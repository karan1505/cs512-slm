# train_slm_lora.py
import math
import argparse
import time
import torch

from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, prepare_model_for_kbit_training


# Args
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data/owm10k_tinyllama_bs128",
                    help="Path to token-packed dataset (load_from_disk).")
parser.add_argument("--base_model_name", type=str,
                    default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                    help="HF id of the TinyLlama base model.")
parser.add_argument("--student_lora_path", type=str,
                    default="baseline_tinyllama_lora_bs4_ga4_e1",
                    help="Directory with baseline LoRA checkpoint (from train_baseline_lora).")
parser.add_argument("--ref_model_name", type=str,
                    default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                    help="Reference model for excess-loss computation.")
parser.add_argument("--selection", type=str, default="topk",
                    choices=["topk", "random", "stochastic"],
                    help="Token selection strategy.")
parser.add_argument("--select_ratio", type=float, default=0.5,
                    help="Fraction of (non-pad) tokens to keep (0–1).")

parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--grad_accum", type=int, default=4)
parser.add_argument("--bf16", type=int, default=1)
parser.add_argument("--load_in_8bit_ref", type=int, default=1)
parser.add_argument("--load_in_8bit_student", type=int, default=1)
parser.add_argument("--grad_ckpt", type=int, default=0)
parser.add_argument("--log_every", type=int, default=20)
parser.add_argument("--save_dir", type=str,
                    default="slm_tinyllama_topk_r05_e1")
args = parser.parse_args()


def p(msg: str):
    print(msg, flush=True)


# Dataset
p("Loading dataset…")
ds = load_from_disk(args.data)
train_ds, val_ds = ds["train"], ds["validation"]


def collate(batch):
    ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    att = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
    return {"input_ids": ids, "attention_mask": att}


train_loader = DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate
)
val_loader = DataLoader(
    val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
)

steps_per_epoch = len(train_loader)
p(
    f"Data ready. Steps per epoch: {steps_per_epoch} "
    f"(batch={args.batch_size}, grad_accum={args.grad_accum})"
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Models (ref + student)
torch.backends.cuda.matmul.allow_tf32 = True
dtype = torch.bfloat16 if args.bf16 else None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

p("Loading reference model…")
ref_model = AutoModelForCausalLM.from_pretrained(
    args.ref_model_name,
    load_in_8bit=bool(args.load_in_8bit_ref),
    device_map="auto",
    torch_dtype=dtype,
)
ref_model.eval()
for p_param in ref_model.parameters():
    p_param.requires_grad = False

p("Loading student base model…")
student_base = AutoModelForCausalLM.from_pretrained(
    args.base_model_name,
    load_in_8bit=bool(args.load_in_8bit_student),
    device_map="auto",
    torch_dtype=dtype,
)

# no caching with gradient checkpointing
if args.grad_ckpt:
    student_base.gradient_checkpointing_enable()
student_base.config.use_cache = False

# prepare for LoRA training (important for 8-bit)
student_base = prepare_model_for_kbit_training(
    student_base, use_gradient_checkpointing=bool(args.grad_ckpt)
)

p(f"Attaching LoRA adapters from {args.student_lora_path} …")
student_model = PeftModel.from_pretrained(
    student_base,
    args.student_lora_path,
    is_trainable=True,          # 👈 THIS is the key fix
)
student_model.train()
student_model.print_trainable_parameters()


# Optimizer
if hasattr(torch.optim, "AdamW"):
    opt = torch.optim.AdamW(
        student_model.parameters(), lr=args.lr, fused=True
    ) if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.optim.AdamW(
        student_model.parameters(), lr=args.lr
    )
else:
    opt = torch.optim.Adam(student_model.parameters(), lr=args.lr)

loss_fct = torch.nn.CrossEntropyLoss(reduction="none")


# Helper: per-token losses
def token_losses(model, input_ids, attention_mask, use_grad: bool):
    """
    Returns per-token next-token prediction loss.
    Shape: [B, T-1]
    """
    ctx = torch.enable_grad() if use_grad else torch.no_grad()
    with ctx:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]
        # shift
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        B, Tm1, V = shift_logits.shape
        loss = loss_fct(
            shift_logits.view(B * Tm1, V),
            shift_labels.view(B * Tm1),
        )
        loss = loss.view(B, Tm1)
    return loss


# Evaluation (full-token CLM)
def evaluate():
    student_model.eval()
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["input_ids"]
            out = student_model(**batch, labels=labels)
            total_loss += out.loss.item()
            n_batches += 1
    student_model.train()
    avg = total_loss / max(n_batches, 1)
    return avg, math.exp(avg)


# SLM training loop
p(
    f"Starting SLM-LoRA training… selection={args.selection}, "
    f"select_ratio={args.select_ratio}"
)
global_step = 0

for epoch in range(args.epochs):
    running = 0.0
    t0 = time.time()
    p(f"Epoch {epoch+1}/{args.epochs} …")

    for i, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # [B, T-1]
        ref_losses = token_losses(ref_model, input_ids, attention_mask, use_grad=False)
        student_losses = token_losses(
            student_model, input_ids, attention_mask, use_grad=True
        )

        # valid mask (exclude padding positions)
        valid = attention_mask[:, 1:].bool()  # [B, T-1]

        # excess loss
        excess = student_losses - ref_losses  # [B, T-1]

        # flatten over valid positions only
        flat_valid_idx = valid.view(-1).nonzero(as_tuple=False).squeeze(1)
        num_valid = flat_valid_idx.numel()
        k_tokens = max(1, int(args.select_ratio * num_valid))

        if args.selection == "topk":
            scores = excess.clone()
            scores[~valid] = -1e9  # mask pads
            flat_scores = scores.view(-1)
            top_vals, top_idx = torch.topk(flat_scores, k_tokens)
            chosen = top_idx

        elif args.selection == "random":
            perm = torch.randperm(num_valid, device=device)
            chosen = flat_valid_idx[perm[:k_tokens]]

        else:  # stochastic
            scores = torch.relu(excess)  # only positive excess
            scores[~valid] = 0.0
            flat_scores = scores.view(-1)
            if flat_scores.sum() <= 0:
                # fallback to random if everything is <= 0
                perm = torch.randperm(num_valid, device=device)
                chosen = flat_valid_idx[perm[:k_tokens]]
            else:
                probs = flat_scores / flat_scores.sum()
                chosen = torch.multinomial(
                    probs, k_tokens, replacement=False
                )

        # build mask
        sel_mask = torch.zeros_like(valid.view(-1), dtype=torch.bool)
        sel_mask[chosen] = True
        sel_mask = sel_mask.view_as(valid)

        selected_losses = student_losses[sel_mask]
        slm_loss = selected_losses.mean()

        (slm_loss / args.grad_accum).backward()

        if (i + 1) % args.grad_accum == 0:
            opt.step()
            opt.zero_grad()
            global_step += 1

        running += slm_loss.item()

        # heartbeat every 100 raw steps
        if (i + 1) % 100 == 0:
            print(".", end="", flush=True)

        # log every log_every optimizer steps
        if (i + 1) % (args.log_every * args.grad_accum) == 0:
            opt_steps_done = (i + 1) // args.grad_accum
            opt_steps_left = (steps_per_epoch - (i + 1)) // args.grad_accum
            dt = time.time() - t0
            avg_loss = running / (args.log_every)
            p(
                f"\nopt_step {opt_steps_done} "
                f"| slm_loss (selected) {avg_loss:.4f} "
                f"| {dt:.1f}s for last {args.log_every} opt steps "
                f"| ~{opt_steps_left} opt steps left"
            )
            running = 0.0
            t0 = time.time()

    vloss, vppl = evaluate()
    p(f"End epoch {epoch+1}: val_loss={vloss:.4f} val_ppl={vppl:.2f}")

p(f"Saving student model (LoRA adapters) → {args.save_dir}")
student_model.save_pretrained(args.save_dir)
tokenizer.save_pretrained(args.save_dir)
p("Done.")
