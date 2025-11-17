# train_baseline_lora.py (chatty, tuned for RTX 4060 8GB)

import math, argparse, time, torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ------------------------------
# Args
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data/owm10k_tinyllama_bs128")
parser.add_argument(
    "--model_name",
    type=str,
    default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--grad_accum", type=int, default=4)
parser.add_argument("--bf16", type=int, default=1)
parser.add_argument("--load_in_8bit", type=int, default=0)
parser.add_argument("--grad_ckpt", type=int, default=0)
parser.add_argument("--use_lora", type=int, default=1)
parser.add_argument("--lora_r", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=32)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--log_every", type=int, default=20)
parser.add_argument("--save_dir", type=str, default="baseline_tinyllama_lora_bs4_ga4_e1")
args = parser.parse_args()


def p(msg: str) -> None:
    print(msg, flush=True)


# ------------------------------
# Data
# ------------------------------
p("Loading dataset…")
ds = load_from_disk(args.data)
train_ds, val_ds = ds["train"], ds["validation"]


def collate(batch):
    ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    att = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
    return {"input_ids": ids, "attention_mask": att}


train_loader = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate,
    num_workers=4,
    pin_memory=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collate,
    num_workers=4,
    pin_memory=True,
)

steps_per_epoch = len(train_loader)
p(
    f"Data ready. Steps per epoch: {steps_per_epoch} "
    f"(batch={args.batch_size}, grad_accum={args.grad_accum})"
)

# ------------------------------
# Tokenizer & model
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

torch.backends.cuda.matmul.allow_tf32 = True
dtype = torch.bfloat16 if args.bf16 else None

p("Loading model… (this can take a couple minutes)")
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    load_in_8bit=bool(args.load_in_8bit),
    device_map="auto",
    dtype=dtype,  # new HF arg; replaces torch_dtype
)

if args.grad_ckpt:
    model.gradient_checkpointing_enable()

model.config.use_cache = False

# prepare for k-bit training only when using 8bit
if args.load_in_8bit:
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=bool(args.grad_ckpt)
    )

if args.use_lora:
    lconf = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lconf)
    model.print_trainable_parameters()

# ------------------------------
# Optimizer
# ------------------------------
try:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)
    p("Using fused AdamW optimizer.")
except TypeError:
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    p("Using standard AdamW optimizer (fused not available).")


# ------------------------------
# Eval
# ------------------------------
def evaluate():
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            labels = batch["input_ids"].to(model.device)
            out = model(
                **{k: v.to(model.device) for k, v in batch.items()},
                labels=labels,
            )
            total += out.loss.item()
            n += 1
    model.train()
    avg = total / max(n, 1)
    return avg, math.exp(avg)


# ------------------------------
# Train loop
# ------------------------------
p("Starting CLM baseline training…")
model.train()
global_step = 0

for epoch in range(args.epochs):
    running, t0 = 0.0, time.time()
    p(f"Epoch {epoch + 1}/{args.epochs} …")

    for i, batch in enumerate(train_loader):
        labels = batch["input_ids"].to(model.device)
        out = model(
            **{k: v.to(model.device) for k, v in batch.items()},
            labels=labels,
        )

        (out.loss / args.grad_accum).backward()

        if (i + 1) % args.grad_accum == 0:
            opt.step()
            opt.zero_grad()
            global_step += 1

        running += out.loss.item()

        # Heartbeat every 100 raw steps
        if (i + 1) % 100 == 0:
            print(".", end="", flush=True)

        # Log every log_every *accum* raw steps (≈ log_every opt steps)
        if (i + 1) % (args.log_every * args.grad_accum) == 0:
            steps_done = (i + 1) // args.grad_accum
            steps_left = (steps_per_epoch - (i + 1)) // args.grad_accum
            dt = time.time() - t0
            avg_loss = running / args.log_every
            p(
                f"\nopt_step {steps_done} | avg_loss {avg_loss:.4f} | "
                f"{dt:.1f}s for last {args.log_every} opt steps | "
                f"~{steps_left} opt steps left"
            )
            running = 0.0
            t0 = time.time()

    vloss, vppl = evaluate()
    p(f"End epoch {epoch + 1}: val_loss={vloss:.4f} val_ppl={vppl:.2f}")

# ------------------------------
# Save
# ------------------------------
p(f"Saving → {args.save_dir}")
model.save_pretrained(args.save_dir)
tokenizer.save_pretrained(args.save_dir)
p("Done.")
