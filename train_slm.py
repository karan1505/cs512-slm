# train_slm.py
import math
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ------------------------------
# Argparse config
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--select_ratio", type=float, default=0.6,
                    help="Fraction of tokens to keep (0–1).")
parser.add_argument("--epochs", type=int, default=1,
                    help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=8,
                    help="Batch size.")
parser.add_argument("--subset_train_blocks", type=int, default=None,
                    help="If set, only use this many training blocks (faster runs).")
args = parser.parse_args()

STUDENT_MODEL_NAME = "gpt2"
REF_MODEL_NAME = "gpt2"
BLOCK_SIZE = 128
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
SELECT_RATIO = args.select_ratio
LEARNING_RATE = 5e-5
LOG_EVERY = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(f"Config: epochs={NUM_EPOCHS}, batch_size={BATCH_SIZE}, select_ratio={SELECT_RATIO}")


# ------------------------------
# Load tokenizer & dataset
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading WikiText-2...")
raw_ds = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized = raw_ds.map(
    tokenize_function,
    batched=True,
    num_proc=4,
    remove_columns=["text"],
)

def group_texts(examples):
    # concatenate then split into fixed-length blocks
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = (len(concatenated["input_ids"]) // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated.items()
    }
    return result

lm_datasets = tokenized.map(
    group_texts,
    batched=True,
    num_proc=4,
)

train_hf = lm_datasets["train"]
val_hf = lm_datasets["validation"]

print("Train blocks:", len(train_hf), "Val blocks:", len(val_hf))

# ------------------------------
# Dataset / DataLoader
# ------------------------------
class HFDataset(Dataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
        }

train_ds = HFDataset(train_hf)
val_ds = HFDataset(val_hf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------------
# Load models
# ------------------------------
print("Loading reference model:", REF_MODEL_NAME)
ref_model = AutoModelForCausalLM.from_pretrained(REF_MODEL_NAME)
ref_model.resize_token_embeddings(len(tokenizer))
ref_model.to(device)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

print("Loading student model:", STUDENT_MODEL_NAME)
student_model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_NAME)
student_model.resize_token_embeddings(len(tokenizer))
student_model.to(device)
student_model.train()

optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)
loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

# ------------------------------
# Helper: compute per-token loss
# ------------------------------
def token_losses(model, input_ids, attention_mask, use_grad: bool):
    """
    Returns token-level losses for next-token prediction.
    Shape: [batch_size, seq_len-1]
    """
    context = torch.enable_grad() if use_grad else torch.no_grad()
    with context:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()  # [B, T-1]

        B, T_minus1, V = shift_logits.size()
        loss = loss_fct(
            shift_logits.view(B * T_minus1, V),
            shift_labels.view(B * T_minus1),
        )
        loss = loss.view(B, T_minus1)  # per-token loss
    return loss

# ------------------------------
# Evaluation: standard CLM loss on ALL tokens
# ------------------------------
def evaluate():
    student_model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss  # average over tokens & batch

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    ppl = math.exp(avg_loss)
    student_model.train()
    return avg_loss, ppl

# ------------------------------
# Training loop with SLM
# ------------------------------
print("Starting SLM training with select_ratio =", SELECT_RATIO)
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0

    for step, batch in enumerate(train_loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # 1) token losses from reference (no grad)
        ref_loss_tokens = token_losses(
            ref_model, input_ids, attention_mask, use_grad=False
        )

        # 2) token losses from student (with grad)
        student_loss_tokens = token_losses(
            student_model, input_ids, attention_mask, use_grad=True
        )

        # 3) excess loss and top-k mask
        excess = student_loss_tokens - ref_loss_tokens  # [B, T-1]
        B, Tm1 = excess.size()
        num_tokens = B * Tm1
        k_tokens = max(1, int(SELECT_RATIO * num_tokens))

        # Flatten, choose top-k, build boolean mask
        flat_excess = excess.view(-1)
        top_vals, top_idx = torch.topk(flat_excess, k_tokens)
        mask = torch.zeros_like(flat_excess, dtype=torch.bool)
        mask[top_idx] = True
        mask = mask.view(B, Tm1)

        # 4) SLM loss: average student loss over selected tokens only
        selected_losses = student_loss_tokens[mask]  # 1D tensor of chosen tokens
        slm_loss = selected_losses.mean()

        optimizer.zero_grad()
        slm_loss.backward()
        optimizer.step()

        running_loss += slm_loss.item()

        if step % LOG_EVERY == 0:
            avg = running_loss / LOG_EVERY
            print(
                f"Epoch {epoch+1} Step {step}/{len(train_loader)} "
                f"- SLM loss (selected tokens only): {avg:.4f}"
            )
            running_loss = 0.0

    # End of epoch: evaluate on full validation set
    val_loss, val_ppl = evaluate()
    print(
        f"End of epoch {epoch+1}: "
        f"val_loss (full tokens) = {val_loss:.4f}, val_ppl = {val_ppl:.2f}"
    )

print("SLM training complete.")
