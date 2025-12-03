# train_baseline.py
import math
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Argparse config
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="gpt2",
                    help="HF model name or local path (default: gpt2).")
parser.add_argument("--epochs", type=int, default=1,
                    help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=8,
                    help="Batch size.")
parser.add_argument("--lr", type=float, default=5e-5,
                    help="Learning rate.")
parser.add_argument("--block_size", type=int, default=128,
                    help="Sequence length for training blocks.")
parser.add_argument("--save_dir", type=str, default=None,
                    help="If set, save model + tokenizer to this directory at the end.")
args = parser.parse_args()

MODEL_NAME = args.model_name
BLOCK_SIZE = args.block_size
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs
LEARNING_RATE = args.lr
LOG_EVERY = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(f"Config: model={MODEL_NAME}, epochs={NUM_EPOCHS}, batch_size={BATCH_SIZE}, lr={LEARNING_RATE}")

# Load tokenizer & dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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

# Group tokens into fixed-length blocks
def group_texts(examples):
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

# Wrap HF dataset in PyTorch Dataset
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

# Load model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

# Training & evaluation loops
def evaluate():
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids  # language modeling

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

print("Starting training...")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for step, batch in enumerate(train_loader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = input_ids

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if step % LOG_EVERY == 0:
            avg = running_loss / LOG_EVERY
            print(f"Epoch {epoch+1} Step {step}/{len(train_loader)} - loss: {avg:.4f}")
            running_loss = 0.0

    # end epoch → evaluate
    val_loss, val_ppl = evaluate()
    print(f"End of epoch {epoch+1}: val_loss = {val_loss:.4f}, val_ppl = {val_ppl:.2f}")

print("Training complete.")

if args.save_dir is not None:
    print(f"Saving model and tokenizer to {args.save_dir} ...")
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print("Save complete.")
