import os
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer

IN_DIR  = os.environ.get("IN_DIR",  "data/owm10k_clean")
OUT_DIR = os.environ.get("OUT_DIR", "data/owm10k_tinyllama_bs128")
MODEL   = os.environ.get("MODEL", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
BLOCK   = int(os.environ.get("BLOCK_SIZE", "128"))

def tokenize_lines(ds_split, tok):
    # tokenize each document; no special tokens; we’ll pack later
    def _tok(batch):
        enc = tok(batch["text"], add_special_tokens=False)
        return {"input_ids": enc["input_ids"]}
    return ds_split.map(_tok, batched=True, remove_columns=ds_split.column_names)

def pack_blocks(tokenized_split, block_size):
    # flatten token lists and repack into fixed-length blocks
    all_ids = []
    for ids in tokenized_split["input_ids"]:
        all_ids.extend(ids)
    n_blocks = len(all_ids) // block_size
    blocks = [all_ids[iblock_size:(i+1)block_size] for i in range(n_blocks)]
    attn   = [[1]*blocksize for  in range(n_blocks)]
    return Dataset.from_dict({"input_ids": blocks, "attention_mask": attn})

def main():
    dsd = load_from_disk(IN_DIR)
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tok_train = tokenize_lines(dsd["train"], tok)
    tok_val   = tokenize_lines(dsd["validation"], tok)

    train_blocks = pack_blocks(tok_train, BLOCK)
    val_blocks   = pack_blocks(tok_val, BLOCK)

    out = DatasetDict({"train": train_blocks, "validation": val_blocks})
    out.save_to_disk(OUT_DIR)
    print(out)
    print(f"Saved packed dataset to {OUT_DIR}")
    print(f"Blocks: train={len(train_blocks)} val={len(val_blocks)} (block_size={BLOCK})")

if name == "main":
    main()