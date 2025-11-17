import argparse, math, os
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def collate(batch):
    import torch
    ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
    att = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
    return {"input_ids": ids, "attention_mask": att}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/owm10k_tinyllama_bs128")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--base_model_name", type=str,
                        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Directory with LoRA adapters (e.g. baseline_tinyllama_lora_bs4_ga4_e1). "
                             "If not set, uses plain base model.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=64,
                        help="How many batches of the split to use for analysis.")
    parser.add_argument("--load_in_8bit", type=int, default=0)
    parser.add_argument("--bf16", type=int, default=1)
    parser.add_argument("--out_path", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading dataset from {args.data} [{args.split}] …", flush=True)
    ds_all = load_from_disk(args.data)
    ds = ds_all[args.split]

    max_examples = min(len(ds), args.num_batches * args.batch_size)
    ds = ds.select(range(max_examples))
    print(f"Using {len(ds)} examples for analysis "
          f"({args.num_batches} batches × {args.batch_size}),", flush=True)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    print(f"Loading tokenizer & base model: {args.base_model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    if args.load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            load_in_8bit=True,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            torch_dtype=dtype,
            device_map="auto",
        )

    if args.lora_path is not None and args.lora_path.lower() != "none":
        print(f"Attaching LoRA adapters from {args.lora_path} …", flush=True)
        model = PeftModel.from_pretrained(model, args.lora_path)

    model.eval()
    device = next(model.parameters()).device
    print(f"Model on device: {device}", flush=True)

    all_losses = []
    all_masks = []

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= args.num_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids)

            logits = outputs.logits  # [B, T, V]
            labels = input_ids       # [B, T]

            vocab = logits.size(-1)
            # Cross-entropy per token (no reduction)
            ce_flat = F.cross_entropy(
                logits.view(-1, vocab),
                labels.view(-1),
                reduction="none"
            )
            ce = ce_flat.view_as(labels)  # [B, T]

            # Mask out pads with NaN so we can ignore them later
            ce = ce.masked_fill(attention_mask == 0, float("nan"))

            all_losses.append(ce.cpu())
            all_masks.append(attention_mask.cpu())

            if (bi + 1) % 10 == 0:
                print(f"  Processed batch {bi+1}/{args.num_batches}", flush=True)

    all_losses = torch.cat(all_losses, dim=0)  # [N, T]
    all_masks = torch.cat(all_masks, dim=0)    # [N, T]

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(
        {"losses": all_losses, "mask": all_masks},
        args.out_path,
    )
    print(f"Saved token losses to {args.out_path}", flush=True)

if __name__ == "__main__":
    main()
