import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def get_grad_vector_from_params(params):
    grads = []
    for p in params:
        if p.grad is not None:
            # keep gradients in 16-bit to reduce memory
            grads.append(p.grad.detach().to(torch.float16).flatten())
    if not grads:
        raise RuntimeError("No gradients found on the selected parameters.")
    return torch.cat(grads)


def compute_cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

def prepare_batch(tokenizer, text, batch_size, max_length, device):
    batch = tokenizer(
        [text] * batch_size,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    labels = batch["input_ids"].clone()
    return batch, labels

def compute_baseline_grad(model_name, tokenizer, batch, labels, device, bf16):
    print("Loading BASELINE model …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
    ).to(device)

    model.train()
    model.zero_grad()
    out = model(**batch, labels=labels)
    loss = out.loss
    print(f"Baseline loss: {loss.item():.4f}")
    loss.backward()

    grad_vec = get_grad_vector_from_params(model.parameters())
    grad_vec = grad_vec.cpu()

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return grad_vec

def compute_slm_grad(model_name, lora_path, tokenizer, batch, labels, device, bf16, tag):
    print(f"\nLoading SLM model ({tag}) from {lora_path} …")
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
    ).to(device)

    slm = PeftModel.from_pretrained(base, lora_path)
    slm.to(device)
    slm.train()
    slm.zero_grad()

    # Enable grads on the *base* model parameters for analysis
    try:
        base_model = slm.get_base_model()
    except AttributeError:
        # Fallback: many PEFT models store base under .base_model.model
        base_model = slm.base_model.model

    for p in base_model.parameters():
        p.requires_grad_(True)

    out = slm(**batch, labels=labels)
    loss = out.loss
    print(f"{tag} loss: {loss.item():.4f}")
    loss.backward()

    # Collect gradients ONLY from the shared base model parameters
    grad_vec = get_grad_vector_from_params(base_model.parameters())
    grad_vec = grad_vec.cpu()

    del slm
    del base
    if device == "cuda":
        torch.cuda.empty_cache()

    return grad_vec

def main(args):
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    print("Loading tokenizer …")
    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Simple mathy batch (same text across models for fair comparison)
    sample_text = (
        "In mathematics, a group is a set equipped with an operation that combines any two elements "
        "to form a third element. The operation satisfies four conditions called the group axioms."
    )

    batch, labels = prepare_batch(
        tok,
        sample_text,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    # 1) Baseline gradient
    g_base = compute_baseline_grad(
        args.model_name, tok, batch, labels, device, bf16=bool(args.bf16)
    )

    # 2) SLM variants (sequential to avoid OOM)
    results = {}

    if args.topk_path:
        g_topk = compute_slm_grad(
            args.model_name, args.topk_path, tok, batch, labels, device, bf16=bool(args.bf16), tag="SLM Top-k"
        )
        results["Top-k"] = g_topk

    if args.random_path:
        g_rand = compute_slm_grad(
            args.model_name, args.random_path, tok, batch, labels, device, bf16=bool(args.bf16), tag="SLM Random"
        )
        results["Random"] = g_rand

    if args.stochastic_path:
        g_stoch = compute_slm_grad(
            args.model_name, args.stochastic_path, tok, batch, labels, device, bf16=bool(args.bf16), tag="SLM Stochastic"
        )
        results["Stochastic"] = g_stoch

    # 3) Cosine similarities
    print("\n=== Cosine similarity of gradients on *base TinyLlama parameters* ===")
    for name, g in results.items():
        cos = compute_cosine_sim(g_base, g)
        print(f"Baseline ↔ {name}: {cos:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str,
                        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    parser.add_argument("--topk_path", type=str, default="slm_tinyllama_topk_r05_e1")
    parser.add_argument("--random_path", type=str, default="slm_tinyllama_random_r05_e1")
    parser.add_argument("--stochastic_path", type=str, default="slm_tinyllama_stochastic_r05_e1")
    parser.add_argument("--bf16", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()
    main(args)
