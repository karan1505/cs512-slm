import argparse
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

def get_lora_vector(model: torch.nn.Module) -> torch.Tensor:
    """
    Flatten all LoRA parameters into a single 1D vector (on CPU, float32).
    We only grab parameters whose names contain 'lora'.
    """
    vecs = []
    for name, p in model.named_parameters():
        if "lora" in name.lower():
            vecs.append(p.detach().to(torch.float32).view(-1).cpu())
    if not vecs:
        # Debug: show some parameter names to help if this ever fails again
        print("Available parameter names (first 20):")
        for i, (n, _) in enumerate(model.named_parameters()):
            if i >= 20:
                break
            print("  ", n)
        raise RuntimeError("Did not find any LoRA parameters (names containing 'lora').")
    return torch.cat(vecs)

def load_base_plus_lora(base_model_name: str, lora_dir: str, device: str):
    """
    Load the base TinyLlama model and attach LoRA adapters from lora_dir.
    We keep everything on CPU by default to avoid OOM.
    """
    print(f"\nLoading base model: {base_model_name}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=None if device == "cpu" else {"": device},
    )
    print(f"Attaching LoRA from: {lora_dir}")
    model = PeftModel.from_pretrained(base, lora_dir)
    model.eval()
    return model


def cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """
    Cosine similarity between two 1D tensors.
    """
    a = vec_a / (vec_a.norm() + 1e-8)
    b = vec_b / (vec_b.norm() + 1e-8)
    return float((a * b).sum().item())


def main(args):
    device = args.device
    print(f"Using device for loading models: {device}")

    # --- Baseline CLM LoRA ---
    baseline_model = load_base_plus_lora(args.base_model_name, args.baseline_lora, device=device)
    baseline_vec = get_lora_vector(baseline_model)
    del baseline_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Top-k SLM LoRA ---
    topk_model = load_base_plus_lora(args.base_model_name, args.topk_lora, device=device)
    topk_vec = get_lora_vector(topk_model)
    del topk_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Random SLM LoRA ---
    random_model = load_base_plus_lora(args.base_model_name, args.random_lora, device=device)
    random_vec = get_lora_vector(random_model)
    del random_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Stochastic SLM LoRA ---
    stochastic_model = load_base_plus_lora(args.base_model_name, args.stochastic_lora, device=device)
    stochastic_vec = get_lora_vector(stochastic_model)
    del stochastic_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Cosine similarities ---
    print("\n=== Cosine similarity between LoRA update directions ===")
    print(f"Vector sizes: baseline={baseline_vec.numel()}, "
          f"topk={topk_vec.numel()}, random={random_vec.numel()}, "
          f"stochastic={stochastic_vec.numel()}")

    def report_pair(name_a, vec_a, name_b, vec_b):
        cos = cosine_similarity(vec_a, vec_b)
        print(f"{name_a} ↔ {name_b}: {cos:.4f}")

    print("\nAgainst baseline CLM LoRA:")
    report_pair("Baseline", baseline_vec, "Top-k SLM", topk_vec)
    report_pair("Baseline", baseline_vec, "Random SLM", random_vec)
    report_pair("Baseline", baseline_vec, "Stochastic SLM", stochastic_vec)

    print("\nBetween SLM variants:")
    report_pair("Top-k SLM", topk_vec, "Random SLM", random_vec)
    report_pair("Top-k SLM", topk_vec, "Stochastic SLM", stochastic_vec)
    report_pair("Random SLM", random_vec, "Stochastic SLM", stochastic_vec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name",
        type=str,
        required=True,
        help="Base model name, e.g. TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    )
    parser.add_argument(
        "--baseline_lora",
        type=str,
        required=True,
        help="Path to baseline CLM LoRA directory (baseline_tinyllama_lora_bs4_ga4_e1).",
    )
    parser.add_argument(
        "--topk_lora",
        type=str,
        required=True,
        help="Path to top-k SLM LoRA directory.",
    )
    parser.add_argument(
        "--random_lora",
        type=str,
        required=True,
        help="Path to random SLM LoRA directory.",
    )
    parser.add_argument(
        "--stochastic_lora",
        type=str,
        required=True,
        help="Path to stochastic SLM LoRA directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used to load models (no backprop is done, so CPU is fine).",
    )

    args = parser.parse_args()
    main(args)