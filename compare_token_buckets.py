import argparse
import torch

def load_losses(path):
    d = torch.load(path)
    # Ensure float32 for quantile and comparisons
    losses = d["losses"].to(torch.float32)
    mask = d["mask"].bool()
    return losses, mask

def summarize_transitions(baseline_losses, mask, method_losses, name, high_quantile=0.7):
    """
    baseline_losses: [N, T] tensor (with NaNs on pads), float
    method_losses:   [N, T] tensor aligned with baseline, float
    mask:            [N, T] bool, True for real tokens
    high_quantile:   e.g. 0.7 -> top 30% losses are 'high'
    """
    # Flatten real tokens
    base_flat = baseline_losses[mask]
    method_flat = method_losses[mask]

    # Make sure everything is float32 for quantile
    base_flat = base_flat.to(torch.float32)
    method_flat = method_flat.to(torch.float32)

    # Remove any NaNs just in case
    valid = torch.isfinite(base_flat) & torch.isfinite(method_flat)
    base_flat = base_flat[valid]
    method_flat = method_flat[valid]

    # Threshold on baseline to define H vs L
    thr = torch.quantile(base_flat, high_quantile)
    hi0 = base_flat >= thr      # high under baseline
    lo0 = ~hi0                  # low under baseline

    hi1 = method_flat >= thr    # high under method
    lo1 = ~hi1

    total = base_flat.numel()
    if total == 0:
        print(f"\n==== {name} ====")
        print("No valid tokens after masking; check inputs.")
        return

    # Buckets
    hh = (hi0 & hi1).sum().item()
    hl = (hi0 & lo1).sum().item()
    lh = (lo0 & hi1).sum().item()
    ll = (lo0 & lo1).sum().item()

    def pct(x): return 100.0 * x / total

    print(f"\n==== {name} ====")
    print(f"Threshold (baseline high-loss) = {thr.item():.4f}")
    print(f"Total tokens considered: {total}")

    print(f"H → H (stayed hard):   {hh} ({pct(hh):.2f}%)")
    print(f"H → L (improved):      {hl} ({pct(hl):.2f}%)")
    print(f"L → H (got worse):     {lh} ({pct(lh):.2f}%)")
    print(f"L → L (stayed easy):   {ll} ({pct(ll):.2f}%)")

    # Extra: mean losses for high bucket
    base_high_mean = base_flat[hi0].mean().item()
    meth_high_mean = method_flat[hi0].mean().item()
    print(f"Mean loss on high-loss tokens: baseline={base_high_mean:.4f}, {name}={meth_high_mean:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_path", type=str, required=True)
    parser.add_argument("--topk_path", type=str, required=True)
    parser.add_argument("--random_path", type=str, required=True)
    parser.add_argument("--stochastic_path", type=str, required=True)
    parser.add_argument("--high_quantile", type=float, default=0.7,
                        help="Quantile to define high-loss tokens (0.7 => top 30% are 'high').")
    args = parser.parse_args()

    print("Loading baseline losses …")
    base_losses, base_mask = load_losses(args.baseline_path)

    print("Loading top-k SLM losses …")
    topk_losses, topk_mask = load_losses(args.topk_path)

    print("Loading random SLM losses …")
    rand_losses, rand_mask = load_losses(args.random_path)

    print("Loading stochastic SLM losses …")
    stoch_losses, stoch_mask = load_losses(args.stochastic_path)

    # We assume masks are identical across runs; use baseline's
    mask = base_mask

    # Summaries
    summarize_transitions(base_losses, mask, base_losses, "Baseline (self)", high_quantile=args.high_quantile)
    summarize_transitions(base_losses, mask, topk_losses, "SLM Top-k r=0.5", high_quantile=args.high_quantile)
    summarize_transitions(base_losses, mask, rand_losses, "SLM Random r=0.5", high_quantile=args.high_quantile)
    summarize_transitions(base_losses, mask, stoch_losses, "SLM Stochastic r=0.5", high_quantile=args.high_quantile)

if __name__ == "__main__":
    main()