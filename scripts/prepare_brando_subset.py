# scripts/prepare_brando_subset.py
import os, random, pandas as pd
from datasets import load_dataset, Dataset

OUT_RAW = os.environ.get("OUT_RAW", "data/owm10k_raw")
OUT_CLEAN = os.environ.get("OUT_CLEAN", "data/owm10k_clean")
SEED = int(os.environ.get("SEED", "42"))

def main():
    random.seed(SEED)

    # Load community tiny subset (≈10k rows)
    ds = load_dataset("brando/small-open-web-math-dataset-v2", split="train")
    ds.save_to_disk(OUT_RAW)

    # Basic cleaning: length filter + exact-dup drop
    df = pd.DataFrame({"text": ds["text"]})
    # keep medium-length docs (tune if you want)
    df = df[df["text"].str.len().between(200, 8000)]
    df = df.drop_duplicates(subset="text")
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # simple train/val split (95/5)
    n = len(df)
    n_val = max(500, int(0.05 * n))
    df_train = df.iloc[:-n_val].copy()
    df_val = df.iloc[-n_val:].copy()

    ds_train = Dataset.from_pandas(df_train, preserve_index=False)
    ds_val = Dataset.from_pandas(df_val, preserve_index=False)

    from datasets import DatasetDict
    dsd = DatasetDict({"train": ds_train, "validation": ds_val})
    dsd.save_to_disk(OUT_CLEAN)

    print(dsd)
    print(f"Saved cleaned dataset to {OUT_CLEAN}")
    print(f"Train docs: {len(ds_train)} | Val docs: {len(ds_val)}")

if __name__ == "__main__":
    main()