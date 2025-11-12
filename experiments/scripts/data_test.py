from datasets import load_dataset

ds = load_dataset("wikitext", "wikitext-2-raw-v1")
print(ds)
print("Train example:", ds["train"][0]["text"][:200])