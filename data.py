from datasets import load_dataset
import tiktoken
import torch

print("Loading dataset...")

dataset = load_dataset("wikitext", "wikitext-103-v1")

enc = tiktoken.get_encoding("cl100k_base")

# ---------------------------
# Tokenize function
# ---------------------------

def tokenize(split):
    tokens = []
    for text in dataset[split]["text"]:
        if text.strip() == "":
            continue
        ids = enc.encode(text)
        tokens.extend(ids + [enc.eot_token])
    return torch.tensor(tokens, dtype=torch.long)

print("Tokenizing train...")
train_data = tokenize("train")

print("Tokenizing validation...")
val_data = tokenize("validation")

max_token_val = enc.n_vocab

print("Train tokens:", len(train_data))
print("Val tokens:", len(val_data))
print("Vocab size:", max_token_val)