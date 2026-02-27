# from datasets import load_dataset
# import tiktoken
# import torch

# print("Loading dataset...")

# dataset = load_dataset("wikitext", "wikitext-103-v1")

# enc = tiktoken.get_encoding("cl100k_base")

# # ---------------------------
# # Tokenize function
# # ---------------------------

# def tokenize(split):
#     tokens = []
#     for text in dataset[split]["text"]:
#         if text.strip() == "":
#             continue
#         ids = enc.encode(text)
#         tokens.extend(ids + [enc.eot_token])
#     return torch.tensor(tokens, dtype=torch.long)

# print("Tokenizing train...")
# train_data = tokenize("train")

# print("Tokenizing validation...")
# val_data = tokenize("validation")

# max_token_val = enc.n_vocab

# # print("Train tokens:", len(train_data))
# # print("Val tokens:", len(val_data))
# # print("Vocab size:", max_token_val)
from datasets import load_dataset
import torch
import tiktoken

print("Loading dataset (streaming)...")

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    split="train",
    streaming=True
)

enc = tiktoken.get_encoding("cl100k_base")
max_token_val = enc.n_vocab
tokens = []

LIMIT = 15_000_000   # number of tokens you want

print("Tokenizing...")

total = 0

for example in dataset:
    text = example["text"]

    if text.strip():
        ids = enc.encode(text)
        tokens.extend(ids + [enc.eot_token])
        total += len(ids)

    if total >= LIMIT:
        break

data = torch.tensor(tokens, dtype=torch.long)

# split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print("Train tokens:", len(train_data))
print("Val tokens:", len(val_data))