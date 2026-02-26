import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 1. TERNARY QUANTIZATION WITH SCALING FACTOR α
# =========================================================

class TernaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        delta = 0.05 * w.abs().mean()

        w_q = torch.zeros_like(w)
        w_q[w > delta] = 1.0
        w_q[w < -delta] = -1.0

        alpha = w.abs().mean()
        ctx.save_for_backward(w)

        return alpha * w_q

    @staticmethod
    def backward(ctx, grad_output):
        w, = ctx.saved_tensors
        grad = grad_output.clone()
        grad[w.abs() > 1] = 0
        return grad


class TernaryActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        delta = 0.1 * x.abs().mean()
        x_q = torch.zeros_like(x)
        x_q[x > delta] = 1.0
        x_q[x < -delta] = -1.0
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp(-1, 1)


# =========================================================
# 2. TERNARY LINEAR LAYER
# =========================================================

class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)

    def forward(self, x):
        x_q = TernaryActivation.apply(x)
        w_q = TernaryQuantize.apply(self.weight)
        return F.linear(x_q, w_q)


# =========================================================
# 3. ATTENTION WITH KV CACHE
# =========================================================

class TernarySelfAttention(nn.Module):
    def __init__(self, dim, heads=6):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.qkv = TernaryLinear(dim, dim * 3)
        self.proj = TernaryLinear(dim, dim)

    def forward(self, x, cache=None):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        k = k.view(B, T, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, T, self.heads, C // self.heads).transpose(1, 2)

        if cache is not None:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)

        new_cache = {"k": k, "v": v}

        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.softmax(dim=-1)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out), new_cache


# =========================================================
# 4. MLP + BLOCK
# =========================================================

class TernaryMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = TernaryLinear(dim, dim * 4)
        self.fc2 = TernaryLinear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = TernarySelfAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = TernaryMLP(dim)

    def forward(self, x, cache=None):
        attn_out, new_cache = self.attn(self.ln1(x), cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


# =========================================================
# 5. FULL TERNARY SLM
# =========================================================

class BitNetSLM(nn.Module):
    def __init__(self, vocab, dim=384, depth=8, heads=6, max_len=512):
        super().__init__()

        self.token_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))

        self.blocks = nn.ModuleList([
            Block(dim, heads) for _ in range(depth)
        ])

        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, idx, cache=None):
        B, T = idx.shape

        x = self.token_emb(idx) + self.pos_emb[:, :T]

        new_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = None if cache is None else cache[i]
            x, new_cache = block(x, layer_cache)
            new_caches.append(new_cache)

        x = self.ln(x)
        logits = self.head(x)
        return logits, new_caches


# =========================================================
# 6. REAL 2-BIT PACKING
# =========================================================

def pack_ternary(t):
    encoded = (t + 1).to(torch.uint8)
    flat = encoded.view(-1)

    packed = torch.zeros((len(flat) + 3) // 4, dtype=torch.uint8)

    for i in range(len(flat)):
        packed[i // 4] |= flat[i] << (2 * (i % 4))

    return packed


# =========================================================
# 7. DATASET PIPELINE (TinyStories)
# =========================================================

print("Loading dataset...")
dataset = load_dataset("roneneldan/TinyStories", split="train")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

dataset = dataset.map(tokenize, remove_columns=["text"])
dataset.set_format(type="torch", columns=["input_ids"])

dataset = dataset.select(range(20000))
# =========================================================
# 8. TRAINING LOOP
# =========================================================

model = BitNetSLM(len(tokenizer)).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

epochs = 1
batch_size = 4

# loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.eos_token_id)
    return {"input_ids": input_ids}

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

print("Training...")

for epoch in range(epochs):
    for batch in tqdm(loader):
        x = batch["input_ids"].to(device)

        logits, _ = model(x[:, :-1])
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("Training done.")


# =========================================================
# 9. GENERATION WITH KV CACHE
# =========================================================

def generate(prompt, steps=50):
    model.eval()
    tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    cache = None

    for _ in range(steps):
        logits, cache = model(tokens[:, -1:], cache)
        next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)

    return tokenizer.decode(tokens[0])


print(generate("Once upon a time"))