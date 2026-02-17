import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import numpy as np
import os
import requests
from config import *
import math
from data import *

# we get the data set
# Load training data
# if not os.path.exists('data/sales_textbook.txt'):
#     url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
#     with open('data/sales_textbook.txt', 'w') as f:
#         f.write(requests.get(url).text)

# with open('data/sales_textbook.txt', 'r', encoding='utf-8') as f:
#     text = f.read()

# encoding  = tiktoken.get_encoding("cl100k_base")
# tokenized_text = encoding.encode(text)
# max_token_val = max(tokenized_text) + 1
# tokenized_text = torch.tensor(tokenized_text, dtype=torch.long) # device=device)

# split_idx = int(len(tokenized_text) * 0.9) # we will use 90%
# train_data = tokenized_text[:split_idx]
# val_data = tokenized_text[split_idx:]

# tokens = np.fromfile("data/train_tokens.bin", dtype=np.uint32)
# tokens = torch.from_numpy(tokens.astype(np.int64))

# split_idx = int(len(tokens) * 0.9)

# train_data = tokens[:split_idx]
# val_data = tokens[split_idx:]

# max_token_val = int(tokens.max()) + 1

class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.d_model * 4, out_features=self.d_model * 4),
            nn.Linear(in_features=self.d_model * 4, out_features=self.d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.ffn(x)
    
class Attention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.context_length = context_length
        self.dropout = dropout

        self.k = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.v = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.q = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones((self.context_length, self.context_length))))
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        B,T,C = x.shape
        assert T <= self.context_length
        assert C == self.d_model
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        # scaled dot prod
        weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # apply mask
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(input=weights, dim=-1)
        weights = self.dropout_layer(weights)

        out = weights @ v
        return out
    
class MHA(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.d_model = d_model
        self.context_length = context_length
        self.dropout = dropout

        self.heads = nn.ModuleList([Attention(head_size=self.head_size) for _ in range(self.num_heads)])
        self.projection_layer = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.projection_layer(out)
        out = self.dropout_layer(out)
        return out

def precompute_rope_freqs(head_dim, max_seq_len):
    assert head_dim % 2 == 0, "head_dim must be even"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    theta = 1.0 / (10000 ** (
        torch.arange(0, head_dim, 2, device=device).float() / head_dim
    ))

    positions = torch.arange(max_seq_len, device=device).float()

    freqs = torch.outer(positions, theta)  # (T, head_dim/2)

    return freqs
def apply_rope(x, freqs):
    # x shape: (B, T, H, D)

    B, T, H, D = x.shape

    # Move to (B, H, T, D)
    x = x.permute(0, 2, 1, 3)

    freqs = freqs[:T]  # (T, D/2)

    # reshape for broadcasting
    freqs = freqs.unsqueeze(0).unsqueeze(0)  # (1,1,T,D/2)

    x1, x2 = x[..., :D//2], x[..., D//2:]

    x = torch.cat([
        x1 * freqs.cos() - x2 * freqs.sin(),
        x1 * freqs.sin() + x2 * freqs.cos()
    ], dim=-1)

    # return to original shape
    return x.permute(0, 2, 1, 3)

class GQAAttention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()

        self.n_q = num_heads           # query heads
        self.n_kv = num_heads // 4       # KV heads (smaller)
        self.head_dim = d_model // self.n_q

        self.q_proj = nn.Linear(d_model, self.n_q * self.head_dim)
        self.k_proj = nn.Linear(d_model, self.n_kv * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.n_kv * self.head_dim)

        self.out_proj = nn.Linear(d_model, d_model)

        self.register_buffer(
            "rope_freqs",
            precompute_rope_freqs(self.head_dim, context_length),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_q, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim)

        # Apply RoPE
        q = apply_rope(q, self.rope_freqs)
        k = apply_rope(k, self.rope_freqs)

        # Expand KV to match Q groups
        repeat = self.n_q // self.n_kv
        k = k.repeat_interleave(repeat, dim=2)
        v = v.repeat_interleave(repeat, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - more stable than LayerNorm"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, T, C)
        norm = x.norm(2, dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)

class TB(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.head_size = d_model // num_heads
        self.nums_heads = num_heads
        self.dropout = dropout

        # self.mha = MHA(head_size=self.head_size)
        self.mha = GQAAttention(head_size=self.head_size)
        self.ffn = FFN()
        self.norm1 = RMSNorm(self.d_model)
        self.norm2 = RMSNorm(self.d_model)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    def forward(self, x):
        x = x + self.mha(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    
class LModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.max_token_value = max_token_val

        self.token_em = nn.Embedding(num_embeddings=self.max_token_value, embedding_dim=self.d_model)
        self.transformer_blocks = nn.Sequential(*(
            [TB(num_heads=self.num_heads) for _ in range(self.num_blocks)] + [nn.LayerNorm(self.d_model)]
        ))
        self.lmoll = nn.Linear(in_features=self.d_model, out_features=self.max_token_value + 1)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        pelt = torch.zeros(self.context_length, self.d_model)
        pos = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0)/self.d_model))
        pelt[:, 0::2] = torch.sin(pos * div_term)
        pelt[:, 1::2] = torch.cos(pos * div_term)

        pos_em = pelt[:T, :].to(device)
        x = self.token_em(idx) + pos_em
        x = self.transformer_blocks(x)

        logits = self.lmoll(x)
        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B*T, C)
            targets_reshaped = targets.view(B*T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss
    def generate(self, idx, max_new_token, temperature=1.0, top_k=None):
        for _ in range(max_new_token):
            idx_corp = idx[:, -self.context_length:]
            logits, _ = self(idx_corp)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
model = LModel()
model = model.to(device)


# Get input embedding batch
def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    x = torch.stack([data[idx:idx + context_length] for idx in idxs])
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs])
    return x.to(device), y.to(device)


# Calculate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if __name__ == "__main__":
    # Use AdamW optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
    tracked_losses = list()
    for step in range(max_iters):
        if step % eval_iters == 0 or step == max_iters - 1:
            losses = estimate_loss()
            tracked_losses.append(losses)
            print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
                round(losses['valid'].item(), 3))

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save the model state dictionary
    torch.save(model.state_dict(), 'model-ckpt.pt')

# # Generate
# model.eval()
# start = 'The salesperson'
# start_ids = encoding.encode(start)
# x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
# y = model.generate(x, max_new_token=100)
# print('---------------')
# print(encoding.decode(y[0].tolist()))
# print('---------------')