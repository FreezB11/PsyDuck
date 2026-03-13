claude2 = """
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from datasets import load_dataset
import tiktoken

# =============================================================================
# Hyperparameters -- edit here
# =============================================================================

DIM         = 768       # embedding dimension
LAYERS      = 12*2        # transformer blocks
HEADS       = 16        # attention heads  (head_dim = 768/12 = 64)
SEQ_LEN     = 512       # context length
BATCH       = 24         # per-GPU batch size
VOCAB       = 50257     # GPT-2 BPE vocab

LR          = 3e-4      # peak learning rate
MIN_LR      = 3e-5      # cosine decay floor
WEIGHT_DECAY= 0.1
BETA1, BETA2= 0.9, 0.95
GRAD_CLIP   = 1.0

WARMUP_STEPS= 15000
TOTAL_STEPS = 200_000

SAVE_EVERY  = 100      # save checkpoint every N steps
LOG_EVERY   = 20       # print loss + run inference every N steps
CKPT_DIR    = "checkpoints"
RESUME_CKPT     = "/kaggle/input/datasets/gpulll/my-llm/model.pt"
# =============================================================================
# Model
# =============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads   = heads
        self.head_dim = dim // heads

        self.qkv  = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim,     bias=False)

        # output projection init: scale down to avoid large residuals at init
        nn.init.normal_(self.proj.weight, std=0.02 / math.sqrt(2 * LAYERS))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # reshape to (B, heads, T, head_dim)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)

        # Flash attention (causal, fused kernel -- no attention matrix stored)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FFN(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4, bias=False)
        self.fc2 = nn.Linear(dim * 4, dim, bias=False)

        # same scaled init as attn proj
        nn.init.normal_(self.fc2.weight, std=0.02 / math.sqrt(2 * LAYERS))

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):

    def __init__(self, dim, heads):
        super().__init__()
        self.ln1  = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads)
        self.ln2  = nn.LayerNorm(dim)
        self.ffn  = FFN(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # pre-norm attention
        x = x + self.ffn(self.ln2(x))    # pre-norm FFN
        return x


class GPT(nn.Module):
    def __init__(self, vocab=VOCAB, dim=DIM, layers=LAYERS, heads=HEADS, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len

        self.token_emb = nn.Embedding(vocab, dim)
        self.pos_emb   = nn.Embedding(seq_len, dim)   # learned positional

        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])

        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

        # weight init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.seq_len, f"sequence length {T} > max {self.seq_len}"

        pos = torch.arange(T, device=idx.device)
        h   = self.token_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            # gradient checkpointing: recompute activations during backward
            # saves ~60% activation memory at cost of ~33% extra compute
            h = grad_checkpoint(block, h, use_reentrant=False)

        return self.head(self.ln_f(h))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, max_new=100, temperature=0.8, top_k=40):

        self.eval()
        for _ in range(max_new):
            # crop to context window
            idx_cond = idx[:, -self.seq_len:]
            logits   = self(idx_cond)[:, -1, :]          # (B, vocab)
            logits   = logits / temperature

            # top-k filter
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        self.train()
        return idx


# =============================================================================
# LR Schedule
# =============================================================================

def get_lr(step):
    # linear warmup
    if step < WARMUP_STEPS:
        return LR * step / max(WARMUP_STEPS, 1)
    # cosine decay to MIN_LR
    t = (step - WARMUP_STEPS) / max(TOTAL_STEPS - WARMUP_STEPS, 1)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * t))


# =============================================================================
# Dataset
# =============================================================================

class StreamDataset(IterableDataset):
    def __init__(self, seq_len=SEQ_LEN, rank=0, world_size=1):
        self.seq_len    = seq_len
        self.rank       = rank
        self.world_size = world_size
        # self.ds  = load_dataset("openwebtext", split="train", streaming=True)
        self.ds = dataset = load_dataset(
                                "HuggingFaceFW/fineweb-edu",
                                split="train",
                                streaming=True
                            )
        self.enc = tiktoken.get_encoding("gpt2")

    def __iter__(self):
        buffer = []
        for i, item in enumerate(self.ds):
            if i % self.world_size != self.rank:   # DDP sharding
                continue
            buffer.extend(self.enc.encode_ordinary(item["text"]))
            while len(buffer) >= self.seq_len + 1:
                chunk  = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]      # slide, keep tail
                yield (
                    torch.tensor(chunk[:self.seq_len],   dtype=torch.long),
                    torch.tensor(chunk[1:self.seq_len+1], dtype=torch.long),
                )


# =============================================================================
# Checkpoint helpers
# =============================================================================

def save_checkpoint(model, optimizer, step, loss, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    # unwrap DDP to get raw model state dict
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save({
        "step":       step,
        "loss":       loss,
        "model":      raw_model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }, path)
    print(f"  OK saved checkpoint -> {path}")


# def load_checkpoint(model, optimizer, path, device):

#     ckpt = torch.load(path, map_location=device)
#     raw_model = model.module if isinstance(model, DDP) else model
#     raw_model.load_state_dict(ckpt["model"])
#     optimizer.load_state_dict(ckpt["optimizer"])
#     print(f"  OK resumed from {path}  (step {ckpt['step']}, loss {ckpt['loss']:.4f})")
#     return ckpt["step"], ckpt["loss"]
def load_checkpoint(model, optimizer, path, device):
    ckpt = torch.load(path, map_location=device)
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(ckpt["model"])
    
    # optimizer state might fail if checkpoint is from different setup
    try:
        optimizer.load_state_dict(ckpt["optimizer"])
        print(f"  OK resumed optimizer state")
    except Exception as e:
        print(f"  WARNING: Could not load optimizer state: {e}")
        print(f"  Starting with fresh optimizer (weights still loaded)")
    
    print(f"  OK resumed from {path}  (step {ckpt['step']}, loss {ckpt['loss']:.4f})")
    return ckpt["step"], ckpt["loss"]

# def latest_checkpoint(ckpt_dir):

#     if not os.path.isdir(ckpt_dir):
#         return None
#     files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
#     if not files:
#         return None
#     # files named step_XXXXXX.pt -- pick highest step
#     files.sort(key=lambda f: int(f.replace("step_", "").replace(".pt", "")))
#     return os.path.join(ckpt_dir, files[-1])
def latest_checkpoint(ckpt_dir):
    # 👇 Check working checkpoints FIRST
    if os.path.isdir(ckpt_dir):
        files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        if files:
            def sort_key(f):
                try:
                    return int(f.replace("step_", "").replace("_final", "").replace(".pt", ""))
                except ValueError:
                    return 0
            files.sort(key=sort_key)
            path = os.path.join(ckpt_dir, files[-1])
            print(f"  Found working checkpoint: {path}")
            return path

    # 👇 Only fall back to input dataset if nothing in working dir
    if os.path.isfile(RESUME_CKPT):
        print(f"  Found resume checkpoint: {RESUME_CKPT}")
        return RESUME_CKPT

    return None

# =============================================================================
# DDP setup
# =============================================================================

def setup_ddp():
    dist.init_process_group("nccl")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


# =============================================================================
# Training loop
# =============================================================================

def train():
    rank, world_size = setup_ddp()
    device = torch.device("cuda", rank)
    is_master = (rank == 0)

    # -- Model ----------------------------------------------------------------
    model = GPT().to(device)
    model = torch.compile(model)
    if is_master:
        n = model.param_count()
        print("")
        print(f"Model parameters: {n:,} ({n/1e6:.1f}M)")
        print("")

    model = DDP(model, device_ids=[rank])

    # -- Optimizer ------------------------------------------------------------
    # Separate weight decay: apply only to weight matrices, NOT biases/norms
    decay_params   = [p for n, p in model.named_parameters()
                      if p.dim() >= 2 and p.requires_grad]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.dim() < 2  and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": decay_params,   "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=LR, betas=(BETA1, BETA2), fused=True)   # fused=True: faster on CUDA

    # -- Resume from checkpoint if one exists ---------------------------------
    start_step = 0
    ckpt_path  = latest_checkpoint(CKPT_DIR)
    if ckpt_path:
        start_step, _ = load_checkpoint(model, optimizer, ckpt_path, device)
        start_step += 1   # continue from next step

    # -- Dataset --------------------------------------------------------------
    dataset = StreamDataset(SEQ_LEN, rank=rank, world_size=world_size)
    loader  = DataLoader(dataset, batch_size=BATCH, pin_memory=True, num_workers=2)

    # -- Tokenizer for inference preview --------------------------------------
    enc = tiktoken.get_encoding("gpt2")

    # -- Training -------------------------------------------------------------
    step = start_step
    for x, y in loader:
        if step >= TOTAL_STEPS:
            break

        x, y = x.to(device), y.to(device)

        # set LR for this step
        current_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        # forward + loss
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss   = F.cross_entropy(
                logits.view(-1, VOCAB),
                y.view(-1),
                ignore_index=-1,
            )

        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # -- Logging ----------------------------------------------------------
        if step % LOG_EVERY == 0:
            ppl = math.exp(min(loss.item(), 20))  # cap to avoid overflow display
            print(f"step {step:>7}  loss {loss.item():.4f}  ppl {ppl:.1f}  lr {current_lr:.2e}")

            # quick inference to see if model is learning to talk
            if step > 0 and step % 7 == 0:
                raw = model.module
                prompt_ids = torch.tensor(
                    enc.encode("The meaning of life is"),
                    dtype=torch.long, device=device
                ).unsqueeze(0)
                out_ids = raw.generate(prompt_ids, max_new=40, temperature=0.8, top_k=40)
                print("  sample:", enc.decode(out_ids[0].tolist()))
            print()

        # -- Checkpoint save ---------------------------------------------------
        if is_master and step > 0 and step % SAVE_EVERY == 0:
            ckpt_file = os.path.join(CKPT_DIR, f"step_model.pt")
            save_checkpoint(model, optimizer, step, loss.item(), ckpt_file)

        step += 1

    # save final
    if is_master:
        save_checkpoint(model, optimizer, step,
                        loss.item(), os.path.join(CKPT_DIR, f"step_{step:07d}_final.pt"))
        print("Training complete.")


if __name__ == "__main__":
    train()
"""
with open("train.py","w") as file: 
    file.write(claude2)

!torchrun --nproc_per_node=2 /kaggle/working/train.py