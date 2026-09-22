import os
import re
import time
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaConfig, LlamaForCausalLM
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
import bitsandbytes as bnb
import faulthandler
from bpe_tokenizer import BPETokenizer, tokenize_all as bpe_tokenize_all

faulthandler.enable()

# ----------------------------
# -2. CLI args (resume support)
# ----------------------------
parser = argparse.ArgumentParser(description="Train the reasoning LLaMA model.")
parser.add_argument(
    "--resume", nargs="?", const="latest", default=None,
    help="Resume training from a checkpoint. Pass a path to a specific "
         "*_train_state.pt file, or omit the value to auto-resume from "
         "the most recent epoch found in CHECKPOINT_DIR."
)
parser.add_argument(
    "--tokens-bin", default="tokens.bin",
    help="Path to a pre-tokenized corpus produced by prepare_data.py. "
         "If it exists, it's memory-mapped and loaded instantly instead "
         "of re-downloading/re-tokenizing the full dataset."
)
cli_args = parser.parse_args()

# ----------------------------
# -1. Timestamped logging
# ----------------------------
# All prints go through log() so every line is prefixed with elapsed
# seconds since the script started. This makes it easy to see how long
# data loading, tokenizing, and each training epoch actually take.
_START_TIME = time.time()


def log(msg=""):
    elapsed = time.time() - _START_TIME
    # e.g. "[   12.3s] Loading openai/gsm8k..."
    for line in str(msg).split("\n"):
        print(f"[{elapsed:8.1f}s] {line}")


# ----------------------------
# 0. Hardware setup
# ----------------------------
assert torch.cuda.is_available(), "No CUDA GPU detected."
DEVICE = "cuda"
gpu_name = torch.cuda.get_device_name(0)
total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
log(f"GPU: {gpu_name} | VRAM: {total_vram:.1f} GB")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

USE_BF16 = torch.cuda.is_bf16_supported()
log(f"Using bf16: {USE_BF16}")

# ----------------------------
# 1. Tokenizer
# ----------------------------
tok = BPETokenizer("tokenizer_16k.bin")
VOCAB_SIZE = tok.total_vocab_size   # your trained vocab + 1 for EOS
EOS_TOKEN_ID = tok.eos_token_id

# ----------------------------
# 2. Model config (optimized for RTX 3050, 6GB VRAM)
# ----------------------------
MODEL_CONFIG = dict(
    vocab_size=VOCAB_SIZE,
    hidden_size=512*2,
    intermediate_size=1280,
    num_hidden_layers=12,
    num_attention_heads=8,
    num_key_value_heads=4,          # GQA
    max_position_embeddings=512 * 2,
    rms_norm_eps=1e-5,
    tie_word_embeddings=True,
    attn_implementation="sdpa",
)

BLOCK_SIZE = 512

# Previously BATCH_SIZE=4 + grad checkpointing left the 6GB card almost
# completely idle (~1GB used) because this model is only ~40-100M params.
# Gradient checkpointing trades VRAM for extra compute (it recomputes
# activations on the backward pass) - useful when you're VRAM-starved,
# actively harmful to speed when you're not. We're not badly VRAM
# constrained, so checkpointing is removed below.
#
# BATCH_SIZE is intentionally a more conservative starting point (not
# maxed out) because removing gradient checkpointing changes activation
# memory in a way that's hard to predict precisely without just running
# it. A probe step below will auto-halve this (and compensate with
# GRAD_ACCUM_STEPS) if it doesn't fit, so this is a safe starting value
# to try first rather than a guaranteed-safe one.
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 1             # effective batch size = 12 * 4 = 48
LR = 3e-4
EPOCHS = 3
CHECKPOINT_DIR = "./math_llama_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Each task gets an explicit tag so the model has a cheap, reliable signal
# for which response format/domain it's in, instead of having to infer
# task type purely from question phrasing. This is prepended to every
# example's text.
TASK_TAGS = {
    "gsm8k": "[MATH-WORD]",
    "metamathqa": "[MATH-WORD]",
    "math_lighteval": "[MATH-SYMBOLIC]",
    "minif2f": "[MATH-PROOF]",
    "chess": "[CHESS]",
    "arc": "[SCIENCE-MC]",
    "commonsense_qa": "[COMMONSENSE-MC]",
    "openbookqa": "[SCIENCE-MC]",
}


# ----------------------------
# 3. Load ALL reasoning datasets
# ----------------------------
def load_all_reasoning_data():
    datasets_list = []
    failed_datasets = []
    # Track token count per dataset (pre-dedup) so we can see where our
    # training tokens are actually going.
    per_dataset_counts = {}

    def record_and_tag(ds, tag_key, name):
        """Prefix every example's text with its task tag and stash a
        pre-dedup example count for later reporting."""
        tag = TASK_TAGS[tag_key]
        ds = ds.map(lambda x: {"text": f"{tag} {x['text']}".strip()})
        per_dataset_counts[name] = len(ds)
        return ds

    # ===== MATH DATASETS =====
    math_datasets = [
        ("openai/gsm8k", "main", "train", "gsm8k"),
        ("meta-math/MetaMathQA", None, "train", "metamathqa"),
        ("DigitalLearningGmbH/MATH-lighteval", None, "train", "math_lighteval"),
        ("Tonic/MiniF2F", None, "train", "minif2f"),
    ]

    for dataset_name, config, split, tag_key in math_datasets:
        try:
            log(f"Loading {dataset_name}...")
            ds = load_dataset(dataset_name, config, split=split) if config else load_dataset(dataset_name, split=split)
            if isinstance(ds, dict):
                ds = ds[split]

            if dataset_name == "openai/gsm8k":
                ds = ds.map(lambda x: {"text": f"Question: {x['question']}\nAnswer: {x['answer']}"})
            elif dataset_name == "meta-math/MetaMathQA":
                ds = ds.map(lambda x: {"text": f"Question: {x['query']}\nAnswer: {x['response']}"})
            elif dataset_name == "DigitalLearningGmbH/MATH-lighteval":
                ds = ds.map(lambda x: {"text": f"Question: {x.get('problem', '')}\nAnswer: {x.get('solution', '')}"})
            elif dataset_name == "Tonic/MiniF2F":
                def format_minif2f(example):
                    conv = example.get('conversation', [])
                    if len(conv) >= 2:
                        question = conv[0].get('content', '')
                        answer = conv[-1].get('content', '')
                        return {"text": f"Question: {question}\nAnswer: {answer}"}
                    return {"text": ""}
                ds = ds.map(format_minif2f)
                ds = ds.filter(lambda x: len(x['text'].strip()) > 20)

            ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
            ds = record_and_tag(ds, tag_key, dataset_name)
            datasets_list.append(ds)
            log(f"  Loaded {dataset_name}: {len(ds):,} examples")
        except Exception as e:
            log(f"  WARNING: Failed to load {dataset_name}: {e}")
            failed_datasets.append(dataset_name)

    # ===== CHESS =====
    try:
        log("Loading Lichess chess puzzles...")
        chess = load_dataset("Lichess/chess-puzzles", split="train[:30000]")
        chess = chess.map(lambda x: {
            "text": f"Chess Position (FEN): {x['FEN']}\nBest move sequence: {x['Moves']}"
        })
        chess = chess.remove_columns([c for c in chess.column_names if c != "text"])
        chess = record_and_tag(chess, "chess", "Lichess/chess-puzzles")
        datasets_list.append(chess)
        log(f"  Loaded Lichess/chess-puzzles: {len(chess):,} examples")
    except Exception as e:
        log(f"  WARNING: Failed to load Lichess/chess-puzzles: {e}")
        failed_datasets.append("Lichess/chess-puzzles")

    # ===== REASONING / SCIENCE =====
    reasoning_datasets = [
        ("allenai/ai2_arc", "ARC-Challenge", "train", "arc"),
        ("tau/commonsense_qa", None, "train", "commonsense_qa"),
        ("allenai/openbookqa", None, "train", "openbookqa"),
    ]

    for dataset_name, config, split, tag_key in reasoning_datasets:
        try:
            log(f"Loading {dataset_name}...")
            slice_split = f"{split}[:20000]"
            ds = load_dataset(dataset_name, config, split=slice_split) if config else load_dataset(dataset_name, split=slice_split)

            if dataset_name == "allenai/ai2_arc":
                ds = ds.map(lambda x: {
                    "text": f"Question: {x['question']}\n"
                            f"Choices: {', '.join(x['choices']['text'])}\n"
                            f"Answer: {x['answerKey']}"
                })
            elif dataset_name == "tau/commonsense_qa":
                ds = ds.map(lambda x: {
                    "text": f"Question: {x['question']}\n"
                            f"Choices: {', '.join(x['choices']['text'])}\n"
                            f"Answer: {x['answerKey']}"
                })
            elif dataset_name == "allenai/openbookqa":
                ds = ds.map(lambda x: {
                    "text": f"Question: {x['question_stem']}\n"
                            f"Choices: {', '.join(x['choices']['text'])}\n"
                            f"Answer: {x['answerKey']}"
                })

            ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
            ds = record_and_tag(ds, tag_key, dataset_name)
            datasets_list.append(ds)
            log(f"  Loaded {dataset_name}: {len(ds):,} examples")
        except Exception as e:
            log(f"  WARNING: Failed to load {dataset_name}: {e}")
            failed_datasets.append(dataset_name)

    # ===== CONCATENATE & DEDUPLICATE =====
    if not datasets_list:
        raise ValueError("No datasets loaded! Check your internet connection.")

    log("Concatenating all datasets...")
    full_dataset = concatenate_datasets(datasets_list)
    log(f"Total examples before deduplication: {len(full_dataset):,}")

    log("Deduplicating dataset...")

    # --- FIX: the previous normalize_text() stripped ALL digits before
    # hashing, which meant two genuinely different math problems that
    # only differ in their numbers (extremely common in GSM8K /
    # MetaMathQA, since problems are frequently templated) were treated
    # as duplicates and silently dropped. That was quietly deleting a
    # large chunk of exactly the math data we most want to keep.
    #
    # New behaviour: only normalize whitespace/case for the dedup key.
    # Numbers and punctuation are kept, so distinct problems with
    # different numbers are correctly treated as distinct examples. We
    # also track how many examples were removed FROM EACH dataset so
    # it's visible if one source is being gutted by the dedup step.
    def normalize_text(text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    seen_normalized = set()
    deduplicated_texts = []
    removed_per_dataset = {name: 0 for name in per_dataset_counts}

    # Rebuild a per-example dataset-name lookup so we can attribute
    # removals correctly. concatenate_datasets preserves order, so we
    # can walk the same boundaries we recorded above.
    boundaries = []
    running = 0
    for ds in datasets_list:
        boundaries.append((running, running + len(ds)))
        running += len(ds)
    names_in_order = list(per_dataset_counts.keys())

    def name_for_index(i):
        for (start, end), name in zip(boundaries, names_in_order):
            if start <= i < end:
                return name
        return "unknown"

    for i, example in enumerate(full_dataset):
        text = example['text']
        if not text.strip():
            continue
        normalized = normalize_text(text)
        if normalized not in seen_normalized:
            seen_normalized.add(normalized)
            deduplicated_texts.append(text)
        else:
            removed_per_dataset[name_for_index(i)] += 1
        if (i + 1) % 10000 == 0:
            log(f"  Processed {i + 1:,} examples...")

    duplicates_removed = len(full_dataset) - len(deduplicated_texts)
    log(f"Removed {duplicates_removed:,} duplicate/near-identical examples")
    log(f"Final unique examples: {len(deduplicated_texts):,}")

    log("Examples kept per dataset (before -> after dedup):")
    for name in names_in_order:
        before = per_dataset_counts[name]
        removed = removed_per_dataset[name]
        after = before - removed
        log(f"  {name:40s} {before:>7,} -> {after:>7,}  (-{removed:,})")

    if failed_datasets:
        log(f"WARNING: Failed to load {len(failed_datasets)} datasets: {failed_datasets}")

    return HFDataset.from_dict({"text": deduplicated_texts})


# ----------------------------
# 4-5. Load tokens: prefer a pre-tokenized .bin, else tokenize fresh
# ----------------------------
# prepare_data.py writes tokens.bin (+ tokens.bin.meta) once. On every
# later run we just memory-map that file, which loads in a fraction of
# a second regardless of corpus size - no dataset re-download, no
# re-tokenizing. If it's missing, we fall back to the original
# load-then-tokenize pipeline (slow, but no separate step required).
meta_path = cli_args.tokens_bin + ".meta"

if os.path.exists(cli_args.tokens_bin) and os.path.exists(meta_path):
    log(f"Found pre-tokenized corpus at {cli_args.tokens_bin} - loading via memmap")
    meta = {}
    with open(meta_path) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                meta[k] = v
    bin_vocab_size = int(meta.get("vocab_size", VOCAB_SIZE))
    if bin_vocab_size != VOCAB_SIZE:
        log(f"WARNING: tokens.bin was built with vocab_size={bin_vocab_size}, "
            f"current tokenizer has vocab_size={VOCAB_SIZE}. This likely means "
            f"tokens.bin is stale for the current tokenizer_16k.bin - consider "
            f"re-running prepare_data.py.")
    all_token_ids = np.memmap(cli_args.tokens_bin, dtype=np.uint32, mode="r")
    log(f"Total tokens (from .bin): {len(all_token_ids):,}")
else:
    log("No pre-tokenized corpus found at " + cli_args.tokens_bin)
    log("Falling back to the SMALL curated-only dataset pipeline built into "
        "this script (tens of millions of tokens, NOT the 1-2B token corpus).")
    log("For the full large-scale corpus (curated datasets + open-web-math, "
        "~1-2B tokens), run `python3 prepare_data.py` first, then re-run this "
        "script - it will pick up tokens.bin automatically.")
    log("=" * 60)
    log("LOADING ALL REASONING DATASETS (small fallback corpus)")
    log("=" * 60)
    raw_data = load_all_reasoning_data()

    def tokenize_all(dataset):
        all_ids = []
        for text in dataset["text"]:
            ids = tok.encode(text)
            ids.append(EOS_TOKEN_ID)
            all_ids.extend(ids)
        return all_ids

    log("Tokenizing corpus...")
    _tokenize_start = time.time()
    all_token_ids = np.array(tokenize_all(raw_data), dtype=np.uint32)
    log(f"Total tokens: {len(all_token_ids):,} (tokenized in {time.time() - _tokenize_start:.1f}s)")

total_len = (len(all_token_ids) // BLOCK_SIZE) * BLOCK_SIZE
all_token_ids = all_token_ids[:total_len]
log(f"Using {total_len:,} tokens for training")


class ChunkedDataset(Dataset):
    """Wraps a flat token array (numpy array or memmap) as fixed-size
    blocks without copying the whole corpus into a Python list of
    tensors up front - each __getitem__ slices directly from the
    underlying array, which keeps memory flat and startup instant even
    for large memmapped corpora."""

    def __init__(self, token_ids, block_size):
        self.token_ids = token_ids
        self.block_size = block_size
        self.num_blocks = len(token_ids) // block_size

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        # np.memmap/np.array -> plain int64 tensor for embedding lookups
        chunk = np.asarray(self.token_ids[start:end], dtype=np.int64)
        ids = torch.from_numpy(chunk)
        return {"input_ids": ids, "labels": ids.clone()}


train_dataset = ChunkedDataset(all_token_ids, BLOCK_SIZE)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    pin_memory=True, num_workers=2, drop_last=True
)
log(f"Train batches per epoch: {len(train_loader):,} (batch_size={BATCH_SIZE})")

# ----------------------------
# 6. Model
# ----------------------------
config = LlamaConfig(**MODEL_CONFIG)
model = LlamaForCausalLM(config)
# NOTE: gradient checkpointing intentionally NOT enabled. It trades
# VRAM for extra recomputation during backward, which only helps when
# you're VRAM-constrained. We removed it to get faster steps, but since
# we can't know activation memory exactly ahead of time, the probe step
# below actually tests the chosen batch size on real hardware before
# committing to the full run, and falls back safely if it doesn't fit.
model.to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
embed_params = model.get_input_embeddings().weight.numel()
log(f"Model: {n_params/1e6:.1f}M params | Embeddings: {embed_params/1e6:.1f}M ({100*embed_params/n_params:.1f}%)")

try:
    model = torch.compile(model)
    log("torch.compile enabled")
except Exception as e:
    log(f"torch.compile unavailable: {e}")

# ----------------------------
# 7. Optimizer
# ----------------------------
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=0.01)

amp_dtype = torch.bfloat16 if USE_BF16 else torch.float16
scaler = torch.cuda.amp.GradScaler(enabled=not USE_BF16)


# ----------------------------
# 7a. Resume from checkpoint
# ----------------------------
# Checkpoints are saved as a single dict per epoch containing model
# weights, optimizer state, and bookkeeping (epoch/global_step), so
# resuming continues training with the optimizer's momentum/variance
# state intact rather than just reloading weights and starting the
# optimizer from scratch.
start_epoch = 0
resumed_global_step = 0


def strip_compile_prefix(state_dict):
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    return state_dict


def find_latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = []
    for fname in os.listdir(checkpoint_dir):
        m = re.match(r"epoch_(\d+)_train_state\.pt$", fname)
        if m:
            candidates.append((int(m.group(1)), os.path.join(checkpoint_dir, fname)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]  # highest epoch number


if cli_args.resume is not None:
    resume_path = cli_args.resume
    if resume_path == "latest":
        resume_path = find_latest_checkpoint(CHECKPOINT_DIR)
        if resume_path is None:
            log(f"WARNING: --resume was given but no *_train_state.pt "
                f"checkpoints were found in {CHECKPOINT_DIR}. Starting fresh.")

    if resume_path is not None:
        log(f"Loading checkpoint: {resume_path}")
        loaded = torch.load(resume_path, map_location="cpu")

        # Two possible checkpoint shapes can show up here:
        #   1. A full train_state dict (what this script itself saves as
        #      epoch_N_train_state.pt): has "model_state_dict",
        #      "optimizer_state_dict", "epoch", etc. True resume: model,
        #      optimizer momentum/variance, and epoch/step all continue.
        #   2. A plain weights-only state_dict (e.g. an old epoch_N.pt,
        #      or any file saved via torch.save(model.state_dict())):
        #      its top-level keys are tensor names directly, no wrapper
        #      dict. There's no optimizer state to restore here, so this
        #      is really "initialize from these weights" rather than a
        #      true resume - epoch/global_step start over from 0.
        is_train_state = isinstance(loaded, dict) and "model_state_dict" in loaded

        if is_train_state:
            train_state = loaded
            model_state = strip_compile_prefix(train_state["model_state_dict"])
            target = model._orig_mod if hasattr(model, "_orig_mod") else model
            target.load_state_dict(model_state, strict=True)

            optimizer.load_state_dict(train_state["optimizer_state_dict"])

            start_epoch = train_state["epoch"] + 1  # continue from the NEXT epoch
            resumed_global_step = train_state.get("global_step", 0)

            if "batch_size" in train_state:
                BATCH_SIZE = train_state["batch_size"]
            if "grad_accum_steps" in train_state:
                GRAD_ACCUM_STEPS = train_state["grad_accum_steps"]

            log(f"Resumed at epoch {start_epoch} (global_step={resumed_global_step}, "
                f"batch_size={BATCH_SIZE}, grad_accum_steps={GRAD_ACCUM_STEPS})")
        else:
            log(f"NOTE: {resume_path} is a weights-only checkpoint (no optimizer "
                f"state, no epoch/step bookkeeping) - not a *_train_state.pt file. "
                f"Loading it as a starting point for the model only. This is NOT "
                f"a true resume: the optimizer starts fresh and epoch/global_step "
                f"start over from 0, so the LR schedule and epoch count in "
                f"CHECKPOINT_DIR filenames will restart too.")
            model_state = strip_compile_prefix(loaded)
            target = model._orig_mod if hasattr(model, "_orig_mod") else model
            target.load_state_dict(model_state, strict=True)
            log("Weights loaded successfully. Starting fresh optimizer/epoch count.")


# ----------------------------
# 7b. OOM probe
# ----------------------------
# Run one real forward+backward pass at the configured BATCH_SIZE
# before committing to the full training loop, to see whether it fits
# in VRAM. If it OOMs, halve BATCH_SIZE and double GRAD_ACCUM_STEPS to
# keep the effective batch size roughly constant, clear the CUDA
# cache, and try again. This avoids discovering an OOM only after
# minutes of data loading/tokenizing.
#
# NOTE: this intentionally does NOT call optimizer.step(). Doing so
# would apply a real update from a garbage dummy-data gradient, which
# would corrupt a freshly-initialized model's starting point and,
# worse, corrupt a resumed optimizer's momentum/variance state right
# after loading it. We only need forward+backward to see peak memory;
# gradients are discarded afterwards without ever stepping.
def probe_batch_size(model, optimizer, batch_size, block_size, device, amp_dtype, use_bf16):
    dummy_input = torch.randint(
        0, VOCAB_SIZE, (batch_size, block_size), dtype=torch.long, device=device
    )
    dummy_labels = dummy_input.clone()
    try:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(input_ids=dummy_input, labels=dummy_labels)
            loss = outputs.loss
        if use_bf16:
            loss.backward()
        else:
            scaler.scale(loss).backward()
        # Discard the dummy gradients without stepping the optimizer.
        optimizer.zero_grad(set_to_none=True)
        return True
    except torch.cuda.OutOfMemoryError:
        optimizer.zero_grad(set_to_none=True)
        return False
    finally:
        del dummy_input, dummy_labels
        torch.cuda.empty_cache()


log("Probing batch size against actual GPU memory before full training run...")
while True:
    mem_before = torch.cuda.memory_allocated() / 1e9
    ok = probe_batch_size(model, optimizer, BATCH_SIZE, BLOCK_SIZE, DEVICE, amp_dtype, USE_BF16)
    mem_after_peak = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    if ok:
        log(f"  BATCH_SIZE={BATCH_SIZE} OK (peak {mem_after_peak:.2f}GB, "
            f"baseline was {mem_before:.2f}GB) - proceeding with training")
        break

    if BATCH_SIZE <= 1:
        raise RuntimeError(
            "OOM even at BATCH_SIZE=1. The model/BLOCK_SIZE combination "
            "doesn't fit in available VRAM. Reduce BLOCK_SIZE or model size."
        )

    old_batch, old_accum = BATCH_SIZE, GRAD_ACCUM_STEPS
    BATCH_SIZE = max(1, BATCH_SIZE // 2)
    GRAD_ACCUM_STEPS = old_accum * (old_batch // BATCH_SIZE)
    log(f"  OOM at batch_size={old_batch} - falling back to "
        f"batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM_STEPS} "
        f"(effective batch size unchanged: {BATCH_SIZE * GRAD_ACCUM_STEPS})")

# Rebuild the DataLoader in case BATCH_SIZE changed during probing.
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    pin_memory=True, num_workers=2, drop_last=True
)
log(f"Final train batches per epoch: {len(train_loader):,} (batch_size={BATCH_SIZE})")

total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps)

# If resuming, fast-forward the scheduler to where it left off so the
# LR curve continues smoothly instead of restarting the OneCycle ramp.
if resumed_global_step > 0:
    for _ in range(min(resumed_global_step, total_steps - 1)):
        scheduler.step()
    log(f"Fast-forwarded scheduler by {resumed_global_step} steps")

# ----------------------------
# 8. Training loop
# ----------------------------
model.train()
global_step = resumed_global_step
optimizer.zero_grad()

log("=" * 60)
log("STARTING TRAINING" if start_epoch == 0 else f"RESUMING TRAINING FROM EPOCH {start_epoch}")
log("=" * 60)

if start_epoch >= EPOCHS:
    log(f"start_epoch={start_epoch} >= EPOCHS={EPOCHS}: this checkpoint already "
        f"completed all configured epochs. Increase EPOCHS if you want to train "
        f"further, otherwise there's nothing more to do.")

_last_log_time = time.time()

for epoch in range(start_epoch, EPOCHS):
    epoch_start = time.time()
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        try:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / GRAD_ACCUM_STEPS

            if USE_BF16:
                loss.backward()
            else:
                scaler.scale(loss).backward()
        except torch.cuda.OutOfMemoryError:
            # A mid-training OOM (e.g. from memory fragmentation after
            # many steps) shouldn't kill hours of progress. Skip this
            # batch, clear gradients/cache, and continue.
            log(f"  OOM on step {step} (epoch {epoch}) - skipping this batch")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            if USE_BF16:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 50 == 0:
                now = time.time()
                steps_per_sec = 50 / (now - _last_log_time) if now > _last_log_time else 0.0
                _last_log_time = now
                mem_used = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                log(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f} | "
                    f"{steps_per_sec:.2f} steps/s | "
                    f"VRAM: {mem_used:.2f}GB / {mem_reserved:.2f}GB")

    epoch_time = time.time() - epoch_start

    # Save a full training-state checkpoint (not just weights) so
    # --resume can continue with the optimizer's momentum/variance
    # state intact, plus enough bookkeeping to reconstruct the run.
    underlying_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    train_state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": underlying_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "model_config": MODEL_CONFIG,
    }
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}_train_state.pt")
    torch.save(train_state, ckpt_path)

    # Also save a plain weights-only file for infer.py, which doesn't
    # need optimizer state and expects a bare state_dict.
    weights_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pt")
    torch.save(underlying_model.state_dict(), weights_path)

    log(f"Saved checkpoint: {ckpt_path} and {weights_path} (epoch took {epoch_time:.1f}s)")

log("=" * 60)
log("TRAINING COMPLETE")
log("=" * 60)

# ----------------------------
# 9. Quantize & save
# ----------------------------
model.eval()
model_cpu = model.to("cpu")
model_int8 = torch.quantization.quantize_dynamic(
    model_cpu, {torch.nn.Linear}, dtype=torch.qint8
)
torch.save(model_int8.state_dict(), os.path.join(CHECKPOINT_DIR, "model_int8.pt"))
log("Saved int8 quantized model")

# ----------------------------
# 10. Inference test
# ----------------------------
model.to(DEVICE)
test_prompts = [
    "[MATH-WORD] Question: If a train travels 60 miles in 1.5 hours, what is its speed?\nAnswer:",
    "[CHESS] Chess Position (FEN): rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\nBest move sequence:",
    "[MATH-WORD] Question: If 3x + 5 = 20, what is x? Show your work.\nAnswer:",
]

log("=" * 60)
log("INFERENCE TESTS")
log("=" * 60)
for prompt in test_prompts:
    input_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long).to(DEVICE)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        output = model.generate(input_ids, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.7, eos_token_id=EOS_TOKEN_ID)
    log("-" * 60)
    log(f"PROMPT: {prompt}")
    log(f"OUTPUT: {tok.decode(output[0].tolist()).decode('utf-8', errors='replace')}")
    log("-" * 60)