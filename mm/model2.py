import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaConfig, LlamaForCausalLM
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
import tiktoken
import bitsandbytes as bnb
import faulthandler
from bpe_tokenizer import BPETokenizer, tokenize_all as bpe_tokenize_all

faulthandler.enable()

# ----------------------------
# 0. Hardware setup
# ----------------------------
assert torch.cuda.is_available(), "No CUDA GPU detected."
DEVICE = "cuda"
gpu_name = torch.cuda.get_device_name(0)
total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu_name} | VRAM: {total_vram:.1f} GB")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

USE_BF16 = torch.cuda.is_bf16_supported()
print(f"Using bf16: {USE_BF16}")

# ----------------------------
# 1. Tokenizer
# ----------------------------
# enc = tiktoken.get_encoding("cl100k_base")
tok = BPETokenizer("tokenizer_16k.bin")
VOCAB_SIZE = tok.total_vocab_size   # your trained vocab + 1 for EOS
EOS_TOKEN_ID = tok.eos_token_id

# ----------------------------
# 2. Model config (optimized for RTX 3050)
# ----------------------------
MODEL_CONFIG = dict(
    vocab_size=VOCAB_SIZE,
    hidden_size=512*2,               # Increased from 384
    intermediate_size=1280,        # Scaled with hidden_size
    num_hidden_layers=12,          # Increased from 8
    num_attention_heads=8,
    num_key_value_heads=4,         # GQA
    max_position_embeddings=512*2,
    rms_norm_eps=1e-5,
    tie_word_embeddings=True,
    attn_implementation="sdpa",
)
BLOCK_SIZE = 512
BATCH_SIZE = 4                   # Keep at 4 for stability on 6GB VRAM
GRAD_ACCUM_STEPS = 8
LR = 3e-4
EPOCHS = 3
CHECKPOINT_DIR = "./math_llama_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ----------------------------
# 3. Load ALL reasoning datasets
# ----------------------------
def load_all_reasoning_data():
    datasets_list = []
    failed_datasets = []

    # ===== MATH DATASETS =====
    math_datasets = [
        ("openai/gsm8k", "main", "train"),
        ("meta-math/MetaMathQA", None, "train"),
        ("DigitalLearningGmbH/MATH-lighteval", None, "train"),
        ("Tonic/MiniF2F", None, "train"),
    ]

    for dataset_name, config, split in math_datasets:
        try:
            print(f"Loading {dataset_name}...")
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
            datasets_list.append(ds)
        except Exception as e:
            print(f"  ⚠️  Failed to load {dataset_name}: {e}")
            failed_datasets.append(dataset_name)

    # ===== CHESS =====
    try:
        print("Loading Lichess chess puzzles...")
        chess = load_dataset("Lichess/chess-puzzles", split="train[:30000]")
        chess = chess.map(lambda x: {
            "text": f"Chess Position (FEN): {x['FEN']}\nBest move sequence: {x['Moves']}"
        })
        chess = chess.remove_columns([c for c in chess.column_names if c != "text"])
        datasets_list.append(chess)
    except Exception as e:
        print(f"  ⚠️  Failed to load Lichess/chess-puzzles: {e}")
        failed_datasets.append("Lichess/chess-puzzles")

    # ===== REASONING / SCIENCE =====
    reasoning_datasets = [
        ("allenai/ai2_arc", "ARC-Challenge", "train"),
        ("tau/commonsense_qa", None, "train"),
        ("allenai/openbookqa", None, "train"),
    ]

    for dataset_name, config, split in reasoning_datasets:
        try:
            print(f"Loading {dataset_name}...")
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
            datasets_list.append(ds)
        except Exception as e:
            print(f"  ⚠️  Failed to load {dataset_name}: {e}")
            failed_datasets.append(dataset_name)

    # ===== CONCATENATE & DEDUPLICATE =====
    if not datasets_list:
        raise ValueError("No datasets loaded! Check your internet connection.")

    print("\nConcatenating all datasets...")
    full_dataset = concatenate_datasets(datasets_list)
    print(f"Total examples before deduplication: {len(full_dataset):,}")

    print("Deduplicating dataset...")
    def normalize_text(text):
        text = re.sub(r'\d+', '', text.lower())
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()

    seen_normalized = set()
    deduplicated_texts = []

    for i, example in enumerate(full_dataset):
        text = example['text']
        if not text.strip():
            continue
        normalized = normalize_text(text)
        if normalized not in seen_normalized:
            seen_normalized.add(normalized)
            deduplicated_texts.append(text)
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1:,} examples...")

    duplicates_removed = len(full_dataset) - len(deduplicated_texts)
    print(f"Removed {duplicates_removed:,} duplicate/near-identical examples")
    print(f"Final unique examples: {len(deduplicated_texts):,}")

    if failed_datasets:
        print(f"\n⚠️  Warning: Failed to load {len(failed_datasets)} datasets: {failed_datasets}")

    return HFDataset.from_dict({"text": deduplicated_texts})


# ----------------------------
# 4. Load and prepare data
# ----------------------------
print("="*60)
print("LOADING ALL REASONING DATASETS")
print("="*60)
raw_data = load_all_reasoning_data()

# ----------------------------
# 5. Tokenize + chunk
# ----------------------------
def tokenize_all(dataset):
    all_ids = []
    for text in dataset["text"]:
        ids = tok.encode(text)
        ids.append(EOS_TOKEN_ID)
        all_ids.extend(ids)
    return all_ids

print("\nTokenizing corpus...")
all_token_ids = tokenize_all(raw_data)
print(f"Total tokens: {len(all_token_ids):,}")

total_len = (len(all_token_ids) // BLOCK_SIZE) * BLOCK_SIZE
all_token_ids = all_token_ids[:total_len]
print(f"Using {total_len:,} tokens for training")

class ChunkedDataset(Dataset):
    def __init__(self, token_ids, block_size):
        self.examples = [token_ids[i:i + block_size] for i in range(0, len(token_ids), block_size)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.examples[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}

train_dataset = ChunkedDataset(all_token_ids, BLOCK_SIZE)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    pin_memory=True, num_workers=2, drop_last=True
)

# ----------------------------
# 6. Model
# ----------------------------
config = LlamaConfig(**MODEL_CONFIG)
model = LlamaForCausalLM(config)
model.gradient_checkpointing_enable()
model.to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
embed_params = model.get_input_embeddings().weight.numel()
print(f"\nModel: {n_params/1e6:.1f}M params | Embeddings: {embed_params/1e6:.1f}M ({100*embed_params/n_params:.1f}%)")

try:
    model = torch.compile(model)
    print("✅ torch.compile enabled")
except Exception as e:
    print(f"⚠️  torch.compile unavailable: {e}")

# ----------------------------
# 7. Optimizer
# ----------------------------
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps)

amp_dtype = torch.bfloat16 if USE_BF16 else torch.float16
scaler = torch.cuda.amp.GradScaler(enabled=not USE_BF16)

# ----------------------------
# 8. Training loop
# ----------------------------
model.train()
global_step = 0
optimizer.zero_grad()

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

for epoch in range(EPOCHS):
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS

        if USE_BF16:
            loss.backward()
        else:
            scaler.scale(loss).backward()

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
                mem_used = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f} | "
                      f"VRAM: {mem_used:.2f}GB / {mem_reserved:.2f}GB")

        # import subprocess, time
        # if (step+1) % 600 == 0:
        #     print("cooling the system")
        #     subprocess.run(["nvidia-smi"])
        #     time.sleep(120)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n✅ Saved checkpoint: {ckpt_path}")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)

# ----------------------------
# 9. Quantize & save
# ----------------------------
model.eval()
model_cpu = model.to("cpu")
model_int8 = torch.quantization.quantize_dynamic(
    model_cpu, {torch.nn.Linear}, dtype=torch.qint8
)
torch.save(model_int8.state_dict(), os.path.join(CHECKPOINT_DIR, "model_int8.pt"))
print("✅ Saved int8 quantized model")

# ----------------------------
# 10. Inference test
# ----------------------------
model.to(DEVICE)
test_prompts = [
    "Question: If a train travels 60 miles in 1.5 hours, what is its speed?\nAnswer:",
    "Chess: What is the best move for White in this position: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O b5 Bb3 Be7?\nAnswer:",
    "Question: If 3x + 5 = 20, what is x? Show your work.\nAnswer:",
]

print("\n" + "="*60)
print("INFERENCE TESTS")
print("="*60)
for prompt in test_prompts:
    input_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long).to(DEVICE)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        output = model.generate(input_ids, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.7, eos_token_id=EOS_TOKEN_ID)
    print("\n" + "-"*60)
    print(f"PROMPT: {prompt}")
    print(f"OUTPUT: {tok.decode(output[0].tolist()).decode('utf-8', errors='replace')}")
    print("-"*60)

import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaConfig, LlamaForCausalLM
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
import tiktoken
import bitsandbytes as bnb
import faulthandler
from bpe_tokenizer import BPETokenizer, tokenize_all as bpe_tokenize_all

faulthandler.enable()

# ----------------------------
# 0. Hardware setup
# ----------------------------
assert torch.cuda.is_available(), "No CUDA GPU detected."
DEVICE = "cuda"
gpu_name = torch.cuda.get_device_name(0)
total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu_name} | VRAM: {total_vram:.1f} GB")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

USE_BF16 = torch.cuda.is_bf16_supported()
print(f"Using bf16: {USE_BF16}")

# ----------------------------
# 1. Tokenizer
# ----------------------------
# enc = tiktoken.get_encoding("cl100k_base")
tok = BPETokenizer("tokenizer_16k.bin")
VOCAB_SIZE = tok.total_vocab_size   # your trained vocab + 1 for EOS
EOS_TOKEN_ID = tok.eos_token_id

# ----------------------------
# 2. Model config (optimized for RTX 3050)
# ----------------------------
MODEL_CONFIG = dict(
    vocab_size=VOCAB_SIZE,
    hidden_size=512,               # Increased from 384
    intermediate_size=1280,        # Scaled with hidden_size
    num_hidden_layers=12,          # Increased from 8
    num_attention_heads=8,
    num_key_value_heads=4,         # GQA
    max_position_embeddings=512*2,
    rms_norm_eps=1e-5,
    tie_word_embeddings=True,
    attn_implementation="sdpa",
)
BLOCK_SIZE = 512
BATCH_SIZE = 4                   # Keep at 4 for stability on 6GB VRAM
GRAD_ACCUM_STEPS = 8
LR = 3e-4
EPOCHS = 3
CHECKPOINT_DIR = "./math_llama_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ----------------------------
# 3. Load ALL reasoning datasets
# ----------------------------
def load_all_reasoning_data():
    datasets_list = []
    failed_datasets = []

    # ===== MATH DATASETS =====
    math_datasets = [
        ("openai/gsm8k", "main", "train"),
        ("meta-math/MetaMathQA", None, "train"),
        ("DigitalLearningGmbH/MATH-lighteval", None, "train"),
        ("Tonic/MiniF2F", None, "train"),
    ]

    for dataset_name, config, split in math_datasets:
        try:
            print(f"Loading {dataset_name}...")
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
            datasets_list.append(ds)
        except Exception as e:
            print(f"  ⚠️  Failed to load {dataset_name}: {e}")
            failed_datasets.append(dataset_name)

    # ===== CHESS =====
    try:
        print("Loading Lichess chess puzzles...")
        chess = load_dataset("Lichess/chess-puzzles", split="train[:30000]")
        chess = chess.map(lambda x: {
            "text": f"Chess Position (FEN): {x['FEN']}\nBest move sequence: {x['Moves']}"
        })
        chess = chess.remove_columns([c for c in chess.column_names if c != "text"])
        datasets_list.append(chess)
    except Exception as e:
        print(f"  ⚠️  Failed to load Lichess/chess-puzzles: {e}")
        failed_datasets.append("Lichess/chess-puzzles")

    # ===== REASONING / SCIENCE =====
    reasoning_datasets = [
        ("allenai/ai2_arc", "ARC-Challenge", "train"),
        ("tau/commonsense_qa", None, "train"),
        ("allenai/openbookqa", None, "train"),
    ]

    for dataset_name, config, split in reasoning_datasets:
        try:
            print(f"Loading {dataset_name}...")
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
            datasets_list.append(ds)
        except Exception as e:
            print(f"  ⚠️  Failed to load {dataset_name}: {e}")
            failed_datasets.append(dataset_name)

    # ===== CONCATENATE & DEDUPLICATE =====
    if not datasets_list:
        raise ValueError("No datasets loaded! Check your internet connection.")

    print("\nConcatenating all datasets...")
    full_dataset = concatenate_datasets(datasets_list)
    print(f"Total examples before deduplication: {len(full_dataset):,}")

    print("Deduplicating dataset...")
    def normalize_text(text):
        text = re.sub(r'\d+', '', text.lower())
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()

    seen_normalized = set()
    deduplicated_texts = []

    for i, example in enumerate(full_dataset):
        text = example['text']
        if not text.strip():
            continue
        normalized = normalize_text(text)
        if normalized not in seen_normalized:
            seen_normalized.add(normalized)
            deduplicated_texts.append(text)
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1:,} examples...")

    duplicates_removed = len(full_dataset) - len(deduplicated_texts)
    print(f"Removed {duplicates_removed:,} duplicate/near-identical examples")
    print(f"Final unique examples: {len(deduplicated_texts):,}")

    if failed_datasets:
        print(f"\n⚠️  Warning: Failed to load {len(failed_datasets)} datasets: {failed_datasets}")

    return HFDataset.from_dict({"text": deduplicated_texts})


# ----------------------------
# 4. Load and prepare data
# ----------------------------
print("="*60)
print("LOADING ALL REASONING DATASETS")
print("="*60)
raw_data = load_all_reasoning_data()

# ----------------------------
# 5. Tokenize + chunk
# ----------------------------
def tokenize_all(dataset):
    all_ids = []
    for text in dataset["text"]:
        ids = tok.encode(text)
        ids.append(EOS_TOKEN_ID)
        all_ids.extend(ids)
    return all_ids

print("\nTokenizing corpus...")
all_token_ids = tokenize_all(raw_data)
print(f"Total tokens: {len(all_token_ids):,}")

total_len = (len(all_token_ids) // BLOCK_SIZE) * BLOCK_SIZE
all_token_ids = all_token_ids[:total_len]
print(f"Using {total_len:,} tokens for training")

class ChunkedDataset(Dataset):
    def __init__(self, token_ids, block_size):
        self.examples = [token_ids[i:i + block_size] for i in range(0, len(token_ids), block_size)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.examples[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}

train_dataset = ChunkedDataset(all_token_ids, BLOCK_SIZE)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    pin_memory=True, num_workers=2, drop_last=True
)

# ----------------------------
# 6. Model
# ----------------------------
config = LlamaConfig(**MODEL_CONFIG)
model = LlamaForCausalLM(config)
model.gradient_checkpointing_enable()
model.to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
embed_params = model.get_input_embeddings().weight.numel()
print(f"\nModel: {n_params/1e6:.1f}M params | Embeddings: {embed_params/1e6:.1f}M ({100*embed_params/n_params:.1f}%)")

try:
    model = torch.compile(model)
    print("✅ torch.compile enabled")
except Exception as e:
    print(f"⚠️  torch.compile unavailable: {e}")

# ----------------------------
# 7. Optimizer
# ----------------------------
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * EPOCHS
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=total_steps)

amp_dtype = torch.bfloat16 if USE_BF16 else torch.float16
scaler = torch.cuda.amp.GradScaler(enabled=not USE_BF16)

# ----------------------------
# 8. Training loop
# ----------------------------
model.train()
global_step = 0
optimizer.zero_grad()

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

for epoch in range(EPOCHS):
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS

        if USE_BF16:
            loss.backward()
        else:
            scaler.scale(loss).backward()

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
                mem_used = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item()*GRAD_ACCUM_STEPS:.4f} | "
                      f"VRAM: {mem_used:.2f}GB / {mem_reserved:.2f}GB")

        # import subprocess, time
        # if (step+1) % 600 == 0:
        #     print("cooling the system")
        #     subprocess.run(["nvidia-smi"])
        #     time.sleep(120)

    ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n✅ Saved checkpoint: {ckpt_path}")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)

# ----------------------------
# 9. Quantize & save
# ----------------------------
model.eval()
model_cpu = model.to("cpu")
model_int8 = torch.quantization.quantize_dynamic(
    model_cpu, {torch.nn.Linear}, dtype=torch.qint8
)
torch.save(model_int8.state_dict(), os.path.join(CHECKPOINT_DIR, "model_int8.pt"))
print("✅ Saved int8 quantized model")

# ----------------------------
# 10. Inference test
# ----------------------------
model.to(DEVICE)
test_prompts = [
    "Question: If a train travels 60 miles in 1.5 hours, what is its speed?\nAnswer:",
    "Chess: What is the best move for White in this position: e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O b5 Bb3 Be7?\nAnswer:",
    "Question: If 3x + 5 = 20, what is x? Show your work.\nAnswer:",
]

print("\n" + "="*60)
print("INFERENCE TESTS")
print("="*60)
for prompt in test_prompts:
    input_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long).to(DEVICE)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        output = model.generate(input_ids, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.7, eos_token_id=EOS_TOKEN_ID)
    print("\n" + "-"*60)
    print(f"PROMPT: {prompt}")
    print(f"OUTPUT: {tok.decode(output[0].tolist()).decode('utf-8', errors='replace')}")
    print("-"*60)