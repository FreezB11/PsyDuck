"""
prepare_data.py

Loads all reasoning datasets, tags + deduplicates them, tokenizes the
result once, and writes the token stream to a single flat binary file
(uint32 little-endian token ids). This lets train_math_llama.py skip
dataset download + tokenization on every run - it just memory-maps the
.bin file, which loads in a fraction of a second regardless of corpus
size.

Usage:
    python3 prepare_data.py
    python3 prepare_data.py --out tokens.bin --tokenizer tokenizer_16k.bin

Output:
    tokens.bin        - raw uint32 token ids, one after another
    tokens.bin.meta   - small text file recording token count + dtype,
                        so train_math_llama.py can sanity-check it
                        without re-deriving anything.
"""

import os
import re
import time
import argparse
import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
from bpe_tokenizer import BPETokenizer

_START_TIME = time.time()


def log(msg=""):
    elapsed = time.time() - _START_TIME
    for line in str(msg).split("\n"):
        print(f"[{elapsed:8.1f}s] {line}")


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


def load_all_reasoning_data():
    datasets_list = []
    failed_datasets = []
    per_dataset_counts = {}

    def record_and_tag(ds, tag_key, name):
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

    if not datasets_list:
        raise ValueError("No datasets loaded! Check your internet connection.")

    log("Concatenating all datasets...")
    full_dataset = concatenate_datasets(datasets_list)
    log(f"Total examples before deduplication: {len(full_dataset):,}")

    log("Deduplicating dataset...")

    # Only normalize whitespace/case for the dedup key - NOT digits.
    # Stripping digits (as an earlier version of this pipeline did)
    # would treat distinct math problems that differ only in their
    # numbers as duplicates and silently drop them.
    def normalize_text(text):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    seen_normalized = set()
    deduplicated_texts = []
    removed_per_dataset = {name: 0 for name in per_dataset_counts}

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


def tokenize_all(dataset, tok, eos_token_id):
    all_ids = []
    for text in dataset["text"]:
        ids = tok.encode(text)
        ids.append(eos_token_id)
        all_ids.extend(ids)
    return all_ids


def main():
    parser = argparse.ArgumentParser(description="Tokenize the reasoning corpus and save it to a .bin file.")
    parser.add_argument("--tokenizer", default="tokenizer_16k.bin", help="Path to the BPE tokenizer file.")
    parser.add_argument("--out", default="tokens.bin", help="Output path for the tokenized corpus.")
    args = parser.parse_args()

    log(f"Loading tokenizer from {args.tokenizer}...")
    tok = BPETokenizer(args.tokenizer)
    eos_token_id = tok.eos_token_id
    vocab_size = tok.total_vocab_size

    log("=" * 60)
    log("LOADING ALL REASONING DATASETS")
    log("=" * 60)
    raw_data = load_all_reasoning_data()

    log("Tokenizing corpus...")
    _tokenize_start = time.time()
    all_token_ids = tokenize_all(raw_data, tok, eos_token_id)
    log(f"Total tokens: {len(all_token_ids):,} (tokenized in {time.time() - _tokenize_start:.1f}s)")

    # uint32 is plenty for a 16k-ish vocab (max id ~16385) and keeps the
    # file 2x smaller than int64 while numpy handles the dtype for us.
    arr = np.array(all_token_ids, dtype=np.uint32)

    log(f"Writing {arr.nbytes / 1e6:.1f} MB to {args.out}...")
    arr.tofile(args.out)

    meta_path = args.out + ".meta"
    with open(meta_path, "w") as f:
        f.write(f"num_tokens={len(arr)}\n")
        f.write(f"dtype=uint32\n")
        f.write(f"vocab_size={vocab_size}\n")
        f.write(f"eos_token_id={eos_token_id}\n")
        f.write(f"tokenizer={args.tokenizer}\n")

    log(f"Wrote metadata to {meta_path}")
    log("Done. Point train_math_llama.py at this .bin via TOKENS_BIN_PATH.")


if __name__ == "__main__":
    main()