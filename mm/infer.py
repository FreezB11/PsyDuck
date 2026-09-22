import os

import torch
import tiktoken

from transformers import LlamaConfig, LlamaForCausalLM
from bpe_tokenizer import BPETokenizer, tokenize_all as bpe_tokenize_all

# ============================================================
# 1. Hardware
# ============================================================

assert torch.cuda.is_available(), "CUDA GPU not detected."

DEVICE = "cuda"

gpu_name = torch.cuda.get_device_name(0)
total_vram = (
    torch.cuda.get_device_properties(0).total_memory / 1e9
)

print(f"GPU: {gpu_name}")
print(f"VRAM: {total_vram:.1f} GB")


# ============================================================
# 2. CUDA configuration
# ============================================================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

USE_BF16 = torch.cuda.is_bf16_supported()

amp_dtype = (
    torch.bfloat16
    if USE_BF16
    else torch.float16
)

print(f"Using BF16: {USE_BF16}")
print(f"AMP dtype: {amp_dtype}")


# ============================================================
# 3. Tokenizer
# ============================================================

tok = BPETokenizer("tokenizer_16k.bin")
VOCAB_SIZE = tok.total_vocab_size   # your trained vocab + 1 for EOS
EOS_TOKEN_ID = tok.eos_token_id

print(f"Tokenizer: cl100k_base")
print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"EOS token ID: {EOS_TOKEN_ID}")


# ============================================================
# 4. Model configuration
#
# MUST exactly match train_math_llama.py
# (hidden_size=512, intermediate_size=1280, num_hidden_layers=12,
#  max_position_embeddings=512*2, tie_word_embeddings=True)
# ============================================================

MODEL_CONFIG = dict(
    vocab_size=VOCAB_SIZE,

    hidden_size=1024,
    intermediate_size=1280,

    num_hidden_layers=12,

    num_attention_heads=8,
    num_key_value_heads=4,

    max_position_embeddings=512 * 2,

    rms_norm_eps=1e-5,

    tie_word_embeddings=True,

    attn_implementation="sdpa",
)


# ============================================================
# 5. Create model
# ============================================================

print("\nCreating model architecture...")

config = LlamaConfig(**MODEL_CONFIG)

model = LlamaForCausalLM(config)

n_params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Model parameters: "
    f"{n_params / 1e6:.2f}M"
)


# ============================================================
# 6. Load checkpoint
# ============================================================

CHECKPOINT_PATH = (
    "math_llama_checkpoints/epoch_0.pt"
)

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

print("\nLoading checkpoint:")
print(f"  {CHECKPOINT_PATH}")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)


# ============================================================
# 7. Handle torch.compile() checkpoint
# ============================================================

# In training you did:
#
#     model = torch.compile(model)
#
# and then:
#
#     torch.save(model.state_dict(), ...)
#
# Therefore the checkpoint keys look like:
#
#     _orig_mod.model.layers.0...
#
# while the fresh model expects:
#
#     model.layers.0...
#
# Remove the prefix.

if any(
    key.startswith("_orig_mod.")
    for key in checkpoint.keys()
):
    print(
        "Detected torch.compile checkpoint."
    )
    print(
        "Removing '_orig_mod.' prefix..."
    )

    checkpoint = {
        key.removeprefix("_orig_mod."): value
        for key, value in checkpoint.items()
    }


# ============================================================
# 8. Load weights
# ============================================================

model.load_state_dict(
    checkpoint,
    strict=True
)

print("Checkpoint loaded successfully.")


# ============================================================
# 9. Move model to GPU
# ============================================================

model = model.to(DEVICE)

model.eval()

print(
    f"Model device: "
    f"{next(model.parameters()).device}"
)


# ============================================================
# 10. Prompts (a series of test questions)
# ============================================================

test_prompts = [
    "Question: If a train travels 60 miles in 1.5 hours, "
    "what is its speed?\nAnswer:",

    "Question: If 3x + 5 = 20, what is x? Show your work.\nAnswer:",

    "Question: What is the capital of France?\nAnswer:",

    "Question: A shirt costs $20 and is on sale for 25% off. "
    "What is the sale price?\nAnswer:",

    "Chess Position (FEN): rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\n"
    "Best move sequence:",

    "Question: Why do objects fall to the ground?\nAnswer:",

    "Question: What is 12 multiplied by 8?\nAnswer:",
]



# ============================================================
# 11-13. Tokenize, generate, and decode for each prompt
# ============================================================

for i, prompt in enumerate(test_prompts, start=1):

    print("\n" + "=" * 70)
    print(f"PROMPT {i}/{len(test_prompts)}")
    print("=" * 70)
    print(prompt)

    tokens = tok.encode(prompt)

    input_ids = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=DEVICE
    )

    print(f"Input tokens: {input_ids.shape[1]}")

    print("\nGenerating...\n")

    with torch.no_grad():

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype
        ):

            output = model.generate(
                input_ids,

                max_new_tokens=100,

                do_sample=True,

                top_p=0.9,
                temperature=0.4,

                eos_token_id=EOS_TOKEN_ID,
            )

    generated_text = tok.decode(
        output[0].tolist()
    ).decode('utf-8', errors='replace')

    print("-" * 70)
    print("OUTPUT")
    print("-" * 70)
    print(generated_text)
    print("-" * 70)


# ============================================================
# 14. VRAM usage
# ============================================================

allocated = (
    torch.cuda.memory_allocated() / 1e9
)

reserved = (
    torch.cuda.memory_reserved() / 1e9
)

print(
    f"\nVRAM: "
    f"{allocated:.2f} GB allocated / "
    f"{reserved:.2f} GB reserved"
)