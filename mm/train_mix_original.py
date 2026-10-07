"""QLoRA fine-tune of Qwen3-8B on a mix of reasoning, tool-calling, code and long-context data.

Sources (all rendered with Qwen3's own chat template, so the format is exactly what the model expects):
  - open-r1/Mixture-of-Thoughts (math + code)    -> reasoning
  - Salesforce/xlam-function-calling-60k (gated)  -> tool calls
  - ise-uiuc/Magicoder-OSS-Instruct-75K           -> general coding
  - THUDM/LongAlign-10k                           -> long context (only examples <= --max_len fit)

Test run (1 GPU, ~50 examples, includes long ones to check memory):
    python train_mix.py --test --out outputs/test

Full run (8x A100 40GB):
    accelerate launch --num_processes 8 train_mix.py
"""
import argparse
import json
import re
import torch
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import PartialState
from trl import SFTTrainer, SFTConfig

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-8B")
p.add_argument("--out", default="outputs/qwen3-8b-mix-lora")
p.add_argument("--n_math", type=int, default=7500)
p.add_argument("--n_code_reason", type=int, default=7500)
p.add_argument("--n_tools", type=int, default=6000)
p.add_argument("--n_code", type=int, default=6000)
p.add_argument("--n_long", type=int, default=1000, help="upper bound; fewer will fit under --max_len")
p.add_argument("--max_len", type=int, default=16384)
p.add_argument("--reason_max_len", type=int, default=8192)
p.add_argument("--epochs", type=float, default=1)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--bs", type=int, default=1, help="per-GPU batch size")
p.add_argument("--accum", type=int, default=2)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--test", action="store_true", help="tiny run: 10 examples per source")
args = p.parse_args()
if args.test:
    args.n_math = args.n_code_reason = args.n_tools = args.n_code = args.n_long = 10

NP = 8
state = PartialState()
tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def log(*a):
    if state.is_main_process:
        print(*a, flush=True)


def render(msgs, tools=None):
    # .rstrip("\n") so the text ends exactly at <|im_end|> and the trainer does not add a second EOS
    return tok.apply_chat_template(msgs, tools=tools, tokenize=False).rstrip("\n")


def finish(d, limit, cap, name):
    """d has a 'text' column. Count tokens, drop too-long examples, keep up to `limit`."""
    d = d.map(lambda x: {"n_tok": len(tok(x["text"], add_special_tokens=False)["input_ids"])}, num_proc=NP)
    d = d.filter(lambda x: x["n_tok"] <= cap)
    if len(d) < limit:
        log(f"[warn] {name}: only {len(d)} examples fit in {cap} tokens (wanted {limit})")
    d = d.select(range(min(limit, len(d))))
    if len(d):
        avg = sum(d["n_tok"]) / len(d)
        log(f"[{name}] {len(d)} examples, avg {avg:.0f} tokens, max {max(d['n_tok'])}")
        log(f"--- sample from {name} ---\n{d[0]['text'][:800]}\n--------------------------")
    return d.select_columns(["text"])


# ---------- reasoning: Mixture-of-Thoughts ----------
def load_mot(subset, limit):
    d = load_dataset("open-r1/Mixture-of-Thoughts", subset, split="train")
    d = d.select_columns(["messages"]).shuffle(seed=args.seed)
    d = d.select(range(min(len(d), limit * 4)))  # pool; many traces are too long
    d = d.map(lambda x: {"text": render(x["messages"])}, remove_columns=["messages"], num_proc=NP)
    return finish(d, limit, args.reason_max_len, f"mot-{subset}")


# ---------- tool calls: xLAM (JSON strings -> OpenAI tool schema -> Qwen format) ----------
TYPE_MAP = {"int": "integer", "integer": "integer", "float": "number", "number": "number",
            "str": "string", "string": "string", "bool": "boolean", "boolean": "boolean",
            "list": "array", "array": "array", "tuple": "array", "set": "array",
            "dict": "object", "object": "object"}


def convert_tool(t):
    params = t.get("parameters", {}) or {}
    if "properties" in params:  # already JSON-schema style
        schema = params
    else:  # xLAM flat style: {"param": {"type": "int", "description": "...", "default": ...}}
        props, required = {}, []
        for k, v in params.items():
            if not isinstance(v, dict):
                v = {"type": str(v)}
            raw = str(v.get("type", "string")).lower()
            optional = "optional" in raw
            m = re.search(r"[a-z]+", raw.replace("optional", ""))
            props[k] = {"type": TYPE_MAP.get(m.group(0) if m else "", "string"),
                        "description": v.get("description", "")}
            if not optional and "default" not in v:
                required.append(k)
        schema = {"type": "object", "properties": props, "required": required}
    return {"type": "function",
            "function": {"name": t["name"], "description": t.get("description", ""), "parameters": schema}}


def xlam_to_text(x):
    try:
        tools = [convert_tool(t) for t in json.loads(x["tools"])]
        names = {t["function"]["name"] for t in tools}
        calls = []
        for a in json.loads(x["answers"]):
            if a["name"] not in names:
                return {"text": ""}  # known bad rows in the dataset: skip
            a_args = a.get("arguments", {})
            if isinstance(a_args, str):
                a_args = json.loads(a_args)
            calls.append({"type": "function", "function": {"name": a["name"], "arguments": a_args}})
        if not calls:
            return {"text": ""}
        msgs = [{"role": "user", "content": x["query"]},
                {"role": "assistant", "content": "", "tool_calls": calls}]
        return {"text": render(msgs, tools)}
    except Exception:
        return {"text": ""}


def load_xlam(limit):
    d = load_dataset("Salesforce/xlam-function-calling-60k", split="train").shuffle(seed=args.seed)
    d = d.select(range(min(len(d), int(limit * 1.4))))
    d = d.map(xlam_to_text, remove_columns=d.column_names, num_proc=NP)
    d = d.filter(lambda x: len(x["text"]) > 0)
    return finish(d, limit, args.max_len, "xlam-tools")


# ---------- general coding: Magicoder ----------
def load_magicoder(limit):
    d = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train").shuffle(seed=args.seed)
    d = d.select(range(min(len(d), int(limit * 1.3))))
    d = d.map(lambda x: {"text": render([{"role": "user", "content": x["problem"]},
                                         {"role": "assistant", "content": x["solution"]}])},
              remove_columns=d.column_names, num_proc=NP)
    return finish(d, limit, args.reason_max_len, "magicoder")


# ---------- long context: LongAlign ----------
def load_long(limit):
    d = load_dataset("THUDM/LongAlign-10k", split="train")
    maxc = args.max_len * 5  # cheap character pre-filter before exact token counting
    d = d.filter(lambda x: sum(len(m["content"]) for m in x["messages"]) <= maxc, num_proc=NP)
    d = d.shuffle(seed=args.seed)
    d = d.map(lambda x: {"text": render(x["messages"])}, remove_columns=d.column_names, num_proc=NP)
    return finish(d, limit, args.max_len, "longalign")


with state.main_process_first():  # rank 0 builds + caches; others reuse
    parts = [
        load_mot("math", args.n_math),
        load_mot("code", args.n_code_reason),
        load_xlam(args.n_tools),
        load_magicoder(args.n_code),
        load_long(args.n_long),
    ]
    ds = concatenate_datasets(parts).shuffle(seed=args.seed)
log(f"TOTAL training examples: {len(ds)}")

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map={"": state.process_index},  # one full copy per GPU (DDP)
)

lora = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

cfg = SFTConfig(
    output_dir=args.out,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.bs,
    gradient_accumulation_steps=args.accum,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_text_field="text",
    packing=False,
    max_length=args.max_len,  # older TRL versions call this max_seq_length
    logging_steps=5,
    save_steps=100,
    save_total_limit=2,
    ddp_find_unused_parameters=False,
    dataloader_num_workers=2,
    report_to="none",
    seed=args.seed,
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)
trainer.train()
trainer.save_model(args.out)  # LoRA adapter only
log("Saved adapter to", args.out)
