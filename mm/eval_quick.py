"""Quick checks: tool calling + long-context retrieval. Run on base model, then with your adapter.

    python eval_quick.py
    python eval_quick.py --adapter outputs/qwen3-8b-mix-lora
    python eval_quick.py --adapter outputs/qwen3-8b-mix-lora --yarn --needle_tokens 60000
"""
import argparse
import json
import random
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-8B")
p.add_argument("--adapter", default=None)
p.add_argument("--yarn", action="store_true", help="extend context to ~128k with YaRN (check model card for your version)")
p.add_argument("--needle_tokens", type=int, default=30000)
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
cfg = AutoConfig.from_pretrained(args.model)
if args.yarn:
    cfg.rope_scaling = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
    cfg.max_position_embeddings = 131072
model = AutoModelForCausalLM.from_pretrained(
    args.model, config=cfg, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()


def gen(messages, tools=None, max_new=400, thinking=False):
    text = tok.apply_chat_template(messages, tools=tools, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=thinking)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)


# ---- 1. tool calling ----
tools = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string", "description": "City name"},
                                      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "calculate", "description": "Evaluate a math expression",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]
out = gen([{"role": "user", "content": "What's the weather in Paris and in Tokyo (celsius)? Also compute 17*23+19."}],
          tools=tools)
print("=== TOOL CALL TEST ===\n", out)
calls = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", out, re.S)
ok = 0
for c in calls:
    try:
        j = json.loads(c)
        ok += 1 if "name" in j and "arguments" in j else 0
    except Exception:
        pass
print(f"-> {ok}/{len(calls)} parseable tool calls (expected 3)\n")

# ---- 2. long-context needle test ----
random.seed(0)
secret = str(random.randint(10000, 99999))
n_lines = args.needle_tokens // 18
lines = [f"Log entry {i}: routine check finished with status code {random.randint(100, 999)}." for i in range(n_lines)]
lines.insert(len(lines) // 2, f"NOTE: The secret launch code is {secret}.")
prompt = "\n".join(lines) + "\n\nWhat is the secret launch code? Answer with just the number."
n_tok = len(tok(prompt)["input_ids"])
ans = gen([{"role": "user", "content": prompt}], max_new=30)
print(f"=== NEEDLE TEST ({n_tok} tokens) ===\nanswer: {ans.strip()}\nexpected: {secret}")
print("-> PASS" if secret in ans else "-> FAIL")
