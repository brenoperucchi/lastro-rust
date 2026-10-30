#!/usr/bin/env python3
"""Gera os textos dos traços de roteamento (M0.5): cada modelo responde aos prompts de tools/ref-texts/prompts-traco.json pelo
llama-server local (GPU, greedy) e a conversa é gravada já no formato de chat do modelo, para o ref_logits.py --experts-only.

  tools/.venv-ref/bin/python -I tools/gen_trace_texts.py --model <dir HF> --server http://127.0.0.1:18080 --out <dir>
"""
import argparse
import json
import os
import urllib.request

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "ref-texts", "prompts-traco.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    os.makedirs(args.out, exist_ok=True)
    for i, prompt in enumerate(json.load(open(args.prompts, encoding="utf-8"))):
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "seed": 1,
        }
        req = urllib.request.Request(
            args.server + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            reply = json.load(r)["choices"][0]["message"]["content"]
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}], tokenize=False
        )
        path = os.path.join(args.out, f"conversa-{i}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(path, len(tok(text).input_ids), "tokens")


if __name__ == "__main__":
    main()
