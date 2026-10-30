#!/usr/bin/env python3
"""Referência de geração gulosa do M1b (plano docs/PLANO-runtime-rust-moe-20261029.md).

Para cada prompt de `prompts-traco.json` (com o chat template do modelo), gera N tokens gulosos no transformers (CPU, float32 a
partir dos pesos BF16, cache KV ligado) e grava em JSON os ids do prompt, os ids gerados, o texto gerado e, por passo, a
diferença entre o primeiro e o segundo logit (para separar divergência real de empate):

  tools/.venv-ref/bin/python -I tools/ref_greedy.py --model <dir> --prompts tools/ref-texts/prompts-traco.json \
      --count 3 --new-tokens 64 --out qual-runs/REF-greedy-granite-1b/greedy.json
"""
import argparse
import json
import os

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def greedy(model, ids, new_tokens, pad_id):
    """Gera `new_tokens` tokens gulosos a partir de `ids` [1, T]; devolve os ids gerados e, por passo, o gap top1-top2."""
    # Máscara explícita: sem ela, com pad_token_id = 0, o generate mascara todo token 0 do prompt, e no Granite o 0 também é
    # o <|end_of_text|> que o chat template põe entre os turnos.
    with torch.inference_mode():
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            eos_token_id=None,
            pad_token_id=pad_id,
        )
    gen = out.sequences[0, ids.shape[1] :].tolist()
    gaps = []
    for s in out.scores:
        top = torch.topk(s[0].float(), 2).values
        gaps.append(float(top[0] - top[1]))
    return gen, gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True, dtype=torch.float32).eval()
    prompts = json.load(open(args.prompts, encoding="utf-8"))[: args.count]
    cases = []
    for text in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, return_tensors="pt")
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"]
        gen, gaps = greedy(model, ids, args.new_tokens, tok.pad_token_id)
        cases.append({"prompt": text, "prompt_ids": ids[0].tolist(), "generated_ids": gen, "generated_text": tok.decode(gen), "top_gap": gaps})
        print(f"prompt {len(cases)}: {ids.shape[1]} tokens de prompt, {len(gen)} gerados, menor gap {min(gaps):.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    meta = {"model": os.path.realpath(args.model), "torch": torch.__version__, "transformers": transformers.__version__, "dtype": "float32 (pesos BF16 convertidos)"}
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "cases": cases}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
