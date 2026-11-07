#!/usr/bin/env python3
"""Referência do degrau 5b (primeira palavra a partir de texto cru): para os 3 primeiros pedidos de `prompts-traco.json`, renderiza o chat
template do Granite com a DATA FIXA (a mesma do critério), roda o modelo no transformers (CPU, float32 a partir dos pesos BF16) e grava
os ids do prompt, o argmax do último logit, o segundo colocado e a diferença entre os dois (para separar divergência real de empate).
SÓ CPU: nenhuma GPU é aberta.

  env CUDA_VISIBLE_DEVICES= LC_ALL=C tools/.venv-ref/bin/python -I tools/ref_primeira_palavra.py --nome 1b \
      --model /home/brenoperucchi/models/granite-3.1-1b-a400m-instruct --prompts tools/ref-texts/prompts-traco.json \
      --out qual-runs/A2-LOTE-20261103/5b-referencia/ref-1b.json

Critério: qual-runs/A2-LOTE-20261103/criterio-5b-primeira-palavra.txt.
"""
import argparse
import datetime as _dt
import json
import os

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import chat_template_utils

DATA = _dt.datetime(2026, 11, 7, 12, 0, 0)  # "November 07, 2026"


class Relogio(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nome", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    chat_template_utils.datetime = Relogio
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True, dtype=torch.float32).eval()
    prompts = json.load(open(a.prompts, encoding="utf-8"))[: a.count]
    casos = []
    for i, p in enumerate(prompts):
        texto = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        ids = tok(texto, return_tensors="pt").input_ids
        with torch.inference_mode():
            lg = model(ids).logits[0, -1].float()
        top = torch.topk(lg, 2)
        t1, t2 = int(top.indices[0]), int(top.indices[1])
        casos.append({
            "indice": i,
            "prompt": p,
            "texto_renderizado": texto,
            "ids": ids[0].tolist(),
            "argmax": t1,
            "argmax_texto": tok.decode([t1]),
            "logit_argmax": float(top.values[0]),
            "top2": t2,
            "logit_top2": float(top.values[1]),
            "gap": float(top.values[0] - top.values[1]),
        })
        print(f"{a.nome} pedido {i}: {ids.shape[1]} ids, argmax {t1} {tok.decode([t1])!r}, gap {casos[-1]['gap']:.4f}")
    meta = {"nome": a.nome, "model": os.path.realpath(a.model), "data_template": "November 07, 2026", "torch": torch.__version__,
            "transformers": transformers.__version__, "dtype": "float32 (pesos BF16 convertidos)", "dispositivo": "cpu"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "casos": casos}, f, indent=1, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
