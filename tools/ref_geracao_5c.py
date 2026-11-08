#!/usr/bin/env python3
"""Referência da etapa 5c (geração livre curta), derivada de `tools/ref_greedy.py`. Para cada texto cru de `5c-textos/textos.json`: renderiza o chat
template do Granite com a DATA FIXA "November 07, 2026" (a do 5b), tokeniza o texto renderizado, gera N tokens gulosos no transformers (CPU,
float32 a partir dos pesos BF16, cache KV, eos_token_id=None: nunca para no EOS) e grava os ids do prompt, os ids gerados, o texto e o gap
top1-top2 por passo (para separar divergência real de empate). SÓ CPU: nenhuma GPU é aberta.

  env CUDA_VISIBLE_DEVICES= LC_ALL=C tools/.venv-ref/bin/python -I tools/ref_geracao_5c.py --nome 1b \
      --model /home/brenoperucchi/models/granite-3.1-1b-a400m-instruct \
      --textos qual-runs/A2-LOTE-20261103/5c-textos --new-tokens 64 \
      --out qual-runs/A2-LOTE-20261103/5c-referencia/ref5c-1b.json
  (idem --nome 3b com o modelo 3b e --out .../ref5c-3b.json)

Critério: qual-runs/A2-LOTE-20261103/criterio-5c-geracao-livre-20261108.md.
"""
import argparse
import datetime as _dt
import hashlib
import json
import os

DATA = _dt.datetime(2026, 11, 7, 12, 0, 0)  # "November 07, 2026"


class Relogio(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return DATA


def greedy(model, ids, new_tokens, pad_id):
    """Gera `new_tokens` tokens gulosos a partir de `ids` [1, T]; devolve os ids gerados e, por passo, o gap top1-top2."""
    import torch

    # Máscara explícita (como no ref_greedy.py): sem ela, com pad_token_id = 0, o generate mascara todo token 0 do prompt.
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
    gen = out.sequences[0, ids.shape[1]:].tolist()
    gaps = []
    for s in out.scores:
        top = torch.topk(s[0].float(), 2).values
        gaps.append(float(top[0] - top[1]))
    return gen, gaps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nome", required=True, help="1b ou 3b")
    ap.add_argument("--model", required=True)
    ap.add_argument("--textos", required=True, help="pasta com textos.json e os texto-*.txt")
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import chat_template_utils

    torch.set_num_threads(a.threads)
    chat_template_utils.datetime = Relogio
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True, dtype=torch.float32).eval()
    manifesto = json.load(open(os.path.join(a.textos, "textos.json"), encoding="utf-8"))
    casos = []
    for i, t in enumerate(manifesto["textos"]):
        bruto = open(os.path.join(a.textos, t["arquivo"]), "rb").read()
        sha = hashlib.sha256(bruto).hexdigest()
        if sha != t["sha256"]:
            raise SystemExit(f"{t['arquivo']}: sha256 {sha} difere do manifesto {t['sha256']}")
        texto = bruto.decode("utf-8")
        renderizado = tok.apply_chat_template([{"role": "user", "content": texto}], tokenize=False, add_generation_prompt=True)
        ids = tok(renderizado, return_tensors="pt").input_ids
        gen, gaps = greedy(model, ids, a.new_tokens, tok.pad_token_id)
        casos.append({
            "indice": i,
            "nome": t["nome"],
            "arquivo": t["arquivo"],
            "texto_sha256": sha,
            "texto_renderizado": renderizado,
            "ids": ids[0].tolist(),
            "gerados": gen,
            "texto_gerado": tok.decode(gen),
            "gaps": gaps,
        })
        print(f"{a.nome} {t['nome']}: {ids.shape[1]} ids de prompt, {len(gen)} gerados, menor gap {min(gaps):.5f}")
    meta = {"nome": a.nome, "model": os.path.realpath(a.model), "data_template": "November 07, 2026", "novos": a.new_tokens, "eos_token_id": None,
            "torch": torch.__version__, "transformers": transformers.__version__, "dtype": "float32 (pesos BF16 convertidos)", "dispositivo": "cpu"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "casos": casos}, f, indent=1, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
