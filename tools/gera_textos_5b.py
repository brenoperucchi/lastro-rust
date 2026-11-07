#!/usr/bin/env python3
"""Textos de velocidade do degrau 5b: pedidos de usuário de cenário C (alvo 500 ids) e A (alvo 2.003 ids), gerados de forma DETERMINÍSTICA
a partir de `tools/ref-texts/texto-a.txt` (as palavras do texto, repetidas em ciclo e cortadas em N palavras; N escolhido pela menor
distância ao alvo, o menor N em empate). A contagem de ids é a do pedido JÁ RENDERIZADO pelo chat template (sistema padrão, data fixa
"November 07, 2026", add_generation_prompt) no transformers, por modelo (os dois tokenizer.json são iguais). SÓ CPU.

  env CUDA_VISIBLE_DEVICES= LC_ALL=C tools/.venv-ref/bin/python -I tools/gera_textos_5b.py

Grava em qual-runs/A2-LOTE-20261103/5b-textos/: pedido-C.txt, pedido-A.txt (sem quebra de linha final) e textos.json (palavras, ids por
modelo, sha256). Critério: qual-runs/A2-LOTE-20261103/criterio-5b-primeira-palavra.txt.
"""
import datetime as _dt
import hashlib
import json
import os

from transformers import AutoTokenizer
from transformers.utils import chat_template_utils

RAIZ = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SAIDA = os.path.join(RAIZ, "qual-runs", "A2-LOTE-20261103", "5b-textos")
MODELOS = {"1b": "/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct", "3b": "/home/brenoperucchi/models/granite-3.1-3b-a800m-instruct"}
ALVOS = {"C": 500, "A": 2003}
DATA = _dt.datetime(2026, 11, 7, 12, 0, 0)


class Relogio(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return DATA


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    chat_template_utils.datetime = Relogio
    base = open(os.path.join(RAIZ, "tools", "ref-texts", "texto-a.txt"), encoding="utf-8").read().split()
    toks = {n: AutoTokenizer.from_pretrained(d, local_files_only=True) for n, d in MODELOS.items()}

    def texto(n):
        return " ".join(base[i % len(base)] for i in range(n))

    def contagem(tok, n):
        r = tok.apply_chat_template([{"role": "user", "content": texto(n)}], tokenize=False, add_generation_prompt=True)
        return len(tok(r).input_ids)

    ref = toks["1b"]
    out = {"fonte": "tools/ref-texts/texto-a.txt", "palavras_na_fonte": len(base), "data_template": "November 07, 2026", "cenarios": {}}
    os.makedirs(SAIDA, exist_ok=True)
    for cen, alvo in ALVOS.items():
        # a contagem cresce com N (quase monotônica): procura por varredura por saltos e depois refina por vizinhança
        lo, hi = 1, 1
        while contagem(ref, hi) < alvo:
            hi *= 2
        while lo < hi:
            m = (lo + hi) // 2
            if contagem(ref, m) < alvo:
                lo = m + 1
            else:
                hi = m
        candidatos = range(max(1, lo - 6), lo + 7)
        n = min(candidatos, key=lambda k: (abs(contagem(ref, k) - alvo), k))
        t = texto(n)
        arq = os.path.join(SAIDA, f"pedido-{cen}.txt")
        with open(arq, "w", encoding="utf-8", newline="") as f:
            f.write(t)
        ids = {nome: contagem(tok, n) for nome, tok in toks.items()}
        out["cenarios"][cen] = {"alvo": alvo, "palavras": n, "ids": ids, "sha256": sha(t.encode("utf-8")), "arquivo": f"pedido-{cen}.txt"}
        print(cen, n, "palavras", ids, "alvo", alvo)
    with open(os.path.join(SAIDA, "textos.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
