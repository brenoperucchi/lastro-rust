#!/usr/bin/env python3
"""Textos da etapa 5c (geração livre curta): 3 pedidos crus em inglês, de conteúdo neutro e FIXO (literais abaixo, nada aleatório), de cerca de
20, 80 e 250 palavras. Grava em qual-runs/A2-LOTE-20261103/5c-textos/: texto-curto.txt, texto-medio.txt, texto-longo.txt (sem quebra de linha
final) e textos.json (palavras, sha256 e nº de ids do pedido JÁ RENDERIZADO pelo template do Granite, data fixa "November 07, 2026", por modelo).
SÓ CPU:

  env CUDA_VISIBLE_DEVICES= LC_ALL=C tools/.venv-ref/bin/python -I tools/gera_textos_5c.py [--sem-tokenizer]

Critério: qual-runs/A2-LOTE-20261103/criterio-5c-geracao-livre-20261108.md.
"""
import argparse
import datetime as _dt
import hashlib
import json
import os

RAIZ = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SAIDA = os.path.join(RAIZ, "qual-runs", "A2-LOTE-20261103", "5c-textos")
MODELOS = {"1b": "/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct", "3b": "/home/brenoperucchi/models/granite-3.1-3b-a800m-instruct"}
DATA = _dt.datetime(2026, 11, 7, 12, 0, 0)

TEXTOS = {
    "curto": (
        "Please explain in simple terms how a bicycle gear system helps a rider climb hills and still go fast on flat roads.",
        (18, 26),
    ),
    "medio": (
        "I am planning a small vegetable garden behind my house. The space gets about six hours of sun each day, the soil is a bit heavy "
        "with clay, and I can only water it a few times per week. I would like to grow tomatoes, carrots, lettuce and some herbs. "
        "Could you describe how I should prepare the soil, how far apart the plants should be, and which of these vegetables are the "
        "easiest for a beginner to look after during the first season?",
        (70, 90),
    ),
    "longo": (
        "Our town library is thinking about changing how it organizes its opening hours, its collections and its public programs, and "
        "the staff would like a clear overview of the main ideas before the next meeting. At the moment the library opens at nine in "
        "the morning and closes at six in the evening on weekdays, with shorter hours on Saturday and no service on Sunday. Many "
        "families say they can only visit after work or on weekends, while students often ask for quiet rooms in the evening. The "
        "collection has grown over many years and includes printed books, magazines, music discs, films and a growing catalog of "
        "digital titles that members can borrow from home. Some shelves are crowded and rarely used, while the children's section is "
        "often full on Saturday mornings. The library also runs reading circles, language practice groups, workshops for older adults "
        "and a weekly story hour for young children, but attendance changes a lot from one month to the next. The budget is fixed, so "
        "adding new services means giving up something else. Please write a short, balanced summary of the options the staff could "
        "consider, explain the advantages and the risks of each one in plain language, and suggest a simple way to collect opinions "
        "from visitors, volunteers and local schools before any decision is made. Keep the tone friendly and practical, and avoid "
        "technical vocabulary so that every member of the team can follow the discussion without extra background.",
        (225, 275),
    ),
}


class Relogio(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return DATA


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sem-tokenizer", action="store_true", help="só grava os textos e os shas (sem contar ids)")
    a = ap.parse_args()
    toks = {}
    if not a.sem_tokenizer:
        from transformers import AutoTokenizer
        from transformers.utils import chat_template_utils

        chat_template_utils.datetime = Relogio
        toks = {n: AutoTokenizer.from_pretrained(d, local_files_only=True) for n, d in MODELOS.items()}
    os.makedirs(SAIDA, exist_ok=True)
    out = {"data_template": "November 07, 2026", "textos": []}
    for nome, (t, (lo, hi)) in TEXTOS.items():
        n = len(t.split())
        assert lo <= n <= hi, f"{nome}: {n} palavras fora de {lo}..{hi}"
        assert "\n" not in t and t == t.strip()
        arq = f"texto-{nome}.txt"
        with open(os.path.join(SAIDA, arq), "w", encoding="utf-8", newline="") as f:
            f.write(t)
        ids = {}
        for m, tok in toks.items():
            r = tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True)
            ids[m] = len(tok(r).input_ids)
        out["textos"].append({"nome": nome, "arquivo": arq, "palavras": n, "sha256": sha(t.encode("utf-8")), "ids_com_template": ids})
        print(nome, n, "palavras", ids, sha(t.encode("utf-8")))
    with open(os.path.join(SAIDA, "textos.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
