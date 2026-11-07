#!/usr/bin/env python3
"""Gera rust/moe-rt/src/tokenizador/tabelas.rs: faixas de \\p{L} e \\p{N} EXATAMENTE como o pré-tokenizador ByteLevel da biblioteca
`tokenizers` (a que gerou os ids de referência) as enxerga, varrendo todos os valores escalares Unicode (CPU, sem rede).

  env CUDA_VISIBLE_DEVICES= tools/.venv-ref/bin/python -I tools/gera_tabelas_5a.py

Sondas (regex GPT-2 do tokenizers): "a"+c é UM pedaço só se c é letra; "1"+c é um pedaço só se c é número (e não letra);
"\\n"+c é um pedaço só se c é espaço. Confere também que char::is_whitespace do Rust coincide com a sonda de espaço (a
tabela de espaço em branco não é gerada: o Rust usa a do std, e o teste Rust/esta varredura falham se divergirem do oracle).
"""
import hashlib
import os
import sys

import tokenizers
from tokenizers import Tokenizer

TOK = "/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct/tokenizer.json"
SAIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rust", "moe-rt", "src", "tokenizador", "tabelas.rs")

# White_Space do Unicode (lista fixa, igual a char::is_whitespace do Rust desde sempre)
WS = {0x9, 0xA, 0xB, 0xC, 0xD, 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000}


def faixas(cps):
    out = []
    for c in cps:
        if out and out[-1][1] + 1 == c:
            out[-1][1] = c
        else:
            out.append([c, c])
    return out


def main():
    pre = Tokenizer.from_file(TOK).pre_tokenizer
    letras, numeros, ws = [], [], []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        if len(pre.pre_tokenize_str("a" + c)) == 1:
            letras.append(cp)
        elif len(pre.pre_tokenize_str("1" + c)) == 1:
            numeros.append(cp)
        if len(pre.pre_tokenize_str("\n" + c)) == 1:
            ws.append(cp)
    if set(ws) != WS:
        sys.exit(f"espaço em branco do oracle difere do White_Space: {sorted(set(ws) ^ WS)[:10]}")
    h = hashlib.sha256(open(TOK, "rb").read()).hexdigest()
    with open(SAIDA, "w", encoding="utf-8") as f:
        f.write("// GERADO por tools/gera_tabelas_5a.py; não editar à mão.\n")
        f.write(f"// Fonte: pré-tokenizador do tokenizers {tokenizers.__version__} com {TOK} (sha256 {h}), varredura de todos os valores escalares.\n")
        f.write(f"// \\p{{L}}: {len(letras)} pontos; \\p{{N}}: {len(numeros)} pontos.\n\n")
        for nome, cps in (("LETRA", letras), ("NUMERO", numeros)):
            fx = faixas(cps)
            f.write(f"pub(super) static {nome}: &[(u32, u32)] = &[\n")
            for i in range(0, len(fx), 6):
                f.write("    " + " ".join(f"(0x{a:X}, 0x{b:X})," for a, b in fx[i : i + 6]) + "\n")
            f.write("];\n\n")
    print("letras", len(letras), "numeros", len(numeros), "espacos", len(ws))


if __name__ == "__main__":
    main()
