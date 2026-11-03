#!/usr/bin/env python3
"""A1-DIAG10, sanidade (criterio.txt v6, seção 7): fidelidade dos intervalos com a injeção ALTERNADA por token (tokens ímpares com injeção, pares sem).
Uso: fidelidade.py PASTA   onde PASTA tem d-<modelo>-<cpu|livro|espera>/diag-motor_evento-10.txt (saída do I com --inject <sítio>:10:alt).
Estatística da fidelidade (fixada por escrito, rodada 36): a MEDIANA das diferenças entre cada token com injeção e o token par anterior, na mesma execução; a faixa
é a do critério (esperado = faltas por token x 10 us, +-30%). A diferença de médias e a média dos pares são reportadas, mas o desvio-padrão do T_cpu do 3B (cerca de
3 ms por token) dá à média um erro padrão do tamanho do sinal, então só a mediana tem poder de decidir. Descritivo: nada daqui entra na árvore."""
import os, statistics as st, sys

COL = {"cpu": 6, "livro": 8, "espera": 9}      # t_cpu, t_livro, t_inj nas linhas do diag-*.txt
NOME = {"cpu": "T_cpu", "livro": "T_livro", "espera": "espera antes do wait (acumulador próprio)"}


def le(path):
    return [ln.split() for ln in open(path) if not ln.startswith("#") and ln.strip()]


def mede(rows, col):
    on = [float(r[col]) for r in rows if r[10] == "1"]
    off = [float(r[col]) for r in rows if r[10] == "0"]
    pares = [float(rows[i][col]) - float(rows[i - 1][col]) for i in range(1, len(rows)) if rows[i][10] == "1" and rows[i - 1][10] == "0"]
    return st.mean(on) - st.mean(off), st.mean(pares), st.median(pares), st.pstdev(off), len(pares)


def main(pasta):
    out = []
    for mod in ("1b", "3b"):
        for sitio in ("cpu", "livro", "espera"):
            f = os.path.join(pasta, f"d-{mod}-{sitio}", "diag-motor_evento-10.txt")
            if not os.path.exists(f):
                continue
            rows = le(f)
            faltas = st.mean(float(r[2]) for r in rows)
            esp = faltas * 0.010
            dm, mp, md, sd, n = mede(rows, COL[sitio])
            lo, hi = 0.7 * esp, 1.3 * esp
            out.append(f"{mod} {NOME[sitio]}: esperado {esp:.3f} ms (faixa {lo:.3f} a {hi:.3f}), faltas por token {faltas:.1f}; mediana dos pares {md:+.3f} "
                       f"({'DENTRO' if lo <= md <= hi else 'FORA'}); diferença de médias {dm:+.3f}, média dos pares {mp:+.3f}, desvio-padrão sem injeção {sd:.2f} ms, n={n}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    print(main(sys.argv[1]), end="")
