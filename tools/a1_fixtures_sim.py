#!/usr/bin/env python3
"""A1.1: fixtures do simulador para o motor D1 no modo com barreira (plano docs/PLANO-runtime-rust-moe-20261029.md, A1.1).

Roda o `Sim` de tools/moe_sim.py com `lfu_cpu`, protocolo `token` e `token_barreira=True` em traços pequenos e aleatórios
(semente fixa), e grava os roteamentos com as contagens esperadas em rust/moe-rt/tests/fixtures/a1_sim_token_barreira.json.
O teste `motor::tests::token_barrier_bate_com_o_simulador` roda o motor com o backend falso nos mesmos roteamentos, em
intercalações sorteadas, e exige as mesmas contagens.

No modo com barreira as contagens não dependem do tempo (o backlog conta todas as cópias do token e a virada espera todas),
então os parâmetros de custo do `Sim` não importam; usa os do 1B.

Uso: taskset -c 16-27 tools/.venv-ref/bin/python -I -B tools/a1_fixtures_sim.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import moe_sim  # noqa: E402

VERSAO = 1
SEMENTE = 20261031
CASOS = 64
SAIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rust", "moe-rt", "tests", "fixtures",
                     "a1_sim_token_barreira.json")
CAMPOS = ("needed", "hits", "cpu", "copies_bg", "dropped", "copies_deferred")


def tracos(rng):
    L = int(rng.integers(2, 5))
    E = int(rng.choice([3, 4, 6, 8]))
    K = int(rng.integers(1, min(3, E) + 1))
    T = int(rng.integers(6, 25))
    cap = int(rng.integers(0, L * E + 1))
    backlog = int(rng.choice([0, 1, 2, 4, 4, 4]))
    # Popularidade enviesada por camada, para haver reúso e despejo.
    pop = rng.dirichlet(np.full(E, 0.4), size=L)
    a = np.zeros((L, T, K), dtype=np.int64)
    for t in range(T):
        for l in range(L):
            a[l, t] = rng.choice(E, size=K, replace=False, p=pop[l])
    return L, E, K, T, cap, backlog, a


def main():
    rng = np.random.default_rng(SEMENTE)
    p = moe_sim.PARAMS_RUNTIME["1b"]
    casos = []
    for i in range(CASOS):
        L, E, K, T, cap, backlog, a = tracos(rng)
        s = moe_sim.Sim("lfu_cpu", cap, L, E, K, p, protocol="token", backlog=backlog, token_barreira=True)
        s.run([a], measure=False)
        casos.append({
            "caso": i, "layers": L, "experts": E, "topk": K, "tokens": T, "slots": cap, "backlog": backlog,
            # routing[t][l] = ids na ordem do top-k
            "routing": [[[int(e) for e in a[l, t]] for l in range(L)] for t in range(T)],
            "esperado": {c: int(s.stats[c]) for c in CAMPOS},
        })
    out = {"versao": VERSAO, "semente": SEMENTE, "gerador": "tools/a1_fixtures_sim.py",
           "sim": "tools/moe_sim.py Sim(lfu_cpu, protocol=token, token_barreira=True)", "half_life": moe_sim.HALF_LIFE,
           "casos": casos}
    with open(SAIDA, "w") as f:
        json.dump(out, f, separators=(",", ":"))
        f.write("\n")
    tot = {c: sum(k["esperado"][c] for k in casos) for c in CAMPOS}
    print(f"{len(casos)} casos em {os.path.normpath(SAIDA)}; totais {tot}")


if __name__ == "__main__":
    main()
