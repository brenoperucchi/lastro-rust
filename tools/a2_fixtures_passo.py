#!/usr/bin/env python3
"""A2-PASSO E4a, U0: fixtures do simulador para o passo de M posições do motor D1 no modo com barreira (contrato-e4a.md v2, oráculo B).

Roda o SimM (qual-runs/A2-MULTIPOS-SIM-20261103/multipos.py) com lfu_cpu, protocolo `token` e `token_barreira=True` em traços pequenos e
aleatórios (semente fixa), em passos de tamanho m (fixo com cauda parcial ou variável, ex. 8,1,7,2), e grava os roteamentos, os tamanhos dos
passos e as contagens esperadas em rust/moe-rt/tests/fixtures/a2_sim_passo_barreira.json. No modo com barreira as contagens não dependem do
tempo (o backlog conta todas as cópias do passo e a virada espera todas), então os parâmetros de custo não importam; usa os do 1B.
Com todos os passos de tamanho 1 reproduz as contagens do a1_sim_token_barreira.json (teste de equivalência em test_a2_fixtures_passo.py).

Uso: taskset -c 16-27 tools/.venv-ref/bin/python -I -B tools/a2_fixtures_passo.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "qual-runs", "A2-MULTIPOS-SIM-20261103"))
import numpy as np  # noqa: E402

import moe_sim  # noqa: E402
import multipos as MP  # noqa: E402

VERSAO = 1
SEMENTE = 20261104
CASOS = 96
SAIDA = os.path.join(HERE, "..", "rust", "moe-rt", "tests", "fixtures", "a2_sim_passo_barreira.json")
CAMPOS = ("needed", "hits", "cpu", "copies_bg", "dropped", "copies_deferred")
M_MAX = 8


class SimPasso(MP.SimM):
    """SimM com barreira e backlog explícitos e a capacidade em slots (não em fração)."""

    def __init__(self, L, E, k, cap, backlog, p):
        moe_sim.Sim.__init__(self, "lfu_cpu", cap, L, E, k, p, protocol="token", backlog=backlog, token_barreira=True)
        self.limite, self.M = "otimista", 1

    def roda_passos(self, a, passos):
        """Uma conversa a[L, T, k] em passos de tamanho m; a soma dos passos tem de ser T."""
        L, T, _ = a.shape
        assert sum(passos) == T and all(1 <= m <= M_MAX for m in passos), (passos, T)
        pos = 0
        for m in passos:
            self.M = m
            blk = a[:, pos:pos + m, :]
            for l in range(L):
                es, ns = MP.primeira_aparicao(blk[l])
                assert len(es) <= min(self.E, m * self.k)
                self.layer_m(l, es, ns)
            self.stats["tokens"] += m
            pos += m


def passos_fixos(T, m):
    """m, m, ..., cauda parcial."""
    return [m] * (T // m) + ([T % m] if T % m else [])


def passos_variaveis(rng, T):
    out, resto = [], T
    while resto:
        m = min(int(rng.choice([1, 1, 2, 3, 4, 7, 8])), resto)
        out.append(m)
        resto -= m
    return out


def tracos(rng):
    L = int(rng.integers(2, 5))
    E = int(rng.choice([3, 4, 6, 8]))
    K = int(rng.integers(1, min(3, E) + 1))
    T = int(rng.integers(8, 41))
    cap = int(rng.integers(0, L * E + 1))
    backlog = int(rng.choice([0, 1, 2, 4, 4, 4]))
    pop = rng.dirichlet(np.full(E, 0.4), size=L)
    a = np.zeros((L, T, K), dtype=np.int64)
    for t in range(T):
        for l in range(L):
            a[l, t] = rng.choice(E, size=K, replace=False, p=pop[l])
    return L, E, K, T, cap, backlog, a


def esperado(L, E, K, cap, backlog, a, passos):
    s = SimPasso(L, E, K, cap, backlog, dict(moe_sim.PARAMS_RUNTIME["1b"]))
    s.roda_passos(a, passos)
    return {c: int(s.stats[c]) for c in CAMPOS}


def gera():
    rng = np.random.default_rng(SEMENTE)
    casos = []
    for i in range(CASOS):
        L, E, K, T, cap, backlog, a = tracos(rng)
        tipo = i % 4
        passos = passos_variaveis(rng, T) if tipo == 3 else passos_fixos(T, (2, 4, 8)[tipo])
        casos.append({
            "caso": i, "layers": L, "experts": E, "topk": K, "posicoes": T, "slots": cap, "backlog": backlog,
            "passos": passos,
            "routing": [[[int(e) for e in a[l, t]] for l in range(L)] for t in range(T)],     # routing[pos][l], top-k na ordem do roteador
            "esperado": esperado(L, E, K, cap, backlog, a, passos),
        })
    return {"versao": VERSAO, "semente": SEMENTE, "gerador": "tools/a2_fixtures_passo.py",
            "sim": "qual-runs/A2-MULTIPOS-SIM-20261103/multipos.py SimM(lfu_cpu, protocol=token, token_barreira=True)",
            "half_life": moe_sim.HALF_LIFE, "casos": casos}


def main():
    out = gera()
    with open(SAIDA, "w") as f:
        json.dump(out, f, separators=(",", ":"))
        f.write("\n")
    tot = {c: sum(k["esperado"][c] for k in out["casos"]) for c in CAMPOS}
    print(f"{len(out['casos'])} casos em {os.path.normpath(SAIDA)}; totais {tot}")


if __name__ == "__main__":
    main()
