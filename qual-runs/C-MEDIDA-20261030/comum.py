"""Partes comuns das previsões e do ajuste da medida do C (rodadas 38 e 39 do plano)."""
import hashlib
import re
import sys

sys.path.insert(0, "tools")
import moe_sim as m  # noqa: E402

TR = {"1b": "qual-runs/TRACOS-20261029/granite-3.1-1b-a400m-instruct", "3b": "qual-runs/TRACOS-20261029/granite-3.1-3b-a800m-instruct"}
FRACS = (0.1, 0.25, 0.5)
# GPU por token do m2b (modo normal) que o passo 3 usou em PARAMS_RUNTIME, e o residente do m2a das referências (29/10).
GPU_M2B = {"1b": 2.99, "3b": 4.83}
RESIDENTE_REF = {"1b": 3.05, "3b": 4.86}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def traces(mod):
    train, test = m.load(TR[mod], "treino"), m.load(TR[mod], "teste")
    L, _, k = test[0].shape
    E = int(max(a.max() for a in train + test)) + 1
    return train, test, L, E, k


def sim(mod_sim, policy, cap, L, E, k, p, train, test, warm, S=1.0, backlog=4):
    s = mod_sim.Sim(policy, cap, L, E, k, p, backlog=backlog)
    if S != 1.0:
        s.S1 = S
    if warm:
        s.run(train, False)
    s.run(test, True)
    st = s.stats
    n = st["tokens"]
    # Com S < 1 o CPU ms das camadas de uma cópia não sai destes contadores; só vale com S = 1.
    cpu_ms = (st["cpu_traffic"] * p["c_cpu"] + st["cpu_quiet"] * p["c_cpu_quiet"]) / n
    return {"ms": st["ms"] / n, "wait": st["wait_ms"] / n, "hit": st["hits"] / st["needed"], "cpu_ms": cpu_ms,
            "bg": st["copies_bg"] / n, "traffic": st["layers_traffic"] / n / L, "cpu_traffic": st["cpu_traffic"] / n,
            "cpu_quiet": st["cpu_quiet"] / n}


def tabela(path):
    """{(política, orçamento %): {ms, cpu_ms}} da tabela principal de um m2a-cache, residente incluído."""
    out = {}
    for line in open(path):
        r = re.match(r"^(resident|lru_cpu|lfu_cpu)\s+(\d+)%\s+\d+\s+\|\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+\|\s+([\d.]+)%\s+([\d.]+)%"
                     r"\s+[\d.]+\s+([\d.]+)\s+\|\s+(\S+)\s+([\d.]+)", line)
        if r:
            out[(r[1], int(r[2]))] = {"ms": float(r[3]), "hit": float(r[4]) / 100, "bg": float(r[6]), "cpu_ms": float(r[8])}
    return out
