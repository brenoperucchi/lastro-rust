"""Previsões do simulador para o m2a do lru_cpu com backlog 0, 1, 2 e 4 (cache vazio, conversas 4-7, sem aquecimento)."""
import sys, hashlib
sys.path.insert(0, "tools")
import moe_sim as m
TR = {"1b": "qual-runs/TRACOS-20261029/granite-3.1-1b-a400m-instruct", "3b": "qual-runs/TRACOS-20261029/granite-3.1-3b-a800m-instruct"}
print("simulador:", hashlib.sha256(open("tools/moe_sim.py", "rb").read()).hexdigest())
print("modelo  orçam. backlog | ms/token acerto CPU ms/token cópias bg/token camadas com tráfego")
for mod in TR:
    test = m.load(TR[mod], "teste"); L, _, k = test[0].shape
    E = int(max(a.max() for a in m.load(TR[mod], "treino") + test)) + 1
    for frac in (0.1, 0.25, 0.5):
        cap = int(frac * L * E)
        for bl in (0, 1, 2, 4):
            s = m.Sim("lru_cpu", cap, L, E, k, m.PARAMS[mod], backlog=bl); s.run(test, True); st = s.stats; n = st["tokens"]
            cpu_ms = (st["cpu_traffic"] * m.PARAMS[mod]["c_cpu"] + st["cpu_quiet"] * m.PARAMS[mod]["c_cpu_quiet"]) / n
            print(f"{mod:5s} {frac:6.0%} {bl:7d} | {st['ms']/n:8.2f} {st['hits']/st['needed']:6.1%} {cpu_ms:10.2f} {st['copies_bg']/n:14.1f} {st['layers_traffic']/n/L:18.1%}")
