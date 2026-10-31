"""A0c, rodada 52: de onde vem a mudança do ganho com as admissões depois da CPU. Para cada célula, o protocolo de hoje, o
token da rodada 51 (moe_sim-rodada51.py, a cópia do simulador congelado na rodada 51, sha 286e06c713556cc1, com a admissão no instante do roteador e a
cópia correndo junto com a CPU da camada) e o token de agora. Imprime a espera, as camadas com tráfego, os experts na CPU
em camada com tráfego e as cópias, por token. Rodar da raiz:
  taskset -c 16-27 tools/.venv-ref/bin/python -I -B qual-runs/A0C-TOKEN-SIM-20261031/decomposicao.py > .../decomposicao.txt"""
import hashlib
import sys
import types

sys.path.insert(0, "tools")
import moe_sim as m  # noqa: E402

R51 = "qual-runs/A0C-TOKEN-SIM-20261031/moe_sim-rodada51.py"
src, src51 = open("tools/moe_sim.py").read(), open(R51).read()
r51 = types.ModuleType("moe_sim_r51")
r51.__file__ = m.__file__
exec(compile(src51, "moe_sim_r51", "exec"), r51.__dict__)

T = "qual-runs/TRACOS-20261029"
MODS = {"1b": f"{T}/granite-3.1-1b-a400m-instruct", "3b": f"{T}/granite-3.1-3b-a800m-instruct"}
sha = lambda t: hashlib.sha256(t.encode()).hexdigest()[:16]  # noqa: E731
print(f"simulador {sha(src)}, rodada 51 {sha(src51)}")
for mod, d in MODS.items():
    p = m.PARAMS_RUNTIME[mod]
    train, test = m.load(d, "treino"), m.load(d, "teste")
    L, _, k = train[0].shape
    E = int(max(x.max() for x in train + test)) + 1
    for c in (0.10, 0.25, 0.50):
        cap = int(c * L * E)
        print(f"\n{mod} {c:.0%}")
        for nome, mod_, kw in (("hoje", m, {}), ("token da rodada 51", r51, {"protocol": "token"}),
                               ("token de agora", m, {"protocol": "token"})):
            s = mod_.Sim("lfu_cpu", cap, L, E, k, p, **kw)
            s.run([train[3]], False)
            s.run(test, True)
            st, n = s.stats, s.stats["tokens"]
            print(f"  {nome:30s} espera {st['wait_ms'] / n:6.3f}  camadas com tráfego {st['layers_traffic'] / n:5.2f}"
                  f"  CPU em camada com tráfego {st['cpu_traffic'] / n:6.2f}  cópias {st['copies_bg'] / n:5.2f}  (por token)")
