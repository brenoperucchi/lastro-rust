"""Estimativa (não medida): ms/token do lfu_cpu e do lru_cpu, protocolo de hoje e protocolo token (D1), com a banda de
cópia do simulador escalada para x16, x8 e x4. Rodar da raiz do projeto:
  taskset -c 16-27 tools/.venv-ref/bin/python -I -B qual-runs/PCIE-X8-20261031/sim_largura.py <pcie_gbs_cpu_x8> > .../sim_largura.txt
pcie_gbs_cpu em x16 é o calibrado (14,1 GB/s); em x8 é o argumento (medido no lru_cpu 10% do 1B em x8, como a calibração);
x4 é metade do x8 (a 3080 Ti no slot do chipset: limite teórico, sem medida). PCIE_GBS (cópia sem CPU) escala na mesma razão.
Só o modelo de custo muda; os traços são os de teste, como na A0c."""
import hashlib
import sys

sys.path.insert(0, "tools")
import moe_sim as m  # noqa: E402

x8 = float(sys.argv[1])
LARG = {"x16": m.PCIE_GBS_CPU_RUNTIME, "x8": x8, "x4": x8 / 2}
T = "qual-runs/TRACOS-20261029"
MODS = {"1b": f"{T}/granite-3.1-1b-a400m-instruct", "3b": f"{T}/granite-3.1-3b-a800m-instruct"}
CAPS = (0.10, 0.25, 0.50)
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]  # noqa: E731
print(f"simulador {sha('tools/moe_sim.py')}; pcie_gbs_cpu " + ", ".join(f"{k} {v:.2f}" for k, v in LARG.items()))
base_gbs = m.PCIE_GBS
for mod, d in MODS.items():
    train, test = m.load(d, "treino"), m.load(d, "teste")
    L, _, k = train[0].shape
    E = int(max(a.max() for a in train + test)) + 1
    warm = [train[3]]
    print(f"\n== {mod}: ms/token simulado (cópias bg por token)")
    print(f"{'política':26s} {'orç.':>5s} | " + " | ".join(f"{w:>17s}" for w in LARG))
    for pol, kw, nome in (("lfu_cpu", {}, "lfu_cpu hoje"), ("lfu_cpu", {"protocol": "token"}, "lfu_cpu token (D1)"),
                          ("lru_cpu", {}, "lru_cpu hoje")):
        for c in CAPS:
            cel = []
            for w, gbs in LARG.items():
                p = dict(m.PARAMS_RUNTIME[mod], pcie_gbs_cpu=gbs)
                m.PCIE_GBS = base_gbs * gbs / m.PCIE_GBS_CPU_RUNTIME
                s = m.Sim(pol, int(c * L * E), L, E, k, p, **kw)
                s.run(warm, False)
                s.run(test, True)
                st = s.stats
                cel.append(f"{st['ms'] / st['tokens']:7.3f} ({st['copies_bg'] / st['tokens']:5.2f})")
            m.PCIE_GBS = base_gbs
            print(f"{nome:26s} {c:5.0%} | " + " | ".join(f"{x:>17s}" for x in cel))
