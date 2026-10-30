"""Fase A da validação fria: reajusta c_cpu e c_cpu_quiet (mínimo quadrado sem pesos até o ponto fixo, como no passo 3) com
o CPU ms/token do lru_cpu com backlog 0 e 4 da sessão (médias das 3 repetições por orçamento), e a GPU pelo residente da
sessão: GPU por token = GPU do m2b do passo 3 × (residente da sessão ÷ residente das referências); g = isso ÷ L − k·h, com h
fixo. o_layer e banda ficam. Depois prevê o lfu_cpu frio (cache vazio, conversas 4-7, backlog 4) com S = 1 (moe_sim.py) e
S = 0,6 (moe_sim_s.py, a cópia instrumentada da rodada 37). Rodar a partir da raiz do projeto."""
import glob
import statistics as stt
import sys

import numpy as np

sys.path.insert(0, "qual-runs/C-MEDIDA-20261030")
from comum import FRACS, GPU_M2B, RESIDENTE_REF, m, sha, sim, tabela, traces  # noqa: E402
import moe_sim_s  # noqa: E402

R = "qual-runs/C-MEDIDA-20261030"
print("simulador:", sha("tools/moe_sim.py"), " cópia instrumentada:", sha(f"{R}/moe_sim_s.py"))
for mod in ("1b", "3b"):
    p = dict(m.PARAMS[mod])
    train, test, L, E, k = traces(mod)
    res, meas = [], []
    for bl in (0, 4):
        files = sorted(glob.glob(f"{R}/faseA-{mod}-b{bl}-*.txt"))
        assert len(files) == 3, files
        rows = [tabela(f) for f in files]
        res += [r[("resident", 100)]["ms"] for r in rows]
        for frac in FRACS:
            v = stt.mean(r[("lru_cpu", round(frac * 100))]["cpu_ms"] for r in rows)
            meas.append((int(frac * L * E), v, bl))
    resid = stt.mean(res)
    gpu_tok = GPU_M2B[mod] * resid / RESIDENTE_REF[mod]
    p["g"] = gpu_tok / L - k * p["h"]
    p["tg_gpu"] = 1000 / gpu_tok
    print(f"\n{mod}: residente da sessão {resid:.3f} ms/token (referência {RESIDENTE_REF[mod]}); GPU por token {gpu_tok:.3f}; g {p['g']:.5f}")
    for it in range(8):
        X, Y = [], []
        for cap, v, bl in meas:
            s = sim(m, "lru_cpu", cap, L, E, k, p, train, test, False, backlog=bl)
            X.append([s["cpu_traffic"], s["cpu_quiet"]])
            Y.append(v)
        X, Y = np.array(X), np.array(Y)
        c = np.linalg.lstsq(X, Y, rcond=None)[0]
        print(f"  iteração {it}: c_cpu {c[0]:.5f} c_cpu_quiet {c[1]:.5f} resíduo máx {np.abs(X @ c - Y).max():.3f} ms")
        done = abs(c[0] - p["c_cpu"]) < 5e-5 and abs(c[1] - p["c_cpu_quiet"]) < 5e-5
        p["c_cpu"], p["c_cpu_quiet"] = round(c[0], 4), round(c[1], 4)
        if done:
            break
    print(f"  parâmetros reajustados: c_cpu {p['c_cpu']} c_cpu_quiet {p['c_cpu_quiet']} g {p['g']:.5f} h {p['h']} o_layer {p['o_layer']}"
          f" pcie_gbs_cpu {p['pcie_gbs_cpu']}")
    print("  previsão do lfu_cpu frio | S = 1: ms/token acerto CPU ms | S = 0,6: ms/token acerto")
    for frac in FRACS:
        cap = int(frac * L * E)
        a = sim(m, "lfu_cpu", cap, L, E, k, p, train, test, False)
        b = sim(moe_sim_s, "lfu_cpu", cap, L, E, k, p, train, test, False, S=0.6)
        print(f"  {frac:6.0%}                  | {a['ms']:8.2f} {a['hit']:6.1%} {a['cpu_ms']:6.2f} | {b['ms']:8.2f} {b['hit']:6.1%}")
