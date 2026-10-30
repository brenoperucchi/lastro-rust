"""Previsão da comparação aquecida (lru_cpu × lfu_cpu, aquecidos nas conversas 0-3, medidos nas 4-7, backlog 4), com os
parâmetros atuais do moe_sim.py. Rodar a partir da raiz do projeto."""
import sys

sys.path.insert(0, "qual-runs/C-MEDIDA-20261030")
from comum import FRACS, m, sha, sim, traces  # noqa: E402

print("simulador:", sha("tools/moe_sim.py"))
print("modelo orçam. | espera lru_cpu  espera lfu_cpu | redução prevista | refutação fora de [metade, dobro] | ms/token lru lfu")
for mod in ("1b", "3b"):
    p = m.PARAMS[mod]
    train, test, L, E, k = traces(mod)
    for frac in FRACS:
        cap = int(frac * L * E)
        a = sim(m, "lru_cpu", cap, L, E, k, p, train, test, True)
        b = sim(m, "lfu_cpu", cap, L, E, k, p, train, test, True)
        red = 1 - b["wait"] / a["wait"]
        print(f"{mod:5s} {frac:6.0%} | {a['wait']:14.2f} {b['wait']:15.2f} | {red:16.1%} | [{red / 2:6.1%}, {min(2 * red, 1):6.1%}]"
              f"                  | {a['ms']:6.2f} {b['ms']:6.2f}")
