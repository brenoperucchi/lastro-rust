"""Análise da medida do C pelo criterio.txt: comparação aquecida (portão e teste do simulador) e validação fria (fase B contra
a previsão gravada em previsao-fase-b.txt, com S = 1; controles contra a fase A). Rodar a partir da raiz do projeto."""
import glob
import re
import statistics as stt
import sys

sys.path.insert(0, "qual-runs/C-MEDIDA-20261030")
from comum import tabela  # noqa: E402

R = "qual-runs/C-MEDIDA-20261030"
BUD = (10, 25, 50)


def waits(pattern, pol):
    """{orçamento: [espera por execução]}, espera = ms/token da política − ms/token do residente da mesma execução."""
    out = {b: [] for b in BUD}
    files = sorted(glob.glob(f"{R}/{pattern}"))
    for f in files:
        t = tabela(f)
        for b in BUD:
            out[b].append(t[(pol, b)]["ms"] - t[("resident", 100)]["ms"])
    return out, files


def col(pattern, pol, key):
    return {b: [tabela(f)[(pol, b)][key] for f in sorted(glob.glob(f"{R}/{pattern}"))] for b in BUD}


prev = {}
for line in open(f"{R}/previsao.txt"):
    m = re.match(r"^(1b|3b)\s+(\d+)%\s+\|\s+([\d.]+)\s+([\d.]+)\s+\|\s+([\d.]+)%", line)
    if m:
        prev[(m[1], int(m[2]))] = float(m[5]) / 100

print("== Comparação aquecida (espera = ms/token − residente da mesma execução; médias das 5)")
print("modelo orçam. | espera lru_cpu (mín–máx)    | espera lfu_cpu (mín–máx)    | redução | prevista | portão | simulador")
gate_ok = False
sim_ok = True
for mod in ("1b", "3b"):
    a, fa = waits(f"quente-{mod}-lru_cpu-*.txt", "lru_cpu")
    b, fb = waits(f"quente-{mod}-lfu_cpu-*.txt", "lfu_cpu")
    assert len(fa) == 5 and len(fb) == 5, (fa, fb)
    for bud in BUD:
        ma, mb = stt.mean(a[bud]), stt.mean(b[bud])
        red = 1 - mb / ma
        p = prev[(mod, bud)]
        gate = red >= 0.05 and max(b[bud]) < min(a[bud])
        inside = p / 2 <= red <= 2 * p
        gate_ok |= gate
        sim_ok &= inside
        print(f"{mod:5s} {bud:5d}% | {ma:6.2f} ({min(a[bud]):6.2f}–{max(a[bud]):6.2f}) | {mb:6.2f} ({min(b[bud]):6.2f}–{max(b[bud]):6.2f})"
              f" | {red:6.1%} | {p:7.1%} | {'passa' if gate else 'não':6s} | {'dentro' if inside else 'FORA'}")
print(f"portão do C (≥ 5% com faixas sem sobreposição em ao menos uma célula): {'PASSA' if gate_ok else 'NÃO PASSA'}")
print(f"teste do simulador (redução dentro de [metade, dobro] da prevista em toda célula): {'PASSA' if sim_ok else 'CAI'}")

print("\n== Acerto e cópias por token na comparação aquecida (médias das 5)")
for mod in ("1b", "3b"):
    for pol in ("lru_cpu", "lfu_cpu"):
        h = col(f"quente-{mod}-{pol}-*.txt", pol, "hit")
        c = col(f"quente-{mod}-{pol}-*.txt", pol, "bg")
        print(f"{mod} {pol:8s} " + "  ".join(f"{bud}%: acerto {stt.mean(h[bud]):.1%} cóp.bg {stt.mean(c[bud]):5.2f}" for bud in BUD))

pb = {}
cur = None
for line in open(f"{R}/previsao-fase-b.txt"):
    m = re.match(r"^(1b|3b):", line)
    if m:
        cur = m[1]
    m = re.match(r"^\s+(\d+)%\s+\|\s+([\d.]+)\s+([\d.]+)%\s+([\d.]+)\s+\|\s+([\d.]+)\s+([\d.]+)%", line)
    if m and cur:
        pb[(cur, int(m[1]))] = {"ms": float(m[2]), "hit": float(m[3]) / 100, "cpu_ms": float(m[4]), "ms_s06": float(m[5])}

print("\n== Validação fria do lfu_cpu (médias das 3; previsão com S = 1; critério ±2 pontos de acerto e ±10% de ms/token)")
print("modelo orçam. | ms/token medido previsto erro | acerto medido previsto | CPU ms medido previsto | ms com S = 0,6 (registro)")
val_ok = True
for mod in ("1b", "3b"):
    ms = col(f"faseB-{mod}-lfu-*.txt", "lfu_cpu", "ms")
    hit = col(f"faseB-{mod}-lfu-*.txt", "lfu_cpu", "hit")
    cpu = col(f"faseB-{mod}-lfu-*.txt", "lfu_cpu", "cpu_ms")
    assert all(len(v) == 3 for v in ms.values()), ms
    for bud in BUD:
        p = pb[(mod, bud)]
        e = stt.mean(ms[bud]) / p["ms"] - 1
        dh = stt.mean(hit[bud]) - p["hit"]
        ok = abs(e) <= 0.10 and abs(dh) <= 0.02
        val_ok &= ok
        print(f"{mod:5s} {bud:5d}% | {stt.mean(ms[bud]):14.2f} {p['ms']:8.2f} {e:+6.1%} | {stt.mean(hit[bud]):13.1%} {p['hit']:8.1%} |"
              f" {stt.mean(cpu[bud]):13.2f} {p['cpu_ms']:8.2f} | {p['ms_s06']:6.2f} {'ok' if ok else 'FORA'}")
print(f"validação fria: {'PASSA' if val_ok else 'REPROVA'}")

print("\n== Deriva: controles lru_cpu backlog 4 frios da fase B contra a média da fase A (±5%)")
for mod in ("1b", "3b"):
    a = col(f"faseA-{mod}-b4-*.txt", "lru_cpu", "ms")
    c = col(f"faseB-{mod}-ctl-*.txt", "lru_cpu", "ms")
    for bud in BUD:
        ref = stt.mean(a[bud])
        ds = [x / ref - 1 for x in c[bud]]
        print(f"{mod} {bud}%: fase A {ref:.2f}; controles " + " ".join(f"{d:+.1%}" for d in ds) + ("" if all(abs(d) <= 0.05 for d in ds) else "  FORA de ±5%"))

print("\n== Residente por execução (ms/token)")
for mod in ("1b", "3b"):
    r = [tabela(f)[("resident", 100)]["ms"] for f in sorted(glob.glob(f"{R}/*-{mod}-*.txt"))]
    print(f"{mod}: {len(r)} execuções, {min(r):.2f}–{max(r):.2f}, média {stt.mean(r):.2f}")

print("\n== tok/s nas execuções aquecidas (1000 ÷ média do ms/token das 5; residente das 10 aquecidas do modelo)")
for mod in ("1b", "3b"):
    res = [tabela(f)[("resident", 100)]["ms"] for f in sorted(glob.glob(f"{R}/quente-{mod}-*.txt"))]
    a = col(f"quente-{mod}-lru_cpu-*.txt", "lru_cpu", "ms")
    b = col(f"quente-{mod}-lfu_cpu-*.txt", "lfu_cpu", "ms")
    parts = []
    for bud in BUD:
        ta, tb = 1000 / stt.mean(a[bud]), 1000 / stt.mean(b[bud])
        parts.append(f"{bud}%: lru {ta:.0f} lfu {tb:.0f} tok/s ({tb / ta - 1:+.0%})")
    print(f"{mod}: residente {1000 / stt.mean(res):.0f} tok/s; " + "; ".join(parts))
