"""Lê as 16 execuções e compara, dentro da sessão, backlog 1 e 2 contra os controles 0 e 4 (médias das 2 repetições).
CPU por expert = CPU ms/token ÷ (fração na CPU · k · L); "parcela da 1ª cópia" = (b − b0) ÷ (b4 − b0) do CPU por expert."""
import re
LK = {"1b": 24 * 8, "3b": 32 * 8}
REF = {"1b": "../M2F-FUSOES-20261029/m2a-1b-final-1.txt", "3b": "../M2F-FUSOES-20261029/m2a-3b-final-1.txt"}
PREV = {}
for line in open("previsao.txt"):
    t = line.split()
    if len(t) > 5 and t[0] in LK and t[3] == "|":
        PREV[(t[0], int(t[1].rstrip("%")), int(t[2]))] = float(t[4])

def rows(path):
    out = {}
    for line in open(path):
        t = line.split()
        if t and t[0] in ("lru_cpu", "resident") and len(t) > 15:
            out[(t[0], int(t[1].rstrip("%")))] = dict(ms=float(t[4]), cpu=float(t[9].rstrip("%")) / 100, cpu_ms=float(t[14]), rot=float(t[13]) if t[13] != "NaN" else 0.0, bg=float(t[11]))
    return out

def avg(d):
    return {k: sum(x[k] for x in d) / len(d) for k in d[0]}

for mod in ("1b", "3b"):
    runs = {b: [rows(f"m2a-{mod}-backlog{b}-{n}.txt") for n in (1, 2)] for b in (0, 1, 2, 4)}
    res = avg([r[("resident", 100)] for b in runs for r in runs[b]])
    ref = rows(REF[mod])
    print(f"== {mod}: residente desta sessão {res['ms']:.2f} ms/token (referência {ref[('resident', 100)]['ms']:.2f})")
    print("  orçam. backlog | ms/tok (rep1 rep2)  prev. | roteador | cóp.bg | CPU ms | CPU/expert µs | parcela 1ª cópia | ms ÷ ms b4 med/prev")
    for f in (10, 25, 50):
        a = {b: avg([r[("lru_cpu", f)] for r in runs[b]]) for b in runs}
        per = {b: a[b]["cpu_ms"] / (a[b]["cpu"] * LK[mod]) * 1e3 for b in a}
        for b in (0, 1, 2, 4):
            r1, r2 = (r[("lru_cpu", f)]["ms"] for r in runs[b])
            part = (per[b] - per[0]) / (per[4] - per[0]) if b else 0.0
            print(f"  {f:5d}% {b:7d} | {a[b]['ms']:6.2f} ({r1:5.2f} {r2:5.2f}) {PREV[(mod, f, b)]:6.2f} | {a[b]['rot']:8.2f} | {a[b]['bg']:6.1f} | {a[b]['cpu_ms']:6.2f} | {per[b]:13.1f} |"
                  f" {part:16.0%} | {a[b]['ms'] / a[4]['ms']:.3f}/{PREV[(mod, f, b)] / PREV[(mod, f, 4)]:.3f}")
        rf = ref.get(("lru_cpu", f))
        if rf:
            print(f"         ref. b4 | {rf['ms']:6.2f}                      | {rf['rot']:8.2f} | {rf['bg']:6.1f} | {rf['cpu_ms']:6.2f} | {rf['cpu_ms'] / (rf['cpu'] * LK[mod]) * 1e3:13.1f} |")
