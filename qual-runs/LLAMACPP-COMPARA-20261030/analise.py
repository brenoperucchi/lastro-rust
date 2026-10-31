"""Análise da comparação com o llama.cpp pelo criterio.txt. Rodar a partir da raiz do projeto, depois do rodar.sh.
Com MEDIDA_DIR, lê outra pasta (só para testar o próprio script)."""
import glob
import os
import re
import statistics as stt
import subprocess
import sys

R = os.environ.get("MEDIDA_DIR", "qual-runs/LLAMACPP-COMPARA-20261030")
BUD = (10, 25, 50)
MODS = ("1b", "3b")
HF = {"1b": "/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct", "3b": "/home/brenoperucchi/models/granite-3.1-3b-a800m-instruct"}
NAMES = {"gpu32": "llama.cpp GPU F32", "gpu16": "llama.cpp GPU F16", "cpumoe": "llama.cpp --cpu-moe", "cheio": "llama.cpp cache cheio",
         "c10": "llama.cpp cache 10%", "c25": "llama.cpp cache 25%", "c50": "llama.cpp cache 50%",
         "ncm10": "llama.cpp -ncmoe 10%", "ncm25": "llama.cpp -ncmoe 25%", "ncm50": "llama.cpp -ncmoe 50%"}


def forcado(path):
    m = re.search(r"^total: (\d+) tokens, ms/tok ([\d.]+), mediana ([\d.]+)", open(path).read(), re.M)
    return float(m[2])


def hit_rate(path):
    m = re.search(r"ubatch <= 8: hits = (\d+), misses = (\d+), hit rate = ([\d.]+)%", open(path).read())
    return float(m[3]) / 100 if m else None


def tabela(path):
    """{(política, orçamento %): {ms, hit}} da tabela principal de um m2a-cache."""
    out = {}
    for line in open(path):
        r = re.match(r"^(resident|lru_cpu|lfu_cpu|lru_sync|cache_full)\s+(\d+)%\s+\d+\s+\|\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+\|\s+([\d.]+)%", line)
        if r:
            out[(r[1], int(r[2]))] = {"ms": float(r[3]), "hit": float(r[4]) / 100}
    return out


MIN_PORTAO = 0.99  # portão de correção entre configurações (criterio.txt)
SLOTS = {"1b": {"c10": 76, "c25": 192, "c50": 384, "cheio": 768}, "3b": {"c10": 128, "c25": 320, "c50": 640, "cheio": 1280}}
LAYERS = {"1b": 24, "3b": 32}


def confere_log(mod, l, log):
    """Lista de problemas no log do llama.cpp de uma execução: FA, tipo do KV, cache (grupo único, slots) e hit rate."""
    t = open(log).read()
    bad = []
    f32 = l != "gpu16"
    if f32 and "flash_attn            = disabled" not in t:
        bad.append("FA não desligado")
    kv = "f32" if f32 else "f16"
    for c in ("K", "V"):  # os dois tipos, separados (rodada 45)
        if not re.search(rf"\b{c} \({kv}\)", t):
            bad.append(f"tipo do {c}")
    if l in SLOTS[mod]:
        g = re.findall(r"(\d+) layers, bf16: +(\d+) slots", t)
        if g != [(str(LAYERS[mod]), str(SLOTS[mod][l]))]:
            bad.append(f"cache {g}")
        h = hit_rate(log)
        if h is None or (l in ("c25", "c50", "cheio") and h == 0):
            bad.append(f"hit rate {h}")
    elif "MoE cache size" in t:
        bad.append("cache ligado onde não devia")
    return bad


def portao(mod, l):
    """Portão de correção da configuração: argmax contra a passada tudo na GPU F32 (1ª repetição), todas as posições."""
    out = subprocess.run([sys.executable, "-I", "-B", "tools/llama-forcado/confere_argmax.py", f"{R}/tokens/{mod}", f"{R}/argmax-{mod}-{l}.txt",
                          HF[mod], "--ref", f"{R}/argmax-{mod}-gpu32.txt", "--todas", "--min", str(MIN_PORTAO)],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()
    return out[-1].endswith("PASSA") and "NÃO" not in out[-1], out[0].split(": ", 1)[1]


def files(pat, n):
    fs = sorted(glob.glob(f"{R}/{pat}"))
    assert len(fs) == n, (pat, fs)
    return fs


# previsões: linhas "nome orçam. | ms/tok tok/s | ..." do previsao.txt, por modelo
prev = {}
cur = None
for line in open(f"{R}/previsao.txt"):
    m = re.match(r"^(1b|3b): ", line)
    if m:
        cur = m[1]
    m = re.match(r"^(llama\.cpp .+?|nosso \S+)\s+([\d.]+)%\s+\|\s+([\d.]+)\s+[\d.]+\s+\|", line)
    if m and cur:
        prev[(cur, m[1].strip(), round(float(m[2])))] = float(m[3])


def pv(mod, name, bud):
    return prev.get((mod, name, bud))


def cell(xs):
    return {"mean": stt.mean(xs), "med": stt.median(xs), "lo": min(xs), "hi": max(xs), "xs": xs}


def cmp_ranges(a, b):
    """'a mais rápido', 'b mais rápido' ou 'empate dentro do ruído' entre duas células do núcleo (faixas)."""
    if a["hi"] < b["lo"]:
        return "a"
    if b["hi"] < a["lo"]:
        return "b"
    return "="


def cmp_base(v, c):
    """valor único de uma linha de base contra uma célula do núcleo: 'lento', 'rápido' ou '='."""
    if v > 1.1 * c["hi"]:
        return "lento"
    if v < c["lo"] / 1.1:
        return "rápido"
    return "="


def err(x, p):
    return "" if p is None else f"{x / p - 1:+6.1%}"


nuc, base, ours_n, ours_b, hits = {}, {}, {}, {}, {}
for mod in MODS:
    for l in ("gpu32", "c10", "c25", "c50"):
        fs = files(f"nucleo-{mod}-{l}-*.txt", 3)
        nuc[(mod, l)] = cell([forcado(f) for f in fs])
        if l != "gpu32":
            hits[(mod, l)] = [hit_rate(f[:-4] + ".log") for f in fs]
    tabs = [tabela(f) for f in files(f"nucleo-{mod}-nosso-*.txt", 3)]
    ours_n[(mod, "resident", 100)] = cell([t[("resident", 100)]["ms"] for t in tabs])
    for pol in ("lru_cpu", "lfu_cpu"):
        for b in BUD:
            ours_n[(mod, pol, b)] = cell([t[(pol, b)]["ms"] for t in tabs])
            ours_n[(mod, pol, b)]["hit"] = stt.mean(t[(pol, b)]["hit"] for t in tabs)
    for l in ("gpu16", "cpumoe", "ncm10", "ncm25", "ncm50", "cheio"):
        base[(mod, l)] = forcado(files(f"base-{mod}-{l}.txt", 1)[0])
    tb = tabela(files(f"base-{mod}-nosso.txt", 1)[0])
    for b in BUD:
        ours_b[(mod, "lru_sync", b)] = tb[("lru_sync", b)]
    ours_b[(mod, "cache_full", 100)] = tb[("cache_full", 100)]
    ours_b[(mod, "resident", 100)] = tb[("resident", 100)]

print("== Núcleo (3 repetições): ms/token médio (mín–máx), mediana das 3, tok/s; erro contra a previsão")
for mod in MODS:
    print(f"\n{mod}:")
    print("linha                     orçam. |  média (mín–máx)            mediana   tok/s | prevista  erro")
    rows = [("llama.cpp GPU F32", 100, nuc[(mod, "gpu32")], pv(mod, "llama.cpp GPU F32", 100))]
    for b in BUD:
        rows.append((f"llama.cpp cache {b}%", b, nuc[(mod, f"c{b}")], [v for (mm, n, bb), v in prev.items() if mm == mod and n.startswith("llama.cpp cache") and "cheio" not in n and bb == b][0]))
    rows.append(("nosso resident", 100, ours_n[(mod, "resident", 100)], pv(mod, "nosso resident", 100)))
    for pol in ("lru_cpu", "lfu_cpu"):
        for b in BUD:
            rows.append((f"nosso {pol}", b, ours_n[(mod, pol, b)], pv(mod, f"nosso {pol}", b)))
    for name, b, c, p in rows:
        print(f"{name:25s} {b:5d}% | {c['mean']:7.2f} ({c['lo']:7.2f}–{c['hi']:7.2f}) {c['med']:9.2f} {1000 / c['mean']:7.1f} | "
              f"{'' if p is None else f'{p:8.2f}'} {err(c['mean'], p)}")

print("\n== Linhas de base (1 execução): ms/token, tok/s; erro contra a previsão")
for mod in MODS:
    print(f"\n{mod}:")
    for l in ("gpu16", "cpumoe", "ncm10", "ncm25", "ncm50", "cheio"):
        name = NAMES[l]
        pkey = [k for k in prev if k[0] == mod and (k[1] == name or (l.startswith("ncm") and k[1].startswith("llama.cpp -ncmoe") and
                abs(k[2] - {"ncm10": 9, "ncm25": 25, "ncm50": 50}[l]) <= 1) or (l == "cheio" and k[1].startswith("llama.cpp cache cheio")))]
        p = prev[pkey[0]] if pkey else None
        print(f"{name:25s} | {base[(mod, l)]:7.2f} {1000 / base[(mod, l)]:7.1f} | {'' if p is None else f'{p:8.2f}'} {err(base[(mod, l)], p)}")
    for (mm, pol, b), t in ours_b.items():
        if mm == mod:
            p = pv(mod, f"nosso {pol}", b)
            print(f"{'nosso ' + pol + f' {b}%':25s} | {t['ms']:7.2f} {1000 / t['ms']:7.1f} | {'' if p is None else f'{p:8.2f}'} {err(t['ms'], p)}  acerto {t['hit']:.1%}")

print("\n== Conferência das execuções do llama.cpp (log de cada execução: FA, tipo do KV, cache, hit rate)")
n_logs = 0
log_bad = set()
for log in sorted(glob.glob(f"{R}/nucleo-*.log") + glob.glob(f"{R}/base-*.log")):
    m = re.match(r".*/(?:nucleo|base)-(1b|3b)-(\w+?)(?:-\d)?\.log$", log)
    n_logs += 1
    bad = confere_log(m[1], m[2], log)
    if bad:
        log_bad.add((m[1], m[2]))
        print(f"FALHA {os.path.basename(log)}: {', '.join(bad)}")
print(f"{n_logs} execuções conferidas")

print(f"\n== Portão de correção por configuração (argmax contra a passada tudo na GPU F32, todas as posições; ≥ {MIN_PORTAO:.0%})")
ok_cfg = {}
for mod in MODS:
    for l in ("c10", "c25", "c50", "gpu16", "cpumoe", "ncm10", "ncm25", "ncm50", "cheio"):
        ok, msg = portao(mod, l)
        ok = ok and (mod, l) not in log_bad and (mod, "gpu32") not in log_bad  # a passada GPU F32 é a referência
        ok_cfg[(mod, l)] = ok
        print(f"{mod} {NAMES[l]:22s}: {'passa' if ok else 'NÃO PASSA'}; {msg}")
    for f in files(f"nucleo-{mod}-nosso-*.txt", 3) + files(f"base-{mod}-nosso.txt", 1):
        ok = "resultado: logits dentro da tolerância em todas as configurações" in open(f).read()
        ok_cfg[(mod, "nosso")] = ok_cfg.get((mod, "nosso"), True) and ok
    print(f"{mod} nosso m2a-cache          : {'passa' if ok_cfg[(mod, 'nosso')] else 'NÃO PASSA'} (logits contra o transformers e a passada residente, limite 1e-4)")
susp = [k for k, v in ok_cfg.items() if not v]
print("configurações suspensas (sem conclusão de velocidade): " + (", ".join(f"{a} {b}" for a, b in susp) if susp else "nenhuma"))

print("\n== Afirmações (uma célula com configuração suspensa não confirma nem refuta)")
for mod in MODS:
    ok_cfg[(mod, "gpu32")] = (mod, "gpu32") not in log_bad


class Afirmacao:
    """Resultado de três estados: confirmada (todas as células decididas e a favor), não confirmada (alguma célula decidida
    contra) ou não decidida (alguma célula suspensa e nenhuma contra)."""

    def __init__(self):
        self.total = self.susp = self.contra = 0

    def celula(self, ok_cfg_cel, a_favor):
        self.total += 1
        if not ok_cfg_cel:
            self.susp += 1
        elif not a_favor:
            self.contra += 1

    def __str__(self):
        if self.contra:
            return f"não confirmada ({self.contra} de {self.total} células contra)"
        if self.susp:
            return f"não decidida ({self.susp} de {self.total} células suspensas)"
        return "confirmada"


A = Afirmacao()
for mod in MODS:
    a, b = ours_n[(mod, "resident", 100)], nuc[(mod, "gpu32")]
    ok = ok_cfg[(mod, "nosso")] and ok_cfg[(mod, "gpu32")]
    r = cmp_ranges(b, a)
    A.celula(ok, r == "a")
    print(f"(a) {mod}: " + ("suspensa (portão de correção)" if not ok else
          f"nosso residente {a['mean']:.2f} contra llama.cpp GPU F32 {b['mean']:.2f} ms ({a['mean'] / b['mean']:.2f}×): "
          f"{ {'a': 'llama.cpp mais rápido', 'b': 'nosso mais rápido', '=': 'empate dentro do ruído'}[r] }"))
print(f"(a) {A}")
B_ = Afirmacao()
for mod in MODS:
    for b in BUD:
        o, l = ours_n[(mod, "lfu_cpu", b)], nuc[(mod, f"c{b}")]
        ok = ok_cfg[(mod, f"c{b}")] and ok_cfg[(mod, "nosso")]
        r = cmp_ranges(o, l)
        B_.celula(ok, r == "a")
        print(f"(b) {mod} {b}%: " + ("suspensa (portão de correção)" if not ok else
              f"lfu_cpu {o['mean']:.2f} contra cache do llama.cpp {l['mean']:.2f} ms (llama.cpp ÷ nosso {l['mean'] / o['mean']:.2f}×): "
              f"{ {'a': 'nosso mais rápido', 'b': 'llama.cpp mais rápido', '=': 'empate dentro do ruído'}[r] }"))
print(f"(b) {B_}")
C = Afirmacao()
for mod in MODS:
    for b in BUD:
        ok = ok_cfg[(mod, f"c{b}")] and ok_cfg[(mod, "cpumoe")]
        r = cmp_base(base[(mod, "cpumoe")], nuc[(mod, f"c{b}")])  # 'rápido' = o --cpu-moe é mais rápido que o cache
        if b in (10, 25):  # a 50% a célula é só relatada
            C.celula(ok, r == "rápido")
        print(f"(c) {mod} {b}%: " + ("suspensa (portão de correção)" if not ok else
              f"cache {nuc[(mod, f'c{b}')]['mean']:.2f} contra --cpu-moe {base[(mod, 'cpumoe')]:.2f} ms: "
              f"{ {'rápido': 'cache mais lento', 'lento': 'cache mais rápido', '=': 'sem diferença decidida'}[r] }"))
print(f"(c) a 10% e 25%: {C}")
D = Afirmacao()
for mod in MODS:
    for b in BUD:
        s_, l = ours_b[(mod, "lru_sync", b)], nuc[(mod, f"c{b}")]
        ok = ok_cfg[(mod, f"c{b}")] and ok_cfg[(mod, "nosso")]
        r = cmp_base(s_["ms"], l)
        if b == 10:
            D.celula(ok, r == "rápido")
        hr = stt.mean(h for h in hits[(mod, f"c{b}")] if h is not None) if any(h is not None for h in hits[(mod, f"c{b}")]) else None
        print(f"(d) {mod} {b}%: " + ("suspensa (portão de correção)" if not ok else
              f"lru_sync {s_['ms']:.2f} ms (acerto {s_['hit']:.1%}) contra cache do llama.cpp {l['mean']:.2f} ms "
              f"(hit rate do log {'' if hr is None else f'{hr:.1%}'}): "
              f"{ {'rápido': 'nosso mais rápido', 'lento': 'nosso mais lento', '=': 'sem diferença decidida'}[r] }"))
print(f"(d) a 10%: {D}")

print("\n== Hit rate do cache do llama.cpp: log (soma aquecimento e medida) contra a réplica do SLRU")
rep = {}
cur = None
for line in open(f"{R}/previsao.txt"):
    m = re.match(r"^(1b|3b): ", line)
    if m:
        cur = m[1]
    m = re.match(r"^hit rate previsto no log do llama.cpp .*?: (.*)$", line)
    if m and cur:
        for b, h in re.findall(r"(\d+)%: ([\d.]+)%", m[1]):
            rep[(cur, int(b))] = float(h) / 100
for mod in MODS:
    for b in BUD:
        hs = hits[(mod, f"c{b}")]
        print(f"{mod} {b}%: log " + " ".join("—" if h is None else f"{h:.2%}" for h in hs) + f"; réplica {rep.get((mod, b), float('nan')):.2%}")

print("\n== Deriva por repetição (±5% da média das 3)")
for mod in MODS:
    for name, c in (("llama.cpp GPU F32", nuc[(mod, "gpu32")]), ("nosso resident", ours_n[(mod, "resident", 100)])):
        ds = [x / c["mean"] - 1 for x in c["xs"]]
        print(f"{mod} {name}: " + " ".join(f"{d:+.1%}" for d in ds) + ("" if all(abs(d) <= 0.05 for d in ds) else "  FORA de ±5%"))

print("\n== Argmax das linhas com cache contra a passada tudo na GPU F32 (1ª repetição)")
for mod in MODS:
    for l in ("c10", "c25", "c50"):
        out = subprocess.run([sys.executable, "-I", "-B", "tools/llama-forcado/confere_argmax.py", f"{R}/tokens/{mod}", f"{R}/argmax-{mod}-{l}.txt",
                              HF[mod], "--ref", f"{R}/argmax-{mod}-gpu32.txt"], capture_output=True, text=True, check=True).stdout
        print(f"{mod} {l}: " + out.strip().splitlines()[-1].split(": ", 1)[1])

print("\n== Linha do servidor (uso normal; ms/token só da resposta)")
for mod in MODS:
    for l in ("gpu32", "c10", "c25", "c50"):
        t = open(files(f"servidor-{mod}-{l}.txt", 1)[0]).read()
        tot = re.search(r"^total: .*$", t, re.M)
        div = len(re.findall(r"diverge do traço", t))
        print(f"{mod} {NAMES[l]:22s}: {tot[0] if tot else 'sem total'}; conversas com texto diferente do traço: {div}/4")
