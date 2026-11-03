#!/usr/bin/env python3
"""A1.4d: análise da janela de 01-02/11. Lê só os arquivos gravados pelo rodar.sh; escreve analise.txt.
Segue o criterio.txt (seções 2 a 7). Rodar da raiz do projeto (usa o simulador e o conferidor de argmax):
  taskset -c 16-27 tools/.venv-ref/bin/python -I -B qual-runs/A1-MEDIDA-20261101/analise.py
Rodada 26: o portão de correção é calculado por bloco, tem controles negativos em memória e uma falha suspende as conclusões
da célula; a L5 (argmax e logs), a L6 contra o simulador e o cruzamento com a carga da GPU entram aqui."""
import os, re, statistics as st, subprocess, sys
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")), "tools"))
import moe_sim  # noqa: E402

R = os.path.dirname(os.path.abspath(__file__))
MODS = ["1b", "3b"]
BUD = [10, 25, 50]
LAYERS = {"1b": 24, "3b": 32}
PRED = {  # (rodada 51, de agora), ms por token, criterio 3.1
    ("1b", 10): (-0.040, -0.040), ("1b", 25): (-0.031, -0.037), ("1b", 50): (-0.021, -0.038),
    ("3b", 10): (-0.067, -0.070), ("3b", 25): (-0.041, -0.051), ("3b", 50): (-0.027, -0.045)}
out = []
def p(*a):
    out.append(" ".join(str(x) for x in a))

def ms(rep, mod, bloco, pref, pol, b):
    f = f"{R}/runs/rep{rep}/{mod}/{bloco}/ms/{pref}-{pol}-{b}.txt"
    return [float(x) for x in open(f).read().split()]

def mean(v): return sum(v) / len(v)
def pct(v, q):
    s = sorted(v); k = (len(s) - 1) * q; i = int(k); j = min(i + 1, len(s) - 1)
    return s[i] + (s[j] - s[i]) * (k - i)

M2, A1 = "m2a-ms", "a1-motor-ms"
data = {}  # (mod, rep) -> dict de séries
for mod in MODS:
    for rep in range(1, 6):
        d = {}
        for b in BUD:
            d[("L2", b)] = ms(rep, mod, "m2a-b4", M2, "lfu_cpu", b)
            d[("L2q", b)] = ms(rep, mod, "m2a-b0", M2, "lfu_cpu", b)
            d[("L1", b)] = ms(rep, mod, "a1", A1, "motor_evento", b)
        d[("L3", 100)] = ms(rep, mod, "m2a-b4", M2, "resident", 100)
        d[("L3q", 100)] = ms(rep, mod, "m2a-b0", M2, "resident", 100)
        d[("L4", 100)] = ms(rep, mod, "a1", A1, "motor_cheio", 100)
        d[("L1res", 100)] = ms(rep, mod, "a1", A1, "resident", 100)
        data[(mod, rep)] = d

# ---------- portão de correção ----------
HF = {"1b": "/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct", "3b": "/home/brenoperucchi/models/granite-3.1-3b-a800m-instruct"}
TOT = {"1b": 82824, "3b": 122048}  # pares (token, camada) das conversas 4 a 7
TOKS = {"1b": 3451, "3b": 3814}
NEEDED = {"1b": 662592, "3b": 976384}
LOGIT_MAX = 1e-4


def valida(t, mod, motor, politicas):
    """Problemas de um bloco (texto da saída do harness): linhas esperadas presentes, logits <= 1e-4 (texto-a e conversas),
    roteamento com no máximo 0,01% dos pares fora do traço, 'true' no needed das linhas do motor e a frase de resultado."""
    bad = []
    rows = {}
    for ln in t.splitlines():
        m = re.match(r"^(resident|lfu_cpu|motor_cheio|motor_evento|motor_barreira)\s+(\d+)%", ln)
        if m and ln.count("|") >= 3 and (m[1], int(m[2])) not in rows:
            rows[(m[1], int(m[2]))] = ln
    for pol, b in politicas:
        ln = rows.get((pol, b))
        if ln is None:
            bad.append(f"falta a linha {pol} {b}%")
            continue
        cols = ln.split("|")
        lg = re.findall(r"([\d.]+e[-+]?\d+)", cols[-1])
        if pol != "resident" and len(lg) != 2 or pol == "resident" and len(lg) != 1:
            bad.append(f"{pol} {b}%: colunas de logits ilegíveis")
        elif any(float(x) > LOGIT_MAX for x in lg):
            bad.append(f"{pol} {b}%: logits acima de 1e-4 {lg}")
        precisa = (not motor) or pol != "resident"
        r = re.search(r"(\d+)/(\d+)", " ".join(cols[3:-1]))
        if precisa:
            if r is None:
                bad.append(f"{pol} {b}%: sem a coluna de roteamento")
            elif int(r[2]) != TOT[mod] or (int(r[2]) - int(r[1])) * 10000 > int(r[2]):
                bad.append(f"{pol} {b}%: roteamento {r[1]}/{r[2]} fora de 0,01%")
        if motor and pol != "resident" and "true" not in cols[-2]:
            bad.append(f"{pol} {b}%: needed não confere")
    if "resultado: logits dentro da tolerância" not in t:
        bad.append("frase de resultado ausente")
    return bad


POL_M2A = [("resident", 100), ("lfu_cpu", 10), ("lfu_cpu", 25), ("lfu_cpu", 50)]
POL_A1 = [("resident", 100), ("motor_cheio", 100), ("motor_evento", 10), ("motor_evento", 25), ("motor_evento", 50)]
POL_BAR = [("resident", 100), ("motor_barreira", 10), ("motor_barreira", 25), ("motor_barreira", 50)]


def controles_negativos():
    """Cada mutação da saída de um bloco real tem de reprovar (rodada 26)."""
    t = open(f"{R}/runs/rep1/1b/a1/saida.txt").read()
    ok = valida(t, "1b", True, POL_A1) == []
    muts = {
        "resumo FALHOU": t.replace("resultado: logits dentro da tolerância", "resultado: FALHOU"),
        "linha ausente": "\n".join(ln for ln in t.splitlines() if not re.match(r"^motor_evento\s+25%", ln)),
        "roteamento fora": t.replace("82824/82824", "82700/82824", 1),
        "logits 2e-4": re.sub(r"5\.5e-6", "2.0e-4", t, count=1),
        "needed falso": t.replace("true rot", "false rot", 1),
    }
    res = {k: valida(v, "1b", True, POL_A1) != [] for k, v in muts.items()}
    return ok, res


p("=== 2. Portão de correção ===")
ok0, neg = controles_negativos()
p("  controles negativos em memória (a saída do rep1/1b/a1 passa; cada mutação tem de reprovar):", "dados atuais passam" if ok0 else "DADOS ATUAIS NÃO PASSAM",
  "; ".join(f"{k}: {'reprova' if v else 'NÃO REPROVA'}" for k, v in neg.items()))
assert ok0 and all(neg.values()), "controle negativo falhou"
bad = 0
valido = {}  # (mod, rep, bloco) -> bool
for mod in MODS:
    for rep in range(1, 6):
        for bl, motor, pols in (("m2a-b4", False, POL_M2A), ("a1", True, POL_A1), ("m2a-b0", False, POL_M2A)):
            t = open(f"{R}/runs/rep{rep}/{mod}/{bl}/saida.txt").read()
            pr = valida(t, mod, motor, pols)
            valido[(mod, rep, bl)] = not pr
            if pr:
                bad += 1
                p(f"  FALHA rep{rep} {mod} {bl}: {pr}")
p("  blocos com falha de logits, roteamento ou needed:", bad, "de 30")
for mod in MODS:
    pr = valida(open(f"{R}/runs/barreira/{mod}/a1/saida.txt").read(), mod, True, POL_BAR)
    valido[(mod, "L6", "a1")] = not pr
    p(f"  L6 {mod}: {'ok' if not pr else pr}")


def celula_valida(mod):
    return all(v for (m, _, _), v in valido.items() if m == mod)


# L5: argmax contra a passada tudo na GPU (1ª repetição), 12 linhas, e as conferências de log de 30/10
SLOTS = {"1b": {"c10": 76, "c25": 192, "c50": 384}, "3b": {"c10": 128, "c25": 320, "c50": 640}}
LAYERS_N = {"1b": 24, "3b": 32}


def hit_rate(path):
    m = re.search(r"ubatch <= 8: hits = (\d+), misses = (\d+), hit rate = ([\d.]+)%", open(path).read())
    return float(m[3]) / 100 if m else None


def confere_log(mod, l, log):
    t = open(log).read()
    bad = []
    if "flash_attn            = disabled" not in t:
        bad.append("FA não desligado")
    for c in ("K", "V"):
        if not re.search(rf"\b{c} \(f32\)", t):
            bad.append(f"tipo do {c}")
    if l in SLOTS[mod]:
        g = re.findall(r"(\d+) layers, bf16: +(\d+) slots", t)
        if g != [(str(LAYERS_N[mod]), str(SLOTS[mod][l]))]:
            bad.append(f"cache {g}")
        h = hit_rate(log)
        if h is None or (l in ("c25", "c50") and h == 0):
            bad.append(f"hit rate {h}")
    elif "MoE cache size" in t:
        bad.append("cache ligado onde não devia")
    return bad


LINHAS5 = ["gpu32", "c10", "c25", "c50", "ncm10", "ncm25", "ncm50"]
p("  L5 argmax (confere_argmax.py contra gpu32, todas as posições, mínimo 99%) e logs do llama.cpp (3 repetições):")
l5_ok = {m: True for m in MODS}
for mod in MODS:
    for l in LINHAS5[1:]:
        r = subprocess.run([sys.executable, "-I", "-B", "tools/llama-forcado/confere_argmax.py", f"qual-runs/LLAMACPP-COMPARA-20261030/tokens/{mod}",
                            f"{R}/llama/rep1/{mod}/argmax-{l}.txt", HF[mod], "--ref", f"{R}/llama/rep1/{mod}/argmax-gpu32.txt", "--todas", "--min", "0.99"],
                           capture_output=True, text=True, env=dict(os.environ, HF_HUB_OFFLINE="1"))
        last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]
        pc = re.search(r"\((\d+\.\d+)%\)", r.stdout)
        okl = r.returncode == 0 and last.endswith("PASSA") and "NÃO" not in last
        l5_ok[mod] &= okl
        p(f"    {mod} {l}: {'PASSA' if okl else 'NÃO PASSA'} {pc[1] + '%' if pc else ''}")
    probs = []
    for rep in (1, 2, 3):
        for l in LINHAS5:
            probs += [f"rep{rep} {l}: {x}" for x in confere_log(mod, l, f"{R}/llama/rep{rep}/{mod}/{l}.txt.log")]
    l5_ok[mod] &= not probs
    p(f"    {mod} logs (FA desligado, K e V em f32, cache com um grupo e os slots certos, hit rate): {len(probs)} problemas {probs[:3] if probs else ''}")

# cruzamento com a carga da GPU (emenda 13c, rodada 27): cada tentativa do carga.log tem de ter observação válida do monitor, nenhum erro de
# consulta e nenhum processo que não seja o PID da própria tentativa dentro do seu intervalo; senão é descartada e suspende o que depende dela
BINS = "A1-MEDIDA-20261101/bin/"


def le_amostras(linhas):
    out = []
    for ln in linhas:
        f = ln.split(" ", 2)
        if len(f) < 2:
            continue
        procs = []
        if f[1] != "ERRO-DE-CONSULTA" and len(f) > 2:
            for x in f[2].strip().split(";"):
                if x.strip():
                    c = [y.strip() for y in x.split(",")]
                    procs.append((int(c[1]), c[2]))
        out.append((float(f[0]), f[1] != "ERRO-DE-CONSULTA", procs))
    return out


def le_tentativas(linhas):
    tent = []
    for ln in linhas:
        m = re.match(r"TENTATIVA (\S+) n=\d+ pid=(\d+) inicio=([\d.]+) fim=([\d.]+) ", ln)
        if m:
            tent.append((m[1], int(m[2]), float(m[3]), float(m[4])))
    return tent


def problemas_gpu(amostras, tent):
    """nome da tentativa -> lista de problemas ([] = válida)."""
    prob = {}
    for nome, pid, a, b in tent:
        pr, obs = [], 0
        for ts, ok, procs in amostras:
            if not (a <= ts <= b):
                continue
            if not ok:
                pr.append("erro de consulta"); continue
            obs += 1
            for pp, nm in procs:
                if pp != pid or BINS not in nm:
                    pr.append(f"processo alheio {pp} {nm}")
        if obs == 0:
            pr.append("sem observação válida")
        prob[nome] = pr
    return prob


AMO = le_amostras(open(f"{R}/gpu-monitor.log"))
TENT = le_tentativas(open(f"{R}/carga.log"))
amostras = sum(1 for _, ok, _ in AMO if ok)
erro = sum(1 for _, ok, _ in AMO if not ok)
estranhas = sum(1 for _, ok, pr in AMO if ok and any(BINS not in nm for _, nm in pr))
prob = problemas_gpu(AMO, TENT)
descartadas = {k: v for k, v in prob.items() if v}
dentro = sum(1 for ts, ok, _ in AMO if any(a <= ts <= b for _, _, a, b in TENT))
p(f"  GPU: {amostras} amostras válidas, {erro} falhas de consulta, {estranhas} com processo que não é do nosso bin/; {dentro} amostras dentro de "
  f"tentativas ({len(TENT)} tentativas no carga.log); tentativas descartadas por carga: {len(descartadas)} {list(descartadas.items())[:3] if descartadas else ''}")


def aplica_descartes(descartes):
    """Tentativa descartada suspende a validade do que depende dela (por modelo, como o portão de correção)."""
    for nome in descartes:
        m = re.match(r"qual-runs/A1-MEDIDA-20261101/runs/rep(\d)/(1b|3b)/(\S+)$", nome)
        if m:
            valido[(m[2], int(m[1]), m[3])] = False; continue
        m = re.match(r"qual-runs/A1-MEDIDA-20261101/runs/barreira/(1b|3b)/a1$", nome)
        if m:
            valido[(m[1], "L6", "a1")] = False; continue
        m = re.match(r"llama-(1b|3b)-", nome)
        if m:
            l5_ok[m[1]] = False; continue
        raise AssertionError(f"tentativa sem mapeamento: {nome}")


def controles_gpu():
    """Mutações em memória do monitor: as 4 primeiras têm de descartar a primeira tentativa; a última não pode atribuí-la."""
    n0, pid0, a0, b0 = TENT[0]
    meio = (a0 + b0) / 2
    inter = [x for x in AMO if a0 <= x[0] <= b0]
    base = [x for x in AMO if not (a0 <= x[0] <= b0)]
    nm = f"qual-runs/{BINS}m2a-cache"
    cas = {
        "erro de consulta": base + inter + [(meio, False, [])],
        "outro programa": base + inter + [(meio, True, [(999999, "/usr/bin/outro-programa")])],
        "outro PID do mesmo binário": base + inter + [(meio, True, [(999999, nm)])],
        "sem observação": base,
    }
    rej = {k: bool(problemas_gpu(v, TENT[:1])[n0]) for k, v in cas.items()}
    fora = base + inter + [(a0 - 1000, True, [(999999, "/usr/bin/outro-programa")])]
    rej["processo alheio só fora do intervalo (não pode descartar)"] = not problemas_gpu(fora, TENT[:1])[n0]
    return rej


ctl = controles_gpu()
p("  controles negativos da GPU (em memória): " + "; ".join(f"{k}: {'ok' if v else 'FALHOU'}" for k, v in ctl.items()))
assert all(ctl.values()), "controle da GPU falhou"
aplica_descartes(descartadas)
carga_ok = not descartadas

# ---------- 3. tempo ----------
p("\n=== 3. Tempo: espera e d_i (ms por token; média dos tokens por repetição) ===")
res = {}
for mod in MODS:
    for b in BUD:
        if not celula_valida(mod):
            p(f"{mod} {b:>2}%: SUSPENSA (portão de correção reprovou; sem conclusão de velocidade)")
            continue
        m1 = [mean(data[(mod, r)][("L1", b)]) for r in range(1, 6)]
        m2 = [mean(data[(mod, r)][("L2", b)]) for r in range(1, 6)]
        l3 = [mean(data[(mod, r)][("L3", 100)]) for r in range(1, 6)]
        l4 = [mean(data[(mod, r)][("L4", 100)]) for r in range(1, 6)]
        e1 = [a - c for a, c in zip(m1, l4)]
        e2 = [a - c for a, c in zip(m2, l3)]
        d = [a - c for a, c in zip(e1, e2)]
        razao = st.median(m1) / st.median(m2)
        pa, pb = PRED[(mod, b)]
        def dentro(x): return min(d) <= x <= max(d)
        if dentro(pa) and dentro(pb): cls = "inconclusivo (contém as duas)"
        elif dentro(pa): cls = "aponta a cobrança da rodada 51"
        elif dentro(pb): cls = "aponta a cobrança de agora"
        else: cls = "nenhuma das duas"
        res[(mod, b)] = (razao, d)
        p(f"{mod} {b:>2}%: L1 med {st.median(m1):.4f} [{min(m1):.4f},{max(m1):.4f}]  L2 med {st.median(m2):.4f} [{min(m2):.4f},{max(m2):.4f}]  "
          f"razão L1/L2 {razao:.4f}")
        p(f"        espera L1 {st.median(e1):.4f}  espera L2 {st.median(e2):.4f}  d_i = {', '.join(f'{x:+.4f}' for x in d)}  "
          f"intervalo [{min(d):+.4f},{max(d):+.4f}]  previsões {pa:+.3f}/{pb:+.3f} -> {cls}")
p("\n3.2 aceitação (mediana L1 <= 1,05 x mediana L2):")
passa = True
for (mod, b), (r, d) in res.items():
    s = "passa" if r <= 1.05 else ("ACIMA DA MARGEM" if r <= 1.15 else "REGRESSÃO")
    if r > 1.05: passa = False
    p(f"  {mod} {b:>2}%: {r:.4f}  {s}")
p("  resultado:", ("SEM CONCLUSÃO (células suspensas)" if len(res) < 6 else ("PASSA nas 6 células" if passa else "NÃO PASSA")))
p("  ruído entre repetições (máx-mín)/mediana de L1 e L2 por célula:")
for mod in MODS:
    for b in BUD:
        for L in ("L1", "L2"):
            m = [mean(data[(mod, r)][(L, b)]) for r in range(1, 6)]
            p(f"    {mod} {b}% {L}: {100*(max(m)-min(m))/st.median(m):.2f}%")
p("  deriva de L3 e L4 entre repetições (critério 8: acima de 5% da média invalida a decisão):")
for mod in MODS:
    for L in ("L3", "L4"):
        m = [mean(data[(mod, r)][(L, 100)]) for r in range(1, 6)]
        p(f"    {mod} {L}: {100*(max(m)-min(m))/mean(m):.2f}%  médias {', '.join(f'{x:.4f}' for x in m)}")

# ---------- 6. cauda ----------
p("\n=== 6. Cauda de latência (ms por token, 5 repetições juntas) ===")
for mod in MODS:
    for b in BUD:
        for L in ("L1", "L2"):
            v = []
            for r in range(1, 6):
                v += [(x, r, i) for i, x in enumerate(data[(mod, r)][(L, b)])]
            xs = [x for x, _, _ in v]
            med = st.median(xs)
            top = sorted(v, reverse=True)[:10]
            if not celula_valida(mod):
                p(f"{mod} {b:>2}% {L}: SUSPENSA"); continue
            p(f"{mod} {b:>2}% {L}: mediana {med:.3f} p95 {pct(xs,.95):.3f} p99 {pct(xs,.99):.3f} máx {max(xs):.3f} "
              f"acima de 3x: {sum(1 for x in xs if x > 3*med)}  maiores (rep:índice=ms) "
              + " ".join(f"{r}:{i}={x:.1f}" for x, r, i in top))

# ---------- 7. L4 - L3 ----------
p("\n=== 7. Diferença líquida L4 - L3 (motor todo residente menos o residente do m2a), ms/token ===")
for mod in MODS:
    dd = [mean(data[(mod, r)][("L4", 100)]) - mean(data[(mod, r)][("L3", 100)]) for r in range(1, 6)]
    ds = [mean(data[(mod, r)][("L1res", 100)]) - mean(data[(mod, r)][("L3", 100)]) for r in range(1, 6)]
    n = LAYERS[mod]
    p(f"{mod}: L4-L3 mediana {st.median(dd):+.4f} [{min(dd):+.4f},{max(dd):+.4f}] por camada {1000*st.median(dd)/n:+.2f} us; "
      f"(a1 residente - m2a residente, mesma base de comparação: {st.median(ds):+.4f})")

# ---------- backlog 0 (insumo da seção 5) ----------
p("\n=== 5. Insumos do ajuste (lfu_cpu backlog 4 e 0, ms/token, mediana das 5 médias) ===")
for mod in MODS:
    for b in BUD:
        b4 = [mean(data[(mod, r)][("L2", b)]) for r in range(1, 6)]
        b0 = [mean(data[(mod, r)][("L2q", b)]) for r in range(1, 6)]
        p(f"{mod} {b:>2}%: backlog4 {st.median(b4):.4f}  backlog0 {st.median(b0):.4f}  diferença {st.median(b4)-st.median(b0):+.4f}")
    r0 = [mean(data[(mod, r)][("L3q", 100)]) for r in range(1, 6)]
    p(f"{mod} residente backlog0 {st.median(r0):.4f}")

# ---------- 4. llama.cpp ----------
p("\n=== 4. llama.cpp (forcado) contra o motor D1 e o lfu_cpu; ms/token, faixa [mín,máx] ===")
def llama(mod, linha):
    v = []
    for rep in (1, 2, 3):
        t = open(f"{R}/llama/rep{rep}/{mod}/{linha}.txt").read()
        m = re.search(r"total: (\d+) tokens, ms/tok ([\d.]+)", t)
        v.append(float(m.group(2)))
    return v
for mod in MODS:
    if not l5_ok[mod]:
        p(f"{mod}: SUSPENSA (portão da L5 ou carga da GPU reprovou)"); continue
    if not celula_valida(mod):
        p(f"{mod}: SUSPENSA (linhas do motor ou do lfu_cpu reprovadas; sem comparação de velocidade com o llama.cpp)"); continue
    g = llama(mod, "gpu32")
    p(f"{mod} tudo na GPU (F32): {st.median(g):.3f} [{min(g):.3f},{max(g):.3f}]")
    for b in BUD:
        m1 = [mean(data[(mod, r)][("L1", b)]) for r in range(1, 6)]
        m2 = [mean(data[(mod, r)][("L2", b)]) for r in range(1, 6)]
        c = llama(mod, f"c{b}"); n = llama(mod, f"ncm{b}")
        def cmp(a, o):
            if max(a) < min(o): return "motor mais rápido"
            if min(a) > max(o): return "motor mais lento"
            return "empate dentro do ruído"
        p(f"{mod} {b:>2}%: motor [{min(m1):.3f},{max(m1):.3f}]  lfu_cpu [{min(m2):.3f},{max(m2):.3f}]  "
          f"llama cache [{min(c):.3f},{max(c):.3f}]  llama -ncmoe [{min(n):.3f},{max(n):.3f}]")
        p(f"        llama cache/motor {st.median(c)/st.median(m1):.2f}x ({cmp(m1,c)});  llama -ncmoe/motor {st.median(n)/st.median(m1):.2f}x ({cmp(m1,n)});  "
          f"llama cache/lfu_cpu {st.median(c)/st.median(m2):.2f}x")

# ---------- L6 ----------
p("\n=== L6. Motor em barreira por token: contadores (uma execução por modelo) ===")
for mod in MODS:
    for ln in open(f"{R}/runs/barreira/{mod}/a1/saida.txt"):
        if ln.startswith("motor_barreira") and "hits" in ln:
            p(f"{mod}: {ln.strip()[:170]}")
    for ln in open(f"{R}/runs/barreira/{mod}/a1/saida.txt"):
        if ln.startswith("motor_barreira") and "hits" not in ln and "rot" in ln:
            p(f"{mod}: {ln.strip()[:130]}")


# ---------- L6 contra o simulador (protocolo token com barreira, mesmas conversas) ----------
p("\n=== L6 contra o simulador (needed hits cpu cópias descartadas adiadas; aquecimento 0-3, medida 4-7) ===")
TRD = {"1b": "granite-3.1-1b-a400m-instruct", "3b": "granite-3.1-3b-a800m-instruct"}
DOORBELL_US = 5.41  # mediana do doorbell no A1.0 (qual-runs/A1-DOORBELL-20261031/resultado.txt)
est = {}
for mod in MODS:
    d = os.path.join("qual-runs", "TRACOS-20261029", TRD[mod])
    tr, te = moe_sim.load(d, "treino"), moe_sim.load(d, "teste")
    L, _, k = tr[0].shape
    E = int(max(a.max() for a in tr + te)) + 1
    pp = dict(moe_sim.PARAMS_RUNTIME[mod])
    linhas = {}
    for ln in open(f"{R}/runs/barreira/{mod}/a1/saida.txt"):
        m = re.match(r"^motor_barreira\s+(\d+)% \| needed\s+(\d+) hits\s+(\d+) cpu\s+(\d+) cópias\s+(\d+) descart\.\s+(\d+) adiadas\s+(\d+)", ln)
        if m:
            linhas[int(m[1])] = tuple(int(x) for x in m.groups()[1:])
    for b in BUD:
        cap = int(b / 100 * L * E)
        s_ = moe_sim.Sim("lfu_cpu", cap, L, E, k, pp, protocol="token", backlog=4, token_barreira=True)
        s_.run(tr, False); s_.run(te, True)
        sim_c = tuple(s_.stats[x] for x in ("needed", "hits", "cpu", "copies_bg", "dropped", "copies_deferred"))
        p(f"{mod} {b:>2}%: motor {linhas.get(b)}  simulador {sim_c}  -> {'iguais' if linhas.get(b) == sim_c else 'DIFERENTES'}")
        # estimativa do doorbell: camadas com falta por token (Sim por evento no protocolo token) x latência do doorbell
        s2 = moe_sim.Sim("lfu_cpu", cap, L, E, k, pp, protocol="token", backlog=4)
        s2.run(tr, False); s2.run(te, True)
        est[(mod, b)] = DOORBELL_US * s2.stats["layers_cpu"] / s2.stats["tokens"] / 1000
p("\n=== 3 (nota de leitura, emenda 5): estimativa do custo do caminho do host, doorbell do A1.0 (5,41 us) x camadas com falta por token ===")
for mod in MODS:
    for b in BUD:
        if (mod, b) in res:
            d = res[(mod, b)][1]
            md = st.median(d)
            cob = (f"a estimativa cobre {100*est[(mod, b)]/md:.0f}% da mediana de d" if md > 0.05 else
                   ("mediana de d positiva mas perto de zero, percentual sem sentido" if md > 0 else "mediana de d não positiva"))
            p(f"{mod} {b:>2}%: estimativa {est[(mod, b)]:.3f} ms por token; d observado [{min(d):+.3f},{max(d):+.3f}] (mediana {md:+.3f}); {cob}")

open(f"{R}/analise.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
