#!/usr/bin/env python3
"""A1-DIAG10: análise da janela do diagnóstico dos 10% (criterio.txt v6). Lê só os arquivos gravados pelo rodar.sh e escreve analise-e<etapa>.txt.
A decisão vem SÓ de decide.py (d dos binários sem instrumentação); os intervalos do binário I são descritivos.
Uso, da raiz do projeto:
  taskset -c 16-27 tools/.venv-ref/bin/python -I -B qual-runs/A1-DIAG10-20261103/analise.py [DIR]      (DIR padrão: a pasta deste script)
Camadas, na ordem do criterio.txt: portão de correção por bloco (com controles negativos) -> cruzamento do monitor da GPU por PID -> regra 0c (frequência) ->
controle de trabalho por par -> orçamento da instrumentação -> passo 0 e regra (a)."""
import os, re, statistics as st, sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import decide as D  # noqa: E402
import amostrador as AM  # noqa: E402

R = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else AQUI
MODS = ["1b", "3b"]
TOKS = {"1b": 3451, "3b": 3814}
TOT = {"1b": 82824, "3b": 122048}          # pares (token, camada) das conversas 4 a 7
C_COPY = {"1b": 0.223, "3b": 0.335}        # escala convencional do tools/moe_sim.py (criterio.txt, seção 3)
LOGIT_MAX = 1e-4
BINS = ("A1-MEDIDA-20261101/bin/", "A1-DIAG10-20261103/bin/")
ORCAM = {"1b": {1: [10, 25], 2: [10]}, "3b": {1: [10], 2: [10]}}
CFGS = {1: ["C", "N0", "I"], 2: ["N0", "A1w", "A1m"]}
BL = ("m2a-b4", "a1")
POL_M2A = [("resident", 100), ("lfu_cpu", 10)]
POL_A1 = [("resident", 100), ("motor_cheio", 100), ("motor_evento", 10)]


def mean(v): return sum(v) / len(v)


def pols(bl, mod, etapa):
    base = POL_M2A if bl == "m2a-b4" else POL_A1
    extra = [("lfu_cpu" if bl == "m2a-b4" else "motor_evento", b) for b in ORCAM[mod][etapa] if b != 10]
    return base + extra


# ---------- portão de correção ----------
def valida(t, mod, motor, politicas):
    """Problemas de um bloco: linhas esperadas, logits <= 1e-4, roteamento (<= 0,01% fora do traço), 'true' no needed do motor e a frase de resultado."""
    bad, rows = [], {}
    for ln in t.splitlines():
        m = re.match(r"^(resident|lfu_cpu|motor_cheio|motor_evento)\s+(\d+)%", ln)
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
        if (not motor) or pol != "resident":
            r = re.search(r"(\d+)/(\d+)", " ".join(cols[3:-1]))
            if r is None:
                bad.append(f"{pol} {b}%: sem a coluna de roteamento")
            elif int(r[2]) != TOT[mod] or (int(r[2]) - int(r[1])) * 10000 > int(r[2]):
                bad.append(f"{pol} {b}%: roteamento {r[1]}/{r[2]} fora de 0,01%")
        if motor and pol != "resident" and "true" not in cols[-2]:
            bad.append(f"{pol} {b}%: needed não confere")
    if "resultado: logits dentro da tolerância" not in t:
        bad.append("frase de resultado ausente")
    return bad


def controle_afinidade(t, cfg, bl):
    """Problemas das linhas CONTROLE da afinidade efetiva (P2-1 da rodada 35): na A1w toda linha diz 'ok' e há uma por linha de política; na A1m (só no a1)
    a thread principal está em 17. Outras configurações não têm linha."""
    ls = [ln for ln in t.splitlines() if ln.startswith("CONTROLE ")]
    if cfg == "A1w":
        nec = 3 if bl == "a1" else 2
        if len(ls) < nec:
            return [f"só {len(ls)} linhas CONTROLE A1w, esperadas {nec}"]
        return [f"controle da A1w: {ln}" for ln in ls if not ln.startswith("CONTROLE A1w") or not ln.rstrip().endswith("ok")]
    if cfg == "A1m" and bl == "a1":
        if not ls:
            return ["sem linha CONTROLE A1m"]
        return [f"controle da A1m: {ln}" for ln in ls if not ln.rstrip().endswith("thread principal em 17")]
    return []


def controles_afinidade():
    ok = "CONTROLE A1w motor_evento 10%: 8 participantes, núcleos físicos distintos: ok\n"
    base = ok * 3
    casos = {
        "A1w com FALHOU": base.replace("distintos: ok", "FALHOU", 1),
        "A1w com linha a menos": ok * 2,
        "A1m com a principal em 0-15": "CONTROLE A1m motor_evento 10%: thread principal em 0-15\n",
    }
    res = {k: controle_afinidade(v, "A1m" if "A1m" in k else "A1w", "a1") != [] for k, v in casos.items()}
    res["A1w correta passa"] = controle_afinidade(base, "A1w", "a1") == []
    res["A1m correta passa"] = controle_afinidade("CONTROLE A1m motor_evento 10%: thread principal em 17\n", "A1m", "a1") == []
    return res


def controles_negativos(t, mod):
    """Cada mutação de uma saída real do a1-motor tem de reprovar; a saída em si tem de passar."""
    ok = valida(t, mod, True, POL_A1) == []
    muts = {
        "resumo FALHOU": t.replace("resultado: logits dentro da tolerância", "resultado: FALHOU"),
        "linha ausente": "\n".join(ln for ln in t.splitlines() if not re.match(r"^motor_evento\s+10%", ln)),
        "roteamento fora": re.sub(rf"(\d+)/{TOT[mod]}", f"{TOT[mod] - 200}/{TOT[mod]}", t, count=1),
        "logits 2e-4": re.sub(r"\d\.\de-\d", "2.0e-4", t, count=1),
        "needed falso": t.replace("true rot", "false rot", 1),
    }
    return ok, {k: valida(v, mod, True, POL_A1) != [] for k, v in muts.items()}


# ---------- leitura dos blocos ----------
def bdir(etapa, rep, mod, cfg, bl): return f"{R}/e{etapa}/rep{rep}/{mod}/{cfg}/{bl}"


def serie(d, bl, pol, b):
    pref = "m2a-ms" if bl == "m2a-b4" else "a1-motor-ms"
    return [float(x) for x in open(f"{d}/ms/{pref}-{pol}-{b}.txt").read().split()]


def contadores(t, motor, pol, b):
    """Contadores exatos da linha (pol, b%) -> dict(hits, cpu, dropped, copies, deferred) ou None."""
    for ln in t.splitlines():
        if ln.count("|") != 2 or not re.match(rf"^{pol}\s+{b}%\s*\|", ln):
            continue
        mid = ln.split("|")[1]
        if motor:
            m = re.search(r"needed\s+(\d+)\s+hits\s+(\d+)\s+cpu\s+(\d+)\s+cópias\s+(\d+)\s+descart\.\s+(\d+)\s+adiadas\s+(\d+)", mid)
            if m:
                return dict(hits=int(m[2]), cpu=int(m[3]), copies=int(m[4]), dropped=int(m[5]), deferred=int(m[6]))
        else:
            m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", mid)
            if m:
                return dict(hits=int(m[1]), cpu=int(m[5]), copies=int(m[2]) + int(m[3]), dropped=int(m[4]), deferred=0)
    return None


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
    """nome da tentativa -> lista de problemas ([] = válida): erro de consulta, processo que não é o PID da tentativa, ou nenhuma observação."""
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
                if pp != pid or not any(x in nm for x in BINS):
                    pr.append(f"processo alheio {pp} {nm}")
        if obs == 0:
            pr.append("sem observação válida")
        prob[nome] = pr
    return prob


def controles_gpu(amo, tent):
    n0, pid0, a0, b0 = tent[0]
    meio = (a0 + b0) / 2
    inter = [x for x in amo if a0 <= x[0] <= b0]
    base = [x for x in amo if not (a0 <= x[0] <= b0)]
    nm = f"qual-runs/{BINS[1]}N/a1-motor"
    cas = {
        "erro de consulta": base + inter + [(meio, False, [])],
        "outro programa": base + inter + [(meio, True, [(999999, "/usr/bin/outro-programa")])],
        "outro PID do mesmo binário": base + inter + [(meio, True, [(999999, nm)])],
        "sem observação": base,
    }
    rej = {k: bool(problemas_gpu(v, tent[:1])[n0]) for k, v in cas.items()}
    fora = base + inter + [(a0 - 1000, True, [(999999, "/usr/bin/outro-programa")])]
    rej["processo alheio só fora do intervalo (não pode descartar)"] = not problemas_gpu(fora, tent[:1])[n0]
    return rej


def analisa(etapa):
    out = []
    def p(*a): out.append(" ".join(str(x) for x in a))
    E = f"{R}/e{etapa}"
    p(f"=== A1-DIAG10, etapa {etapa}: portão de correção, carga, frequência, trabalho, instrumentação e decisão (criterio.txt v6) ===")
    # --- 1. portão de correção ---
    valido, motivo = {}, {}          # (mod, rep, cfg, bl) -> bool ; -> texto
    texto = {}
    amostra_a1 = None
    for mod in MODS:
        for rep in range(1, 6):
            for cfg in CFGS[etapa]:
                for bl in BL:
                    d = bdir(etapa, rep, mod, cfg, bl)
                    k = (mod, rep, cfg, bl)
                    try:
                        t = open(f"{d}/saida.txt").read()
                    except OSError:
                        valido[k], motivo[k] = False, "saída ausente"
                        continue
                    texto[k] = t
                    pr = valida(t, mod, bl == "a1", pols(bl, mod, etapa)) + controle_afinidade(t, cfg, bl)
                    if not pr:
                        for pol, b in pols(bl, mod, etapa):
                            try:
                                serie(d, bl, pol, b)
                            except OSError:
                                pr.append(f"série ms ausente {pol} {b}%")
                    valido[k], motivo[k] = not pr, "; ".join(pr)
                    if bl == "a1" and not pr and amostra_a1 is None:
                        amostra_a1 = (t, mod)
    if amostra_a1 is None:
        p("  nenhum bloco a1 válido para os controles negativos: sem conclusões")
        return out, {}
    ok0, neg = controles_negativos(*amostra_a1)
    p("  controles negativos em memória (uma saída real passa; cada mutação tem de reprovar):", "saída real passa" if ok0 else "SAÍDA REAL NÃO PASSA",
      "; ".join(f"{k}: {'reprova' if v else 'NÃO REPROVA'}" for k, v in neg.items()))
    assert ok0 and all(neg.values()), "controle negativo do portão falhou"
    caf = controles_afinidade()
    p("  controles da afinidade efetiva (em memória): " + "; ".join(f"{k}: {'ok' if v else 'FALHOU'}" for k, v in caf.items()))
    assert all(caf.values()), "controle da afinidade falhou"
    ruins = {k: m for k, m in motivo.items() if not valido[k] and m}
    p(f"  blocos com falha de logits, roteamento, needed ou arquivos: {sum(1 for k in valido if not valido[k])} de {len(valido)}")
    for k, m in sorted(ruins.items()):
        p(f"    FALHA {k}: {m}")
    # --- 2. GPU ---
    arq_g, arq_c = f"{R}/gpu-monitor-e{etapa}.log", f"{R}/carga-e{etapa}.log"
    try:
        AMO, TENT = le_amostras(open(arq_g)), le_tentativas(open(arq_c))
    except OSError:
        p("  monitor da GPU ou carga.log ausente: nenhuma tentativa pode ser validada, o que sobra é SEM CONCLUSÃO")
        return out, {}
    ctl = controles_gpu(AMO, TENT)
    p("  controles negativos da GPU (em memória): " + "; ".join(f"{k}: {'ok' if v else 'FALHOU'}" for k, v in ctl.items()))
    assert all(ctl.values()), "controle da GPU falhou"
    prob = problemas_gpu(AMO, TENT)
    desc = {k: v for k, v in prob.items() if v}
    p(f"  GPU: {sum(1 for _, ok, _ in AMO if ok)} amostras válidas, {sum(1 for _, ok, _ in AMO if not ok)} falhas de consulta; {len(TENT)} tentativas; descartadas por carga: {len(desc)}")
    visto = set()
    for nome in prob:
        m = re.search(rf"/e{etapa}/rep(\d)/(1b|3b)/(\w+)/(m2a-b4|a1)$", nome)
        if not m:
            raise AssertionError(f"tentativa sem mapeamento: {nome}")
        k = (m[2], int(m[1]), m[3], m[4])
        visto.add(k)
        if prob[nome]:
            valido[k], motivo[k] = False, f"carga na GPU: {prob[nome][:2]}"
    for k in valido:
        if k not in visto and valido[k]:
            valido[k], motivo[k] = False, "bloco sem tentativa registrada no carga.log"
    # blocos suspensos pelo rodar.sh
    try:
        for ln in open(f"{R}/suspensas-e{etapa}.txt"):
            p("  suspensa pelo rodar.sh:", ln.strip())
    except OSError:
        p("  nenhum bloco suspenso pelo rodar.sh")
    # --- 3. regra 0c ---
    for mod in MODS:
        for rep in range(1, 6):
            for cfg in CFGS[etapa]:
                f1, f2 = f"{bdir(etapa, rep, mod, cfg, 'm2a-b4')}/freq.txt", f"{bdir(etapa, rep, mod, cfg, 'a1')}/freq.txt"
                try:
                    pr = AM.compara(f1, f2)
                except OSError:
                    pr = ["freq.txt ausente"]
                if pr:
                    p(f"  0c reprova {mod} rep{rep} {cfg}: {pr}")
                    for bl in BL:
                        valido[(mod, rep, cfg, bl)], motivo[(mod, rep, cfg, bl)] = False, f"0c: {pr}"
    # --- 4. séries e contadores ---
    L, CT = {}, {}
    for mod in MODS:
        for rep in range(1, 6):
            for cfg in CFGS[etapa]:
                if not all(valido[(mod, rep, cfg, bl)] for bl in BL):
                    continue
                a, m = bdir(etapa, rep, mod, cfg, "a1"), bdir(etapa, rep, mod, cfg, "m2a-b4")
                L[(mod, rep, cfg)] = {"L1": mean(serie(a, "a1", "motor_evento", 10)), "L4": mean(serie(a, "a1", "motor_cheio", 100)),
                                      "L2": mean(serie(m, "m2a-b4", "lfu_cpu", 10)), "L3": mean(serie(m, "m2a-b4", "resident", 100))}
                if 25 in ORCAM[mod][etapa]:
                    L[(mod, rep, cfg)].update(L1_25=mean(serie(a, "a1", "motor_evento", 25)), L2_25=mean(serie(m, "m2a-b4", "lfu_cpu", 25)))
                CT[(mod, rep, cfg)] = {"L1": contadores(texto[(mod, rep, cfg, "a1")], True, "motor_evento", 10),
                                       "L2": contadores(texto[(mod, rep, cfg, "m2a-b4")], False, "lfu_cpu", 10)}
    def par_ok(mod, rep, ref, alt):
        if (mod, rep, ref) not in CT or (mod, rep, alt) not in CT:
            return False
        a, b = CT[(mod, rep, ref)], CT[(mod, rep, alt)]
        if None in (a["L1"], a["L2"], b["L1"], b["L2"]):
            return False
        return D.par_valido(a["L1"], b["L1"], TOKS[mod], C_COPY[mod]) and D.par_valido(a["L2"], b["L2"], TOKS[mod], C_COPY[mod])
    def lista(mod, cfg):
        return [L[(mod, rep, cfg)] for rep in range(1, 6)] if all((mod, rep, cfg) in L for rep in range(1, 6)) else None
    dd = lambda x: (x["L1"] - x["L4"]) - (x["L2"] - x["L3"])
    # --- 5. por célula ---
    res = {}
    for mod in MODS:
        p(f"\n--- célula {mod} 10% ---")
        ls = {c: lista(mod, c) for c in CFGS[etapa]}
        for c, v in ls.items():
            if v is None:
                faltam = [rep for rep in range(1, 6) if (mod, rep, c) not in L]
                p(f"  {c}: blocos inválidos ou ausentes nas repetições {faltam}")
        for c, v in ls.items():
            if v:
                p(f"  d_i({c}) = " + " ".join(f"{dd(x):+.3f}" for x in v) + f"  mediana {st.median(dd(x) for x in v):+.3f}")
                p(f"    L1 " + " ".join(f"{x['L1']:.3f}" for x in v) + "  L2 " + " ".join(f"{x['L2']:.3f}" for x in v)
                  + "  L3 " + " ".join(f"{x['L3']:.3f}" for x in v) + "  L4 " + " ".join(f"{x['L4']:.3f}" for x in v))
        if 25 in ORCAM[mod][etapa]:
            for c, v in ls.items():
                if v:
                    d25 = [(x["L1_25"] - x["L4"]) - (x["L2_25"] - x["L3"]) for x in v]
                    p(f"  [descritivo, não entra em regra] d_i({c}) a 25% = " + " ".join(f"{x:+.3f}" for x in d25))
        if etapa == 1:
            if not all(ls.values()):
                res[mod] = "SEM CONCLUSÃO (portão de correção, carga ou frequência: blocos inválidos)"
            else:
                vp = {pair: [par_ok(mod, rep, *pair) for rep in range(1, 6)] for pair in (("C", "N0"), ("N0", "I"))}
                p("  controle de trabalho por par (L1 e L2 juntas):", "; ".join(f"{a}->{b}: {sum(v)}/5 válidos" for (a, b), v in vp.items()))
                if sum(vp[("C", "N0")]) < 5:
                    res[mod] = D.SEM_CONCLUSAO_TRABALHO + " (C contra N0)"
                else:
                    r0 = D.passo0_etapa1(ls["C"], ls["N0"])
                    res[mod] = r0 if r0 != "OK" else "PASSO 0 OK: o excesso se reproduz com C e N0; a etapa 2 pode rodar"
                    p(f"  passo 0: d(C) mediana {st.median(D.d_list(ls['C'])):+.3f}, d(N0) mediana {st.median(D.d_list(ls['N0'])):+.3f} -> {r0}")
                # orçamento da instrumentação (I contra N0)
                if ls["I"]:
                    dif = [dd(a) - dd(b) for a, b in zip(ls["I"], ls["N0"])]
                    custo = {k: st.median((a[k] - b[k]) / b[k] for a, b in zip(ls["I"], ls["N0"])) for k in ("L1", "L2", "L3", "L4")}
                    viol = abs(st.median(dif)) > 0.05 or any(abs(v) > 0.01 for v in custo.values())
                    p("  orçamento da instrumentação (I contra N0): mediana(d(I)-d(N0)) "
                      f"{st.median(dif):+.3f} ms, dispersão (min..max) {min(dif):+.3f}..{max(dif):+.3f}; custo relativo por linha " + " ".join(f"{k} {100 * v:+.2f}%" for k, v in custo.items())
                      + f" -> {'VIOLA: intervalos descritivos NÃO CONFIÁVEIS' if viol else 'dentro do orçamento'}")
                    if not all(vp[('N0', 'I')]):
                        p("  AVISO: par N0/I com contadores fora do limite em alguma repetição; os intervalos descritivos não são confiáveis")
        else:
            if not all(ls.values()):
                res[mod] = "SEM CONCLUSÃO (portão de correção, carga ou frequência: blocos inválidos)"
            else:
                # a etapa 1 vem de outra pasta de resultados; o d(N) dela entra por um arquivo com a mediana (escrito pela análise da etapa 1)
                try:
                    d_e1 = float(open(f"{R}/d-etapa1-{mod}.txt").read().split()[0])
                except OSError:
                    res[mod] = "SEM CONCLUSÃO (falta d-etapa1-" + mod + ".txt, escrito pela análise da etapa 1)"
                    continue
                vw = [par_ok(mod, rep, "N0", "A1w") for rep in range(1, 6)]
                vm = [par_ok(mod, rep, "N0", "A1m") for rep in range(1, 6)]
                p(f"  controle de trabalho: N0->A1w {sum(vw)}/5 válidos; N0->A1m {sum(vm)}/5 válidos")
                # o passo 0 já foi decidido na etapa 1; aqui entra a referência contemporânea reaplicando o portão (fator())
                e2 = {"N0": ls["N0"], "A1w": ls["A1w"], "A1m": ls["A1m"], "valid": {"A1w": vw, "A1m": vm}}
                r = {}
                for x in ("A1w", "A1m"):
                    if sum(e2["valid"][x]) < 5:
                        r[x] = (D.SEM_CONCLUSAO_TRABALHO, {})
                    else:
                        r[x] = D.fator(e2["N0"], e2[x], d_e1)
                    s_, det = r[x]
                    if det:
                        p(f"  {x}: dL1 {det['dL1']:+.3f} dL2 {det['dL2']:+.3f} dL3 {det['dL3']:+.3f} dL4 {det['dL4']:+.3f}"
                          + (f"; R {det['R']:.2f} Mfr {det['Mfr']:.2f}; r_i " + " ".join(f"{v:+.3f}" for v in det["r"]) + "; m_i " + " ".join(f"{v:+.3f}" for v in det["m"])
                             + "; c_i " + " ".join(f"{v:+.3f}" for v in det["c"]) if "R" in det else ""))
                    p(f"  {x} -> {s_}")
                res[mod] = D.combina(r["A1w"], r["A1m"])
    p("\n=== RESULTADO POR CÉLULA ===")
    for mod in MODS:
        p(f"  {mod} 10%: {res.get(mod)}")
    if etapa == 2 and res.get("1b") and res.get("3b"):
        p("  Regra do critério: o 3B só confirma o sinal do 1B; se o 1B der SUSTENTADA e o 3B não, vale o resultado do 1B com a nota 'o 3B não confirma'.")
    if etapa == 1:
        for mod in MODS:
            ls = lista(mod, "N0")
            if ls:
                open(f"{R}/d-etapa1-{mod}.txt", "w").write(f"{st.median(dd(x) for x in ls):.6f}\n")
    return out, res


def main():
    for etapa in (1, 2):
        if os.path.isdir(f"{R}/e{etapa}"):
            out, _ = analisa(etapa)
            txt = "\n".join(out) + "\n"
            open(f"{R}/analise-e{etapa}.txt", "w").write(txt)
            print(txt)


if __name__ == "__main__":
    main()
