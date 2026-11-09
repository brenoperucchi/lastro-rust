#!/usr/bin/env python3
"""Extrator do perfil do prefill do Lastro (criterio-perfil-prefill-20261109.md, versão 3, Parte B), só por medição, descritivo.

    python3 -I perfil-extrai.py --saida DIR_DA_JANELA [--ids cmp-ids-prompt.json] [--refs 5c-referencia] [--textos 5c-textos/textos.json]

Só biblioteca padrão. NÃO roda nada, NÃO usa GPU: lê o que roda-perfil-janela.sh gravou e grava DIR/perfil-resultado.json e DIR/perfil-resultado.md.
Sem meta e sem PASSA: P1, P2 e P3 são PREVISÕES pré-registradas, e o resultado de cada uma é "dentro da previsão" ou "fora da previsão" com o dado ao lado.

Layout lido (escrito por roda-perfil-janela.sh e, por baixo, roda-cmp-lastro.sh):
  DIR/integridade-antes.txt, integridade-depois.txt, integridade-fixos.txt, ambiente.txt, orcamento.json ("modelos": um só), ordem.txt
  DIR/smoke-L100/veredito.txt e DIR/smoke-LC/veredito.txt (julgados por cmp-extrai.py --smoke [--degrau2])
  DIR/celulas/lastro-<modelo>-<série>-<texto>-quente/ (celula.json, monitor.csv, portao.txt, pK/...)   série: L10 (padrão), L100 (--fracao-slots 1.0), LC (--camadas-cpu N)
  DIR/p4/lastro-<modelo>-<série>-<texto>-quente.gpu.csv  ("timestamp, utilização %", nvidia-smi -lms 100)  e  .cpu.csv  ("epoch_ms,pid,cpu_pct", por /proc, ~100 ms)

A validade de cada célula REAPROVEITA o le_celula_lastro de cmp-extrai.py (portão, monitor, ids, contagens, integro=1, rc), ligando os modos dos degraus: L100 herda as regras da
Emenda 1 do degrau 1 (linha `slots N de N`, cpu/needed == 0, cmdline com --fracao-slots 1.0, smoke) e LC as da Emenda 1 do degrau 2 (linha `camadas-cpu N de L; slots S de T`,
cpu*L == needed*N exato por medida, smoke); L10 exige a linha `slots` com a fração PADRÃO (10%) impressa e a linha de comando sem --fracao-slots nem --camadas-cpu.
Definições: ms/posição = mediana do prefill_ms da célula / nº de posições do prompt (89, 170 e 364, template incluído); razão P3 = ms/posição / mediana do passo de decode da MESMA célula.
Célula ausente = NÃO RODOU; sem prova = SEM PROVA. Nada é estimado, nada vira PASSA.
"""
import argparse
import importlib.util
import json
import math
import os
import re
import statistics
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))


def carrega_cmp():
    spec = importlib.util.spec_from_file_location("cmp_extrai", os.path.join(AQUI, "cmp-extrai.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


X = carrega_cmp()

MODELOS = ("1b", "3b")
TEXTOS = ("curto", "medio", "longo")
SERIES = ("L10", "L100", "LC")
N_POS = {"curto": 89, "medio": 170, "longo": 364}          # posições do prompt (template incluído), contagens do ensaio e de cmp-ids-prompt.json
SERIE_META = {"L10": "L", "L100": "L100", "LC": "LC"}      # o "serie" que roda-cmp-lastro.sh grava em celula.json
ROTULO = {
    "L10": "L10: Lastro padrão (10% dos experts na GPU)",
    "L100": "L100: Lastro 100% dos experts na GPU (--fracao-slots 1.0)",
    "LC": "LC: Lastro camadas inteiras na CPU (--camadas-cpu 22 no 1B, 29 no 3B)",
}
FRACAO_PADRAO = 0.10        # FRACAO_SLOTS do binário (slots = int(0,10 x camadas x experts), mesma conta f64 do Rust)
P1_TOL = 0.15               # P1: ±15% entre os três textos, em torno da média da série
P2_TOL = 0.10               # P2: ±10% do valor da janela anterior (Parte A do critério)
P3_TOL = 0.15               # P3: ±15% sobre a razão da Parte A
P4_PREFILL_DICA_MS = 3000.0  # P4: com prefill >= ~3 s a leitura é só uma dica; abaixo disso, sem resolução
# valores de referência da Parte A (ms por posição, mediana quente das janelas já feitas); P2 compara célula a célula
REF_MS_POS = {
    "1b": {"L10": {"curto": 8.9, "medio": 9.0, "longo": 8.5}, "L100": {"curto": 1.58, "medio": 1.58, "longo": 1.58}, "LC": {"curto": 8.8, "medio": 8.8, "longo": 8.7}},
    "3b": {"L10": {"curto": 17.5, "medio": 17.7, "longo": 17.4}, "L100": {"curto": 2.52, "medio": 2.54, "longo": 2.55}, "LC": {"curto": 17.5, "medio": 17.4, "longo": 17.2}},
}
# P3: razão ms/posição ÷ passo de decode; faixa = (0,66 a 0,67) ±15% no L100 e (0,41 a 0,43) ±15% no L10 e no LC
P3_FAIXA = {"L100": (0.66 * (1 - P3_TOL), 0.67 * (1 + P3_TOL)), "L10": (0.41 * (1 - P3_TOL), 0.43 * (1 + P3_TOL)), "LC": (0.41 * (1 - P3_TOL), 0.43 * (1 + P3_TOL))}


# integridade FIXA (Emenda 1 do critério): chaves congeladas que o manifesto integridade-fixos.txt TEM de trazer, além do tokenizer do modelo da janela (mesma lista do FIXO de roda-perfil-janela.sh)
FIXO_CHAVES = ("criterio-perfil", "binario-lastro", "script-perfil-extrai", "script-amostra-cpu", "script-cmp-extrai", "script-lastro", "script-monitor", "script-janela", "script-limite",
               "criterio-cmp", "criterio-degrau1", "criterio-degrau2", "cmp-ids-prompt", "texto-curto", "texto-medio", "texto-longo", "textos-json", "ref5c-1b", "ref5c-3b")
# binários das janelas de referência da Parte A (P2): L10 das janelas cmp, L100 das d1, LC das d2 (integridade-antes.txt de cada uma)
BIN_REF = {"L10": "f0a272db2df8aca6642073b830cc8234f2d2434e01c8ae6ef618886e6e3dc2df", "L100": "936d0c146d750f4849b7724ba5ae7ddd099618629a4e12bd2984758b801c3551",
           "LC": "e4c18f3a9dea7d41236d1b721046f0d69a3be07863a5be8a2cc1a262b9e331d4"}
AVISO_P2 = "P2 compara com binário diferente no L10 e L100 (mudanças aditivas) e com amostragem P4 a 10 Hz; fora da previsão não isola a causa"


def integridade_fixos(saida, m):
    """Manifesto FIXO OBRIGATÓRIO: arquivo legível e não vazio, UMA entrada completa (`chave esperado observado OK`) para cada chave congelada da janela, esperado == observado (64 hex),
    observado == o sha de integridade-antes.txt. Ausente, vazio, truncado, chave faltando, NAO-CONFERIDO, FALHA ou ilegível = problema global (todas as células SEM PROVA)."""
    arq = os.path.join(saida, "integridade-fixos.txt")
    txt = X.le_txt(arq)
    if txt is None:
        return ["integridade FIXA: manifesto integridade-fixos.txt ausente ou ilegível (janela interrompida antes de gravá-lo?)"]
    linhas = [x for x in txt.splitlines() if x.strip()]
    if not linhas:
        return ["integridade FIXA: manifesto integridade-fixos.txt vazio"]
    mot, vistas = [], {}
    antes = X.le_shas(os.path.join(saida, "integridade-antes.txt"))
    for x in linhas:
        c = x.split()
        if len(c) != 4:
            mot.append("integridade FIXA: linha truncada ou ilegível no manifesto: %r" % x.strip()[:80])
            continue
        k, esp, obs, est = c
        if k in vistas:
            mot.append("integridade FIXA: chave %s repetida no manifesto" % k)
        vistas[k] = c
        if est != "OK":
            mot.append("integridade FIXA: %s com estado %s (esperado OK)" % (k, est))
        elif not (re.fullmatch(r"[0-9a-f]{64}", esp) and esp == obs):
            mot.append("integridade FIXA: %s com sha esperado/observado diferentes ou inválidos" % k)
        elif antes.get(k) is None:
            mot.append("integridade FIXA: %s ausente de integridade-antes.txt (o snapshot inicial tem de ter toda chave obrigatória)" % k)
        elif antes.get(k) != obs:
            mot.append("integridade FIXA: %s com sha diferente do de integridade-antes.txt" % k)
    falta = [k for k in FIXO_CHAVES + ("tokenizer-%s" % m,) if k not in vistas]
    if falta:
        mot.append("integridade FIXA: chave(s) faltando no manifesto: %s" % ", ".join(falta))
    return sorted(set(mot))


def compara_com_referencia(refs, m, t, meds):
    """Ids gerados de CADA medida contra a referência 5c, antes de descartar as medidas: primeira divergência, ids medido/referência, gap da referência, processo/repetição.
    Referência ausente fica registrada explicitamente (nunca igualdade). Só reporta: tempos e P3 seguem (as contagens são conferidas à parte)."""
    ref = refs.ref(m, t)
    if not ref or not isinstance(ref.get("gerados"), list):
        return {"referencia_disponivel": False, "sequencias_diferentes": None, "divergencias": [], "marca": "referência 5c ausente: igualdade de sequência NÃO verificada"}
    refg, gaps, div = ref["gerados"], ref.get("gaps") or [], []
    for x in meds:
        g = x.get("gerados") or []
        i = X.primeira_divergencia(list(g), list(refg))
        if i is not None:
            div.append({"processo": x["processo"], "rotulo": x["rotulo"], "passo": i, "id_medido": g[i] if i < len(g) else None, "id_referencia": refg[i] if i < len(refg) else None,
                        "gap_referencia": gaps[i] if i < len(gaps) else None})
    return {"referencia_disponivel": True, "sequencias_diferentes": bool(div), "divergencias": div, "n_medidas_divergentes": len(div), "n_medidas": len(meds),
            "marca": ("sequências diferentes da referência 5c em %d de %d medidas (tempos e P3 mantidos)" % (len(div), len(meds))) if div else "mesma sequência da referência 5c"}


def chave(m, s, t):
    return "lastro-%s-%s-%s-quente" % (m, s, t)


def linha_slots_padrao(m):
    """Linha `slots N de M` esperada na fração padrão: M = camadas x experts do modelo, N = int(0,10 x M)."""
    L_, E_ = X.CAMADAS_EXPERTS[m]
    tot = L_ * E_
    return (int(FRACAO_PADRAO * tot), tot)


# ---------------------------------------------------------------------------------------------------------------------------------- P4 (descritivo)
def le_gpu_util(p, ini=None, fim=None):
    """([(carimbo, % de utilização)] DENTRO de [ini, fim] epoch, nº de amostras legíveis FORA do intervalo ou com carimbo ilegível); None se o arquivo não existe.
    Linhas `timestamp, 87 %`; sem intervalo legível (ini ou fim None) nada é aceito (tudo conta como fora)."""
    t = X.le_txt(p)
    if t is None:
        return None
    r, fora = [], 0
    for linha in t.splitlines():
        c = [x.strip() for x in linha.split(",")]
        if len(c) < 2:
            continue
        m_ = re.match(r"(\d+(?:\.\d+)?)", c[-1])
        if not m_:
            continue
        v = float(m_.group(1))
        if math.isfinite(v) and 0 <= v <= 100:
            ts = X.ts_epoch(c[0]) if (ini is not None and fim is not None) else None
            if ts is not None and ini <= ts <= fim:
                r.append((c[0], v))
            else:
                fora += 1
    return r, fora


def le_cpu_amostras(p):
    """[(epoch_ms, pid, %CPU de um núcleo)] das linhas legíveis `epoch_ms,pid,cpu_pct`; None se o arquivo não existe."""
    t = X.le_txt(p)
    if t is None:
        return None
    r = []
    for linha in t.splitlines():
        c = [x.strip() for x in linha.split(",")]
        if len(c) < 3:
            continue
        try:
            ts, pid, v = float(c[0]), int(c[1]), float(c[2])
        except ValueError:
            continue
        if math.isfinite(ts) and math.isfinite(v) and v >= 0:
            r.append((ts, pid, v))
    return r


def resumo_serie(v):
    if not v:
        return {"n": 0, "media": None, "max": None, "mediana": None}
    return {"n": len(v), "media": sum(v) / len(v), "max": max(v), "mediana": statistics.median(v)}


def analisa_p4(saida, k, prefill_med_ms, meta=None):
    """P4 da célula: só descrição da célula INTEIRA (carga, criação do motor, prefill, decode), sem atribuir nada ao prefill. Amostras ausentes = sem resolução."""
    meta = meta or {}
    ini, fim = (meta.get(x) if isinstance(meta.get(x), (int, float)) and not isinstance(meta.get(x), bool) else None for x in ("inicio_coleta_epoch", "fim_epoch"))
    g, fora = le_gpu_util(os.path.join(saida, "p4", k + ".gpu.csv"), ini, fim) or (None, 0)
    c = le_cpu_amostras(os.path.join(saida, "p4", k + ".cpu.csv"))
    gpu = resumo_serie([x[1] for x in g or []])
    cpu = resumo_serie([x[2] for x in c or []])
    motivos = []
    if gpu["n"] == 0:
        motivos.append("sem amostras de utilização da GPU dentro do intervalo de coleta da célula (arquivo %s; %d fora do intervalo)" % ("ausente" if g is None else "vazio ou ilegível", fora))
    if cpu["n"] == 0:
        motivos.append("sem amostras de %%CPU do processo (arquivo %s)" % ("ausente" if c is None else "vazio ou ilegível"))
    if motivos:
        res = "sem resolução: " + "; ".join(motivos)
    elif prefill_med_ms is not None and prefill_med_ms >= P4_PREFILL_DICA_MS:
        res = "dica: prefill >= ~3 s; a leitura cobre a célula inteira e não separa o prefill"
    else:
        res = "sem resolução: prefill < ~3 s (a leitura cobre a célula inteira, 100 ms é a cadência de consulta, não a resolução do contador)"
    return {"gpu_util_pct": gpu, "cpu_pct_processo": cpu, "gpu_amostras_fora_do_intervalo": fora, "gpu_intervalo_epoch": [ini, fim], "resolucao": res, "cobre": "célula inteira (carga, criação do motor, prefill e decode), sem atribuir causa ao prefill"}


# ---------------------------------------------------------------------------------------------------------------------------------- células
def valida_l10(d, m, mot):
    """L10 (série padrão): linha `slots` com a fração padrão impressa em CADA processo; linha de comando sem --fracao-slots nem --camadas-cpu (regra do critério: validade)."""
    esp = linha_slots_padrao(m)
    cmdl = X.le_txt(os.path.join(d, "cmdline.txt")) or ""
    linhas = [x for x in cmdl.splitlines() if x.strip()]
    if not linhas:
        mot.append("sem cmdline.txt (linha de comando do Lastro não gravada)")
    elif any("--fracao-slots" in x or "--camadas-cpu" in x for x in linhas):
        mot.append("linha de comando do L10 com --fracao-slots ou --camadas-cpu: a série não é a padrão")
    procs = sorted(x for x in os.listdir(d) if x.startswith("p") and x[1:].isdigit())
    if not procs:
        mot.append("sem pasta de processo (pK)")
    for p in procs:
        ls = X.le_linhas_slots(X.le_txt(os.path.join(d, p, "stderr.txt")))
        if not ls:
            mot.append("%s: sem a linha `slots N de M` no stderr do binário (a fração padrão não está provada)" % p)
        elif any(x != esp for x in ls):
            mot.append("%s: linha slots diferente da esperada na fração padrão (slots %d de %d): impressa %s" % (p, esp[0], esp[1], "; ".join("slots %d de %d" % x for x in ls)))


def le_celula(saida, m, s, t, refs):
    """Lê UMA célula quente do Lastro; devolve o resumo (status OK|SEM PROVA|NÃO RODOU, motivos, medidas, ms/posição, razão P3, P4)."""
    k = chave(m, s, t)
    d = os.path.join(saida, "celulas", k)
    base = {"chave": k, "modelo": m, "serie": s, "texto": t, "estado": "quente", "n_posicoes": N_POS[t]}
    meta = X.le_json(os.path.join(d, "celula.json"))
    if not isinstance(meta, dict):
        base.update({"status": "NÃO RODOU", "motivos": ["a janela não rodou esta célula (corte de orçamento, smoke falho ou janela interrompida)"], "n_medidas": 0})
        return base
    meta["chave"] = k
    mot = []
    if meta.get("lado") != "lastro" or meta.get("modelo") != m or meta.get("texto") != t or meta.get("estado") != "quente":
        mot.append("celula.json não é de %s (lado/modelo/texto/estado)" % k)
    if meta.get("serie") != SERIE_META[s]:
        mot.append("celula.json com serie %r, esperado %r" % (meta.get("serie"), SERIE_META[s]))
    if s == "L10" and (meta.get("fracao_slots") not in (None, "") or meta.get("camadas_cpu") not in (None, "")):
        mot.append("celula.json do L10 com fracao_slots/camadas_cpu")
    if meta.get("n_medidas") not in (3, 5):
        mot.append("n_medidas previstas %r: só 5 (nominal) ou 3 (corte de orçamento)" % (meta.get("n_medidas"),))
    X.D1, X.D2 = (s == "L100"), (s == "LC")
    try:
        dados, mot2 = X.le_celula_lastro(d, dict(meta, lado="lastro", modelo=m, texto=t, estado="quente", serie=SERIE_META[s]), refs)
    finally:
        X.D1 = X.D2 = False
    mot += mot2
    if s == "L10":
        valida_l10(d, m, mot)
    r = X.resume_celula(dict(meta, lado="lastro", modelo=m, texto=t, estado="quente", serie=SERIE_META[s]), dados, mot)
    meds = r.pop("_meds", [])
    for x in meds:
        if x["ids_n"] != N_POS[t]:
            r["motivos"] = sorted(set(r["motivos"] + ["%s/%s: prompt com %d posições, esperado %d" % (x["processo"], x["rotulo"], x["ids_n"], N_POS[t])]))
            r["status"] = "SEM PROVA"
    if refs.n_esperado(m, t) != N_POS[t]:
        r["motivos"] = sorted(set(r["motivos"] + ["cmp-ids-prompt.json com %s posições para %s/%s, esperado %d" % (refs.n_esperado(m, t), m, t, N_POS[t])]))
        r["status"] = "SEM PROVA"
    pre = (r.get("prefill") or {}).get("mediana")
    dec = (r.get("decode") or {}).get("mediana")
    r["n_posicoes"] = N_POS[t]
    r["prefill_ms_mediana"] = pre
    r["ms_por_posicao"] = (pre / N_POS[t]) if pre is not None else None
    r["decode_ms_passo_mediana"] = dec
    r["razao_p3"] = (r["ms_por_posicao"] / dec) if (r["ms_por_posicao"] is not None and dec) else None
    r["p4"] = analisa_p4(saida, k, pre, meta)
    r["sequencia"] = compara_com_referencia(refs, m, t, meds)
    return r


def status_serie(cels):
    st = [c["status"] for c in cels]
    if all(x == "OK" for x in st):
        return "OK"
    if all(x == "NÃO RODOU" for x in st):
        return "NÃO RODOU"
    return "SEM PROVA" if any(x == "SEM PROVA" for x in st) else "NÃO RODOU"


def previsoes(m, s, cels, bin_atual=None):
    """P1, P2 e P3 de uma série (só com as três células OK); 'dentro da previsão' ou 'fora da previsão', sempre com o dado ao lado."""
    st = status_serie(cels)
    if st != "OK":
        motivo = {"SEM PROVA": "SEM PROVA: série sem as três células válidas", "NÃO RODOU": "NÃO RODOU: série incompleta"}[st]
        return {x: {"resultado": motivo} for x in ("P1", "P2", "P3")}
    v = [c["ms_por_posicao"] for c in cels]
    media = sum(v) / len(v)
    desv = [abs(x / media - 1) for x in v]
    p1 = {"resultado": "dentro da previsão" if max(desv) <= P1_TOL else "fora da previsão", "tolerancia": P1_TOL, "media_ms_por_posicao": media, "ms_por_posicao": dict(zip(TEXTOS, v)),
          "desvio_max": max(desv), "desvios": dict(zip(TEXTOS, desv)), "max_sobre_min": max(v) / min(v)}
    ref = REF_MS_POS[m][s]
    d2 = {t: abs(c["ms_por_posicao"] / ref[t] - 1) for t, c in zip(TEXTOS, cels)}
    p2 = {"resultado": "dentro da previsão" if max(d2.values()) <= P2_TOL else "fora da previsão", "tolerancia": P2_TOL, "referencia_ms_por_posicao": dict(ref),
          "ms_por_posicao": dict(zip(TEXTOS, v)), "desvios": d2, "desvio_max": max(d2.values()),
          "binario_referencia": BIN_REF[s], "binario_atual": bin_atual, "aviso": AVISO_P2}
    lo, hi = P3_FAIXA[s]
    rz = [c["razao_p3"] for c in cels]
    if any(x is None for x in rz):
        p3 = {"resultado": "SEM PROVA: passo de decode ausente", "razoes": dict(zip(TEXTOS, rz))}
    else:
        p3 = {"resultado": "dentro da previsão" if all(lo <= x <= hi for x in rz) else "fora da previsão", "faixa": [lo, hi], "tolerancia": P3_TOL, "razoes": dict(zip(TEXTOS, rz)),
              "mediana_da_serie": statistics.median(rz), "decode_ms_passo": dict(zip(TEXTOS, [c["decode_ms_passo_mediana"] for c in cels]))}
    # marca de sequência ao lado de P3 (Emenda 1): vem das células; tempos e P3 seguem mantidos
    sq = [c["sequencia"] for c in cels]
    if any(not x["referencia_disponivel"] for x in sq):
        marca = "referência 5c ausente em %s: igualdade de sequência NÃO verificada" % ", ".join(c["texto"] for c, x in zip(cels, sq) if not x["referencia_disponivel"])
    elif any(x["sequencias_diferentes"] for x in sq):
        marca = "sequências diferentes da referência 5c em %s (tempos e P3 mantidos)" % ", ".join(c["texto"] for c, x in zip(cels, sq) if x["sequencias_diferentes"])
    else:
        marca = "mesma sequência da referência 5c nos três textos"
    p3["sequencia"] = marca
    p3["sequencias_diferentes"] = any(bool(x["sequencias_diferentes"]) for x in sq)
    return {"P1": p1, "P2": p2, "P3": p3}


def smoke_serie(saida, s):
    """Veredito do smoke da série (L100 e LC): None = sem exigência (L10); (ok, motivo)."""
    if s == "L10":
        return None
    v = X.le_chave_valor(os.path.join(saida, "smoke-%s" % s, "veredito.txt"))
    if not v:
        return (False, "smoke ausente (smoke-%s/veredito.txt): sem a prova da série, nenhuma previsão é publicada" % s)
    if v.get("veredito") != "OK":
        return (False, "smoke %s FALHOU: %s" % (s, v.get("motivo", "sem motivo gravado")))
    return (True, "")


def processa(saida, ids_arq, refs_dir, textos_arq):
    refs = X.Refs(ids_arq, refs_dir, textos_arq)
    integ = X.integridade(saida)
    orc = X.le_json(os.path.join(saida, "orcamento.json"))
    orc = orc if isinstance(orc, dict) else {}
    decl = orc.get("modelos")
    if not (isinstance(decl, list) and len(decl) == 1 and decl[0] in MODELOS):
        raise ValueError("orcamento.json sem um modelo único declarado (\"modelos\": [\"1b\"] ou [\"3b\"]): a janela do perfil é de UM modelo")
    m = decl[0]
    integ += integridade_fixos(saida, m)
    bin_atual = X.le_shas(os.path.join(saida, "integridade-antes.txt")).get("binario-lastro") or None
    amb = X.le_chave_valor(os.path.join(saida, "ambiente.txt"))
    obs = []
    mem = amb.get("memoria_mts")
    if mem is None:
        obs.append("memória do host (MT/s) não conferida pelo roteiro: declarar à parte")
    else:
        try:
            if int(float(mem)) != X.MEM_NOMINAL:
                integ.append("memória do host em %s MT/s, fora de %d MT/s" % (mem, X.MEM_NOMINAL))
        except ValueError:
            obs.append("memoria_mts ilegível: %s" % mem)
    celulas, series = {}, {}
    smokes = {}
    for s in SERIES:
        cels = [le_celula(saida, m, s, t, refs) for t in TEXTOS]
        sm = smoke_serie(saida, s)
        smokes[s] = None if sm is None else {"ok": sm[0], "motivo": sm[1]}
        # smoke da série ausente ou falho: as células que existirem não valem (as ausentes seguem NÃO RODOU)
        if sm is not None and not sm[0]:
            for c in cels:
                if c["status"] == "OK":
                    c["status"] = "SEM PROVA"
                    c["motivos"] = sorted(set(c["motivos"] + [sm[1]]))
        if integ:
            for c in cels:
                if c["status"] == "OK":
                    c["status"] = "SEM PROVA"
                    c["motivos"] = sorted(set(c["motivos"] + integ))
        for c in cels:
            celulas[c["chave"]] = c
        series[s] = {"rotulo": ROTULO[s], "status": status_serie(cels), "previsoes": previsoes(m, s, cels, bin_atual)}
    ordem = (X.le_txt(os.path.join(saida, "ordem.txt")) or "").splitlines()
    res = {
        "tipo": "perfil-resultado", "criterio": "criterio-perfil-prefill-20261109.md (versão 3; herda o critério base e as Emendas 1 dos degraus 1 e 2 para a validade)", "descritivo": True,
        "modelo_da_janela": m,
        "aviso": "Descritivo: sem meta e sem PASSA. P1, P2 e P3 são previsões pré-registradas, só conferidas ('dentro da previsão' / 'fora da previsão'); nenhuma prova causa. "
                 "P4 descreve a célula inteira (inclui carga e criação do motor) e não atribui nada ao prefill.",
        "binario_da_janela": bin_atual, "integridade_problemas": integ, "observacoes_ambiente": obs, "smokes": smokes, "orcamento": orc, "ordem": ordem, "celulas": celulas, "series": series,
    }
    X.limpa_nao_finitos(res)
    return res


# ---------------------------------------------------------------------------------------------------------------------------------- relatório
def f(x, n=2):
    return "-" if x is None else ("%.*f" % (n, x))


def pc(x):
    return "-" if x is None else ("%.1f%%" % (100 * x))


def relatorio(res):
    m = res["modelo_da_janela"]
    o = ["# Perfil do prefill do Lastro, janela do modelo %s (criterio-perfil-prefill-20261109.md, versão 3, Parte B)" % m.upper(), "",
         res["aviso"], ""]
    if res["integridade_problemas"]:
        o += ["**Integridade QUEBRADA: tudo SEM PROVA.**", ""] + ["- %s" % x for x in res["integridade_problemas"]] + [""]
    if res["observacoes_ambiente"]:
        o += ["Observações de ambiente: " + "; ".join(res["observacoes_ambiente"]), ""]
    o += ["## Tabela série x texto (mediana das medidas quentes; ms/posição = prefill_ms / posições)", "",
          "| série | texto | posições | prefill_ms (mediana; min a máx) | ms/posição | decode ms/passo | razão ms/posição ÷ decode | medidas | status |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in SERIES:
        for t in TEXTOS:
            c = res["celulas"][chave(m, s, t)]
            p = c.get("prefill") or {}
            faixa = "%s (%s a %s)" % (f(p.get("mediana"), 1), f(p.get("min"), 1), f(p.get("max"), 1)) if p else "-"
            o.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s |" % (s, t, N_POS[t], faixa, f(c.get("ms_por_posicao"), 3), f(c.get("decode_ms_passo_mediana"), 2), f(c.get("razao_p3"), 3),
                                                                      "%s/%s" % (c.get("n_medidas"), c.get("n_medidas_previstas", "-")) if c["status"] != "NÃO RODOU" else "-", c["status"]))
    o += ["", "## Previsões pré-registradas (conferência; nenhuma é meta)", ""]
    for s in SERIES:
        sr = res["series"][s]
        o += ["### %s (%s)" % (sr["rotulo"], sr["status"]), ""]
        pv = sr["previsoes"]
        p1, p2, p3 = pv["P1"], pv["P2"], pv["P3"]
        o.append("- P1 (ms/posição varia no máximo ±15%% entre 89, 170 e 364): %s" % p1["resultado"])
        if "desvios" in p1:
            o.append("  - média %s ms/posição; desvios %s; maior/menor %s" % (f(p1["media_ms_por_posicao"], 3), ", ".join("%s %s" % (t, pc(p1["desvios"][t])) for t in TEXTOS), f(p1["max_sobre_min"], 3)))
        o.append("- P2 (cada célula dentro de ±10%% do valor da janela anterior): %s" % p2["resultado"])
        if "desvios" in p2:
            o.append("  - " + "; ".join("%s %s contra ref. %s (desvio %s)" % (t, f(p2["ms_por_posicao"][t], 3), f(p2["referencia_ms_por_posicao"][t], 3), pc(p2["desvios"][t])) for t in TEXTOS))
        o.append("  - binário da janela de referência %s; binário de agora %s" % (BIN_REF[s], res.get("binario_da_janela") or "(ilegível)"))
        o.append("  - %s" % AVISO_P2)
        o.append("- P3 (razão ms/posição ÷ decode na faixa): %s" % p3["resultado"])
        if "faixa" in p3:
            o.append("  - faixa %s a %s; razões %s; mediana da série %s" % (f(p3["faixa"][0], 3), f(p3["faixa"][1], 3), ", ".join("%s %s" % (t, f(p3["razoes"][t], 3)) for t in TEXTOS), f(p3["mediana_da_serie"], 3)))
            o.append("  - sequência: %s" % p3["sequencia"])
        o.append("")
    o += ["## P4 (descritivo): utilização da GPU e %CPU do processo, célula inteira", "",
          "A leitura cobre carga, criação do motor, prefill e decode; os carimbos de início e fim do prefill não saem do binário, então nada é atribuído ao prefill. "
          "A média de GPU só usa amostras dentro de [inicio_coleta_epoch, fim_epoch] da célula (o portão ocioso fica de fora; o nº de descartadas vai entre parênteses). "
          "100 ms é a cadência de consulta, não a resolução do contador. Com prefill >= ~3 s (L10 e LC) é só uma dica; nas demais, sem resolução.", "",
          "| série | texto | GPU n (fora do intervalo) | GPU média % | GPU máx % | CPU n | CPU média % (1 núcleo = 100) | CPU máx % | leitura |", "|---|---|---|---|---|---|---|---|---|"]
    for s in SERIES:
        for t in TEXTOS:
            c = res["celulas"][chave(m, s, t)]
            p4 = c.get("p4")
            if not p4 or c["status"] == "NÃO RODOU":
                o.append("| %s | %s | - | - | - | - | - | - | NÃO RODOU |" % (s, t))
                continue
            g, cp = p4["gpu_util_pct"], p4["cpu_pct_processo"]
            o.append("| %s | %s | %d (%d) | %s | %s | %d | %s | %s | %s |" % (s, t, g["n"], p4.get("gpu_amostras_fora_do_intervalo", 0), f(g["media"], 1), f(g["max"], 1), cp["n"], f(cp["media"], 1), f(cp["max"], 1), p4["resolucao"]))
    o += ["", "## Sequência gerada contra a referência 5c (só reportada, sem ajuste nem repetição)", ""]
    for c in res["celulas"].values():
        sq = c.get("sequencia")
        if not sq or c["status"] == "NÃO RODOU":
            continue
        o.append("- %s: %s" % (c["chave"], sq["marca"]))
        for dv in sq["divergencias"]:
            o.append("  - %s/%s: primeira divergência no passo %s, id medido %s, id da referência %s, gap da referência %s" % (dv["processo"], dv["rotulo"], dv["passo"], dv["id_medido"], dv["id_referencia"], f(dv["gap_referencia"], 4)))
    nao = [c for c in res["celulas"].values() if c["status"] != "OK"]
    if nao:
        o += ["", "## Células sem resultado", ""]
        for c in nao:
            o.append("- %s: %s: %s" % (c["chave"], c["status"], "; ".join(c["motivos"])))
    sm = [(s, v) for s, v in res["smokes"].items() if v and not v["ok"]]
    if sm:
        o += ["", "## Smoke", ""] + ["- %s: %s" % (s, v["motivo"]) for s, v in sm]
    orc = res.get("orcamento") or {}
    o += ["", "## Orçamento", "", "decorrido %s s de %s s; cortes: %s; não rodou: %s" % (orc.get("decorrido_s", "?"), orc.get("limite_s", "?"), "; ".join(orc.get("cortes") or []) or "nenhum",
                                                                                      "; ".join(orc.get("nao_rodou") or []) or "nada")]
    o += ["", "## Conclusão", "",
          "O que a Parte B acrescentou e a recomendação do próximo passo (instrumento dentro do motor, só com autorização de código do Breno) são redigidos por quem lê estas tabelas; "
          "este relatório não conclui causa e não declara PASSA."]
    return "\n".join(o) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="extrator do perfil do prefill do Lastro")
    ap.add_argument("--saida")
    ap.add_argument("--ids", default=os.path.join(AQUI, "cmp-ids-prompt.json"))
    ap.add_argument("--refs", default=os.path.join(AQUI, "5c-referencia"))
    ap.add_argument("--textos", default=os.path.join(AQUI, "5c-textos", "textos.json"))
    a = ap.parse_args(argv)
    if not a.saida or not os.path.isdir(a.saida):
        print("pasta de saída inexistente: %s" % a.saida, file=sys.stderr)
        return 2
    try:
        res = processa(a.saida, a.ids, a.refs, a.textos)
    except ValueError as ex:
        print("perfil-extrai: %s" % ex, file=sys.stderr)
        return 2
    with open(os.path.join(a.saida, "perfil-resultado.json"), "w", encoding="utf-8") as fo:
        json.dump(res, fo, ensure_ascii=False, indent=1, allow_nan=False)
    with open(os.path.join(a.saida, "perfil-resultado.md"), "w", encoding="utf-8") as fo:
        fo.write(relatorio(res))
    n = {"OK": 0, "SEM PROVA": 0, "NÃO RODOU": 0}
    for c in res["celulas"].values():
        n[c["status"]] += 1
    print("perfil-extrai: células OK %d, SEM PROVA %d, NÃO RODOU %d; gravados perfil-resultado.json e perfil-resultado.md" % (n["OK"], n["SEM PROVA"], n["NÃO RODOU"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
