#!/usr/bin/env python3
"""Extrator do instrumento de prefill do Lastro (criterio-instrumento-prefill-20261110.md), descritivo, sem veredito geral.

    python3 -I instrumento-extrai.py --saida DIR_DA_JANELA [--ids cmp-ids-prompt.json] [--refs 5c-referencia] [--textos 5c-textos/textos.json]

Só biblioteca padrão. NÃO roda nada, NÃO usa GPU: lê o que roda-instrumento-janela.sh gravou e grava DIR/instrumento-resultado.json e DIR/instrumento-resultado.md.
Aplica: o par de controle (ordem D L L D D L, magnitude |desvio| de prefill_ms: <= 3% comprovada, <= 10% com ressalva, > 10% sem prova), as classes H1 a H6 do plano
(seção 1.3, mesmos limiares e mesmos >= <= < > do módulo perfil_bloco.rs), a conservação já avaliada pelo binário (campo `valido` por pedido), a cobertura do host (<= 3% do prefill_ms)
e do serviço (mediana de S_fora/I_g <= 10%) e os recortes de H3. RECUSA (código 2, sem classificar nada) dados sem a prova de potência de 600.00 W nos dois extremos ou sem a linha
`perfil-bloco: ligado|desligado` em nenhum processo; processo avulso sem a linha invalida só a sua célula.

Layout lido (escrito por roda-instrumento-janela.sh e, por baixo, roda-cmp-lastro.sh):
  DIR/integridade-antes.txt, integridade-depois.txt, integridade-fixos.txt, ambiente.txt, orcamento.json ("modelos": um só), ordem.txt
  DIR/smoke-L100/veredito.txt e DIR/smoke-LC/veredito.txt (julgados por cmp-extrai.py --smoke [--degrau2])
  DIR/celulas/ctrl-<modelo>-<série>-medio-<NN>-<D|L>/   processos do par de controle (série L10 ou L100; NN = 01..06; um processo por pasta, p1 com --repete 3)
  DIR/celulas/lastro-<modelo>-<série>-<texto>-quente/   células principais (série L10, L100 ou LC; processo L, 1 aquecimento + N medidas)
  em cada pasta de processo: p1/stdout.txt (com a linha `perfil-bloco: ...`), p1/saida/perfil-bloco.json (só o braço L) e o resto que roda-cmp-lastro.sh grava.
Interpretações onde o plano deixa margem estão no critério (seção 'Interpretações') e nos comentários abaixo; nada aqui inventa limiar.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))


def carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(AQUI, arq))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


P = carrega("perfil_extrai", "f2e-perfil-extrai.py")     # leitura de célula, valida_l10, referência 5c (reaproveita cmp-extrai.py por baixo)
X = P.X

MODELOS = ("1b", "3b")
TEXTOS = ("curto", "medio", "longo")
SERIES = ("L10", "L100", "LC")
SERIES_CONTROLE = ("L10", "L100")
N_POS = P.N_POS
BLOCO = 8
ORDEM_CONTROLE = "DLLDDL"
EPS = 1e-12
LIM_COMPROVADA = 0.03
LIM_RESSALVA = 0.10
LIM_COB_HOST = 0.03
LIM_COB_SERVICO = 0.10
LIM_EXTRAPOLACAO = 0.10
POT_TOK = "600.00_W"
LINHA_RE = re.compile(r"^perfil-bloco: (ligado|desligado)$")
CTRL_RE = re.compile(r"^ctrl-(1b|3b)-(L10|L100)-medio-(\d\d)-([DL])$")
CEL_RE = re.compile(r"^lastro-(1b|3b)-(L10|L100|LC)-(curto|medio|longo)-quente$")
S_CAMPOS = ("s_espera_ms", "s_monta_ms", "s_compute_ms", "s_answer_ms", "s_livro_ms", "s_end_espera_ms", "s_end_trab_ms")
S_RECORTE = tuple(c.replace("_ms", "_recorte_ms") for c in S_CAMPOS)

# FIXO que o manifesto integridade-fixos.txt TEM de trazer (mesmas chaves do FIXO de roda-instrumento-janela.sh, mais o tokenizer do modelo da janela e a potência)
FIXO_CHAVES = ("criterio-instrumento", "plano-instrumento", "binario-instr", "script-instrumento-extrai", "script-perfil-extrai", "script-cmp-extrai", "script-lastro", "script-monitor",
               "script-janela", "script-limite", "criterio-cmp", "criterio-degrau1", "criterio-degrau2", "cmp-ids-prompt", "texto-curto", "texto-medio", "texto-longo", "textos-json",
               "ref5c-1b", "ref5c-3b")


class Recusa(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------------------------- funções puras
def plano_blocos(n, mp=BLOCO):
    """Espelho de motor/leitura.rs::plano_blocos."""
    mp = max(mp, 1)
    v = [mp] * (n // mp)
    if n % mp:
        v.append(n % mp)
    return v


def fin(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def mediana(v):
    v = [x for x in v if fin(x) is not None]
    return statistics.median(v) if v else None


def razao(a, b):
    return (a / b) if (fin(a) is not None and fin(b) is not None and b > 0) else None


def classifica_controle(procs, orcamento_ok=True, incompleto_motivo=None):
    """Classifica o par de controle de uma (modelo, série). `procs`: lista na ordem em que rodou, cada um com braco ('D'|'L'), prefill_ms_mediana, passo_ms_mediana (ou None), imprimiu_ligado,
    produziu_perfil, valido. Devolve dict com `classe` em {'comprovada', 'com ressalva', 'SEM PROVA', 'controle incompleto'}, `motivo`, `desvio` (com sinal), `pior_caso`, `desvio_passo`.
    A magnitude manda (o sinal não aprova nada); 3% e 10% são inclusivos (com EPS contra erro de ponto flutuante). Controle incompleto = nenhuma classificação de H e custo SEM PROVA. `incompleto_motivo` (ordem efetiva do ordem.txt ausente, truncada ou incompatível) força o incompleto."""
    def inc(m):
        return {"classe": "controle incompleto", "motivo": m, "desvio": None, "pior_caso": None, "desvio_passo": None}
    if not orcamento_ok:
        return inc("orçamento insuficiente")
    if incompleto_motivo:
        return inc(incompleto_motivo)
    if len(procs) != len(ORDEM_CONTROLE):
        return inc("%d processos de controle em vez de 6 (mínimo de 3 válidos por braço)" % len(procs))
    if "".join(p["braco"] for p in procs) != ORDEM_CONTROLE:
        return inc("ordem diferente de D L L D D L")
    if any((not p["valido"]) or fin(p["prefill_ms_mediana"]) is None or p["prefill_ms_mediana"] <= 0 for p in procs):
        return inc("processo de controle abortado ou inválido")
    if any(p["braco"] == "D" and (p["produziu_perfil"] or p["imprimiu_ligado"]) for p in procs):
        return inc("braço D produziu perfil-bloco.json ou imprimiu perfil-bloco: ligado")
    if any(p["braco"] == "L" and not (p["produziu_perfil"] and p["imprimiu_ligado"]) for p in procs):
        return inc("braço L sem perfil-bloco.json ou sem a linha ligado")
    d = [p["prefill_ms_mediana"] for p in procs if p["braco"] == "D"]
    l = [p["prefill_ms_mediana"] for p in procs if p["braco"] == "L"]
    md, ml = statistics.median(d), statistics.median(l)
    desvio = (ml - md) / md
    pior = (max(l) - min(d)) / min(d)
    mag = abs(desvio)
    classe = "comprovada" if mag <= LIM_COMPROVADA + EPS else ("com ressalva" if mag <= LIM_RESSALVA + EPS else "SEM PROVA")
    pd_ = [p.get("passo_ms_mediana") for p in procs if p["braco"] == "D"]
    pl_ = [p.get("passo_ms_mediana") for p in procs if p["braco"] == "L"]
    dpasso = None
    if all(fin(x) is not None and x > 0 for x in pd_ + pl_):
        dpasso = (statistics.median(pl_) - statistics.median(pd_)) / statistics.median(pd_)
    return {"classe": classe, "motivo": "", "desvio": desvio, "pior_caso": pior, "desvio_passo": dpasso, "prefill_D_mediana_ms": md, "prefill_L_mediana_ms": ml}


def controle_permite(c):
    return c in ("comprovada", "com ressalva")


def faixas_h6(r):
    return (0.5 + r / 16.0, (r / 8.0) * 1.25)


def h6_descritivo(r):
    a, b = faixas_h6(r)
    return a <= b


def _composta(a, b, dentro, fora):
    a, b = fin(a), fin(b)
    if (a is not None and a < fora[0]) or (b is not None and b < fora[1]):
        return "fora"
    if a is not None and b is not None:
        return "dentro" if (a >= dentro[0] and b >= dentro[1]) else "sem leitura"
    return "sem dado"


def _ge(x, d, f):
    x = fin(x)
    if x is None:
        return "sem dado"
    return "dentro" if x >= d else ("fora" if x < f else "sem leitura")


def _le(x, d, f):
    x = fin(x)
    if x is None:
        return "sem dado"
    return "dentro" if x <= d else ("fora" if x > f else "sem leitura")


def classifica_h(h, serie, m, controle):
    """Classe de H1..H6 numa série (L10|L100|LC), dadas as medianas `m` (dict: f1_b, f2_b, f3_b, f3_passo, r3, s_compute_b, r5, r6, resto) e a classe do controle.
    Resultado em {'dentro', 'fora', 'sem leitura', 'sem dado', 'SEM PROVA', 'descritivo'}. Mesmas regras de perfil_bloco.rs::classifica_h:
    controle que não permite = SEM PROVA antes de olhar número; série sem previsão (LC; L10 em H1, H2, H4) = 'sem dado'; faixa intermediária = 'sem leitura'."""
    if not controle_permite(controle):
        return "SEM PROVA"
    if h == "H1":
        if serie != "L100":
            return "sem dado"
        dentro = fin(m.get("f2_b")) is not None and m["f2_b"] >= 0.70
        fora = fin(m.get("f3_b")) is not None and m["f3_b"] >= 0.50
        if dentro and fora:
            return "sem dado"
        if dentro:
            return "dentro"
        if fora:
            return "fora"
        return "sem leitura" if fin(m.get("f2_b")) is not None and fin(m.get("f3_b")) is not None else "sem dado"
    if h == "H2":
        return _composta(m.get("f3_b"), m.get("f3_passo"), (0.50, 3.0), (0.25, 1.5)) if serie == "L100" else "sem dado"
    if h == "H3":
        if serie == "L10":
            return _composta(m.get("r3"), m.get("s_compute_b"), (0.50, 0.20), (0.30, 0.10))
        return _le(m.get("r3"), 0.10, 0.25) if serie == "L100" else "sem dado"
    if h == "H4":
        return _ge(m.get("f1_b"), 0.05, 0.02) if serie == "L100" else "sem dado"
    if h == "H5":
        if serie == "L10":
            return _ge(m.get("r5"), 0.10, 0.03)
        return _le(m.get("r5"), 0.02, 0.05) if serie == "L100" else "sem dado"
    if h == "H6":
        r = m.get("resto", 0)
        if not (1 <= r <= BLOCO - 1) or h6_descritivo(r):
            return "descritivo"
        r6 = fin(m.get("r6"))
        if r6 is None:
            return "sem dado"
        a, b = faixas_h6(r)
        return "dentro" if r6 >= a else ("fora" if r6 <= b else "sem leitura")
    raise ValueError("hipótese desconhecida: %r" % (h,))


# ---------------------------------------------------------------------------------------------------------------------------------- leitura
def le_linha_perfil(d):
    """Estado impresso pelo binário em CADA processo (p*/stdout.txt): 'ligado', 'desligado' ou None (ausente, repetida com valores diferentes ou ilegível). Devolve (estado, [motivos])."""
    mot, estados = [], []
    procs = sorted(x for x in os.listdir(d) if x.startswith("p") and x[1:].isdigit())
    if not procs:
        return None, ["sem pasta de processo (pK)"]
    for p in procs:
        txt = X.le_txt(os.path.join(d, p, "stdout.txt")) or ""
        achados = [LINHA_RE.match(x.strip()).group(1) for x in txt.splitlines() if LINHA_RE.match(x.strip())]
        if len(achados) != 1:
            mot.append("%s: %d linha(s) `perfil-bloco: ligado|desligado` no stdout do binário (esperada 1)" % (p, len(achados)))
        else:
            estados.append(achados[0])
    if mot or len(set(estados)) != 1:
        return None, mot or ["processos da célula com estados diferentes de perfil-bloco"]
    return estados[0], []


def acha_perfis(d):
    r = []
    for raiz, _, arqs in os.walk(d):
        if "perfil-bloco.json" in arqs:
            r.append(os.path.join(raiz, "perfil-bloco.json"))
    return sorted(r)


def valida_pedidos(arq, m, t, n_medidas, mot):
    """Lê perfil-bloco.json de UM processo L: devolve a lista dos pedidos MEDIDOS (sem o aquecimento). Falhas entram em `mot` (a célula vira SEM PROVA)."""
    j = X.le_json(arq)
    if not isinstance(j, dict) or j.get("tipo") != "perfil-bloco" or j.get("versao") != 2 or not isinstance(j.get("pedidos"), list):
        mot.append("perfil-bloco.json ilegível ou fora do esquema (tipo perfil-bloco, versao 2)")
        return []
    peds = j["pedidos"]
    if len(peds) != n_medidas + 1:
        mot.append("perfil-bloco.json com %d entradas, esperadas %d (1 aquecimento e %d medidas)" % (len(peds), n_medidas + 1, n_medidas))
    medidos = []
    for i, p in enumerate(peds):
        if not isinstance(p, dict):
            mot.append("entrada %d do perfil-bloco.json inválida" % i)
            continue
        if p.get("modelo") != m:
            mot.append("entrada %d do perfil-bloco.json é do modelo %r, não %r" % (i, p.get("modelo"), m))
        if p.get("indice") != i:
            mot.append("entrada %d com índice %r" % (i, p.get("indice")))
        if bool(p.get("aquecimento")) != (i == 0):
            mot.append("entrada %d com aquecimento=%r (só a 0 é aquecimento)" % (i, p.get("aquecimento")))
        if i == 0:
            continue
        if p.get("n") != N_POS[t]:
            mot.append("pedido %d com n=%r posições, esperado %d" % (i, p.get("n"), N_POS[t]))
        if p.get("plano_m") != plano_blocos(N_POS[t]) or [b.get("m") for b in p.get("blocos", [])] != plano_blocos(N_POS[t]):
            mot.append("pedido %d com plano de blocos diferente de plano_blocos(%d, 8)" % (i, N_POS[t]))
        if p.get("valido") is not True:
            mot.append("pedido %d inválido pelo próprio binário: %s" % (i, "; ".join(str(x) for x in (p.get("problemas") or ["sem motivo"]))[:300]))
        if any(b.get("conserva") is not True for b in p.get("blocos", [])):
            mot.append("pedido %d com bloco que não conserva F0+F1+F2+F3 = B" % i)
        medidos.append(p)
    return medidos


def compara5c(d, m, t, refs):
    """Ids de CADA pedido (aquecimento inclusive; os tempos do aquecimento seguem fora das medianas) contra a referência 5c e contra o prompt esperado. A referência tem de estar disponível:
    ausente = `referencia_disponivel` False (bloqueio, nunca igualdade)."""
    ref = refs.ref(m, t)
    disp = bool(ref) and isinstance(ref.get("gerados"), list) and bool(ref["gerados"])
    esperado = refs.ids_prompt(m, t)
    div = []
    for p in sorted(x for x in os.listdir(d) if x.startswith("p") and x[1:].isdigit()):
        mj = X.le_json(os.path.join(d, p, "saida", "medidas-%s.json" % m))
        for e in (mj or {}).get("medidas", []) if isinstance(mj, dict) else []:
            if not isinstance(e, dict) or e.get("falha"):
                continue
            rot = {"processo": p, "rotulo": e.get("rotulo"), "aquecimento": bool(e.get("descartada"))}
            if disp:
                g = list(e.get("gerados") or [])
                i = X.primeira_divergencia(g, list(ref["gerados"]))
                if i is not None:
                    div.append(dict(rot, passo=i, tipo="gerados"))
            if esperado is not None and list(e.get("ids") or []) != list(esperado):
                div.append(dict(rot, passo=None, tipo="ids do prompt"))
    return {"referencia_disponivel": disp, "divergencias": div, "n_divergentes_no_aquecimento": sum(1 for x in div if x["aquecimento"])}


def confere_cardinalidade(d, m, t, n_prev, refs, mot):
    """No bruto de CADA processo (D ou L, com ou sem perfil): exatamente 1 aquecimento (o primeiro) seguido de n_prev medidas, ids dos n_prev+1 pedidos conferidos por compara5c (aquecimento incluído); as medianas usam só as n_prev."""
    ps = sorted(x for x in os.listdir(d) if x.startswith("p") and x[1:].isdigit())
    if not ps:
        mot.append("sem pasta de processo (p1) com o bruto de medidas")
    for p in ps:
        mj = X.le_json(os.path.join(d, p, "saida", "medidas-%s.json" % m))
        ent = [e for e in ((mj or {}).get("medidas") or []) if isinstance(e, dict)] if isinstance(mj, dict) else []
        aq = [bool(e.get("descartada")) for e in ent]
        if len(ent) != n_prev + 1 or not aq or aq[0] is not True or any(aq[1:]):
            mot.append("%s: bruto com %d entradas (%d aquecimentos), esperados 1 aquecimento seguido de %d medidas" % (p, len(ent), sum(aq), n_prev))
        if any(e.get("falha") for e in ent):
            mot.append("%s: pedido com falha no bruto" % p)


def le_dir_celula(saida, nome, m, s, t, n_prev, refs, bin_ref=None):
    """Lê UMA pasta de processo/célula (controle ou principal). Devolve dict com `status` (OK|SEM PROVA|NÃO RODOU, o conjunto), `status_temporal` (só rc, portão, medidas, ids, linha, D sem perfil: é o que vale
    para o PROCESSO de controle) e `status_perfil` (só o conteúdo de perfil-bloco.json: marca a célula, nunca o controle), motivos de cada um, meds, pedidos medidos, linha, tem_perfil e a comparação com a 5c."""
    d = os.path.join(saida, "celulas", nome)
    meta = X.le_json(os.path.join(d, "celula.json"))
    base = {"nome": nome, "modelo": m, "serie": s, "texto": t, "n_previstas": n_prev}
    if not isinstance(meta, dict):
        base.update({"status": "NÃO RODOU", "status_temporal": "NÃO RODOU", "status_perfil": "NÃO RODOU", "motivos": ["a janela não rodou esta pasta"], "motivos_temporais": [], "motivos_perfil": [],
                     "meds": [], "pedidos": [], "estado_linha": None, "tem_perfil": False, "seq": {"referencia_disponivel": False, "divergencias": [], "n_divergentes_no_aquecimento": 0}})
        return base
    mot, mot_p = [], []
    if meta.get("lado") != "lastro" or meta.get("modelo") != m or meta.get("texto") != t or meta.get("estado") != "quente":
        mot.append("celula.json não é de %s (lado/modelo/texto/estado)" % nome)
    if meta.get("serie") != P.SERIE_META[s]:
        mot.append("celula.json com serie %r, esperado %r" % (meta.get("serie"), P.SERIE_META[s]))
    if meta.get("n_medidas") != n_prev:
        mot.append("n_medidas previstas %r, esperado %d" % (meta.get("n_medidas"), n_prev))
    X.D1, X.D2 = (s == "L100"), (s == "LC")
    try:
        dados, mot2 = X.le_celula_lastro(d, dict(meta, lado="lastro", modelo=m, texto=t, estado="quente", serie=P.SERIE_META[s]), refs)
    finally:
        X.D1 = X.D2 = False
    mot += mot2
    if s == "L10":
        P.valida_l10(d, m, mot)
    meds = dados["meds"]
    if len(meds) != n_prev:
        mot.append("%d medidas aceitas de %d previstas" % (len(meds), n_prev))
    for x in meds:
        if x["ids_n"] != N_POS[t]:
            mot.append("%s/%s: prompt com %d posições, esperado %d" % (x["processo"], x["rotulo"], x["ids_n"], N_POS[t]))
    confere_cardinalidade(d, m, t, n_prev, refs, mot)
    estado, mot3 = le_linha_perfil(d)
    mot += mot3
    perfis = acha_perfis(d)
    base.update({"estado_linha": estado, "tem_perfil": bool(perfis), "meds": meds, "pedidos": [], "seq": compara5c(d, m, t, refs)})
    if estado == "ligado":
        if len(perfis) != 1:
            mot.append("braço ligado com %d perfil-bloco.json (esperado 1)" % len(perfis))
        else:
            base["pedidos"] = valida_pedidos(perfis[0], m, t, n_prev, mot_p)
    elif estado == "desligado" and perfis:
        mot.append("processo com perfil-bloco: desligado produziu perfil-bloco.json")
    base["status_temporal"] = "SEM PROVA" if mot else "OK"
    base["status_perfil"] = "SEM PROVA" if mot_p else "OK"
    base["status"] = "SEM PROVA" if (mot or mot_p) else "OK"
    base["motivos_temporais"] = sorted(set(mot))
    base["motivos_perfil"] = sorted(set(mot_p))
    base["motivos"] = sorted(set(mot + mot_p))
    return base


def mediana_passo(meds):
    return mediana([x.get("mediana_passos_descritiva") for x in meds])


def processo_controle(c, braco):
    meds = c.get("meds") or []
    return {"braco": braco, "prefill_ms_mediana": mediana([x["prefill_ms"] for x in meds]), "passo_ms_mediana": mediana_passo(meds), "imprimiu_ligado": c.get("estado_linha") == "ligado",
            "produziu_perfil": bool(c.get("tem_perfil")), "valido": c.get("status_temporal") == "OK" and c.get("estado_linha") == ("ligado" if braco == "L" else "desligado")}


# ---------------------------------------------------------------------------------------------------------------------------------- estatísticas por (série, texto)
def faixa(v):
    v = [x for x in v if fin(x) is not None]
    return {"n": len(v), "mediana": statistics.median(v), "min": min(v), "max": max(v)} if v else {"n": 0, "mediana": None, "min": None, "max": None}


def estatisticas(pedidos, resto):
    """Medianas de H1..H6 e o resumo descritivo, sobre blocos cheios (m = 8) dos pedidos MEDIDOS (já sem aquecimento). Razões = mediana do numerador / mediana de B sobre o MESMO conjunto de blocos.
    F0 exclui os blocos f0_pendente (reportados à parte). Pedido com serviço aberto sai das estatísticas de serviço (contado). Serviço só entra se fechado e coerente."""
    f = {k: [] for k in ("f0", "f1", "f2", "f3", "b")}
    f0_pend = []
    sv = {k: [] for k in S_CAMPOS + S_RECORTE + ("i_ms", "s_fora_ms", "s_fora_frac", "inter_bloco_ms")}
    b_serv = []
    r6s, cob_host, abertos, invalidos_serv = [], [], 0, 0
    for p in pedidos:
        cob_host.append(p.get("cobertura_host_frac"))
        if fin(p.get("r6")) is not None:
            r6s.append(p["r6"])
        aberto = bool(p.get("servico_aberto"))
        abertos += 1 if aberto else 0
        servs = p.get("servico") or []
        for i, b in enumerate(p.get("blocos", [])):
            if b.get("m") != BLOCO:
                continue
            for k in ("f1", "f2", "f3", "b"):
                f[k].append(b.get(k + "_ms"))
            if b.get("f0_pendente"):
                f0_pend.append(b.get("f0_ms"))
            else:
                f["f0"].append(b.get("f0_ms"))
            s = servs[i] if i < len(servs) else None
            if aberto or not isinstance(s, dict) or s.get("aberto") or s.get("coerente") is not True:
                invalidos_serv += 0 if aberto else 1
                continue
            b_serv.append(b.get("b_ms"))
            for k in S_CAMPOS + S_RECORTE + ("i_ms", "s_fora_ms", "s_fora_frac", "inter_bloco_ms"):
                sv[k].append(s.get(k))
    mb = mediana(f["b"])
    mbs = mediana(b_serv)
    m = {"f1_b": razao(mediana(f["f1"]), mb), "f2_b": razao(mediana(f["f2"]), mb), "f3_b": razao(mediana(f["f3"]), mb), "resto": resto,
         "r6": mediana(r6s)}
    soma = lambda a, b: [x + y for x, y in zip(sv[a], sv[b]) if fin(x) is not None and fin(y) is not None]
    m["r3"] = razao(mediana(soma("s_compute_ms", "s_livro_ms")), mbs)
    m["s_compute_b"] = razao(mediana(sv["s_compute_ms"]), mbs)
    m["r5"] = razao(mediana(soma("s_livro_ms", "s_end_trab_ms")), mbs)
    m["r3_recorte"] = razao(mediana(soma("s_compute_recorte_ms", "s_livro_recorte_ms")), mbs)
    desc = {
        "n_pedidos": len(pedidos), "n_blocos_cheios": len(f["b"]), "n_blocos_com_servico": len(b_serv), "pedidos_servico_aberto": abertos, "blocos_servico_incoerente_ou_ausente": invalidos_serv,
        "fases_host_ms": {k: faixa(f[k]) for k in ("f0", "f1", "f2", "f3", "b")},
        "fases_host_pct_de_b": {k: razao(mediana(f[k]), mb) for k in ("f0", "f1", "f2", "f3")},
        "f0_bloco0_reutilizado_ms": faixa(f0_pend),
        "servico_ms": {k: faixa(sv[k]) for k in S_CAMPOS + ("i_ms", "s_fora_ms", "inter_bloco_ms")},
        "servico_recorte_ms": {k: faixa(sv[k]) for k in S_RECORTE},
        "servico_razao_de_b": {k: razao(mediana(sv[k]), mbs) for k in S_CAMPOS},
        "s_fora_frac": faixa(sv["s_fora_frac"]),
        "cobertura_host_frac": faixa(cob_host),
        "r6": faixa(r6s), "resto": resto,
    }
    return m, desc, sv["s_fora_frac"], mb


# ---------------------------------------------------------------------------------------------------------------------------------- integridade e processamento
def integridade_fixos(saida, m):
    """Manifesto FIXO obrigatório: uma linha `chave esperado observado OK` por chave congelada (mais tokenizer do modelo), esperado == observado == o de integridade-antes.txt.
    A linha `potencia` é tratada à parte (recusa)."""
    txt = X.le_txt(os.path.join(saida, "integridade-fixos.txt"))
    if txt is None:
        return ["integridade FIXA: manifesto integridade-fixos.txt ausente ou ilegível"]
    linhas = [x for x in txt.splitlines() if x.strip()]
    if not linhas:
        return ["integridade FIXA: manifesto vazio"]
    mot, vistas = [], {}
    antes = X.le_shas(os.path.join(saida, "integridade-antes.txt"))
    for x in linhas:
        c = x.split()
        if len(c) != 4:
            mot.append("integridade FIXA: linha truncada ou ilegível: %r" % x.strip()[:80])
            continue
        k, esp, obs, est = c
        if k in vistas:
            mot.append("integridade FIXA: chave %s repetida" % k)
        vistas[k] = c
        if k == "potencia":
            continue
        if est != "OK":
            mot.append("integridade FIXA: %s com estado %s (esperado OK)" % (k, est))
        elif not (re.fullmatch(r"[0-9a-f]{64}", esp) and esp == obs):
            mot.append("integridade FIXA: %s com sha esperado/observado diferentes ou inválidos" % k)
        elif antes.get(k) != obs:
            mot.append("integridade FIXA: %s com sha diferente do de integridade-antes.txt ou ausente dele" % k)
    falta = [k for k in FIXO_CHAVES + ("tokenizer-%s" % m,) if k not in vistas]
    if falta:
        mot.append("integridade FIXA: chave(s) faltando no manifesto: %s" % ", ".join(falta))
    return sorted(set(mot))


def prova_potencia(saida):
    """Lista de motivos de RECUSA: a prova `potencia 600.00_W 600.00_W OK` (única) no manifesto, coerente com potencia_inicio e potencia_fim de ambiente.txt."""
    mot = []
    fixos = X.le_txt(os.path.join(saida, "integridade-fixos.txt"))
    amb = X.le_chave_valor(os.path.join(saida, "ambiente.txt"))
    if fixos is None:
        return ["sem prova de potência: integridade-fixos.txt ausente"]
    pot = [l.split() for l in fixos.splitlines() if l.split()[:1] == ["potencia"]]
    if len(pot) != 1:
        mot.append("sem prova de potência: linha 'potencia' %s no manifesto" % ("ausente" if not pot else "duplicada"))
    elif pot[0] != ["potencia", POT_TOK, POT_TOK, "OK"]:
        mot.append("sem prova de potência: o manifesto diz %s (esperado a prova de 600.00 W no início e no fim)" % " ".join(pot[0]))
    if amb.get("potencia_inicio") != POT_TOK or amb.get("potencia_fim") != POT_TOK:
        mot.append("sem prova de potência: ambiente.txt com início %r e fim %r (esperado %s nos dois)" % (amb.get("potencia_inicio"), amb.get("potencia_fim"), POT_TOK))
    return mot


def smoke_ok(saida, s):
    if s == "L10":
        return True, ""
    v = X.le_chave_valor(os.path.join(saida, "smoke-%s" % s, "veredito.txt"))
    if not v:
        return False, "smoke ausente (smoke-%s/veredito.txt)" % s
    if v.get("veredito") != "OK":
        return False, "smoke %s FALHOU: %s" % (s, v.get("motivo", "sem motivo gravado"))
    return True, ""


ORDEM_RE = r"^(\d\d) (ctrl-%s-%s-medio-(\d\d)-([DL])) n=(\d+) braco=([DL]) inicio=\S+ fim=\S+ rc=(\d+)$"


def confere_ordem(saida, m, s, procs_ctrl):
    """Ordem EFETIVA do controle da série, lida do ordem.txt (registro cronológico do roteiro) e cruzada com as 6 pastas: mesmas chaves na mesma ordem (D L L D D L), braço do registro igual ao do nome,
    n = 3 e rc = 0. Registro ausente, truncado ou incompatível devolve o motivo (controle incompleto); None = confere."""
    txt = X.le_txt(os.path.join(saida, "ordem.txt"))
    if txt is None:
        return "ordem.txt ausente: a ordem efetiva do controle de %s não foi provada" % s
    rx = re.compile(ORDEM_RE % (re.escape(m), re.escape(s)))
    linhas = []
    for x in txt.splitlines():
        mt = rx.match(x.strip())
        if mt:
            linhas.append({"chave": mt.group(2), "nn": int(mt.group(3)), "braco_nome": mt.group(4), "n": int(mt.group(5)), "braco": mt.group(6), "rc": int(mt.group(7))})
        elif x.strip().split(" ")[1:2] and x.split(" ")[1].startswith("ctrl-%s-%s-" % (m, s)):
            return "linha de ordem.txt ilegível para o controle de %s: %r" % (s, x.strip()[:80])
    if len(linhas) != len(ORDEM_CONTROLE):
        return "ordem.txt com %d linhas do controle de %s (esperadas 6): registro truncado ou ausente" % (len(linhas), s)
    esperado = ["ctrl-%s-%s-medio-%02d-%s" % (m, s, i + 1, b) for i, b in enumerate(ORDEM_CONTROLE)]
    if [x["chave"] for x in linhas] != esperado:
        return "ordem efetiva de ordem.txt (%s) diferente de D L L D D L com os nomes esperados" % " ".join(x["chave"][-4:] for x in linhas)
    if any(x["braco"] != x["braco_nome"] for x in linhas):
        return "braço registrado em ordem.txt diferente do braço do nome da pasta"
    if any(x["rc"] != 0 or x["n"] != 3 for x in linhas):
        return "ordem.txt com processo de controle de rc != 0 ou n != 3"
    if sorted(x[2] for x in procs_ctrl) != sorted(esperado):
        return "pastas do controle diferentes das 6 esperadas"
    return None


def sha_arq(p):
    if not os.path.isfile(p):
        return None
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def confere_referencias(saida, m, ids_arq, refs_dir, textos_arq):
    """Os arquivos que o extrator ABRIU (ids, referência 5c do modelo, textos) têm de ter o sha registrado no manifesto da janela (observado). Devolve motivos de bloqueio."""
    obs = {}
    for x in (X.le_txt(os.path.join(saida, "integridade-fixos.txt")) or "").splitlines():
        c = x.split()
        if len(c) == 4:
            obs[c[0]] = c[2]
    mot = []
    for chave, arq in (("cmp-ids-prompt", ids_arq), ("ref5c-%s" % m, os.path.join(refs_dir, "ref5c-%s.json" % m)), ("textos-json", textos_arq)):
        h = sha_arq(arq)
        if h is None:
            mot.append("referência 5c: arquivo %s ausente (%s)" % (chave, os.path.basename(arq)))
        elif obs.get(chave) != h:
            mot.append("referência 5c: o arquivo aberto para %s tem sha %s…, o manifesto registra %s…" % (chave, h[:12], (obs.get(chave) or "nada")[:12]))
    return mot


def processa(saida, ids_arq, refs_dir, textos_arq):
    refs = X.Refs(ids_arq, refs_dir, textos_arq)
    orc = X.le_json(os.path.join(saida, "orcamento.json"))
    orc = orc if isinstance(orc, dict) else {}
    decl = orc.get("modelos")
    if not (isinstance(decl, list) and len(decl) == 1 and decl[0] in MODELOS):
        raise Recusa("orcamento.json sem um modelo único declarado (\"modelos\": [\"1b\"] ou [\"3b\"])")
    m = decl[0]
    rec = prova_potencia(saida)
    if rec:
        raise Recusa("; ".join(rec))
    nomes = sorted(os.listdir(os.path.join(saida, "celulas"))) if os.path.isdir(os.path.join(saida, "celulas")) else []
    algum = False
    for n_ in nomes:
        if (CTRL_RE.match(n_) or CEL_RE.match(n_)) and os.path.isdir(os.path.join(saida, "celulas", n_)):
            if le_linha_perfil(os.path.join(saida, "celulas", n_))[0] is not None:
                algum = True
    if not algum:
        raise Recusa("sem a linha `perfil-bloco: ligado|desligado` em nenhum processo: não é saída do binário instrumentado")
    integ = sorted(set(X.integridade(saida) + integridade_fixos(saida, m)))
    ref_prob = confere_referencias(saida, m, ids_arq, refs_dir, textos_arq)
    bin_atual = X.le_shas(os.path.join(saida, "integridade-antes.txt")).get("binario-instr") or None
    nao_rodou = [str(x) for x in (orc.get("nao_rodou") or [])]

    ctrl, cels = {}, {}
    for n_ in nomes:
        mt = CTRL_RE.match(n_)
        if mt and mt.group(1) == m:
            ctrl.setdefault(mt.group(2), []).append((int(mt.group(3)), mt.group(4), n_))
            continue
        mt = CEL_RE.match(n_)
        if mt and mt.group(1) == m:
            cels.setdefault((mt.group(2), mt.group(3)), []).append(n_)
    leituras = {}

    def le(nome, s, t, n):
        if nome not in leituras:
            leituras[nome] = le_dir_celula(saida, nome, m, s, t, n, refs)
        return leituras[nome]

    smokes = {s: smoke_ok(saida, s) for s in SERIES}
    resultado_series = {}
    for s in SERIES:
        sm_ok, sm_mot = smokes[s]
        ctl, procs_ctrl = None, []
        if s in SERIES_CONTROLE:
            for (nn, braco, nome) in sorted(ctrl.get(s, [])):
                procs_ctrl.append((nn, braco, nome, le(nome, s, "medio", 3)))
            nums = [x[0] for x in procs_ctrl]
            ordem_ok = nums == list(range(1, len(nums) + 1))
            orc_ok = not any(x.startswith("ctrl-%s-%s-" % (m, s)) for x in nao_rodou)   # o roteiro grava em nao_rodou a pasta do processo de controle que não coube/rodou
            mot_ordem = confere_ordem(saida, m, s, procs_ctrl) if procs_ctrl else None
            if not ordem_ok and not mot_ordem:
                mot_ordem = "numeração das pastas do controle não é 01..%02d" % len(nums)
            ctl = classifica_controle([processo_controle(c, b) for (_, b, _, c) in procs_ctrl], orc_ok, mot_ordem)
            ctl["ordem_efetiva"] = mot_ordem or "confere com ordem.txt (D L L D D L, rc 0)"
            ctl["processos"] = [{"nn": nn, "braco": b, "pasta": nome, "status": c["status"], "status_temporal": c["status_temporal"], "status_perfil": c["status_perfil"],
                                 "estado_linha": c.get("estado_linha"), "tem_perfil": c.get("tem_perfil"), "prefill_ms_mediana": mediana([x["prefill_ms"] for x in c.get("meds") or []]),
                                 "passo_ms_mediana": mediana_passo(c.get("meds") or []), "motivos_temporais": c["motivos_temporais"], "motivos_perfil": c["motivos_perfil"]}
                                for (nn, b, nome, c) in procs_ctrl]
        else:
            ctl = {"classe": "sem controle", "motivo": "a série LC não tem par de controle: descritiva, sempre com ressalva", "desvio": None, "pior_caso": None, "desvio_passo": None}
        passo_d = mediana([processo_controle(c, b)["passo_ms_mediana"] for (_, b, _, c) in procs_ctrl if b == "D"]) if procs_ctrl else None
        por_texto, b_por_texto, fora_frac_serie = {}, {}, []
        for t in TEXTOS:
            pedidos, motivos_t, estado_t, usadas, fontes, seqs = [], [], "NÃO RODOU", [], [], []
            if s in SERIES_CONTROLE and t == "medio":
                fontes += [(nome, c) for (_, b, nome, c) in procs_ctrl if b == "L"]
            for nome in cels.get((s, t), []):
                mj = X.le_json(os.path.join(saida, "celulas", nome, "celula.json")) or {}
                fontes.append((nome, le(nome, s, t, mj.get("n_medidas") if mj.get("n_medidas") in (3, 5) else 5)))
            if fontes:
                ok = [c for _, c in fontes if c["status"] == "OK"]
                estado_t = "OK" if ok and len(ok) == len(fontes) else ("SEM PROVA" if any(c["status"] != "NÃO RODOU" for _, c in fontes) else "NÃO RODOU")
                for nome, c in fontes:
                    motivos_t += ["%s: %s" % (nome, x) for x in c["motivos"]] if c["status"] != "OK" else []
                    seqs.append(c.get("seq"))
                    if c["status"] == "OK":
                        pedidos += c["pedidos"]
                        usadas.append(nome)
            m_h, desc, fora_frac, mb = estatisticas(pedidos, N_POS[t] % BLOCO) if pedidos else ({}, {}, [], None)
            fora_frac_serie += fora_frac
            b_por_texto[t] = mb
            por_texto[t] = {"status": estado_t, "motivos": sorted(set(motivos_t)), "celulas": usadas, "medidas": m_h, "descricao": desc, "n_posicoes": N_POS[t],
                            "ref_ausente": any((not sq["referencia_disponivel"]) for sq in seqs if sq) if fontes else False}
        # extrapolação do controle (checagem INTERNA, B mediano observado em L; não estima L-D): por texto, |B_t - B_médio| / B_médio <= 10%; exige B médio válido; só o texto que falha fica SEM PROVA
        bm = b_por_texto.get("medio")
        extra = {"formula": "|B_texto - B_medio| / B_medio <= 10%", "b_medio_ms": bm, "textos": {}}
        for t in ("curto", "longo"):
            bt = b_por_texto.get(t)
            if fin(bm) is None or bm <= 0 or fin(bt) is None:
                extra["textos"][t] = {"b_ms": bt, "desvio": None, "ok": False, "motivo": "sem B médio válido" if (fin(bm) is None or bm <= 0) else "sem B do texto"}
            else:
                dv = abs(bt - bm) / bm
                extra["textos"][t] = {"b_ms": bt, "desvio": dv, "ok": dv <= LIM_EXTRAPOLACAO + EPS, "motivo": ""}
        extra["textos"]["medio"] = {"b_ms": bm, "desvio": 0.0 if fin(bm) is not None else None, "ok": fin(bm) is not None, "motivo": ""}
        cobs = mediana(fora_frac_serie)
        cob_serv_ressalva = cobs is not None and cobs > LIM_COB_SERVICO
        # ids contra a 5c em QUALQUER processo da série, D ou L, aquecimento inclusive
        div_serie, ref_ausente_serie = [], []
        for (_, b, nome, c) in procs_ctrl:
            div_serie += [dict(x, pasta=nome, braco=b) for x in c["seq"]["divergencias"]]
            if c["status"] != "NÃO RODOU" and not c["seq"]["referencia_disponivel"]:
                ref_ausente_serie.append((nome, "medio"))
        for t in TEXTOS:
            for nome in cels.get((s, t), []):
                c = leituras.get(nome) or {}
                div_serie += [dict(x, pasta=nome, texto=t) for x in (c.get("seq") or {}).get("divergencias", [])]
                if c and c.get("status") != "NÃO RODOU" and not (c.get("seq") or {}).get("referencia_disponivel"):
                    ref_ausente_serie.append((nome, t))
        ids_l_d = None
        if procs_ctrl:
            gl = {tuple(x["gerados"]) for (_, b, _, c) in procs_ctrl if b == "L" for x in c.get("meds") or []}
            gd = {tuple(x["gerados"]) for (_, b, _, c) in procs_ctrl if b == "D" for x in c.get("meds") or []}
            ids_l_d = bool(gl) and bool(gd) and gl == gd and len(gl) == 1
        # bloqueios da série (todas, LC inclusive: integridade, smoke e referências; a LC só escapa da comparação de ids contra a 5c e da falta de controle; medidas brutas preservadas): valem para a classe EFETIVA do controle e do custo, e para as H
        bloq = []
        if integ:
            bloq.append("integridade quebrada")
        if not sm_ok:
            bloq.append(sm_mot)
        if ref_prob:
            bloq.append("; ".join(ref_prob))
        if ref_ausente_serie:
            bloq.append("referência 5c indisponível para texto com dados submetidos: " + ", ".join(sorted({"%s (%s)" % (t_, n_) for (n_, t_) in ref_ausente_serie})))
        if s != "LC" and div_serie:   # exceção da LC: só a comparação de ids com a 5c (reportada, sem bloquear) e a falta de controle
            bloq.append("ids diferentes da referência 5c (instrumento inválido nesta série; inclui o aquecimento)")
        if s in SERIES_CONTROLE and bloq and controle_permite(ctl["classe"]):
            ctl = dict(ctl, classe_calculada=ctl["classe"], classe="SEM PROVA", motivo="bloqueio da série: " + "; ".join(bloq) + " (as medidas calculadas ficam só como descrição)")
        hs = {}
        for t in TEXTOS:
            pt = por_texto[t]
            pt["f3_passo"] = razao(pt["descricao"]["fases_host_ms"]["f3"]["mediana"], passo_d) if pt["descricao"] else None   # denominador de H2: passo de decode do braço D
            r = {}
            for h in ("H1", "H2", "H3", "H4", "H5", "H6"):
                if pt["status"] != "OK" or not pt["medidas"]:
                    r[h] = "sem dado" if pt["status"] == "NÃO RODOU" else "SEM PROVA"
                    continue
                if bloq:
                    r[h] = "SEM PROVA"
                    continue
                if s == "LC":
                    r[h] = "descritivo"
                    continue
                if pt["ref_ausente"] or not extra["textos"][t]["ok"]:
                    r[h] = "SEM PROVA"
                    continue
                mm = dict(pt["medidas"])
                mm["f3_passo"] = pt["f3_passo"]
                r[h] = classifica_h(h, s, mm, ctl["classe"])
            hs[t] = r
        ressalvas = []
        if ctl["classe"] == "com ressalva":
            ressalvas.append("controle com ressalva (3% < |desvio| <= 10%): frações relativas ao tempo medido com instrumento, nunca ao do braço D")
        if s == "LC":
            ressalvas.append("LC sem controle: descritiva, sempre com ressalva (divergência contra a 5c, se houver, só reportada)")
        if cob_serv_ressalva:
            ressalvas.append("com ressalva de cobertura do serviço (mediana de S_fora/I_g %.1f%% > 10%%)" % (100 * cobs))
        for t in TEXTOS:
            cb = (por_texto[t]["descricao"].get("cobertura_host_frac") or {}).get("mediana") if por_texto[t]["descricao"] else None
            if cb is not None and cb > LIM_COB_HOST:
                ressalvas.append("com ressalva de cobertura do host em %s (mediana %.1f%% > 3%%)" % (t, 100 * cb))
        resultado_series[s] = {"controle": ctl, "passo_d_ms": passo_d, "textos": por_texto, "extrapolacao": extra, "cobertura_servico_mediana": cobs, "bloqueios": bloq, "hipoteses": hs,
                               "ressalvas": ressalvas, "ids_L_igual_D": ids_l_d, "divergencias_5c_serie": div_serie}
    return {"tipo": "instrumento-resultado", "criterio": "criterio-instrumento-prefill-20261110.md", "descritivo": True, "modelo_da_janela": m, "binario_da_janela": bin_atual,
            "aviso": "Descritivo: sem veredito geral e sem PASSA. H1 a H6 são previsões pré-registradas, só conferidas ('dentro', 'fora', 'sem leitura', 'sem dado', 'SEM PROVA'); nada prova causa; "
                     "o tempo com instrumento não é o tempo de produção; H7 não é avaliada nesta unidade.",
            "integridade_problemas": integ, "referencia_problemas": ref_prob, "smokes": {s: {"ok": smokes[s][0], "motivo": smokes[s][1]} for s in SERIES}, "orcamento": orc, "series": resultado_series}


# ---------------------------------------------------------------------------------------------------------------------------------- relatório
def f(x, n=2):
    return "-" if x is None else ("%.*f" % (n, x))


def pc(x):
    return "-" if x is None else ("%.1f%%" % (100 * x))


def relatorio(res):
    m = res["modelo_da_janela"]
    o = ["# Instrumento de prefill do Lastro, janela do modelo %s (criterio-instrumento-prefill-20261110.md)" % m.upper(), "", res["aviso"], ""]
    if res.get("referencia_problemas"):
        o += ["**Referência 5c sem prova (bloqueia todas as séries):**", ""] + ["- %s" % x for x in res["referencia_problemas"]] + [""]
    if res["integridade_problemas"]:
        o += ["**Integridade QUEBRADA: todas as séries SEM PROVA, inclusive a LC (medidas brutas preservadas só como descrição, com o motivo).**", ""] + ["- %s" % x for x in res["integridade_problemas"]] + [""]
    for s, sr in res["series"].items():
        c = sr["controle"]
        o += ["## Série %s" % s, ""]
        o.append("Controle do custo do instrumento: **%s**%s" % (c["classe"], (" (%s)" % c["motivo"]) if c.get("motivo") else ""))
        if c.get("classe_calculada"):
            o.append("- classe calculada antes do bloqueio: %s (medidas abaixo preservadas só como descrição)" % c["classe_calculada"])
        if c.get("ordem_efetiva"):
            o.append("- ordem efetiva do controle (ordem.txt): %s" % c["ordem_efetiva"])
        if c.get("desvio") is not None:
            o.append("- desvio de prefill_ms (L menos D, sobre D, com sinal): %s; pior caso (maior L menos menor D, sobre menor D): %s; desvio do passo de decode: %s; mediana D %s ms, mediana L %s ms"
                     % (pc(c["desvio"]), pc(c["pior_caso"]), pc(c.get("desvio_passo")), f(c.get("prefill_D_mediana_ms"), 2), f(c.get("prefill_L_mediana_ms"), 2)))
            blocos = len(plano_blocos(N_POS["medio"]))
            o.append("- desvio em ms por bloco: %s ms (diferença das medianas / %d blocos do texto médio)" % (f((c["prefill_L_mediana_ms"] - c["prefill_D_mediana_ms"]) / blocos, 3), blocos))
        for p in c.get("processos", []):
            mots = p["motivos_temporais"] + ["perfil: " + x for x in p["motivos_perfil"]]
            o.append("  - %02d %s %s: temporal %s, perfil %s, linha %s, perfil-bloco.json %s, prefill mediano %s ms, passo %s ms%s" % (
                p["nn"], p["braco"], p["pasta"], p["status_temporal"], p["status_perfil"], p["estado_linha"], "sim" if p["tem_perfil"] else "não", f(p["prefill_ms_mediana"], 2), f(p["passo_ms_mediana"], 2),
                ("; " + "; ".join(mots)) if mots else ""))
        if sr["extrapolacao"]["textos"]:
            e = sr["extrapolacao"]
            o.append("- extrapolação do médio (checagem interna: B mediano observado em L, não estimativa de L menos D; %s; B médio %s ms): %s" % (
                e["formula"], f(e["b_medio_ms"], 2), "; ".join("%s B %s ms, desvio %s, %s" % (t, f(v["b_ms"], 2), pc(v["desvio"]), "sustenta" if v["ok"] else "BLOQUEIA" + ((" (%s)" % v["motivo"]) if v["motivo"] else "")) for t, v in e["textos"].items() if t != "medio")))
        o.append("- ids do braço L iguais aos do D (processos do controle): %s" % ({True: "sim", False: "não", None: "sem dado"}[sr["ids_L_igual_D"]]))
        if sr["divergencias_5c_serie"]:
            o.append("- DIVERGÊNCIA contra a referência 5c em %d pedido(s), aquecimento incluído (%s)" % (len(sr["divergencias_5c_serie"]), "; ".join("%s %s passo %s" % (x.get("pasta") or x.get("texto"), x.get("rotulo"), x.get("passo")) for x in sr["divergencias_5c_serie"][:6])))
        if sr["bloqueios"]:
            o.append("- BLOQUEIOS (H em SEM PROVA): " + "; ".join(sr["bloqueios"]))
        for r in sr["ressalvas"]:
            o.append("- ressalva: " + r)
        o += ["", "| texto | status | pedidos | blocos cheios | F0 ms (mediana; faixa; % de B) | F1 | F2 | F3 | B ms | F0 do bloco 0 reutilizado (ms) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for t, pt in sr["textos"].items():
            d = pt["descricao"]
            if not d:
                o.append("| %s | %s | - | - | - | - | - | - | - | - |" % (t, pt["status"]))
                continue
            fh, pb = d["fases_host_ms"], d["fases_host_pct_de_b"]
            cel = lambda k: "%s (%s a %s; %s)" % (f(fh[k]["mediana"], 3), f(fh[k]["min"], 3), f(fh[k]["max"], 3), pc(pb[k]))
            o.append("| %s | %s | %d | %d | %s | %s | %s | %s | %s | %s |" % (t, pt["status"], d["n_pedidos"], d["n_blocos_cheios"], cel("f0"), cel("f1"), cel("f2"), cel("f3"), f(fh["b"]["mediana"], 3), f(d["f0_bloco0_reutilizado_ms"]["mediana"], 3)))
        o += ["", "Serviço (fora da partição do host, com sobreposição; ms por bloco e razão de B; interseção com [t0, t4] à parte):", "",
              "| texto | S_espera | S_monta | S_compute | S_answer | S_livro | S_end_espera | S_end_trab | S_fora | inter. [t0,t4] | R3 integral | R3 recortado | blocos com serviço | pedidos com serviço aberto |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t, pt in sr["textos"].items():
            d = pt["descricao"]
            if not d:
                continue
            sm_, rz = d["servico_ms"], d["servico_razao_de_b"]
            o.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %d | %d |" % (
                t, *["%s (%s)" % (f(sm_[k]["mediana"], 3), pc(rz[k])) for k in S_CAMPOS], f(sm_["s_fora_ms"]["mediana"], 3), f(sm_["inter_bloco_ms"]["mediana"], 3),
                pc(pt["medidas"].get("r3")), pc(pt["medidas"].get("r3_recorte")), d["n_blocos_com_servico"], d["pedidos_servico_aberto"]))
        o += ["", "R6 e resto por texto: " + "; ".join("%s: resto %d, R6 %s (faixas: custo fixo se >= %s, custo por posição se <= %s%s)" % (
            t, pt["n_posicoes"] % BLOCO, f(pt["medidas"].get("r6"), 3), f(faixas_h6(pt["n_posicoes"] % BLOCO)[0], 4), f(faixas_h6(pt["n_posicoes"] % BLOCO)[1], 4),
            "; faixas se tocam: texto descritivo" if h6_descritivo(pt["n_posicoes"] % BLOCO) else "") for t, pt in sr["textos"].items()), ""]
        o += ["| texto | H1 | H2 | H3 | H4 | H5 | H6 |", "|---|---|---|---|---|---|---|"]
        for t in TEXTOS:
            o.append("| %s | %s |" % (t, " | ".join(sr["hipoteses"][t][h] for h in ("H1", "H2", "H3", "H4", "H5", "H6"))))
        o += ["", "Medidas usadas nas classes (razões de medianas): " + "; ".join("%s: F1/B %s, F2/B %s, F3/B %s, F3/passo(D) %s, R3 %s, S_compute/B %s, R5 %s" % (
            t, pc(pt["medidas"].get("f1_b")), pc(pt["medidas"].get("f2_b")), pc(pt["medidas"].get("f3_b")),
            f(razao(pt["descricao"]["fases_host_ms"]["f3"]["mediana"], sr["passo_d_ms"]) if pt["descricao"] else None, 2), pc(pt["medidas"].get("r3")), pc(pt["medidas"].get("s_compute_b")), pc(pt["medidas"].get("r5")))
            for t, pt in sr["textos"].items() if pt["medidas"]), ""]
        for t, pt in sr["textos"].items():
            if pt["motivos"]:
                o.append("- %s: %s" % (t, "; ".join(pt["motivos"])))
        o.append("")
    orc = res.get("orcamento") or {}
    o += ["## Orçamento", "", "decorrido %s s de %s s; cortes: %s; não rodou: %s" % (orc.get("decorrido_s", "?"), orc.get("limite_s", "?"), "; ".join(orc.get("cortes") or []) or "nenhum", "; ".join(orc.get("nao_rodou") or []) or "nada"),
          "", "## Conclusão", "", "Este relatório não declara PASSA nem conclui causa; H7 e o nível GPU ficam para a segunda unidade."]
    return "\n".join(o) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="extrator do instrumento de prefill do Lastro")
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
    except Recusa as ex:
        print("instrumento-extrai: RECUSADO: %s" % ex, file=sys.stderr)
        with open(os.path.join(a.saida, "instrumento-resultado.json"), "w", encoding="utf-8") as fo:
            json.dump({"tipo": "instrumento-resultado", "recusado": True, "motivo": str(ex)}, fo, ensure_ascii=False, indent=1)
        with open(os.path.join(a.saida, "instrumento-resultado.md"), "w", encoding="utf-8") as fo:
            fo.write("# Instrumento de prefill: RECUSADO\n\n%s\n\nNenhuma classificação foi feita; os dados brutos ficam na pasta.\n" % ex)
        return 2
    X.limpa_nao_finitos(res)
    with open(os.path.join(a.saida, "instrumento-resultado.json"), "w", encoding="utf-8") as fo:
        json.dump(res, fo, ensure_ascii=False, indent=1, allow_nan=False)
    with open(os.path.join(a.saida, "instrumento-resultado.md"), "w", encoding="utf-8") as fo:
        fo.write(relatorio(res))
    cl = {s: res["series"][s]["controle"]["classe"] for s in SERIES}
    print("instrumento-extrai: controle %s; gravados instrumento-resultado.json e instrumento-resultado.md" % ", ".join("%s %s" % (s, c) for s, c in cl.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
