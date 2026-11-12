#!/usr/bin/env python3
"""Extrator da F2 do host por operação em L100 (criterio-f2-host-20261111.md), descritivo, sem veredito geral e sem frase causal.

    python3 -I f2-extrai.py --saida DIR_DA_JANELA [--ids cmp-ids-prompt.json] [--refs 5c-referencia] [--textos 5c-textos/textos.json]

Só biblioteca padrão. NÃO roda nada, NÃO usa GPU: lê o que roda-f2-janela.sh gravou e grava DIR/f2-host-resultado.json e DIR/f2-host-resultado.md (um modelo por janela).
Reaproveita, por baixo, instrumento-extrai.py (leitura de célula, 5c, integridade comum) e recalcula em Python as invariantes I1 a I8 a partir das operações brutas de perfil-f2.json
(nunca confia nas flags gravadas pelo binário).

Layout lido (escrito por roda-f2-janela.sh e, por baixo, roda-cmp-lastro.sh):
  DIR/integridade-antes.txt, integridade-depois.txt, integridade-fixos.txt, ambiente.txt, orcamento.json ("modelos": um só), ordem.txt, teto-excedido.txt
  DIR/celulas/smoke-<modelo>-L100-curto-B/          smoke: 1 processo no curto, braço B, 1 aquecimento e 1 medida
  DIR/celulas/ctrl-<modelo>-L100-medio-<NN>-<D|A|B>/  controle de 9 processos (ordem D A B, B D A, A B D; --repete 3)
  DIR/celulas/lastro-<modelo>-L100-<curto|longo>-quente/  células principais em B (1 aquecimento + 5 ou 3 medidas)
  em cada pasta de processo: p1/stdout.txt (linhas `perfil-bloco: ...` e `perfil-f2: ...`), p1/saida/perfil-bloco.json (A e B) e p1/saida/perfil-f2.json (só B).
Regras (critério, seções 4 a 8 e pontos 1 a 10 da seção 10): D = nada ligado, A = só perfil-bloco, B = perfil-bloco e perfil-f2; controle por |B contra A| (<= 3% comprovada, <= 10% com
ressalva, > 10% SEM PROVA), B contra D só como apoio; HA a HE calculadas POR PEDIDO e agregadas pela mediana entre os pedidos da célula; mínimo de 3 pedidos válidos em curto e longo e
9 no médio (senão SEM PROVA na classificação afetada); controle incompleto invalida todas as H do modelo; HF, HG e HI sem classe de dentro/fora.
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


def carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(AQUI, arq))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


I = carrega("instrumento_extrai", "f2e-instrumento-extrai.py")
X, P = I.X, I.P

MODELOS = ("1b", "3b")
TEXTOS = ("curto", "medio", "longo")
N_POS = I.N_POS
BLOCO = 8
ORDEM_CONTROLE = "DABBDAABD"
BLOCOS_POR_PEDIDO_MEDIO = len(I.plano_blocos(N_POS["medio"]))   # 22: blocos totais executados no texto médio (170 = 21 x 8 + 2)
EPS = 1e-12
TOL_NS = 1000                      # TOL_RELOGIO_NS
LIM_COBERTURA_BORDA = 0.03
MIN_PEDIDOS = {"curto": 3, "medio": 9, "longo": 3}
POT_TOK = "600.00_W"
LINHA_BLOCO_RE = re.compile(r"^perfil-bloco: (ligado|desligado)$")
LINHA_F2_RE = re.compile(r"^perfil-f2: (ligado|desligado)$")
CTRL_RE = re.compile(r"^ctrl-(1b|3b)-L100-medio-(\d\d)-([DAB])$")
CEL_RE = re.compile(r"^lastro-(1b|3b)-L100-(curto|longo)-quente$")
SMOKE_RE = re.compile(r"^smoke-(1b|3b)-L100-curto-B$")
FIXO_CHAVES = ("criterio-f2-host", "plano-f2-host", "binario-f2", "script-f2-extrai", "script-instrumento-extrai", "script-perfil-extrai", "script-cmp-extrai", "script-lastro", "script-monitor",
               "script-janela", "script-limite", "criterio-cmp", "criterio-degrau1", "criterio-degrau2", "cmp-ids-prompt", "texto-curto", "texto-medio", "texto-longo", "textos-json",
               "ref5c-1b", "ref5c-3b")
# códigos das operações do host (perfil_bloco.rs)
H_ENTRADA, H_BEGIN_STEP, H_UPLOAD_TABELA, H_WAIT_EVENT, H_POST_LAYERS, H_SEQ_FIM, H_SAIDA = range(7)
H_PRE_LAYER, H_ROUTE, H_SEQ, H_LAUNCH_HITS, H_WAIT_FLAG, H_ADD_Y = range(7, 13)
OPS_CAMADA = (H_PRE_LAYER, H_ROUTE, H_SEQ, H_LAUNCH_HITS, H_WAIT_FLAG, H_ADD_Y)
LAYERS_DO_MODELO = {"1b": 24, "3b": 32}
EXIGE_SEM_CAMADA = None
fin = I.fin
mediana = I.mediana


class Recusa(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------------------------- funções puras
def percentil_posto(v, p):
    v = sorted(x for x in v if fin(x) is not None)
    if not v:
        return None
    k = min(max(math.ceil(p * len(v)), 1), len(v))
    return v[k - 1]


def dist(v):
    v = [x for x in v if fin(x) is not None]
    if not v:
        return {"n": 0, "min": None, "mediana": None, "p90": None, "max": None}
    return {"n": len(v), "min": min(v), "mediana": statistics.median(v), "p90": percentil_posto(v, 0.9), "max": max(v)}


def razao(a, b):
    return (a / b) if (fin(a) is not None and fin(b) is not None and b > 0) else None


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


def classifica_ha(x):
    return _ge(x, 0.80, 0.50)


def classifica_hb(frac, maior):
    frac = fin(frac)
    if frac is None:
        return "sem dado"
    if frac < 0.25:
        return "fora"
    if maior is None:
        return "sem dado"
    return "dentro" if (maior and frac >= 0.40) else "sem leitura"


def classifica_hc(x):
    return _le(x, 0.05, 0.15)


def classifica_hd(x):
    return _le(x, 1.5, 3.0)


def classifica_he(x):
    return _le(x, 0.05, 0.20)


def classifica_hh(x):
    x = fin(x)
    if x is None:
        return "sem dado"
    return "dentro" if x >= 1.5 else "descritivo"


def faixa_hf(r):
    c = r / BLOCO
    return (c * 0.75, c * 1.25)


def classe_por_desvio(d):
    a = abs(d)
    return "comprovada" if a <= 0.03 + EPS else ("com ressalva" if a <= 0.10 + EPS else "SEM PROVA")


def controle_permite(c):
    return c in ("comprovada", "com ressalva")


def classifica_controle_f2(procs, orcamento_ok=True, incompleto_motivo=None, blocos_por_pedido=BLOCOS_POR_PEDIDO_MEDIO):
    """Controle de três braços (espelho de perfil_bloco.rs::classifica_controle_f2). `procs`: ordem de execução, cada um com braco D|A|B, prefill_ms_mediana, passo_ms_mediana,
    imprimiu_bloco, imprimiu_f2, produziu_bloco, produziu_f2, valido. Classe pela magnitude de |B contra A|; B contra D só apoio. Controle incompleto invalida todas as H do modelo."""
    def inc(m):
        return {"classe": "controle incompleto", "motivo": m, "principal": None, "apoio": None, "classe_apoio": None}
    if not orcamento_ok:
        return inc("orçamento insuficiente")
    if incompleto_motivo:
        return inc(incompleto_motivo)
    if len(procs) != len(ORDEM_CONTROLE):
        return inc("%d processos de controle em vez de 9 (mínimo de 3 válidos por braço)" % len(procs))
    if "".join(p["braco"] for p in procs) != ORDEM_CONTROLE:
        return inc("ordem diferente de D A B, B D A, A B D")
    if any((not p["valido"]) or fin(p["prefill_ms_mediana"]) is None or p["prefill_ms_mediana"] <= 0 or fin(p.get("passo_ms_mediana")) is None or p["passo_ms_mediana"] <= 0 for p in procs):
        return inc("processo de controle abortado ou inválido")
    for p in procs:
        b = p["braco"]
        ok = ((not p["imprimiu_bloco"] and not p["imprimiu_f2"] and not p["produziu_bloco"] and not p["produziu_f2"]) if b == "D" else
              (p["imprimiu_bloco"] and not p["imprimiu_f2"] and p["produziu_bloco"] and not p["produziu_f2"]) if b == "A" else
              (p["imprimiu_bloco"] and p["imprimiu_f2"] and p["produziu_bloco"] and p["produziu_f2"]))
        if not ok:
            return inc("braço %s com linhas de estado ou arquivos de perfil que não são os do braço" % b)

    def de(b, k):
        return [p[k] for p in procs if p["braco"] == b]

    def desvio(alvo, ref):
        a, r = de(alvo, "prefill_ms_mediana"), de(ref, "prefill_ms_mediana")
        ma, mr = statistics.median(a), statistics.median(r)
        pa, pr = statistics.median(de(alvo, "passo_ms_mediana")), statistics.median(de(ref, "passo_ms_mediana"))
        return {"desvio": (ma - mr) / mr, "pior_caso": (max(a) - min(r)) / min(r), "ms_por_bloco": (ma - mr) / max(blocos_por_pedido, 1), "desvio_passo": (pa - pr) / pr,
                "mediana_alvo_ms": ma, "mediana_ref_ms": mr}
    pri, apo = desvio("B", "A"), desvio("B", "D")
    return {"classe": classe_por_desvio(pri["desvio"]), "motivo": "", "principal": pri, "apoio": apo, "classe_apoio": classe_por_desvio(apo["desvio"])}


# ---------------------------------------------------------------------------------------------------------------------------------- invariantes (espelho de perfil_bloco.rs)
def ordem_esperada(layers):
    v = [(H_ENTRADA, None), (H_BEGIN_STEP, None), (H_UPLOAD_TABELA, None), (H_WAIT_EVENT, None)]
    for l in range(layers):
        v += [(o, l) for o in OPS_CAMADA]
    return v + [(H_POST_LAYERS, None), (H_SEQ_FIM, None), (H_SAIDA, None)]


def bloco_ok_forma(b):
    return isinstance(b, dict) and isinstance(b.get("ops"), list) and all(isinstance(o, list) and len(o) == 4 and all(isinstance(x, int) or x is None for x in o) for o in b["ops"]) \
        and all(isinstance(b.get(k), int) for k in ("t2_ns", "t3_ns", "entrada_ns", "saida_ns"))


def dur(o):
    return max(o[3] - o[2], 0)


def f2_ns(b):
    return max(b["t3_ns"] - b["t2_ns"], 0)


def soma_ops(b):
    return sum(dur(o) for o in b["ops"])


def r_borda(b):
    return max(b["entrada_ns"] - b["t2_ns"], 0) + max(b["t3_ns"] - b["saida_ns"], 0)


def soma_op(b, op):
    return sum(dur(o) for o in b["ops"] if o[0] == op)


def soma_camada(b, l):
    return sum(dur(o) for o in b["ops"] if o[1] == l and o[0] in OPS_CAMADA)


def i3(b, layers):
    return [(o[0], o[1]) for o in b["ops"]] == ordem_esperada(layers)


def i1(b):
    ops = b["ops"]
    if not ops or b["entrada_ns"] != ops[0][2]:
        return False
    dentro = lambda x: x + TOL_NS >= b["t2_ns"] and x <= b["t3_ns"] + TOL_NS
    prev = ops[0][2]
    for o in ops:
        if o[2] < prev or o[3] < o[2] or not dentro(o[2]) or not dentro(o[3]):
            return False
        prev = o[3]
    return prev == b["saida_ns"] and dentro(b["saida_ns"]) and dentro(b["entrada_ns"])


def i2(b):
    return abs((soma_ops(b) + r_borda(b)) - f2_ns(b)) <= 2 * TOL_NS


def i5(b, layers):
    por_camada = sum(soma_camada(b, l) for l in range(layers))
    fixas = sum(dur(o) for o in b["ops"] if o[1] is None)
    return por_camada + fixas == soma_ops(b)


def ressalva_borda(b):
    return f2_ns(b) > 0 and r_borda(b) / f2_ns(b) > LIM_COBERTURA_BORDA


def fim_seq(b, l):
    for o in b["ops"]:
        if o[0] == H_SEQ and o[1] == l:
            return o[3]
    return None


def ini_seq(b, l):
    for o in b["ops"]:
        if o[0] == H_SEQ and o[1] == l:
            return o[2]
    return None


def i8_problemas(b, tb, layers):
    """Espelho de perfil_bloco.rs::i8_problemas: `tb_obs == 0` acusado à parte; causal tb_obs_l >= ini(H_SEQ_l) - 1 µs; defasagem negativa NÃO é violação."""
    if not isinstance(tb, list):
        return ["bloco gen %s: sem carimbos do serviço" % b.get("gen")]
    p = []
    if len(tb) != layers:
        p.append("bloco gen %s: %d camadas com carimbo de serviço, esperadas %d" % (b.get("gen"), len(tb), layers))
    for i, c in enumerate(tb):
        if not isinstance(c, dict) or not all(isinstance(c.get(k), int) for k in ("camada", "gen", "tb_obs_ns", "tb_ns")):
            p.append("bloco gen %s: registro de camada %d ilegível" % (b.get("gen"), i))
            continue
        if c["camada"] != i:
            p.append("bloco gen %s: camada %d fora de ordem na posição %d" % (b.get("gen"), c["camada"], i))
        if c["gen"] != b.get("gen"):
            p.append("bloco gen %s: camada %d com geração de serviço %d" % (b.get("gen"), c["camada"], c["gen"]))
        if c["tb_obs_ns"] == 0:
            p.append("bloco gen %s: camada %d: tb_obs ausente" % (b.get("gen"), c["camada"]))
            continue
        if c["tb_obs_ns"] > c["tb_ns"]:
            p.append("bloco gen %s: camada %d com tb_obs > tb" % (b.get("gen"), c["camada"]))
        ini = ini_seq(b, i)
        if ini is None:
            p.append("bloco gen %s: camada %d sem H_SEQ no host" % (b.get("gen"), c["camada"]))
        elif c["tb_obs_ns"] + TOL_NS < ini:
            p.append("bloco gen %s: camada %d com tb_obs anterior ao início de H_SEQ" % (b.get("gen"), c["camada"]))
        if i > 0 and isinstance(tb[i - 1], dict) and isinstance(tb[i - 1].get("tb_ns"), int):
            a = tb[i - 1]
            if c["tb_ns"] < a["tb_ns"] or c["tb_obs_ns"] < a.get("tb_obs_ns", 0):
                p.append("bloco gen %s: carimbos não crescentes na camada %d" % (b.get("gen"), c["camada"]))
    return p


def defasagens(b, tb):
    out = []
    for i, c in enumerate(tb):
        fim = fim_seq(b, c["camada"])
        if fim is None:
            continue
        ant = tb[i - 1] if i > 0 else None
        out.append({"camada": c["camada"], "defasagem_obs": c["tb_obs_ns"] - fim, "defasagem": c["tb_ns"] - fim,
                    "intervalo_obs": (c["tb_obs_ns"] - ant["tb_obs_ns"]) if ant else None, "intervalo": (c["tb_ns"] - ant["tb_ns"]) if ant else None})
    return out


def valida_pedido_f2(pf2, pbloco, layers):
    """Invariantes de UM pedido. Devolve (problemas I1..I5 que invalidam o pedido, problemas I8 que invalidam só a HI, ressalva_borda)."""
    prob, hi, ress = [], [], False
    blocos = pf2.get("blocos") if isinstance(pf2, dict) else None
    bb = pbloco.get("blocos") if isinstance(pbloco, dict) else None
    if not isinstance(blocos, list) or not isinstance(bb, list):
        return ["pedido sem blocos legíveis"], hi, ress
    if pf2.get("layers") != layers:
        prob.append("layers %r no perfil-f2.json, esperadas %d" % (pf2.get("layers"), layers))
    if len(blocos) != len(bb):
        prob.append("%d blocos de F2 contra %d do perfil-bloco" % (len(blocos), len(bb)))
        return prob, hi, ress
    if pf2.get("host_extra"):
        prob.append("%s registros de F2 sem bloco correspondente" % pf2.get("host_extra"))
    for i, (b, r) in enumerate(zip(blocos, bb)):
        if b.get("sem_host") or not bloco_ok_forma(b):
            prob.append("bloco %d: sem registro de F2 do host" % i)
            continue
        if not i3(b, layers):
            prob.append("bloco %d: I3 (sequência de operações diferente da esperada; %d operações, esperadas %d)" % (i, len(b["ops"]), 6 * layers + 7))
            continue
        if not i1(b):
            prob.append("bloco %d: I1 (monotonia/contenção)" % i)
        if not i2(b):
            prob.append("bloco %d: I2 (soma das operações mais R_borda contra t3 - t2)" % i)
        t = r.get("t_ns") or [None] * 5
        if b.get("gen") != r.get("gen") or b.get("m") != r.get("m") or b.get("pos_ini") != r.get("pos_ini"):
            prob.append("bloco %d: I4 (gen/m/pos_ini do F2 diferem do perfil-bloco)" % i)
        elif b["t2_ns"] != t[2] or b["t3_ns"] != t[3] or abs(soma_ops(b) + r_borda(b) - (t[3] - t[2])) > TOL_NS:
            prob.append("bloco %d: I4 (soma das operações mais R_borda contra F2 do registro por bloco)" % i)
        if not i5(b, layers):
            prob.append("bloco %d: I5 (soma por camada)" % i)
        if ressalva_borda(b):
            ress = True
        hi += ["bloco %d: %s" % (i, x) for x in i8_problemas(b, b.get("tb"), layers)]
    if pf2.get("tb_extra"):
        hi.append("%s registros de serviço sem bloco correspondente" % pf2.get("tb_extra"))
    return prob, hi, ress


# ---------------------------------------------------------------------------------------------------------------------------------- medidas por pedido e por célula
def elegiveis(pf2):
    """Blocos cheios (m = 8) do pedido, sem o primeiro bloco do pedido (reportado à parte para a HH)."""
    return [b for i, b in enumerate(pf2["blocos"]) if i > 0 and b.get("m") == BLOCO and not b.get("sem_host")]


def he_do_pedido(bl):
    grupos, f2t = {}, 0
    for b in bl:
        f2t += f2_ns(b)
        for o in b["ops"]:
            grupos.setdefault((o[0], o[1]), []).append(dur(o))
    longas = 0
    for d in grupos.values():
        md = statistics.median(d)
        longas += sum(x for x in d if x > 10 * md)
    return (longas / f2t) if f2t > 0 else None


def medidas_pedido(bl, layers):
    if not bl:
        return {}
    f2 = mediana([f2_ns(b) for b in bl])
    grupo = lambda ops: mediana([sum(soma_op(b, o) for o in ops) for b in bl])
    por_camada = [mediana([soma_camada(b, l) for b in bl]) for l in range(layers)]
    hd = None
    if layers > 0 and all(x is not None for x in por_camada) and min(por_camada) > 0:
        hd = max(por_camada) / min(por_camada)
    tipos = [grupo([o]) for o in OPS_CAMADA]
    return {"ha": razao(grupo([H_PRE_LAYER, H_ROUTE, H_LAUNCH_HITS, H_ADD_Y]), f2), "hb_frac": razao(grupo([H_PRE_LAYER]), f2),
            "hc": razao(grupo([H_SEQ, H_WAIT_FLAG, H_SEQ_FIM, H_UPLOAD_TABELA, H_WAIT_EVENT, H_BEGIN_STEP]), f2), "hd": hd, "he": he_do_pedido(bl), "tipos": tipos}


def medidas_celula(pedidos_f2, layers, resto):
    """HA a HE por pedido (nunca com blocos de pedidos misturados) e mediana entre pedidos; HB compara as medianas entre pedidos dos valores por tipo; HF e HH por pedido, mediana entre pedidos."""
    por = []
    hf, hh = [], []
    for pf in pedidos_f2:
        bl = elegiveis(pf)
        if not bl:
            continue
        por.append(medidas_pedido(bl, layers))
        f2m = mediana([f2_ns(b) for b in bl])
        blocos = [b for b in pf["blocos"] if not b.get("sem_host")]
        if blocos and blocos[0].get("m") == BLOCO:
            hh.append(razao(f2_ns(blocos[0]), f2m))
        if blocos and blocos[-1].get("m", BLOCO) < BLOCO:
            hf.append(razao(f2_ns(blocos[-1]), f2m))
    agrega = lambda k: mediana([m[k] for m in por if m.get(k) is not None])
    tipos = None
    if por and all(all(x is not None for x in m["tipos"]) for m in por):
        tipos = [statistics.median([m["tipos"][i] for m in por]) for i in range(6)]
    hb_maior = (all(tipos[0] > tipos[i] for i in range(1, 6))) if tipos else None
    return {"ha": agrega("ha"), "hb_frac": agrega("hb_frac"), "hb_maior": hb_maior, "hc": agrega("hc"), "hd": agrega("hd"), "he": agrega("he"),
            "hf_razao": mediana(hf), "hh": mediana(hh), "n_pedidos_com_blocos": len(por), "por_pedido": [{k: v for k, v in m.items() if k != "tipos"} for m in por],
            "faixa_hf": faixa_hf(resto) if resto in (1, 2, 4) else None}


def relatorio_hi(pedidos_f2, pedidos_pb, layers):
    """HI descritiva (sem classe): distribuições com sinal e medianas por camada, só de pedidos com a coleta de serviço válida (I8), blocos elegíveis."""
    amostras = []
    for pf in pedidos_f2:
        for b in elegiveis(pf):
            if isinstance(b.get("tb"), list):
                amostras.append((b, defasagens(b, b["tb"])))
    todas = lambda k: [d[k] for _, ds in amostras for d in ds if d.get(k) is not None]
    por_camada = lambda k: [{"camada": l, "n": len([1 for _, ds in amostras for d in ds if d["camada"] == l and d.get(k) is not None]),
                             "mediana": mediana([d[k] for _, ds in amostras for d in ds if d["camada"] == l and d.get(k) is not None])} for l in range(layers)]
    return {"tipo": "hi", "descritivo": True, "unidade": "ns, com sinal", "amostras": len(amostras),
            "distribuicao": {k: dist(todas(k)) for k in ("defasagem_obs", "defasagem", "intervalo_obs", "intervalo")},
            "mediana_por_camada": dict({k: por_camada(k) for k in ("defasagem_obs", "defasagem", "intervalo_obs", "intervalo")},
                                       host_seis_operacoes=[{"camada": l, "n": len(amostras), "mediana_ns": mediana([soma_camada(b, l) for b, _ in amostras])} for l in range(layers)]),
            "limites": "defasagem negativa e conservada com sinal; tb_obs e tb nao medem a GPU nem identificam o mecanismo de espera; tb inclui leitura, copia e escalonamento do servico"}


# ---------------------------------------------------------------------------------------------------------------------------------- leitura
def le_linhas(d, rx, nome):
    """Estado impresso em CADA processo (p*/stdout.txt): 'ligado'/'desligado' ou None; devolve (estado, motivos)."""
    mot, estados = [], []
    procs = sorted(x for x in os.listdir(d) if x.startswith("p") and x[1:].isdigit())
    if not procs:
        return None, ["sem pasta de processo (pK)"]
    for p in procs:
        txt = X.le_txt(os.path.join(d, p, "stdout.txt")) or ""
        ach = [rx.match(x.strip()).group(1) for x in txt.splitlines() if rx.match(x.strip())]
        if len(ach) != 1:
            mot.append("%s: %d linha(s) `%s: ligado|desligado` no stdout do binário (esperada 1)" % (p, len(ach), nome))
        else:
            estados.append(ach[0])
    if mot or len(set(estados)) != 1:
        return None, mot or ["processos da célula com estados diferentes de %s" % nome]
    return estados[0], []


def le_f2_json(d, m, n_medidas, layers):
    """perfil-f2.json de UM processo B: devolve (pedidos_f2 por índice, motivos de processo). Esquema tipo perfil-f2 versao 1."""
    arqs = []
    for raiz, _, a in os.walk(d):
        if "perfil-f2.json" in a:
            arqs.append(os.path.join(raiz, "perfil-f2.json"))
    if len(arqs) != 1:
        return [], ["braço B com %d perfil-f2.json (esperado 1)" % len(arqs)]
    j = X.le_json(arqs[0])
    if not isinstance(j, dict) or j.get("tipo") != "perfil-f2" or j.get("versao") != 1 or not isinstance(j.get("pedidos"), list):
        return [], ["perfil-f2.json ilegível ou fora do esquema (tipo perfil-f2, versao 1)"]
    peds = j["pedidos"]
    mot = []
    if len(peds) != n_medidas + 1:
        mot.append("perfil-f2.json com %d entradas, esperadas %d (1 aquecimento e %d medidas)" % (len(peds), n_medidas + 1, n_medidas))
    for i, p in enumerate(peds):
        if not isinstance(p, dict) or p.get("modelo") != m or p.get("indice") != i:
            mot.append("entrada %d do perfil-f2.json com modelo/índice incoerentes" % i)
    return peds, mot


def problemas_identidade(p, m, i):
    """Identidade do pedido no perfil-base (mesmas regras de valida_pedidos): modelo, índice e marca de aquecimento (só o 0 é aquecimento)."""
    if not isinstance(p, dict):
        return ["pedido do perfil-bloco ilegível"]
    out = []
    if p.get("modelo") != m:
        out.append("perfil-bloco: modelo %r, não %r" % (p.get("modelo"), m))
    if p.get("indice") != i:
        out.append("perfil-bloco: índice %r na posição %d" % (p.get("indice"), i))
    if bool(p.get("aquecimento")) != (i == 0):
        out.append("perfil-bloco: aquecimento=%r na posição %d (só a 0 é aquecimento)" % (p.get("aquecimento"), i))
    return out


def problemas_base(p, t):
    """Problemas estruturais do pedido MEDIDO no perfil-base (mesmas regras de instrumento-extrai.valida_pedidos): plano de blocos completo, n, valido, conservação. Pedido com problema NÃO conta como válido."""
    if not isinstance(p, dict):
        return ["pedido do perfil-bloco ilegível"]
    out = []
    plano = I.plano_blocos(N_POS[t])
    if p.get("n") != N_POS[t]:
        out.append("perfil-bloco: n=%r posições, esperado %d" % (p.get("n"), N_POS[t]))
    if p.get("plano_m") != plano or [b.get("m") for b in p.get("blocos", []) if isinstance(b, dict)] != plano:
        out.append("perfil-bloco: plano de blocos diferente de plano_blocos(%d, 8)" % N_POS[t])
    if p.get("valido") is not True:
        out.append("perfil-bloco: pedido inválido pelo próprio binário")
    if any(not isinstance(b, dict) or b.get("conserva") is not True for b in p.get("blocos", [])):
        out.append("perfil-bloco: bloco que não conserva F0+F1+F2+F3 = B")
    return out


def le_pasta(saida, nome, m, t, n_prev, braco, refs, layers):
    """Pasta de processo (controle, célula ou smoke): base de instrumento-extrai.le_dir_celula mais as checagens da F2. `valido_controle` = processo temporalmente válido, com as
    linhas e arquivos do braço, sem estouro de teto; pedidos_f2 = pedidos MEDIDOS B válidos (I1..I5), com a marca hi_valida."""
    c = I.le_dir_celula(saida, nome, m, "L100", t, n_prev, refs)
    if c["status"] == "NÃO RODOU":
        c.update({"estado_f2": None, "tem_f2": False, "motivos_f2": [], "pedidos_f2": [], "aquecimento_f2": None, "valido_controle": False})
        return c
    d = os.path.join(saida, "celulas", nome)
    est_f2, mot_l = le_linhas(d, LINHA_F2_RE, "perfil-f2")
    tem_f2 = any("perfil-f2.json" in a for _, _, a in os.walk(d))
    mot = list(mot_l)
    esperado_f2 = "ligado" if braco == "B" else "desligado"
    esperado_bl = "desligado" if braco == "D" else "ligado"
    if est_f2 is not None and est_f2 != esperado_f2:
        mot.append("braço %s com `perfil-f2: %s` (esperado %s)" % (braco, est_f2, esperado_f2))
    if c.get("estado_linha") is not None and c["estado_linha"] != esperado_bl:
        mot.append("braço %s com `perfil-bloco: %s` (esperado %s)" % (braco, c["estado_linha"], esperado_bl))
    if braco != "B" and tem_f2:
        mot.append("braço %s produziu perfil-f2.json" % braco)
    pedidos_f2, aq_f2, mot_f2 = [], None, []
    if braco == "B" and est_f2 == "ligado":
        peds, mj = le_f2_json(d, m, n_prev, layers)
        mot += mj
        pbs, arq_ok = None, True
        for raiz, _, a in os.walk(d):
            if "perfil-bloco.json" in a:
                jb = X.le_json(os.path.join(raiz, "perfil-bloco.json"))
                # falha do ARQUIVO (mesma regra de instrumento-extrai.valida_pedidos): esquema inválido impede a admissão de TODOS os pedidos desta fonte
                if not isinstance(jb, dict) or jb.get("tipo") != "perfil-bloco" or jb.get("versao") != 2 or not isinstance(jb.get("pedidos"), list):
                    arq_ok = False
                    mot_f2.append("perfil-bloco.json ilegível ou fora do esquema (tipo perfil-bloco, versao 2): nenhum pedido desta fonte é admitido")
                else:
                    pbs = jb["pedidos"]
        pbs = pbs if isinstance(pbs, list) else []
        for i, pf in enumerate(peds):
            if not arq_ok:
                continue
            if i >= len(pbs):
                mot_f2.append("pedido %d sem perfil-bloco correspondente" % i)
                continue
            prob, hi, ress = valida_pedido_f2(pf, pbs[i], layers)
            prob = problemas_identidade(pbs[i], m, i) + prob       # falha de PEDIDO: modelo, índice e marca de aquecimento esperados
            if i > 0:
                prob = problemas_base(pbs[i], t) + prob     # a validade estrutural do perfil-base (plano de blocos completo etc.) vale para a admissão do pedido
            info = {"indice": i, "problemas": prob, "problemas_hi": hi, "hi_valida": not hi, "ressalva_borda": ress, "f2": pf}
            if i == 0:
                aq_f2 = info      # o prefill do aquecimento continua coletado e validado, mas fora das medianas
            elif prob:
                mot_f2.append("pedido %d inválido: %s" % (i, "; ".join(prob)[:300]))
            else:
                pedidos_f2.append(info)
    elif braco == "B":
        mot.append("braço B sem `perfil-f2: ligado`")
    c.update({"estado_f2": est_f2, "tem_f2": tem_f2, "motivos_f2": sorted(set(mot)), "pedidos_f2": pedidos_f2, "aquecimento_f2": aq_f2, "motivos_pedidos_f2": sorted(set(mot_f2))})
    c["valido_controle"] = c["status_temporal"] == "OK" and not mot and (bool(c.get("estado_linha")) and True)
    return c


def integridade_fixos(saida, m):
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


ORDEM_RE = r"^(\d\d) (ctrl-%s-L100-medio-(\d\d)-([DAB])) n=(\d+) braco=([DAB]) inicio=\S+ fim=\S+ rc=(\d+)$"


def confere_ordem(saida, m, nomes_pastas):
    """Ordem EFETIVA do controle lida do ordem.txt, cruzada com as 9 pastas (D A B, B D A, A B D; n = 3; rc = 0). Devolve o motivo (controle incompleto) ou None."""
    txt = X.le_txt(os.path.join(saida, "ordem.txt"))
    if txt is None:
        return "ordem.txt ausente: a ordem efetiva do controle não foi provada"
    rx = re.compile(ORDEM_RE % re.escape(m))
    linhas = []
    for x in txt.splitlines():
        mt = rx.match(x.strip())
        if mt:
            linhas.append({"chave": mt.group(2), "braco_nome": mt.group(4), "n": int(mt.group(5)), "braco": mt.group(6), "rc": int(mt.group(7))})
        elif len(x.split(" ")) > 1 and x.split(" ")[1].startswith("ctrl-%s-L100-" % m):
            return "linha de ordem.txt ilegível para o controle: %r" % x.strip()[:80]
    if len(linhas) != len(ORDEM_CONTROLE):
        return "ordem.txt com %d linhas do controle (esperadas 9): registro truncado ou ausente" % len(linhas)
    esperado = ["ctrl-%s-L100-medio-%02d-%s" % (m, i + 1, b) for i, b in enumerate(ORDEM_CONTROLE)]
    if [x["chave"] for x in linhas] != esperado:
        return "ordem efetiva de ordem.txt (%s) diferente de D A B, B D A, A B D com os nomes esperados" % " ".join(x["chave"][-1] for x in linhas)
    if any(x["braco"] != x["braco_nome"] for x in linhas):
        return "braço registrado em ordem.txt diferente do braço do nome da pasta"
    if any(x["rc"] != 0 or x["n"] != 3 for x in linhas):
        return "ordem.txt com processo de controle de rc != 0 ou n != 3"
    if sorted(nomes_pastas) != sorted(esperado):
        return "pastas do controle diferentes das 9 esperadas"
    return None


def teto_excedido(saida):
    txt = X.le_txt(os.path.join(saida, "teto-excedido.txt")) or ""
    return {x.split()[0] for x in txt.splitlines() if x.strip()}


# ---------------------------------------------------------------------------------------------------------------------------------- processamento
def processa(saida, ids_arq, refs_dir, textos_arq):
    refs = X.Refs(ids_arq, refs_dir, textos_arq)
    orc = X.le_json(os.path.join(saida, "orcamento.json"))
    orc = orc if isinstance(orc, dict) else {}
    decl = orc.get("modelos")
    if not (isinstance(decl, list) and len(decl) == 1 and decl[0] in MODELOS):
        raise Recusa("orcamento.json sem um modelo único declarado (\"modelos\": [\"1b\"] ou [\"3b\"])")
    m = decl[0]
    layers = LAYERS_DO_MODELO[m]
    rec = I.prova_potencia(saida)
    if rec:
        raise Recusa("; ".join(rec))
    nomes = sorted(os.listdir(os.path.join(saida, "celulas"))) if os.path.isdir(os.path.join(saida, "celulas")) else []
    algum = False
    for n_ in nomes:
        if (CTRL_RE.match(n_) or CEL_RE.match(n_) or SMOKE_RE.match(n_)) and os.path.isdir(os.path.join(saida, "celulas", n_)):
            if le_linhas(os.path.join(saida, "celulas", n_), LINHA_F2_RE, "perfil-f2")[0] is not None:
                algum = True
    if not algum:
        raise Recusa("sem a linha `perfil-f2: ligado|desligado` em nenhum processo: não é saída do binário da F2")
    integ = sorted(set(X.integridade(saida) + integridade_fixos(saida, m)))
    ref_prob = I.confere_referencias(saida, m, ids_arq, refs_dir, textos_arq)
    nao_rodou = [str(x) for x in (orc.get("nao_rodou") or [])]
    estouros = teto_excedido(saida)
    leituras = {}

    def le(nome, t, n, braco):
        if nome not in leituras:
            c = le_pasta(saida, nome, m, t, n, braco, refs, layers)
            if nome in estouros:
                c["valido_controle"] = False
                c["motivos_f2"] = sorted(set(c.get("motivos_f2", []) + ["teto de processo excedido (processo inválido, sem repetição)"]))
            leituras[nome] = c
        return leituras[nome]

    # smoke: 1 processo no curto, em B, 1 aquecimento e 1 medida
    sm_nome = "smoke-%s-L100-curto-B" % m
    smoke_ok, smoke_mot = False, ["smoke ausente (%s)" % sm_nome]
    if sm_nome in nomes:
        c = le(sm_nome, "curto", 1, "B")
        mots = list(c.get("motivos") or []) + list(c.get("motivos_f2") or []) + list(c.get("motivos_pedidos_f2") or [])
        mots += ["aquecimento do smoke: " + "; ".join(c["aquecimento_f2"]["problemas"]) for _ in [0] if c.get("aquecimento_f2") and c["aquecimento_f2"]["problemas"]]
        mots += ["aquecimento do smoke com I8 falha: " + "; ".join(c["aquecimento_f2"]["problemas_hi"])[:200] for _ in [0] if c.get("aquecimento_f2") and c["aquecimento_f2"]["problemas_hi"]]
        mots += ["smoke com I8 falha: " + "; ".join(p["problemas_hi"])[:200] for p in c.get("pedidos_f2", []) if p["problemas_hi"]]
        if len(c.get("pedidos_f2", [])) != 1 and not mots:
            mots.append("smoke sem o pedido medido válido")
        smoke_ok, smoke_mot = (not mots, "; ".join(mots)[:500])
    # controle
    ctrl = sorted((int(CTRL_RE.match(n_).group(2)), CTRL_RE.match(n_).group(3), n_) for n_ in nomes if CTRL_RE.match(n_) and CTRL_RE.match(n_).group(1) == m)
    procs = []
    for (nn, b, nome) in ctrl:
        c = le(nome, "medio", 3, b)
        meds = c.get("meds") or []
        procs.append((nn, b, nome, c, {"braco": b, "prefill_ms_mediana": mediana([x["prefill_ms"] for x in meds]), "passo_ms_mediana": I.mediana_passo(meds),
                                       "imprimiu_bloco": c.get("estado_linha") == "ligado", "imprimiu_f2": c.get("estado_f2") == "ligado", "produziu_bloco": bool(c.get("tem_perfil")),
                                       "produziu_f2": bool(c.get("tem_f2")), "valido": bool(c.get("valido_controle"))}))
    orc_ok = not any(x.startswith("ctrl-%s-L100-" % m) for x in nao_rodou)
    mot_ordem = confere_ordem(saida, m, [x[2] for x in procs]) if procs else "sem processos de controle"
    nums = [x[0] for x in procs]
    if mot_ordem is None and nums != list(range(1, len(nums) + 1)):
        mot_ordem = "numeração das pastas do controle não é 01..09"
    ctl = classifica_controle_f2([x[4] for x in procs], orc_ok, mot_ordem)
    ctl["ordem_efetiva"] = mot_ordem or "confere com ordem.txt (D A B, B D A, A B D, rc 0)"
    ctl["processos"] = [{"nn": nn, "braco": b, "pasta": nome, "status_temporal": c["status_temporal"], "estado_bloco": c.get("estado_linha"), "estado_f2": c.get("estado_f2"),
                         "tem_perfil_bloco": c.get("tem_perfil"), "tem_perfil_f2": c.get("tem_f2"), "valido": c.get("valido_controle"), "prefill_ms_mediana": pr["prefill_ms_mediana"],
                         "passo_ms_mediana": pr["passo_ms_mediana"], "motivos": sorted(set(c["motivos_temporais"] + c.get("motivos_f2", [])))} for (nn, b, nome, c, pr) in procs]
    # divergência de ids contra a 5c em qualquer processo (D, A ou B), aquecimento incluído; ids de B iguais aos de D (I7)
    div = []
    for (_, b, nome, c, _) in procs:
        div += [dict(x, pasta=nome, braco=b) for x in c["seq"]["divergencias"]]
    celnomes = {t: [n_ for n_ in nomes if CEL_RE.match(n_) and CEL_RE.match(n_).group(1) == m and CEL_RE.match(n_).group(2) == t] for t in ("curto", "longo")}
    ref_ausente = []
    for (_, b, nome, c, _) in procs:
        if c["status"] != "NÃO RODOU" and not c["seq"]["referencia_disponivel"]:
            ref_ausente.append(nome)
    for t, lst in celnomes.items():
        for nome in lst:
            mj = X.le_json(os.path.join(saida, "celulas", nome, "celula.json")) or {}
            c = le(nome, t, mj.get("n_medidas") if mj.get("n_medidas") in (3, 5) else 5, "B")
            div += [dict(x, pasta=nome, texto=t) for x in c["seq"]["divergencias"]]
            if c["status"] != "NÃO RODOU" and not c["seq"]["referencia_disponivel"]:
                ref_ausente.append(nome)
    gb = {tuple(x["gerados"]) for (_, b, _, c, _) in procs if b == "B" for x in c.get("meds") or []}
    gd = {tuple(x["gerados"]) for (_, b, _, c, _) in procs if b == "D" for x in c.get("meds") or []}
    ids_b_d = bool(gb) and bool(gd) and gb == gd and len(gb) == 1
    bloq = []
    if integ:
        bloq.append("integridade quebrada")
    if not smoke_ok:
        bloq.append("smoke FALHOU ou ausente: " + smoke_mot)
    if ref_prob:
        bloq.append("; ".join(ref_prob))
    if ref_ausente:
        bloq.append("referência 5c indisponível em: " + ", ".join(sorted(set(ref_ausente))))
    if div:
        bloq.append("ids diferentes da referência 5c (I7: instrumento inválido; inclui o aquecimento)")
    ctl_final = ctl
    if bloq and controle_permite(ctl["classe"]):
        ctl_final = dict(ctl, classe_calculada=ctl["classe"], classe="SEM PROVA", motivo="bloqueio: " + "; ".join(bloq) + " (as medidas ficam só como descrição)")
    # textos
    textos = {}
    for t in TEXTOS:
        fontes = []
        if t == "medio":
            fontes = [(nome, c) for (_, b, nome, c, _) in procs if b == "B"]
        else:
            fontes = [(nome, le(nome, t, 5, "B")) for nome in celnomes[t]]
        resto = N_POS[t] % BLOCO
        if not fontes:
            textos[t] = {"status": "NÃO RODOU", "motivos": [], "celulas": [], "pedidos_validos": 0, "minimo_pedidos": MIN_PEDIDOS[t], "medidas": {}, "hi": None, "hipoteses": {}, "ressalvas": [], "resto": resto}
            continue
        motivos, pedidos, usadas, ress, pbl = [], [], [], [], []
        for nome, c in fontes:
            if c["status"] == "NÃO RODOU":
                motivos.append("%s: não rodou" % nome)
                continue
            if c["status_temporal"] != "OK" or c.get("motivos_f2"):
                motivos += ["%s: %s" % (nome, x) for x in c["motivos_temporais"] + c.get("motivos_f2", [])]
                continue   # processo temporalmente inválido ou sem as linhas/arquivos do braço: nenhum pedido seu entra
            if c.get("motivos_perfil"):
                motivos += ["%s: perfil: %s" % (nome, x) for x in c["motivos_perfil"]]
            motivos += ["%s: %s" % (nome, x) for x in c.get("motivos_pedidos_f2", [])]
            usadas.append(nome)
            for p in c["pedidos_f2"]:
                pedidos.append(p)
        ok_peds = [p for p in pedidos]
        med = medidas_celula([p["f2"] for p in ok_peds], layers, resto) if ok_peds else {}
        hi_peds = [p["f2"] for p in ok_peds if p["hi_valida"]]
        hi = relatorio_hi(hi_peds, None, layers) if hi_peds else None
        for p in ok_peds:
            if p["ressalva_borda"]:
                ress.append("R_borda/F2 acima de 3%% em bloco do pedido %d (ressalva, não invalida)" % p["indice"])
        n_hi_inv = len(ok_peds) - len(hi_peds)
        if n_hi_inv:
            ress.append("HI: %d pedido(s) com coleta de serviço inválida (I8), fora do relatório da HI" % n_hi_inv)
        enough = len(ok_peds) >= MIN_PEDIDOS[t]
        hip = {}
        for h in ("HA", "HB", "HC", "HD", "HE", "HF", "HG", "HH", "HI"):
            if not usadas and not ok_peds:
                hip[h] = "SEM PROVA"
            elif bloq or not controle_permite(ctl_final["classe"]):
                hip[h] = "SEM PROVA"
            elif not enough:
                hip[h] = "SEM PROVA"
            elif h == "HG":
                hip[h] = "sem dado"
            elif h == "HF":
                hip[h] = "descritivo" if fin(med.get("hf_razao")) is not None else "sem dado"
            elif h == "HI":
                hip[h] = "descritivo" if (hi and hi["amostras"] > 0) else "sem dado"
            elif t == "curto" and h in ("HD", "HE"):
                hip[h] = "descritivo"
            else:
                hip[h] = {"HA": lambda: classifica_ha(med.get("ha")), "HB": lambda: classifica_hb(med.get("hb_frac"), med.get("hb_maior")), "HC": lambda: classifica_hc(med.get("hc")),
                          "HD": lambda: classifica_hd(med.get("hd")), "HE": lambda: classifica_he(med.get("he")), "HH": lambda: classifica_hh(med.get("hh"))}[h]()
        if not enough:
            ress.append("mínimo de pedidos válidos não atingido: %d de %d exigidos (SEM PROVA na classificação)" % (len(ok_peds), MIN_PEDIDOS[t]))
        estado = "OK" if (enough and not motivos) else ("SEM PROVA" if (usadas or motivos) else "NÃO RODOU")
        n_amost = sum(len(elegiveis(p["f2"])) for p in ok_peds)
        textos[t] = {"status": estado, "motivos": sorted(set(motivos)), "celulas": usadas, "pedidos_validos": len(ok_peds), "minimo_pedidos": MIN_PEDIDOS[t], "amostras_elegiveis": n_amost,
                     "medidas": med, "hi": hi, "hipoteses": hip, "ressalvas": ress, "resto": resto}
    return {"tipo": "f2-host-resultado", "criterio": "criterio-f2-host-20261111.md", "descritivo": True, "modelo_da_janela": m, "layers": layers,
            "aviso": "Descritivo: sem veredito geral e sem PASSA. HA a HH são previsões pré-registradas, só conferidas ('dentro', 'fora', 'sem leitura', 'sem dado', 'SEM PROVA', 'descritivo'); "
                     "HF, HG e HI não têm classe de dentro/fora; nada prova causa nem identifica o mecanismo de espera; o tempo com instrumento não é o tempo de produção.",
            "integridade_problemas": integ, "referencia_problemas": ref_prob, "smoke": {"ok": smoke_ok, "motivo": smoke_mot}, "bloqueios": bloq, "controle": ctl_final,
            "ids_B_igual_D": ids_b_d, "divergencias_5c": div, "textos": textos, "orcamento": orc, "teto_excedido": sorted(estouros)}


# ---------------------------------------------------------------------------------------------------------------------------------- relatório
def f(x, n=2):
    return "-" if x is None else ("%.*f" % (n, x))


def pc(x):
    return "-" if x is None else ("%.1f%%" % (100 * x))


def relatorio(res):
    m = res["modelo_da_janela"]
    o = ["# F2 do host por operação em L100, janela do modelo %s (criterio-f2-host-20261111.md)" % m.upper(), "", res["aviso"], ""]
    if res["referencia_problemas"]:
        o += ["**Referência 5c sem prova:**", ""] + ["- %s" % x for x in res["referencia_problemas"]] + [""]
    if res["integridade_problemas"]:
        o += ["**Integridade QUEBRADA: todas as hipóteses SEM PROVA (medidas brutas preservadas só como descrição).**", ""] + ["- %s" % x for x in res["integridade_problemas"]] + [""]
    o += ["Smoke (1 processo no curto, em B): **%s**%s" % ("OK" if res["smoke"]["ok"] else "FALHOU", "" if res["smoke"]["ok"] else " (%s)" % res["smoke"]["motivo"]), ""]
    c = res["controle"]
    o += ["## Controle do custo do instrumento (D, A, B)", "", "Classe (|B contra A|): **%s**%s" % (c["classe"], (" (%s)" % c["motivo"]) if c.get("motivo") else "")]
    if c.get("classe_calculada"):
        o.append("- classe calculada antes do bloqueio: %s" % c["classe_calculada"])
    o.append("- ordem efetiva (ordem.txt): %s" % c.get("ordem_efetiva"))
    for rot, k in (("B contra A (principal)", "principal"), ("B contra D (apoio)", "apoio")):
        d = c.get(k)
        if d:
            o.append("- %s: desvio de prefill_ms %s; pior caso %s; diferença de prefill por bloco executado, incluindo primeiro e resto, %s ms (%d blocos); desvio do passo de decode %s" % (
                rot, pc(d["desvio"]), pc(d["pior_caso"]), f(d["ms_por_bloco"], 3), BLOCOS_POR_PEDIDO_MEDIO, pc(d["desvio_passo"])))
    if c.get("classe_apoio"):
        o.append("- classe descritiva de B contra D: %s (não decide)" % c["classe_apoio"])
    for p in c.get("processos", []):
        o.append("  - %02d %s %s: temporal %s, bloco %s, f2 %s, perfil-bloco.json %s, perfil-f2.json %s, válido %s, prefill mediano %s ms, passo %s ms%s" % (
            p["nn"], p["braco"], p["pasta"], p["status_temporal"], p["estado_bloco"], p["estado_f2"], "sim" if p["tem_perfil_bloco"] else "não", "sim" if p["tem_perfil_f2"] else "não",
            p["valido"], f(p["prefill_ms_mediana"]), f(p["passo_ms_mediana"]), ("; " + "; ".join(p["motivos"])) if p["motivos"] else ""))
    o.append("- ids de B iguais aos de D (I7, processos do controle): %s" % {True: "sim", False: "não"}[bool(res["ids_B_igual_D"])])
    if res["divergencias_5c"]:
        o.append("- DIVERGÊNCIA contra a referência 5c em %d pedido(s)" % len(res["divergencias_5c"]))
    if res["bloqueios"]:
        o.append("- BLOQUEIOS (hipóteses em SEM PROVA): " + "; ".join(res["bloqueios"]))
    o.append("")
    for t, tx in res["textos"].items():
        o += ["## Texto %s (resto %d)" % (t, tx["resto"]), "", "status **%s**; pedidos válidos %d (mínimo %d); amostras elegíveis %s; células: %s" % (
            tx["status"], tx["pedidos_validos"], tx["minimo_pedidos"], tx.get("amostras_elegiveis", "-"), ", ".join(tx["celulas"]) or "nenhuma")]
        md = tx["medidas"]
        if md:
            o.append("- HA %s; HB F2 %s (maior soma: %s); HC %s; HD %s; HE %s; HH %s; HF razão %s (faixa %s)" % (
                pc(md.get("ha")), pc(md.get("hb_frac")), md.get("hb_maior"), pc(md.get("hc")), f(md.get("hd"), 3), pc(md.get("he")), f(md.get("hh"), 3), f(md.get("hf_razao"), 3),
                ("%.4f a %.4f" % tuple(md["faixa_hf"])) if md.get("faixa_hf") else "r = 6 ou 7: descritivo"))
        o += ["", "| HA | HB | HC | HD | HE | HF | HG | HH | HI |", "|---|---|---|---|---|---|---|---|---|"]
        if tx["hipoteses"]:
            o.append("| " + " | ".join(tx["hipoteses"][h] for h in ("HA", "HB", "HC", "HD", "HE", "HF", "HG", "HH", "HI")) + " |")
        if tx["hi"]:
            hi = tx["hi"]
            o += ["", "HI (somente descritiva, ns com sinal, %d amostras): " % hi["amostras"] + "; ".join(
                "%s min %s, mediana %s, p90 %s, max %s" % (k, f(v["min"], 0), f(v["mediana"], 0), f(v["p90"], 0), f(v["max"], 0)) for k, v in hi["distribuicao"].items()), "", hi["limites"]]
        for r in tx["ressalvas"]:
            o.append("- ressalva: " + r)
        for x in tx["motivos"]:
            o.append("- %s" % x)
        o.append("")
    orc = res.get("orcamento") or {}
    o += ["## Orçamento", "", "decorrido %s s de %s s; cortes: %s; não rodou: %s; teto de processo excedido: %s" % (
        orc.get("decorrido_s", "?"), orc.get("limite_s", "?"), "; ".join(orc.get("cortes") or []) or "nenhum", "; ".join(orc.get("nao_rodou") or []) or "nada",
        ", ".join(res.get("teto_excedido") or []) or "nenhum"), "", "## Conclusão", "", "Este relatório não declara PASSA nem conclui causa; nível B e qualquer mudança de desempenho exigem nova decisão do Breno."]
    return "\n".join(o) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="extrator da F2 do host por operação")
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
        print("f2-extrai: RECUSADO: %s" % ex, file=sys.stderr)
        with open(os.path.join(a.saida, "f2-host-resultado.json"), "w", encoding="utf-8") as fo:
            json.dump({"tipo": "f2-host-resultado", "recusado": True, "motivo": str(ex)}, fo, ensure_ascii=False, indent=1)
        with open(os.path.join(a.saida, "f2-host-resultado.md"), "w", encoding="utf-8") as fo:
            fo.write("# F2 do host: RECUSADO\n\n%s\n\nNenhuma classificação foi feita; os dados brutos ficam na pasta.\n" % ex)
        return 2
    X.limpa_nao_finitos(res)
    with open(os.path.join(a.saida, "f2-host-resultado.json"), "w", encoding="utf-8") as fo:
        json.dump(res, fo, ensure_ascii=False, indent=1, allow_nan=False)
    with open(os.path.join(a.saida, "f2-host-resultado.md"), "w", encoding="utf-8") as fo:
        fo.write(relatorio(res))
    print("f2-extrai: controle %s; gravados f2-host-resultado.json e f2-host-resultado.md" % res["controle"]["classe"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
