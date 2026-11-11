#!/usr/bin/env python3
"""Extrator da F2 nível B, só o pré-roteador (criterio-f2-nivel-b-20261111.md), descritivo, sem veredito geral e sem frase causal.

    python3 -I f2b-extrai-v2.py --saida DIR_DA_JANELA [--ids cmp-ids-prompt.json] [--refs 5c-referencia] [--textos 5c-textos/textos.json] [--f2-anterior DIR_DA_JANELA_F2_DE_11H]

Só biblioteca padrão. NÃO roda nada, NÃO usa GPU: lê o que roda-f2b-janela.sh gravou e grava DIR/f2b-host-resultado.json e DIR/f2b-host-resultado.md (um modelo por janela).
Reaproveita f2-extrai.py (e, por baixo, instrumento-extrai.py): leitura de célula, 5c, integridade, identidade e estrutura do perfil-base, invariantes I1 a I5 do nível A e a leitura do perfil-f2.json.
Recalcula em Python IB1 a IB5 a partir dos registros brutos de perfil-f2b.json (nunca confia nas flags gravadas pelo binário).

Layout lido (escrito por roda-f2b-janela.sh):
  DIR/celulas/smoke-<modelo>-L100-curto-B2/            smoke: 1 processo no curto, braço B2, 1 aquecimento e 1 medida
  DIR/celulas/ctrl-<modelo>-L100-medio-<NN>-<D|A|B2>/  controle de 9 processos (ordem D A B2, B2 D A, A B2 D; --repete 3)
  DIR/celulas/lastro-<modelo>-L100-<curto|longo>-quente/ células principais em B2 (1 aquecimento + 5 ou 3 medidas)
  em cada pasta de processo: p1/stdout.txt (linhas `perfil-bloco:`, `perfil-f2:` e `perfil-f2b:`), p1/saida/perfil-bloco.json (A e B2), perfil-f2.json e perfil-f2b.json (só B2).
Regras (critério): D = nada ligado, A = só perfil-bloco, B2 = as três; controle por |B2 contra A| (<= 3% comprovada, <= 10% com ressalva, > 10% SEM PROVA), B2 contra D só como apoio; HB1 a HB6 por pedido e
mediana entre os pedidos da célula; mínimo de 3 pedidos válidos em curto e longo e 9 no médio; controle incompleto invalida todas as H do modelo. HB3 (decisão do Breno 12:03): dentro >= 0,60, fora < 0,40.
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


F = carrega("f2_extrai", "f2-extrai.py")
I, X = F.I, F.X
MODELOS, TEXTOS, N_POS, BLOCO = F.MODELOS, F.TEXTOS, F.N_POS, F.BLOCO
LAYERS_DO_MODELO = F.LAYERS_DO_MODELO
MIN_PEDIDOS = F.MIN_PEDIDOS
TOL_NS = F.TOL_NS
EPS = F.EPS
fin, mediana = F.fin, F.mediana
ORDEM_CONTROLE = ["D", "A", "B2", "B2", "D", "A", "A", "B2", "D"]
BLOCOS_POR_PEDIDO_MEDIO = F.BLOCOS_POR_PEDIDO_MEDIO
LIM_HB3_DENTRO, LIM_HB3_FORA = 0.60, 0.40       # decisão do Breno 12:03
ROTULO_LANCA = "tempo da chamada de envio de trabalho (a expressão async_on)"
LINHA_F2B_RE = re.compile(r"^perfil-f2b: (ligado|desligado)$")
CTRL_RE = re.compile(r"^ctrl-(1b|3b)-L100-medio-(\d\d)-(D|A|B2)$")
CEL_RE = F.CEL_RE
SMOKE_RE = re.compile(r"^smoke-(1b|3b)-L100-curto-B2$")
FIXO_CHAVES = ("criterio-f2b", "plano-f2b", "binario-f2b", "script-f2b-extrai", "script-f2-extrai", "script-instrumento-extrai", "script-perfil-extrai", "script-cmp-extrai", "script-lastro",
               "script-monitor", "script-janela", "script-limite", "criterio-cmp", "criterio-degrau1", "criterio-degrau2", "cmp-ids-prompt", "texto-curto", "texto-medio", "texto-longo", "textos-json",
               "ref5c-1b", "ref5c-3b")
# códigos de B_* (perfil_bloco.rs)
B_LOCK, B_SETUP, B_GENERICS, B_ATTN_PREP, B_LANCA, B_DTOD = range(6)
# Contrato v1: instrumento de 14:10 (seis cópias device-para-device por posição e camada, 21 registros por posição). Só para dados já coletados.
ORDEM_POS_V1 = [B_SETUP, B_GENERICS, B_LANCA, B_LANCA, B_LANCA, B_LANCA, B_DTOD, B_DTOD, B_LANCA, B_LANCA, B_DTOD, B_DTOD, B_DTOD, B_ATTN_PREP, B_LANCA, B_LANCA, B_DTOD, B_LANCA, B_LANCA, B_LANCA, B_SETUP]
# Contrato v2: instrumento do plano de cópias (O1: as quatro cópias de reforma viraram reshape, 17 registros por posição, sem registro fictício). Espelho de perfil_bloco.rs::ORDEM_POS_B.
ORDEM_POS_V2 = [B_SETUP, B_GENERICS, B_LANCA, B_LANCA, B_LANCA, B_LANCA, B_LANCA, B_LANCA, B_DTOD, B_DTOD, B_ATTN_PREP, B_LANCA, B_LANCA, B_LANCA, B_LANCA, B_LANCA, B_SETUP]
ORDENS = {1: ORDEM_POS_V1, 2: ORDEM_POS_V2}
ORDEM_POS = ORDEM_POS_V1   # nome antigo: o contrato v1 (preservado para os dados de 14:10)
H_PRE_LAYER = F.H_PRE_LAYER
LIM_RESIDUO = 0.03


class Recusa(Exception):
    pass


def _ge(x, d, f):
    return F._ge(x, d, f)


def classifica_hb3(x):
    return _ge(x, LIM_HB3_DENTRO, LIM_HB3_FORA)


def classifica_descritiva(x):
    return "descritivo" if fin(x) is not None else "sem dado"


def classifica_controle_b(procs, orcamento_ok=True, incompleto_motivo=None, blocos_por_pedido=BLOCOS_POR_PEDIDO_MEDIO):
    """Controle D, A, B2 (espelho de perfil_bloco.rs::classifica_controle_f2b com CONTROLE_B_PLANO). Classe pela magnitude de |B2 contra A|; B2 contra D só apoio."""
    def inc(m):
        return {"classe": "controle incompleto", "motivo": m, "principal": None, "apoio": None, "classe_apoio": None}
    if not orcamento_ok:
        return inc("orçamento insuficiente")
    if incompleto_motivo:
        return inc(incompleto_motivo)
    if len(procs) != len(ORDEM_CONTROLE):
        return inc("%d processos de controle em vez de 9 (mínimo de 3 válidos por braço)" % len(procs))
    if [p["braco"] for p in procs] != ORDEM_CONTROLE:
        return inc("ordem diferente de D A B2, B2 D A, A B2 D")
    if any((not p["valido"]) or fin(p["prefill_ms_mediana"]) is None or p["prefill_ms_mediana"] <= 0 or fin(p.get("passo_ms_mediana")) is None or p["passo_ms_mediana"] <= 0 for p in procs):
        return inc("processo de controle abortado ou inválido")
    esp = {"D": (False, False, False), "A": (True, False, False), "B2": (True, True, True)}
    for p in procs:
        e = esp[p["braco"]]
        if (p["imprimiu_bloco"], p["imprimiu_f2"], p["imprimiu_f2b"]) != e or (p["produziu_bloco"], p["produziu_f2"], p["produziu_f2b"]) != e:
            return inc("braço %s com linhas de estado ou arquivos de perfil que não são os do braço" % p["braco"])

    def de(b, k):
        return [p[k] for p in procs if p["braco"] == b]

    def desvio(alvo, ref):
        a, r = de(alvo, "prefill_ms_mediana"), de(ref, "prefill_ms_mediana")
        ma, mr = statistics.median(a), statistics.median(r)
        pa, pr = statistics.median(de(alvo, "passo_ms_mediana")), statistics.median(de(ref, "passo_ms_mediana"))
        return {"desvio": (ma - mr) / mr, "pior_caso": (max(a) - min(r)) / min(r), "ms_por_bloco": (ma - mr) / max(blocos_por_pedido, 1), "desvio_passo": (pa - pr) / pr,
                "mediana_alvo_ms": ma, "mediana_ref_ms": mr}
    pri, apo = desvio("B2", "A"), desvio("B2", "D")
    return {"classe": F.classe_por_desvio(pri["desvio"]), "motivo": "", "principal": pri, "apoio": apo, "classe_apoio": F.classe_por_desvio(apo["desvio"])}


# ---------------------------------------------------------------------------------------------------------------------------------- invariantes IB1 a IB5 (espelho de perfil_bloco.rs)
def bloco_b_forma(b):
    return isinstance(b, dict) and isinstance(b.get("regs"), list) and all(isinstance(r, list) and len(r) == 5 and all(isinstance(x, int) or x is None for x in r) for r in b["regs"]) \
        and all(isinstance(b.get(k), int) for k in ("gen", "m", "pos_ini", "perdidos"))


def esperados_ib3(b, layers, contrato=1):
    return layers * (len(ORDENS[contrato]) * b["m"] + 1)


def ib3(b, layers, contrato=1):
    """Sequência esperada na ordem de emissão: por camada um B_LOCK sem posição e, por posição 0..m-1, os registros da ordem do contrato (v1: 21, v2: 17); nenhum perdido.
    O contrato é escolhido por quem chama (fixtures e dados antigos só pelo v1, novos só pelo v2): bloco de um contrato recusado pelo outro."""
    m = b["m"]
    ordem = ORDENS[contrato]
    if b["perdidos"] > 0 or m <= 0 or len(b["regs"]) != layers * (len(ordem) * m + 1):
        return False
    it = iter(b["regs"])
    for l in range(layers):
        r = next(it)
        if not (r[0] == B_LOCK and r[1] == l and r[2] is None):
            return False
        for p in range(m):
            for op in ordem:
                r = next(it)
                if not (r[0] == op and r[1] == l and r[2] == p):
                    return False
    return True


def host_pre(host, l):
    for o in host["ops"]:
        if o[0] == H_PRE_LAYER and o[1] == l:
            return o
    return None


def dur_regs(b, l, ops=None):
    return sum(max(r[4] - r[3], 0) for r in b["regs"] if r[1] == l and (ops is None or r[0] in ops))


def ib1(b, host):
    regs = b["regs"]
    if not regs:
        return False
    prev = 0
    for r in regs:
        if r[4] < r[3] or r[3] < prev:
            return False
        prev = r[4]
    for l in sorted({r[1] for r in regs}):
        h = host_pre(host, l)
        da = [r for r in regs if r[1] == l]
        if h is None or not da or da[0][3] + TOL_NS < h[2] or da[-1][4] > h[3] + TOL_NS:
            return False
    return True


def residuo_camada(b, host, l):
    h = host_pre(host, l)
    return None if h is None else F.dur(h) - dur_regs(b, l)


def ib2(b, host, layers):
    return all(residuo_camada(b, host, l) is not None and residuo_camada(b, host, l) >= -TOL_NS for l in range(layers))


def razao_residuo(b, host, layers):
    res = sum(r for r in (residuo_camada(b, host, l) for l in range(layers)) if r is not None)
    tot = sum(F.dur(host_pre(host, l)) for l in range(layers) if host_pre(host, l) is not None)
    return (res / tot) if tot > 0 else 0.0


def valida_pedido_b(pfb, pf2, pbloco, layers, contrato=1):
    """Problemas que invalidam o pedido (IB1 a IB4; IB3 invalida o pedido inteiro) e ressalva da IB5. Exige o perfil-f2 e o perfil-bloco do mesmo pedido."""
    prob, ress = [], False
    blocos = pfb.get("blocos") if isinstance(pfb, dict) else None
    hosts = pf2.get("blocos") if isinstance(pf2, dict) else None
    bb = pbloco.get("blocos") if isinstance(pbloco, dict) else None
    if not isinstance(blocos, list) or not isinstance(hosts, list) or not isinstance(bb, list):
        return ["F2B: pedido sem blocos legíveis"], ress
    if pfb.get("layers") != layers:
        prob.append("F2B: layers %r, esperadas %d" % (pfb.get("layers"), layers))
    if len(blocos) != len(hosts) or len(blocos) != len(bb):
        prob.append("F2B: %d blocos de F2B contra %d de F2 e %d do perfil-bloco" % (len(blocos), len(hosts), len(bb)))
        return prob, ress
    if pfb.get("extra"):
        prob.append("F2B: %s registros sem bloco correspondente" % pfb.get("extra"))
    for i, (b, h, r) in enumerate(zip(blocos, hosts, bb)):
        if not isinstance(b, dict) or b.get("sem_registro") or not bloco_b_forma(b):
            prob.append("bloco %d: sem registro de F2B" % i)
            continue
        if h.get("sem_host") or not F.bloco_ok_forma(h):
            prob.append("bloco %d: sem registro de F2 do host para conferir a F2B" % i)
            continue
        if not ib3(b, layers, contrato):
            prob.append("bloco %d: IB3 (contrato v%d, sequência por posição: %d registros, %d perdidos, esperados %d)" % (i, contrato, len(b["regs"]), b["perdidos"], esperados_ib3(b, layers, contrato)))
            continue
        if not ib1(b, h):
            prob.append("bloco %d: IB1 (ordem, sobreposição ou contenção em H_PRE_LAYER)" % i)
        if not ib2(b, h, layers):
            prob.append("bloco %d: IB2 (soma dos intervalos maior que H_PRE_LAYER)" % i)
        if b["gen"] != h.get("gen") or b["m"] != h.get("m") or b["pos_ini"] != h.get("pos_ini") or b["gen"] != r.get("gen") or b["m"] != r.get("m") or b["pos_ini"] != r.get("pos_ini"):
            prob.append("bloco %d: IB4 (gen/m/pos_ini da F2B diferem da F2 ou do perfil-bloco)" % i)
        if not F.i2(h):
            prob.append("bloco %d: IB4 (soma de H_PRE_LAYER e das demais operações mais R_borda contra t3 - t2)" % i)
        if razao_residuo(b, h, layers) > LIM_RESIDUO:
            ress = True
    return prob, ress


# ---------------------------------------------------------------------------------------------------------------------------------- medidas
def pares_elegiveis(pf2, pfb):
    """(bloco F2B, bloco F2 do host) dos blocos cheios (m = 8) do pedido, sem o primeiro bloco."""
    out = []
    for i, (h, b) in enumerate(zip(pf2["blocos"], pfb["blocos"])):
        if i > 0 and h.get("m") == BLOCO and not h.get("sem_host") and isinstance(b, dict) and not b.get("sem_registro"):
            out.append((b, h))
    return out


def medidas_pedido_b(pares, layers):
    if not pares:
        return {}
    pre = lambda b, h, l: F.dur(host_pre(h, l))
    soma_med = lambda f: sum(mediana([f(b, h, l) for b, h in pares]) for l in range(layers))
    den = soma_med(pre)
    parc = lambda ops: F.razao(soma_med(lambda b, h, l: dur_regs(b, l, ops)), den)
    por_camada = [[pre(b, h, l) for b, h in pares] for l in range(layers)]
    med_cam = [mediana(v) for v in por_camada]
    p90 = [F.percentil_posto(v, 0.9) / m for v, m in zip(por_camada, med_cam) if m and m > 0 and F.percentil_posto(v, 0.9) is not None]
    return {"hb1": parc({B_LOCK}), "hb2": parc({B_SETUP, B_GENERICS, B_ATTN_PREP}), "hb3": parc({B_LANCA}), "hb4": parc({B_DTOD}),
            "hb5_p90_sobre_mediana": mediana(p90) if p90 else None,
            "hb5_dispersao_camadas": (max(med_cam) / min(med_cam)) if med_cam and min(med_cam) > 0 else None,
            "h_pre_layer_mediana_camada_ns": med_cam,
            "hb6_blocos": [{"pos_ini": h["pos_ini"], "m": h["m"], "h_pre_layer_total_ns": sum(pre(b, h, l) for l in range(layers))} for b, h in pares]}


def medidas_celula_b(pedidos, layers):
    """HB1 a HB5 por pedido (nunca blocos de pedidos misturados) e mediana entre pedidos."""
    por = [medidas_pedido_b(pares, layers) for pares in pedidos if pares]
    por = [p for p in por if p]
    agrega = lambda k: mediana([p[k] for p in por if p.get(k) is not None]) if any(p.get(k) is not None for p in por) else None
    cam = [mediana([p["h_pre_layer_mediana_camada_ns"][l] for p in por]) for l in range(layers)] if por else []
    return {"hb1": agrega("hb1"), "hb2": agrega("hb2"), "hb3": agrega("hb3"), "hb4": agrega("hb4"), "hb5_p90_sobre_mediana": agrega("hb5_p90_sobre_mediana"),
            "hb5_dispersao_camadas": agrega("hb5_dispersao_camadas"), "h_pre_layer_mediana_camada_ns": cam, "n_pedidos_com_blocos": len(por),
            "hb6_blocos": [{"pedido": i, **x} for i, p in enumerate(por) for x in p["hb6_blocos"]],
            "por_pedido": [{k: v for k, v in p.items() if k not in ("hb6_blocos", "h_pre_layer_mediana_camada_ns")} for p in por]}


def pool_pre_layer(pedidos_f2, layers):
    """Mediana, entre todos os blocos cheios elegíveis, de H_PRE_LAYER por camada (usada só para a comparação entre janelas)."""
    v = [[] for _ in range(layers)]
    for pf in pedidos_f2:
        for h in F.elegiveis(pf):
            for l in range(layers):
                o = host_pre(h, l)
                if o is not None:
                    v[l].append(F.dur(o))
    return [mediana(x) if x else None for x in v]


ARQS_ANTERIOR = []  # arquivos da janela de 11:00 lidos na comparação (com sha256), para registro


def le_anterior(dir_ant, m, layers):
    """Mediana de H_PRE_LAYER por camada do B de 11:00 (F2 sem F2B): perfil-f2.json dos processos B do médio, pedidos medidos. None se ausente ou ilegível."""
    if not dir_ant or not os.path.isdir(os.path.join(dir_ant, "celulas")):
        return None
    peds = []
    ARQS_ANTERIOR.clear()
    for n_ in sorted(os.listdir(os.path.join(dir_ant, "celulas"))):
        if re.match(r"^ctrl-%s-L100-medio-\d\d-B$" % m, n_):
            for raiz, _, a in os.walk(os.path.join(dir_ant, "celulas", n_)):
                if "perfil-f2.json" in a:
                    caminho = os.path.join(raiz, "perfil-f2.json")
                    try:
                        ARQS_ANTERIOR.append({"arquivo": os.path.relpath(caminho, dir_ant), "sha256": hashlib.sha256(open(caminho, "rb").read()).hexdigest()})
                    except OSError:
                        pass
                    j = X.le_json(os.path.join(raiz, "perfil-f2.json"))
                    if isinstance(j, dict) and j.get("tipo") == "perfil-f2" and isinstance(j.get("pedidos"), list):
                        peds += [p for i, p in enumerate(j["pedidos"]) if i > 0 and isinstance(p, dict) and isinstance(p.get("blocos"), list)]
    ok = [p for p in peds if all(F.bloco_ok_forma(b) for b in p["blocos"] if not b.get("sem_host"))]
    return pool_pre_layer(ok, layers) if ok else None


# ---------------------------------------------------------------------------------------------------------------------------------- leitura
def le_f2b_json(d, m, n_medidas, contrato=1):
    arqs = []
    for raiz, _, a in os.walk(d):
        if "perfil-f2b.json" in a:
            arqs.append(os.path.join(raiz, "perfil-f2b.json"))
    if len(arqs) != 1:
        return [], ["braço B2 com %d perfil-f2b.json (esperado 1)" % len(arqs)]
    j = X.le_json(arqs[0])
    if not isinstance(j, dict) or j.get("tipo") != "perfil-f2b" or j.get("versao") != 1 or not isinstance(j.get("pedidos"), list):
        return [], ["perfil-f2b.json ilegível ou fora do esquema (tipo perfil-f2b, versao 1)"]
    if j.get("contrato_b", 1) != contrato:   # arquivo sem o campo é do instrumento de 14:10 (v1)
        return [], ["perfil-f2b.json declara contrato_b %r e o extrator foi chamado com --contrato %d" % (j.get("contrato_b", 1), contrato)]
    peds = j["pedidos"]
    mot = []
    if len(peds) != n_medidas + 1:
        mot.append("perfil-f2b.json com %d entradas, esperadas %d (1 aquecimento e %d medidas)" % (len(peds), n_medidas + 1, n_medidas))
    for i, p in enumerate(peds):
        if not isinstance(p, dict) or p.get("modelo") != m or p.get("indice") != i:
            mot.append("entrada %d do perfil-f2b.json com modelo/índice incoerentes" % i)
    return peds, mot


def tem_arq(d, nome):
    return any(nome in a for _, _, a in os.walk(d))


def le_pasta_b(saida, nome, m, t, n_prev, braco, refs, layers, contrato=1):
    """Pasta de processo: base de f2-extrai.le_pasta (linhas, perfil-bloco, perfil-f2) mais a F2B. `valido_controle` = processo temporalmente válido com as linhas e os arquivos do braço."""
    c = I.le_dir_celula(saida, nome, m, "L100", t, n_prev, refs)
    vazio = {"estado_f2": None, "estado_f2b": None, "tem_f2": False, "tem_f2b": False, "motivos_f2": [], "pedidos_b": [], "aquecimento_b": None, "valido_controle": False, "motivos_pedidos_f2": []}
    if c["status"] == "NÃO RODOU":
        c.update(vazio)
        return c
    d = os.path.join(saida, "celulas", nome)
    est_f2, m1 = F.le_linhas(d, F.LINHA_F2_RE, "perfil-f2")
    est_b, m2 = F.le_linhas(d, LINHA_F2B_RE, "perfil-f2b")
    tem_f2, tem_b = tem_arq(d, "perfil-f2.json"), tem_arq(d, "perfil-f2b.json")
    mot = list(m1) + list(m2)
    esp_f2 = esp_b = "ligado" if braco == "B2" else "desligado"
    esp_bl = "desligado" if braco == "D" else "ligado"
    if est_f2 is not None and est_f2 != esp_f2:
        mot.append("braço %s com `perfil-f2: %s` (esperado %s)" % (braco, est_f2, esp_f2))
    if est_b is not None and est_b != esp_b:
        mot.append("braço %s com `perfil-f2b: %s` (esperado %s)" % (braco, est_b, esp_b))
    if c.get("estado_linha") is not None and c["estado_linha"] != esp_bl:
        mot.append("braço %s com `perfil-bloco: %s` (esperado %s)" % (braco, c["estado_linha"], esp_bl))
    if braco != "B2" and (tem_f2 or tem_b):
        mot.append("braço %s produziu perfil-f2.json ou perfil-f2b.json" % braco)
    pedidos_b, aq_b, mot_p = [], None, []
    if braco == "B2" and est_f2 == "ligado" and est_b == "ligado":
        p2, mj2 = F.le_f2_json(d, m, n_prev, layers)
        pb_, mjb = le_f2b_json(d, m, n_prev, contrato)
        mot += mj2 + mjb
        pbs, arq_ok = None, True
        n_pb = sum(1 for _, _, a in os.walk(d) if "perfil-bloco.json" in a)
        if n_pb != 1:
            arq_ok = False
            mot_p.append("braço B2 com %d perfil-bloco.json (esperado 1)" % n_pb)
        for raiz, _, a in os.walk(d):
            if "perfil-bloco.json" in a:
                jb = X.le_json(os.path.join(raiz, "perfil-bloco.json"))
                if not isinstance(jb, dict) or jb.get("tipo") != "perfil-bloco" or jb.get("versao") != 2 or not isinstance(jb.get("pedidos"), list):
                    arq_ok = False
                    mot_p.append("perfil-bloco.json ilegível ou fora do esquema (tipo perfil-bloco, versao 2): nenhum pedido desta fonte é admitido")
                else:
                    pbs = jb["pedidos"]
        pbs = pbs if isinstance(pbs, list) else []
        for i, pf in enumerate(p2):
            if not arq_ok:
                continue
            if i >= len(pbs) or i >= len(pb_):
                mot_p.append("pedido %d sem perfil-bloco ou perfil-f2b correspondente" % i)
                continue
            prob, hi, _res = F.valida_pedido_f2(pf, pbs[i], layers)  # hi = I8 do nível A: preservada por pedido; só o smoke a usa (como em f2-extrai.py)
            prob = F.problemas_identidade(pbs[i], m, i) + prob
            if i > 0:
                prob = F.problemas_base(pbs[i], t) + prob
            pb_prob, ress = valida_pedido_b(pb_[i], pf, pbs[i], layers, contrato)
            prob = prob + pb_prob
            info = {"indice": i, "problemas": prob, "problemas_hi": hi, "ressalva_residuo": ress, "f2": pf, "f2b": pb_[i]}
            if i == 0:
                aq_b = info
            elif prob:
                mot_p.append("pedido %d inválido: %s" % (i, "; ".join(prob)[:300]))
            else:
                pedidos_b.append(info)
    elif braco == "B2":
        mot.append("braço B2 sem `perfil-f2: ligado` e `perfil-f2b: ligado`")
    c.update({"estado_f2": est_f2, "estado_f2b": est_b, "tem_f2": tem_f2, "tem_f2b": tem_b, "motivos_f2": sorted(set(mot)), "pedidos_b": pedidos_b, "aquecimento_b": aq_b,
              "motivos_pedidos_f2": sorted(set(mot_p))})
    c["valido_controle"] = c["status_temporal"] == "OK" and not mot and bool(c.get("estado_linha"))
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


ORDEM_RE = r"^(\d\d) (ctrl-%s-L100-medio-(\d\d)-(D|A|B2)) n=(\d+) braco=(D|A|B2) inicio=\S+ fim=\S+ rc=(\d+)$"


def confere_ordem(saida, m, nomes_pastas):
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
        return "ordem efetiva de ordem.txt diferente de D A B2, B2 D A, A B2 D com os nomes esperados"
    if any(x["braco"] != x["braco_nome"] for x in linhas):
        return "braço registrado em ordem.txt diferente do braço do nome da pasta"
    if any(x["rc"] != 0 or x["n"] != 3 for x in linhas):
        return "ordem.txt com processo de controle de rc != 0 ou n != 3"
    if sorted(nomes_pastas) != sorted(esperado):
        return "pastas do controle diferentes das 9 esperadas"
    return None


# ---------------------------------------------------------------------------------------------------------------------------------- processamento
HIPS = ("HB1", "HB2", "HB3", "HB4", "HB5", "HB6")


def processa(saida, ids_arq, refs_dir, textos_arq, anterior=None, contrato=1):
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
            if F.le_linhas(os.path.join(saida, "celulas", n_), LINHA_F2B_RE, "perfil-f2b")[0] is not None:
                algum = True
    if not algum:
        raise Recusa("sem a linha `perfil-f2b: ligado|desligado` em nenhum processo: não é saída do binário do nível B")
    integ = sorted(set(X.integridade(saida) + integridade_fixos(saida, m)))
    ref_prob = I.confere_referencias(saida, m, ids_arq, refs_dir, textos_arq)
    nao_rodou = [str(x) for x in (orc.get("nao_rodou") or [])]
    estouros = F.teto_excedido(saida)
    leituras = {}

    def le(nome, t, n, braco):
        if nome not in leituras:
            c = le_pasta_b(saida, nome, m, t, n, braco, refs, layers, contrato)
            if nome in estouros:
                c["valido_controle"] = False
                c["motivos_f2"] = sorted(set(c.get("motivos_f2", []) + ["teto de processo excedido (processo inválido, sem repetição)"]))
            leituras[nome] = c
        return leituras[nome]

    sm_nome = "smoke-%s-L100-curto-B2" % m
    smoke_ok, smoke_mot = False, ["smoke ausente (%s)" % sm_nome]
    if sm_nome in nomes:
        c = le(sm_nome, "curto", 1, "B2")
        mots = list(c.get("motivos") or []) + list(c.get("motivos_f2") or []) + list(c.get("motivos_pedidos_f2") or [])
        if c.get("aquecimento_b") and c["aquecimento_b"]["problemas"]:
            mots.append("aquecimento do smoke: " + "; ".join(c["aquecimento_b"]["problemas"])[:300])
        if c.get("aquecimento_b") and c["aquecimento_b"].get("problemas_hi"):
            mots.append("aquecimento do smoke com I8 falha: " + "; ".join(c["aquecimento_b"]["problemas_hi"])[:200])
        mots += ["smoke com I8 falha: " + "; ".join(p_["problemas_hi"])[:200] for p_ in c.get("pedidos_b", []) if p_.get("problemas_hi")]
        sq = c.get("seq") or {}
        if c.get("status") != "NÃO RODOU" and not sq.get("referencia_disponivel"):
            mots.append("smoke sem referência 5c disponível")
        if sq.get("divergencias"):
            mots.append("smoke com divergência de ids contra a referência 5c (aquecimento incluído): %d" % len(sq["divergencias"]))
        if len(c.get("pedidos_b", [])) != 1 and not mots:
            mots.append("smoke sem o pedido medido válido")
        smoke_ok, smoke_mot = (not mots, "; ".join(mots)[:500])
    ctrl = sorted((int(CTRL_RE.match(n_).group(2)), CTRL_RE.match(n_).group(3), n_) for n_ in nomes if CTRL_RE.match(n_) and CTRL_RE.match(n_).group(1) == m)
    procs = []
    for (nn, b, nome) in ctrl:
        c = le(nome, "medio", 3, b)
        meds = c.get("meds") or []
        procs.append((nn, b, nome, c, {"braco": b, "prefill_ms_mediana": mediana([x["prefill_ms"] for x in meds]), "passo_ms_mediana": I.mediana_passo(meds),
                                       "imprimiu_bloco": c.get("estado_linha") == "ligado", "imprimiu_f2": c.get("estado_f2") == "ligado", "imprimiu_f2b": c.get("estado_f2b") == "ligado",
                                       "produziu_bloco": bool(c.get("tem_perfil")), "produziu_f2": bool(c.get("tem_f2")), "produziu_f2b": bool(c.get("tem_f2b")),
                                       "valido": bool(c.get("valido_controle"))}))
    orc_ok = not any(x.startswith("ctrl-%s-L100-" % m) for x in nao_rodou)
    mot_ordem = confere_ordem(saida, m, [x[2] for x in procs]) if procs else "sem processos de controle"
    nums = [x[0] for x in procs]
    if mot_ordem is None and nums != list(range(1, len(nums) + 1)):
        mot_ordem = "numeração das pastas do controle não é 01..09"
    ctl = classifica_controle_b([x[4] for x in procs], orc_ok, mot_ordem)
    ctl["ordem_efetiva"] = mot_ordem or "confere com ordem.txt (D A B2, B2 D A, A B2 D, rc 0)"
    ctl["processos"] = [{"nn": nn, "braco": b, "pasta": nome, "status_temporal": c["status_temporal"], "estado_bloco": c.get("estado_linha"), "estado_f2": c.get("estado_f2"),
                         "estado_f2b": c.get("estado_f2b"), "tem_perfil_bloco": c.get("tem_perfil"), "tem_perfil_f2": c.get("tem_f2"), "tem_perfil_f2b": c.get("tem_f2b"),
                         "valido": c.get("valido_controle"), "prefill_ms_mediana": pr["prefill_ms_mediana"], "passo_ms_mediana": pr["passo_ms_mediana"],
                         "motivos": sorted(set(c["motivos_temporais"] + c.get("motivos_f2", [])))} for (nn, b, nome, c, pr) in procs]
    div = []
    for (_, b, nome, c, _) in procs:
        div += [dict(x, pasta=nome, braco=b) for x in c["seq"]["divergencias"]]
    celnomes = {t: [n_ for n_ in nomes if CEL_RE.match(n_) and CEL_RE.match(n_).group(1) == m and CEL_RE.match(n_).group(2) == t] for t in ("curto", "longo")}
    ref_ausente = [nome for (_, b, nome, c, _) in procs if c["status"] != "NÃO RODOU" and not c["seq"]["referencia_disponivel"]]
    for t, lst in celnomes.items():
        for nome in lst:
            mj = X.le_json(os.path.join(saida, "celulas", nome, "celula.json")) or {}
            c = le(nome, t, mj.get("n_medidas") if mj.get("n_medidas") in (3, 5) else 5, "B2")
            div += [dict(x, pasta=nome, texto=t) for x in c["seq"]["divergencias"]]
            if c["status"] != "NÃO RODOU" and not c["seq"]["referencia_disponivel"]:
                ref_ausente.append(nome)
    gb = {tuple(x["gerados"]) for (_, b, _, c, _) in procs if b == "B2" for x in c.get("meds") or []}
    gd = {tuple(x["gerados"]) for (_, b, _, c, _) in procs if b == "D" for x in c.get("meds") or []}
    ga = {tuple(x["gerados"]) for (_, b, _, c, _) in procs if b == "A" for x in c.get("meds") or []}
    ids_ok = bool(gb) and bool(gd) and bool(ga) and gb == gd == ga and len(gb) == 1
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
    if bloq and F.controle_permite(ctl["classe"]):
        ctl_final = dict(ctl, classe_calculada=ctl["classe"], classe="SEM PROVA", motivo="bloqueio: " + "; ".join(bloq) + " (as medidas ficam só como descrição)")
    textos = {}
    for t in TEXTOS:
        if t == "medio":
            fontes = [(nome, c) for (_, b, nome, c, _) in procs if b == "B2"]
        else:
            fontes = [(nome, le(nome, t, 5, "B2")) for nome in celnomes[t]]
        resto = N_POS[t] % BLOCO
        if not fontes:
            textos[t] = {"status": "NÃO RODOU", "motivos": [], "celulas": [], "pedidos_validos": 0, "minimo_pedidos": MIN_PEDIDOS[t], "medidas": {}, "hipoteses": {}, "ressalvas": [], "resto": resto}
            continue
        motivos, pedidos, usadas, ress = [], [], [], []
        for nome, c in fontes:
            if c["status"] == "NÃO RODOU":
                motivos.append("%s: não rodou" % nome)
                continue
            if c["status_temporal"] != "OK" or c.get("motivos_f2"):
                motivos += ["%s: %s" % (nome, x) for x in c["motivos_temporais"] + c.get("motivos_f2", [])]
                continue
            if c.get("motivos_perfil"):
                motivos += ["%s: perfil: %s" % (nome, x) for x in c["motivos_perfil"]]
            motivos += ["%s: %s" % (nome, x) for x in c.get("motivos_pedidos_f2", [])]
            usadas.append(nome)
            pedidos += c["pedidos_b"]
        pares = [pares_elegiveis(p["f2"], p["f2b"]) for p in pedidos]
        med = medidas_celula_b(pares, layers) if pedidos else {}
        for p in pedidos:
            if p["ressalva_residuo"]:
                ress.append("B_RESIDUO/H_PRE_LAYER acima de 3%% em bloco do pedido %d (ressalva, não invalida)" % p["indice"])
        enough = len(pedidos) >= MIN_PEDIDOS[t]
        hip = {}
        for h in HIPS:
            if not usadas and not pedidos:
                hip[h] = "SEM PROVA"
            elif bloq or not F.controle_permite(ctl_final["classe"]):
                hip[h] = "SEM PROVA"
            elif not enough:
                hip[h] = "SEM PROVA"
            elif h == "HB3":
                hip[h] = classifica_hb3(med.get("hb3"))
            elif h == "HB5":
                hip[h] = "descritivo" if fin(med.get("hb5_p90_sobre_mediana")) is not None and fin(med.get("hb5_dispersao_camadas")) is not None else "sem dado"
            elif h == "HB6":
                hip[h] = "descritivo" if med.get("h_pre_layer_mediana_camada_ns") else "sem dado"
            else:
                hip[h] = classifica_descritiva(med.get({"HB1": "hb1", "HB2": "hb2", "HB4": "hb4"}[h]))
        if not enough:
            ress.append("mínimo de pedidos válidos não atingido: %d de %d exigidos (SEM PROVA na classificação)" % (len(pedidos), MIN_PEDIDOS[t]))
        estado = "OK" if (enough and not motivos) else ("SEM PROVA" if (usadas or motivos) else "NÃO RODOU")
        textos[t] = {"status": estado, "motivos": sorted(set(motivos)), "celulas": usadas, "pedidos_validos": len(pedidos), "minimo_pedidos": MIN_PEDIDOS[t],
                     "amostras_elegiveis": sum(len(x) for x in pares), "medidas": med, "hipoteses": hip, "ressalvas": ress, "resto": resto}
    # H_PRE_LAYER do B2 ao lado do B de 11:00 (entre janelas)
    b2_peds = [p["f2"] for (_, b, _, c, _) in procs if b == "B2" for p in c.get("pedidos_b", [])]
    cam_b2 = pool_pre_layer(b2_peds, layers) if b2_peds else None
    cam_ant = le_anterior(anterior, m, layers)
    comp = {"b2_mediana_por_camada_ns": cam_b2, "b_1100_mediana_por_camada_ns": cam_ant, "ressalva": "entre janelas", "arquivos_b_1100_lidos": list(ARQS_ANTERIOR),
            "razao_mediana_b2_sobre_b": (mediana([x / y for x, y in zip(cam_b2, cam_ant) if x is not None and y]) if (cam_b2 and cam_ant) else None)}
    return {"tipo": "f2b-host-resultado", "contrato_b": contrato, "criterio": "criterio-f2-nivel-b-20261111.md", "descritivo": True, "modelo_da_janela": m, "layers": layers, "rotulo_b_lanca": ROTULO_LANCA,
            "aviso": "Descritivo: sem veredito geral e sem PASSA. HB1 a HB6 são previsões pré-registradas, só conferidas ('dentro', 'fora', 'sem leitura', 'sem dado', 'SEM PROVA', 'descritivo'); "
                     "B_LANCA é o tempo da chamada de envio de trabalho; o tempo com instrumento não é o tempo de produção.",
            "nota_i8": "I8 do nível A (carimbos do serviço): calculada por pedido e usada só no smoke; fora do smoke não invalida medidas HB do nível B.", "integridade_problemas": integ, "referencia_problemas": ref_prob, "smoke": {"ok": smoke_ok, "motivo": smoke_mot}, "bloqueios": bloq, "controle": ctl_final,
            "ids_B2_igual_D_e_A": ids_ok, "divergencias_5c": div, "textos": textos, "comparacao_h_pre_layer": comp, "orcamento": orc, "teto_excedido": sorted(estouros)}


# ---------------------------------------------------------------------------------------------------------------------------------- relatório
f, pc = F.f, F.pc


def relatorio(res):
    m = res["modelo_da_janela"]
    o = ["# F2 nível B, pré-roteador, janela do modelo %s (criterio-f2-nivel-b-20261111.md)" % m.upper(), "", res["aviso"], ""]
    if res["referencia_problemas"]:
        o += ["**Referência 5c sem prova:**", ""] + ["- %s" % x for x in res["referencia_problemas"]] + [""]
    if res["integridade_problemas"]:
        o += ["**Integridade QUEBRADA: todas as hipóteses SEM PROVA (medidas brutas preservadas só como descrição).**", ""] + ["- %s" % x for x in res["integridade_problemas"]] + [""]
    o += ["Smoke (1 processo no curto, em B2): **%s**%s" % ("OK" if res["smoke"]["ok"] else "FALHOU", "" if res["smoke"]["ok"] else " (%s)" % res["smoke"]["motivo"]), ""]
    c = res["controle"]
    o += ["## Controle do custo do instrumento (D, A, B2)", "", "Classe (|B2 contra A|): **%s**%s" % (c["classe"], (" (%s)" % c["motivo"]) if c.get("motivo") else "")]
    if c.get("classe_calculada"):
        o.append("- classe calculada antes do bloqueio: %s" % c["classe_calculada"])
    o.append("- ordem efetiva (ordem.txt): %s" % c.get("ordem_efetiva"))
    for rot, k in (("B2 contra A (principal)", "principal"), ("B2 contra D (apoio)", "apoio")):
        d = c.get(k)
        if d:
            o.append("- %s: desvio de prefill_ms %s; pior caso %s; diferença de prefill por bloco executado, incluindo primeiro e resto, %s ms (%d blocos); desvio do passo de decode %s" % (
                rot, pc(d["desvio"]), pc(d["pior_caso"]), f(d["ms_por_bloco"], 3), BLOCOS_POR_PEDIDO_MEDIO, pc(d["desvio_passo"])))
    if c.get("classe_apoio"):
        o.append("- classe descritiva de B2 contra D: %s (não decide)" % c["classe_apoio"])
    for p in c.get("processos", []):
        o.append("  - %02d %s %s: temporal %s, bloco %s, f2 %s, f2b %s, arquivos bloco/f2/f2b %s/%s/%s, válido %s, prefill mediano %s ms, passo %s ms%s" % (
            p["nn"], p["braco"], p["pasta"], p["status_temporal"], p["estado_bloco"], p["estado_f2"], p["estado_f2b"], "sim" if p["tem_perfil_bloco"] else "não", "sim" if p["tem_perfil_f2"] else "não",
            "sim" if p["tem_perfil_f2b"] else "não", p["valido"], f(p["prefill_ms_mediana"]), f(p["passo_ms_mediana"]), ("; " + "; ".join(p["motivos"])) if p["motivos"] else ""))
    o.append("- ids de B2 iguais aos de D e de A (I7, processos do controle): %s" % {True: "sim", False: "não"}[bool(res["ids_B2_igual_D_e_A"])])
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
            o.append("- parcelas de H_PRE_LAYER (mediana por camada somada): B_LOCK %s (mede o custo de pegar uma trava livre); B_SETUP + B_GENERICS + B_ATTN_PREP %s; B_LANCA %s (%s); B_DTOD %s" % (
                pc(md.get("hb1")), pc(md.get("hb2")), pc(md.get("hb3")), res["rotulo_b_lanca"], pc(md.get("hb4"))))
            o.append("- uniformidade: p90/mediana por camada %s; maior/menor mediana entre camadas %s" % (f(md.get("hb5_p90_sobre_mediana"), 3), f(md.get("hb5_dispersao_camadas"), 3)))
        o += ["", "| HB1 | HB2 | HB3 | HB4 | HB5 | HB6 |", "|---|---|---|---|---|---|"]
        if tx["hipoteses"]:
            o.append("| " + " | ".join(tx["hipoteses"][h] for h in HIPS) + " |")
        for r in tx["ressalvas"]:
            o.append("- ressalva: " + r)
        for x in tx["motivos"]:
            o.append("- %s" % x)
        o.append("")
    cp = res["comparacao_h_pre_layer"]
    o += ["## H_PRE_LAYER por camada: B2 ao lado do B de 11:00 (entre janelas)", ""]
    if cp["b2_mediana_por_camada_ns"] and cp["b_1100_mediana_por_camada_ns"]:
        o.append("razão mediana B2/B entre as camadas: %s (janelas diferentes; o B2 já inclui o custo do instrumento novo)" % f(cp["razao_mediana_b2_sobre_b"], 3))
        o.append("")
        o.append("| camada | B2 (ns) | B de 11:00 (ns) |")
        o.append("|---|---|---|")
        for l, (x, y) in enumerate(zip(cp["b2_mediana_por_camada_ns"], cp["b_1100_mediana_por_camada_ns"])):
            o.append("| %d | %s | %s |" % (l, f(x, 0), f(y, 0)))
    else:
        o.append("sem dado: B2 ou B de 11:00 ausente (`--f2-anterior` não informado ou ilegível).")
    o.append("")
    orc = res.get("orcamento") or {}
    o += ["## Orçamento", "", "decorrido %s s de %s s; cortes: %s; não rodou: %s; teto de processo excedido: %s" % (
        orc.get("decorrido_s", "?"), orc.get("limite_s", "?"), "; ".join(orc.get("cortes") or []) or "nenhum", "; ".join(orc.get("nao_rodou") or []) or "nada",
        ", ".join(res.get("teto_excedido") or []) or "nenhum"), "", "## Conclusão", "", "Este relatório é descritivo e não declara veredito; qualquer mudança de desempenho no motor exige nova decisão do Breno."]
    return "\n".join(o) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="extrator da F2 nível B")
    ap.add_argument("--saida")
    ap.add_argument("--ids", default=os.path.join(AQUI, "cmp-ids-prompt.json"))
    ap.add_argument("--refs", default=os.path.join(AQUI, "5c-referencia"))
    ap.add_argument("--textos", default=os.path.join(AQUI, "5c-textos", "textos.json"))
    ap.add_argument("--f2-anterior", default=None)
    ap.add_argument("--contrato", type=int, choices=(1, 2), default=1, help="contrato de registros do nível B: 1 = instrumento de 14:10 (padrão, dados antigos), 2 = instrumento do plano de cópias")
    a = ap.parse_args(argv)
    if not a.saida or not os.path.isdir(a.saida):
        print("pasta de saída inexistente: %s" % a.saida, file=sys.stderr)
        return 2
    try:
        res = processa(a.saida, a.ids, a.refs, a.textos, a.f2_anterior, a.contrato)
    except Recusa as ex:
        print("f2b-extrai: RECUSADO: %s" % ex, file=sys.stderr)
        with open(os.path.join(a.saida, "f2b-host-resultado.json"), "w", encoding="utf-8") as fo:
            json.dump({"tipo": "f2b-host-resultado", "recusado": True, "motivo": str(ex)}, fo, ensure_ascii=False, indent=1)
        with open(os.path.join(a.saida, "f2b-host-resultado.md"), "w", encoding="utf-8") as fo:
            fo.write("# F2 nível B: RECUSADO\n\n%s\n\nNenhuma classificação foi feita; os dados brutos ficam na pasta.\n" % ex)
        return 2
    X.limpa_nao_finitos(res)
    with open(os.path.join(a.saida, "f2b-host-resultado.json"), "w", encoding="utf-8") as fo:
        json.dump(res, fo, ensure_ascii=False, indent=1, allow_nan=False)
    with open(os.path.join(a.saida, "f2b-host-resultado.md"), "w", encoding="utf-8") as fo:
        fo.write(relatorio(res))
    print("f2b-extrai: controle %s; gravados f2b-host-resultado.json e f2b-host-resultado.md" % res["controle"]["classe"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
