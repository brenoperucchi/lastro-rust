#!/usr/bin/env python3
"""E2E sem GPU do f2d-extrai.py: janela sintética de 1B (9 processos em quadrado latino, contrato v3 de 15 registros por posição), main -> processa -> f2d-resultado.json/.md.
Cobre: sequência de 15 registros no aquecimento de cada B2_O3 (faz o papel do smoke), I8 e ids do aquecimento, recusa de bloco v1 e v2 no contrato v3 e vice-versa, limites HC3/HC4/HC6, ordem, binários.
    python3 -I test-f2c-extrai-e2e.py      (mkdtemp; apaga a pasta-mãe só se tudo passar; --guarda mantém)
"""
import copy
import hashlib
import importlib.util
import json
import os
import sys
import shutil
import tempfile

AQUI = os.path.dirname(os.path.abspath(__file__))
GUARDA = "--guarda" in sys.argv   # --guarda: mantém a pasta-mãe dos cenários mesmo se tudo passar
RAIZ_TESTE = tempfile.mkdtemp(prefix="f2d-e2e-")   # UMA pasta-mãe por execução; todo mkdtemp seguinte cai dentro dela
tempfile.tempdir = RAIZ_TESTE
s = importlib.util.spec_from_file_location("c", os.path.join(AQUI, "f2d-extrai.py"))
C = importlib.util.module_from_spec(s)
s.loader.exec_module(C)
B, F, X = C.B, C.F, C.X
falhas = 0


def confere(nome, cond):
    global falhas
    print(("ok: " if cond else "FALHOU: ") + nome)
    if not cond:
        falhas += 1


T2, REG, LAY, G = 1_000_000, 20, 24, list(range(64))
V3 = B.ORDEM_POS_V3
V2 = B.ORDEM_POS_V2
V1 = B.ORDEM_POS_V1


def pre_dur(m, n=len(V3)):
    return (n * m + 1) * REG + 10   # F2D: o resíduo fixo cai de 20 para 10 ns por camada e bloco, para a razão do V3 (15 por posição) ficar abaixo de 3% como a do V2


def host(m, gen, pos_ini, t2, pre=None):
    durs = {(F.H_PRE_LAYER, l): (pre if pre is not None else pre_dur(m)) for l in range(LAY)}
    t = t2 + 10
    entrada, ops = t, []
    for (op, l) in F.ordem_esperada(LAY):
        d = durs.get((op, l), 100)
        ops.append([op, l, t, t + d])
        t += d
    b = {"gen": gen, "m": m, "pos_ini": pos_ini, "t2_ns": t2, "t3_ns": t + 10, "entrada_ns": entrada, "saida_ns": t, "ops": ops}
    b["tb"] = [{"camada": l, "gen": gen, "tb_obs_ns": next(o[2] for o in ops if o[0] == F.H_SEQ and o[1] == l) + 5, "tb_ns": next(o[3] for o in ops if o[0] == F.H_SEQ and o[1] == l) + 20} for l in range(LAY)]
    return b


def regs(h, ordem):
    out = []
    for l in range(LAY):
        pre = next(o for o in h["ops"] if o[0] == F.H_PRE_LAYER and o[1] == l)
        t = pre[2] + 10
        out.append([B.B_LOCK, l, None, t, t + REG])
        t += REG
        for p in range(h["m"]):
            for op in ordem:
                out.append([op, l, p, t, t + REG])
                t += REG
    return {"gen": h["gen"], "m": h["m"], "pos_ini": h["pos_ini"], "regs": out, "perdidos": 0}


def perfis(n_meas, ordem, pre):
    plano = F.I.plano_blocos(F.N_POS["medio"])
    pbs, f2s, fbs = [], [], []
    for i in range(n_meas + 1):
        hs, pbb, pos = [], [], 0
        for bi, mb in enumerate(plano):
            hh = host(mb, bi, pos, T2 + bi * 10_000_000, pre=pre(mb))
            pos += mb
            hs.append(hh)
            pbb.append({"m": mb, "conserva": True, "gen": hh["gen"], "pos_ini": hh["pos_ini"], "t_ns": [0, 0, hh["t2_ns"], hh["t3_ns"], 0]})
        pbs.append({"modelo": "1b", "indice": i, "aquecimento": i == 0, "n": F.N_POS["medio"], "plano_m": plano, "blocos": pbb, "valido": True, "problemas": []})
        f2s.append({"modelo": "1b", "indice": i, "layers": LAY, "blocos": hs})
        fbs.append({"modelo": "1b", "indice": i, "layers": LAY, "blocos": [regs(hh, ordem) for hh in hs], "extra": 0})
    return pbs, f2s, fbs


def sha(x):
    return hashlib.sha256(x).hexdigest()


def fake_le_celula_lastro(dd, meta, refs):
    meds = []
    for pn in sorted(x for x in os.listdir(dd) if x.startswith("p") and x[1:].isdigit()):
        mj = X.le_json(os.path.join(dd, pn, "saida", "medidas-%s.json" % meta["modelo"]))
        for e in mj["medidas"]:
            if not e.get("descartada"):
                meds.append({"processo": pn, "rotulo": e["rotulo"], "prefill_ms": e["prefill_ms"], "mediana_passos_descritiva": e.get("passo_ms", 1.0), "gerados": e["gerados"], "ids_n": len(e["ids"])})
    return {"meds": meds, "cargas_limite_ms": [], "esperado_n": None, "temp_max": None, "slots_impressos": []}, []


X.le_celula_lastro = fake_le_celula_lastro
F.I.X.le_celula_lastro = fake_le_celula_lastro
BASE, CAND = "a" * 64, "b" * 64


def monta(d, pre_novo=100.0, pre_b2=100.0, passo_novo=1.0, ordem_b2=None, ordem=None, bin_por_braco=None, contrato_b=3, pre_host=None, prova_extra=None):
    w = lambda p, txt: (os.makedirs(os.path.dirname(p), exist_ok=True), open(p, "w", encoding="utf-8").write(txt))
    ids = list(range(F.N_POS["medio"]))
    w(os.path.join(d, "ids.json"), json.dumps({"modelos": {"1b": {t: {"ids": list(range(F.N_POS[t]))} for t in F.TEXTOS}}}))
    w(os.path.join(d, "refs", "ref5c-1b.json"), json.dumps({"casos": [{"nome": t, "gerados": G} for t in F.TEXTOS]}))
    w(os.path.join(d, "textos.json"), json.dumps({"textos": []}))
    obs = {"cmp-ids-prompt": sha(open(os.path.join(d, "ids.json"), "rb").read()), "ref5c-1b": sha(open(os.path.join(d, "refs", "ref5c-1b.json"), "rb").read()),
           "textos-json": sha(open(os.path.join(d, "textos.json"), "rb").read())}
    for k in C.FIXO_CHAVES + ("tokenizer-1b",):
        obs.setdefault(k, sha(k.encode()))
    obs["binario-base"], obs["binario-cand"] = BASE, CAND
    sj = os.path.join(d, "janela")
    extra = {"config-1b": sha(b"config-1b"), "safetensors-1b": sha(b"safetensors-1b")}   # só em integridade-antes/depois (como no roteiro real), fora do manifesto FIXO
    w(os.path.join(sj, "integridade-antes.txt"), "".join("%s %s\n" % kv for kv in list(obs.items()) + list(extra.items())))
    w(os.path.join(sj, "integridade-depois.txt"), "".join("%s %s\n" % kv for kv in list(obs.items()) + list(extra.items())))
    w(os.path.join(sj, "teto-excedido.txt"), "")
    w(os.path.join(sj, "integridade-fixos.txt"), "".join("%s %s %s OK\n" % (k, h_, h_) for k, h_ in obs.items()) + "potencia 370.00_W 370.00_W OK\n")
    w(os.path.join(sj, "ambiente.txt"), "uuid=GPU-b22d9465-8552-9f97-812e-380b590645f8\npotencia_inicio=370.00_W\npotencia_fim=370.00_W\nbinario_base=%s\nbinario_cand=%s\n" % (BASE, CAND))
    w(os.path.join(sj, "orcamento.json"), json.dumps({"modelos": ["1b"], "nao_rodou": [], "cortes": [], "decorrido_s": 500, "limite_s": 1200}))
    seq = ordem or C.ORDEM_CONTROLE
    w(os.path.join(sj, "ordem.txt"), "".join("%02d ctrl-1b-L100-medio-%02d-%s n=3 braco=%s inicio=%ds fim=%ds rc=0\n" % (k + 1, k + 1, b, b, 20 * k, 20 * k + 19) for k, b in enumerate(seq)))
    w(os.path.join(sj, "prova-variavel.txt"), "".join("ctrl-1b-L100-medio-%02d-%s braco=%s bin=%s env=ok\n" % (k + 1, b, b, (bin_por_braco or {}).get(b, BASE if b == "DO1" else CAND)) for k, b in enumerate(seq)))
    if prova_extra:
        open(os.path.join(sj, "prova-variavel.txt"), "a").write(prova_extra)
    for k, b in enumerate(seq):
        nome = "ctrl-1b-L100-medio-%02d-%s" % (k + 1, b)
        pd = os.path.join(sj, "celulas", nome)
        pre = {"DO1": 100.0, "DO3": pre_novo, "B2O3": pre_b2}[b]
        passo = {"DO1": 1.0, "DO3": passo_novo, "B2O3": passo_novo}[b]
        w(os.path.join(pd, "celula.json"), json.dumps({"lado": "lastro", "serie": "L100", "modelo": "1b", "texto": "medio", "estado": "quente", "n_medidas": 3}))
        lig = b == "B2O3"
        w(os.path.join(pd, "p1", "stdout.txt"), "perfil-bloco: %s\nperfil-f2: %s\nperfil-f2b: %s\n" % (("ligado",) * 3 if lig else ("desligado",) * 3))
        med = [{"rotulo": "r%d" % i, "descartada": i == 0, "falha": None, "gerados": G, "ids": ids, "prefill_ms": pre, "passo_ms": passo} for i in range(4)]
        w(os.path.join(pd, "p1", "saida", "medidas-1b.json"), json.dumps({"modelo": "1b", "texto": "medio", "medidas": med}))
        if lig:
            pbs, f2s, fbs = perfis(3, ordem_b2 or V3, pre_host or (lambda mb: None))
            w(os.path.join(pd, "p1", "saida", "perfil-bloco.json"), json.dumps({"tipo": "perfil-bloco", "versao": 2, "pedidos": pbs}))
            w(os.path.join(pd, "p1", "saida", "perfil-f2.json"), json.dumps({"tipo": "perfil-f2", "versao": 1, "pedidos": f2s}))
            doc = {"tipo": "perfil-f2b", "versao": 1, "pedidos": fbs}
            if contrato_b is not None:
                doc["contrato_b"] = contrato_b
            w(os.path.join(pd, "p1", "saida", "perfil-f2b.json"), json.dumps(doc))
    return sj


def roda(sj):
    b = os.path.dirname(sj)
    rc = C.main(["--saida", sj, "--ids", os.path.join(b, "ids.json"), "--refs", os.path.join(b, "refs"), "--textos", os.path.join(b, "textos.json")])
    return rc, json.load(open(os.path.join(sj, "f2d-resultado.json"))), open(os.path.join(sj, "f2d-resultado.md"), encoding="utf-8").read()


def novo(prefixo, **kw):
    return monta(tempfile.mkdtemp(prefix="f2d-e2e-" + prefixo + "."), **kw)


def sem_prova(R):
    return R["comparacao"]["HC3"] == R["comparacao"]["HC4"] == R["comparacao"]["HC6"] == "SEM PROVA"


rc, R, md = roda(novo("base"))
confere("base: código 0, sem bloqueios nem motivos, ordem confere, HC1 dentro, HC3/HC4 dentro, HC6 comprovada, nenhuma regressão",
        rc == 0 and R["bloqueios"] == [] and R["motivos_controle"] == [] and R["integridade_problemas"] == [] and R["HC1"] == "dentro" and R["comparacao"]["HC3"] == "dentro do limite"
        and R["comparacao"]["HC4"] == "dentro do limite" and R["comparacao"]["HC6"] == "comprovada" and R["regressao"] == [])
confere("base: HC1b SEM PROVA sem os arquivos do gancho (coleta é outra janela); HC2a descritivo; HC5 comprovada; relatório sem 'causa'", R["HC1b"]["classe"] == "SEM PROVA" and R["HC2a"] == "descritivo" and R["HC5"] == "comprovada"
        and "causa" not in md.lower() and len(R["h_pre_layer_mediana_camada_ns_b2_o3"]) == LAY)
rc, R, md = roda(novo("pref", pre_novo=101.6))
confere("D_O3 com prefill +1,6%: HC3 REGRESSÃO", R["comparacao"]["HC3"] == "REGRESSÃO" and "HC3" in R["regressao"])
rc, R, md = roda(novo("pref2", pre_novo=101.5))
confere("D_O3 com prefill +1,5% exato: dentro do limite", R["comparacao"]["HC3"] == "dentro do limite")
rc, R, md = roda(novo("efq", pre_novo=97.0))
confere("D_O3 com prefill -3% (3 medianas todas abaixo das de D_O1): efeito distinguível no prefill, decode sem efeito, nenhuma regressão, relatório traz os dois rótulos",
        R["comparacao"]["efeito_prefill"]["classe"] == "efeito distinguível" and R["comparacao"]["efeito_decode"]["classe"] == "sem efeito distinguível" and R["regressao"] == []
        and "efeito no prefill" in md and "efeito distinguível" in md and "sem efeito distinguível" in md)
rc, R, md = roda(novo("efn", pre_novo=99.5))
confere("D_O3 com prefill -0,5%: sem efeito distinguível nas duas métricas", R["comparacao"]["efeito_prefill"]["classe"] == "sem efeito distinguível" and R["comparacao"]["efeito_decode"]["classe"] == "sem efeito distinguível")
rc, R, md = roda(novo("efr", pre_novo=101.6))
confere("D_O3 com prefill +1,6%: efeito REGRESSÃO no prefill", R["comparacao"]["efeito_prefill"]["classe"] == "REGRESSÃO")
rc, R, md = roda(novo("dec", passo_novo=1.03))
confere("D_O3 com decode +3%: HC4 REGRESSÃO", R["comparacao"]["HC4"] == "REGRESSÃO" and "HC4" in R["regressao"])
rc, R, md = roda(novo("b2", pre_b2=120.0))
confere("B2_O3 20% acima de D_O3: HC6 SEM PROVA e conta como regressão", R["comparacao"]["HC6"] == "SEM PROVA" and "HC6" in R["regressao"])
# bloco v1 (21 registros) e bloco v2 (17, o da O1) num instrumento declarado v3: o aquecimento e todos os pedidos falham IB3; tudo SEM PROVA
rc, R, md = roda(novo("v1em3", ordem_b2=V1))
confere("bloco v1 (21 por posição) no contrato v3: recusado por IB3, controle inválido, todas as H SEM PROVA", sem_prova(R) and any("IB3" in x for x in R["motivos_controle"]))
rc, R, md = roda(novo("v2em3", ordem_b2=V2))
confere("bloco v2 (17 por posição, com os 2 B_DTOD da O1) no contrato v3: recusado por IB3, controle inválido, todas as H SEM PROVA", sem_prova(R) and any("IB3" in x for x in R["motivos_controle"]))
# vice-versa: bloco v3 lido pelos contratos v1 e v2 (função de validação do f2b) e arquivo sem contrato_b ou com contrato_b 1 ou 2 lido como v3
h_ = host(8, 0, 0, T2)
b3 = regs(h_, V3)
confere("bloco v3 (15 por posição) pelos contratos v1 e v2: IB3 falha; pelo v3 passa", not B.ib3(b3, LAY, 1) and not B.ib3(b3, LAY, 2) and B.ib3(b3, LAY, 3))
b2_ = regs(host(8, 0, 0, T2, pre=pre_dur(8, 17)), V2)
confere("bloco v2 pelo contrato v3: IB3 falha; pelo v2 passa", not B.ib3(b2_, LAY, 3) and B.ib3(b2_, LAY, 2))
b1 = regs(host(8, 0, 0, T2, pre=pre_dur(8, 21)), V1)
confere("bloco v1 pelo contrato v3: IB3 falha; pelo v1 passa", not B.ib3(b1, LAY, 3) and B.ib3(b1, LAY, 1))
rc, R, md = roda(novo("semcb", contrato_b=None))
confere("perfil-f2b.json sem contrato_b (é v1) lido no contrato v3: recusado, tudo SEM PROVA", sem_prova(R) and R["motivos_controle"] != [])
rc, R, md = roda(novo("cb1", contrato_b=1))
confere("perfil-f2b.json declarando contrato_b 1 no contrato v3: recusado, tudo SEM PROVA", sem_prova(R) and R["motivos_controle"] != [])
rc, R, md = roda(novo("cb2", contrato_b=2))
confere("perfil-f2b.json declarando contrato_b 2 no contrato v3: recusado, tudo SEM PROVA", sem_prova(R) and R["motivos_controle"] != [])
# aquecimento (papel do smoke): I8 e ids
def altera(sj, f):
    for k, b in enumerate(C.ORDEM_CONTROLE):
        if b == "B2O3":
            f(os.path.join(sj, "celulas", "ctrl-1b-L100-medio-%02d-%s" % (k + 1, b), "p1", "saida"))
            return
sj = novo("i8aq")
def f_i8(pd):
    p = os.path.join(pd, "perfil-f2.json")
    j = json.load(open(p))
    for bb in j["pedidos"][0]["blocos"]:
        for c in bb["tb"]:
            c["tb_obs_ns"] = 0
    json.dump(j, open(p, "w"))
altera(sj, f_i8)
rc, R, md = roda(sj)
confere("I8 do aquecimento de um B2_O3 alterada: motivo I8 explícito, controle incompleto, todas as H SEM PROVA", sem_prova(R) and any("I8" in x for x in R["motivos_controle"]))
sj = novo("idaq")
def f_id(pd):
    p = os.path.join(pd, "medidas-1b.json")
    j = json.load(open(p))
    j["medidas"][0]["ids"] = [x + 1 for x in j["medidas"][0]["ids"]]
    json.dump(j, open(p, "w"))
altera(sj, f_id)
rc, R, md = roda(sj)
confere("um id do aquecimento alterado: divergência registrada (HC1 SEM PROVA ou REGRESSÃO), nunca 'dentro'", R["HC1"] != "dentro" and (R["divergencias_5c"] != [] or sem_prova(R)))
sj = novo("ib3aq")
def f_ib3(pd):
    p = os.path.join(pd, "perfil-f2b.json")
    j = json.load(open(p))
    for bl in j["pedidos"][0]["blocos"]:
        for r in bl["regs"]:
            if r[0] != B.B_LOCK:
                r[2] = 0
    json.dump(j, open(p, "w"))
altera(sj, f_ib3)
rc, R, md = roda(sj)
confere("sequência de 15 registros do aquecimento com posições zeradas (totais iguais): IB3 falha, tudo SEM PROVA", sem_prova(R) and any("IB3" in x for x in R["motivos_controle"]))
# ordem e binários
rc, R, md = roda(novo("ord", ordem=["DO1", "DO3", "B2O3", "DO1", "DO3", "B2O3", "DO1", "DO3", "B2O3"]))
confere("ordem fora do quadrado latino de 3x3 do roteiro: recusada", sem_prova(R) and any("ordem" in x for x in R["motivos_controle"]))
rc, R, md = roda(novo("bin", bin_por_braco={"DO3": BASE}))
confere("D_O3 rodou com o binário base: integridade quebrada, tudo SEM PROVA", sem_prova(R) and any("binário" in x for x in R["integridade_problemas"]))
sj = novo("igual")
txt = open(os.path.join(sj, "ambiente.txt")).read().replace(CAND, BASE)
open(os.path.join(sj, "ambiente.txt"), "w").write(txt)
open(os.path.join(sj, "prova-variavel.txt"), "w").write(open(os.path.join(sj, "prova-variavel.txt")).read().replace(CAND, BASE))
rc, R, md = roda(sj)
confere("binário base e candidato com o mesmo sha: recusado, tudo SEM PROVA", sem_prova(R) and any("mesmo sha" in x for x in R["integridade_problemas"]))
d = tempfile.mkdtemp(prefix="f2d-e2e-sem.")
os.makedirs(os.path.join(d, "janela", "celulas"))
confere("sem orcamento.json: RECUSADO (código 2) e resultado marcado recusado", C.main(["--saida", os.path.join(d, "janela")]) == 2 and json.load(open(os.path.join(d, "janela", "f2d-resultado.json")))["recusado"] is True)

# ---- rev-1 P2-2: B2_O3 inválido com motivos estruturais de pedido ou sem os 3 pedidos medidos válidos
def zera_pedidos(sj, so_primeiro=False, indices=(1, 2, 3)):
    feitos = 0
    for k, b in enumerate(C.ORDEM_CONTROLE):
        if b != "B2O3" or (so_primeiro and feitos):
            continue
        feitos += 1
        p_ = os.path.join(sj, "celulas", "ctrl-1b-L100-medio-%02d-%s" % (k + 1, b), "p1", "saida", "perfil-f2b.json")
        j = json.load(open(p_))
        for i in indices:
            for bl in j["pedidos"][i]["blocos"]:
                for r in bl["regs"]:
                    if r[0] != B.B_LOCK:
                        r[2] = 0
        json.dump(j, open(p_, "w"))
sj = novo("p2todos")
zera_pedidos(sj)
rc, R, md = roda(sj)
confere("todos os pedidos medidos dos 3 B2_O3 com IB3 falha: controle inválido, tudo SEM PROVA, motivo IB3 preservado, sem HC6 comprovada", sem_prova(R) and R["comparacao"]["HC6"] != "comprovada" and any("IB3" in x for x in R["motivos_controle"]) and R["HC1"] == "SEM PROVA")
sj = novo("p2um")
zera_pedidos(sj, so_primeiro=True, indices=(2,))
rc, R, md = roda(sj)
confere("um único pedido medido inválido num B2_O3 (2 de 3 válidos): controle deixa de ser completo, tudo SEM PROVA, motivo IB3", sem_prova(R) and any("2 de 3 pedidos" in x and "IB3" in x for x in R["motivos_controle"]))
rc, R, md = roda(novo("ib5", pre_host=lambda mb: pre_dur(mb) + 20000))
confere("IB5 acima de 3% segue ressalva: controle completo, HC5 ressalva, HC3/HC4 classificadas", R["motivos_controle"] == [] and R["HC5"] == "ressalva" and R["comparacao"]["HC3"] == "dentro do limite")

# ---- rev-1 P2-3: prova de binário por processo canônico
sj = novo("trocaba")
pv = os.path.join(sj, "prova-variavel.txt")
txt = open(pv).read().replace("ctrl-1b-L100-medio-02-DO3 braco=DO3 bin=" + CAND, "ctrl-1b-L100-medio-02-DO3 braco=DO1 bin=" + BASE)
assert "braco=DO1 bin=" + BASE in txt.split("\n")[1]
open(pv, "w").write(txt)
rc, R, md = roda(sj)
confere("processo DO3 com linha declarando braco=DO1 e o sha base: integridade quebrada, tudo SEM PROVA", sem_prova(R) and any("não coincidem" in x for x in R["integridade_problemas"]))
sj = novo("nove")
pv = os.path.join(sj, "prova-variavel.txt")
l1 = open(pv).read().split("\n")[0]
open(pv, "w").write((l1 + "\n") * 9)
rc, R, md = roda(sj)
confere("nove repetições da mesma linha de prova: duplicatas e ausentes recusadas", sem_prova(R) and any("duplicada" in x for x in R["integridade_problemas"]) and any("sem prova de binário" in x for x in R["integridade_problemas"]))
rc, R, md = roda(novo("estranho", prova_extra="ctrl-3b-L100-medio-01-DO1 braco=DO1 bin=%s env=ok\n" % BASE))
confere("prova de processo de outro modelo: recusada", sem_prova(R) and any("estranho ou de outro modelo" in x for x in R["integridade_problemas"]))
sj = novo("ambman")
am = os.path.join(sj, "ambiente.txt")
t_am = open(am).read().replace("binario_cand=" + CAND, "binario_cand=" + "c" * 64)
open(am, "w").write(t_am)
rc, R, md = roda(sj)
confere("ambiente.txt com sha de binário diferente do manifesto de integridade: recusado", sem_prova(R) and any("manifesto de integridade" in x for x in R["integridade_problemas"]))

# ---- rev-1 P2-1: HC1b com envelope, cobertura e procedência
HEXN = lambda x: hashlib.sha256(str(x).encode()).hexdigest()


def doc_gancho(modelo="1b", n_rep=3, layers=24, ids=None, gerados=None, mut=None):
    ids = list(range(F.N_POS["medio"])) if ids is None else ids
    gerados = G if gerados is None else gerados
    peds = [{"indice": i, "aquecimento": i == 0, "ids": ids, "gerados": gerados, "logits": [HEXN(("l", i, t)) for t in range(len(gerados))], "kv": [[HEXN(("k", i, l)), HEXN(("v", i, l))] for l in range(layers)]} for i in range(n_rep + 1)]
    d = {"tipo": "gancho-logits", "versao": 1, "variavel": "LASTRO_GANCHO_LOGITS", "modelo": modelo, "pedidos": peds}
    if mut:
        mut(d)
    return d


def monta_coleta(prefixo, mut_o=None, mut_c=None, mesmo_arquivo=False, manifesto_extra=None, n_rep=3, bin_c=None, par=None, mut_arq=None):
    d = tempfile.mkdtemp(prefix="f2d-coleta-" + prefixo + ".")
    sj = monta(d)
    cj = os.path.join(d, "coleta")
    os.makedirs(os.path.join(cj, "gancho-original"))
    os.makedirs(os.path.join(cj, "gancho-candidato"))
    pb, pc = par or (BASE, CAND)
    for nm in ("integridade-antes.txt", "integridade-depois.txt", "integridade-fixos.txt", "ambiente.txt"):
        t_ = open(os.path.join(sj, nm)).read().replace(BASE, "@B@").replace(CAND, "@C@").replace("@B@", pb).replace("@C@", pc)
        open(os.path.join(cj, nm), "w").write(t_)
    ao, ac = os.path.join(cj, "gancho-original", "gancho-logits-1b.json"), os.path.join(cj, "gancho-candidato", "gancho-logits-1b.json")
    json.dump(doc_gancho(n_rep=n_rep, mut=mut_o), open(ao, "w"))
    if mesmo_arquivo:
        os.link(ao, ac)
    else:
        json.dump(doc_gancho(n_rep=n_rep, mut=mut_c), open(ac, "w"))
    shf = lambda a: hashlib.sha256(open(a, "rb").read()).hexdigest()
    ids_sha = open(os.path.join(sj, "integridade-antes.txt")).read()
    ids_sha = dict(x.split() for x in ids_sha.splitlines())
    man = {"modelo": "1b", "n_repete": n_rep, "binario_original": pb, "binario_candidato": bin_c or pc, "gancho_original_sha": shf(ao), "gancho_candidato_sha": shf(ac),
           "ids_sha": ids_sha["cmp-ids-prompt"], "refs_sha": ids_sha["ref5c-1b"]}
    man.update(manifesto_extra or {})
    open(os.path.join(cj, "coleta-manifesto.txt"), "w").write("".join("%s=%s\n" % kv for kv in man.items()))
    if mut_arq:
        mut_arq(cj)
    return sj, cj


def roda_hc1b(sj, cj):
    b = os.path.dirname(sj)
    rc = C.main(["--saida", sj, "--ids", os.path.join(b, "ids.json"), "--refs", os.path.join(b, "refs"), "--textos", os.path.join(b, "textos.json"), "--coleta", cj])
    return json.load(open(os.path.join(sj, "f2d-resultado.json")))["HC1b"]


def um_ulp(d):
    d["pedidos"][2]["logits"][5] = HEXN("um ulp")


h = roda_hc1b(*monta_coleta("ok"))
confere("HC1b: duas coleções completas, binários e entradas corretos, hashes iguais: dentro", h["classe"] == "dentro" and h["divergencias"] == [])
h = roda_hc1b(*monta_coleta("ulp", mut_c=um_ulp))
confere("HC1b: 1 ULP trocado num hash de logits (ids iguais): REGRESSÃO", h["classe"] == "REGRESSÃO" and h["divergencias"])
def kv_dif(d): d["pedidos"][1]["kv"][3][1] = HEXN("kv")
h = roda_hc1b(*monta_coleta("kv", mut_c=kv_dif))
confere("HC1b: KV de uma camada diferente: REGRESSÃO", h["classe"] == "REGRESSÃO")
def outro_modelo(d): d["modelo"] = "3b"
def so_um(d): d["pedidos"] = [dict(d["pedidos"][0], indice=99, ids=[999], gerados=[888], logits=[HEXN(1)], kv=[[HEXN(2), HEXN(3)]])]
sj_, cj_ = monta_coleta("rev1", mut_o=lambda d: (outro_modelo(d), so_um(d)), mut_c=lambda d: (outro_modelo(d), so_um(d)))
h = roda_hc1b(sj_, cj_)
confere("HC1b: mesma coleção malformada (modelo 3B, 1 pedido de índice 99, prompt [999]) nos dois lados: SEM PROVA, nunca dentro", h["classe"] == "SEM PROVA" and any("modelo" in x for x in h["problemas"]))
def trunc(d): d["pedidos"] = d["pedidos"][:2]
h = roda_hc1b(*monta_coleta("trunc", mut_o=trunc, mut_c=trunc))
confere("HC1b: coleta truncada igualmente nos dois lados: SEM PROVA", h["classe"] == "SEM PROVA" and any("pedidos (esperados" in x for x in h["problemas"]))
def sem_kv(d): [p.__setitem__("kv", p["kv"][:5]) for p in d["pedidos"]]
h = roda_hc1b(*monta_coleta("camadas", mut_o=sem_kv, mut_c=sem_kv))
confere("HC1b: camadas faltando nos dois lados: SEM PROVA", h["classe"] == "SEM PROVA" and any("KV sem 24 camadas" in x for x in h["problemas"]))
def sem_passos(d): [p.__setitem__("logits", p["logits"][:10]) for p in d["pedidos"]]
h = roda_hc1b(*monta_coleta("passos", mut_o=sem_passos, mut_c=sem_passos))
confere("HC1b: passos de decode faltando nos dois lados: SEM PROVA", h["classe"] == "SEM PROVA")
def mal(d): d["pedidos"][1]["logits"][0] = "xyz"
h = roda_hc1b(*monta_coleta("hash", mut_o=mal, mut_c=mal))
confere("HC1b: hash malformado (não 64 hexadecimais): SEM PROVA", h["classe"] == "SEM PROVA")
def ids_outros(d): [p.__setitem__("ids", [1, 2, 3]) for p in d["pedidos"]]
h = roda_hc1b(*monta_coleta("prompt", mut_o=ids_outros, mut_c=ids_outros))
confere("HC1b: outro prompt nos dois lados: SEM PROVA", h["classe"] == "SEM PROVA" and any("ids de prompt" in x for x in h["problemas"]))
h = roda_hc1b(*monta_coleta("mesmo", mesmo_arquivo=True))
confere("HC1b: o mesmo arquivo nos dois papéis: SEM PROVA", h["classe"] == "SEM PROVA" and any("mesmo arquivo" in x for x in h["problemas"]))
h = roda_hc1b(*monta_coleta("binerr", bin_c="d" * 64))
confere("HC1b: binário do candidato no manifesto diferente do congelado: SEM PROVA", h["classe"] == "SEM PROVA" and any("binário" in x for x in h["problemas"]))
h = roda_hc1b(*monta_coleta("shaarq", manifesto_extra={"gancho_candidato_sha": "e" * 64}))
confere("HC1b: sha do arquivo diferente do manifesto: SEM PROVA", h["classe"] == "SEM PROVA" and any("sha do arquivo candidato" in x for x in h["problemas"]))
h = roda_hc1b(*monta_coleta("modelo", manifesto_extra={"modelo": "3b"}))
confere("HC1b: manifesto de outro modelo: SEM PROVA", h["classe"] == "SEM PROVA")
rc, R, md = roda(novo("semcol"))
confere("HC1b sem a pasta da coleta: SEM PROVA, e a janela de medida não fecha a condição bit a bit", R["HC1b"]["classe"] == "SEM PROVA" and "execução própria" in R["HC1b"]["motivo"])

# ---- rev-1 r167 P2-1: o par da coleta tem de ser o da janela de medida
def hc1b_janela(par=None, **kw):
    sj_, cj_ = monta_coleta("par", par=par, **kw)
    return roda_hc1b(sj_, cj_), sj_, cj_


h, sj_, cj_ = hc1b_janela(par=("c" * 64, "d" * 64))
confere("coleta íntegra de outro par (C/D) com janela A/B: SEM PROVA, motivo par incompatível, nunca dentro", h["classe"] == "SEM PROVA" and any("par incompatível" in x for x in h["problemas"]))
rc_ = C.main(["--coleta-so", "--saida", cj_, "--ids", os.path.join(os.path.dirname(sj_), "ids.json"), "--refs", os.path.join(os.path.dirname(sj_), "refs")])
rj = json.load(open(os.path.join(cj_, "f2d-coleta-resultado.json")))
confere("--coleta-so da coleta C/D: HC1b dentro (análise independente) mas identifica o par ao qual a prova se aplica", rc_ == 0 and rj["HC1b"]["classe"] == "dentro" and rj["HC1b"]["par_aplicavel"] == {"base": "c" * 64, "candidato": "d" * 64}
        and "vale só para o par" in open(os.path.join(cj_, "f2d-coleta-resultado.md"), encoding="utf-8").read() and rj["teste_gpu"]["executado"] is False)
h, _, _ = hc1b_janela(par=(CAND, BASE))
confere("coleta com os papéis invertidos (original = candidato da janela): SEM PROVA, par incompatível", h["classe"] == "SEM PROVA" and any("par incompatível" in x for x in h["problemas"]))
h, _, _ = hc1b_janela(par=(BASE, "d" * 64))
confere("coleta com só o candidato trocado: SEM PROVA, par incompatível", h["classe"] == "SEM PROVA" and any("par incompatível" in x for x in h["problemas"]))
h, _, _ = hc1b_janela()
confere("coleta do par A/B da janela A/B: dentro", h["classe"] == "dentro")


def entrada_outra(cj):
    for nm in ("integridade-antes.txt", "integridade-depois.txt", "integridade-fixos.txt"):
        pth = os.path.join(cj, nm)
        t_ = open(pth).read()
        ob = dict(x.split()[:2] for x in t_.splitlines() if x.strip())["textos-json"]
        open(pth, "w").write(t_.replace(ob, "9" * 64) if nm != "integridade-fixos.txt" else "\n".join(" ".join(("9" * 64) if (i_ in (1, 2) and ln.startswith("textos-json ")) else x for i_, x in enumerate(ln.split())) for ln in t_.splitlines()) + "\n")


h, _, _ = hc1b_janela(mut_arq=entrada_outra)
confere("coleta com entrada (textos) diferente da janela: SEM PROVA, entrada diferente", h["classe"] == "SEM PROVA" and any("textos-json" in x for x in h["problemas"]))

# ---- rev-1 r167 P2-2: integridade antes x depois da coleta
def depois_mutado(cj):
    pth = os.path.join(cj, "integridade-depois.txt")
    ln = open(pth).read().splitlines()
    ln = [(l.split()[0] + " " + "7" * 64) if l.startswith("binario-cand ") else l for l in ln]
    open(pth, "w").write("\n".join(ln) + "\n")


def depois_ausente(cj):
    os.rename(os.path.join(cj, "integridade-depois.txt"), os.path.join(cj, "integridade-depois.txt.fora"))


def depois_truncado(cj):
    pth = os.path.join(cj, "integridade-depois.txt")
    ln = open(pth).read().splitlines()
    open(pth, "w").write("\n".join(ln[:3]) + "\nbinario-ba")


for nome_, f_, frag in (("sha mutado só no depois", depois_mutado, "sha mudou entre antes e depois"), ("manifesto depois ausente", depois_ausente, "faltam integridade-antes.txt/integridade-depois.txt"), ("manifesto depois truncado", depois_truncado, "antes x depois")):
    h, sj_, cj_ = hc1b_janela(mut_arq=f_)
    ok_ = h["classe"] == "SEM PROVA" and any("(antes x depois)" in x and (frag in x or frag == "antes x depois") for x in h["problemas"])
    C.main(["--coleta-so", "--saida", cj_, "--ids", os.path.join(os.path.dirname(sj_), "ids.json"), "--refs", os.path.join(os.path.dirname(sj_), "refs")])
    rj = json.load(open(os.path.join(cj_, "f2d-coleta-resultado.json")))
    confere("integridade da coleta, %s: SEM PROVA com motivo específico, também no --coleta-so" % nome_, ok_ and rj["HC1b"]["classe"] == "SEM PROVA" and any("(antes x depois)" in x for x in rj["HC1b"]["problemas"]))

# ---- rev-1 r167 P3: n_repete 3 e exatamente 4 pedidos
for n_ in (1, 2):
    h, _, _ = hc1b_janela(n_rep=n_)
    confere("coleta completa autodeclarada com n_repete %d: SEM PROVA (o contrato exige 3)" % n_, h["classe"] == "SEM PROVA" and any("n_repete" in x for x in h["problemas"]))

# ---- rev-2 r167 P3-2: o resultado traz o teste de GPU
def com_teste(rc_t):
    def f_(cj):
        open(os.path.join(cj, "teste-gpu.rc"), "w").write("%d\n" % rc_t)
        man_p = os.path.join(cj, "coleta-manifesto.txt")
        open(man_p, "a").write("teste_gpu_commit=%s\nteste_gpu_exe_sha=%s\nteste_gpu_exe_sha_reforma=%s\nteste_gpu_exe_sha_o3=%s\n" % ("a1" * 20, "5" * 64, "6" * 64, "7" * 64))
    return f_


for rc_t in (0, 101):
    h, sj_, cj_ = hc1b_janela(mut_arq=com_teste(rc_t))
    C.main(["--coleta-so", "--saida", cj_, "--ids", os.path.join(os.path.dirname(sj_), "ids.json"), "--refs", os.path.join(os.path.dirname(sj_), "refs")])
    rj = json.load(open(os.path.join(cj_, "f2d-coleta-resultado.json")))
    md_ = open(os.path.join(cj_, "f2d-coleta-resultado.md"), encoding="utf-8").read()
    confere("teste_gpu no resultado da coleta (rc %d): executado, rc, passou, commit e sha do executável; linha no .md" % rc_t, rj["teste_gpu"] == {"executado": True, "rc": rc_t, "passou": rc_t == 0, "procedencia_completa": True, "commit_worktree": "a1" * 20, "sha_executavel": "5" * 64, "sha_executavel_reforma": "6" * 64, "sha_executavel_o3": "7" * 64}
            and ("passou" in md_ if rc_t == 0 else "FALHOU (rc 101)" in md_))

# ---- rev-1/rev-2 r168: modelo (safetensors/config) vinculado; teste_gpu.passou exige procedência; teto-excedido.txt obrigatório
def safetensors_outro(cj):
    for nm in ("integridade-antes.txt", "integridade-depois.txt"):
        pth = os.path.join(cj, nm)
        t_ = "".join(("safetensors-1b " + "8" * 64 + "\n") if l.startswith("safetensors-1b ") else l + "\n" for l in open(pth).read().splitlines())
        open(pth, "w").write(t_)


h, _, _ = hc1b_janela(mut_arq=safetensors_outro)
confere("sha de safetensors-1b trocado só na coleta: SEM PROVA, modelo diferente entre janela e coleta", h["classe"] == "SEM PROVA" and any("safetensors-1b" in x for x in h["problemas"]))
def config_ausente(cj):
    for nm in ("integridade-antes.txt", "integridade-depois.txt"):
        pth = os.path.join(cj, nm)
        t_ = "".join(l + "\n" for l in open(pth).read().splitlines() if not l.startswith("config-1b "))
        open(pth, "w").write(t_)
h, _, _ = hc1b_janela(mut_arq=config_ausente)
confere("config-1b ausente só da coleta: SEM PROVA (conjunto de chaves diferente)", h["classe"] == "SEM PROVA" and any("config-1b" in x for x in h["problemas"]))


def so_rc(rc_t):
    def f_(cj):
        open(os.path.join(cj, "teste-gpu.rc"), "w").write("%d\n" % rc_t)
    return f_


h, sj_, cj_ = hc1b_janela(mut_arq=so_rc(0))
C.main(["--coleta-so", "--saida", cj_, "--ids", os.path.join(os.path.dirname(sj_), "ids.json"), "--refs", os.path.join(os.path.dirname(sj_), "refs")])
rj = json.load(open(os.path.join(cj_, "f2d-coleta-resultado.json")))
confere("teste de GPU rc 0 sem commit nem sha do executável: passou False, procedência incompleta no JSON e no .md", rj["teste_gpu"]["executado"] and rj["teste_gpu"]["rc"] == 0 and rj["teste_gpu"]["passou"] is False and rj["teste_gpu"]["procedencia_completa"] is False
        and "rc 0, procedência incompleta" in open(os.path.join(cj_, "f2d-coleta-resultado.md"), encoding="utf-8").read())
def commit_curto(cj):
    open(os.path.join(cj, "teste-gpu.rc"), "w").write("0\n")
    open(os.path.join(cj, "coleta-manifesto.txt"), "a").write("teste_gpu_commit=abc123\nteste_gpu_exe_sha=%s\n" % ("5" * 64))
h, sj_, cj_ = hc1b_janela(mut_arq=commit_curto)
C.main(["--coleta-so", "--saida", cj_, "--ids", os.path.join(os.path.dirname(sj_), "ids.json"), "--refs", os.path.join(os.path.dirname(sj_), "refs")])
confere("commit com menos de 40 hex: passou False", json.load(open(os.path.join(cj_, "f2d-coleta-resultado.json")))["teste_gpu"]["passou"] is False)
def sem_sha_reforma(cj):
    open(os.path.join(cj, "teste-gpu.rc"), "w").write("0\n")
    open(os.path.join(cj, "coleta-manifesto.txt"), "a").write("teste_gpu_commit=%s\nteste_gpu_exe_sha=%s\n" % ("a1" * 20, "5" * 64))
h, sj_, cj_ = hc1b_janela(mut_arq=sem_sha_reforma)
C.main(["--coleta-so", "--saida", cj_, "--ids", os.path.join(os.path.dirname(sj_), "ids.json"), "--refs", os.path.join(os.path.dirname(sj_), "refs")])
confere("sha do executável do motor_reforma_gpu ausente (só o do motor_passo_gpu): passou False", json.load(open(os.path.join(cj_, "f2d-coleta-resultado.json")))["teste_gpu"]["passou"] is False)
def sem_sha_o3(cj):
    open(os.path.join(cj, "teste-gpu.rc"), "w").write("0\n")
    open(os.path.join(cj, "coleta-manifesto.txt"), "a").write("teste_gpu_commit=%s\nteste_gpu_exe_sha=%s\nteste_gpu_exe_sha_reforma=%s\n" % ("a1" * 20, "5" * 64, "6" * 64))
h, sj_, cj_ = hc1b_janela(mut_arq=sem_sha_o3)
C.main(["--coleta-so", "--saida", cj_, "--ids", os.path.join(os.path.dirname(sj_), "ids.json"), "--refs", os.path.join(os.path.dirname(sj_), "refs")])
confere("sha do executável do o3_kernels_gpu ausente (os outros dois presentes): passou False", json.load(open(os.path.join(cj_, "f2d-coleta-resultado.json")))["teste_gpu"]["passou"] is False)
sj = novo("semteto")
os.rename(os.path.join(sj, "teto-excedido.txt"), os.path.join(sj, "teto-excedido.txt.fora"))
rc, R, md = roda(sj)
confere("teto-excedido.txt ausente da janela: integridade quebrada, tudo SEM PROVA", sem_prova(R) and any("teto-excedido.txt" in x for x in R["integridade_problemas"]))
print("TODOS OS TESTES PASSARAM" if falhas == 0 else "%d FALHA(S)" % falhas)
if falhas == 0 and not GUARDA:
    shutil.rmtree(RAIZ_TESTE)   # só o caminho literal que o mkdtemp devolveu
else:
    print("pasta de teste mantida:", RAIZ_TESTE)
sys.exit(1 if falhas else 0)
