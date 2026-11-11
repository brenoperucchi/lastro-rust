#!/usr/bin/env python3
"""Testes SEM GPU do f2b-extrai-v2.py: registros brutos sintéticos, IB1 a IB5 recalculadas em Python, classificadores (HB3 0,60/0,40), controle D/A/B2, agregação por pedido, recusas e ponta a ponta.
    python3 -I test-f2b-extrai-v2.py        (usa mkdtemp para os cenários; apaga a pasta-mãe só se tudo passar; --guarda mantém)
"""
import hashlib
import importlib.util
import json
import os
import sys
import shutil
import tempfile

AQUI = os.path.dirname(os.path.abspath(__file__))
GUARDA = "--guarda" in sys.argv   # --guarda: mantém a pasta-mãe dos cenários mesmo se tudo passar
RAIZ_TESTE = tempfile.mkdtemp(prefix="f2b-e2e-")   # UMA pasta-mãe por execução; todo mkdtemp seguinte cai dentro dela
tempfile.tempdir = RAIZ_TESTE
spec = importlib.util.spec_from_file_location("f2bx", os.path.join(AQUI, "f2b-extrai-v2.py"))
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
F = B.F
falhas = 0


def confere(nome, cond):
    global falhas
    if cond:
        print("ok:", nome)
    else:
        print("FALHOU:", nome)
        falhas += 1


T2 = 1_000_000
REG = 20


def pre_dur(m):
    return (21 * m + 1) * REG + 20


def host(layers, m=8, gen=0, pos_ini=0, t2=T2, pre=None):
    durs = {(F.H_PRE_LAYER, l): (pre if pre is not None else pre_dur(m)) for l in range(layers)}
    ordem = F.ordem_esperada(layers)
    t = t2 + 10
    entrada = t
    ops = []
    for (op, l) in ordem:
        d = durs.get((op, l), 100)
        ops.append([op, l, t, t + d])
        t += d
    b = {"gen": gen, "m": m, "pos_ini": pos_ini, "t2_ns": t2, "t3_ns": t + 10, "entrada_ns": entrada, "saida_ns": t, "ops": ops}
    b["tb"] = [{"camada": l, "gen": gen, "tb_obs_ns": next(o[2] for o in ops if o[0] == F.H_SEQ and o[1] == l) + 5, "tb_ns": next(o[3] for o in ops if o[0] == F.H_SEQ and o[1] == l) + 20} for l in range(layers)]
    return b


ORDEM_FIX = B.ORDEM_POS_V1   # as fixtures do arquivo são do contrato v1 (14:10); a seção v2 no fim troca para a v2
CONTRATO_FIX = 1


def regs(h, layers, m=8, lock_dur=REG):
    """Registros brutos contíguos dentro de cada H_PRE_LAYER do host, na ordem de emissão."""
    out = []
    for l in range(layers):
        pre = next(o for o in h["ops"] if o[0] == F.H_PRE_LAYER and o[1] == l)
        t = pre[2] + 10
        out.append([B.B_LOCK, l, None, t, t + lock_dur])
        t += lock_dur
        for p in range(m):
            for op in ORDEM_FIX:
                out.append([op, l, p, t, t + REG])
                t += REG
    return {"gen": h["gen"], "m": m, "pos_ini": h["pos_ini"], "regs": out, "perdidos": 0}


def pb_do(h):
    return {"gen": h["gen"], "m": h["m"], "pos_ini": h["pos_ini"], "t_ns": [0, 0, h["t2_ns"], h["t3_ns"], 0]}


LAY = 2
h = host(LAY)
b = regs(h, LAY)
confere("ordem de emissão tem 21 registros por posição: 2/1/1/11/6", len(B.ORDEM_POS) == 21 and B.ORDEM_POS.count(B.B_SETUP) == 2 and B.ORDEM_POS.count(B.B_GENERICS) == 1
        and B.ORDEM_POS.count(B.B_ATTN_PREP) == 1 and B.ORDEM_POS.count(B.B_LANCA) == 11 and B.ORDEM_POS.count(B.B_DTOD) == 6)
confere("bloco íntegro: IB1, IB2, IB3", B.ib1(b, h) and B.ib2(b, h, LAY) and B.ib3(b, LAY))
confere("pedido íntegro: sem problemas e sem ressalva", B.valida_pedido_b({"layers": LAY, "blocos": [b]}, {"blocos": [h]}, {"blocos": [pb_do(h)]}, LAY) == ([], False))
# IB3: mutante que troca as posições por zero, repetidas, fora de ordem; m = 1, restos
import copy
z = copy.deepcopy(b)
for r in z["regs"]:
    if r[0] != B.B_LOCK:
        r[2] = 0
confere("IB3: posições não sentinela todas zero (totais iguais) falham", len(z["regs"]) == len(b["regs"]) and not B.ib3(z, LAY))
rp = copy.deepcopy(b)
for r in rp["regs"]:
    if r[0] != B.B_LOCK and r[2] == 3:
        r[2] = 2
    elif r[0] != B.B_LOCK and r[2] == 5:
        r[2] = 4
confere("IB3: duas posições repetidas no lugar de outras falham", not B.ib3(rp, LAY))
fo = copy.deepcopy(b)
for r in fo["regs"]:
    if r[0] != B.B_LOCK and r[2] in (1, 2):
        r[2] = 3 - r[2]
confere("IB3: posições fora de ordem falham", not B.ib3(fo, LAY))
pz, _ = B.valida_pedido_b({"layers": LAY, "blocos": [z]}, {"blocos": [h]}, {"blocos": [pb_do(h)]}, LAY)
confere("pedido com mutante de posições: inválido, motivo IB3", any("IB3" in x for x in pz))
for m_ in (1, 2, 7):
    hm = host(LAY, m=m_)
    confere("IB3 íntegro para m = %d" % m_, B.ib3(regs(hm, LAY, m=m_), LAY))
perd = dict(b, perdidos=1)
confere("IB3: registro perdido por capacidade falha", not B.ib3(perd, LAY))
vazio = dict(b, regs=[])
confere("IB3 e IB1: sem registros (o Fake) falham", not B.ib3(vazio, LAY) and not B.ib1(vazio, h))
# IB1
tr = copy.deepcopy(b)
tr["regs"][5], tr["regs"][6] = tr["regs"][6], tr["regs"][5]
confere("IB1: ordem trocada de dois registros (soma e contagem iguais) falha, IB3 segue", B.ib3(tr, LAY) and not B.ib1(tr, h))
so = copy.deepcopy(b)
so["regs"][7][3] = so["regs"][6][4] - 1
confere("IB1: sobreposição falha", not B.ib1(so, h))
fr = copy.deepcopy(b)
fr["regs"][-1][4] = fr["regs"][-1][4] + 5 * B.TOL_NS
confere("IB1: último fim fora do H_PRE_LAYER (acima de 1 us) falha; IB2 acusa o excesso", not B.ib1(fr, h) and not B.ib2(fr, h, LAY))
tol = copy.deepcopy(b)
tol["regs"][0][3] = tol["regs"][0][3] - 10 - B.TOL_NS
confere("IB1: 1 us de tolerância na borda vale", B.ib1(tol, h))
# IB2 e IB5
confere("residuo da camada = H_PRE_LAYER - soma dos registros", B.residuo_camada(b, h, 0) == pre_dur(8) - 169 * REG)
conf = host(LAY, pre=pre_dur(8) + 20000)
confere("IB5: resíduo acima de 3% é ressalva, não invalida", B.razao_residuo(regs(conf, LAY), conf, LAY) > 0.03
        and B.valida_pedido_b({"layers": LAY, "blocos": [regs(conf, LAY)]}, {"blocos": [conf]}, {"blocos": [pb_do(conf)]}, LAY) == ([], True))
# IB4
g = dict(b, gen=9)
confere("IB4: gen diferente da F2 invalida o pedido", any("IB4" in x for x in B.valida_pedido_b({"layers": LAY, "blocos": [g]}, {"blocos": [h]}, {"blocos": [pb_do(h)]}, LAY)[0]))
hr = copy.deepcopy(h)
next(o for o in hr["ops"] if o[0] == F.H_ROUTE)[3] += 10_000
confere("IB4: F2 que não fecha t3 - t2 invalida", any("IB4" in x for x in B.valida_pedido_b({"layers": LAY, "blocos": [b]}, {"blocos": [hr]}, {"blocos": [pb_do(hr)]}, LAY)[0]))
confere("blocos a menos no perfil-f2b invalidam", B.valida_pedido_b({"layers": LAY, "blocos": []}, {"blocos": [h]}, {"blocos": [pb_do(h)]}, LAY)[0] != [])
confere("layers errado invalida", B.valida_pedido_b({"layers": 5, "blocos": [b]}, {"blocos": [h]}, {"blocos": [pb_do(h)]}, LAY)[0] != [])
confere("sem_registro invalida", any("sem registro" in x for x in B.valida_pedido_b({"layers": LAY, "blocos": [{"sem_registro": True}]}, {"blocos": [h]}, {"blocos": [pb_do(h)]}, LAY)[0]))

# classificadores
confere("HB3: >= 0,60 dentro; < 0,40 fora; entre, sem leitura; sem número, sem dado", (B.classifica_hb3(0.60), B.classifica_hb3(0.8), B.classifica_hb3(0.39), B.classifica_hb3(0.40), B.classifica_hb3(0.59), B.classifica_hb3(None))
        == ("dentro", "dentro", "fora", "sem leitura", "sem leitura", "sem dado"))
confere("HB1, HB2, HB4: descritivas", (B.classifica_descritiva(0.01), B.classifica_descritiva(None)) == ("descritivo", "sem dado"))
# medidas
pares = [(regs(host(LAY, gen=i), LAY), host(LAY, gen=i)) for i in range(4)]
mp = B.medidas_pedido_b(pares, LAY)
confere("parcelas: B_LANCA 11/21 dos registros de posição; soma das quatro <= 1; HB5 e HB6 presentes", abs(mp["hb3"] / mp["hb4"] - 11 / 6) < 1e-9 and mp["hb1"] + mp["hb2"] + mp["hb3"] + mp["hb4"] <= 1 + 1e-9
        and mp["hb5_dispersao_camadas"] == 1.0 and len(mp["h_pre_layer_mediana_camada_ns"]) == LAY and len(mp["hb6_blocos"]) == 4)
lento = [(regs(host(LAY, gen=i, pre=pre_dur(8) * 3), LAY), host(LAY, gen=i, pre=pre_dur(8) * 3)) for i in range(4)]
cel = B.medidas_celula_b([pares, pares, lento], LAY)
confere("célula: mediana entre pedidos (o pedido lento não arrasta) e valores por pedido guardados", abs(cel["hb3"] - mp["hb3"]) < 1e-12 and len(cel["por_pedido"]) == 3)

# controle D/A/B2
def proc(b_, pre, passo=1.0, **kw):
    d = {"braco": b_, "prefill_ms_mediana": pre, "passo_ms_mediana": passo, "valido": True, "imprimiu_bloco": b_ != "D", "imprimiu_f2": b_ == "B2", "imprimiu_f2b": b_ == "B2",
         "produziu_bloco": b_ != "D", "produziu_f2": b_ == "B2", "produziu_f2b": b_ == "B2"}
    d.update(kw)
    return d


def nove(pre_a=100.0, pre_b=100.0, pre_d=100.0):
    return [proc(x, {"D": pre_d, "A": pre_a, "B2": pre_b}[x]) for x in B.ORDEM_CONTROLE]


confere("ordem do controle D A B2, B2 D A, A B2 D", B.ORDEM_CONTROLE == ["D", "A", "B2", "B2", "D", "A", "A", "B2", "D"])
confere("controle: <= 3% comprovada; <= 10% com ressalva; > 10% SEM PROVA", (B.classifica_controle_b(nove(100, 102.5))["classe"], B.classifica_controle_b(nove(100, 108))["classe"], B.classifica_controle_b(nove(100, 120))["classe"])
        == ("comprovada", "com ressalva", "SEM PROVA"))
confere("controle: B2 contra D só apoio", (lambda r: r["classe"] == "comprovada" and r["classe_apoio"] == "SEM PROVA")(B.classifica_controle_b(nove(100, 100, 150))))
confere("controle: ordem trocada, 8 processos, orçamento e motivo = incompleto", all(B.classifica_controle_b(x)["classe"] == "controle incompleto" for x in (list(reversed(nove())), nove()[:8])) and
        B.classifica_controle_b(nove(), orcamento_ok=False)["classe"] == "controle incompleto" and B.classifica_controle_b(nove(), incompleto_motivo="x")["motivo"] == "x")
for nome_, i_, k_ in (("D com perfil-f2b", 4, "produziu_f2b"), ("B2 sem perfil-f2b.json", 2, "produziu_f2b"), ("A com perfil-f2", 1, "produziu_f2"), ("B2 sem linha f2b", 3, "imprimiu_f2b")):
    r_ = nove(); r_[i_] = dict(r_[i_], **{k_: (nome_.startswith("D") or nome_.startswith("A"))})
    confere("controle: %s = incompleto" % nome_, B.classifica_controle_b(r_)["classe"] == "controle incompleto")
r_ = nove(); r_[1] = dict(r_[1], valido=False)
confere("controle: processo inválido = incompleto", B.classifica_controle_b(r_)["classe"] == "controle incompleto")
confere("controle: ms por bloco usa 22 blocos", abs(B.classifica_controle_b(nove(100, 102.2))["principal"]["ms_por_bloco"] - 2.2 / 22) < 1e-9)

# recusas
d = tempfile.mkdtemp(prefix="f2b-extrai-teste.")
os.makedirs(os.path.join(d, "celulas"))
confere("sem orcamento.json: RECUSADO e resultado marcado recusado", B.main(["--saida", d]) == 2 and json.load(open(os.path.join(d, "f2b-host-resultado.json")))["recusado"] is True)
confere("pasta inexistente: código 2", B.main(["--saida", os.path.join(d, "nao-existe")]) == 2)
json.dump({"modelos": ["1b", "3b"]}, open(os.path.join(d, "orcamento.json"), "w"))
confere("orçamento com dois modelos: RECUSADO", B.main(["--saida", d]) == 2)

# ======================================================================================================================================
# Ponta a ponta: janela sintética completa de 1B (24 camadas, m = 8), main -> processa -> f2b-host-resultado.md/.json
L1B = 24
G = list(range(64))
X_ = F.X


def sha_b(x):
    return hashlib.sha256(x).hexdigest()


def fake_le_celula_lastro(dd, meta, refs):
    meds = []
    for pn in sorted(x for x in os.listdir(dd) if x.startswith("p") and x[1:].isdigit()):
        mj = X_.le_json(os.path.join(dd, pn, "saida", "medidas-%s.json" % meta["modelo"]))
        for e in mj["medidas"]:
            if not e.get("descartada"):
                meds.append({"processo": pn, "rotulo": e["rotulo"], "prefill_ms": e["prefill_ms"], "mediana_passos_descritiva": 1.0, "gerados": e["gerados"], "ids_n": len(e["ids"])})
    return {"meds": meds, "cargas_limite_ms": [], "esperado_n": None, "temp_max": None, "slots_impressos": []}, []


def peds_perfil(t, n_meas, trunc=(), mut_b=None):
    plano = F.I.plano_blocos(F.N_POS[t])
    pbs, f2s, fbs = [], [], []
    for i in range(n_meas + 1):
        hs, pbb_, pos = [], [], 0
        for bi, mb in enumerate(plano):
            hh = host(L1B, m=mb, gen=bi, pos_ini=pos, t2=T2 + bi * 10_000_000)
            pos += mb
            hs.append(hh)
            pbb_.append({"m": mb, "conserva": True, "gen": hh["gen"], "pos_ini": hh["pos_ini"], "t_ns": [0, 0, hh["t2_ns"], hh["t3_ns"], 0]})
        bs = [regs(hh, L1B, m=hh["m"]) for hh in hs]
        if mut_b and i > 0:
            mut_b(i, bs)
        if i in trunc:
            keep = sorted({0, 1, len(hs) - 1})
            hs, pbb_, bs = [hs[k] for k in keep], [pbb_[k] for k in keep], [bs[k] for k in keep]
        pbs.append({"modelo": "1b", "indice": i, "aquecimento": i == 0, "n": F.N_POS[t], "plano_m": plano, "blocos": pbb_, "valido": True, "problemas": []})
        f2s.append({"modelo": "1b", "indice": i, "layers": L1B, "blocos": hs})
        fbs.append({"modelo": "1b", "indice": i, "layers": L1B, "blocos": bs, "extra": 0})
    return pbs, f2s, fbs


def monta_janela(d, longo_n=5, trunc_curto=(), smoke_mut=None, medio_mut=None, b_pre=100.5, mut_pb_curto=None, curto_mut=None, sem_f2b_em=None, f2b_ligado_em_d=False):
    w = lambda p, txt: (os.makedirs(os.path.dirname(p), exist_ok=True), open(p, "w", encoding="utf-8").write(txt))
    ids = {t: list(range(F.N_POS[t])) for t in F.TEXTOS}
    w(os.path.join(d, "ids.json"), json.dumps({"modelos": {"1b": {t: {"ids": ids[t]} for t in F.TEXTOS}}}))
    w(os.path.join(d, "refs", "ref5c-1b.json"), json.dumps({"casos": [{"nome": t, "gerados": G} for t in F.TEXTOS]}))
    w(os.path.join(d, "textos.json"), json.dumps({"textos": []}))
    obs = {"cmp-ids-prompt": sha_b(open(os.path.join(d, "ids.json"), "rb").read()), "ref5c-1b": sha_b(open(os.path.join(d, "refs", "ref5c-1b.json"), "rb").read()),
           "textos-json": sha_b(open(os.path.join(d, "textos.json"), "rb").read())}
    sj = os.path.join(d, "janela")
    for k in B.FIXO_CHAVES + ("tokenizer-1b",):
        obs.setdefault(k, sha_b(k.encode()))
    w(os.path.join(sj, "integridade-antes.txt"), "".join("%s %s\n" % kv for kv in obs.items()))
    w(os.path.join(sj, "integridade-depois.txt"), "".join("%s %s\n" % kv for kv in obs.items()))
    w(os.path.join(sj, "integridade-fixos.txt"), "".join("%s %s %s OK\n" % (k, h_, h_) for k, h_ in obs.items()) + "potencia 370.00_W 370.00_W OK\n")
    w(os.path.join(sj, "ambiente.txt"), "uuid=GPU-b22d9465-8552-9f97-812e-380b590645f8\npotencia_inicio=370.00_W\npotencia_fim=370.00_W\n")
    w(os.path.join(sj, "orcamento.json"), json.dumps({"modelos": ["1b"], "nao_rodou": [], "cortes": [], "decorrido_s": 500, "limite_s": 1200}))
    ordem = ["01 smoke-1b-L100-curto-B2 n=1 braco=B2 inicio=0s fim=10s rc=0"]
    for k, b_ in enumerate(B.ORDEM_CONTROLE):
        ordem.append("%02d ctrl-1b-L100-medio-%02d-%s n=3 braco=%s inicio=%ds fim=%ds rc=0" % (k + 2, k + 1, b_, b_, 10 + 20 * k, 30 + 20 * k))
    w(os.path.join(sj, "ordem.txt"), "\n".join(ordem) + "\n")

    def pasta(nome, t, n, braco, pre, trunc=(), mut_pb=None, mut_b=None):
        pd = os.path.join(sj, "celulas", nome)
        w(os.path.join(pd, "celula.json"), json.dumps({"lado": "lastro", "serie": "L100", "modelo": "1b", "texto": t, "estado": "quente", "n_medidas": n}))
        f2b_linha = "ligado" if (braco == "B2" or (f2b_ligado_em_d and braco == "D")) else "desligado"
        w(os.path.join(pd, "p1", "stdout.txt"), "perfil-bloco: %s\nperfil-f2: %s\nperfil-f2b: %s\n" % ("desligado" if braco == "D" else "ligado", "ligado" if braco == "B2" else "desligado", f2b_linha))
        med = [{"rotulo": "r%d" % i, "descartada": i == 0, "falha": None, "gerados": G, "ids": ids[t], "prefill_ms": pre} for i in range(n + 1)]
        w(os.path.join(pd, "p1", "saida", "medidas-1b.json"), json.dumps({"modelo": "1b", "texto": t, "medidas": med}))
        if braco != "D":
            pbs, f2s, fbs = peds_perfil(t, n, trunc, mut_b)
            doc = {"tipo": "perfil-bloco", "versao": 2, "pedidos": pbs}
            if mut_pb:
                mut_pb(doc)
            w(os.path.join(pd, "p1", "saida", "perfil-bloco.json"), json.dumps(doc))
            if braco == "B2":
                w(os.path.join(pd, "p1", "saida", "perfil-f2.json"), json.dumps({"tipo": "perfil-f2", "versao": 1, "pedidos": f2s}))
                if sem_f2b_em != nome:
                    w(os.path.join(pd, "p1", "saida", "perfil-f2b.json"), json.dumps({"tipo": "perfil-f2b", "versao": 1, **({"contrato_b": 2} if CONTRATO_FIX == 2 else {}), "pedidos": fbs}))
    pasta("smoke-1b-L100-curto-B2", "curto", 1, "B2", b_pre, mut_b=smoke_mut)
    for k, b_ in enumerate(B.ORDEM_CONTROLE):
        pasta("ctrl-1b-L100-medio-%02d-%s" % (k + 1, b_), "medio", 3, b_, {"D": 100.0, "A": 100.0, "B2": b_pre}[b_], mut_b=(medio_mut if (medio_mut and k + 1 == 3) else None))
    pasta("lastro-1b-L100-curto-quente", "curto", 5, "B2", b_pre, trunc=trunc_curto, mut_pb=mut_pb_curto, mut_b=curto_mut)
    pasta("lastro-1b-L100-longo-quente", "longo", longo_n, "B2", b_pre)
    return sj


def roda_extrator(sj, anterior=None, extra=()):
    base = os.path.dirname(sj)
    args = ["--saida", sj, "--ids", os.path.join(base, "ids.json"), "--refs", os.path.join(base, "refs"), "--textos", os.path.join(base, "textos.json"), *extra]
    if anterior:
        args += ["--f2-anterior", anterior]
    rc = B.main(args)
    return rc, json.load(open(os.path.join(sj, "f2b-host-resultado.json"))), open(os.path.join(sj, "f2b-host-resultado.md"), encoding="utf-8").read()


F.I.X.le_celula_lastro = fake_le_celula_lastro
base_d = tempfile.mkdtemp(prefix="f2b-e2e-base.")
rc, R, md = roda_extrator(monta_janela(base_d))
confere("E2E base: código 0, janela íntegra, smoke OK, sem bloqueios, controle comprovada, ids de B2 iguais aos de D e A", rc == 0 and R["integridade_problemas"] == [] and R["smoke"]["ok"] and R["bloqueios"] == []
        and R["controle"]["classe"] == "comprovada" and R["ids_B2_igual_D_e_A"])
confere("E2E base: médio com 9 pedidos válidos; curto e longo com 5; status OK", R["textos"]["medio"]["pedidos_validos"] == 9 and R["textos"]["curto"]["pedidos_validos"] == 5
        and R["textos"]["longo"]["pedidos_validos"] == 5 and all(R["textos"][t]["status"] == "OK" for t in F.TEXTOS))
hm = R["textos"]["medio"]["hipoteses"]
confere("E2E base: classes do médio (HB3 sem leitura: 11/21 de quase todo H_PRE_LAYER dá ~0,52; demais descritivas)", hm["HB3"] == "sem leitura" and hm["HB1"] == hm["HB2"] == hm["HB4"] == hm["HB5"] == hm["HB6"] == "descritivo")
parc = R["textos"]["medio"]["medidas"]
confere("E2E base: HB3 sintética = 11 x 8 x 20 ns sobre H_PRE_LAYER, entre 0,40 e 0,60", abs(parc["hb3"] - (11 * 8 * REG) / pre_dur(8)) < 0.02 and 0.40 <= parc["hb3"] < 0.60)
confere("E2E base: relatório com rótulo do plano, frase do HB1, comparação entre janelas, e sem cuLaunchKernel nem 'causa'", "tempo da chamada de envio de trabalho" in md and "mede o custo de pegar uma trava livre" in md
        and "entre janelas" in md and "cuLaunchKernel" not in md and "causa" not in md.lower() and "cuLaunchKernel" not in json.dumps(R) and "## Controle" in md and "## Orçamento" in md)
confere("E2E base: sem --f2-anterior a comparação é 'sem dado'", R["comparacao_h_pre_layer"]["b_1100_mediana_por_camada_ns"] is None and "sem dado" in md)
# comparação com a janela de 11:00 sintética (perfil-f2.json dos B do médio)
ant = tempfile.mkdtemp(prefix="f2b-ant.")
for k, b_ in enumerate(B.ORDEM_CONTROLE):
    if b_ == "B2":
        _, f2s, _ = peds_perfil("medio", 3)
        p = os.path.join(ant, "celulas", "ctrl-1b-L100-medio-%02d-B" % (k + 1), "p1", "saida")
        os.makedirs(p)
        json.dump({"tipo": "perfil-f2", "versao": 1, "pedidos": f2s}, open(os.path.join(p, "perfil-f2.json"), "w"))
d_c = tempfile.mkdtemp(prefix="f2b-e2e-comp.")
rc, R, md = roda_extrator(monta_janela(d_c), anterior=ant)
confere("E2E comparação com o B de 11:00: razão mediana 1,0 (mesmos dados) e tabela no relatório", rc == 0 and abs(R["comparacao_h_pre_layer"]["razao_mediana_b2_sobre_b"] - 1.0) < 1e-9 and "| 0 |" in md)
# mutantes
d1 = tempfile.mkdtemp(prefix="f2b-e2e-z.")
def mut_zero(i, bs):
    for bb in bs:
        for r in bb["regs"]:
            if r[0] != B.B_LOCK:
                r[2] = 0
rc, R, md = roda_extrator(monta_janela(d1, curto_mut=mut_zero))
cu = R["textos"]["curto"]
confere("E2E posições zeradas em 5 pedidos do curto: 0 válidos, H do curto SEM PROVA, motivo IB3 preservado, médio intacto", cu["pedidos_validos"] == 0 and set(cu["hipoteses"].values()) == {"SEM PROVA"} and any("IB3" in x for x in cu["motivos"])
        and R["textos"]["medio"]["pedidos_validos"] == 9)
d2 = tempfile.mkdtemp(prefix="f2b-e2e-um.")
rc, R, md = roda_extrator(monta_janela(d2, curto_mut=lambda i, bs: mut_zero(i, bs) if i == 2 else None))
confere("E2E um pedido de 5 com IB3 falha: 4 válidos, classes seguem, motivo registrado", R["textos"]["curto"]["pedidos_validos"] == 4 and R["textos"]["curto"]["hipoteses"]["HB3"] == "sem leitura" and any("IB3" in x for x in R["textos"]["curto"]["motivos"]))
d3 = tempfile.mkdtemp(prefix="f2b-e2e-dois.")
rc, R, md = roda_extrator(monta_janela(d3, curto_mut=lambda i, bs: mut_zero(i, bs) if i in (1, 3, 5) else None))
confere("E2E dois válidos (abaixo de 3): SEM PROVA pelo mínimo", R["textos"]["curto"]["pedidos_validos"] == 2 and set(R["textos"]["curto"]["hipoteses"].values()) == {"SEM PROVA"})
d4 = tempfile.mkdtemp(prefix="f2b-e2e-smoke.")
def mut_aq(i, bs):
    pass
sj4 = monta_janela(d4)
pj = os.path.join(sj4, "celulas", "smoke-1b-L100-curto-B2", "p1", "saida", "perfil-f2b.json")
jj = json.load(open(pj))
for bb in jj["pedidos"][0]["blocos"]:
    bb["perdidos"] = 1
json.dump(jj, open(pj, "w"))
rc, R, md = roda_extrator(sj4)
confere("E2E IB falha no aquecimento do smoke: smoke FALHOU, bloqueio aplicado, todas as H SEM PROVA", (not R["smoke"]["ok"]) and "aquecimento do smoke" in R["smoke"]["motivo"] and set(R["textos"]["medio"]["hipoteses"].values()) == {"SEM PROVA"})

# I8 do smoke (rodada 161 rev-1 P2): so a I8 do aquecimento; so a do pedido medido; defasagem negativa permitida segue valida
def _smoke_i8(idx, campo, valor):
    dd = tempfile.mkdtemp(prefix="f2b-e2e-i8.")
    sj_ = monta_janela(dd)
    pj_ = os.path.join(sj_, "celulas", "smoke-1b-L100-curto-B2", "p1", "saida", "perfil-f2.json")
    jj_ = json.load(open(pj_))
    for bb in jj_["pedidos"][idx]["blocos"]:
        for c_ in bb["tb"]:
            c_[campo] = valor(c_) if callable(valor) else valor
    json.dump(jj_, open(pj_, "w"))
    return roda_extrator(sj_)
rc, R, md = _smoke_i8(0, "tb_obs_ns", 0)
confere("E2E so a I8 do aquecimento do smoke alterada: smoke FALHOU, motivo I8 explicito, todas as H SEM PROVA", (not R["smoke"]["ok"]) and "aquecimento do smoke com I8 falha" in R["smoke"]["motivo"] and all(set(R["textos"][t]["hipoteses"].values()) == {"SEM PROVA"} for t in F.TEXTOS))
rc, R, md = _smoke_i8(1, "tb_obs_ns", 0)
confere("E2E so a I8 do pedido medido do smoke alterada: smoke FALHOU, motivo I8 explicito, todas as H SEM PROVA", (not R["smoke"]["ok"]) and "smoke com I8 falha" in R["smoke"]["motivo"] and all(set(R["textos"][t]["hipoteses"].values()) == {"SEM PROVA"} for t in F.TEXTOS))
rc, R, md = _smoke_i8(1, "tb_obs_ns", lambda c_: c_["tb_obs_ns"] - 500)
confere("E2E defasagem negativa permitida no smoke segue aceita", R["smoke"]["ok"] and R["bloqueios"] == [])
# I8 fora do smoke (cmp medio) nao invalida as medidas HB
dd_ = tempfile.mkdtemp(prefix="f2b-e2e-i8f.")
sj_ = monta_janela(dd_)
pj_ = os.path.join(sj_, "celulas", "lastro-1b-L100-curto-quente", "p1", "saida", "perfil-f2.json")
jj_ = json.load(open(pj_))
for bb in jj_["pedidos"][2]["blocos"]:
    for c_ in bb["tb"]:
        c_["tb_obs_ns"] = 0
json.dump(jj_, open(pj_, "w"))
rc, R, md = roda_extrator(sj_)
confere("E2E I8 fora do smoke nao invalida as medidas HB do nivel B (curto com 5 validos, smoke OK) e a nota da I8 e registrada", R["smoke"]["ok"] and R["textos"]["curto"]["pedidos_validos"] == 5 and "nota_i8" in R)
# ids do smoke e referencia (rev-1 P2 item 2)
dd_ = tempfile.mkdtemp(prefix="f2b-e2e-idq.")
sj_ = monta_janela(dd_)
pj_ = os.path.join(sj_, "celulas", "smoke-1b-L100-curto-B2", "p1", "saida", "medidas-1b.json")
jj_ = json.load(open(pj_))
jj_["medidas"][0]["ids"] = [x + 1 for x in jj_["medidas"][0]["ids"]]
json.dump(jj_, open(pj_, "w"))
rc, R, md = roda_extrator(sj_)
print("  [diag] smoke motivo:", R["smoke"]["motivo"][:200])
confere("E2E um id do aquecimento do smoke alterado: smoke FALHOU, motivo de divergencia, todas as H SEM PROVA", (not R["smoke"]["ok"]) and ("diverg" in R["smoke"]["motivo"]) and all(set(R["textos"][t]["hipoteses"].values()) == {"SEM PROVA"} for t in F.TEXTOS))
dd_ = tempfile.mkdtemp(prefix="f2b-e2e-ref.")
sj_ = monta_janela(dd_)
pr_ = os.path.join(dd_, "refs", "ref5c-1b.json")
json.dump({"casos": [{"nome": t, "gerados": G} for t in F.TEXTOS if t != "curto"]}, open(pr_, "w"))
# a referencia mudou: a integridade sintetica a acusaria; regrava o sha observado
os_ = lambda nome: os.path.join(sj_, nome)
novo = sha_b(open(pr_, "rb").read())
for nome in ("integridade-antes.txt", "integridade-depois.txt", "integridade-fixos.txt"):
    tx = open(os_(nome)).read().split("\n")
    open(os_(nome), "w").write("\n".join(" ".join(novo if (i2 == 1 or i2 == 2) and ln.startswith("ref5c-1b ") else x for i2, x in enumerate(ln.split(" "))) for ln in tx))
rc, R, md = roda_extrator(sj_)
confere("E2E referencia do texto do smoke ausente: smoke FALHOU por referencia, leitura descritiva dos brutos preservada (medio com 9 validos)", (not R["smoke"]["ok"]) and "refer" in R["smoke"]["motivo"] and R["textos"]["medio"]["pedidos_validos"] == 9 and rc in (0, 1))
d5 = tempfile.mkdtemp(prefix="f2b-e2e-i3.")
rc, R, md = roda_extrator(monta_janela(d5, medio_mut=lambda i, bs: mut_zero(i, bs) if i == 1 else None))
confere("E2E médio com um pedido IB3 inválido: 8 válidos, SEM PROVA pelo mínimo de 9, curto segue", R["textos"]["medio"]["pedidos_validos"] == 8 and set(R["textos"]["medio"]["hipoteses"].values()) == {"SEM PROVA"} and R["textos"]["curto"]["pedidos_validos"] == 5)
d6 = tempfile.mkdtemp(prefix="f2b-e2e-corte.")
rc, R, md = roda_extrator(monta_janela(d6, longo_n=3))
confere("E2E longo cortado para 3 medidas: 3 válidos, status OK", R["textos"]["longo"]["pedidos_validos"] == 3 and R["textos"]["longo"]["status"] == "OK")
d7 = tempfile.mkdtemp(prefix="f2b-e2e-ctrl.")
rc, R, md = roda_extrator(monta_janela(d7, b_pre=120.0))
confere("E2E controle com B2 20% acima de A: SEM PROVA, todas as H do modelo SEM PROVA", R["controle"]["classe"] == "SEM PROVA" and all(set(R["textos"][t]["hipoteses"].values()) == {"SEM PROVA"} for t in F.TEXTOS))
d8 = tempfile.mkdtemp(prefix="f2b-e2e-semb.")
rc, R, md = roda_extrator(monta_janela(d8, sem_f2b_em="ctrl-1b-L100-medio-03-B2"))
confere("E2E B2 do controle sem perfil-f2b.json: controle incompleto, todas as H SEM PROVA", R["controle"]["classe"] == "controle incompleto" and all(set(R["textos"][t]["hipoteses"].values()) == {"SEM PROVA"} for t in F.TEXTOS))
d9 = tempfile.mkdtemp(prefix="f2b-e2e-dl.")
rc, R, md = roda_extrator(monta_janela(d9, f2b_ligado_em_d=True))
confere("E2E D que imprime perfil-f2b: ligado: controle incompleto", R["controle"]["classe"] == "controle incompleto")
# esquema e identidade do perfil-base (rodada 155)
def _ver(doc): doc["versao"] = 99
def _modelo(doc): [x.__setitem__("modelo", "3b") for x in doc["pedidos"][1:]]
def _indice(doc): [x.__setitem__("indice", 99) for x in doc["pedidos"][1:]]
def _aquec(doc): [x.__setitem__("aquecimento", True) for x in doc["pedidos"][1:]]
def _um(doc): doc["pedidos"][3]["modelo"] = "3b"
for nome_, mut_ in (("versão 99", _ver), ("modelo trocado nos 5", _modelo), ("índice 99 nos 5", _indice), ("marca de aquecimento incoerente nos 5", _aquec)):
    dx = tempfile.mkdtemp(prefix="f2b-e2e-id.")
    rc, R, md = roda_extrator(monta_janela(dx, mut_pb_curto=mut_))
    cu = R["textos"]["curto"]
    confere("E2E %s no perfil-bloco do curto: 0 válidos, H do curto SEM PROVA, motivo preservado, médio intacto" % nome_, cu["pedidos_validos"] == 0 and set(cu["hipoteses"].values()) == {"SEM PROVA"} and cu["motivos"] != []
            and R["textos"]["medio"]["pedidos_validos"] == 9)
dx = tempfile.mkdtemp(prefix="f2b-e2e-id1.")
rc, R, md = roda_extrator(monta_janela(dx, mut_pb_curto=_um))
confere("E2E um pedido de modelo trocado: 4 válidos, classes seguem", R["textos"]["curto"]["pedidos_validos"] == 4 and any("perfil-bloco: " in x for x in R["textos"]["curto"]["motivos"]))
dxt = tempfile.mkdtemp(prefix="f2b-e2e-trunc.")
rc, R, md = roda_extrator(monta_janela(dxt, trunc_curto=(1, 2, 3, 4, 5)))
confere("E2E blocos removidos dos três perfis em 5 pedidos do curto: 0 válidos, SEM PROVA, motivo do plano de blocos", R["textos"]["curto"]["pedidos_validos"] == 0 and set(R["textos"]["curto"]["hipoteses"].values()) == {"SEM PROVA"}
        and any("plano de blocos" in x for x in R["textos"]["curto"]["motivos"]))
d10 = tempfile.mkdtemp(prefix="f2b-e2e-pot.")
sj = monta_janela(d10)
open(os.path.join(sj, "ambiente.txt"), "w").write("potencia_inicio=370.00_W\npotencia_fim=300.00_W\n")
rc, R, md = roda_extrator(sj)
confere("E2E potência 300 W ao fim: RECUSADO (código 2)", rc == 2 and R.get("recusado") is True)
d11 = tempfile.mkdtemp(prefix="f2b-e2e-sem-linha.")
sj = monta_janela(d11)
for raiz, _, a in os.walk(sj):
    if "stdout.txt" in a:
        p_ = os.path.join(raiz, "stdout.txt")
        open(p_, "w").write("".join(x for x in open(p_) if not x.startswith("perfil-f2b")))
rc, R, md = roda_extrator(sj)
confere("E2E saída sem a linha perfil-f2b em nenhum processo: RECUSADO (não é o binário do nível B)", rc == 2)
# ======================================================================================================================================
# Contrato v2 (plano de cópias, O1): 17 registros por posição. Fixtures antigas só pelo v1, novas só pelo v2; v1 preservado para 14:10.
confere("v2: ordem de 17 registros 2/1/1/11/2; v1 continua de 21", len(B.ORDEM_POS_V2) == 17 and [B.ORDEM_POS_V2.count(x) for x in (B.B_SETUP, B.B_GENERICS, B.B_ATTN_PREP, B.B_LANCA, B.B_DTOD)] == [2, 1, 1, 11, 2]
        and len(B.ORDEM_POS_V1) == 21 and B.ORDEM_POS is B.ORDEM_POS_V1 and B.ORDENS == {1: B.ORDEM_POS_V1, 2: B.ORDEM_POS_V2})
confere("v2: é o v1 sem as 4 cópias de reforma (lançamentos na mesma ordem, 2 cópias do cache ficam)", [x for x in B.ORDEM_POS_V1 if x != B.B_DTOD] == [x for x in B.ORDEM_POS_V2 if x != B.B_DTOD])
ORDEM_FIX, CONTRATO_FIX = B.ORDEM_POS_V2, 2
h2, hv1 = host(LAY), host(LAY)
b2 = regs(h2, LAY)
ORDEM_FIX = B.ORDEM_POS_V1
bv1 = regs(hv1, LAY)
ORDEM_FIX = B.ORDEM_POS_V2
confere("v2: bloco novo passa a IB3 v2 e é recusado pela v1", len(b2["regs"]) == LAY * (17 * 8 + 1) and B.ib3(b2, LAY, 2) and not B.ib3(b2, LAY, 1) and not B.ib3(b2, LAY))
confere("v1: bloco antigo passa a IB3 v1 (padrão) e é recusado pela v2", B.ib3(bv1, LAY) and B.ib3(bv1, LAY, 1) and not B.ib3(bv1, LAY, 2))
pv2, _ = B.valida_pedido_b({"layers": LAY, "blocos": [b2]}, {"blocos": [h2]}, {"blocos": [pb_do(h2)]}, LAY, 2)
pv2_como_v1, _ = B.valida_pedido_b({"layers": LAY, "blocos": [b2]}, {"blocos": [h2]}, {"blocos": [pb_do(h2)]}, LAY, 1)
pv1_como_v2, _ = B.valida_pedido_b({"layers": LAY, "blocos": [bv1]}, {"blocos": [hv1]}, {"blocos": [pb_do(hv1)]}, LAY, 2)
confere("v2: pedido novo íntegro pelo v2; pelo v1 recusado com IB3 v1; pedido antigo recusado pelo v2 com IB3 v2",
        pv2 == [] and any("IB3 (contrato v1" in x for x in pv2_como_v1) and any("IB3 (contrato v2" in x for x in pv1_como_v2))
import copy as _copy
falta = _copy.deepcopy(b2)
i_ = next(i for i, r in enumerate(falta["regs"]) if r[0] == B.B_DTOD)
del falta["regs"][i_]
troca = _copy.deepcopy(b2)
i_ = next(i for i, r in enumerate(troca["regs"]) if r[0] == B.B_DTOD)
troca["regs"][i_][0] = B.B_LANCA
ficticio = _copy.deepcopy(b2)
i_ = next(i for i, r in enumerate(ficticio["regs"]) if r[0] == B.B_DTOD)
ficticio["regs"].insert(i_, list(ficticio["regs"][i_]))
confere("v2: registro ausente, trocado ou fictício a mais continua recusado", not B.ib3(falta, LAY, 2) and not B.ib3(troca, LAY, 2) and not B.ib3(ficticio, LAY, 2))
mpv2 = B.medidas_pedido_b([(b2, h2)], LAY)
confere("v2: parcela B_LANCA/B_DTOD = 11/2 dos registros de posição (descritiva)", abs(mpv2["hb3"] / mpv2["hb4"] - 11 / 2) < 1e-9)
# E2E v2: janela sintética com perfil-f2b.json declarando contrato_b 2, extrator com --contrato 2
d_v2 = tempfile.mkdtemp(prefix="f2c-e2e-v2.")
sj_v2 = monta_janela(d_v2)
rc, R, md = roda_extrator(sj_v2, extra=("--contrato", "2"))
confere("E2E v2: com --contrato 2 a janela nova é íntegra, 9 pedidos no médio, status OK, resultado declara contrato_b 2",
        rc == 0 and R["contrato_b"] == 2 and R["integridade_problemas"] == [] and R["bloqueios"] == [] and R["textos"]["medio"]["pedidos_validos"] == 9 and all(R["textos"][t]["status"] == "OK" for t in F.TEXTOS))
rc, R, md = roda_extrator(sj_v2)
confere("E2E v2: dados novos com o extrator no contrato v1 (padrão) são recusados pelo contrato declarado, 0 válidos",
        R["contrato_b"] == 1 and R["textos"]["medio"]["pedidos_validos"] == 0 and any("contrato_b" in x for x in R["textos"]["medio"]["motivos"]))
ORDEM_FIX, CONTRATO_FIX = B.ORDEM_POS_V1, 1
d_v1 = tempfile.mkdtemp(prefix="f2c-e2e-v1.")
sj_v1 = monta_janela(d_v1)
rc, R, md = roda_extrator(sj_v1, extra=("--contrato", "2"))
confere("E2E v1: dados antigos (sem contrato_b) com --contrato 2 são recusados, 0 válidos", R["textos"]["medio"]["pedidos_validos"] == 0 and any("contrato_b" in x for x in R["textos"]["medio"]["motivos"]))
rc, R, md = roda_extrator(sj_v1)
confere("E2E v1: dados antigos pelo contrato v1 seguem íntegros", rc == 0 and R["contrato_b"] == 1 and R["textos"]["medio"]["pedidos_validos"] == 9)
print("TODOS OS TESTES PASSARAM" if falhas == 0 else "%d FALHA(S)" % falhas)
if falhas == 0 and not GUARDA:
    shutil.rmtree(RAIZ_TESTE)   # só o caminho literal que o mkdtemp devolveu
else:
    print("pasta de teste mantida:", RAIZ_TESTE)
sys.exit(1 if falhas else 0)
