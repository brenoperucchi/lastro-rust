#!/usr/bin/env python3
"""Testes SEM GPU do f2-extrai.py: blocos sintéticos (2 camadas), invariantes I1 a I5 e I8, classificadores, controle de três braços, agregação por pedido e recusas.
    python3 -I test-f2-extrai.py        (usa mktemp -d para o teste de recusa; NÃO apaga nada)
"""
import importlib.util
import json
import os
import sys
import tempfile

AQUI = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("f2x", os.path.join(AQUI, "f2-extrai.py"))
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)

falhas = 0


def confere(nome, cond):
    global falhas
    if cond:
        print("ok:", nome)
    else:
        print("FALHOU:", nome)
        falhas += 1


LAY = 2
T2 = 1_000_000


def bloco(gen=0, m=8, pos_ini=0, durs=None, tb_mode="ok", t2=T2, r_in=10, r_out=10, perturba=None, lay=None):
    """Bloco sintético com LAY camadas. durs: dict (op, camada) -> ns (padrão 100)."""
    LAYX = lay or LAY
    ordem = F.ordem_esperada(LAYX)
    t = t2 + r_in
    entrada = t
    ops = []
    for (op, l) in ordem:
        d = (durs or {}).get((op, l), 100)
        ops.append([op, l, t, t + d])
        t += d
    saida = t
    t3 = saida + r_out
    b = {"gen": gen, "m": m, "pos_ini": pos_ini, "t2_ns": t2, "t3_ns": t3, "entrada_ns": entrada, "saida_ns": saida, "ops": ops}
    if tb_mode != "sem":
        tb = []
        for l in range(LAYX):
            ini = next(o[2] for o in ops if o[0] == F.H_SEQ and o[1] == l)
            fim = next(o[3] for o in ops if o[0] == F.H_SEQ and o[1] == l)
            tb.append({"camada": l, "gen": gen, "tb_obs_ns": ini + 5 if tb_mode != "zero" else 0, "tb_ns": fim + 20})
        if tb_mode == "causal":
            tb[0]["tb_obs_ns"] = next(o[2] for o in ops if o[0] == F.H_SEQ and o[1] == 0) - 5000
        b["tb"] = tb
    if perturba:
        perturba(b)
    return b


def pb_do(b, i=0):
    return {"gen": b["gen"], "m": b["m"], "pos_ini": b["pos_ini"], "t_ns": [0, 0, b["t2_ns"], b["t3_ns"], 0]}


# ---- estrutura e invariantes
b = bloco()
confere("ordem esperada: 6*L+7 operações", len(F.ordem_esperada(LAY)) == 6 * LAY + 7 and len(b["ops"]) == 19)
confere("bloco válido: I1, I2, I3, I5", F.i1(b) and F.i2(b) and F.i3(b, LAY) and F.i5(b, LAY))
confere("R_borda = entrada - t2 + t3 - saida", F.r_borda(b) == 20 and F.f2_ns(b) == F.soma_ops(b) + 20)
b2 = bloco(perturba=lambda x: x["ops"].pop())
confere("I3 falha com operação faltando", not F.i3(b2, LAY))
b3 = bloco(perturba=lambda x: x["ops"][5].__setitem__(2, x["ops"][5][2] - 500))
confere("I1 falha com carimbo fora de ordem", not F.i1(b3))
def _antes_de_t2(x):
    x["ops"][0][2] -= 5000
    x["entrada_ns"] = x["ops"][0][2] - 0
b4 = bloco(perturba=_antes_de_t2)
confere("I2 falha quando a entrada fica 5 us antes de t2 (R_borda fica truncada em zero)", not F.i2(b4))
confere("ressalva de borda só acima de 3%", not F.ressalva_borda(bloco()) and F.ressalva_borda(bloco(r_in=500, r_out=500)))
confere("I8 ok com tb válido; defasagem negativa NÃO é violação", F.i8_problemas(b, b["tb"], LAY) == []
        and F.i8_problemas(bloco(), [dict(c, tb_ns=c["tb_ns"] - 70) for c in b["tb"]], LAY) == [])
confere("I8: tb_obs == 0 acusado à parte (tb_obs ausente)", any("tb_obs ausente" in x for x in F.i8_problemas(bloco(tb_mode="zero"), bloco(tb_mode="zero")["tb"], LAY)))
bc = bloco(tb_mode="causal")
confere("I8: tb_obs anterior ao início de H_SEQ viola a causalidade", any("anterior ao início de H_SEQ" in x for x in F.i8_problemas(bc, bc["tb"], LAY)))
confere("I8: sem tb é acusado", F.i8_problemas(bloco(tb_mode="sem"), None, LAY) != [])

# ---- pedido (I4 contra o perfil-bloco, contagem de blocos, I8 só para a HI)
blocos = [bloco(gen=i, m=8, pos_ini=8 * i, t2=T2 + i * 1_000_000) for i in range(3)]
pf = {"layers": LAY, "blocos": blocos}
pbb = {"blocos": [pb_do(x) for x in blocos]}
prob, hi, ress = F.valida_pedido_f2(pf, pbb, LAY)
confere("pedido válido: sem problemas, sem I8, sem ressalva", prob == [] and hi == [] and not ress)
pbb2 = {"blocos": [pb_do(x) for x in blocos][:2]}
confere("pedido com número de blocos diferente do perfil-bloco: inválido", F.valida_pedido_f2(pf, pbb2, LAY)[0] != [])
pbb3 = {"blocos": [dict(pb_do(x), pos_ini=999) if i == 1 else pb_do(x) for i, x in enumerate(blocos)]}
confere("I4: pos_ini diferente do perfil-bloco invalida o pedido", any("I4" in x for x in F.valida_pedido_f2(pf, pbb3, LAY)[0]))
blocos_i8 = [bloco(gen=i, m=8, pos_ini=8 * i, t2=T2 + i * 1_000_000, tb_mode="zero" if i == 1 else "ok") for i in range(3)]
prob, hi, _ = F.valida_pedido_f2({"layers": LAY, "blocos": blocos_i8}, {"blocos": [pb_do(x) for x in blocos_i8]}, LAY)
confere("falha de I8 NÃO invalida o pedido (só a HI)", prob == [] and hi != [])
confere("layers errado invalida o pedido", F.valida_pedido_f2({"layers": 5, "blocos": blocos}, pbb, LAY)[0] != [])

# ---- classificadores (limiares do critério)
confere("HA: >= 0,80 dentro; < 0,50 fora; entre, sem leitura", (F.classifica_ha(0.8), F.classifica_ha(0.49), F.classifica_ha(0.6), F.classifica_ha(None)) == ("dentro", "fora", "sem leitura", "sem dado"))
confere("HB: < 25% fora; >= 40% e maior soma dentro", (F.classifica_hb(0.2, True), F.classifica_hb(0.45, True), F.classifica_hb(0.45, False), F.classifica_hb(0.3, True)) == ("fora", "dentro", "sem leitura", "sem leitura"))
confere("HC: <= 5% dentro; > 15% fora", (F.classifica_hc(0.05), F.classifica_hc(0.16), F.classifica_hc(0.1)) == ("dentro", "fora", "sem leitura"))
confere("HD: <= 1,5 dentro; > 3,0 fora", (F.classifica_hd(1.5), F.classifica_hd(3.1), F.classifica_hd(2.0)) == ("dentro", "fora", "sem leitura"))
confere("HE: <= 5% dentro; > 20% fora", (F.classifica_he(0.05), F.classifica_he(0.21), F.classifica_he(0.1)) == ("dentro", "fora", "sem leitura"))
confere("HH: >= 1,5 dentro, senão descritivo", (F.classifica_hh(1.6), F.classifica_hh(1.0)) == ("dentro", "descritivo"))
lo, hi_ = F.faixa_hf(4)
confere("HF: faixa 0,75 a 1,25 vezes r/8", abs(lo - 0.375) < 1e-9 and abs(hi_ - 0.625) < 1e-9)

# ---- HE por pedido
bl_he = [bloco(durs={(F.H_ROUTE, 0): 100}) for _ in range(5)]
bl_he[2] = bloco(durs={(F.H_ROUTE, 0): 5000})
he = F.he_do_pedido(bl_he)
confere("HE: só a operação acima de 10x a mediana do seu grupo conta", he is not None and 0.1 < he < 0.9)
confere("HE: sem outlier dá 0", F.he_do_pedido([bloco() for _ in range(4)]) == 0)

# ---- agregação POR PEDIDO e mediana entre pedidos
def pedido(dur_pre):
    return {"layers": LAY, "blocos": [bloco(gen=i, m=8, pos_ini=8 * i, t2=T2 + i * 1_000_000, durs={(F.H_PRE_LAYER, 0): dur_pre, (F.H_PRE_LAYER, 1): dur_pre}) for i in range(4)] + [bloco(gen=4, m=2, pos_ini=32)]}
peds = [pedido(100), pedido(100), pedido(1000)]
med = F.medidas_celula(peds, LAY, 2)
por = [F.medidas_pedido(F.elegiveis(p), LAY)["hb_frac"] for p in peds]
confere("elegíveis: sem o primeiro bloco e sem o resto (m < 8)", len(F.elegiveis(peds[0])) == 3)
confere("HB da célula é a mediana entre pedidos (não a mistura de blocos)", abs(med["hb_frac"] - sorted(por)[1]) < 1e-12 and por[2] > por[0])
confere("célula guarda os valores por pedido e a faixa de HF do resto", len(med["por_pedido"]) == 3 and med["faixa_hf"] is not None)
confere("HF: razão do bloco do resto contra a mediana dos cheios", med["hf_razao"] is not None)
hi_rel = F.relatorio_hi(peds, None, LAY)
confere("HI: descritiva, com sinal, por camada, só dos blocos elegíveis", hi_rel["descritivo"] and hi_rel["amostras"] == 9 and len(hi_rel["mediana_por_camada"]["defasagem"]) == LAY)

# ---- controle de três braços
def proc(b, pre, passo=1.0, **kw):
    d = {"braco": b, "prefill_ms_mediana": pre, "passo_ms_mediana": passo, "valido": True,
         "imprimiu_bloco": b != "D", "imprimiu_f2": b == "B", "produziu_bloco": b != "D", "produziu_f2": b == "B"}
    d.update(kw)
    return d


def nove(pre_a=100.0, pre_b=100.0, pre_d=100.0):
    return [proc(b, {"D": pre_d, "A": pre_a, "B": pre_b}[b]) for b in F.ORDEM_CONTROLE]


confere("controle: |B contra A| <= 3% comprovada", F.classifica_controle_f2(nove(100, 102.5))["classe"] == "comprovada")
confere("controle: <= 10% com ressalva", F.classifica_controle_f2(nove(100, 108))["classe"] == "com ressalva")
confere("controle: > 10% SEM PROVA", F.classifica_controle_f2(nove(100, 120))["classe"] == "SEM PROVA")
confere("controle: B contra D só apoio (A e B iguais, D longe: continua comprovada, apoio SEM PROVA)", (lambda r: r["classe"] == "comprovada" and r["classe_apoio"] == "SEM PROVA")(F.classifica_controle_f2(nove(100, 100, 150))))
confere("controle: ordem diferente = incompleto", F.classifica_controle_f2(list(reversed(nove())))["classe"] == "controle incompleto")
confere("controle: 8 processos = incompleto", F.classifica_controle_f2(nove()[:8])["classe"] == "controle incompleto")
confere("controle: orçamento insuficiente = incompleto", F.classifica_controle_f2(nove(), orcamento_ok=False)["classe"] == "controle incompleto")
confere("controle: ordem.txt desconfere = incompleto, com o motivo", F.classifica_controle_f2(nove(), incompleto_motivo="x")["motivo"] == "x")
ruim = nove(); ruim[4] = dict(ruim[4], imprimiu_f2=True)
confere("controle: D que imprime perfil-f2: ligado = incompleto", F.classifica_controle_f2(ruim)["classe"] == "controle incompleto")
ruim = nove(); ruim[2] = dict(ruim[2], produziu_f2=False)
confere("controle: B sem perfil-f2.json = incompleto", F.classifica_controle_f2(ruim)["classe"] == "controle incompleto")
ruim = nove(); ruim[1] = dict(ruim[1], valido=False)
confere("controle: processo inválido (teto, rc) = incompleto", F.classifica_controle_f2(ruim)["classe"] == "controle incompleto")
confere("controle: ms por bloco usa 22 blocos executados", F.BLOCOS_POR_PEDIDO_MEDIO == 22 and abs(F.classifica_controle_f2(nove(100, 102.2))["principal"]["ms_por_bloco"] - 2.2 / 22) < 1e-9)

# ---- recusas e saída: pasta sem orçamento
d = tempfile.mkdtemp(prefix="f2-extrai-teste.")
os.makedirs(os.path.join(d, "celulas"))
rc = F.main(["--saida", d])
confere("sem orcamento.json: RECUSADO (código 2) e resultado marcado recusado", rc == 2 and json.load(open(os.path.join(d, "f2-host-resultado.json")))["recusado"] is True)
confere("pasta inexistente: código 2", F.main(["--saida", os.path.join(d, "nao-existe")]) == 2)
json.dump({"modelos": ["1b", "3b"]}, open(os.path.join(d, "orcamento.json"), "w"))
confere("orçamento com dois modelos: RECUSADO", F.main(["--saida", d]) == 2)

# ======================================================================================================================================
# Ponta a ponta (rodada 154, P2-1): janela sintética COMPLETA de 1B no formato do Rust, `main` -> processa -> classes -> f2-host-resultado.md/.json.
# O manifesto de integridade é escrito já com os shas simulados em OK (o modo de teste do roteiro, SEM_SHA_FIXO, não é usado aqui nem é afrouxado em produção);
# a leitura bruta do cmp (X.le_celula_lastro: monitor, portão, cmdline) é substituída por um leitor mínimo das medidas sintéticas; todo o resto é o código real do extrator.
import hashlib
import copy
L1B = 24
G = list(range(64))
X_ = F.X


def sha_b(b):
    return hashlib.sha256(b).hexdigest()


def fake_le_celula_lastro(dd, meta, refs):
    meds = []
    for pn in sorted(x for x in os.listdir(dd) if x.startswith("p") and x[1:].isdigit()):
        mj = X_.le_json(os.path.join(dd, pn, "saida", "medidas-%s.json" % meta["modelo"]))
        for e in mj["medidas"]:
            if not e.get("descartada"):
                meds.append({"processo": pn, "rotulo": e["rotulo"], "prefill_ms": e["prefill_ms"], "mediana_passos_descritiva": 1.0, "gerados": e["gerados"], "ids_n": len(e["ids"])})
    return {"meds": meds, "cargas_limite_ms": [], "esperado_n": None, "temp_max": None, "slots_impressos": []}, []


def peds_perfil(t, n_meas, trunc=(), i8_aq=False):
    plano = F.I.plano_blocos(F.N_POS[t])
    pbs, f2s = [], []
    for i in range(n_meas + 1):
        bl, pbb_, pos = [], [], 0
        for bi, mb in enumerate(plano):
            b = bloco(gen=bi, m=mb, pos_ini=pos, t2=T2 + bi * 1_000_000, lay=L1B, tb_mode="zero" if (i8_aq and i == 0) else "ok")
            pos += mb
            bl.append(b)
            pbb_.append({"m": mb, "conserva": True, "gen": b["gen"], "pos_ini": b["pos_ini"], "t_ns": [0, 0, b["t2_ns"], b["t3_ns"], 0]})
        if i in trunc:
            keep = sorted({0, 1, len(bl) - 1})
            bl, pbb_ = [bl[k] for k in keep], [pbb_[k] for k in keep]
        pbs.append({"modelo": "1b", "indice": i, "aquecimento": i == 0, "n": F.N_POS[t], "plano_m": plano, "blocos": pbb_, "valido": True, "problemas": []})
        f2s.append({"modelo": "1b", "indice": i, "layers": L1B, "blocos": bl})
    return pbs, f2s


def monta_janela(d, longo_n=5, trunc_curto=(), smoke_i8=False, medio_i3=None, b_pre=100.5, mut_pb_curto=None):
    w = lambda p, txt: (os.makedirs(os.path.dirname(p), exist_ok=True), open(p, "w", encoding="utf-8").write(txt))
    ids = {t: list(range(F.N_POS[t])) for t in F.TEXTOS}
    w(os.path.join(d, "ids.json"), json.dumps({"modelos": {"1b": {t: {"ids": ids[t]} for t in F.TEXTOS}}}))
    w(os.path.join(d, "refs", "ref5c-1b.json"), json.dumps({"casos": [{"nome": t, "gerados": G} for t in F.TEXTOS]}))
    w(os.path.join(d, "textos.json"), json.dumps({"textos": []}))
    obs = {"cmp-ids-prompt": sha_b(open(os.path.join(d, "ids.json"), "rb").read()), "ref5c-1b": sha_b(open(os.path.join(d, "refs", "ref5c-1b.json"), "rb").read()),
           "textos-json": sha_b(open(os.path.join(d, "textos.json"), "rb").read())}
    sj = os.path.join(d, "janela")
    for k in F.FIXO_CHAVES + ("tokenizer-1b",):
        obs.setdefault(k, sha_b(k.encode()))
    w(os.path.join(sj, "integridade-antes.txt"), "".join("%s %s\n" % kv for kv in obs.items()))
    w(os.path.join(sj, "integridade-depois.txt"), "".join("%s %s\n" % kv for kv in obs.items()))
    w(os.path.join(sj, "integridade-fixos.txt"), "".join("%s %s %s OK\n" % (k, h, h) for k, h in obs.items()) + "potencia 370.00_W 370.00_W OK\n")
    w(os.path.join(sj, "ambiente.txt"), "uuid=GPU-b22d9465-8552-9f97-812e-380b590645f8\npotencia_inicio=370.00_W\npotencia_fim=370.00_W\n")
    w(os.path.join(sj, "orcamento.json"), json.dumps({"modelos": ["1b"], "nao_rodou": [], "cortes": [], "decorrido_s": 500, "limite_s": 1200}))
    ordem = ["01 smoke-1b-L100-curto-B n=1 braco=B inicio=0s fim=10s rc=0"]
    for k, b in enumerate(F.ORDEM_CONTROLE):
        ordem.append("%02d ctrl-1b-L100-medio-%02d-%s n=3 braco=%s inicio=%ds fim=%ds rc=0" % (k + 2, k + 1, b, b, 10 + 20 * k, 30 + 20 * k))
    w(os.path.join(sj, "ordem.txt"), "\n".join(ordem) + "\n")

    def pasta(nome, t, n, braco, pre, trunc=(), i8=False, i3=False, mut=None):
        pd = os.path.join(sj, "celulas", nome)
        w(os.path.join(pd, "celula.json"), json.dumps({"lado": "lastro", "serie": "L100", "modelo": "1b", "texto": t, "estado": "quente", "n_medidas": n}))
        w(os.path.join(pd, "p1", "stdout.txt"), "perfil-bloco: %s\nperfil-f2: %s\n" % ("desligado" if braco == "D" else "ligado", "ligado" if braco == "B" else "desligado"))
        med = [{"rotulo": "r%d" % i, "descartada": i == 0, "falha": None, "gerados": G, "ids": ids[t], "prefill_ms": pre} for i in range(n + 1)]
        w(os.path.join(pd, "p1", "saida", "medidas-1b.json"), json.dumps({"modelo": "1b", "texto": t, "medidas": med}))
        if braco != "D":
            pbs, f2s = peds_perfil(t, n, trunc, i8)
            if i3:
                f2s[1]["blocos"][0]["ops"].pop()
            doc = {"tipo": "perfil-bloco", "versao": 2, "pedidos": pbs}
            if mut:
                mut(doc)
            w(os.path.join(pd, "p1", "saida", "perfil-bloco.json"), json.dumps(doc))
            if braco == "B":
                w(os.path.join(pd, "p1", "saida", "perfil-f2.json"), json.dumps({"tipo": "perfil-f2", "versao": 1, "pedidos": f2s}))
    pasta("smoke-1b-L100-curto-B", "curto", 1, "B", b_pre, i8=smoke_i8)
    for k, b in enumerate(F.ORDEM_CONTROLE):
        pasta("ctrl-1b-L100-medio-%02d-%s" % (k + 1, b), "medio", 3, b, {"D": 100.0, "A": 100.0, "B": b_pre}[b], i3=(medio_i3 == k + 1))
    pasta("lastro-1b-L100-curto-quente", "curto", 5, "B", b_pre, trunc=trunc_curto, mut=mut_pb_curto)
    pasta("lastro-1b-L100-longo-quente", "longo", longo_n, "B", b_pre)
    return sj


def roda_extrator(sj):
    base = os.path.dirname(sj)
    rc = F.main(["--saida", sj, "--ids", os.path.join(base, "ids.json"), "--refs", os.path.join(base, "refs"), "--textos", os.path.join(base, "textos.json")])
    return rc, json.load(open(os.path.join(sj, "f2-host-resultado.json"))), open(os.path.join(sj, "f2-host-resultado.md"), encoding="utf-8").read()


F.I.X.le_celula_lastro = fake_le_celula_lastro
try:
    base_d = tempfile.mkdtemp(prefix="f2-e2e-base.")
    rc, R, md = roda_extrator(monta_janela(base_d))
    confere("E2E base: código 0, janela íntegra, smoke OK, sem bloqueios, controle comprovada, B igual a D nos ids", rc == 0 and R["integridade_problemas"] == [] and R["smoke"]["ok"] and R["bloqueios"] == [] and R["controle"]["classe"] == "comprovada" and R["ids_B_igual_D"])
    confere("E2E base: médio com 9 pedidos válidos e curto e longo com 5, status OK", R["textos"]["medio"]["pedidos_validos"] == 9 and R["textos"]["curto"]["pedidos_validos"] == 5 and R["textos"]["longo"]["pedidos_validos"] == 5 and all(R["textos"][t]["status"] == "OK" for t in F.TEXTOS))
    hm = R["textos"]["medio"]["hipoteses"]
    confere("E2E base: classes do médio conforme os números sintéticos (HA sem leitura, HC fora, HD e HE dentro, HG sem dado, HH e HF descritivos)", (hm["HA"], hm["HC"], hm["HD"], hm["HE"], hm["HG"], hm["HH"], hm["HF"]) == ("sem leitura", "fora", "dentro", "dentro", "sem dado", "descritivo", "descritivo"))
    hc_ = R["textos"]["curto"]["hipoteses"]
    confere("E2E base: no curto HD e HE descritivas; HI descritiva com amostras", hc_["HD"] == "descritivo" and hc_["HE"] == "descritivo" and hc_["HI"] == "descritivo" and R["textos"]["curto"]["hi"]["amostras"] > 0)
    confere("E2E base: relatório .md com controle, os três textos, orçamento e sem PASSA como veredito", "Controle do custo" in md and "## Texto medio" in md and "## Texto curto" in md and "## Texto longo" in md and "## Orçamento" in md and "comprovada" in md and "PASSA nem conclui causa" in md)
    # P2 da rev-1: pedidos com blocos ausentes nos DOIS perfis (plano incompleto) não contam
    d2 = tempfile.mkdtemp(prefix="f2-e2e-trunc.")
    rc, R, md = roda_extrator(monta_janela(d2, trunc_curto=(1, 2, 3, 4, 5)))
    cu = R["textos"]["curto"]
    confere("E2E blocos removidos dos dois perfis em 5 pedidos do curto: 0 válidos, todas as H do curto SEM PROVA, motivos e bruto conservados", rc == 0 and cu["pedidos_validos"] == 0 and set(cu["hipoteses"].values()) == {"SEM PROVA"} and any("plano de blocos" in x for x in cu["motivos"]) and cu["status"] != "OK" and R["textos"]["medio"]["pedidos_validos"] == 9)
    d3 = tempfile.mkdtemp(prefix="f2-e2e-um.")
    rc, R, md = roda_extrator(monta_janela(d3, trunc_curto=(2,)))
    cu = R["textos"]["curto"]
    confere("E2E um pedido inválido de 5 (4 válidos, acima do limite de 3): classes seguem calculadas, motivo registrado", cu["pedidos_validos"] == 4 and cu["hipoteses"]["HC"] == "fora" and any("plano de blocos" in x for x in cu["motivos"]))
    d4 = tempfile.mkdtemp(prefix="f2-e2e-tres.")
    rc, R, md = roda_extrator(monta_janela(d4, trunc_curto=(1, 4)))
    confere("E2E limite de três: 3 válidos ainda classificam", R["textos"]["curto"]["pedidos_validos"] == 3 and R["textos"]["curto"]["hipoteses"]["HC"] == "fora")
    d5 = tempfile.mkdtemp(prefix="f2-e2e-dois.")
    rc, R, md = roda_extrator(monta_janela(d5, trunc_curto=(1, 3, 5)))
    confere("E2E dois válidos (abaixo de 3): SEM PROVA pelo mínimo", R["textos"]["curto"]["pedidos_validos"] == 2 and set(R["textos"]["curto"]["hipoteses"].values()) == {"SEM PROVA"})
    # P2 da rev-1: I8 do aquecimento do smoke entra na decisão do smoke
    d6 = tempfile.mkdtemp(prefix="f2-e2e-smoke.")
    rc, R, md = roda_extrator(monta_janela(d6, smoke_i8=True))
    confere("E2E I8 do aquecimento do smoke: smoke.ok falso com o motivo, bloqueio aplicado, todas as H SEM PROVA", (not R["smoke"]["ok"]) and "tb_obs ausente" in R["smoke"]["motivo"] and any("smoke FALHOU" in x for x in R["bloqueios"]) and set(R["textos"]["medio"]["hipoteses"].values()) == {"SEM PROVA"})
    # médio: um pedido I3 inválido deixa 8 de 9
    d7 = tempfile.mkdtemp(prefix="f2-e2e-i3.")
    rc, R, md = roda_extrator(monta_janela(d7, medio_i3=3))
    confere("E2E médio com um pedido I3 inválido: 8 válidos, SEM PROVA pelo mínimo de 9, curto segue", R["textos"]["medio"]["pedidos_validos"] == 8 and set(R["textos"]["medio"]["hipoteses"].values()) == {"SEM PROVA"} and R["textos"]["curto"]["pedidos_validos"] == 5 and any("I3" in x for x in R["textos"]["medio"]["motivos"]))
    # longo cortado para 3 medidas (perfil-f2.json com 4 entradas)
    d8 = tempfile.mkdtemp(prefix="f2-e2e-corte.")
    rc, R, md = roda_extrator(monta_janela(d8, longo_n=3))
    confere("E2E longo cortado para 3 medidas: 3 válidos, status OK, classes calculadas", R["textos"]["longo"]["pedidos_validos"] == 3 and R["textos"]["longo"]["status"] == "OK" and R["textos"]["longo"]["hipoteses"]["HC"] == "fora")
    # controle fora da faixa: B 20% acima de A
    d9 = tempfile.mkdtemp(prefix="f2-e2e-ctrl.")
    rc, R, md = roda_extrator(monta_janela(d9, b_pre=120.0))
    confere("E2E controle com B 20% acima de A: SEM PROVA, todas as H do modelo SEM PROVA", R["controle"]["classe"] == "SEM PROVA" and all(set(R["textos"][t]["hipoteses"].values()) == {"SEM PROVA"} for t in F.TEXTOS))
    # rodada 155: esquema do arquivo e identidade por pedido do perfil-base
    def _ver(doc): doc["versao"] = 99
    def _modelo(doc): [x.__setitem__("modelo", "3b") for x in doc["pedidos"][1:]]
    def _indice(doc): [x.__setitem__("indice", 99) for x in doc["pedidos"][1:]]
    def _aquec(doc): [x.__setitem__("aquecimento", True) for x in doc["pedidos"][1:]]
    def _um(doc): doc["pedidos"][3]["modelo"] = "3b"
    def _um_ind(doc): doc["pedidos"][2]["indice"] = 99
    for nome_, mut_ in (("versão 99", _ver), ("modelo trocado nos 5", _modelo), ("índice 99 nos 5", _indice), ("marca de aquecimento incoerente nos 5", _aquec)):
        dx = tempfile.mkdtemp(prefix="f2-e2e-id.")
        rc, R, md = roda_extrator(monta_janela(dx, mut_pb_curto=mut_))
        cu = R["textos"]["curto"]
        confere("E2E %s no perfil-bloco do curto: 0 válidos, todas as H do curto SEM PROVA, motivo preservado, médio intacto" % nome_, cu["pedidos_validos"] == 0 and set(cu["hipoteses"].values()) == {"SEM PROVA"} and cu["status"] != "OK" and cu["motivos"] != [] and R["textos"]["medio"]["pedidos_validos"] == 9)
    for nome_, mut_ in (("um pedido de modelo trocado", _um), ("um pedido de índice 99", _um_ind)):
        dx = tempfile.mkdtemp(prefix="f2-e2e-id1.")
        rc, R, md = roda_extrator(monta_janela(dx, mut_pb_curto=mut_))
        cu = R["textos"]["curto"]
        confere("E2E %s: restam 4 válidos, motivo preservado, classes seguem" % nome_, cu["pedidos_validos"] == 4 and cu["hipoteses"]["HC"] == "fora" and any("perfil-bloco: " in x for x in cu["motivos"]))
    # manifesto sem prova de potência: recusa
    d10 = tempfile.mkdtemp(prefix="f2-e2e-pot.")
    sj = monta_janela(d10)
    open(os.path.join(sj, "ambiente.txt"), "w").write("potencia_inicio=370.00_W\npotencia_fim=300.00_W\n")
    rc, R, md = roda_extrator(sj)
    confere("E2E potência 300 W ao fim: RECUSADO (código 2), nada classificado", rc == 2 and R.get("recusado") is True)
finally:
    pass
print("pasta de teste (não apagada):", d)
print("TODOS OS TESTES PASSARAM" if falhas == 0 else "%d FALHA(S)" % falhas)
sys.exit(1 if falhas else 0)
