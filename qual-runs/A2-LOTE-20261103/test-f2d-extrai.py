#!/usr/bin/env python3
"""Testes sem GPU do f2d-extrai.py: classificador (limites e bordas), rótulos de efeito por métrica, quadrado latino, comparador de hashes do gancho (um ULP com ids iguais falha)."""
import importlib.util
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
s = importlib.util.spec_from_file_location("c", os.path.join(AQUI, "f2d-extrai.py"))
C = importlib.util.module_from_spec(s)
s.loader.exec_module(C)
n = [0]


def ok(c, msg):
    if not c:
        print("FALHOU:", msg)
        sys.exit(1)
    n[0] += 1
    print("ok:", msg)


def cl(pn, pv, dn, dv, bn=None):
    return C.classifica_janela_c({"DO1": pv, "DO3": pn, "B2O3": bn if bn is not None else pn}, {"DO1": dv, "DO3": dn, "B2O3": dn})


ok(cl(101.5, 100, 100, 100)["HC3"] == "dentro do limite", "prefill +1,5% exato é dentro do limite")
ok(cl(101.6, 100, 100, 100)["HC3"] == "REGRESSÃO", "prefill +1,6% é regressão")
ok(cl(90, 100, 100, 100)["HC3"] == "dentro do limite", "queda de prefill nunca é regressão")
ok(cl(100, 100, 102, 100)["HC4"] == "dentro do limite", "decode +2% exato é dentro do limite")
ok(cl(100, 100, 102.1, 100)["HC4"] == "REGRESSÃO", "decode +2,1% é regressão")
ok(cl(100, 100, 100, 100, bn=103)["HC6"] == F_OK if (F_OK := C.F.classe_por_desvio(0.03)) else False, "HC6 em +3% usa a classe do critério comum")
r = cl(100, 100, 100, 100, bn=111)
ok(r["HC6"] == "SEM PROVA" and "HC6" in r["regressao"], "HC6 acima de 10% é SEM PROVA e conta como regressão")
ok(cl(None, 100, 100, 100)["HC3"] == "sem dado", "sem dado não vira aprovação")
ok(C.quadrado_latino_ok(C.ORDEM_CONTROLE), "a ordem do roteiro é um quadrado latino 3x3")
ok(not C.quadrado_latino_ok(["DO1"] * 9), "ordem degenerada recusada")
ok(not C.quadrado_latino_ok(C.ORDEM_CONTROLE[:8]), "ordem curta recusada")
base = [{"indice": 0, "aquecimento": False, "ids": [1, 2], "gerados": [3, 4], "logits": ["a", "b", "c"], "kv": [["k0", "v0"], ["k1", "v1"]]}]
import copy
ok(C.compara_gancho(base, copy.deepcopy(base)) == [], "original contra original passa")
mut = copy.deepcopy(base)
mut[0]["logits"][1] = "b1ulp"
d = C.compara_gancho(base, mut)
ok(len(d) == 1 and "passo 1" in d[0], "1 ULP num logit com ids iguais falha")
mut = copy.deepcopy(base)
mut[0]["kv"][1][1] = "x"
ok(any("camada 1 (V)" in x for x in C.compara_gancho(base, mut)), "KV divergente falha")
ok(C.compara_gancho([], []) != [], "conjunto vazio não passa")
ok(C.hash_bits([1.0]) == C.hash_bits([1.0]) and C.hash_bits([1.0]) != C.hash_bits([1.0000001]), "hash dos bits distingue 1 ULP")

# ---- efeito por métrica (plano da O3, decisão 6): distinguível só abaixo de -1% com as 3 medianas do novo todas abaixo das 3 do velho
ef = C.classifica_efeito
ok(ef(-0.02, [97, 98, 98.5], [100, 100.2, 100.4], 0.015)["classe"] == "efeito distinguível", "queda de 2% com as 3 medianas todas abaixo: efeito distinguível")
ok(ef(-0.02, [97, 99.9, 98.5], [100, 99.5, 100.4], 0.015)["classe"] == "sem efeito distinguível", "queda de 2% com medianas sobrepostas: sem efeito distinguível")
ok(ef(-0.011, [98.9, 98.9, 98.9], [100, 100, 100], 0.015)["classe"] == "efeito distinguível", "queda de 1,1% com medianas separadas: efeito distinguível")
ok(ef(-0.01, [99, 99, 99], [100, 100, 100], 0.015)["classe"] == "sem efeito distinguível", "queda de exatamente 1% não passa de -1%: sem efeito distinguível")
ok(ef(-0.005, [99.5, 99.5, 99.5], [100, 100, 100], 0.015)["classe"] == "sem efeito distinguível", "queda de 0,5% fica na zona do ruído: sem efeito distinguível")
ok(ef(0.008, [100.8] * 3, [100] * 3, 0.015)["classe"] == "sem efeito distinguível", "razão positiva abaixo do limite (+0,8%): sem efeito distinguível")
ok(ef(0.015, [101.5] * 3, [100] * 3, 0.015)["classe"] == "sem efeito distinguível", "+1,5% exato no prefill é dentro do limite: sem efeito distinguível")
ok(ef(0.0151, [101.51] * 3, [100] * 3, 0.015)["classe"] == "REGRESSÃO", "+1,51% no prefill é regressão")
ok(ef(0.02, [102] * 3, [100] * 3, 0.02)["classe"] == "sem efeito distinguível" and ef(0.021, [102.1] * 3, [100] * 3, 0.02)["classe"] == "REGRESSÃO", "limite do decode em +2%")
ok(ef(None, [1, 2, 3], [1, 2, 3], 0.015)["classe"] == "sem dado" and ef(-0.05, [1, 2], [1, 2, 3], 0.015)["classe"] == "sem dado" and ef(-0.05, [1, 2, None], [1, 2, 3], 0.015)["classe"] == "sem dado", "razão ou medianas ausentes: sem dado")
r = C.classifica_janela_c({"DO1": 100, "DO3": 98, "B2O3": 98}, {"DO1": 1.0, "DO3": 1.0, "B2O3": 1.0}, {"DO1": [100, 100.1, 100.2], "DO3": [97.9, 98, 98.1]}, {"DO1": [1.0, 1.0, 1.0], "DO3": [1.0, 1.0, 1.0]})
ok(r["efeito_prefill"]["classe"] == "efeito distinguível" and r["efeito_decode"]["classe"] == "sem efeito distinguível" and r["regressao"] == [], "uma métrica com efeito e a outra sem: rótulo por métrica, sem regressão")
r = C.classifica_janela_c({"DO1": 100, "DO3": 102, "B2O3": 102}, {"DO1": 1.0, "DO3": 1.0, "B2O3": 1.0}, {"DO1": [100] * 3, "DO3": [102] * 3}, {"DO1": [1.0] * 3, "DO3": [1.0] * 3})
ok(r["efeito_prefill"]["classe"] == "REGRESSÃO" and r["HC3"] == "REGRESSÃO" and "HC3" in r["regressao"], "prefill +2% é regressão no efeito e no HC3")
ok(C.classifica_janela_c({"DO1": 100, "DO3": 98, "B2O3": 98}, {"DO1": 1.0, "DO3": 1.0, "B2O3": 1.0})["efeito_prefill"]["classe"] == "sem dado", "sem as medianas por processo o efeito é sem dado")
ok(C.DEGRAU["1b"] == {"antes_ate": 11, "transicao": 12, "patamar_de": 13} and C.DEGRAU["3b"] == {"antes_ate": 10, "transicao": None, "patamar_de": 11}, "cortes por camada refixados com os dados da F2C")
ok(C.CONTRATO == 3, "o extrator da F2D lê o contrato v3 do nível B")
print("TODOS OS TESTES PASSARAM (%d)" % n[0])
