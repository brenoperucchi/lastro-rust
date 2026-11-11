#!/usr/bin/env python3
"""Testes sem GPU do f2c-extrai.py: classificador (limites e bordas), quadrado latino, comparador de hashes do gancho (um ULP com ids iguais falha)."""
import importlib.util
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
s = importlib.util.spec_from_file_location("c", os.path.join(AQUI, "f2c-extrai.py"))
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
    return C.classifica_janela_c({"Dvelho": pv, "Dnovo": pn, "B2novo": bn if bn is not None else pn}, {"Dvelho": dv, "Dnovo": dn, "B2novo": dn})


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
ok(not C.quadrado_latino_ok(["Dvelho"] * 9), "ordem degenerada recusada")
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
print("TODOS OS TESTES PASSARAM (%d)" % n[0])
