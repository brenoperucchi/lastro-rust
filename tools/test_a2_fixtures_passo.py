"""Testes do gerador de fixtures do passo de M posições (U0 do E4a). Uso: taskset -c 16-27 tools/.venv-ref/bin/python -I -B tools/test_a2_fixtures_passo.py"""
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import a2_fixtures_passo as F  # noqa: E402

A1 = os.path.join(HERE, "..", "rust", "moe-rt", "tests", "fixtures", "a1_sim_token_barreira.json")


def traco(caso):
    L, T, K = caso["layers"], len(caso["routing"]), caso["topk"]
    a = np.zeros((L, T, K), dtype=np.int64)
    for t in range(T):
        for l in range(L):
            a[l, t] = caso["routing"][t][l]
    return a


class Testes(unittest.TestCase):
    def test_passo_1_reproduz_as_fixtures_do_a1(self):
        """Com todos os passos de tamanho 1 o SimPasso dá as contagens do a1_sim_token_barreira.json (nos 64 casos)."""
        with open(A1) as f:
            a1 = json.load(f)
        for c in a1["casos"]:
            a = traco(dict(c, topk=c["topk"]))
            esp = F.esperado(c["layers"], c["experts"], c["topk"], c["slots"], c["backlog"], a, [1] * len(c["routing"]))
            self.assertEqual(esp, c["esperado"], f"caso {c['caso']}")

    def test_o_arquivo_gravado_bate_com_a_regeracao(self):
        with open(F.SAIDA) as f:
            gravado = json.load(f)
        self.assertEqual(gravado, F.gera())

    def test_passos_cobrem_as_posicoes_e_respeitam_o_limite(self):
        for c in F.gera()["casos"]:
            self.assertEqual(sum(c["passos"]), c["posicoes"])
            self.assertTrue(all(1 <= m <= F.M_MAX for m in c["passos"]))
            self.assertEqual(len(c["routing"]), c["posicoes"])

    def test_invariantes_das_contagens(self):
        """needed = hits + cpu; needed = posições·camadas·topk (um pedido por posição, camada e posto)."""
        for c in F.gera()["casos"]:
            e = c["esperado"]
            self.assertEqual(e["needed"], e["hits"] + e["cpu"], c["caso"])
            self.assertEqual(e["needed"], c["posicoes"] * c["layers"] * c["topk"], c["caso"])

    def test_cobertura_dos_tipos_de_passo(self):
        casos = F.gera()["casos"]
        self.assertTrue(any(len(set(c["passos"])) > 1 and 8 in c["passos"] and 1 in c["passos"] for c in casos))
        self.assertTrue(any(c["passos"][-1] < c["passos"][0] for c in casos), "falta cauda parcial")
        for m in (2, 4, 8):
            self.assertTrue(any(c["passos"][0] == m and len(c["passos"]) > 1 for c in casos), m)

    def test_o_passo_muda_as_contagens_de_cache(self):
        """A fixture não é degenerada: em vários casos M>1 dá contagens diferentes de M=1 (a tabela é fixa dentro do passo)."""
        dif = 0
        for c in F.gera()["casos"]:
            a = traco(c)
            um = F.esperado(c["layers"], c["experts"], c["topk"], c["slots"], c["backlog"], a, [1] * c["posicoes"])
            if um != c["esperado"] and max(c["passos"]) > 1:
                dif += 1
        self.assertGreaterEqual(dif, 30)

    def test_mutante_tamanho_do_passo_ignorado(self):
        """Rodar com passos errados (tudo M=2 em vez do tipo do caso) muda alguma contagem: o tamanho do passo conta."""
        mudou = 0
        for c in F.gera()["casos"]:
            if c["passos"][0] == 2 and len(set(c["passos"])) == 1:
                continue
            a = traco(c)
            errado = F.esperado(c["layers"], c["experts"], c["topk"], c["slots"], c["backlog"], a, F.passos_fixos(c["posicoes"], 2))
            mudou += errado != c["esperado"]
        self.assertGreater(mudou, 0)


if __name__ == "__main__":
    unittest.main()
