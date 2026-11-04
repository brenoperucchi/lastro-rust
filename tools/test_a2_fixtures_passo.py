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
GERADO = F.gera()
CASOS = GERADO["casos"]


def traco(caso):
    L, T, K = caso["layers"], len(caso["routing"]), caso["topk"]
    a = np.zeros((L, T, K), dtype=np.int64)
    for t in range(T):
        for l in range(L):
            a[l, t] = caso["routing"][t][l]
    return a


def roda(c, cls=None, passos=None):
    return F.esperado(c["layers"], c["experts"], c["topk"], c["slots"], c["backlog"], traco(c), passos or c["passos"], cls)


# Mutantes das regras do passo (um por vez). Cada um tem de mudar alguma contagem do arquivo gravado.
class RelogioUm(F.SimPasso):
    """O relógio da frequência anda 1 por camada (o serve_layer de hoje), não m."""
    def layer_m(self, l, es, ns):
        super().layer_m(l, es, ns)
        self.step -= self.M - 1


class RelogioMax(F.SimPasso):
    """O relógio anda max_positions por camada, mesmo no passo parcial."""
    def layer_m(self, l, es, ns):
        super().layer_m(l, es, ns)
        self.step += F.M_MAX - self.M


class TouchUmaVez(F.SimPasso):
    """A frequência é tocada uma vez por chave da união, não n_e vezes."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._tocadas = set()

    def touch_freq(self, key):
        if (key, self.step) in self._tocadas:
            return
        self._tocadas.add((key, self.step))
        super().touch_freq(key)


class UniaoOrdenada(F.SimPasso):
    """A união sai ordenada por chave, não pela primeira aparição."""
    def uniao(self, blk_l):
        es, ns = super().uniao(blk_l)
        par = sorted(zip(es, ns))
        return [e for e, _ in par], [n for _, n in par]


class UniaoInversa(F.SimPasso):
    """A união sai em ordem inversa (muda a ordem das admissões)."""
    def uniao(self, blk_l):
        es, ns = super().uniao(blk_l)
        return es[::-1], ns[::-1]


MUTANTES = (RelogioUm, RelogioMax, TouchUmaVez, UniaoOrdenada, UniaoInversa)


class Testes(unittest.TestCase):
    def test_passo_1_reproduz_as_fixtures_do_a1(self):
        """Com todos os passos de tamanho 1 o SimPasso dá as contagens do a1_sim_token_barreira.json (nos 64 casos)."""
        with open(A1) as f:
            a1 = json.load(f)
        for c in a1["casos"]:
            self.assertEqual(roda(c, passos=[1] * len(c["routing"])), c["esperado"], f"caso {c['caso']}")

    def test_o_arquivo_gravado_bate_com_a_regeracao(self):
        with open(F.SAIDA) as f:
            self.assertEqual(json.load(f), GERADO)

    def test_composicao_do_arquivo(self):
        tipos = [c["tipo"] for c in CASOS]
        self.assertEqual((tipos.count("curto"), tipos.count("longo")), (F.CASOS, F.LONGOS))
        self.assertEqual([t for t in tipos if t.startswith("dirigido")], ["dirigido_relogio", "dirigido_mao_2_2", "dirigido_mao_1_1_1_1"])
        self.assertTrue(all(8 <= c["posicoes"] <= 40 for c in CASOS if c["tipo"] == "curto"))
        self.assertTrue(all(600 <= c["posicoes"] <= 1000 for c in CASOS if c["tipo"] == "longo"))

    def test_passos_cobrem_as_posicoes_e_respeitam_o_limite(self):
        for c in CASOS:
            self.assertEqual(sum(c["passos"]), c["posicoes"])
            self.assertTrue(all(1 <= m <= F.M_MAX for m in c["passos"]))
            self.assertEqual(len(c["routing"]), c["posicoes"])

    def test_invariantes_das_contagens(self):
        """needed = hits + cpu; needed = posições·camadas·topk (um pedido por posição, camada e posto)."""
        for c in CASOS:
            e = c["esperado"]
            self.assertEqual(e["needed"], e["hits"] + e["cpu"], c["caso"])
            self.assertEqual(e["needed"], c["posicoes"] * c["layers"] * c["topk"], c["caso"])

    def test_cobertura_dos_tipos_de_passo(self):
        self.assertTrue(any(len(set(c["passos"])) > 1 and 8 in c["passos"] and 1 in c["passos"] for c in CASOS))
        self.assertTrue(any(c["passos"][-1] < c["passos"][0] for c in CASOS), "falta cauda parcial")
        for m in (2, 4, 8):
            self.assertTrue(any(c["passos"][0] == m and len(c["passos"]) > 1 for c in CASOS), m)

    def test_o_passo_muda_as_contagens_de_cache(self):
        """A fixture não é degenerada: em vários casos M>1 dá contagens diferentes de M=1 (a tabela é fixa dentro do passo)."""
        dif = sum(1 for c in CASOS if max(c["passos"]) > 1 and roda(c, passos=[1] * c["posicoes"]) != c["esperado"])
        self.assertGreaterEqual(dif, 30)

    def test_valores_a_mao_da_tabela_fixa_no_passo(self):
        """Expert 0 nas quatro posições, 1 slot, backlog 1. Em passos [2,2] a tabela do 2º passo já tem o expert (cópia da barreira):
        as duas primeiras posições vão à CPU e as duas seguintes acertam. Em passos de 1 só a primeira vai à CPU."""
        por_tipo = {c["tipo"]: c["esperado"] for c in CASOS}
        self.assertEqual(por_tipo["dirigido_mao_2_2"], {"needed": 4, "hits": 2, "cpu": 2, "copies_bg": 1, "dropped": 0, "copies_deferred": 0})
        self.assertEqual(por_tipo["dirigido_mao_1_1_1_1"], {"needed": 4, "hits": 3, "cpu": 1, "copies_bg": 1, "dropped": 0, "copies_deferred": 0})

    def test_caso_dirigido_do_relogio(self):
        """Valores da revisão 58: com o relógio andando m, hits 96; com o relógio andando 1 (mutante), hits 80."""
        c = next(c for c in CASOS if c["tipo"] == "dirigido_relogio")
        self.assertEqual(c["esperado"], {"needed": 192, "hits": 96, "cpu": 96, "copies_bg": 4, "dropped": 8, "copies_deferred": 2})
        self.assertEqual(roda(c, RelogioUm)["hits"], 80)

    def test_o_controle_nao_muda_nenhum_caso(self):
        for c in CASOS:
            self.assertEqual(roda(c, F.SimPasso), c["esperado"], c["caso"])

    def test_cada_mutante_muda_alguma_contagem_gravada(self):
        """O arquivo gravado mata cada mutante das regras do passo em pelo menos um caso. Fora: a versão da chave subindo n_e vezes (equivalente
        nas contagens, coberta por teste de caixa-branca na U2), a admissão por pedido e os pins só da posição 0 (precisam de cópia do layer_m)."""
        for mut in MUTANTES:
            mortos = [c["caso"] for c in CASOS if roda(c, mut) != c["esperado"]]
            self.assertTrue(mortos, f"{mut.__name__} sobreviveu a todos os casos")

    def test_o_relogio_so_aparece_nos_casos_longos_ou_dirigidos(self):
        """Documenta a lacuna da revisão 58: nos 96 casos curtos o relógio errado não muda nada; só os longos e o dirigido o pegam."""
        curtos = [c for c in CASOS if c["tipo"] == "curto"]
        self.assertFalse([c["caso"] for c in curtos if roda(c, RelogioUm) != c["esperado"]])
        longos = [c for c in CASOS if c["tipo"] in ("longo", "dirigido_relogio")]
        self.assertTrue([c["caso"] for c in longos if roda(c, RelogioUm) != c["esperado"]])
        self.assertTrue([c["caso"] for c in longos if roda(c, RelogioMax) != c["esperado"]])

    def test_mutante_tamanho_do_passo_ignorado(self):
        """Rodar com passos errados (tudo M=2) muda alguma contagem: o tamanho do passo conta."""
        mudou = sum(1 for c in CASOS if roda(c, passos=F.passos_fixos(c["posicoes"], 2)) != c["esperado"])
        self.assertGreater(mudou, 0)


if __name__ == "__main__":
    unittest.main()
