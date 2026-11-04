"""Regressão do multipos.py (A2-MULTIPOS-SIM). Rodar: tools/.venv-ref/bin/python -I -B qual-runs/A2-MULTIPOS-SIM-20261103/test_multipos.py"""
import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multipos as MP  # noqa: E402
S = MP.moe_sim
P = S.PARAMS_RUNTIME["1b"]


def traco(seed, L=3, T=64, E=16, k=2):
    """Traço sintético com k experts DISTINTOS por posição, como o top-k real."""
    rng = np.random.default_rng(seed)
    return np.array([[rng.choice(E, size=k, replace=False) for _ in range(T)] for _ in range(L)])


def ref_token(mod, frac, treino, teste, bw=MP.BW_GBS):
    """moe_sim de referência: protocolo token, lfu_cpu, mesma banda."""
    p = dict(S.PARAMS_RUNTIME[mod])
    p["pcie_gbs_cpu"] = bw
    L, _, k = treino[0].shape
    E = int(max(a.max() for a in treino + teste)) + 1
    s = S.Sim("lfu_cpu", int(frac * L * E), L, E, k, p, protocol="token")
    s.run(treino, False)
    s.run(teste, True)
    return s.stats


class T(unittest.TestCase):
    def test_m1_igual_ao_moe_sim_token_lfu_cpu(self):
        for seed, frac in ((1, 0.25), (2, 0.10), (3, 0.5)):
            tr, te = [traco(seed, T=80), traco(seed + 10, T=50)], [traco(seed + 20, T=70)]
            ref = ref_token("1b", frac, tr, te)
            for lim in MP.LIMITES:
                s = MP.SimM("1b", frac, 3, 16, 2, lim)
                s.run_blocks(tr, 1, False)
                s.run_blocks(te, 1, True)
                for kk in ("ms", "hits", "needed", "cpu", "copies_bg", "copies_deferred", "dropped", "tokens"):
                    self.assertAlmostEqual(s.stats[kk], ref[kk], places=9, msg=f"{seed} {frac} {lim} {kk}")

    def test_m1_igual_ao_moe_sim_nos_tracos_reais(self):
        tr, te = MP.carrega("1b")
        tr, te = [a[:, :150] for a in tr[:2]], [a[:, :150] for a in te[:2]]
        ref = ref_token("1b", 0.10, tr, te)
        r = MP.roda("1b", 0.10, 1, "otimista", tr, te)
        self.assertAlmostEqual(r["ms"], ref["ms"] / ref["tokens"], places=9)
        self.assertAlmostEqual(r["cop"], ref["copies_bg"] / ref["tokens"], places=9)

    def test_sem_slots_tudo_vai_a_cpu_quieta(self):
        s = MP.SimM("1b", 0.0, 1, 4, 2, "otimista")
        s.layer_m(0, np.array([0, 1]), np.array([1, 1]))
        self.assertAlmostEqual(s.now, P["g"] + 2 * P["c_cpu_quiet"], places=9)

    def test_m2_otimista_le_o_expert_uma_vez_e_pessimista_cobra_cada_posicao(self):
        for lim, esperado in (("otimista", P["g"] + 3 * P["c_cpu_quiet"]), ("pessimista", 2 * P["g"] + 4 * P["c_cpu_quiet"])):
            s = MP.SimM("1b", 0.0, 1, 4, 2, lim)
            s.M = 2
            s.layer_m(0, np.array([0, 1, 2]), np.array([2, 1, 1]))
            self.assertAlmostEqual(s.now, esperado, places=9, msg=lim)

    def test_acertos_em_lote_na_gpu(self):
        # expert 0 residente na tabela, pedido em 3 posições: otimista g + h; pessimista 3g + 3h
        for lim, esperado in (("otimista", P["g"] + P["h"]), ("pessimista", 3 * P["g"] + 3 * P["h"])):
            s = MP.SimM("1b", 1.0, 1, 4, 1, lim)
            s.M = 3
            s.resident[(0, 0)] = None
            s.layer_m(0, np.array([0]), np.array([3]))
            self.assertAlmostEqual(s.now, esperado, places=9, msg=lim)
            self.assertEqual(s.stats["hits"], 3)

    def test_falta_com_copia_em_voo_nao_admite_de_novo(self):
        s = MP.SimM("1b", 1.0, 1, 4, 1, "otimista")
        s.layer_m(0, np.array([0]), np.array([1]))        # falta: admitida
        n1 = len(s.inflight)
        s.layer_m(0, np.array([0]), np.array([1]))        # ainda em voo ou assentada: sem segunda cópia da mesma chave
        self.assertEqual(n1, 1)
        self.assertEqual(len(s.inflight) + len(s.resident), 1)

    def test_frequencia_conta_todos_os_pedidos_antes_de_admitir(self):
        # 1 slot ocupado por (0,0) com frequência 1; bloco M=4 pede (0,1) 4 vezes: o candidato vence a vítima na própria camada
        s = MP.SimM("1b", 0.25, 1, 4, 1, "otimista")
        s.M = 1
        s.layer_m(0, np.array([0]), np.array([1]))
        s.run_blocks([], 1, False)
        s.M = 4
        for _ in range(3):
            s.layer_m(0, np.array([1]), np.array([4]))
        self.assertGreaterEqual(s.stats["copies_bg"], 2)

    def test_frequencia_acumula_n_pedidos_e_o_relogio_anda_M_por_camada(self):
        import math
        s = MP.SimM("1b", 1.0, 1, 4, 1, "otimista")
        s.M = 3
        s.layer_m(0, np.array([2]), np.array([3]))
        self.assertAlmostEqual(s.freq[(0, 2)][0], math.log2(3), places=9)     # três pedidos = valor 3 (um toque por pedido)
        self.assertEqual(s.step, 3)                                           # o relógio do decaimento anda M posições por camada

    def test_tokens_so_do_teste_e_sem_bloco_incompleto(self):
        tr, te = [traco(1, T=33)], [traco(2, T=10), traco(3, T=7)]
        r = MP.roda("1b", 0.25, 4, "otimista", tr, te)
        self.assertEqual(r["tokens"], 8 + 4)               # 10//4*4 + 7//4*4; o treino e os restos não contam

    def test_treino_aquece_e_so_o_teste_e_medido(self):
        a, b = traco(1, T=40), traco(2, T=40)
        r_ab = MP.roda("1b", 0.25, 2, "otimista", [a], [b])
        r_ba = MP.roda("1b", 0.25, 2, "otimista", [b], [a])
        self.assertNotEqual(r_ab["ms"], r_ba["ms"])        # trocar treino e teste muda a medida

    def test_pessimista_nunca_menor_que_otimista_e_determinismo(self):
        a = traco(5, T=64)
        for M in (2, 4, 8):
            o = MP.roda("1b", 0.25, M, "otimista", [a], [a])["ms"]
            p = MP.roda("1b", 0.25, M, "pessimista", [a], [a])["ms"]
            self.assertGreaterEqual(p, o)
        self.assertEqual(MP.roda("1b", 0.25, 4, "otimista", [a], [a]), MP.roda("1b", 0.25, 4, "otimista", [a], [a]))

    def test_acertos_contam_n_pedidos(self):
        s = MP.SimM("1b", 1.0, 1, 4, 1, "otimista")
        s.M = 3
        s.resident[(0, 0)] = None
        s.layer_m(0, np.array([0]), np.array([3]))
        self.assertEqual((s.stats["needed"], s.stats["hits"]), (3, 3))

    def test_ordem_de_primeira_aparicao(self):
        es, ns = MP.primeira_aparicao(np.array([[5, 3], [3, 7]]))
        self.assertEqual((es, ns), ([5, 3, 7], [1, 2, 1]))

    def test_vitima_nao_e_expert_da_propria_camada(self):
        # cap = 1; (0,0) residente e pedido (acerto) 1 vez; (0,1) falta pedida 5 vezes (frequência maior): sem despejar o fixo da camada, não há cópia
        s = MP.SimM("1b", 0.25, 1, 4, 1, "otimista")
        s.M = 6
        s.resident[(0, 0)] = None
        s.push((0, 0))
        s.layer_m(0, np.array([0, 1]), np.array([1, 5]))
        self.assertEqual(s.stats["copies_bg"], 0)
        self.assertEqual(s.stats["dropped"], 1)

    def test_calibracao_limites(self):
        base = {"ms": MP.MEDIDO_L1["1b"], "cop": MP.MEDIDO_COP["1b"], "acerto": MP.MEDIDO_ACERTO["1b"]}
        self.assertTrue(MP.avalia_calibracao("1b", base)[0])
        self.assertTrue(MP.avalia_calibracao("1b", dict(base, cop=base["cop"] * 1.149))[0])
        self.assertFalse(MP.avalia_calibracao("1b", dict(base, cop=base["cop"] * 1.16))[0])      # só as cópias fora
        self.assertFalse(MP.avalia_calibracao("1b", dict(base, ms=base["ms"] * 1.11))[0])        # só o ms fora
        self.assertTrue(MP.avalia_calibracao("1b", dict(base, acerto=base["acerto"] + 0.019))[0])
        self.assertFalse(MP.avalia_calibracao("1b", dict(base, acerto=base["acerto"] + 0.021))[0])  # só o acerto fora
        b3 = {"ms": MP.MEDIDO_L1["3b"], "cop": MP.MEDIDO_COP["3b"], "acerto": 0.0}                # o 3B não confere acerto
        self.assertTrue(MP.avalia_calibracao("3b", b3)[0])

    def test_principal_nao_varre_se_a_calibracao_reprova(self):
        chamadas = []
        orig = (MP.calibra, MP.varredura)
        try:
            MP.calibra, MP.varredura = (lambda: False), (lambda: chamadas.append("varre"))
            self.assertEqual(MP.principal(["x"]), 1)
            self.assertEqual(chamadas, [])
            MP.calibra = lambda: True
            self.assertEqual(MP.principal(["x", "--calibra"]), 0)
            self.assertEqual(chamadas, [])
            self.assertEqual(MP.principal(["x"]), 0)
            self.assertEqual(chamadas, ["varre"])
        finally:
            MP.calibra, MP.varredura = orig

    def test_metricas_a_mao(self):
        aeq, folga = MP.metricas(10.0, 6.0, 4)
        self.assertAlmostEqual(aeq, 6.0 / 10.0 / 0.85, places=12)
        self.assertAlmostEqual(folga, 4 * (0.85 * 10.0 - 6.0), places=12)
        self.assertLess(MP.metricas(10.0, 9.0, 2)[1], 0)                                          # ganho menor que 15%: folga negativa

    def test_achata_usa_distancia_absoluta(self):
        self.assertEqual(MP.achata([0.0, 0.30, 0.40, 0.35, 0.20, 0.10]), 32)                      # não monotônica: M=2 está a 20 pontos de M=32
        self.assertEqual(MP.achata([0.0, 0.10, 0.20, 0.29, 0.305, 0.31]), 16)
        self.assertEqual(MP.achata([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]), 32)

    def test_decisao(self):
        def res(gp, go, so_em=None):
            r = {}
            for m in ("1b", "3b"):
                for M in MP.MS:
                    gv = (lambda g: g if (so_em is None and M <= MP.M_MAX) or M == so_em else 0.0)
                    r[(m, 0.10, "pessimista", M)] = gv(gp)
                    r[(m, 0.10, "otimista", M)] = gv(go)
            return r
        self.assertTrue(MP.decide(res(0.20, 0.30)).startswith("VALE PROTOTIPAR"))
        self.assertTrue(MP.decide(res(0.05, 0.30)).startswith("VALE MEDIR"))
        self.assertEqual(MP.decide(res(0.05, 0.10)), "NÃO VALE AGORA")
        self.assertTrue(MP.decide(res(0.15, 0.15)).startswith("VALE PROTOTIPAR"))           # 15% exato vale
        self.assertTrue(MP.decide(res(0.20, 0.30, so_em=8)).startswith("VALE PROTOTIPAR"))   # M = 8 conta
        self.assertEqual(MP.decide(res(0.20, 0.30, so_em=16)), "NÃO VALE AGORA")             # M = 16 não conta
        r = res(0.20, 0.30)
        for M in MP.MS:
            r[("3b", 0.10, "pessimista", M)] = 0.05 if M <= MP.M_MAX else 0.9
        self.assertTrue(MP.decide(r).startswith("VALE MEDIR"))                               # o modelo mais fraco decide


if __name__ == "__main__":
    unittest.main()
