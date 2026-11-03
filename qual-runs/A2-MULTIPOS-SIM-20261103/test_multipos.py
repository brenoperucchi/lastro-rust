"""Regressão do multipos.py (A2-MULTIPOS-SIM). Rodar: tools/.venv-ref/bin/python -I -B qual-runs/A2-MULTIPOS-SIM-20261103/test_multipos.py"""
import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multipos as MP  # noqa: E402
P = MP.moe_sim.PARAMS_RUNTIME["1b"]


def traco(seed, L=3, T=64, E=16, k=2):
    """Traço sintético com k experts DISTINTOS por posição, como o top-k real."""
    rng = np.random.default_rng(seed)
    return np.array([[rng.choice(E, size=k, replace=False) for _ in range(T)] for _ in range(L)])


def passo(L, E, k, frac, lim, blk):
    s = MP.Passo("1b", frac, L, E, k, lim)
    return s, s.passo(np.array(blk), True)


class T(unittest.TestCase):
    def test_sem_slots_tudo_vai_a_cpu_quieta(self):
        # 1 camada, nenhum slot: tempo = g + 2 faltas na CPU sem tráfego (M=1)
        _, t = passo(1, 4, 2, 0.0, "otimista", [[[0, 1]]])
        self.assertAlmostEqual(t, P["g"] + 2 * P["c_cpu_quiet"], places=9)

    def test_m2_otimista_le_o_expert_uma_vez(self):
        # posições {0,1} e {0,2}: união 3, n = (2,1,1)
        _, t = passo(1, 4, 2, 0.0, "otimista", [[[0, 1], [0, 2]]])
        self.assertAlmostEqual(t, P["g"] + 3 * P["c_cpu_quiet"], places=9)

    def test_m2_pessimista_cobra_cada_posicao(self):
        _, t = passo(1, 4, 2, 0.0, "pessimista", [[[0, 1], [0, 2]]])
        self.assertAlmostEqual(t, 2 * P["g"] + 4 * P["c_cpu_quiet"], places=9)

    def test_m1_os_dois_limites_coincidem(self):
        a = traco(1, T=40)
        tr, te = [a], [a]
        r = [MP.roda("1b", 0.25, 1, lim, tr, te) for lim in MP.LIMITES]
        self.assertEqual(r[0], r[1])

    def test_pessimista_nunca_menor_que_otimista(self):
        a = traco(2)
        for M in (2, 4, 8):
            o = MP.roda("1b", 0.25, M, "otimista", [a], [a])[0]
            p = MP.roda("1b", 0.25, M, "pessimista", [a], [a])[0]
            self.assertGreaterEqual(p, o)

    def test_determinismo(self):
        a = traco(3)
        self.assertEqual(MP.roda("1b", 0.25, 4, "otimista", [a], [a]), MP.roda("1b", 0.25, 4, "otimista", [a], [a]))

    def test_acertos_nao_passam_do_pedido_e_residentes_cabem_nos_slots(self):
        a = traco(4)
        s = MP.Passo("1b", 0.25, 3, 16, 2, "otimista")
        for b in range(64):
            s.passo(a[:, b:b + 1, :], True)
            self.assertLessEqual(len(s.resident), s.cap)
        self.assertLessEqual(s.stats["hits"], s.stats["needed"])

    def test_decisao(self):
        def res(gp, go):
            r = {}
            for m in ("1b", "3b"):
                for M in MP.MS:
                    r[(m, 0.10, "pessimista", M)] = gp if M <= MP.M_MAX else 0.9
                    r[(m, 0.10, "otimista", M)] = go if M <= MP.M_MAX else 0.9
            return r
        self.assertEqual(MP.decide(res(0.20, 0.30)), "VALE IMPLEMENTAR NO MOTOR")
        self.assertTrue(MP.decide(res(0.05, 0.30)).startswith("VALE MEDIR"))
        self.assertEqual(MP.decide(res(0.05, 0.10)), "NÃO VALE AGORA")
        # o ganho de M > 8 não conta
        r = res(0.0, 0.0)
        self.assertEqual(MP.decide(r), "NÃO VALE AGORA")
        # o modelo mais fraco decide
        r = res(0.20, 0.30)
        for M in MP.MS:
            r[("3b", 0.10, "pessimista", M)] = 0.05 if M <= MP.M_MAX else 0.9
        self.assertTrue(MP.decide(r).startswith("VALE MEDIR"))
        # exatamente 15% vale
        self.assertEqual(MP.decide(res(0.15, 0.15)), "VALE IMPLEMENTAR NO MOTOR")


if __name__ == "__main__":
    unittest.main()
