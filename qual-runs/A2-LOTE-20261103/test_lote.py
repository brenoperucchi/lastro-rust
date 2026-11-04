"""Testes do lote.py. Rodar: tools/.venv-ref/bin/python -I -B qual-runs/A2-LOTE-20261103/test_lote.py"""
import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lote as LT  # noqa: E402
MP = LT.MP


def gpu_txt(f1b, f3b, err=1e-5, t1=(4.96, 5.19)):
    out = []
    for mod, t, f in (("1b", t1[0], f1b), ("3b", t1[1], f3b)):
        for n in LT.NS:
            out.append(f"GPU {mod} n={n} lote_us={t * f(n):.2f} serial_us={t * n:.2f} err_lote_serial={err:.2e} err_ref={err:.2e}")
    return "\n".join(out)


def cpu_txt(f, err=1e-7, t1=(40.1, 61.0)):
    return "\n".join(f"CPU {mod} n={n} us_por_expert={t * f(n):.2f} max_err_rel={err:.2e}" for mod, t in zip(("1b", "3b"), t1) for n in LT.NS)


class Portoes(unittest.TestCase):
    def tabs(self, **kw):
        return LT.parse(gpu_txt(lambda n: 1, lambda n: 1, **kw.get("g", {})), "gpu"), LT.parse(cpu_txt(lambda n: 1, **kw.get("c", {})), "cpu")

    def test_passa(self):
        self.assertEqual(LT.portoes(*self.tabs()), [])

    def test_ausente(self):
        g, c = self.tabs()
        del g[("3b", 8)]
        self.assertTrue(any("ausente" in m for m in LT.portoes(g, c)))

    def test_correcao_gpu_e_cpu(self):
        for kw in ({"g": {"err": 2e-3}}, {"c": {"err": 2e-5}}):
            self.assertTrue(any("correção" in m for m in LT.portoes(*self.tabs(**kw))), kw)
        self.assertEqual(LT.portoes(*self.tabs(g={"err": 1e-3})), [])  # no limite passa

    def test_nan_reprova(self):
        g, c = self.tabs()
        g[("1b", 2)]["err"] = float("nan")
        self.assertTrue(LT.portoes(g, c))

    def test_representatividade(self):
        self.assertTrue(any("GPU 1b n=1" in m for m in LT.portoes(*self.tabs(g={"t1": (4.96 * 1.5, 5.19)}))))
        self.assertEqual(LT.portoes(*self.tabs(g={"t1": (4.96 * 1.39, 5.19)})), [])
        self.assertTrue(any("CPU 3b n=1" in m for m in LT.portoes(*self.tabs(c={"t1": (40.1, 61.0 * 0.6)}))))

    def test_linha_repetida(self):
        with self.assertRaises(AssertionError):
            LT.parse("CPU 1b n=1 us_por_expert=1.0 max_err_rel=0\nCPU 1b n=1 us_por_expert=0.5 max_err_rel=0", "cpu")


class TabelaW(unittest.TestCase):
    def test_w_e_interpolacao(self):
        g = LT.parse(gpu_txt(lambda n: 1 + 0.1 * (n - 1), lambda n: 1), "gpu")
        w, pts = LT.tabela_w(g, "1b")
        self.assertAlmostEqual(w(1), 1.0, places=2)
        self.assertAlmostEqual(w(8), 1.7, places=2)
        self.assertAlmostEqual(w(3), 1.2, places=2)      # entre n=2 (1,1) e n=4 (1,3)
        self.assertAlmostEqual(w(32), 4.1, places=2)
        with self.assertRaises(AssertionError):
            w(33)


class Decisao(unittest.TestCase):
    def ganhos(self, g1, gM):
        return {(m, modo, M): (v if M > 1 else 0.0) for m in ("1b", "3b") for modo, v in (("g1", g1), ("gM", gM)) for M in MP.MS}

    def test_tres_saidas(self):
        self.assertIn("VALE PROTOTIPAR", LT.decide(self.ganhos(0.5, 0.2), True))
        self.assertIn("DEPENDE", LT.decide(self.ganhos(0.5, 0.1), True))
        self.assertIn("NÃO VALE", LT.decide(self.ganhos(0.1, 0.0), True))
        self.assertIn("SEM DECISÃO", LT.decide({}, False))

    def test_um_modelo_fraco_decide(self):
        g = self.ganhos(0.5, 0.5)
        for M in MP.MS:
            g[("3b", "gM", M)] = 0.1 if M > 1 else 0.0
        self.assertIn("DEPENDE", LT.decide(g, True))

    def test_so_M_ate_8_conta(self):
        g = self.ganhos(0.5, 0.0)
        for m in ("1b", "3b"):
            g[(m, "gM", 16)] = 0.9
        self.assertIn("DEPENDE", LT.decide(g, True))


class Simulador(unittest.TestCase):
    def traco(self, seed, L=3, T=64, E=16, k=2):
        rng = np.random.default_rng(seed)
        return np.array([[rng.choice(E, size=k, replace=False) for _ in range(T)] for _ in range(L)])

    def roda(self, M, wg, wc, modo):
        tr, te = [self.traco(1), self.traco(2)], [self.traco(3)]
        return LT.roda("1b", 0.25, M, wg, wc, modo, tr, te)

    def test_M1_igual_ao_a2(self):
        tr, te = [self.traco(1)], [self.traco(3)]
        a = MP.roda("1b", 0.25, 1, "otimista", tr, te)["ms"]
        b = LT.roda("1b", 0.25, 1, lambda n: 1.0, lambda n: 1.0, "g1", tr, te)
        self.assertEqual(a, b)

    def test_w_identidade_reproduz_otimista_e_pessimista(self):
        tr, te = [self.traco(1)], [self.traco(3)]
        o = MP.roda("1b", 0.25, 4, "otimista", tr, te)["ms"]
        p = MP.roda("1b", 0.25, 4, "pessimista", tr, te)["ms"]
        self.assertEqual(o, LT.roda("1b", 0.25, 4, lambda n: 1.0, lambda n: 1.0, "g1", tr, te))
        self.assertEqual(p, LT.roda("1b", 0.25, 4, lambda n: n, lambda n: n, "gM", tr, te))

    def test_custo_medido_maior_custa_mais(self):
        a = self.roda(4, lambda n: 1.0, lambda n: 1.0, "g1")
        b = self.roda(4, lambda n: 1 + 0.2 * (n - 1), lambda n: 1 + 0.2 * (n - 1), "g1")
        c = self.roda(4, lambda n: 1.0, lambda n: 1.0, "gM")
        self.assertLess(a, b)
        self.assertLess(a, c)

    def test_gpu_e_cpu_usam_tabelas_diferentes(self):
        base = self.roda(4, lambda n: 1.0, lambda n: 1.0, "g1")
        so_gpu = self.roda(4, lambda n: 500.0, lambda n: 1.0, "g1")
        so_cpu = self.roda(4, lambda n: 1.0, lambda n: 500.0, "g1")
        self.assertGreater(so_gpu, base)
        self.assertNotEqual(so_cpu, so_gpu)


if __name__ == "__main__":
    unittest.main()
