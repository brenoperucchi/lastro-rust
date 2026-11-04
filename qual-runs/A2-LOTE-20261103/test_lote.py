"""Testes do lote.py. Rodar: tools/.venv-ref/bin/python -I -B qual-runs/A2-LOTE-20261103/test_lote.py"""
import contextlib, io, os, sys, tempfile, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lote as LT  # noqa: E402
MP = LT.MP


def gpu_txt(f1b, f3b, err=1e-5, t1=(4.96, 5.19), err_ref=None, cab=True):
    out = [f"# dispositivo: NVIDIA GeForce RTX 4090 uuid={LT.UUID_4090}", "# modo=completo reps=15 replays=10"] if cab else []
    for mod, t, f in (("1b", t1[0], f1b), ("3b", t1[1], f3b)):
        for n in LT.NS:
            out.append(f"GPU {mod} n={n} lote_us={t * f(n):.2f} serial_us={t * n:.2f} err_lote_serial={err:.2e} err_ref={(err if err_ref is None else err_ref):.2e}")
    return "\n".join(out)


def cpu_txt(f, err=1e-7, t1=(40.1, 61.0), cab=True):
    return "\n".join(["# modo=completo reps=15 camadas_por_rep=106 experts_por_camada=6"] * cab + [f"CPU {mod} n={n} us_por_expert={t * f(n):.2f} max_err_rel={err:.2e}" for mod, t in zip(("1b", "3b"), t1) for n in LT.NS])


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


class Reforcos(unittest.TestCase):
    def tabs(self, **kw):
        return LT.parse(gpu_txt(lambda n: 1, lambda n: 1, **kw.get("g", {})), "gpu"), LT.parse(cpu_txt(lambda n: 1, **kw.get("c", {})), "cpu")

    def test_erros_gpu_sao_independentes(self):
        # referência alta com lote/serial baixo e vice-versa: o max dos dois decide
        self.assertTrue(any("correção" in m for m in LT.portoes(*self.tabs(g={"err": 0.0, "err_ref": 2e-3}))))
        self.assertTrue(any("correção" in m for m in LT.portoes(*self.tabs(g={"err": 2e-3, "err_ref": 0.0}))))

    def test_cpu_representatividade_usa_c_cpu_quiet(self):
        self.assertEqual(LT.portoes(*self.tabs(c={"t1": (40.1 * 1.2, 61.0)})), [])
        self.assertTrue(any("CPU 1b n=1" in m for m in LT.portoes(*self.tabs(c={"t1": (40.1 * 1.35, 61.0)}))))

    def test_tempo_zero_ou_nao_finito_reprova(self):
        for campo, v in (("lote", 0.0), ("lote", float("nan")), ("serial", 0.0), ("lote", float("inf"))):
            g, c = self.tabs()
            g[("1b", 8)][campo] = v
            self.assertTrue(any("tempo inválido" in m for m in LT.portoes(g, c)), (campo, v))
        g, c = self.tabs()
        c[("3b", 4)]["lote"] = 0.0
        self.assertTrue(any("tempo inválido" in m for m in LT.portoes(g, c)))

    def test_erro_negativo_ou_nan_reprova(self):
        for v in (-1e-9, float("nan"), float("inf")):
            g, c = self.tabs()
            g[("3b", 2)]["err"] = v
            self.assertTrue(LT.portoes(g, c), v)

    def test_texto_com_nan_nao_passa(self):
        t = gpu_txt(lambda n: 1, lambda n: 1).replace("err_ref=1.00e-05", "err_ref=nan", 1)
        g = LT.parse(t, "gpu")
        c = LT.parse(cpu_txt(lambda n: 1), "cpu")
        self.assertTrue(LT.portoes(g, c))

    def test_cabecalho(self):
        self.assertEqual(LT.cabecalho(gpu_txt(lambda n: 1, lambda n: 1), "GPU"), [])
        self.assertEqual(LT.cabecalho(cpu_txt(lambda n: 1), "CPU"), [])
        self.assertTrue(LT.cabecalho(gpu_txt(lambda n: 1, lambda n: 1, cab=False), "GPU"))
        self.assertTrue(LT.cabecalho(cpu_txt(lambda n: 1, cab=False), "CPU"))
        self.assertTrue(LT.cabecalho(cpu_txt(lambda n: 1).replace("completo reps=15", "rapido reps=3"), "CPU"))
        outra = gpu_txt(lambda n: 1, lambda n: 1).replace(LT.UUID_4090, "GPU-00000000-0000-0000-0000-000000000000")
        self.assertTrue(any("uuid" in m for m in LT.cabecalho(outra, "GPU")))

    def test_decide_so_M8_e_igualdade(self):
        g = {(m, modo, M): 0.0 for m in ("1b", "3b") for modo in ("g1", "gM") for M in MP.MS}
        for m in ("1b", "3b"):
            g[(m, "gM", 8)] = MP.GANHO_MIN                  # exatamente 15% em M=8 vale
        self.assertIn("VALE PROTOTIPAR", LT.decide(g, True))
        for m in ("1b", "3b"):
            g[(m, "gM", 8)] = MP.GANHO_MIN - 1e-9
        self.assertIn("NÃO VALE", LT.decide(g, True))


class CamposGpu(unittest.TestCase):
    def test_cada_campo_invalido_reprova_no_parser(self):
        base = gpu_txt(lambda n: 1, lambda n: 1)
        c = LT.parse(cpu_txt(lambda n: 1), "cpu")
        for campo in ("err_lote_serial", "err_ref"):
            for v in ("-0.1", "-inf", "inf", "nan"):
                t = base.replace(f"{campo}=1.00e-05", f"{campo}={v}", 1)
                self.assertNotEqual(t, base)
                self.assertTrue(LT.portoes(LT.parse(t, "gpu"), c), (campo, v))
        self.assertEqual(LT.portoes(LT.parse(base, "gpu"), c), [])


class Principal(unittest.TestCase):
    def traco(self, seed, L=3, T=64, E=16, k=2):
        rng = np.random.default_rng(seed)
        return np.array([[rng.choice(E, size=k, replace=False) for _ in range(T)] for _ in range(L)])

    def carrega(self, mod):
        return [self.traco(1), self.traco(2)], [self.traco(3)]

    def roda_principal(self, gtxt, ctxt):
        with tempfile.TemporaryDirectory() as d:
            g, c = os.path.join(d, "g.txt"), os.path.join(d, "c.txt")
            with open(g, "w") as f, open(c, "w") as h:
                f.write(gtxt)
                h.write(ctxt)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = LT.principal(g, c, self.carrega)
            return rc, buf.getvalue()

    def test_portao_reprovado_nao_simula(self):
        chamadas = []
        orig = LT.roda
        LT.roda = lambda *a, **k: chamadas.append(1)
        try:
            rc, out = self.roda_principal(gpu_txt(lambda n: 1, lambda n: 1, cab=False), cpu_txt(lambda n: 1))
        finally:
            LT.roda = orig
        self.assertEqual(rc, 1)
        self.assertEqual(chamadas, [])
        self.assertIn("SEM DECISÃO", out)

    def test_cpu_sem_cabecalho_ou_rapido_nao_decide(self):
        rc, out = self.roda_principal(gpu_txt(lambda n: 1, lambda n: 1), cpu_txt(lambda n: 1, cab=False))
        self.assertEqual(rc, 1)
        rc, _ = self.roda_principal(gpu_txt(lambda n: 1, lambda n: 1), cpu_txt(lambda n: 1).replace("completo reps=15", "rapido reps=3"))
        self.assertEqual(rc, 1)

    def test_decide_so_ve_10_por_cento_e_roda_usa_coluna_lote(self):
        vistos, wgs = [], []
        od, orr = LT.decide, LT.roda
        LT.decide = lambda g, ok: vistos.append(set(g)) or "x"
        def roda(mod, frac, M, wg, wc, modo, tr, te):
            wgs.append((mod, wg(8), wc(8)))
            return orr(mod, frac, M, wg, wc, modo, tr, te)
        LT.roda = roda
        try:
            gt = gpu_txt(lambda n: 1 + 0.5 * (n - 1), lambda n: 1 + 0.5 * (n - 1))
            self.roda_principal(gt, cpu_txt(lambda n: 1 + 0.02 * (n - 1)))
        finally:
            LT.decide, LT.roda = od, orr
        self.assertEqual(len(vistos[0]), 2 * 2 * len(MP.MS))        # 2 modelos x 2 modos x 6 M, só 10%
        g = LT.parse(gt, "gpu")
        esperado = LT.tabela_w(g, "1b")[0](8)
        self.assertTrue(any(m == "1b" and abs(a - esperado) < 1e-9 for m, a, _ in wgs))
        self.assertFalse(any(abs(a - 8.0) < 1e-6 for _, a, _ in wgs))   # a coluna serial daria w=8

    def test_ganhos_usam_as_tabelas_certas_a_10_e_25(self):
        g = LT.parse(gpu_txt(lambda n: 1 + 0.5 * (n - 1), lambda n: 1 + 0.5 * (n - 1)), "gpu")
        c = LT.parse(cpu_txt(lambda n: 1 + 0.02 * (n - 1)), "cpu")
        gan, tem = LT.calcula(g, c, self.carrega, (0.10, 0.25))
        for mod in ("1b", "3b"):
            wg, _ = LT.tabela_w(g, mod)
            wc, _ = LT.tabela_w(c, mod)
            tr, te = self.carrega(mod)
            for frac in (0.10, 0.25):
                t1 = LT.roda(mod, frac, 1, wg, wc, "g1", tr, te)
                for M in (2, 8):
                    t = LT.roda(mod, frac, M, wg, wc, "gM", tr, te)
                    self.assertAlmostEqual(gan[(mod, frac, "gM", M)], 1 - t / t1)
                    # tabelas trocadas dão outro número
                    t_troca = LT.roda(mod, frac, M, wc, wg, "gM", tr, te)
                    self.assertNotAlmostEqual(t, t_troca, places=6)
        self.assertEqual({k[1] for k in gan}, {0.10, 0.25})

    def test_fluxo_completo_imprime_decisao(self):
        rc, out = self.roda_principal(gpu_txt(lambda n: 1 + 0.05 * (n - 1), lambda n: 1 + 0.05 * (n - 1)), cpu_txt(lambda n: 1 + 0.1 * (n - 1)))
        self.assertEqual(rc, 0)
        self.assertIn("DECISÃO:", out)
        self.assertIn("[descritivo, não decide]", out)


if __name__ == "__main__":
    unittest.main()
