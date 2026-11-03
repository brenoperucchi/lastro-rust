"""Testes sintéticos do classificador decide.py (criterio.txt v6, seção 5). Cada caso tem uma saída única e coerente com o sinal e a incerteza."""
import itertools, os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decide as D  # noqa: E402

# Referência sintética: d0 = (L1-L4) - (L2-L3) = 5,4 - 4,9 = 0,5 ms
def rep(l1=7.6, l2=7.6, l3=2.7, l4=2.2):
    return {"L1": l1, "L2": l2, "L3": l3, "L4": l4}


def cfg(n=5, **kw):
    return [rep(**kw) for _ in range(n)]


def com_d(d, n=5):
    """Configuração com o d desejado, mexendo só em L1."""
    return [rep(l1=7.1 + d) for _ in range(n)]   # L1-L4 = 4,9 + d ; L2-L3 = 4,9


CONT = {"hits": 165887, "cpu": 496705, "dropped": 492509, "copies": 4040, "deferred": 2205}
TOK, CC = 3451, 0.223


def cont(**kw):
    c = dict(CONT)
    c.update(kw)
    return c


class Passo0(unittest.TestCase):
    def test_reproduz_estavel(self):
        self.assertEqual(D.passo0_etapa1(com_d(0.5), com_d(0.5)), "OK")

    def test_nao_reproduzido_quase_zero(self):
        self.assertEqual(D.passo0_etapa1(com_d(0.05), com_d(0.05)), D.NAO_REPRODUZIDO)

    def test_nao_reproduzido_negativo(self):
        self.assertEqual(D.passo0_etapa1(com_d(-0.5), com_d(0.5)), D.NAO_REPRODUZIDO)

    def test_nao_reproduzido_disperso(self):
        c = [rep(l1=7.1 + x) for x in (0.9, 0.9, 0.9, -0.01, 0.9)]   # mediana alta mas um d_i <= 0
        self.assertEqual(D.passo0_etapa1(c, com_d(0.5)), D.NAO_REPRODUZIDO)

    def test_fronteira_03(self):
        self.assertEqual(D.passo0_etapa1(com_d(0.3), com_d(0.3)), "OK")            # mediana >= 0,3 inclusive
        self.assertEqual(D.passo0_etapa1(com_d(0.2999), com_d(0.3)), D.NAO_REPRODUZIDO)

    def test_build_mudou(self):
        self.assertEqual(D.passo0_etapa1(com_d(0.5), com_d(0.05)), D.SEM_CONCLUSAO_BUILD)    # N perdeu o excesso
        self.assertEqual(D.passo0_etapa1(com_d(0.5), com_d(0.7)), D.SEM_CONCLUSAO_BUILD)     # |0,7-0,5| > 0,1
        self.assertEqual(D.passo0_etapa1(com_d(0.5), com_d(0.6)), "OK")                      # fronteira 0,1 inclusive


class Fator(unittest.TestCase):
    N0 = cfg()   # d0 = 0,5

    def run_fator(self, **kw):
        return D.fator(self.N0, cfg(**kw), 0.5)

    def test_l1_cai_sustentada(self):
        s, det = self.run_fator(l1=7.2)                      # m = 0,4, R = Mfr = 0,8
        self.assertEqual(s, D.SUSTENTADA)
        self.assertAlmostEqual(det["R"], 0.8)

    def test_l2_piora_nao_e_motor(self):
        s, det = self.run_fator(l2=8.0)                      # r = 0,4 vem do comparador, m = 0
        self.assertEqual(s, D.SEM_CONCLUSAO_NAO_MOTOR)

    def test_l1_quase_igual_l2_piora(self):
        s, _ = self.run_fator(l1=7.599, l2=8.0)              # sinal de dL1 negativo, mas fração do motor ~0
        self.assertEqual(s, D.SEM_CONCLUSAO_NAO_MOTOR)

    def test_l1_igual_l2_piora_sem_ramo_vazio(self):
        s, _ = self.run_fator(l1=7.6, l2=8.0)                # dL1 = 0, R = 0,8: tem de ter saída única
        self.assertIn(s, D.SAIDAS)

    def test_base_movida_l4(self):
        s, _ = self.run_fator(l4=2.6)                        # L4 +0,4 com L1 igual: d "cai" pela base
        self.assertEqual(s, D.SEM_CONCLUSAO_BASE)

    def test_base_movida_l3(self):
        s, _ = self.run_fator(l3=3.1)
        self.assertEqual(s, D.SEM_CONCLUSAO_BASE)

    def test_base_fronteira_005(self):
        s, _ = self.run_fator(l1=7.2, l4=2.25)               # |dL4| = 0,05 exatamente: permitido
        self.assertNotEqual(s, D.SEM_CONCLUSAO_BASE)

    def test_nao_sustentada(self):
        s, det = self.run_fator(l1=7.5)                      # m = 0,1, R = 0,2 < 0,3
        self.assertEqual(s, D.NAO_SUSTENTADA)

    def test_parcial(self):
        s, det = self.run_fator(l1=7.4)                      # m = 0,2: R = Mfr = 0,4
        self.assertEqual(s, D.PARCIAL)

    def test_r_alto_com_um_par_nao_positivo_e_parcial(self):
        n1 = [rep(l1=7.2) for _ in range(4)] + [rep(l1=7.6)]  # um m_i = 0
        s, _ = D.fator(self.N0, n1, 0.5)
        self.assertEqual(s, D.PARCIAL)

    def test_um_m_nao_positivo_com_r_positivo_e_parcial(self):
        # 4 pares com m = 0,4; o quinto tem m = 0 mas o comparador piora (c = -0,4), então r_i = 0,4 > 0 em todos e R = Mfr = 0,8:
        # sem a exigência dos 5 m_i > 0 isto seria SUSTENTADA
        n1 = [rep(l1=7.2) for _ in range(4)] + [rep(l1=7.6, l2=8.0)]
        s, det = D.fator(self.N0, n1, 0.5)
        self.assertTrue(all(x > 0 for x in det["r"]))
        self.assertEqual(s, D.PARCIAL)

    def test_a0_da_etapa2_sem_excesso_nao_confirma(self):
        # d(A0) = 0,05 e d(A1w) = 0,04: a melhora pareada é 0,01 ms e a referência já perdeu o excesso
        N0 = com_d(0.05)
        N1 = com_d(0.04)
        s, _ = D.fator(N0, N1, 0.5)
        self.assertEqual(s, D.NAO_REPRODUZIDO)

    def test_a0_da_etapa2_longe_da_etapa1(self):
        s, _ = D.fator(com_d(0.8), com_d(0.0), 0.5)          # reproduz, mas |0,8-0,5| > 0,1
        self.assertEqual(s, D.SEM_CONCLUSAO_BUILD)

    def test_fronteiras_r(self):
        # R = 0,3 exatamente (m = 0,15): >= 0,3 e Mfr >= 0,3 -> PARCIAL, não NAO_SUSTENTADA
        s, det = self.run_fator(l1=7.45)
        self.assertEqual(s, D.PARCIAL, det.get("R"))
        # R = 0,7 exatamente (m = 0,35) com 5 r_i > 0 e 5 m_i > 0 -> SUSTENTADA
        s, det = self.run_fator(l1=7.25)
        self.assertEqual(s, D.SUSTENTADA, det.get("R"))


class Particao(unittest.TestCase):
    def test_toda_combinacao_tem_uma_saida(self):
        # grade de R e Mfr (lados e fronteiras de 0,3 e 0,7) x sinais dos r_i e m_i: nenhuma combinação fica sem saída
        d0 = 0.5
        n0 = com_d(d0)
        vistos = set()
        for m_frac, c_frac, tem_nao_pos in itertools.product((-0.2, 0.0, 0.29, 0.3, 0.5, 0.69, 0.7, 0.9, 1.2), (-0.3, 0.0, 0.2, 0.5), (False, True)):
            m = m_frac * d0
            c = c_frac * d0
            vals = [(m, c)] * 5
            if tem_nao_pos:
                vals[-1] = (0.0, 0.0)
            n1 = [rep(l1=7.6 - mm, l2=7.6 - cc) for mm, cc in vals]   # base parada, d0 = 0,5 em N0
            n0b = cfg()
            s, _ = D.fator(n0b, n1, 0.5)
            self.assertIn(s, D.SAIDAS)
            vistos.add(s)
        self.assertTrue({D.SUSTENTADA, D.PARCIAL, D.NAO_SUSTENTADA, D.SEM_CONCLUSAO_NAO_MOTOR} <= vistos)


class Opostos(unittest.TestCase):
    def test_efeitos_opostos_inconclusivo(self):
        N0 = cfg()
        w = D.fator(N0, cfg(l1=7.2), 0.5)      # r = +0,4
        m = D.fator(N0, cfg(l1=8.0), 0.5)      # r = -0,4
        self.assertEqual(D.combina(w, m), D.INCONCLUSIVO_OPOSTOS)

    def test_efeito_pequeno_nao_e_oposto(self):
        N0 = cfg()
        w = D.fator(N0, cfg(l1=7.2), 0.5)
        m = D.fator(N0, cfg(l1=7.65), 0.5)     # r = -0,05: dentro de 0,1
        self.assertEqual(D.combina(w, m), {"w": D.SUSTENTADA, "m": D.NAO_SUSTENTADA})


class Trabalho(unittest.TestCase):
    def test_mais_12_copias_e_6_adiadas_passa(self):
        self.assertTrue(D.par_valido(CONT, cont(copies=4052, deferred=2211), TOK, CC))

    def test_mais_300_copias_reprova(self):
        self.assertFalse(D.par_valido(CONT, cont(copies=4340, deferred=2370), TOK, CC))

    def test_descartes_menos_2000_reprova(self):
        self.assertFalse(D.par_valido(CONT, cont(dropped=490509), TOK, CC))

    def test_acertos_fora_de_01_por_cento_reprova(self):
        self.assertFalse(D.par_valido(CONT, cont(hits=165887 + 200), TOK, CC))

    def test_cpu_fora_de_01_por_cento_reprova(self):
        self.assertFalse(D.par_valido(CONT, cont(cpu=496705 + 600), TOK, CC))

    def test_adiadas_so_reportadas(self):
        self.assertTrue(D.par_valido(CONT, cont(deferred=0), TOK, CC))        # não há inequação própria das adiadas

    def test_menos_de_5_pares_validos(self):
        N0 = cfg()
        e2 = {"N0": N0, "A1w": cfg(l1=7.2), "A1m": cfg(l1=7.2),
              "cont": {"N0": [CONT] * 5, "A1w": [CONT] * 4 + [cont(copies=4340)], "A1m": [CONT] * 5}}
        r = D.celula(com_d(0.5), com_d(0.5), e2, TOK, CC)
        self.assertEqual(r["w"], D.SEM_CONCLUSAO_TRABALHO)
        self.assertEqual(r["m"], D.SUSTENTADA)


class ValidadeDeFora(unittest.TestCase):
    def test_valid_externo_substitui_os_contadores(self):
        N0 = cfg()
        e2 = {"N0": N0, "A1w": cfg(l1=7.2), "A1m": cfg(l1=7.2), "valid": {"A1w": [True] * 4 + [False], "A1m": [True] * 5}}
        r = D.celula(com_d(0.5), com_d(0.5), e2, TOK, CC)
        self.assertEqual(r["w"], D.SEM_CONCLUSAO_TRABALHO)
        self.assertEqual(r["m"], D.SUSTENTADA)


class Pipeline(unittest.TestCase):
    def test_passo0_bloqueia_as_regras(self):
        e2 = {"N0": cfg(), "A1w": cfg(l1=7.2), "A1m": cfg(l1=7.2), "cont": {"N0": [CONT] * 5, "A1w": [CONT] * 5, "A1m": [CONT] * 5}}
        self.assertEqual(D.celula(com_d(0.05), com_d(0.05), e2, TOK, CC), D.NAO_REPRODUZIDO)


if __name__ == "__main__":
    unittest.main()
