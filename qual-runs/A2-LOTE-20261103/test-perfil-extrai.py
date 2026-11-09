#!/usr/bin/env python3
"""Testes do perfil-extrai.py com dados SINTÉTICOS. Sem GPU, sem rodar nada do motor.

    python3 -I qual-runs/A2-LOTE-20261103/test-perfil-extrai.py
"""
import importlib.util
import io
import json
import os
import tempfile
import time
import unittest
from contextlib import redirect_stdout

AQUI = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("perfil_extrai", os.path.join(AQUI, "perfil-extrai.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

EPOCH_INI, EPOCH_FIM = 1_700_000_000, 1_700_000_010
GERADOS = list(range(100, 164))
NPOS = {"curto": 89, "medio": 170, "longo": 364}
IDS = {t: list(range(n)) for t, n in NPOS.items()}
SERIES = ("L10", "L100", "LC")
TEXTOS = ("curto", "medio", "longo")
DADOS_MODELO = {"1b": {"L": 24, "E": 32, "N": 22}, "3b": {"L": 32, "E": 40, "N": 29}}
RATIO = {"L10": 0.42, "L100": 0.665, "LC": 0.42}      # razão ms/posição ÷ decode usada nos dados "bons"


def grava(p, conteudo):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(conteudo if isinstance(conteudo, str) else json.dumps(conteudo))


class Base(unittest.TestCase):
    modelo = "1b"

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.d = self._t.name
        self.saida = os.path.join(self.d, "saida")
        os.makedirs(self.saida)
        self.ids = os.path.join(self.d, "ids.json")
        grava(self.ids, {"modelos": {m: {t: {"ids": IDS[t]} for t in IDS} for m in ("1b", "3b")}})
        self.refs = os.path.join(self.d, "refs")
        for m in ("1b", "3b"):
            grava(os.path.join(self.refs, "ref5c-%s.json" % m), {"casos": [{"nome": t, "ids": IDS[t], "gerados": GERADOS, "texto_gerado": "x", "gaps": [5.0] * 64} for t in IDS]})
        self.textos = os.path.join(self.d, "textos.json")
        grava(self.textos, {"textos": [{"nome": t, "ids_com_template": {"1b": NPOS[t], "3b": NPOS[t]}} for t in IDS]})
        grava(os.path.join(self.saida, "integridade-antes.txt"), "binario abc\ngguf def\n")
        grava(os.path.join(self.saida, "integridade-depois.txt"), "binario abc\ngguf def\n")
        grava(os.path.join(self.saida, "orcamento.json"), {"modelos": [self.modelo], "limite_s": 1200, "decorrido_s": 600, "cortes": [], "nao_rodou": []})
        grava(os.path.join(self.saida, "ambiente.txt"), "modelo=%s\nmemoria_mts=2400\n" % self.modelo)
        for s in ("L100", "LC"):
            self.smoke(s, True)

    def tearDown(self):
        self._t.cleanup()

    def smoke(self, s, ok, motivo="memória usada da GPU 95% > 90%"):
        grava(os.path.join(self.saida, "smoke-%s" % s, "veredito.txt"), "veredito=OK\n" if ok else "veredito=FALHA\nmotivo=%s\n" % motivo)

    # ---- construtor de célula
    def monitor(self, d):
        linhas = ["%s.000, 5, 2000, 10000, 60, 100" % time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(t_)) for t_ in range(EPOCH_INI, EPOCH_FIM + 1)]
        grava(os.path.join(d, "monitor.csv"), "\n".join(linhas) + "\n")

    def ms_pos_ref(self, s, t):
        return P.REF_MS_POS[self.modelo][s][t]

    def celula(self, s, t, ms_pos=None, decode=None, n=5, prefills=None, cpu=None, needed=None, stderr=None, cmdline=None, serie_meta=None, ids=None, p4=True,
               gpu=True, falha=None, portao=True, extra_meta=None, m=None):
        m = m or self.modelo
        dm = DADOS_MODELO[m]
        if ms_pos is None:
            ms_pos = P.REF_MS_POS[m][s][t]
        if decode is None:
            decode = ms_pos / RATIO[s]
        nome = "lastro-%s-%s-%s-quente" % (m, s, t)
        d = os.path.join(self.saida, "celulas", nome)
        meta = {"lado": "lastro", "serie": serie_meta or P.SERIE_META[s], "fracao_slots": "1.0" if s == "L100" else "", "modelo": m, "texto": t, "estado": "quente", "n_medidas": n,
                "portao_ok": portao, "portao_motivo": "" if portao else "CPU ocupada", "rc": 0, "inicio_epoch": EPOCH_INI, "inicio_coleta_epoch": EPOCH_INI, "fim_epoch": EPOCH_FIM, "monitor_falha": ""}
        if s == "LC":
            meta["camadas_cpu"] = str(dm["N"])
        meta.update(extra_meta or {})
        grava(os.path.join(d, "celula.json"), meta)
        self.monitor(d)
        grava(os.path.join(d, "portao.txt"), "pmon_antes_veredito=OK\ncompute_apps_veredito=OK\nportao=OK\n")
        base = "pp --gera --celula %s:/m --so-texto %s " % (m, t)
        if cmdline is None:
            cmdline = {"L10": base + "--repete %d --saida x" % n, "L100": base + "--fracao-slots 1.0 --repete %d --saida x" % n, "LC": base + "--camadas-cpu %d --repete %d --saida x" % (dm["N"], n)}[s]
        grava(os.path.join(d, "cmdline.txt"), cmdline + "\n")
        L, E, N = dm["L"], dm["E"], dm["N"]
        if stderr is None:
            stderr = {"L10": "slots %d de %d" % (int(0.10 * L * E), L * E), "L100": "slots %d de %d" % (L * E, L * E),
                      "LC": "camadas-cpu %d de %d; slots %d de %d" % (N, L, (L - N) * E, L * E)}[s]
        if needed is None:
            needed = 24 * 100 if s == "LC" else 1000
        if cpu is None:
            cpu = {"L10": needed * 9 // 10, "L100": 0, "LC": needed * N // L}[s]
        pref = prefills if prefills is not None else [ms_pos * NPOS[t]] * n

        def entrada(rot, pre, desc):
            ps = [decode] * 63
            return {"rotulo": rot, "descartada": desc, "ids": ids if ids is not None else IDS[t], "gerados": GERADOS, "passo_ms": ps, "prefill_ms": pre, "total_ms": pre + sum(ps) + 500.0,
                    "T_sequencia_ms": 500.0, "media_por_passo_ms": sum(ps) / len(ps), "cont": {"needed": needed, "hits": 5, "cpu": cpu}, "falha": falha}
        ents = [entrada("aquecimento", pref[0], True)] + [entrada("quente rep %d" % (i + 1), pref[i], False) for i in range(len(pref))]
        pd = os.path.join(d, "p1")
        grava(os.path.join(pd, "rc.txt"), "0\n")
        grava(os.path.join(pd, "wall_ms.txt"), "%d\n" % (sum(e["total_ms"] for e in ents) + 3000))
        grava(os.path.join(pd, "stderr.txt"), stderr + "\n")
        grava(os.path.join(pd, "stdout.txt"), "")
        grava(os.path.join(pd, "saida", "medidas-%s.json" % m), {"tipo": "medidas-gera", "modelo": m, "texto": t, "medidas": ents})
        grava(os.path.join(pd, "saida", "gera-final.txt"), "tipo=gera-final\nintegro=1\n")
        grava(os.path.join(pd, "saida", "gera-%s.json" % m), {"falha": None, "integro": True})
        if p4:
            if not gpu and os.path.exists(os.path.join(self.saida, "p4", nome + ".gpu.csv")):
                os.remove(os.path.join(self.saida, "p4", nome + ".gpu.csv"))
            if gpu:
                grava(os.path.join(self.saida, "p4", nome + ".gpu.csv"), "2026/11/09 04:00:00.000, 10 %\n2026/11/09 04:00:00.100, 90 %\n2026/11/09 04:00:00.200, 50 %\n")
            grava(os.path.join(self.saida, "p4", nome + ".cpu.csv"), "1794000000000,111,100.0\n1794000000100,111,120.0\n1794000000200,111,110.0\n")
        return d

    def tudo(self, **kw):
        for s in SERIES:
            for t in TEXTOS:
                self.celula(s, t, **kw)

    def roda(self):
        return P.processa(self.saida, self.ids, self.refs, self.textos)


class TestDentroDaPrevisao(Base):
    def test_tudo_bom(self):
        self.tudo()
        r = self.roda()
        self.assertTrue(all(c["status"] == "OK" for c in r["celulas"].values()), [c["motivos"] for c in r["celulas"].values() if c["status"] != "OK"])
        for s in SERIES:
            self.assertEqual(r["series"][s]["status"], "OK")
            for p in ("P1", "P2", "P3"):
                self.assertEqual(r["series"][s]["previsoes"][p]["resultado"], "dentro da previsão", (s, p, r["series"][s]["previsoes"][p]))

    def test_tabela_e_numeros(self):
        self.tudo()
        r = self.roda()
        c = r["celulas"]["lastro-1b-L10-curto-quente"]
        self.assertEqual(c["n_posicoes"], 89)
        self.assertAlmostEqual(c["prefill_ms_mediana"], 8.9 * 89, places=6)
        self.assertAlmostEqual(c["ms_por_posicao"], 8.9, places=9)
        self.assertAlmostEqual(c["razao_p3"], 0.42, places=9)
        self.assertEqual(r["celulas"]["lastro-1b-L10-longo-quente"]["n_posicoes"], 364)
        self.assertEqual(r["celulas"]["lastro-1b-L10-medio-quente"]["n_posicoes"], 170)
        self.assertEqual(c["n_medidas"], 5)

    def test_3b_com_referencias_do_3b(self):
        self.modelo = "3b"
        grava(os.path.join(self.saida, "orcamento.json"), {"modelos": ["3b"]})
        self.tudo()
        r = self.roda()
        self.assertEqual(r["modelo_da_janela"], "3b")
        self.assertTrue(all(c["status"] == "OK" for c in r["celulas"].values()), [c["motivos"] for c in r["celulas"].values() if c["status"] != "OK"])
        self.assertEqual(r["series"]["L10"]["previsoes"]["P2"]["referencia_ms_por_posicao"]["medio"], 17.7)

    def test_tres_medidas_aceitas_quatro_nao(self):
        self.celula("L10", "curto", n=3)
        self.assertEqual(self.roda()["celulas"]["lastro-1b-L10-curto-quente"]["status"], "OK")
        self.celula("L10", "curto", n=4)
        c = self.roda()["celulas"]["lastro-1b-L10-curto-quente"]
        self.assertEqual(c["status"], "SEM PROVA")
        self.assertTrue(any("n_medidas" in x for x in c["motivos"]))

    def test_mediana_resiste_a_uma_medida_ruim(self):
        self.tudo()
        base = 8.9 * 89
        self.celula("L10", "curto", prefills=[base, base, base, base, base * 3])   # um outlier: a mediana não muda
        c = self.roda()["celulas"]["lastro-1b-L10-curto-quente"]
        self.assertAlmostEqual(c["prefill_ms_mediana"], base, places=6)
        self.assertEqual(c["prefill"]["max"], base * 3)


class TestPrevisoes(Base):
    def test_p1_fora_p2_fora_p3_fora_num_texto(self):
        self.tudo()
        self.celula("L10", "longo", ms_pos=12.0)       # 12 contra 8,9 e 9,0: >15% da média; 41% acima da referência (8,5); razão 12/20,2 = 0,59
        r = self.roda()
        pv = r["series"]["L10"]["previsoes"]
        self.assertEqual(pv["P1"]["resultado"], "fora da previsão")
        self.assertGreater(pv["P1"]["desvio_max"], 0.15)
        self.assertEqual(pv["P2"]["resultado"], "fora da previsão")
        self.assertGreater(pv["P2"]["desvios"]["longo"], 0.10)
        self.assertLess(pv["P2"]["desvios"]["curto"], 0.01)
        # as outras séries não são afetadas
        self.assertEqual(r["series"]["L100"]["previsoes"]["P1"]["resultado"], "dentro da previsão")

    def test_p1_dentro_na_borda_e_fora_logo_acima(self):
        # médias: valores 100, 100, X; desvio de X em torno da média; X=130 => média 110, desvio 18% (fora); X=115 => média 105, desvio 9,5% (dentro)
        self.tudo()
        for t, v in (("curto", 8.9), ("medio", 8.9), ("longo", 8.9 * 1.15)):
            self.celula("L10", t, ms_pos=v)
        self.assertEqual(self.roda()["series"]["L10"]["previsoes"]["P1"]["resultado"], "dentro da previsão")
        for t, v in (("curto", 8.9), ("medio", 8.9), ("longo", 8.9 * 1.30)):
            self.celula("L10", t, ms_pos=v)
        self.assertEqual(self.roda()["series"]["L10"]["previsoes"]["P1"]["resultado"], "fora da previsão")

    def test_p2_fora_com_p1_dentro(self):
        self.tudo()
        for t in TEXTOS:
            self.celula("L10", t, ms_pos=P.REF_MS_POS["1b"]["L10"][t] * 1.30)     # todos 30% acima: uniforme
        pv = self.roda()["series"]["L10"]["previsoes"]
        self.assertEqual(pv["P1"]["resultado"], "dentro da previsão")
        self.assertEqual(pv["P2"]["resultado"], "fora da previsão")
        self.assertAlmostEqual(pv["P2"]["desvio_max"], 0.30, places=6)

    def test_p2_bordas(self):
        self.tudo()
        for t in TEXTOS:
            self.celula("L100", t, ms_pos=1.58 * 1.09)
        self.assertEqual(self.roda()["series"]["L100"]["previsoes"]["P2"]["resultado"], "dentro da previsão")
        for t in TEXTOS:
            self.celula("L100", t, ms_pos=1.58 * 1.11)
        self.assertEqual(self.roda()["series"]["L100"]["previsoes"]["P2"]["resultado"], "fora da previsão")
        for t in TEXTOS:
            self.celula("L100", t, ms_pos=1.58 * 0.89)
        self.assertEqual(self.roda()["series"]["L100"]["previsoes"]["P2"]["resultado"], "fora da previsão")

    def test_p3_fora_quando_o_decode_muda(self):
        self.tudo()
        for t in TEXTOS:
            self.celula("L100", t, decode=1.58 / 0.665 * 2.0)     # decode 2x mais lento: razão 0,33, fora da faixa 0,56 a 0,77
        pv = self.roda()["series"]["L100"]["previsoes"]
        self.assertEqual(pv["P3"]["resultado"], "fora da previsão")
        self.assertAlmostEqual(pv["P3"]["razoes"]["curto"], 0.3325, places=3)
        self.assertEqual(pv["P1"]["resultado"], "dentro da previsão")
        self.assertEqual(pv["P2"]["resultado"], "dentro da previsão")

    def test_p3_faixas_por_serie(self):
        lo, hi = P.P3_FAIXA["L100"]
        self.assertAlmostEqual(lo, 0.66 * 0.85)
        self.assertAlmostEqual(hi, 0.67 * 1.15)
        lo, hi = P.P3_FAIXA["L10"]
        self.assertAlmostEqual(lo, 0.41 * 0.85)
        self.assertAlmostEqual(hi, 0.43 * 1.15)
        self.assertEqual(P.P3_FAIXA["LC"], P.P3_FAIXA["L10"])
        self.tudo()
        # 0,56 no L100 está fora de 0,561...: fora; 0,57 dentro; no L10 0,5 fora e 0,48 dentro
        for t in TEXTOS:
            self.celula("L100", t, decode=1.58 / 0.56)
            self.celula("L10", t, decode=P.REF_MS_POS["1b"]["L10"][t] / 0.48)
        pv = self.roda()["series"]
        self.assertEqual(pv["L100"]["previsoes"]["P3"]["resultado"], "fora da previsão")
        self.assertEqual(pv["L10"]["previsoes"]["P3"]["resultado"], "dentro da previsão")

    def test_p3_uma_celula_fora_basta(self):
        self.tudo()
        self.celula("LC", "longo", decode=8.7 / 0.30)
        self.assertEqual(self.roda()["series"]["LC"]["previsoes"]["P3"]["resultado"], "fora da previsão")


class TestValidade(Base):
    def test_lc_posicionamento_violado_sem_prova(self):
        self.tudo()
        self.celula("LC", "curto", cpu=2200 + 1)       # cpu*24 != needed*22
        r = self.roda()
        c = r["celulas"]["lastro-1b-LC-curto-quente"]
        self.assertEqual(c["status"], "SEM PROVA")
        self.assertTrue(any("posicionamento" in x for x in c["motivos"]), c["motivos"])
        self.assertEqual(r["series"]["LC"]["status"], "SEM PROVA")
        for p in ("P1", "P2", "P3"):
            self.assertIn("SEM PROVA", r["series"]["LC"]["previsoes"][p]["resultado"])
        self.assertEqual(r["series"]["L10"]["status"], "OK")

    def test_lc_exato_vale(self):
        self.tudo()
        self.celula("LC", "curto", needed=2400, cpu=2200)      # 2200*24 == 2400*22
        self.assertEqual(self.roda()["celulas"]["lastro-1b-LC-curto-quente"]["status"], "OK")

    def test_lc_linha_camadas_errada_ou_ausente(self):
        self.tudo()
        self.celula("LC", "medio", stderr="camadas-cpu 21 de 24; slots 96 de 768")
        self.assertEqual(self.roda()["celulas"]["lastro-1b-LC-medio-quente"]["status"], "SEM PROVA")
        self.celula("LC", "medio", stderr="nada")
        self.assertEqual(self.roda()["celulas"]["lastro-1b-LC-medio-quente"]["status"], "SEM PROVA")

    def test_l100_com_cpu_diferente_de_zero(self):
        self.tudo()
        self.celula("L100", "curto", cpu=5)
        c = self.roda()["celulas"]["lastro-1b-L100-curto-quente"]
        self.assertEqual(c["status"], "SEM PROVA")
        self.assertTrue(any("cpu/needed" in x for x in c["motivos"]), c["motivos"])

    def test_l100_slots_menor_que_o_total_ou_sem_linha(self):
        self.tudo()
        self.celula("L100", "curto", stderr="slots 76 de 768")
        self.assertEqual(self.roda()["celulas"]["lastro-1b-L100-curto-quente"]["status"], "SEM PROVA")
        self.celula("L100", "curto", stderr="")
        self.assertEqual(self.roda()["celulas"]["lastro-1b-L100-curto-quente"]["status"], "SEM PROVA")

    def test_l10_exige_a_linha_slots_padrao(self):
        self.tudo()
        self.celula("L10", "curto", stderr="")
        c = self.roda()["celulas"]["lastro-1b-L10-curto-quente"]
        self.assertEqual(c["status"], "SEM PROVA")
        self.assertTrue(any("slots" in x for x in c["motivos"]), c["motivos"])
        self.celula("L10", "curto", stderr="slots 768 de 768")       # fração 100% impressa na série padrão
        self.assertEqual(self.roda()["celulas"]["lastro-1b-L10-curto-quente"]["status"], "SEM PROVA")
        self.celula("L10", "curto", stderr="slots 77 de 768")
        self.assertEqual(self.roda()["celulas"]["lastro-1b-L10-curto-quente"]["status"], "SEM PROVA")
        self.celula("L10", "curto", stderr="slots 76 de 768")
        self.assertEqual(self.roda()["celulas"]["lastro-1b-L10-curto-quente"]["status"], "OK")

    def test_l10_3b_linha_padrao(self):
        self.modelo = "3b"
        grava(os.path.join(self.saida, "orcamento.json"), {"modelos": ["3b"]})
        self.assertEqual(P.linha_slots_padrao("3b"), (128, 1280))
        self.assertEqual(P.linha_slots_padrao("1b"), (76, 768))
        self.celula("L10", "curto", stderr="slots 128 de 1280")
        self.assertEqual(self.roda()["celulas"]["lastro-3b-L10-curto-quente"]["status"], "OK")
        self.celula("L10", "curto", stderr="slots 76 de 768")
        self.assertEqual(self.roda()["celulas"]["lastro-3b-L10-curto-quente"]["status"], "SEM PROVA")

    def test_l10_com_flag_de_outra_serie_na_linha_de_comando(self):
        self.celula("L10", "curto", cmdline="pp --gera --so-texto curto --fracao-slots 1.0 --repete 5")
        self.assertEqual(self.roda()["celulas"]["lastro-1b-L10-curto-quente"]["status"], "SEM PROVA")
        self.celula("L10", "curto", cmdline="")
        self.assertEqual(self.roda()["celulas"]["lastro-1b-L10-curto-quente"]["status"], "SEM PROVA")

    def test_serie_do_celula_json_errada(self):
        self.celula("L100", "curto", serie_meta="L")
        self.assertEqual(self.roda()["celulas"]["lastro-1b-L100-curto-quente"]["status"], "SEM PROVA")

    def test_posicoes_erradas_ou_ids_diferentes(self):
        self.celula("L10", "curto", ids=list(range(88)))
        c = self.roda()["celulas"]["lastro-1b-L10-curto-quente"]
        self.assertEqual(c["status"], "SEM PROVA")
        self.celula("L10", "curto", ids=[7] * 89)       # contagem certa, ids diferentes de cmp-ids-prompt.json
        self.assertEqual(self.roda()["celulas"]["lastro-1b-L10-curto-quente"]["status"], "SEM PROVA")

    def test_cmp_ids_com_contagem_diferente_de_89_170_364(self):
        grava(self.ids, {"modelos": {m: {t: {"ids": list(range(10))} for t in IDS} for m in ("1b", "3b")}})
        self.celula("L10", "curto", ids=list(range(10)))
        c = self.roda()["celulas"]["lastro-1b-L10-curto-quente"]
        self.assertEqual(c["status"], "SEM PROVA")
        self.assertTrue(any("esperado 89" in x for x in c["motivos"]), c["motivos"])

    def test_portao_falho_e_medida_com_falha(self):
        self.celula("L10", "curto", portao=False)
        self.assertEqual(self.roda()["celulas"]["lastro-1b-L10-curto-quente"]["status"], "SEM PROVA")
        self.celula("L10", "curto", falha="erro de kernel")
        self.assertEqual(self.roda()["celulas"]["lastro-1b-L10-curto-quente"]["status"], "SEM PROVA")

    def test_celula_ausente_e_nao_rodou(self):
        for t in ("curto", "medio"):
            self.celula("LC", t)
        r = self.roda()
        c = r["celulas"]["lastro-1b-LC-longo-quente"]
        self.assertEqual(c["status"], "NÃO RODOU")
        self.assertIsNone(c.get("ms_por_posicao"))
        self.assertEqual(r["series"]["LC"]["status"], "NÃO RODOU")
        self.assertTrue(r["series"]["LC"]["previsoes"]["P1"]["resultado"].startswith("NÃO RODOU"))
        self.assertNotEqual(r["series"]["LC"]["previsoes"]["P1"]["resultado"], "dentro da previsão")
        # sem nenhuma célula: série inteira NÃO RODOU
        self.assertEqual(r["series"]["L10"]["status"], "NÃO RODOU")
        self.assertTrue(r["series"]["L10"]["previsoes"]["P1"]["resultado"].startswith("NÃO RODOU"))

    def test_serie_parcial_vira_nao_rodou_ou_sem_prova_nunca_previsao(self):
        self.celula("LC", "curto")
        self.celula("LC", "medio")
        r = self.roda()
        self.assertEqual(r["series"]["LC"]["status"], "NÃO RODOU")
        self.assertTrue(r["series"]["LC"]["previsoes"]["P2"]["resultado"].startswith("NÃO RODOU"))

    def test_smoke_falho_ou_ausente_invalida_a_serie(self):
        self.tudo()
        self.smoke("L100", False)
        r = self.roda()
        self.assertEqual(r["series"]["L100"]["status"], "SEM PROVA")
        self.assertTrue(any("smoke L100 FALHOU" in x for x in r["celulas"]["lastro-1b-L100-curto-quente"]["motivos"]))
        self.assertEqual(r["series"]["LC"]["status"], "OK")
        self.assertFalse(r["smokes"]["L100"]["ok"])
        os.remove(os.path.join(self.saida, "smoke-LC", "veredito.txt"))
        r = self.roda()
        self.assertEqual(r["series"]["LC"]["status"], "SEM PROVA")
        self.assertTrue(any("smoke ausente" in x for x in r["celulas"]["lastro-1b-LC-curto-quente"]["motivos"]))
        self.assertEqual(r["series"]["L10"]["status"], "OK")       # o L10 não exige smoke

    def test_integridade_quebrada_sem_prova_em_tudo(self):
        self.tudo()
        grava(os.path.join(self.saida, "integridade-depois.txt"), "binario ZZZ\ngguf def\n")
        r = self.roda()
        self.assertTrue(r["integridade_problemas"])
        self.assertTrue(all(c["status"] == "SEM PROVA" for c in r["celulas"].values()))
        self.assertTrue(all(s["status"] == "SEM PROVA" for s in r["series"].values()))
        grava(os.path.join(self.saida, "integridade-depois.txt"), "binario abc\ngguf def\n")
        grava(os.path.join(self.saida, "integridade-fixos.txt"), "criterio-perfil aaa bbb FALHA\n")
        r = self.roda()
        self.assertTrue(all(c["status"] == "SEM PROVA" for c in r["celulas"].values()))

    def test_memoria_do_host_fora_do_nominal(self):
        self.tudo()
        grava(os.path.join(self.saida, "ambiente.txt"), "memoria_mts=2133\n")
        self.assertTrue(all(c["status"] == "SEM PROVA" for c in self.roda()["celulas"].values()))

    def test_orcamento_sem_modelo_unico_recusa(self):
        grava(os.path.join(self.saida, "orcamento.json"), {"modelos": ["1b", "3b"]})
        with self.assertRaises(ValueError):
            self.roda()
        self.assertEqual(P.main(["--saida", self.saida, "--ids", self.ids, "--refs", self.refs, "--textos", self.textos]), 2)


class TestP4(Base):
    def test_amostras_e_resolucao(self):
        self.tudo()
        r = self.roda()
        p4 = r["celulas"]["lastro-1b-L10-curto-quente"]["p4"]
        self.assertEqual(p4["gpu_util_pct"]["n"], 3)
        self.assertAlmostEqual(p4["gpu_util_pct"]["media"], 50.0)
        self.assertEqual(p4["gpu_util_pct"]["max"], 90.0)
        self.assertEqual(p4["cpu_pct_processo"]["n"], 3)
        self.assertAlmostEqual(p4["cpu_pct_processo"]["media"], 110.0)
        self.assertEqual(p4["cpu_pct_processo"]["max"], 120.0)
        self.assertIn("célula inteira", p4["cobre"])
        # prefill 89*8,9 = 792 ms (< 3 s): sem resolução; L10 longo 364*8,5 = 3094 ms (>= 3 s): dica
        self.assertTrue(p4["resolucao"].startswith("sem resolução: prefill < ~3 s"), p4["resolucao"])
        self.assertTrue(r["celulas"]["lastro-1b-L10-longo-quente"]["p4"]["resolucao"].startswith("dica"), r["celulas"]["lastro-1b-L10-longo-quente"]["p4"]["resolucao"])
        # L100 (1,58 ms/posição) nunca chega a 3 s no 1B
        self.assertTrue(r["celulas"]["lastro-1b-L100-longo-quente"]["p4"]["resolucao"].startswith("sem resolução"))

    def test_gpu_ausente_sem_resolucao_e_a_celula_segue_valida(self):
        self.tudo()
        self.celula("L10", "longo", gpu=False)       # prefill >= 3 s, mas sem amostras de GPU
        r = self.roda()
        c = r["celulas"]["lastro-1b-L10-longo-quente"]
        self.assertEqual(c["status"], "OK")
        self.assertEqual(c["p4"]["gpu_util_pct"]["n"], 0)
        self.assertIsNone(c["p4"]["gpu_util_pct"]["media"])
        self.assertTrue(c["p4"]["resolucao"].startswith("sem resolução"), c["p4"]["resolucao"])
        self.assertIn("GPU", c["p4"]["resolucao"])
        self.assertEqual(r["series"]["L10"]["previsoes"]["P1"]["resultado"], "dentro da previsão")

    def test_arquivos_p4_vazios_ou_com_lixo(self):
        self.tudo()
        grava(os.path.join(self.saida, "p4", "lastro-1b-L10-curto-quente.gpu.csv"), "lixo\n\n,\nnvidia-smi falhou\n2026/11/09 04:00:00.000, 130 %\n")
        grava(os.path.join(self.saida, "p4", "lastro-1b-L10-curto-quente.cpu.csv"), "")
        p4 = self.roda()["celulas"]["lastro-1b-L10-curto-quente"]["p4"]
        self.assertEqual(p4["gpu_util_pct"]["n"], 0)
        self.assertEqual(p4["cpu_pct_processo"]["n"], 0)
        self.assertIn("CPU", p4["resolucao"])

    def test_sem_pasta_p4(self):
        self.tudo(p4=False)
        r = self.roda()
        c = r["celulas"]["lastro-1b-L10-longo-quente"]
        self.assertEqual(c["status"], "OK")
        self.assertTrue(c["p4"]["resolucao"].startswith("sem resolução"))


class TestSaida(Base):
    def test_main_grava_json_estrito_e_md_sem_passa(self):
        self.tudo()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = P.main(["--saida", self.saida, "--ids", self.ids, "--refs", self.refs, "--textos", self.textos])
        self.assertEqual(rc, 0)
        self.assertIn("células OK 9", buf.getvalue())
        with open(os.path.join(self.saida, "perfil-resultado.json"), encoding="utf-8") as f:
            j = json.load(f, parse_constant=lambda c: self.fail("constante não finita: " + c))
        self.assertTrue(j["descritivo"])
        with open(os.path.join(self.saida, "perfil-resultado.md"), encoding="utf-8") as fm:
            md = fm.read()
        self.assertNotIn("PASSA", md.replace("sem PASSA", "").replace("não declara PASSA", "").replace("sem meta e sem PASSA", ""))
        for s in SERIES:
            self.assertIn("| %s | curto | 89 |" % s, md)
            self.assertIn("| %s | longo | 364 |" % s, md)
        self.assertIn("P1", md)
        self.assertIn("dentro da previsão", md)
        self.assertIn("P4 (descritivo)", md)
        self.assertIn("célula inteira", md)

    def test_md_mostra_nao_rodou_e_sem_prova(self):
        self.celula("L10", "curto")
        self.celula("LC", "curto", cpu=1)
        r = self.roda()
        md = P.relatorio(r)
        self.assertIn("NÃO RODOU", md)
        self.assertIn("SEM PROVA", md)
        self.assertIn("## Células sem resultado", md)

    def test_pasta_inexistente(self):
        self.assertEqual(P.main(["--saida", os.path.join(self.d, "nao-existe")]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=1)
