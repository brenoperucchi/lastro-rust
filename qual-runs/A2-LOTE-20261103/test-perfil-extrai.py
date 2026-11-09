#!/usr/bin/env python3
"""Testes do perfil-extrai.py com dados SINTÉTICOS. Sem GPU, sem rodar nada do motor.

    python3 -I qual-runs/A2-LOTE-20261103/test-perfil-extrai.py
"""
import importlib.util
import io
import json
import os
import re
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


def ts(t):
    return time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(t)) + ".500"


def le(p):
    with open(p, encoding="utf-8") as fh:
        return fh.read()


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
        # integridade FIXA completa e coerente (cada chave com sha de 64 hex; o binário tem o sha da janela)
        self.shas = {k: ("%02d" % i) * 32 for i, k in enumerate(P.FIXO_CHAVES + ("tokenizer-%s" % self.modelo,))}
        self.shas["binario-lastro"] = "e4c18f3a9dea7d41236d1b721046f0d69a3be07863a5be8a2cc1a262b9e331d4"
        self.shas["gguf-extra"] = "d" * 64
        self.manifesto()
        grava(os.path.join(self.saida, "orcamento.json"), {"modelos": [self.modelo], "limite_s": 1200, "decorrido_s": 600, "cortes": [], "nao_rodou": []})
        grava(os.path.join(self.saida, "ambiente.txt"), "modelo=%s\nmemoria_mts=2400\n" % self.modelo)
        for s in ("L100", "LC"):
            self.smoke(s, True)

    def tearDown(self):
        self._t.cleanup()

    def para_3b(self):
        self.modelo = "3b"
        grava(os.path.join(self.saida, "orcamento.json"), {"modelos": ["3b"]})
        self.shas.pop("tokenizer-1b")
        self.shas["tokenizer-3b"] = "c" * 64
        self.manifesto()

    def manifesto(self, chaves=None, estado="OK", depois_igual=True):
        ks = [k for k in self.shas if k != "gguf-extra"] if chaves is None else chaves
        antes = "".join("%s %s\n" % (k, h) for k, h in self.shas.items())
        grava(os.path.join(self.saida, "integridade-antes.txt"), antes)
        grava(os.path.join(self.saida, "integridade-depois.txt"), antes)
        grava(os.path.join(self.saida, "integridade-fixos.txt"), "".join("%s %s %s %s\n" % (k, self.shas[k], self.shas[k], estado) for k in ks))

    def smoke(self, s, ok, motivo="memória usada da GPU 95% > 90%"):
        grava(os.path.join(self.saida, "smoke-%s" % s, "veredito.txt"), "veredito=OK\n" if ok else "veredito=FALHA\nmotivo=%s\n" % motivo)

    # ---- construtor de célula
    def monitor(self, d):
        linhas = ["%s.000, 5, 2000, 10000, 60, 100" % time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(t_)) for t_ in range(EPOCH_INI, EPOCH_FIM + 1)]
        grava(os.path.join(d, "monitor.csv"), "\n".join(linhas) + "\n")

    def ms_pos_ref(self, s, t):
        return P.REF_MS_POS[self.modelo][s][t]

    def celula(self, s, t, ms_pos=None, decode=None, n=5, prefills=None, cpu=None, needed=None, stderr=None, cmdline=None, serie_meta=None, ids=None, p4=True,
               gpu=True, falha=None, portao=True, extra_meta=None, m=None, gerados_reps=None, gpu_linhas=None):
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

        def entrada(rot, pre, desc, i_rep=None):
            ps = [decode] * 63
            return {"rotulo": rot, "descartada": desc, "ids": ids if ids is not None else IDS[t], "gerados": (gerados_reps[i_rep] if gerados_reps and i_rep is not None else GERADOS), "passo_ms": ps, "prefill_ms": pre, "total_ms": pre + sum(ps) + 500.0,
                    "T_sequencia_ms": 500.0, "media_por_passo_ms": sum(ps) / len(ps), "cont": {"needed": needed, "hits": 5, "cpu": cpu}, "falha": falha}
        ents = [entrada("aquecimento", pref[0], True)] + [entrada("quente rep %d" % (i + 1), pref[i], False, i) for i in range(len(pref))]
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
                grava(os.path.join(self.saida, "p4", nome + ".gpu.csv"), gpu_linhas if gpu_linhas is not None else "%s, 10 %%\n%s, 90 %%\n%s, 50 %%\n" % (ts(EPOCH_INI + 1), ts(EPOCH_INI + 2), ts(EPOCH_INI + 3)))
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
        self.para_3b()
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
        self.para_3b()
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
        self.manifesto()
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
        grava(os.path.join(self.saida, "p4", "lastro-1b-L10-curto-quente.gpu.csv"), "lixo\n\n,\nnvidia-smi falhou\n%s, 130 %%\n" % ts(EPOCH_INI + 1))
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


class TestManifestoFixo(Base):
    def todas_sem_prova(self, r):
        self.assertTrue(r["integridade_problemas"])
        self.assertTrue(all(c["status"] == "SEM PROVA" for c in r["celulas"].values()))
        for s in SERIES:
            self.assertEqual(r["series"][s]["status"], "SEM PROVA")
            for p in ("P1", "P2", "P3"):
                self.assertTrue(r["series"][s]["previsoes"][p]["resultado"].startswith("SEM PROVA"), r["series"][s]["previsoes"][p])
                self.assertNotIn("desvios", r["series"][s]["previsoes"][p])

    def test_completo_e_coerente_preserva_o_caminho_valido(self):
        self.tudo()
        r = self.roda()
        self.assertEqual(r["integridade_problemas"], [])
        self.assertTrue(all(c["status"] == "OK" for c in r["celulas"].values()))
        self.assertEqual(r["series"]["L10"]["previsoes"]["P1"]["resultado"], "dentro da previsão")

    def test_manifesto_ausente(self):
        self.tudo()
        os.remove(os.path.join(self.saida, "integridade-fixos.txt"))
        r = self.roda()
        self.todas_sem_prova(r)
        self.assertTrue(any("ausente" in x for x in r["integridade_problemas"]))

    def test_interrupcao_entre_depois_e_fixos(self):
        # simula a janela morta depois de gravar integridade-depois.txt e antes do manifesto (e do orcamento.json final): sem o manifesto, nada vale
        self.tudo()
        self.assertTrue(os.path.exists(os.path.join(self.saida, "integridade-depois.txt")))
        os.remove(os.path.join(self.saida, "integridade-fixos.txt"))
        self.todas_sem_prova(self.roda())
        # o roteiro grava o manifesto em arquivo temporário e o move de uma vez: nunca fica truncado
        sh = le(os.path.join(AQUI, "roda-perfil-janela.sh"))
        self.assertIn('mv "$SAIDA/.integridade-fixos.tmp" "$SAIDA/integridade-fixos.txt"', sh)
        self.assertNotIn('>> "$SAIDA/integridade-fixos.txt"', sh)

    def test_manifesto_vazio(self):
        self.tudo()
        grava(os.path.join(self.saida, "integridade-fixos.txt"), "")
        r = self.roda()
        self.todas_sem_prova(r)
        self.assertTrue(any("vazio" in x for x in r["integridade_problemas"]))

    def test_manifesto_truncado(self):
        self.tudo()
        arq = os.path.join(self.saida, "integridade-fixos.txt")
        txt = le(arq)
        grava(arq, txt[:len(txt) - 40])        # corta no meio de uma linha
        r = self.roda()
        self.todas_sem_prova(r)
        self.assertTrue(any("truncada" in x or "faltando" in x for x in r["integridade_problemas"]))
        grava(arq, txt[:len(txt) // 2])
        self.todas_sem_prova(self.roda())

    def test_chave_removida(self):
        self.tudo()
        for k in ("binario-lastro", "tokenizer-1b", "criterio-perfil", "ref5c-3b"):
            self.manifesto(chaves=[x for x in self.shas if x not in ("gguf-extra", k)])
            r = self.roda()
            self.todas_sem_prova(r)
            self.assertTrue(any("faltando" in x and k in x for x in r["integridade_problemas"]), (k, r["integridade_problemas"]))

    def test_nao_conferido(self):
        self.tudo()
        self.manifesto(estado="NAO-CONFERIDO")
        r = self.roda()
        self.todas_sem_prova(r)
        self.assertTrue(any("NAO-CONFERIDO" in x for x in r["integridade_problemas"]))

    def test_hash_esperado_diferente_do_observado_ou_ilegivel(self):
        self.tudo()
        arq = os.path.join(self.saida, "integridade-fixos.txt")
        linhas = le(arq).splitlines()
        k, e, o, st = linhas[0].split()
        grava(arq, "\n".join(["%s %s %s OK" % (k, e, "f" * 64)] + linhas[1:]) + "\n")
        self.todas_sem_prova(self.roda())
        grava(arq, "\n".join(["%s - - FALHA ilegível" % k] + linhas[1:]) + "\n")
        self.todas_sem_prova(self.roda())
        grava(arq, "\n".join(["%s curto curto OK" % k] + linhas[1:]) + "\n")
        self.todas_sem_prova(self.roda())

    def test_sha_do_manifesto_diferente_do_de_antes(self):
        self.tudo()
        arq = os.path.join(self.saida, "integridade-fixos.txt")
        linhas = le(arq).splitlines()
        k, e, o, st = linhas[0].split()
        grava(arq, "\n".join(["%s %s %s OK" % (k, "e" * 64, "e" * 64)] + linhas[1:]) + "\n")
        self.todas_sem_prova(self.roda())

    def test_chaves_do_extrator_batem_com_o_FIXO_do_roteiro(self):
        sh = le(os.path.join(AQUI, "roda-perfil-janela.sh"))
        bloco = sh[sh.index("declare -A FIXO=("):]
        bloco = bloco[:bloco.index("\n)\n")]
        chaves = re.findall(r"^\s+\[([a-z0-9-]+)\]=", bloco, re.M)
        esperado = set(P.FIXO_CHAVES) | {"tokenizer-1b", "tokenizer-3b"}
        self.assertEqual(set(chaves), esperado)


class TestSequencia(Base):
    def test_identica(self):
        self.tudo()
        r = self.roda()
        for s in SERIES:
            self.assertEqual(r["series"][s]["previsoes"]["P3"]["sequencia"], "mesma sequência da referência 5c nos três textos")
        sq = r["celulas"]["lastro-1b-L10-curto-quente"]["sequencia"]
        self.assertTrue(sq["referencia_disponivel"])
        self.assertFalse(sq["sequencias_diferentes"])
        self.assertEqual(sq["divergencias"], [])

    def test_uma_repeticao_diverge(self):
        self.tudo()
        ruim = list(GERADOS); ruim[7] = 12345
        self.celula("L10", "curto", gerados_reps=[GERADOS, GERADOS, ruim, GERADOS, GERADOS])
        r = self.roda()
        c = r["celulas"]["lastro-1b-L10-curto-quente"]
        self.assertEqual(c["status"], "OK")                         # tempos mantidos
        dv = c["sequencia"]["divergencias"]
        self.assertEqual(len(dv), 1)
        self.assertEqual((dv[0]["processo"], dv[0]["rotulo"], dv[0]["passo"], dv[0]["id_medido"], dv[0]["id_referencia"], dv[0]["gap_referencia"]), ("p1", "quente rep 3", 7, 12345, GERADOS[7], 5.0))
        pv = r["series"]["L10"]["previsoes"]["P3"]
        self.assertEqual(pv["resultado"], "dentro da previsão")
        self.assertTrue(pv["sequencias_diferentes"])
        self.assertIn("sequências diferentes", pv["sequencia"])
        self.assertIn("curto", pv["sequencia"])
        md = P.relatorio(r)
        self.assertIn("sequência: sequências diferentes", md)
        self.assertIn("p1/quente rep 3: primeira divergência no passo 7, id medido 12345, id da referência %d, gap da referência 5.0000" % GERADOS[7], md)

    def test_todas_divergem(self):
        self.tudo()
        ruim = list(GERADOS); ruim[0] = 1
        self.celula("L100", "longo", gerados_reps=[ruim] * 5)
        c = self.roda()["celulas"]["lastro-1b-L100-longo-quente"]
        self.assertEqual(c["sequencia"]["n_medidas_divergentes"], 5)
        self.assertEqual(c["sequencia"]["n_medidas"], 5)
        self.assertEqual(c["status"], "OK")
        self.assertIn("5 de 5", c["sequencia"]["marca"])

    def test_999_vezes_64(self):
        self.tudo()
        self.celula("LC", "medio", gerados_reps=[[999] * 64] * 5)
        r = self.roda()
        c = r["celulas"]["lastro-1b-LC-medio-quente"]
        dv = c["sequencia"]["divergencias"][0]
        self.assertEqual((dv["passo"], dv["id_medido"], dv["id_referencia"], dv["gap_referencia"]), (0, 999, GERADOS[0], 5.0))
        self.assertEqual(c["status"], "OK")
        self.assertTrue(r["series"]["LC"]["previsoes"]["P3"]["sequencias_diferentes"])

    def test_referencia_ausente_nunca_vira_igualdade(self):
        self.tudo()
        for m in ("1b", "3b"):
            grava(os.path.join(self.refs, "ref5c-%s.json" % m), {"casos": [{"nome": "medio", "ids": IDS["medio"], "gerados": GERADOS, "gaps": [5.0] * 64}, {"nome": "longo", "ids": IDS["longo"], "gerados": GERADOS, "gaps": [5.0] * 64}]})
        r = self.roda()
        c = r["celulas"]["lastro-1b-L10-curto-quente"]
        self.assertFalse(c["sequencia"]["referencia_disponivel"])
        self.assertIsNone(c["sequencia"]["sequencias_diferentes"])
        self.assertIn("ausente", c["sequencia"]["marca"])
        pv = r["series"]["L10"]["previsoes"]["P3"]
        self.assertIn("NÃO verificada", pv["sequencia"])
        self.assertNotIn("mesma sequência", pv["sequencia"])
        self.assertIn("NÃO verificada", P.relatorio(r))


class TestP2Binario(Base):
    def test_md_traz_binarios_e_aviso_fixo(self):
        self.tudo()
        r = self.roda()
        md = P.relatorio(r)
        self.assertEqual(r["binario_da_janela"], self.shas["binario-lastro"])
        for s, b in (("L10", "f0a272db2df8aca6642073b830cc8234f2d2434e01c8ae6ef618886e6e3dc2df"), ("L100", "936d0c146d750f4849b7724ba5ae7ddd099618629a4e12bd2984758b801c3551"),
                     ("LC", "e4c18f3a9dea7d41236d1b721046f0d69a3be07863a5be8a2cc1a262b9e331d4")):
            self.assertEqual(P.BIN_REF[s], b)
            self.assertEqual(r["series"][s]["previsoes"]["P2"]["binario_referencia"], b)
            self.assertIn("binário da janela de referência %s; binário de agora %s" % (b, self.shas["binario-lastro"]), md)
        self.assertEqual(md.count("P2 compara com binário diferente no L10 e L100 (mudanças aditivas) e com amostragem P4 a 10 Hz; fora da previsão não isola a causa"), 3)


class TestGpuIntervalo(Base):
    def test_amostras_anteriores_ao_inicio_ficam_de_fora_e_sao_contadas(self):
        self.tudo()
        linhas = "".join("%s, 0 %%\n" % ts(EPOCH_INI - 5 + i) for i in range(5)) + "%s, 80 %%\n%s, 100 %%\n%s, 0 %%\n" % (ts(EPOCH_INI + 1), ts(EPOCH_FIM - 1), ts(EPOCH_FIM + 3))
        self.celula("L10", "curto", gpu_linhas=linhas)
        p4 = self.roda()["celulas"]["lastro-1b-L10-curto-quente"]["p4"]
        self.assertEqual(p4["gpu_util_pct"]["n"], 2)
        self.assertAlmostEqual(p4["gpu_util_pct"]["media"], 90.0)         # as 5 do portão ocioso (0%) não entram na média
        self.assertEqual(p4["gpu_amostras_fora_do_intervalo"], 6)         # 5 antes + 1 depois do fim
        self.assertEqual(p4["gpu_intervalo_epoch"], [EPOCH_INI, EPOCH_FIM])
        self.assertIn("(6)", P.relatorio(self.roda()))

    def test_so_amostras_fora_sem_resolucao(self):
        self.tudo()
        self.celula("L10", "curto", gpu_linhas="%s, 0 %%\n" % ts(EPOCH_INI - 3))
        p4 = self.roda()["celulas"]["lastro-1b-L10-curto-quente"]["p4"]
        self.assertEqual(p4["gpu_util_pct"]["n"], 0)
        self.assertEqual(p4["gpu_amostras_fora_do_intervalo"], 1)
        self.assertTrue(p4["resolucao"].startswith("sem resolução"))

    def test_carimbo_ilegivel_conta_como_fora(self):
        self.tudo()
        self.celula("L10", "curto", gpu_linhas="ontem, 50 %%\n%s, 20 %%\n" % ts(EPOCH_INI + 2))
        p4 = self.roda()["celulas"]["lastro-1b-L10-curto-quente"]["p4"]
        self.assertEqual((p4["gpu_util_pct"]["n"], p4["gpu_amostras_fora_do_intervalo"]), (1, 1))


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
