#!/usr/bin/env python3
"""Testes do cmp-extrai.py com dados SINTÉTICOS (e a linha real do 1B curto da 5c, se o arquivo da janela existir). Sem GPU.

    python3 -I qual-runs/A2-LOTE-20261103/test-cmp-extrai.py
"""
import importlib.machinery
import importlib.util
import json
import os
import tempfile
import time
import unittest

AQUI = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_loader("cmp_extrai", importlib.machinery.SourceFileLoader("cmp_extrai", os.path.join(AQUI, "cmp-extrai.py")))
X = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X)

EPOCH_INI, EPOCH_FIM = 1_700_000_000, 1_700_000_010      # início e fim (s) das células sintéticas; o monitor.csv as cobre a 1 Hz
GERADOS = list(range(100, 164))          # 64 ids
IDS = {"curto": [1, 2, 3, 4, 5], "medio": [1, 2, 3, 4, 5, 6, 7], "longo": [9] * 11}


def grava(p, conteudo):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(conteudo if isinstance(conteudo, str) else json.dumps(conteudo))


class Base(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.d = self._t.name
        self.saida = os.path.join(self.d, "saida")
        os.makedirs(self.saida)
        self.ids = os.path.join(self.d, "ids.json")
        grava(self.ids, {"modelos": {m: {t: {"ids": IDS[t]} for t in IDS} for m in ("1b", "3b")}})
        self.refs = os.path.join(self.d, "refs")
        for m in ("1b", "3b"):
            grava(os.path.join(self.refs, "ref5c-%s.json" % m),
                  {"casos": [{"nome": t, "ids": IDS[t], "gerados": GERADOS, "texto_gerado": "txt " + t, "gaps": [5.0] * 64} for t in IDS]})
        self.textos = os.path.join(self.d, "textos.json")
        grava(self.textos, {"textos": [{"nome": t, "ids_com_template": {"1b": len(IDS[t]), "3b": len(IDS[t])}} for t in IDS]})
        self.integra()

    def tearDown(self):
        self._t.cleanup()

    def integra(self, depois=None):
        grava(os.path.join(self.saida, "integridade-antes.txt"), "binario abc\ngguf def\n")
        grava(os.path.join(self.saida, "integridade-depois.txt"), depois or "binario abc\ngguf def\n")

    # ---- construtores de célula
    def celula_meta(self, nome, **kw):
        kw.setdefault("inicio_epoch", EPOCH_INI)
        kw.setdefault("fim_epoch", EPOCH_FIM)
        grava(os.path.join(self.saida, "celulas", nome, "celula.json"), kw)
        d = os.path.join(self.saida, "celulas", nome)
        self.monitor(d, [60.0])
        return d

    def monitor(self, d, temps, ini=EPOCH_INI, fim=EPOCH_FIM, passo=1):
        """monitor.csv sintético no formato do nvidia-smi: uma amostra a cada `passo` s de ini a fim (a coluna 5 cicla por `temps`)."""
        linhas = []
        for i, t_ in enumerate(range(ini, fim + 1, passo)):
            linhas.append("%s.000, 5, 2000, 10000, %s, 100" % (time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(t_)), temps[i % len(temps)]))
        grava(os.path.join(d, "monitor.csv"), "\n".join(linhas) + "\n")

    def lastro(self, m="1b", t="curto", estado="frio", n=3, prefill=100.0, passo=2.0, passos=None, gerados=None, ids=None, cpu=90, needed=100, portao=True,
               prefill_k=None, rc=0):
        nome = "lastro-%s-L-%s-%s" % (m, t, estado)
        d = self.celula_meta(nome, lado="lastro", serie="L", modelo=m, texto=t, estado=estado, n_medidas=n, portao_ok=portao, portao_motivo="" if portao else "CPU ocupada")
        def entrada(i, rot, desc):
            ps = (passos[i] if passos else [passo] * 63)
            mp = sum(ps) / len(ps)
            pre = prefill_k[i] if prefill_k else prefill
            return {"rotulo": rot, "descartada": desc, "ids": ids if ids is not None else IDS[t], "gerados": gerados if gerados is not None else GERADOS, "passo_ms": ps,
                    "prefill_ms": pre, "total_ms": pre + sum(ps) + 500.0, "T_sequencia_ms": 500.0, "media_por_passo_ms": mp,
                    "cont": {"needed": needed, "hits": 5, "cpu": cpu}, "fracao_cpu_needed": cpu / needed, "falha": None}
        if estado == "frio":
            procs = [(k, [entrada(k - 1 if k else 0, "frio", False)]) for k in range(0, n + 1)]
        else:
            procs = [(1, [entrada(0, "aquecimento", True)] + [entrada(i, "quente rep %d" % (i + 1), False) for i in range(n)])]
        for k, ents in procs:
            pd = os.path.join(d, "p%d" % k)
            if estado == "frio" and k >= 1:
                ents = [entrada(k - 1, "frio", False)]
            grava(os.path.join(pd, "rc.txt"), "%d\n" % rc)
            grava(os.path.join(pd, "wall_ms.txt"), "%d\n" % (sum(e["total_ms"] for e in ents) + 3000))
            grava(os.path.join(pd, "saida", "medidas-%s.json" % m), {"tipo": "medidas-gera", "modelo": m, "texto": t, "medidas": ents})
            grava(os.path.join(pd, "saida", "gera-final.txt"), "tipo=gera-final\nintegro=1\n")
            grava(os.path.join(pd, "saida", "gera-%s.json" % m), {"falha": None, "integro": True})
        return d

    def llama(self, m="1b", t="curto", estado="frio", serie="S1", n=3, prompt_n=None, prefill=50.0, per_token=1.8, predicted_n=64, tokens=None, quente_cmd=None,
              warm_log=False, com_per_token=True, portao=True, ms_total=None):
        nome = "llamacpp-%s-%s-%s-%s" % (m, serie, t, estado)
        d = self.celula_meta(nome, lado="llamacpp", serie=serie, modelo=m, texto=t, estado=estado, n_medidas=n, portao_ok=portao, portao_motivo="" if portao else "GPU ocupada")
        cmd = "llama-server -m x.gguf -t 8 --host 127.0.0.1 --no-webui -ngl 99 -fa on --port <PORTA> " + ("--no-warmup" if estado == "frio" else "")
        grava(os.path.join(d, "cmdline.txt"), quente_cmd or cmd)
        grava(os.path.join(d, "version.txt"), "version: 1 (0893ec9)")
        servs = range(0, n + 1) if estado == "frio" else [0]
        for k in servs:
            grava(os.path.join(d, "servidor-%d.log" % k), ("warming up the model with an empty run\n" if warm_log else "") + "model loaded\n")
            grava(os.path.join(d, "carga-%d.ms" % k), "1234\n")
        for k in range(0, n + 1):
            pm = ms_total if ms_total is not None else per_token * 63
            t_ = {"prompt_n": len(IDS[t]) if prompt_n is None else prompt_n, "prompt_ms": prefill, "predicted_n": predicted_n, "predicted_ms": pm}
            if com_per_token:
                t_["predicted_per_token_ms"] = pm / 63
            grava(os.path.join(d, "resp-%d.json" % k), {"content": "txt " + t, "tokens": tokens if tokens is not None else GERADOS, "timings": t_})
            grava(os.path.join(d, "resp-%d.rc" % k), "0 200\n")
        return d

    def roda(self):
        return X.processa(self.saida, self.ids, self.refs, self.textos)

    def razao(self, res, m="1b", t="curto", estado="frio", serie="S1"):
        r = [x for x in res["razoes"] if (x["modelo"], x["texto"], x["estado"], x["serie"]) == (m, t, estado, serie)]
        self.assertEqual(len(r), 1)
        return r[0]


class TestEstatistica(unittest.TestCase):
    def test_ruidosa_com_2_3_e_5_medidas(self):
        # 2 medidas: mediana = média das duas; (max-min)/mediana = 20/110 = 18% > 15%
        e2 = X.estat([100.0, 120.0])
        self.assertEqual(e2["mediana"], 110.0)
        self.assertTrue(e2["ruidosa"])
        self.assertFalse(X.estat([100.0, 110.0])["ruidosa"])        # 10/105 = 9,5%
        # 3 medidas dispersas: (max-min)/mediana = 40/100 = 40%
        self.assertTrue(X.estat([80.0, 100.0, 120.0])["ruidosa"])
        self.assertFalse(X.estat([98.0, 100.0, 104.0])["ruidosa"])
        # 5 medidas: uma só discrepante já basta pela regra da emenda 4 (max-min sobre a mediana)
        self.assertTrue(X.estat([100.0, 101.0, 100.0, 99.0, 130.0])["ruidosa"])
        self.assertFalse(X.estat([100.0, 101.0, 100.0, 99.0, 105.0])["ruidosa"])
        # 1 medida nunca é ruidosa
        self.assertFalse(X.estat([100.0])["ruidosa"])

    def test_decode_lastro_e_a_media_por_passo_nao_a_diferenca_de_totais(self):
        # Linha real da 5c (1B curto, frio): (7337,0 - 3262,7)/63 = 64,7 ms estava ERRADO; a média por passo é ~21,9 ms.
        arq = os.path.join(AQUI, "gera-5c-janela-20261108-1914", "gera-1b.json")
        if not os.path.exists(arq):
            self.skipTest("gera-1b.json da 5c ausente")
        g = json.load(open(arq, encoding="utf-8"))
        p = g["pedidos"][0]["geracao"]
        mp = X.media_por_passo(p["passo_ms"])
        self.assertEqual(len(p["passo_ms"]), 63)
        self.assertAlmostEqual(mp, 21.9, delta=0.15)
        errado = (p["total_ms"] - p["prefill_ms"]) / 63
        self.assertAlmostEqual(errado, 64.7, delta=0.1)
        self.assertNotAlmostEqual(mp, errado, delta=10)

    def test_decode_llama_predicted_ms_sobre_63(self):
        # sem predicted_per_token_ms: predicted_ms/(predicted_n-1) = /63, nunca /64
        v, fonte = X.passo_llama({"predicted_n": 64, "predicted_ms": 126.0})
        self.assertAlmostEqual(v, 2.0)
        self.assertNotAlmostEqual(v, 126.0 / 64, places=3)
        self.assertIn("predicted_n-1", fonte)
        # com o campo, é ele que vale
        v2, f2 = X.passo_llama({"predicted_n": 64, "predicted_ms": 126.0, "predicted_per_token_ms": 2.0})
        self.assertEqual((v2, f2), (2.0, "predicted_per_token_ms"))


class TestCelulas(Base):
    def tudo_ok(self, n_l=3, n_c=3):
        for est in ("frio", "quente"):
            n = n_l if est == "frio" else 5
            self.lastro(estado=est, n=n)
            for s in ("S1", "S2"):
                self.llama(estado=est, serie=s, n=n_c if est == "frio" else 5)

    def test_celula_ok_publica_razao_rotulada_e_fracao_cpu(self):
        self.lastro(estado="frio", n=3, prefill=100.0, passo=21.9, cpu=894, needed=1000)
        self.llama(estado="frio", serie="S1", n=3, prefill=50.0, per_token=2.19 * 0 + 1.8)
        r = self.razao(self.roda())
        self.assertEqual(r["status"], "OK")
        self.assertTrue(r["rotulo"].startswith("Lastro 10% experts GPU contra llama.cpp S1"))
        self.assertAlmostEqual(r["prefill"]["razao_lastro_sobre_llama_cpp"], 2.0)
        self.assertAlmostEqual(r["decode"]["razao_lastro_sobre_llama_cpp"], 21.9 / 1.8, places=6)
        self.assertAlmostEqual(r["fracao_cpu_needed"]["mediana"], 0.894)
        self.assertEqual(r["marca_sequencia"], "mesma sequência")
        self.assertEqual(r["prefill"]["lastro"]["n"], 3)

    def test_lastro_usa_media_por_passo_e_nao_total_ms(self):
        # total_ms enorme (criação do motor) não pode entrar no decode
        self.lastro(estado="frio", n=2, passo=20.0)
        self.llama(estado="frio", serie="S1", n=2, per_token=2.0)
        r = self.razao(self.roda())
        self.assertAlmostEqual(r["decode"]["lastro"]["mediana"], 20.0)
        self.assertAlmostEqual(r["decode"]["llama_cpp"]["mediana"], 2.0)

    def test_s1_e_s2_nao_se_misturam(self):
        self.lastro(estado="quente", n=5, passo=20.0, prefill=100.0)
        self.llama(estado="quente", serie="S1", n=5, per_token=2.0, prefill=10.0)
        self.llama(estado="quente", serie="S2", n=5, per_token=10.0, prefill=40.0)
        res = self.roda()
        r1, r2 = self.razao(res, estado="quente", serie="S1"), self.razao(res, estado="quente", serie="S2")
        self.assertAlmostEqual(r1["decode"]["razao_lastro_sobre_llama_cpp"], 10.0)
        self.assertAlmostEqual(r2["decode"]["razao_lastro_sobre_llama_cpp"], 2.0)
        self.assertAlmostEqual(r1["prefill"]["razao_lastro_sobre_llama_cpp"], 10.0)
        self.assertAlmostEqual(r2["prefill"]["razao_lastro_sobre_llama_cpp"], 2.5)
        self.assertIn("S1", r1["rotulo"])
        self.assertIn("S2", r2["rotulo"])
        self.assertNotEqual(r1["rotulo"], r2["rotulo"])
        self.assertTrue(all("Lastro 10% experts GPU contra llama.cpp" in x["rotulo"] for x in res["razoes"]))
        md = X.relatorio(res)
        self.assertIn("## Lastro 10% experts GPU contra llama.cpp S1", md)
        self.assertIn("## Lastro 10% experts GPU contra llama.cpp S2", md)
        self.assertIn("op offload", md)

    def test_ruidosa_publicada_com_marca(self):
        self.lastro(estado="frio", n=3, prefill_k=[100.0, 100.0, 100.0])
        self.llama(estado="frio", serie="S1", n=3)
        # llama.cpp com 3 medidas dispersas no prefill: reescreve as respostas 1..3
        d = os.path.join(self.saida, "celulas", "llamacpp-1b-S1-curto-frio")
        for k, v in ((1, 40.0), (2, 50.0), (3, 70.0)):
            r = json.load(open(os.path.join(d, "resp-%d.json" % k)))
            r["timings"]["prompt_ms"] = v
            grava(os.path.join(d, "resp-%d.json" % k), r)
        r = self.razao(self.roda())
        self.assertEqual(r["status"], "OK")
        self.assertTrue(r["prefill"]["ruidosa"])
        self.assertTrue(r["prefill"]["llama_cpp"]["ruidosa"])
        self.assertFalse(r["decode"]["ruidosa"])
        self.assertIn("ruidosa", X.relatorio(self.roda()))

    def test_sem_prova_por_contagem_de_prompt_divergente(self):
        self.lastro(estado="frio", n=3)
        self.llama(estado="frio", serie="S1", n=3, prompt_n=99)
        res = self.roda()
        r = self.razao(res)
        self.assertEqual(r["status"], "SEM PROVA")
        self.assertNotIn("prefill", r)
        self.assertTrue(any("prompt_n 99" in x for x in r["motivos"]))
        self.assertEqual(res["celulas"]["llamacpp-1b-S1-curto-frio"]["status"], "SEM PROVA")

    def test_sem_prova_por_contagem_de_gerados_e_sem_timings(self):
        self.lastro(estado="frio", n=3)
        self.llama(estado="frio", serie="S1", n=3, predicted_n=40, tokens=GERADOS[:40])
        self.assertEqual(self.razao(self.roda())["status"], "SEM PROVA")
        self.llama(estado="frio", serie="S1", n=3)           # reescreve ok
        d = os.path.join(self.saida, "celulas", "llamacpp-1b-S1-curto-frio")
        grava(os.path.join(d, "resp-2.json"), {"content": "x", "tokens": GERADOS})
        r = self.razao(self.roda())
        self.assertEqual(r["status"], "SEM PROVA")
        self.assertTrue(any("timings" in x for x in r["motivos"]))

    def test_sem_prova_lastro_ids_de_prompt_diferentes_e_passos_errados(self):
        self.lastro(estado="frio", n=2, ids=[1, 2, 3])
        self.llama(estado="frio", serie="S1", n=2)
        self.assertEqual(self.razao(self.roda())["status"], "SEM PROVA")
        self.lastro(estado="frio", n=2, passos=[[2.0] * 62] * 3)
        self.assertEqual(self.razao(self.roda())["status"], "SEM PROVA")

    def test_sequencias_diferentes_marcadas_e_razoes_publicadas(self):
        dif = list(GERADOS)
        dif[10] = 999
        self.lastro(estado="frio", n=3, gerados=dif)
        self.llama(estado="frio", serie="S1", n=3)
        res = self.roda()
        r = self.razao(res)
        self.assertEqual(r["status"], "OK")
        self.assertEqual(r["marca_sequencia"], "sequências diferentes")
        self.assertEqual(r["decode"]["marca"], "sequências diferentes")
        self.assertIsNotNone(r["decode"]["razao_lastro_sobre_llama_cpp"])      # razões continuam publicadas, com a marca
        dv = [x for x in r["sequencia"]["divergencias"] if x["par"] == "Lastro x referência"][0]
        self.assertEqual(dv["passo"], 10)
        self.assertEqual(dv["gap_referencia"], 5.0)
        self.assertFalse(dv["empate"])
        self.assertIn("sequências diferentes", X.relatorio(res))

    def test_divergencia_em_empate_e_reportada_como_empate(self):
        for m in ("1b",):
            grava(os.path.join(self.refs, "ref5c-%s.json" % m),
                  {"casos": [{"nome": t, "ids": IDS[t], "gerados": GERADOS, "texto_gerado": "txt " + t, "gaps": [5.0] * 10 + [0.0005] + [5.0] * 53} for t in IDS]})
        dif = list(GERADOS)
        dif[10] = 999
        self.lastro(estado="frio", n=2, gerados=dif)
        self.llama(estado="frio", serie="S1", n=2)
        r = self.razao(self.roda())
        dv = [x for x in r["sequencia"]["divergencias"] if x["par"] == "Lastro x referência"][0]
        self.assertTrue(dv["empate"])

    def test_nao_rodou_e_repeticoes_diferentes(self):
        self.lastro(estado="frio", n=3)
        self.llama(estado="frio", serie="S1", n=2)
        res = self.roda()
        self.assertEqual(self.razao(res, serie="S1")["status"], "SEM PROVA")
        self.assertTrue(any("repetições diferentes" in x for x in self.razao(res, serie="S1")["motivos"]))
        self.assertEqual(self.razao(res, serie="S2")["status"], "NÃO RODOU")
        self.assertEqual(self.razao(res, serie="S3")["status"], "NÃO RODOU")
        self.assertEqual(res["celulas"]["llamacpp-3b-S1-longo-quente"]["status"], "NÃO RODOU")
        md = X.relatorio(res)
        self.assertIn("NÃO RODOU", md)
        for r in res["razoes"]:
            if r["status"] == "NÃO RODOU":
                self.assertNotIn("prefill", r)

    def test_portao_falho_e_frio_com_warmup_sao_sem_prova(self):
        self.lastro(estado="frio", n=3, portao=False)
        self.llama(estado="frio", serie="S1", n=3)
        r = self.razao(self.roda())
        self.assertEqual(r["status"], "SEM PROVA")
        self.assertTrue(any("portão" in x for x in r["motivos"]))
        # emenda 6 (2): a linha 'warming up' no log do frio é só INFORMATIVA (LOG_TRC, não sai na verbosidade padrão): não é prova nem falha
        self.lastro(estado="frio", n=3)
        self.llama(estado="frio", serie="S1", n=3, warm_log=True)
        res = self.roda()
        r = self.razao(res)
        self.assertEqual(r["status"], "OK")
        self.assertTrue(any("warming up" in x for x in res["celulas"]["llamacpp-1b-S1-curto-frio"]["observacoes"]))
        self.assertIn("Observações informativas", X.relatorio(res))
        # sem a linha no log (o caso normal) o frio continua OK: a ausência não prova nada, a prova é a linha de comando
        self.llama(estado="frio", serie="S1", n=3)
        self.assertEqual(self.razao(self.roda())["status"], "OK")
        self.llama(estado="frio", serie="S1", n=3, quente_cmd="llama-server --host 127.0.0.1 -ngl 99")
        self.assertTrue(any("--no-warmup" in x for x in self.razao(self.roda())["motivos"]))
        self.llama(estado="quente", serie="S1", n=5, quente_cmd="llama-server --host 127.0.0.1 --no-warmup")
        self.lastro(estado="quente", n=5)
        self.assertEqual(self.razao(self.roda(), estado="quente")["status"], "SEM PROVA")

    def test_temperatura_acima_de_80_sem_prova_e_integridade_quebrada(self):
        self.lastro(estado="frio", n=3)
        d = self.llama(estado="frio", serie="S1", n=3)
        self.monitor(d, [81.0, 70.0])
        r = self.razao(self.roda())
        self.assertEqual(r["status"], "SEM PROVA")
        self.assertTrue(any("temperatura" in x for x in r["motivos"]))
        self.monitor(d, [79.0])        # 79 C passa
        self.assertEqual(self.razao(self.roda())["status"], "OK")
        self.integra(depois="binario xxx\ngguf def\n")
        res = self.roda()
        self.assertEqual(self.razao(res)["status"], "SEM PROVA")
        self.assertTrue(res["integridade_problemas"])
        self.assertIn("Integridade QUEBRADA", X.relatorio(res))

    def test_medidas_inconsistentes_com_gera_final(self):
        d = self.lastro(estado="frio", n=2)
        grava(os.path.join(d, "p1", "saida", "gera-final.txt"), "tipo=gera-final\nintegro=0\n")
        self.llama(estado="frio", serie="S1", n=2)
        r = self.razao(self.roda())
        self.assertEqual(r["status"], "SEM PROVA")
        self.assertTrue(any("gera-final" in x for x in r["motivos"]))

    def test_processo_com_erro_e_media_guardada_errada(self):
        d = self.lastro(estado="frio", n=2)
        grava(os.path.join(d, "p2", "rc.txt"), "1\n")
        self.llama(estado="frio", serie="S1", n=2)
        self.assertEqual(self.razao(self.roda())["status"], "SEM PROVA")
        d = self.lastro(estado="frio", n=2)
        mj = json.load(open(os.path.join(d, "p1", "saida", "medidas-1b.json")))
        mj["medidas"][0]["media_por_passo_ms"] = 64.7
        grava(os.path.join(d, "p1", "saida", "medidas-1b.json"), mj)
        r = self.razao(self.roda())
        self.assertEqual(r["status"], "SEM PROVA")
        self.assertTrue(any("media_por_passo_ms" in x for x in r["motivos"]))

    def test_razoes_diferem_mais_de_1_5x_publica_fatias_e_candidata(self):
        # prefill 1x e decode 10x -> diferem; T_sequencia 500 de total ~ 100+126+500 = 726 -> 69% > 30% e sem equivalente
        self.lastro(estado="frio", n=3, prefill=100.0, passo=2.0)
        self.llama(estado="frio", serie="S1", n=3, prefill=100.0, per_token=0.2)
        r = self.razao(self.roda())
        self.assertTrue(r["razoes_diferem_mais_de_1_5x"])
        g = r["gargalo"]
        self.assertTrue(g["hipotese_nao_conclusao"])
        self.assertTrue(g["fatias_do_lastro"]["T_sequencia_ms"]["candidata_a_gargalo"])
        self.assertFalse(g["fatias_do_lastro"]["prefill_ms"]["candidata_a_gargalo"])
        self.assertIsNotNone(g["fracao_cpu_needed"])
        # razões parecidas: sem bloco de gargalo
        self.llama(estado="frio", serie="S1", n=3, prefill=100.0, per_token=2.0)
        self.assertFalse(self.razao(self.roda())["razoes_diferem_mais_de_1_5x"])
        self.assertNotIn("gargalo", self.razao(self.roda()))

    def test_carga_lastro_e_limite_superior_e_quente_com_aquecimento(self):
        self.lastro(estado="frio", n=2)
        self.llama(estado="frio", serie="S1", n=2)
        c = self.roda()["celulas"]["lastro-1b-L-curto-frio"]
        # wall = total + 3000 ; carga_limite = wall - total + T_sequencia = 3000 + 500
        self.assertAlmostEqual(c["carga_lastro_limite_superior_ms"]["mediana"], 3500.0)
        md = X.relatorio(self.roda())
        self.assertIn("LIMITE SUPERIOR", md)

    def test_arquivos_de_saida_gravados(self):
        self.lastro(estado="frio", n=2)
        self.llama(estado="frio", serie="S1", n=2)
        self.assertEqual(X.main(["--saida", self.saida, "--ids", self.ids, "--refs", self.refs, "--textos", self.textos]), 0)
        r = json.load(open(os.path.join(self.saida, "cmp-resultado.json"), encoding="utf-8"))
        self.assertTrue(r["descritivo"])
        self.assertIn("Lastro 10% experts GPU", open(os.path.join(self.saida, "cmp-resultado.md"), encoding="utf-8").read())
        self.assertEqual(X.main(["--saida", os.path.join(self.d, "nao-existe")]), 2)


class TestMonitorFechado(Base):
    """r111, rev-1 P2: sem prova de temperatura (monitor ausente, vazio, ilegível, coleta que não cobre a célula) a célula é SEM PROVA e a razão não sai."""

    def par(self):
        self.lastro(estado="frio", n=2)
        return self.llama(estado="frio", serie="S1", n=2)

    def sem_prova(self, trecho):
        r = self.razao(self.roda())
        self.assertEqual(r["status"], "SEM PROVA", r)
        self.assertNotIn("prefill", r)
        self.assertNotIn("decode", r)
        self.assertTrue(any(trecho in x for x in r["motivos"]), r["motivos"])

    def test_baseline_ok_e_79_81(self):
        d = self.par()
        self.assertEqual(self.razao(self.roda())["status"], "OK")
        self.monitor(d, [79.0])
        self.assertEqual(self.razao(self.roda())["status"], "OK")
        self.monitor(d, [81.0])
        self.sem_prova("temperatura")

    def test_monitor_ausente(self):
        d = self.par()
        os.remove(os.path.join(d, "monitor.csv"))
        self.sem_prova("monitor.csv ausente")

    def test_monitor_vazio_e_so_linhas_ilegiveis(self):
        d = self.par()
        grava(os.path.join(d, "monitor.csv"), "")
        self.sem_prova("sem amostra de temperatura legível")
        grava(os.path.join(d, "monitor.csv"), "No devices were found\nlixo, x\nt, 1, 2, 3, abc, 5\nt, 1, 2, 3, nan, 5\n")
        self.sem_prova("sem amostra de temperatura legível")

    def test_coleta_morta_depois_de_uma_amostra_inicial(self):
        d = self.par()
        self.monitor(d, [60.0], fim=EPOCH_INI)          # só a amostra do 1º segundo; a célula durou 10 s
        self.sem_prova("a coleta do monitor parou")

    def test_lacuna_na_coleta(self):
        d = self.par()
        self.monitor(d, [60.0], passo=20)                # amostras em 0, 20 e... só 0 (fim 10): reforça com intervalo longo
        grava(os.path.join(d, "celula.json"), dict(json.load(open(os.path.join(d, "celula.json"))), fim_epoch=EPOCH_INI + 40))
        self.monitor(d, [60.0], ini=EPOCH_INI, fim=EPOCH_INI + 40, passo=20)
        self.sem_prova("lacuna")

    def test_falha_de_coleta_registrada_no_metadata_e_sem_epochs(self):
        d = self.par()
        meta = json.load(open(os.path.join(d, "celula.json")))
        grava(os.path.join(d, "celula.json"), dict(meta, monitor_falha="inicia rc=3"))
        self.sem_prova("falha do monitor registrada")
        m2 = dict(meta)
        del m2["fim_epoch"]
        grava(os.path.join(d, "celula.json"), m2)
        self.sem_prova("inicio_epoch/fim_epoch")

    def test_carimbo_ilegivel_nao_verifica_cobertura(self):
        d = self.par()
        grava(os.path.join(d, "monitor.csv"), "t, 5, 2000, 10000, 60, 100\n")
        self.sem_prova("carimbos de tempo")

    def test_lastro_tambem(self):
        d = self.lastro(estado="frio", n=2)
        self.llama(estado="frio", serie="S1", n=2)
        os.remove(os.path.join(d, "monitor.csv"))
        self.sem_prova("monitor.csv ausente")


class TestNumerosFechados(Base):
    """r111, rev-1 P2: tempos negativos ou não finitos nunca viram razão OK; o JSON final é estrito."""

    def par(self, **kw):
        self.lastro(estado="frio", n=2, **kw)
        self.llama(estado="frio", serie="S1", n=2)

    def sem_prova(self, trecho=None):
        res = self.roda()
        r = self.razao(res)
        self.assertEqual(r["status"], "SEM PROVA", r)
        self.assertNotIn("prefill", r)
        self.assertNotIn("decode", r)
        if trecho:
            self.assertTrue(any(trecho in x for x in r["motivos"]), r["motivos"])
        self.estrito()
        return res

    def estrito(self):
        def rej(c):
            raise AssertionError("constante não finita no JSON final: " + c)
        self.assertEqual(X.main(["--saida", self.saida, "--ids", self.ids, "--refs", self.refs, "--textos", self.textos]), 0)
        with open(os.path.join(self.saida, "cmp-resultado.json"), encoding="utf-8") as f:
            texto = f.read()
        json.loads(texto, parse_constant=rej)
        for lixo in ("NaN", "Infinity"):
            self.assertNotIn(lixo, texto)

    def reescreve(self, arq, de, para):
        with open(arq, encoding="utf-8") as f:
            t = f.read()
        self.assertIn(de, t)
        with open(arq, "w", encoding="utf-8") as f:
            f.write(t.replace(de, para, 1))

    def medidas(self, p="p1"):
        return os.path.join(self.saida, "celulas", "lastro-1b-L-curto-frio", p, "saida", "medidas-1b.json")

    def resp(self, k=1):
        return os.path.join(self.saida, "celulas", "llamacpp-1b-S1-curto-frio", "resp-%d.json" % k)

    def test_baseline_ok(self):
        self.par()
        self.assertEqual(self.razao(self.roda())["status"], "OK")
        self.estrito()

    def test_passo_negativo_lastro(self):
        self.par(passos=[[2.0] * 62 + [-1.0]] * 3)
        self.sem_prova("passo_ms fora do domínio")

    def test_passo_zero_lastro(self):
        self.par(passos=[[0.0] * 63] * 3)
        self.sem_prova("passo_ms fora do domínio")

    def test_prefill_negativo_lastro(self):
        self.par(prefill=-5.0)
        self.sem_prova("prefill_ms fora do domínio")

    def test_passo_negativo_e_prefill_negativo_llama(self):
        self.par()
        r = json.load(open(self.resp()))
        r["timings"]["predicted_per_token_ms"] = -2.0
        grava(self.resp(), r)
        self.sem_prova("predicted_per_token_ms fora do domínio")
        r["timings"]["predicted_per_token_ms"] = 2.0
        r["timings"]["prompt_ms"] = -1.0
        grava(self.resp(), r)
        self.sem_prova("prompt_ms fora do domínio")

    def test_nan_e_infinity_nos_arquivos_json(self):
        self.par()
        for ruim in ("NaN", "Infinity", "-Infinity"):
            self.par()
            self.reescreve(self.medidas(), '"prefill_ms": 100.0', '"prefill_ms": %s' % ruim)
            self.sem_prova()
            self.par()
            self.reescreve(self.resp(), '"prompt_ms": 50.0', '"prompt_ms": %s' % ruim)
            self.sem_prova()

    def test_nan_e_infinity_na_media_guardada_e_passos(self):
        for ruim in ("NaN", "Infinity"):
            self.par()
            self.reescreve(self.medidas(), '"media_por_passo_ms": 2.0', '"media_por_passo_ms": %s' % ruim)
            self.sem_prova()
        self.par()
        self.reescreve(self.medidas("p1"), "[2.0, ", "[NaN, ")
        self.sem_prova()

    def test_expoente_que_estoura(self):
        self.par()
        self.reescreve(self.medidas(), '"prefill_ms": 100.0', '"prefill_ms": 1e999')
        self.sem_prova("prefill_ms")
        self.par()
        self.reescreve(self.medidas(), "[2.0, ", "[1e999, ")
        self.sem_prova("passo_ms")
        self.par()
        self.reescreve(self.resp(), '"predicted_per_token_ms": 1.8', '"predicted_per_token_ms": 1e999')
        self.sem_prova("predicted_per_token_ms")
        self.par()
        self.reescreve(self.resp(), '"prompt_ms": 50.0', '"prompt_ms": 1e999')
        self.sem_prova("prompt_ms")

    def test_soma_que_estoura_o_float(self):
        # cada passo é finito, mas a soma não: o extrator não pode devolver média infinita
        self.par()
        with open(self.medidas(), encoding="utf-8") as f:
            mj = json.load(f)
        mj["medidas"][0]["passo_ms"] = [1e308] * 63
        mj["medidas"][0]["media_por_passo_ms"] = 1e308
        grava(self.medidas(), mj)
        self.sem_prova("não finita")

    def test_total_e_wall_invalidos(self):
        self.par()
        self.reescreve(self.medidas(), '"total_ms": 726.0', '"total_ms": -1')
        self.sem_prova("total_ms fora do domínio")
        self.par()
        grava(os.path.join(self.saida, "celulas", "lastro-1b-L-curto-frio", "p1", "wall_ms.txt"), "nan\n")
        self.sem_prova("wall_ms.txt ilegível ou não finito")
        self.par()
        grava(os.path.join(self.saida, "celulas", "lastro-1b-L-curto-frio", "p1", "wall_ms.txt"), "-5\n")
        self.sem_prova("wall_ms.txt negativo")
        self.par()
        grava(os.path.join(self.saida, "celulas", "llamacpp-1b-S1-curto-frio", "carga-1.ms"), "inf\n")
        self.sem_prova("carga-1.ms ilegível ou não finito")

    def test_primitivos(self):
        for ruim in (float("nan"), float("inf"), True, "2", None, -1.0):
            with self.assertRaises(ValueError):
                X.num(ruim, "x")
        for ruim in (0, 0.0, -0.1):
            with self.assertRaises(ValueError):
                X.num(ruim, "x", True)
        with self.assertRaises(ValueError):
            X.num(10 ** 400, "x")
        self.assertEqual(X.num(0, "x"), 0.0)
        for t in ({"predicted_per_token_ms": float("nan")}, {"predicted_per_token_ms": -1}, {"predicted_n": 1, "predicted_ms": 5.0}, {"predicted_n": 64, "predicted_ms": float("inf")},
                  {"predicted_n": True, "predicted_ms": 5.0}, {"predicted_n": 64, "predicted_ms": 1e-320 * 0}):
            with self.assertRaises(ValueError):
                X.passo_llama(t)


class TestModelosDaJanela(Base):
    """Emenda 6: uma janela por modelo; o outro modelo não vira NÃO RODOU."""

    def test_so_o_modelo_declarado(self):
        self.lastro(estado="frio", n=2)
        self.llama(estado="frio", serie="S1", n=2)
        grava(os.path.join(self.saida, "orcamento.json"), {"modelos": ["1b"], "cortes": []})
        res = self.roda()
        self.assertTrue(all(r["modelo"] == "1b" for r in res["razoes"]))
        self.assertFalse(any(k.startswith(("lastro-3b", "llamacpp-3b")) for k in res["celulas"]))
        self.assertEqual(res["modelos_da_janela"], ["1b"])
        self.assertIn("Janela do(s) modelo(s): 1b", X.relatorio(res))
        grava(os.path.join(self.saida, "orcamento.json"), {"modelos": ["3b"], "cortes": []})
        res = self.roda()
        self.assertTrue(all(r["modelo"] == "3b" and r["status"] == "NÃO RODOU" for r in res["razoes"]))
        grava(os.path.join(self.saida, "orcamento.json"), {"cortes": []})
        self.assertEqual({r["modelo"] for r in self.roda()["razoes"]}, {"1b", "3b"})


if __name__ == "__main__":
    unittest.main(verbosity=1)
