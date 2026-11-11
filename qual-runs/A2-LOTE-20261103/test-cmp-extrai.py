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
        # série da 3080 Ti (Emenda 10): manifesto com a prova única de potência e ambiente coerente
        grava(os.path.join(self.saida, "integridade-fixos.txt"), "binario abc abc OK\npotencia 370.00_W 370.00_W OK\n")
        grava(os.path.join(self.saida, "ambiente.txt"), "uuid=GPU-b22d9465-8552-9f97-812e-380b590645f8\npotencia_inicio=370.00_W\npotencia_fim=370.00_W\n")

    # ---- construtores de célula
    def celula_meta(self, nome, **kw):
        kw.setdefault("inicio_epoch", EPOCH_INI)
        kw.setdefault("fim_epoch", EPOCH_FIM)
        kw.setdefault("inicio_coleta_epoch", EPOCH_INI)
        grava(os.path.join(self.saida, "celulas", nome, "celula.json"), kw)
        d = os.path.join(self.saida, "celulas", nome)
        self.monitor(d, [60.0])
        # emenda 9: portao.txt é obrigatório; a fixture válida traz a evidência explícita do portão (testes específicos sobrescrevem ou removem)
        grava(os.path.join(d, "portao.txt"), "pmon_antes_veredito=OK\ncompute_apps_veredito=OK\nportao=OK\n")
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


class TestPotencia(Base):
    """Emenda 10: a série da 3080 Ti exige a prova positiva única `potencia 370.00_W 370.00_W OK`; ausência nunca vira contrato antigo."""
    def prepara(self):
        self.lastro()
        self.llama()

    def fixos(self, txt):
        grava(os.path.join(self.saida, "integridade-fixos.txt"), txt)

    def amb(self, txt):
        grava(os.path.join(self.saida, "ambiente.txt"), txt)

    def sem_prova(self, trecho):
        res = self.roda()
        self.assertTrue(any(trecho in m for m in res["integridade_problemas"]), res["integridade_problemas"])
        r = self.razao(res)
        self.assertNotEqual(r.get("status"), "OK")
        self.assertIsNone(r.get("decode"))
        self.assertIsNone(r.get("prefill"))

    def test_prova_valida(self):
        self.prepara()
        res = self.roda()
        self.assertEqual(res["integridade_problemas"], [])
        self.assertEqual(self.razao(res)["status"], "OK")

    def test_sem_linha_de_potencia(self):
        self.prepara()
        self.fixos("binario abc abc OK\n")
        self.sem_prova("'potencia' ausente")

    def test_linha_duplicada(self):
        self.prepara()
        self.fixos("potencia 370.00_W 370.00_W OK\npotencia 370.00_W 370.00_W OK\n")
        self.sem_prova("duplicada")

    def test_linha_truncada(self):
        self.prepara()
        self.fixos("binario abc abc OK\npotencia 370.00_W\n")
        self.sem_prova("não é a prova")

    def test_extremo_diferente(self):
        self.prepara()
        self.fixos("potencia 370.00_W 300.00_W OK\n")
        self.sem_prova("não é a prova")

    def test_manifesto_ausente(self):
        self.prepara()
        os.remove(os.path.join(self.saida, "integridade-fixos.txt"))
        self.sem_prova("integridade-fixos.txt ausente")

    def test_ambiente_divergente(self):
        self.prepara()
        self.amb("uuid=GPU-b22d9465-8552-9f97-812e-380b590645f8\npotencia_inicio=370.00_W\npotencia_fim=300.00_W\n")
        self.sem_prova("incoerente")

    def test_ambiente_sem_potencia(self):
        self.prepara()
        self.amb("uuid=GPU-b22d9465-8552-9f97-812e-380b590645f8\n")
        self.sem_prova("incoerente")

    def test_historico_4090_so_por_uuid(self):
        self.prepara()
        self.fixos("binario abc abc OK\n")
        self.amb("uuid=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd\n")
        res = self.roda()
        self.assertEqual(res["integridade_problemas"], [])
        self.amb("loadavg=1 1 1\n")        # sem uuid: a ausência da linha NÃO seleciona o contrato antigo
        self.sem_prova("'potencia' ausente")


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

    def meta_com(self, d, **kw):
        meta = json.load(open(os.path.join(d, "celula.json")))
        meta.update(kw)
        grava(os.path.join(d, "celula.json"), meta)

    def test_unica_amostra_perto_do_fim_de_celula_longa(self):
        # r112 (rev-1 P2): célula de 100 s, única amostra em 99 s
        d = self.par()
        self.meta_com(d, fim_epoch=EPOCH_INI + 100)
        self.monitor(d, [60.0], ini=EPOCH_INI + 99, fim=EPOCH_INI + 99)
        self.sem_prova("só começou")

    def test_inicio_tardio_seguido_de_amostras_regulares(self):
        d = self.par()
        self.meta_com(d, fim_epoch=EPOCH_INI + 100)
        self.monitor(d, [60.0], ini=EPOCH_INI + 40, fim=EPOCH_INI + 100)
        self.sem_prova("só começou")

    def test_portao_longo_antes_da_coleta_continua_valido(self):
        # inicio_epoch 20 s antes do marco de coleta; a coleta cobre tudo a partir do marco
        d = self.par()
        self.meta_com(d, inicio_epoch=EPOCH_INI - 20, inicio_coleta_epoch=EPOCH_INI, fim_epoch=EPOCH_INI + 100)
        self.monitor(d, [60.0], ini=EPOCH_INI + 1, fim=EPOCH_INI + 100)
        self.assertEqual(self.razao(self.roda())["status"], "OK")

    def test_primeira_amostra_no_limite_de_5_s(self):
        d = self.par()
        self.meta_com(d, fim_epoch=EPOCH_INI + 100)
        self.monitor(d, [60.0], ini=EPOCH_INI + 5, fim=EPOCH_INI + 100)
        self.assertEqual(self.razao(self.roda())["status"], "OK")

    def test_sem_marco_de_inicio_da_coleta(self):
        d = self.par()
        meta = json.load(open(os.path.join(d, "celula.json")))
        del meta["inicio_coleta_epoch"]
        grava(os.path.join(d, "celula.json"), meta)
        self.sem_prova("inicio_coleta_epoch")

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


class TestPortaoPmon(Base):
    """Emendas 8 e 9: portao.txt OBRIGATÓRIO com `pmon_antes_veredito=OK` e `compute_apps_veredito=OK`; seção 'Condição do desktop' no relatório."""

    def escreve_portao(self, nome, linhas):
        with open(os.path.join(self.saida, "celulas", nome, "portao.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(linhas) + "\n")

    def celulas(self):
        return os.listdir(os.path.join(self.saida, "celulas"))

    def test_veredito_ok_vale_e_aparece_no_relatorio(self):
        self.lastro(estado="frio", n=3)
        self.llama(estado="frio", serie="S1", n=3)
        r = self.razao(self.roda())
        self.assertEqual(r["status"], "OK")
        md = X.relatorio(self.roda())
        self.assertIn("## Condição do desktop (emendas 8 e 9)", md)
        self.assertIn("COMUM dos dois lados", md)
        self.assertIn("o efeito pode não ser igual entre eles", md)
        self.assertIn("pico do pmon (soma G)", md)
        self.assertIn("PICO DE AMOSTRA ESPARSA, não a fração da placa", md)
        self.assertIn("dmon", md)

    def test_veredito_de_falha_ou_ausente_e_sem_prova(self):
        self.lastro(estado="frio", n=3)
        self.llama(estado="frio", serie="S1", n=3)
        for k in self.celulas():
            self.escreve_portao(k, ["pmon_antes_veredito=processo(s) de compute (tipo C ou C+G) fora da árvore medida no pmon: 1352017 (voxtype)", "compute_apps_veredito=OK", "portao=OK"])
        r = self.razao(self.roda())
        self.assertEqual(r["status"], "SEM PROVA")
        self.assertTrue(any("pmon" in x for x in r["motivos"]))
        for k in self.celulas():
            self.escreve_portao(k, ["gpu_util_pct=3", "portao=OK"])  # formato antigo, sem amostra pmon: falha fechada
        r = self.razao(self.roda())
        self.assertEqual(r["status"], "SEM PROVA")
        self.assertTrue(any("sem a amostra pmon" in x for x in r["motivos"]))

    def test_compute_apps_com_falha_ou_sem_linha_e_sem_prova(self):
        self.lastro(estado="frio", n=3)
        self.llama(estado="frio", serie="S1", n=3)
        for k in self.celulas():
            self.escreve_portao(k, ["pmon_antes_veredito=OK", "compute_apps_veredito=processo(s) de compute fora da árvore: 1352017 (voxtype)", "portao=OK"])
        r = self.razao(self.roda())
        self.assertEqual(r["status"], "SEM PROVA")
        self.assertTrue(any("compute-apps" in x for x in r["motivos"]))
        for k in self.celulas():
            self.escreve_portao(k, ["pmon_antes_veredito=OK", "portao=OK"])
        r = self.razao(self.roda())
        self.assertEqual(r["status"], "SEM PROVA")
        self.assertTrue(any("sem o veredito de compute-apps" in x for x in r["motivos"]))

    def test_portao_txt_ausente_ou_ilegivel_e_sem_prova_e_sem_razao_publicada(self):
        """rev-1 (r114): apagar o portao.txt, com o metadata intacto (portao_ok true), NÃO pode publicar razão; vale para os dois lados."""
        self.lastro(estado="frio", n=3)
        self.llama(estado="frio", serie="S1", n=3)
        self.assertEqual(self.razao(self.roda())["status"], "OK")
        for k in self.celulas():
            os.remove(os.path.join(self.saida, "celulas", k, "portao.txt"))
        res = self.roda()
        r = self.razao(res)
        self.assertEqual(r["status"], "SEM PROVA")
        self.assertTrue(any("portao.txt ausente" in x for x in r["motivos"]))
        self.assertNotIn("razao", {k for k, v in r.items() if v})  # sem razão numérica publicada
        for c in res["celulas"].values():
            if c["status"] != "NÃO RODOU":
                self.assertEqual(c["status"], "SEM PROVA")
        for k in self.celulas():  # portao.txt ilegível: uma pasta no lugar do arquivo
            os.makedirs(os.path.join(self.saida, "celulas", k, "portao.txt"))
        self.assertEqual(self.razao(self.roda())["status"], "SEM PROVA")

    def test_desktop_sm_do_celula_json_vai_para_o_resultado(self):
        self.lastro(estado="frio", n=3)
        for k in os.listdir(os.path.join(self.saida, "celulas")):
            arq = os.path.join(self.saida, "celulas", k, "celula.json")
            with open(arq, encoding="utf-8") as fh:
                m = json.load(fh)
            m["desktop_g_sm_antes"], m["desktop_g_sm_depois"], m["dmon_sm_antes"] = 25, None, 4
            grava(arq, m)
        res = self.roda()
        c = [c for c in res["celulas"].values() if c["lado"] == "lastro" and c["status"] != "NÃO RODOU"][0]
        self.assertEqual((c["desktop_g_sm_antes"], c["desktop_g_sm_depois"], c["dmon_sm_antes"]), (25, None, 4))


class BaseD1(Base):
    """Degrau 1: L100 contra S1 (criterio-degrau1-100pct-20261109.md)."""

    def smoke_ok(self, ids_iguais=True, **kw):
        sd = os.path.join(self.saida, "smoke")
        os.makedirs(sd, exist_ok=True)
        grava(os.path.join(sd, "rc.txt"), "0\n")
        grava(os.path.join(sd, "p1", "rc.txt"), "0\n")
        grava(os.path.join(sd, "cmdline.txt"), "pp --gera --celula 1b:/m --so-texto curto --fracao-slots 1.0 --saida /s/p1/saida\n")
        # intervalo do binário: 1.000.000 a 1.004.000 ms; leituras carimbadas (ts, usada, total); `mem` = lista (ts, usada, total) ou texto cru
        mem = kw.get("mem", [(1000500, 1000, 24000), (1001500, 9000, 24000), (1002500, 9000, 24000), (1003500, 1000, 24000)])
        grava(os.path.join(sd, "gpumem.csv"), mem if isinstance(mem, str) else "".join("%d, %s, %s\n" % x for x in mem))
        if kw.get("tempos", True):
            grava(os.path.join(sd, "smoke-tempos.txt"), "coletor_inicio_ms=999900\nbin_inicio_ms=1000000\nbin_fim_ms=1004000\ncoletor_fim_ms=1004100\n")
        grava(os.path.join(sd, "gpumem-pos-saida.csv"), "1004200, 300, 24000\n")
        g = list(GERADOS) if ids_iguais else [x + 1 for x in GERADOS]
        grava(os.path.join(sd, "p1", "saida", "medidas-1b.json"), {"medidas": [{"descartada": False, "falha": None, "gerados": g, "cont": {"needed": kw.get("needed", 100), "cpu": kw.get("cpu", 0), "hits": 100}}]})
        grava(os.path.join(sd, "p1", "stdout.txt"), kw.get("stdout", ""))
        grava(os.path.join(sd, "p1", "stderr.txt"), kw.get("stderr", "slots 768 de 768\n"))
        return sd

    def avalia(self, sd, m="1b"):
        return X.avalia_smoke(sd, m, self.ids, self.refs, self.textos)

    def lastro100(self, m="1b", t="curto", estado="frio", n=3, cmdline=True, fracao="1.0", slots_err="slots 768 de 768\n", **kw):
        kw.setdefault("cpu", 0)
        d = self.lastro(m=m, t=t, estado=estado, n=n, **kw)
        novo = d.replace("-L-", "-L100-")
        os.rename(d, novo)
        arq = os.path.join(novo, "celula.json")
        meta = json.load(open(arq, encoding="utf-8"))
        meta["serie"] = "L100"
        meta["fracao_slots"] = fracao
        grava(arq, meta)
        if cmdline:
            procs = sorted(x for x in os.listdir(novo) if x.startswith("p") and x[1:].isdigit())
            grava(os.path.join(novo, "cmdline.txt"), "".join("%s: pp --gera --celula %s:/m --so-texto %s --fracao-slots 1.0 --saida x\n" % (p, m, t) for p in procs))
        for p in sorted(x for x in os.listdir(novo) if x.startswith("p") and x[1:].isdigit()):
            grava(os.path.join(novo, p, "stderr.txt"), slots_err)
        return novo

    def roda1(self):
        return X.processa(self.saida, self.ids, self.refs, self.textos, degrau1=True)

    def razao1(self, res, m="1b", t="curto", estado="frio"):
        return self.razao(res, m, t, estado, "S1")


class TestDegrau1Smoke(BaseD1):
    def test_smoke_ok(self):
        ok, lin = self.avalia(self.smoke_ok())
        self.assertTrue(ok, lin)
        self.assertIn("veredito=OK", lin)
        self.assertIn("ids=IGUAIS", lin)
        self.assertIn("gpu_pico_pct=37.5", lin)
        self.assertTrue(any(x.startswith("slots_impressos=slots 768 de 768") for x in lin))

    def test_smoke_rc_diferente_de_zero_falha(self):
        sd = self.smoke_ok()
        grava(os.path.join(sd, "rc.txt"), "1\n")
        ok, lin = self.avalia(sd)
        self.assertFalse(ok)
        self.assertTrue(any("código 0" in x for x in lin))

    def test_smoke_memoria_acima_de_90_falha(self):
        ok, lin = self.avalia(self.smoke_ok(mem=[(1000500 + 1000 * i, 22000, 24000) for i in range(4)]))
        self.assertFalse(ok)
        self.assertTrue(any("90%" in x for x in lin))
        ok2, _ = self.avalia(self.smoke_ok(mem=[(1000500 + 1000 * i, 21600, 24000) for i in range(4)]))     # exatamente 90%: passa
        self.assertTrue(ok2)

    def test_smoke_pico_acima_de_90_so_numa_leitura_viva_falha(self):
        ok, _ = self.avalia(self.smoke_ok(mem=[(1000500, 1000, 24000), (1001500, 23000, 24000), (1002500, 1000, 24000), (1003500, 1000, 24000)]))
        self.assertFalse(ok)

    def test_smoke_sem_leitura_de_memoria_falha_fechada(self):
        ok, lin = self.avalia(self.smoke_ok(mem="lixo\n"))
        self.assertFalse(ok)

    def test_smoke_sem_amostras_no_intervalo_falha_mesmo_com_pos_saida_boa(self):
        # monitor que falhou durante toda a execução e só respondeu depois de o binário sair: nada dentro do intervalo
        ok, lin = self.avalia(self.smoke_ok(mem=[(1004200, 300, 24000), (1004700, 300, 24000), (1005200, 300, 24000)]))
        self.assertFalse(ok)
        self.assertTrue(any("durante a execução do binário" in x for x in lin))
        ok2, _ = self.avalia(self.smoke_ok(mem=[]))
        self.assertFalse(ok2)

    def test_smoke_menos_de_3_leituras_no_intervalo_falha(self):
        ok, _ = self.avalia(self.smoke_ok(mem=[(1000500, 1000, 24000), (1003500, 1000, 24000)]))
        self.assertFalse(ok)
        ok2, _ = self.avalia(self.smoke_ok(mem=[(1000500, 1000, 24000), (1002000, 1000, 24000), (1003500, 1000, 24000)]))
        self.assertTrue(ok2)

    def test_smoke_coleta_interrompida_ou_com_lacuna_falha(self):
        # coletor morreu cedo: 3 leituras no começo, depois nada até o fim (lacuna final > 3 s)
        ok, lin = self.avalia(self.smoke_ok(mem=[(1000100, 1000, 24000), (1000300, 1000, 24000), (1000600, 1000, 24000)]))
        self.assertFalse(ok)
        self.assertTrue(any("não cobrem" in x for x in lin))
        # lacuna no meio > 3 s (intervalo longo)
        sd = self.smoke_ok(mem=[(1000500, 1, 24000), (1001000, 1, 24000), (1005000, 1, 24000), (1009500, 1, 24000), (1010000, 1, 24000)])
        grava(os.path.join(sd, "smoke-tempos.txt"), "bin_inicio_ms=1000000\nbin_fim_ms=1010500\n")
        ok2, _ = self.avalia(sd)
        self.assertFalse(ok2)
        # início tardio (> 3 s depois do começo)
        ok3, _ = self.avalia(self.smoke_ok(mem=[(1003100, 1, 24000), (1003300, 1, 24000), (1003500, 1, 24000)]))
        self.assertFalse(ok3)
        # exatamente 3 s de lacuna: passa
        ok4, _ = self.avalia(self.smoke_ok(mem=[(1003000, 1, 24000), (1003500, 1, 24000), (1003800, 1, 24000)]))
        self.assertTrue(ok4)

    def test_smoke_sem_tempos_falha_fechada(self):
        ok, lin = self.avalia(self.smoke_ok(tempos=False))
        self.assertFalse(ok)
        self.assertTrue(any("smoke-tempos" in x for x in lin))

    def test_smoke_pos_saida_e_so_descritiva(self):
        ok, lin = self.avalia(self.smoke_ok())
        self.assertTrue(any(x.startswith("gpu_pos_saida_descritiva=1004200, 300, 24000") for x in lin))

    def test_smoke_slots_ausente_ou_diferente_falha_fechada(self):
        ok, lin = self.avalia(self.smoke_ok(stderr=""))
        self.assertFalse(ok)
        self.assertTrue(any("slots" in x and "não imprimiu" in x for x in lin))
        ok2, lin2 = self.avalia(self.smoke_ok(stderr="slots 77 de 768\n"))
        self.assertFalse(ok2)
        self.assertTrue(any("não é 100%" in x for x in lin2))
        ok3, _ = self.avalia(self.smoke_ok(stderr="", stdout="slots 768 de 768\n"))   # só o stderr conta
        self.assertFalse(ok3)

    def test_smoke_cpu_ausente_ou_null_falha_e_zero_explicito_passa(self):
        for mod in ("remove", "null", "needed_remove", "cont_remove"):
            sd = self.smoke_ok()
            arq = os.path.join(sd, "p1", "saida", "medidas-1b.json")
            j = json.load(open(arq, encoding="utf-8"))
            c = j["medidas"][0]
            if mod == "remove": del c["cont"]["cpu"]
            elif mod == "null": c["cont"]["cpu"] = None
            elif mod == "needed_remove": del c["cont"]["needed"]
            else: del c["cont"]
            grava(arq, j)
            ok, lin = self.avalia(sd)
            self.assertFalse(ok, mod)
            self.assertTrue(any("contadores do motor ilegíveis" in x for x in lin), (mod, lin))
        ok, lin = self.avalia(self.smoke_ok(cpu=0))
        self.assertTrue(ok, lin)
        self.assertTrue(any(x.startswith("cpu_needed=0") for x in lin))

    def test_smoke_erro_de_alocacao_falha(self):
        ok, lin = self.avalia(self.smoke_ok(stderr="slots 768 de 768\nCUDA error: out of memory\n"))
        self.assertFalse(ok)
        self.assertTrue(any("alocação" in x for x in lin))

    def test_smoke_cpu_needed_diferente_de_zero_falha(self):
        ok, lin = self.avalia(self.smoke_ok(cpu=3))
        self.assertFalse(ok)
        self.assertTrue(any("cpu/needed" in x for x in lin))
        ok2, _ = self.avalia(self.smoke_ok(needed=0))
        self.assertFalse(ok2)

    def test_smoke_ids_divergentes_so_reporta(self):
        ok, lin = self.avalia(self.smoke_ok(ids_iguais=False))
        self.assertTrue(ok, lin)
        self.assertTrue(any(x.startswith("ids=DIVERGEM no passo 0") for x in lin))

    def test_smoke_sem_flag_na_linha_de_comando_falha(self):
        sd = self.smoke_ok()
        grava(os.path.join(sd, "cmdline.txt"), "pp --gera --celula 1b:/m --so-texto curto\n")
        ok, _ = self.avalia(sd)
        self.assertFalse(ok)

    def test_cli_smoke_grava_veredito_e_codigo(self):
        sd = self.smoke_ok(cpu=1)
        rc = X.main(["--smoke", sd, "--modelo", "1b", "--ids", self.ids, "--refs", self.refs, "--textos", self.textos])
        self.assertEqual(rc, 1)
        self.assertIn("veredito=FALHA", open(os.path.join(sd, "veredito.txt"), encoding="utf-8").read())


class TestDegrau1Extrator(BaseD1):
    def veredito(self, ok=True, motivo=""):
        grava(os.path.join(self.saida, "smoke", "veredito.txt"), "veredito=%s\n%s" % ("OK" if ok else "FALHA", ("motivo=%s\n" % motivo) if motivo else ""))

    def test_razao_l100_contra_s1_com_rotulo_exato_e_so_s1(self):
        self.veredito()
        self.lastro100(estado="frio", n=3, prefill=100.0, passo=18.0)
        self.llama(estado="frio", serie="S1", n=3, prefill=50.0, per_token=2.0)
        res = self.roda1()
        self.assertEqual({r["serie"] for r in res["razoes"]}, {"S1"})
        self.assertFalse(any(c["serie"] in ("S2", "S3", "L") for c in res["celulas"].values()))
        r = self.razao1(res)
        self.assertEqual(r["status"], "OK", r.get("motivos"))
        self.assertEqual(r["rotulo"], "Lastro 100% experts GPU contra llama.cpp S1; KV f32 no Lastro contra f16 no llama.cpp")
        self.assertAlmostEqual(r["prefill"]["razao_lastro_sobre_llama_cpp"], 2.0)
        self.assertAlmostEqual(r["decode"]["razao_lastro_sobre_llama_cpp"], 9.0)
        self.assertEqual(r["fracao_cpu_needed"]["mediana"], 0.0)
        self.assertTrue(res["degrau1"])

    def test_relatorio_cita_faixas_e_onde_o_dado_caiu(self):
        self.veredito()
        self.lastro100(estado="frio", n=3, passo=18.0)
        self.llama(estado="frio", serie="S1", n=3, per_token=2.0)
        md = X.relatorio(self.roda1())
        self.assertIn("Lastro 100% experts GPU contra llama.cpp S1; KV f32 no Lastro contra f16 no llama.cpp", md)
        self.assertIn("decode L100/S1 <= 2", md)
        self.assertIn("decode L100/S1 entre 2 e 6", md)
        self.assertIn("decode L100/S1 > 6", md)
        self.assertIn("razão de decode 9.000 cai em 'decode L100/S1 > 6", md)
        self.assertIn("Prefill: a razão será provavelmente maior", md)
        self.assertNotIn("Lastro 10% experts GPU contra", md)
        self.assertNotIn("PASSA", md.replace("PASSA/REPROVA", ""))

    def test_faixas_nos_limites(self):
        self.assertIn("<= 2", X.faixa_d1(2.0))
        self.assertIn("entre 2 e 6", X.faixa_d1(2.0001))
        self.assertIn("entre 2 e 6", X.faixa_d1(6.0))
        self.assertIn("> 6", X.faixa_d1(6.0001))

    def test_sem_smoke_nenhuma_razao(self):
        self.lastro100(estado="frio", n=3)
        self.llama(estado="frio", serie="S1", n=3)
        r = self.razao1(self.roda1())
        self.assertEqual(r["status"], "SEM PROVA")
        self.assertIn("smoke", " ".join(r["motivos"]))

    def test_smoke_falho_vira_l100_nao_sustenta_100(self):
        self.veredito(ok=False, motivo="cpu/needed = 0.1000, não 0")
        self.lastro100(estado="frio", n=3)
        self.llama(estado="frio", serie="S1", n=3)
        res = self.roda1()
        r = self.razao1(res)
        self.assertNotEqual(r["status"], "OK")
        self.assertIsNone(r.get("decode"))
        self.assertEqual(res["resultado_smoke"], "L100 NÃO SUSTENTA 100%: cpu/needed = 0.1000, não 0")
        md = X.relatorio(res)
        self.assertIn("L100 NÃO SUSTENTA 100%: cpu/needed = 0.1000, não 0", md)
        self.assertNotIn("Integridade QUEBRADA", md)

    def test_cpu_needed_diferente_de_zero_na_celula_e_sem_prova(self):
        self.veredito()
        self.lastro100(estado="frio", n=3, cpu=5)
        self.llama(estado="frio", serie="S1", n=3)
        r = self.razao1(self.roda1())
        self.assertEqual(r["status"], "SEM PROVA")
        self.assertTrue(any("não é 100%" in x for x in r["motivos"]))

    def zera(self):
        import shutil
        shutil.rmtree(self.saida)
        os.makedirs(self.saida)
        self.integra()

    def edita_cont(self, d, f):
        for p in sorted(x for x in os.listdir(d) if x.startswith("p") and x[1:].isdigit()):
            arq = os.path.join(d, p, "saida", "medidas-1b.json")
            j = json.load(open(arq, encoding="utf-8"))
            for e in j["medidas"]:
                f(e)
            grava(arq, j)

    def test_celula_l100_cpu_ausente_ou_null_e_sem_prova_e_zero_explicito_vale(self):
        def ausente(e): del e["cont"]["cpu"]
        def nulo(e): e["cont"]["cpu"] = None
        for nome, f in (("ausente", ausente), ("null", nulo)):
            self.veredito()
            d = self.lastro100(estado="frio", n=3)
            self.llama(estado="frio", serie="S1", n=3)
            self.edita_cont(d, f)
            r = self.razao1(self.roda1())
            self.assertEqual(r["status"], "SEM PROVA", nome)
            self.assertIsNone(r.get("decode"), nome)
            self.assertTrue(any("cont.cpu" in x for x in r["motivos"]), (nome, r["motivos"]))
            self.zera()
        self.veredito()
        self.lastro100(estado="frio", n=3, cpu=0)
        self.llama(estado="frio", serie="S1", n=3)
        r = self.razao1(self.roda1())
        self.assertEqual(r["status"], "OK", r.get("motivos"))

    def test_celula_l100_sem_ou_com_slots_diferentes_e_sem_prova(self):
        for err in ("", "slots 77 de 768\n"):
            self.veredito()
            self.lastro100(estado="frio", n=3, slots_err=err)
            self.llama(estado="frio", serie="S1", n=3)
            r = self.razao1(self.roda1())
            self.assertEqual(r["status"], "SEM PROVA", err)
            self.zera()

    def test_celula_sem_cmdline_ou_sem_flag_e_sem_prova(self):
        self.veredito()
        self.lastro100(estado="frio", n=3, cmdline=False)
        self.llama(estado="frio", serie="S1", n=3)
        r = self.razao1(self.roda1())
        self.assertEqual(r["status"], "SEM PROVA")
        self.assertTrue(any("cmdline.txt" in x for x in r["motivos"]))

    def test_sequencias_diferentes_marcadas_mas_razao_publicada(self):
        self.veredito()
        outro = [x + 1 for x in GERADOS]
        self.lastro100(estado="frio", n=3, gerados=outro)
        self.llama(estado="frio", serie="S1", n=3)
        r = self.razao1(self.roda1())
        self.assertEqual(r["status"], "OK")
        self.assertEqual(r["marca_sequencia"], "sequências diferentes")

    def test_ruidosa_e_nao_rodou_e_tudo_ok_nas_12_celulas_do_modelo(self):
        self.veredito()
        for est, n in (("frio", 3), ("quente", 5)):
            for t in IDS:
                self.lastro100(t=t, estado=est, n=n)
                self.llama(t=t, estado=est, serie="S1", n=n)
        res = self.roda1()
        self.assertEqual(len(res["razoes"]), 12)
        self.assertTrue(all(r["status"] == "OK" for r in res["razoes"] if r["modelo"] == "1b"), [r.get("motivos") for r in res["razoes"]])
        # a janela é do 1B só quando orcamento.json o declara; sem declaração, o 3B aparece como NÃO RODOU
        self.assertTrue(all(r["status"] == "NÃO RODOU" for r in res["razoes"] if r["modelo"] == "3b"))

    def test_main_degrau1_grava_resultado(self):
        self.veredito()
        self.lastro100(estado="frio", n=3)
        self.llama(estado="frio", serie="S1", n=3)
        rc = X.main(["--saida", self.saida, "--ids", self.ids, "--refs", self.refs, "--textos", self.textos, "--degrau1"])
        self.assertEqual(rc, 0)
        res = json.load(open(os.path.join(self.saida, "cmp-resultado.json"), encoding="utf-8"))
        self.assertTrue(res["degrau1"])
        self.assertIn("Degrau 1", open(os.path.join(self.saida, "cmp-resultado.md"), encoding="utf-8").read())

    def test_modo_antigo_continua_com_l_e_rotulo_10(self):
        self.lastro(estado="frio", n=3)
        self.llama(estado="frio", serie="S1", n=3)
        res = self.roda()
        self.assertFalse(res.get("degrau1"))
        self.assertTrue(self.razao(res)["rotulo"].startswith("Lastro 10% experts GPU contra llama.cpp S1"))


class BaseD2(BaseD1):
    """Degrau 2: LC (Lastro --camadas-cpu N) contra S2 (criterio-degrau2-camadas-20261109.md)."""
    LINHA = {"1b": "camadas-cpu 22 de 24; slots 64 de 768\n", "3b": "camadas-cpu 29 de 32; slots 120 de 1280\n"}
    NN = {"1b": 22, "3b": 29}

    EXATO = {"1b": (24 * 8 * 152, 22 * 8 * 152), "3b": (32 * 8 * 152, 29 * 8 * 152)}   # (needed, cpu) = (L, N) * k * posições

    def smoke2(self, m="1b", cmdline=None, **kw):
        kw.setdefault("needed", self.EXATO[m][0])
        kw.setdefault("cpu", self.EXATO[m][1])
        kw.setdefault("stderr", self.LINHA[m])
        sd = self.smoke_ok(**kw)
        grava(os.path.join(sd, "cmdline.txt"), cmdline if cmdline is not None else "pp --gera --celula %s:/m --so-texto curto --camadas-cpu %d --saida /s/p1/saida\n" % (m, self.NN[m]))
        if m != "1b":
            os.rename(os.path.join(sd, "p1", "saida", "medidas-1b.json"), os.path.join(sd, "p1", "saida", "medidas-%s.json" % m))
        return sd

    def avalia2(self, sd, m="1b"):
        return X.avalia_smoke(sd, m, self.ids, self.refs, self.textos, d2=True)

    def lastroLC(self, m="1b", t="curto", estado="frio", n=3, cmdline=True, camadas=None, linha=None, **kw):
        kw.setdefault("needed", self.EXATO[m][0])
        kw.setdefault("cpu", self.EXATO[m][1])
        d = self.lastro(m=m, t=t, estado=estado, n=n, **kw)
        novo = d.replace("-L-", "-LC-")
        os.rename(d, novo)
        arq = os.path.join(novo, "celula.json")
        meta = json.load(open(arq, encoding="utf-8"))
        meta["serie"] = "LC"
        meta["camadas_cpu"] = camadas if camadas is not None else str(self.NN[m])
        grava(arq, meta)
        procs = sorted(x for x in os.listdir(novo) if x.startswith("p") and x[1:].isdigit())
        if cmdline:
            grava(os.path.join(novo, "cmdline.txt"), "".join("%s: pp --gera --celula %s:/m --so-texto %s --camadas-cpu %d --saida x\n" % (p, m, t, self.NN[m]) for p in procs))
        for p in procs:
            grava(os.path.join(novo, p, "stderr.txt"), linha if linha is not None else self.LINHA[m])
        return novo

    def roda2(self):
        return X.processa(self.saida, self.ids, self.refs, self.textos, degrau2=True)


class TestDegrau2Smoke(BaseD2):
    def test_smoke_ok_com_cpu_needed_alto(self):
        ok, lin = self.avalia2(self.smoke2())
        self.assertTrue(ok, lin)
        self.assertIn("veredito=OK", lin)
        self.assertIn("cpu_needed=%s" % (22 / 24), lin)
        self.assertIn("camadas_cpu=22", lin)
        self.assertIn("ids=IGUAIS", lin)

    def test_smoke_3b_ok(self):
        ok, lin = self.avalia2(self.smoke2(m="3b"), m="3b")
        self.assertTrue(ok, lin)

    def test_smoke_cpu_zero_ou_exato_menos_um_falha_e_exato_passa_nos_dois_modelos(self):
        for m in ("1b", "3b"):
            needed, exato = self.EXATO[m]
            for cpu in (0, exato - 1, exato + 1):
                ok, lin = self.avalia2(self.smoke2(m=m, cpu=cpu), m=m)
                self.assertFalse(ok, (m, cpu))
                self.assertTrue(any("esperado %d/%d exato" % (self.NN[m], X.CAMADAS_EXPERTS[m][0]) in x for x in lin), lin)
            ok, lin = self.avalia2(self.smoke2(m=m), m=m)
            self.assertTrue(ok, lin)

    def test_smoke_linha_ausente_errada_ou_antiga_falha(self):
        for erro in ("", "slots 768 de 768\n", "camadas-cpu 21 de 24; slots 96 de 768\n", "camadas-cpu 22 de 24; slots 768 de 768\n",
                     "camadas-cpu 22 de 24; slots 64 de 700\n", "camadas-cpu 29 de 32; slots 120 de 1280\n"):
            ok, lin = self.avalia2(self.smoke2(stderr=erro))
            self.assertFalse(ok, erro)
            self.assertTrue(any(x.startswith("motivo=") and ("camadas-cpu N" in x or "linha de camadas" in x) for x in lin), lin)

    def test_smoke_3b_com_linha_do_1b_falha(self):
        ok, _ = self.avalia2(self.smoke2(m="3b", stderr=self.LINHA["1b"]), m="3b")
        self.assertFalse(ok)

    def test_smoke_cmdline_sem_flag_ou_com_fracao_falha(self):
        for cl in ("pp --gera --so-texto curto --saida x\n", "pp --gera --so-texto curto --fracao-slots 1.0 --saida x\n",
                   "pp --gera --so-texto curto --camadas-cpu 220 --saida x\n", "pp --gera --camadas-cpu 22 --fracao-slots 1.0 --saida x\n", "pp --gera --camadas-cpu 29 --saida x\n"):
            ok, lin = self.avalia2(self.smoke2(cmdline=cl))
            self.assertFalse(ok, cl)
            self.assertTrue(any("sem --camadas-cpu 22" in x for x in lin), lin)

    def test_smoke_memoria_rc_e_leituras_seguem_as_regras_da_emenda_1(self):
        self.assertFalse(self.avalia2(self.smoke2(mem=[(1000500, 23000, 24000), (1001500, 23000, 24000), (1002500, 23000, 24000), (1003500, 23000, 24000)]))[0])
        self.assertFalse(self.avalia2(self.smoke2(mem=""))[0])
        self.assertFalse(self.avalia2(self.smoke2(mem=[(1000500, 1000, 24000), (1001500, 1000, 24000)]))[0])
        sd = self.smoke2()
        grava(os.path.join(sd, "rc.txt"), "1\n")
        self.assertFalse(self.avalia2(sd)[0])
        self.setUp()
        self.assertFalse(self.avalia2(self.smoke2(tempos=False))[0])

    def test_smoke_cont_ausente_null_ou_needed_zero_falha(self):
        sd = self.smoke2()
        arq = os.path.join(sd, "p1", "saida", "medidas-1b.json")
        for cont in ({"needed": 100, "hits": 1}, {"needed": 100, "cpu": None}, {"cpu": 5}, {"needed": 0, "cpu": 0}):
            grava(arq, {"medidas": [{"descartada": False, "falha": None, "gerados": list(GERADOS), "cont": cont}]})
            ok, lin = self.avalia2(sd)
            self.assertFalse(ok, cont)

    def test_smoke_ids_divergentes_so_reporta(self):
        ok, lin = self.avalia2(self.smoke2(ids_iguais=False))
        self.assertTrue(ok, lin)
        self.assertTrue(any(x.startswith("ids=DIVERGEM") for x in lin))

    def test_smoke_modo_d1_continua_exigindo_cpu_zero_e_fracao(self):
        ok, _ = self.avalia(self.smoke_ok(cpu=7))
        self.assertFalse(ok)

    def test_cli_smoke_degrau2(self):
        sd = self.smoke2()
        rc = X.main(["--smoke", sd, "--modelo", "1b", "--degrau2", "--ids", self.ids, "--refs", self.refs, "--textos", self.textos])
        self.assertEqual(rc, 0)
        self.assertIn("veredito=OK", open(os.path.join(sd, "veredito.txt"), encoding="utf-8").read())
        sd2 = self.smoke2(stderr="")
        self.assertEqual(X.main(["--smoke", sd2, "--modelo", "1b", "--degrau2", "--ids", self.ids, "--refs", self.refs, "--textos", self.textos]), 1)
        self.assertEqual(X.main(["--smoke", sd2, "--modelo", "1b", "--degrau1", "--degrau2", "--ids", self.ids, "--refs", self.refs, "--textos", self.textos]), 2)


class TestDegrau2Extrator(BaseD2):
    def veredito(self, ok=True, motivo=""):
        grava(os.path.join(self.saida, "smoke", "veredito.txt"), "veredito=%s\n%s" % ("OK" if ok else "FALHA", ("motivo=%s\n" % motivo) if motivo else ""))

    def test_razao_lc_contra_s2_com_rotulo_exato_e_so_s2(self):
        self.veredito()
        self.lastroLC(estado="frio", n=3, prefill=100.0, passo=18.0)
        self.llama(estado="frio", serie="S2", n=3, prefill=50.0, per_token=2.0)
        res = self.roda2()
        self.assertEqual({r["serie"] for r in res["razoes"]}, {"S2"})
        self.assertFalse(any(c["serie"] in ("S1", "S3", "L", "L100") for c in res["celulas"].values()))
        r = self.razao(res, "1b", "curto", "frio", "S2")
        self.assertEqual(r["status"], "OK", r.get("motivos"))
        self.assertEqual(r["rotulo"], "Lastro camadas inteiras (22 na CPU) contra llama.cpp S2 (--n-cpu-moe 22); KV f32 no Lastro contra f16 no llama.cpp; "
                                      "op offload ligado no llama.cpp, prefill da S2 não é leitura de experts na CPU")
        self.assertAlmostEqual(r["prefill"]["razao_lastro_sobre_llama_cpp"], 2.0)
        self.assertAlmostEqual(r["decode"]["razao_lastro_sobre_llama_cpp"], 9.0)
        self.assertGreater(r["fracao_cpu_needed"]["mediana"], 0.5, "cpu/needed alto é reportado, não reprovado")
        self.assertTrue(res["degrau2"])

    def test_rotulo_do_3b_leva_29(self):
        self.veredito()
        self.lastroLC(m="3b", estado="frio", n=3)
        self.llama(m="3b", estado="frio", serie="S2", n=3)
        r = self.razao(self.roda2(), "3b", "curto", "frio", "S2")
        self.assertEqual(r["status"], "OK", r.get("motivos"))
        self.assertIn("(29 na CPU) contra llama.cpp S2 (--n-cpu-moe 29)", r["rotulo"])

    def test_faixas_nos_limites(self):
        self.assertTrue(X.faixa_d2(1.25).endswith("equivale"))
        self.assertTrue(X.faixa_d2(1.2500001).endswith("diferença moderada"))
        self.assertTrue(X.faixa_d2(2.0).endswith("diferença moderada"))
        self.assertTrue(X.faixa_d2(2.0000001).endswith("diferença grande"))
        self.assertTrue(X.faixa_d2(0.5).endswith("equivale"))

    def test_relatorio_cita_faixas_onde_o_dado_caiu_e_nao_decide(self):
        self.veredito()
        self.lastroLC(estado="frio", n=3, passo=18.0)
        self.llama(estado="frio", serie="S2", n=3, per_token=2.0)
        res = self.roda2()
        md = X.relatorio(res)
        for t in ("decode LC/S2 <= 1,25: equivale", "entre 1,25 e 2: diferença moderada", "decode LC/S2 > 2: diferença grande", "Onde o dado caiu", "Degrau 2"):
            self.assertIn(t, md) if t != "Onde o dado caiu" else None
        self.assertIn("1b curto frio: razão de decode 9.000 cai em 'decode LC/S2 > 2: diferença grande'", md)
        self.assertNotIn("PASSA", md.replace("sem PASSA/REPROVA", ""))
        self.assertNotIn("L100", md)

    def test_sem_smoke_ou_smoke_falho_nenhuma_razao(self):
        self.lastroLC(estado="frio", n=3)
        self.llama(estado="frio", serie="S2", n=3)
        res = self.roda2()
        self.assertTrue(all(r["status"] != "OK" for r in res["razoes"]))
        self.assertTrue(any("smoke do degrau 2 ausente" in x for x in res["integridade_problemas"]))
        self.veredito(ok=False, motivo="linha de camadas/slots diferente")
        res = self.roda2()
        self.assertTrue(all(r["status"] != "OK" for r in res["razoes"]))
        self.assertEqual(res["resultado_smoke"], "SMOKE DO DEGRAU 2 FALHOU: linha de camadas/slots diferente")
        self.assertIn("SMOKE DO DEGRAU 2 FALHOU", X.relatorio(res))

    def test_celula_lc_cpu_zero_ou_exato_menos_um_e_sem_prova_e_exato_vale_nos_dois_modelos(self):
        for m in ("1b", "3b"):
            for cpu in (0, self.EXATO[m][1] - 1):
                self.setUp()
                self.veredito()
                self.lastroLC(m=m, estado="frio", n=3, cpu=cpu)
                self.llama(m=m, estado="frio", serie="S2", n=3)
                r = self.razao(self.roda2(), m, "curto", "frio", "S2")
                self.assertEqual(r["status"], "SEM PROVA", (m, cpu))
                self.assertTrue(any("esperado %d/%d exato: o posicionamento por camada não é o declarado" % (self.NN[m], X.CAMADAS_EXPERTS[m][0]) in x for x in r["motivos"]), r["motivos"])
            self.setUp()
            self.veredito()
            self.lastroLC(m=m, estado="frio", n=3)
            self.llama(m=m, estado="frio", serie="S2", n=3)
            self.assertEqual(self.razao(self.roda2(), m, "curto", "frio", "S2")["status"], "OK")

    def test_celula_lc_sem_linha_ou_linha_errada_e_sem_prova(self):
        self.veredito()
        for linha in ("", "slots 768 de 768\n", "camadas-cpu 22 de 24; slots 768 de 768\n", "camadas-cpu 29 de 32; slots 120 de 1280\n"):
            self.lastroLC(estado="frio", n=3, linha=linha)
            self.llama(estado="frio", serie="S2", n=3)
            r = self.razao(self.roda2(), "1b", "curto", "frio", "S2")
            self.assertEqual(r["status"], "SEM PROVA", linha)
            self.assertTrue(any("camadas-cpu N" in x or "linha de camadas" in x for x in r["motivos"]), r["motivos"])
            self.setUp()
            self.veredito()

    def test_celula_lc_sem_cmdline_ou_com_camadas_erradas_e_sem_prova(self):
        self.veredito()
        self.lastroLC(estado="frio", n=3, cmdline=False)
        self.llama(estado="frio", serie="S2", n=3)
        self.assertEqual(self.razao(self.roda2(), "1b", "curto", "frio", "S2")["status"], "SEM PROVA")
        self.setUp()
        self.veredito()
        self.lastroLC(estado="frio", n=3, camadas="29")
        self.llama(estado="frio", serie="S2", n=3)
        self.assertEqual(self.razao(self.roda2(), "1b", "curto", "frio", "S2")["status"], "SEM PROVA")

    def test_cont_ausente_ou_null_na_celula_lc_e_sem_prova(self):
        self.veredito()
        d = self.lastroLC(estado="frio", n=3)
        self.llama(estado="frio", serie="S2", n=3)
        for p in ("p1",):
            arq = os.path.join(d, p, "saida", "medidas-1b.json")
            j = json.load(open(arq, encoding="utf-8"))
            for e in j["medidas"]:
                e["cont"] = {"needed": 100, "cpu": None}
            grava(arq, j)
        self.assertEqual(self.razao(self.roda2(), "1b", "curto", "frio", "S2")["status"], "SEM PROVA")

    def test_nao_rodou_s1_s3_nao_aparecem_e_main_degrau2_grava_resultado(self):
        self.veredito()
        self.lastroLC(estado="frio", n=3)
        self.llama(estado="frio", serie="S2", n=3)
        rc = X.main(["--saida", self.saida, "--ids", self.ids, "--refs", self.refs, "--textos", self.textos, "--degrau2"])
        self.assertEqual(rc, 0)
        res = json.load(open(os.path.join(self.saida, "cmp-resultado.json"), encoding="utf-8"))
        self.assertTrue(res["degrau2"])
        self.assertFalse(any(k.startswith("llamacpp") and ("-S1-" in k or "-S3-" in k) for k in res["celulas"]))
        self.assertIn("Degrau 2", open(os.path.join(self.saida, "cmp-resultado.md"), encoding="utf-8").read())
        self.assertEqual(X.main(["--saida", self.saida, "--degrau1", "--degrau2"]), 2)

    def test_modos_antigo_e_degrau1_intactos_depois_do_degrau2(self):
        self.veredito()
        self.lastroLC(estado="frio", n=3)
        self.llama(estado="frio", serie="S2", n=3)
        self.roda2()
        self.setUp()
        self.lastro(estado="frio", n=3)
        self.llama(estado="frio", serie="S1", n=3)
        res = self.roda()
        self.assertFalse(res.get("degrau2"))
        self.assertTrue(self.razao(res)["rotulo"].startswith("Lastro 10% experts GPU contra llama.cpp S1"))


class TestSoS3(Base):
    """Emenda 12 do critério base: --so-s3, janela só com a S3 do llama.cpp; sem Lastro, sem razão."""

    def roda3(self):
        return X.processa(self.saida, self.ids, self.refs, self.textos, so_s3=True)

    def test_s3_sozinha_so_medidas_descritivas_sem_razao_e_sem_celula_do_lastro(self):
        self.llama(estado="frio", serie="S3", n=3, prefill=50.0, per_token=2.0)
        self.llama(estado="quente", serie="S3", n=5, prefill=40.0, per_token=2.0)
        grava(os.path.join(self.saida, "orcamento.json"), {"modelos": ["1b"], "cortes": [], "so_s3": True})
        res = self.roda3()
        self.assertTrue(res["so_s3"])
        self.assertEqual(res["razoes"], [], "sem Lastro na MESMA janela não há razão")
        self.assertEqual({c["lado"] for c in res["celulas"].values()}, {"llamacpp"})
        self.assertEqual({c["serie"] for c in res["celulas"].values()}, {"S3"})
        self.assertEqual(len(res["celulas"]), 6, "3 textos x frio/quente do modelo, nada de S1/S2")
        c = res["celulas"]["llamacpp-1b-S3-curto-frio"]
        self.assertEqual(c["status"], "OK", c.get("motivos"))
        self.assertAlmostEqual(c["prefill"]["mediana"], 50.0)
        nr = [k for k, v in res["celulas"].items() if v["status"] == "NÃO RODOU"]
        self.assertEqual(len(nr), 4)
        self.assertFalse(any(k.startswith("lastro") for k in res["celulas"]), "Lastro nem aparece como NÃO RODOU")

    def test_relatorio_diz_explicitamente_que_nao_ha_razao_e_nao_inventa_numero(self):
        self.llama(estado="frio", serie="S3", n=3)
        md = X.relatorio(self.roda3())
        self.assertIn("SEM razão", md)
        self.assertIn("NÃO tem células do Lastro", md)
        self.assertIn("MESMA janela", md)
        self.assertNotIn("razão prefill", md)
        self.assertNotIn("| Lastro prefill", md)
        self.assertIn("llamacpp-1b-S3-curto-frio", md)

    def test_integridade_quebrada_vira_sem_prova(self):
        self.llama(estado="frio", serie="S3", n=3)
        self.integra(depois={"binario-lastro": "outro"})
        res = self.roda3()
        self.assertTrue(res["integridade_problemas"])
        self.assertEqual(res["celulas"]["llamacpp-1b-S3-curto-frio"]["status"], "SEM PROVA")
        self.assertIn("Integridade QUEBRADA", X.relatorio(res))

    def test_celula_com_erro_fica_sem_prova_e_modelo_declarado_filtra(self):
        self.llama(estado="frio", serie="S3", n=3, predicted_n=10)
        grava(os.path.join(self.saida, "orcamento.json"), {"modelos": ["3b"], "cortes": [], "so_s3": True})
        res = self.roda3()
        self.assertEqual({c["modelo"] for c in res["celulas"].values()}, {"3b"})
        self.assertTrue(all(c["status"] == "NÃO RODOU" for c in res["celulas"].values()))

    def test_main_so_s3_grava_resultado_e_e_exclusivo(self):
        self.llama(estado="frio", serie="S3", n=3)
        base = ["--saida", self.saida, "--ids", self.ids, "--refs", self.refs, "--textos", self.textos]
        self.assertEqual(X.main(base + ["--so-s3"]), 0)
        res = json.load(open(os.path.join(self.saida, "cmp-resultado.json"), encoding="utf-8"))
        self.assertTrue(res["so_s3"])
        self.assertEqual(res["razoes"], [])
        self.assertIn("S3 do llama.cpp sozinha", open(os.path.join(self.saida, "cmp-resultado.md"), encoding="utf-8").read())
        for extra in (["--degrau1"], ["--degrau2"]):
            self.assertEqual(X.main(base + ["--so-s3"] + extra), 2)
        self.assertEqual(X.main(["--smoke", self.saida, "--modelo", "1b", "--so-s3"]), 2)

    def test_modos_antigos_intactos(self):
        self.lastro(estado="frio", n=3)
        self.llama(estado="frio", serie="S1", n=3)
        res = self.roda()
        self.assertFalse(res.get("so_s3"))
        self.assertTrue(res["razoes"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
