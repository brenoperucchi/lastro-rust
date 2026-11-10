#!/usr/bin/env python3
"""Testes do instrumento-extrai.py com dados SINTÉTICOS. Sem GPU, sem rodar nada do motor.

    python3 -I qual-runs/A2-LOTE-20261103/test-instrumento-extrai.py

Reaproveita o construtor de células de test-perfil-extrai.py (mesmo layout de roda-cmp-lastro.sh) e acrescenta o perfil-bloco.json sintético (versão 2).
"""
import importlib.util
import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout

AQUI = os.path.dirname(os.path.abspath(__file__))


def carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(AQUI, arq))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


T = carrega("test_perfil_extrai", "test-perfil-extrai.py")
E = carrega("instrumento_extrai", "instrumento-extrai.py")
grava = T.grava
NPOS = T.NPOS
S_CAMPOS = E.S_CAMPOS

PADRAO = {"f0": 0.1, "f1": 0.5, "f2": 7.0, "f3": 2.4, "b_pend": 0.0,
          "s_espera": 2.0, "s_monta": 0.2, "s_compute": 0.3, "s_answer": 0.1, "s_livro": 0.2, "s_end_espera": 4.0, "s_end_trab": 0.1, "s_fora": 0.2}


def pedido(m, t, i, v=None, aberto=False, valido=True, cob=0.01, r6=0.9, n=None, plano=None):
    v = dict(PADRAO, **(v or {}))
    n = n if n is not None else NPOS[t]
    plano = plano if plano is not None else E.plano_blocos(n)
    blocos, servico, t0 = [], [], 0
    for k, mk in enumerate(plano):
        esc = 1.0 if mk == 8 else mk / 8.0
        f0 = 0.0 if (i > 0 and k == 0) else v["f0"]
        f = [f0, v["f1"] * esc, v["f2"] * esc, v["f3"] * esc]
        b = sum(f)
        blocos.append({"indice": k, "gen": k, "m": mk, "pos_ini": 8 * k, "f0_pendente": bool(i > 0 and k == 0), "f0_ms": f[0], "f1_ms": f[1], "f2_ms": f[2], "f3_ms": f[3], "b_ms": b,
                       "t_ns": [t0, t0, t0, t0, t0 + int(b * 1e6)], "conserva": True})
        sv = {"indice": k, "gen": k, "aberto": bool(aberto and k == len(plano) - 1)}
        if not sv["aberto"]:
            soma = 0.0
            for c in S_CAMPOS:
                sv[c] = v[c[:-3]]
                sv[c.replace("_ms", "_recorte_ms")] = v[c[:-3]] * 0.9
                soma += v[c[:-3]]
            ig = soma + v["s_fora"]
            sv.update({"ini_ns": t0, "fim_ns": t0 + int(ig * 1e6), "i_ms": ig, "s_fora_ms": v["s_fora"], "s_fora_frac": v["s_fora"] / ig, "inter_bloco_ms": ig / 2, "coerente": True, "n_ops": 20})
        servico.append(sv)
        t0 += int(b * 1e6) + 1000
    pre = sum(bl["b_ms"] for bl in blocos) + 0.5
    return {"modelo": m, "indice": i, "aquecimento": i == 0, "n": n, "plano_m": plano, "prefill_ms": pre, "decode_passo_ms_mediana": 5.0, "blocos": blocos, "servico": servico,
            "f5_ms": 0.3, "r_ini_ms": 0.1, "r_entre_ms": [], "r_fim_ms": 0.1, "residuo_host_ms": 0.2, "cobertura_host_frac": cob, "cobertura_host_ressalva": cob > 0.03,
            "armar_ate_desarmar_ms": pre, "r6": r6, "servico_aberto": bool(aberto), "servico_extra": 0, "s_fora_frac_mediana": 0.05, "valido": valido, "problemas": [] if valido else ["bloco 3: carimbos fora de ordem"]}


def perfil(m, t, n, **kw):
    return {"tipo": "perfil-bloco", "versao": 2, "variavel": "LASTRO_PERFIL_BLOCO", "bloco": 8, "unidades": "ms", "tolerancias": {}, "pedidos": [pedido(m, t, i, **kw) for i in range(n + 1)]}


class Base(T.Base):
    def setUp(self):
        super().setUp()
        self.shas = {k: ("%02d" % i) * 32 for i, k in enumerate(E.FIXO_CHAVES + ("tokenizer-%s" % self.modelo,))}
        self.shas["binario-instr"] = "ae14cbde3965493808818f3aa4a8e7665b6397531e36c2abf020c35fc4f3d969"
        self.shas["gguf-extra"] = "d" * 64
        self.shas["cmp-ids-prompt"] = E.sha_arq(self.ids)
        self.shas["textos-json"] = E.sha_arq(self.textos)
        for mm in ("1b", "3b"):
            self.shas["ref5c-%s" % mm] = E.sha_arq(os.path.join(self.refs, "ref5c-%s.json" % mm))
        self.manifesto()
        for s in ("L100", "LC"):
            self.smoke(s, True)

    # ---- construtores
    def proc(self, s, nn, braco, ms_pos=None, linha=None, perfil_json=True, prefills=None, ids=None, rc_estado=None, **kw):
        """Um processo do controle: pasta ctrl-<m>-<s>-medio-NN-<braco> com 1 aquecimento + 3 medidas."""
        ms_pos = ms_pos if ms_pos is not None else 5.0
        d = self.celula(s, "medio", ms_pos=ms_pos, n=3, p4=False, prefills=prefills, ids=ids, decode=(kw.get("decode_d", 8.0) if braco == "D" else kw.get("decode_l", 8.0)), **{k: v for k, v in kw.items() if k in ("cpu", "needed", "falha", "gerados_reps")})
        novo = os.path.join(self.saida, "celulas", "ctrl-%s-%s-medio-%02d-%s" % (self.modelo, s, nn, braco))
        os.rename(d, novo)
        ligado = (braco == "L") if linha is None else (linha == "ligado")
        txt = "linha de integridade\n" + ("perfil-bloco: ligado\n" if ligado else "perfil-bloco: desligado\n") if linha != "ausente" else "nada\n"
        grava(os.path.join(novo, "p1", "stdout.txt"), txt)
        if perfil_json and (braco == "L" if perfil_json is True else True):
            grava(os.path.join(novo, "p1", "saida", "perfil-bloco.json"), perfil(self.modelo, "medio", 3, **{k: v for k, v in kw.items() if k in ("v", "aberto", "valido", "cob", "r6")}))
        return novo

    def registra_ordem(self, s, ordem="DLLDDL", sequencia=None, rc=0, n=3):
        """Acrescenta ao ordem.txt as linhas do controle da série (na ordem do registro; `sequencia` = lista de NN para inverter só o registro)."""
        with open(os.path.join(self.saida, "ordem.txt"), "a", encoding="utf-8") as fh:
            for k, nn in enumerate(sequencia or range(1, len(ordem) + 1), 1):
                b = ordem[nn - 1]
                fh.write("%02d ctrl-%s-%s-medio-%02d-%s n=%d braco=%s inicio=%ds fim=%ds rc=%d\n" % (k, self.modelo, s, nn, b, n, b, k, k + 1, rc))

    def controle(self, s, dev=0.0, ms_pos=5.0, ordem="DLLDDL", registro=True, **kw):
        for nn, b in enumerate(ordem, 1):
            self.proc(s, nn, b, ms_pos=ms_pos * (1 + dev) if b == "L" else ms_pos, **kw)
        if registro:
            self.registra_ordem(s, ordem)

    def principal(self, s, t, n=5, ms_pos=5.0, v=None, linha=None, ids=None, **kw):
        d = self.celula(s, t, ms_pos=ms_pos, n=n, p4=False, ids=ids, decode=8.0)
        grava(os.path.join(d, "p1", "stdout.txt"), "x\n" + ("perfil-bloco: ligado\n" if linha is None else linha))
        grava(os.path.join(d, "p1", "saida", "perfil-bloco.json"), perfil(self.modelo, t, n, v=v, **kw))
        return d

    def tudo(self, dev=0.0, v10=None, v100=None, **kw):
        for s, v in (("L10", v10), ("L100", v100)):
            self.controle(s, dev=dev, v=v)
            for t in ("curto", "longo"):
                self.principal(s, t, v=v)

    def roda(self):
        return E.processa(self.saida, self.ids, self.refs, self.textos)


class TestControle(unittest.TestCase):
    def procs(self, dev, ordem="DLLDDL", **mod):
        r = []
        for b in ordem:
            r.append({"braco": b, "prefill_ms_mediana": 100.0 * (1 + dev) if b == "L" else 100.0, "passo_ms_mediana": 5.0, "imprimiu_ligado": b == "L", "produziu_perfil": b == "L", "valido": True})
        return r

    def test_magnitude_e_limites_exatos(self):
        casos = [(0.02, "comprovada"), (0.05, "com ressalva"), (0.12, "SEM PROVA"), (-0.02, "comprovada"), (-0.05, "com ressalva"), (-0.12, "SEM PROVA"),
                 (0.03, "comprovada"), (-0.03, "comprovada"), (0.10, "com ressalva"), (-0.10, "com ressalva"), (0.1001, "SEM PROVA"), (0.0301, "com ressalva")]
        for dev, esperado in casos:
            r = E.classifica_controle(self.procs(dev))
            self.assertEqual(r["classe"], esperado, (dev, r))
            self.assertAlmostEqual(r["desvio"], dev, places=9)

    def test_pior_caso_e_passo(self):
        p = self.procs(0.05)
        p[1]["prefill_ms_mediana"] = 120.0
        r = E.classifica_controle(p)
        self.assertAlmostEqual(r["pior_caso"], 0.20)
        p[1]["passo_ms_mediana"], p[2]["passo_ms_mediana"], p[5]["passo_ms_mediana"] = 6.0, 6.0, 6.0
        self.assertAlmostEqual(E.classifica_controle(p)["desvio_passo"], 0.2)

    def test_incompleto(self):
        self.assertEqual(E.classifica_controle(self.procs(0.0)[:5])["classe"], "controle incompleto")           # par truncado
        self.assertEqual(E.classifica_controle(self.procs(0.0, "DLDLDL"))["classe"], "controle incompleto")      # ordem trocada
        self.assertEqual(E.classifica_controle(self.procs(0.0), orcamento_ok=False)["classe"], "controle incompleto")
        p = self.procs(0.0); p[3]["produziu_perfil"] = True
        self.assertIn("braço D", E.classifica_controle(p)["motivo"])
        p = self.procs(0.0); p[3]["imprimiu_ligado"] = True
        self.assertEqual(E.classifica_controle(p)["classe"], "controle incompleto")
        p = self.procs(0.0); p[1]["produziu_perfil"] = False
        self.assertIn("braço L", E.classifica_controle(p)["motivo"])
        p = self.procs(0.0); p[2]["valido"] = False
        self.assertIn("abortado", E.classifica_controle(p)["motivo"])
        r = E.classifica_controle(self.procs(0.0), incompleto_motivo="ordem efetiva errada")
        self.assertEqual((r["classe"], r["motivo"]), ("controle incompleto", "ordem efetiva errada"))
        for c in E.classifica_controle(self.procs(0.0)[:5]), E.classifica_controle(self.procs(0.0), orcamento_ok=False):
            self.assertIsNone(c["desvio"])


class TestHipoteses(unittest.TestCase):
    def h(self, h, serie, ctl="comprovada", **m):
        base = {"f1_b": None, "f2_b": None, "f3_b": None, "f3_passo": None, "r3": None, "s_compute_b": None, "r5": None, "r6": None, "resto": 1}
        base.update(m)
        return E.classifica_h(h, serie, base, ctl)

    def test_h1(self):
        self.assertEqual(self.h("H1", "L100", f2_b=0.70, f3_b=0.2), "dentro")
        self.assertEqual(self.h("H1", "L100", f2_b=0.5, f3_b=0.50), "fora")
        self.assertEqual(self.h("H1", "L100", f2_b=0.69, f3_b=0.49), "sem leitura")
        self.assertEqual(self.h("H1", "L100", f2_b=0.8, f3_b=0.6), "sem dado")
        self.assertEqual(self.h("H1", "L100", f2_b=0.8), "dentro")
        self.assertEqual(self.h("H1", "L10", f2_b=0.8, f3_b=0.1), "sem dado")

    def test_h2(self):
        self.assertEqual(self.h("H2", "L100", f3_b=0.5, f3_passo=3.0), "dentro")
        self.assertEqual(self.h("H2", "L100", f3_b=0.24, f3_passo=4.0), "fora")
        self.assertEqual(self.h("H2", "L100", f3_b=0.6, f3_passo=1.4), "fora")
        self.assertEqual(self.h("H2", "L100", f3_b=0.3, f3_passo=2.0), "sem leitura")
        self.assertEqual(self.h("H2", "L100", f3_b=0.6), "sem dado")
        self.assertEqual(self.h("H2", "L100", f3_b=0.1), "fora")

    def test_h3(self):
        self.assertEqual(self.h("H3", "L10", r3=0.5, s_compute_b=0.2), "dentro")
        self.assertEqual(self.h("H3", "L10", r3=0.29, s_compute_b=0.3), "fora")
        self.assertEqual(self.h("H3", "L10", r3=0.6, s_compute_b=0.09), "fora")
        self.assertEqual(self.h("H3", "L10", r3=0.4, s_compute_b=0.15), "sem leitura")
        self.assertEqual(self.h("H3", "L100", r3=0.10), "dentro")
        self.assertEqual(self.h("H3", "L100", r3=0.26), "fora")
        self.assertEqual(self.h("H3", "L100", r3=0.2), "sem leitura")
        self.assertEqual(self.h("H3", "LC", r3=0.2), "sem dado")

    def test_h4_h5(self):
        self.assertEqual(self.h("H4", "L100", f1_b=0.05), "dentro")
        self.assertEqual(self.h("H4", "L100", f1_b=0.019), "fora")
        self.assertEqual(self.h("H4", "L100", f1_b=0.03), "sem leitura")
        self.assertEqual(self.h("H5", "L10", r5=0.10), "dentro")
        self.assertEqual(self.h("H5", "L10", r5=0.029), "fora")
        self.assertEqual(self.h("H5", "L10", r5=0.05), "sem leitura")
        self.assertEqual(self.h("H5", "L100", r5=0.02), "dentro")
        self.assertEqual(self.h("H5", "L100", r5=0.051), "fora")
        self.assertEqual(self.h("H5", "L100", r5=0.03), "sem leitura")
        self.assertEqual(self.h("H4", "L100"), "sem dado")

    def test_h6_faixas_dos_restos(self):
        for r, (a, b) in ((1, (0.5625, 0.15625)), (2, (0.625, 0.3125)), (4, (0.75, 0.625))):
            self.assertAlmostEqual(E.faixas_h6(r)[0], a)
            self.assertAlmostEqual(E.faixas_h6(r)[1], b)
            self.assertFalse(E.h6_descritivo(r), r)
        self.assertTrue(E.h6_descritivo(6))
        self.assertTrue(E.h6_descritivo(7))
        self.assertFalse(E.h6_descritivo(5))
        self.assertEqual(self.h("H6", "L10", resto=1, r6=0.5625), "dentro")
        self.assertEqual(self.h("H6", "L10", resto=1, r6=0.15625), "fora")
        self.assertEqual(self.h("H6", "L10", resto=1, r6=0.3), "sem leitura")
        self.assertEqual(self.h("H6", "L10", resto=4, r6=0.74), "sem leitura")
        self.assertEqual(self.h("H6", "L10", resto=6, r6=0.9), "descritivo")
        self.assertEqual(self.h("H6", "L10", resto=0, r6=0.9), "descritivo")
        self.assertEqual(self.h("H6", "L10", resto=2), "sem dado")

    def test_controle_que_nao_permite_da_sem_prova(self):
        for ctl in ("SEM PROVA", "controle incompleto", "sem controle"):
            for h in ("H1", "H2", "H3", "H4", "H5", "H6"):
                self.assertEqual(self.h(h, "L100", ctl=ctl, f2_b=0.8, f3_b=0.1, r3=0.05, f1_b=0.1, r5=0.01, r6=0.9, f3_passo=5.0), "SEM PROVA", (ctl, h))
        self.assertEqual(self.h("H4", "L100", ctl="com ressalva", f1_b=0.1), "dentro")


class TestProcessa(Base):
    def test_tudo_bom_classes_e_relatorio(self):
        # L100: F2 domina o host (7/10), F3 baixa; L10 com serviço grande (compute+livro 0,5/B = 5%): só confere as classes calculadas
        self.tudo(dev=0.02, v100={"f2": 7.5, "f3": 1.5, "f1": 0.6, "s_compute": 0.2, "s_livro": 0.05, "s_end_trab": 0.05})
        r = self.roda()
        s10, s100 = r["series"]["L10"], r["series"]["L100"]
        self.assertEqual(s100["controle"]["classe"], "comprovada")
        self.assertEqual(s10["controle"]["classe"], "comprovada")
        self.assertEqual(len(s100["controle"]["processos"]), 6)
        # H1 dentro (F2/B = 7,5/9,7 >= 0,70), H4 dentro (0,6/9,7 = 6%)
        h = s100["hipoteses"]["medio"]
        self.assertEqual(h["H1"], "dentro", s100["textos"]["medio"]["medidas"])
        self.assertEqual(h["H4"], "dentro")
        self.assertEqual(h["H3"], "dentro")        # R3 = (0,2+0,05)/B pequeno
        self.assertEqual(s10["hipoteses"]["medio"]["H1"], "sem dado")
        self.assertEqual(s100["hipoteses"]["curto"]["H1"], "dentro")   # extrapolação sustentada (B por bloco igual entre textos)
        self.assertTrue(all(v["ok"] for v in s100["extrapolacao"]["textos"].values()))
        self.assertEqual(s100["textos"]["medio"]["descricao"]["n_pedidos"], 9)   # 3 processos L x 3 medidas
        self.assertEqual(s100["textos"]["curto"]["descricao"]["n_pedidos"], 5)
        # LC não rodou: sem dado, e o resultado não tem veredito geral
        self.assertEqual(r["series"]["LC"]["hipoteses"]["medio"]["H1"], "sem dado")
        md = E.relatorio(r)
        self.assertNotIn("PASSA", md.replace("sem PASSA", "").replace("nem PASSA", "").replace("declara PASSA", ""))
        self.assertIn("Controle do custo do instrumento: **comprovada**", md)
        self.assertIn("F0 do bloco 0 reutilizado", md)

    def test_f0_pendente_fica_fora_da_mediana_de_f0(self):
        self.tudo()
        d = self.roda()["series"]["L10"]["textos"]["medio"]["descricao"]
        self.assertAlmostEqual(d["fases_host_ms"]["f0"]["mediana"], 0.1)
        self.assertEqual(d["f0_bloco0_reutilizado_ms"]["n"], 9)     # 1 por pedido medido (bloco 0 reutilizado)

    def test_h6_usa_r6_dos_pedidos(self):
        self.tudo()
        h = self.roda()["series"]["L100"]["hipoteses"]
        self.assertEqual(h["medio"]["H6"], "dentro")       # r6 = 0,9 >= 0,625 (resto 2)
        self.assertEqual(h["curto"]["H6"], "dentro")

    def test_controle_com_ressalva_e_sem_prova(self):
        self.tudo(dev=0.05)
        r = self.roda()
        self.assertEqual(r["series"]["L100"]["controle"]["classe"], "com ressalva")
        self.assertNotEqual(r["series"]["L100"]["hipoteses"]["medio"]["H1"], "SEM PROVA")
        self.assertTrue(any("com ressalva" in x for x in r["series"]["L100"]["ressalvas"]))

    def test_controle_acima_de_10_sem_prova_em_todas_as_h_e_dados_preservados(self):
        self.tudo(dev=-0.12)
        r = self.roda()
        for s in ("L10", "L100"):
            self.assertEqual(r["series"][s]["controle"]["classe"], "SEM PROVA")
            for t in ("curto", "medio", "longo"):
                self.assertTrue(all(x == "SEM PROVA" for x in r["series"][s]["hipoteses"][t].values()), (s, t))
            self.assertGreater(r["series"][s]["textos"]["medio"]["descricao"]["n_blocos_cheios"], 0)   # bruto preservado

    def test_braco_d_que_imprimiu_ligado_e_controle_incompleto(self):
        self.controle("L100", v=None)
        # troca o D 01 por um D que imprimiu `ligado` e produziu perfil
        os.rename(os.path.join(self.saida, "celulas", "ctrl-1b-L100-medio-01-D"), os.path.join(self.saida, "descartada-01-D"))
        self.proc("L100", 1, "D", linha="ligado", perfil_json=True)
        for t in ("curto", "longo"):
            self.principal("L100", t)
        r = self.roda()
        c = r["series"]["L100"]["controle"]
        self.assertEqual(c["classe"], "controle incompleto", c)
        self.assertTrue(all(x == "SEM PROVA" for x in r["series"]["L100"]["hipoteses"]["medio"].values()))

    def test_braco_d_com_perfil_json_sem_linha_ligado_e_incompleto(self):
        self.controle("L100")
        grava(os.path.join(self.saida, "celulas", "ctrl-1b-L100-medio-04-D", "p1", "saida", "perfil-bloco.json"), perfil("1b", "medio", 3))
        c = self.roda()["series"]["L100"]["controle"]
        self.assertEqual(c["classe"], "controle incompleto")

    def test_ordem_trocada_e_par_truncado(self):
        self.controle("L10", ordem="DLDLDL")
        self.assertEqual(self.roda()["series"]["L10"]["controle"]["classe"], "controle incompleto")
        os.rename(os.path.join(self.saida, "celulas", "ctrl-1b-L10-medio-04-L"), os.path.join(self.saida, "descartada-04-L"))
        self.assertEqual(self.roda()["series"]["L10"]["controle"]["classe"], "controle incompleto")

    def test_processo_abortado_e_nao_rodou_no_orcamento(self):
        self.controle("L100")
        grava(os.path.join(self.saida, "celulas", "ctrl-1b-L100-medio-03-L", "p1", "rc.txt"), "1\n")
        self.assertEqual(self.roda()["series"]["L100"]["controle"]["classe"], "controle incompleto")
        self.controle("L10")
        grava(os.path.join(self.saida, "orcamento.json"), {"modelos": ["1b"], "nao_rodou": ["ctrl-1b-L10-medio-06-L: não coube"]})
        self.assertEqual(self.roda()["series"]["L10"]["controle"]["classe"], "controle incompleto")

    def test_ids_diferentes_da_5c_invalidam_a_serie(self):
        self.tudo()
        for pasta in ("ctrl-1b-L100-medio-02-L",):
            p = os.path.join(self.saida, "celulas", pasta, "p1", "saida", "medidas-1b.json")
            with open(p, encoding="utf-8") as fh:
                j = json.load(fh)
            for e in j["medidas"]:
                e["gerados"] = [x + 1 for x in e["gerados"]]
            grava(p, j)
        r = self.roda()["series"]["L100"]
        self.assertTrue(r["divergencias_5c_serie"])
        self.assertTrue(all(x == "SEM PROVA" for x in r["hipoteses"]["medio"].values()))
        self.assertNotEqual(self.roda()["series"]["L10"]["hipoteses"]["medio"]["H5"], "SEM PROVA")

    def test_servico_aberto_sai_das_estatisticas_de_servico(self):
        self.tudo(v100=None)
        for pasta in ("lastro-1b-L100-curto-quente",):
            grava(os.path.join(self.saida, "celulas", pasta, "p1", "saida", "perfil-bloco.json"), perfil("1b", "curto", 5, aberto=True))
        d = self.roda()["series"]["L100"]["textos"]["curto"]["descricao"]
        self.assertEqual(d["pedidos_servico_aberto"], 5)
        self.assertEqual(d["n_blocos_com_servico"], 0)
        self.assertTrue(d["n_blocos_cheios"] > 0)

    def test_cobertura_do_servico_ressalva(self):
        self.tudo(v100={"s_fora": 3.0})
        r = self.roda()["series"]["L100"]
        self.assertGreater(r["cobertura_servico_mediana"], 0.10)
        self.assertTrue(any("cobertura do serviço" in x for x in r["ressalvas"]))
        self.assertFalse(any("cobertura do serviço" in x for x in self.roda()["series"]["L10"]["ressalvas"]))

    def test_cobertura_do_host_ressalva(self):
        self.tudo()
        for pasta in ("lastro-1b-L10-curto-quente",):
            grava(os.path.join(self.saida, "celulas", pasta, "p1", "saida", "perfil-bloco.json"), perfil("1b", "curto", 5, cob=0.04))
        self.assertTrue(any("cobertura do host em curto" in x for x in self.roda()["series"]["L10"]["ressalvas"]))

    def test_pedido_invalido_pelo_binario_invalida_so_a_celula(self):
        self.tudo()
        grava(os.path.join(self.saida, "celulas", "lastro-1b-L10-longo-quente", "p1", "saida", "perfil-bloco.json"), perfil("1b", "longo", 5, valido=False))
        r = self.roda()["series"]["L10"]
        self.assertEqual(r["textos"]["longo"]["status"], "SEM PROVA")
        self.assertEqual(r["textos"]["medio"]["status"], "OK")
        self.assertEqual(r["textos"]["curto"]["status"], "OK")

    def test_plano_de_blocos_errado_invalida_a_celula(self):
        self.tudo()
        j = perfil("1b", "curto", 5, plano=[8] * 11)
        grava(os.path.join(self.saida, "celulas", "lastro-1b-L100-curto-quente", "p1", "saida", "perfil-bloco.json"), j)
        r = self.roda()["series"]["L100"]["textos"]["curto"]
        self.assertEqual(r["status"], "SEM PROVA")
        self.assertTrue(any("plano" in x for x in r["motivos"]))

    def test_celula_sem_a_linha_perfil_bloco_invalida_so_ela(self):
        self.tudo()
        grava(os.path.join(self.saida, "celulas", "lastro-1b-L10-curto-quente", "p1", "stdout.txt"), "sem linha\n")
        r = self.roda()["series"]["L10"]["textos"]
        self.assertEqual(r["curto"]["status"], "SEM PROVA")
        self.assertEqual(r["longo"]["status"], "OK")

    def test_lc_e_descritiva(self):
        self.tudo()
        self.principal("LC", "curto", n=3)
        r = self.roda()["series"]["LC"]
        self.assertEqual(r["controle"]["classe"], "sem controle")
        self.assertEqual(r["hipoteses"]["curto"]["H1"], "descritivo")
        self.assertTrue(any("LC sem controle" in x for x in r["ressalvas"]))

    def test_integridade_quebrada_vira_sem_prova_sem_recusar(self):
        self.tudo()
        self.manifesto(estado="FALHA")
        r = self.roda()
        self.assertTrue(r["integridade_problemas"])
        for s in ("L10", "L100"):
            self.assertTrue(all(x == "SEM PROVA" for x in r["series"][s]["hipoteses"]["medio"].values()))
            c = r["series"][s]["controle"]
            self.assertEqual(c["classe"], "SEM PROVA")                      # a classe EFETIVA do custo
            self.assertEqual(c["classe_calculada"], "comprovada")           # medidas preservadas como descrição, com o motivo
            self.assertIn("integridade quebrada", c["motivo"])
            self.assertIsNotNone(c["desvio"])
        self.assertIn("classe calculada antes do bloqueio", E.relatorio(r))

    def test_smoke_reprovado_bloqueia_so_a_serie(self):
        self.tudo()
        self.smoke("L100", False)
        r = self.roda()["series"]
        self.assertTrue(all(x == "SEM PROVA" for x in r["L100"]["hipoteses"]["medio"].values()))
        self.assertNotEqual(r["L10"]["hipoteses"]["medio"]["H5"], "SEM PROVA")

    # ---- rodada 139
    def extrapola(self, b_curto_f2, b_longo_f2):
        for s in ("L10", "L100"):
            self.controle(s)
            self.principal(s, "curto", v={"f2": b_curto_f2})
            self.principal(s, "longo", v={"f2": b_longo_f2})
        return self.roda()["series"]["L100"]

    def test_extrapolacao_por_texto_94_100_106_sustenta_as_duas(self):
        r = self.extrapola(6.4, 7.6)       # B = 3,0 + f2: 9,4 / 10,0 / 10,6 (6% do médio cada)
        self.assertAlmostEqual(r["extrapolacao"]["b_medio_ms"], 10.0)
        self.assertTrue(r["extrapolacao"]["textos"]["curto"]["ok"] and r["extrapolacao"]["textos"]["longo"]["ok"])
        self.assertAlmostEqual(r["extrapolacao"]["textos"]["curto"]["desvio"], 0.06)
        self.assertNotEqual(r["hipoteses"]["curto"]["H1"], "SEM PROVA")      # classificado (F2/B = 68%: faixa intermediária)
        self.assertNotEqual(r["hipoteses"]["longo"]["H1"], "SEM PROVA")

    def test_extrapolacao_94_100_111_bloqueia_so_o_longo(self):
        r = self.extrapola(6.4, 8.1)       # longo a 11,0% do médio
        self.assertTrue(r["extrapolacao"]["textos"]["curto"]["ok"])
        self.assertFalse(r["extrapolacao"]["textos"]["longo"]["ok"])
        self.assertNotEqual(r["hipoteses"]["curto"]["H1"], "SEM PROVA")
        self.assertTrue(all(x == "SEM PROVA" for x in r["hipoteses"]["longo"].values()))
        self.assertEqual(r["hipoteses"]["medio"]["H1"], "dentro")
        self.assertIn("BLOQUEIA", E.relatorio(self.roda()))

    def test_extrapolacao_limite_exato_de_10_por_cento_sustenta(self):
        r = self.extrapola(7.0, 8.0)       # longo com B = 11,0 ms: exatamente 10% do médio, inclusivo
        self.assertTrue(r["extrapolacao"]["textos"]["longo"]["ok"])

    def test_extrapolacao_sem_b_medio_valido_nenhuma_extensao(self):
        for s in ("L10", "L100"):
            self.principal(s, "curto")
            self.principal(s, "longo")
        r = self.roda()["series"]["L100"]
        self.assertIsNone(r["extrapolacao"]["b_medio_ms"])
        self.assertFalse(any(v["ok"] for v in r["extrapolacao"]["textos"].values()))
        self.assertTrue(all(x == "SEM PROVA" for t in ("curto", "longo") for x in r["hipoteses"][t].values()))

    def test_ordem_efetiva_invertida_so_no_registro_e_controle_incompleto(self):
        self.controle("L10", registro=False)
        self.registra_ordem("L10", sequencia=[2, 1, 3, 4, 5, 6])      # mesmas pastas e medições, só o registro cronológico trocado
        c = self.roda()["series"]["L10"]["controle"]
        self.assertEqual(c["classe"], "controle incompleto", c)
        self.assertIn("ordem efetiva", c["motivo"])

    def test_ordem_correta_mantem_o_controle_e_registro_ausente_ou_truncado_nao(self):
        self.controle("L10")
        self.assertEqual(self.roda()["series"]["L10"]["controle"]["classe"], "comprovada")
        with open(os.path.join(self.saida, "ordem.txt"), encoding="utf-8") as fh:
            linhas = fh.read().splitlines()
        grava(os.path.join(self.saida, "ordem.txt"), "\n".join(linhas[:5]) + "\n")
        c = self.roda()["series"]["L10"]["controle"]
        self.assertEqual(c["classe"], "controle incompleto")
        self.assertIn("truncado", c["motivo"])
        grava(os.path.join(self.saida, "ordem.txt"), "")
        self.assertEqual(self.roda()["series"]["L10"]["controle"]["classe"], "controle incompleto")

    def test_ordem_registra_braco_ou_rc_incompativel(self):
        self.controle("L10", registro=False)
        self.registra_ordem("L10", rc=1)
        self.assertEqual(self.roda()["series"]["L10"]["controle"]["classe"], "controle incompleto")

    def test_divergencia_5c_so_em_um_processo_d_bloqueia_a_serie_inteira(self):
        self.tudo()
        p = os.path.join(self.saida, "celulas", "ctrl-1b-L100-medio-04-D", "p1", "saida", "medidas-1b.json")
        with open(p, encoding="utf-8") as fh:
            j = json.load(fh)
        for e in j["medidas"]:
            e["gerados"] = [x + 1 for x in e["gerados"]]
        grava(p, j)
        r = self.roda()["series"]
        c = r["L100"]["controle"]
        self.assertEqual(c["classe"], "SEM PROVA")
        self.assertEqual(c["classe_calculada"], "comprovada")
        self.assertIsNotNone(c["desvio"])
        for t in ("curto", "medio", "longo"):
            self.assertTrue(all(x == "SEM PROVA" for x in r["L100"]["hipoteses"][t].values()), t)
        self.assertTrue(any(x.get("braco") == "D" for x in r["L100"]["divergencias_5c_serie"]))
        self.assertEqual(r["L10"]["controle"]["classe"], "comprovada")

    def altera_so_aquecimento(self, pasta, alvo="gerados"):
        p = os.path.join(self.saida, "celulas", pasta, "p1", "saida", "medidas-1b.json")
        with open(p, encoding="utf-8") as fh:
            j = json.load(fh)
        ent = [e for e in j["medidas"] if e.get("descartada")]
        self.assertEqual(len(ent), 1)
        ent[0][alvo] = [x + 1 for x in ent[0][alvo]]
        grava(p, j)

    def test_id_alterado_so_no_aquecimento_bloqueia_a_classificacao(self):
        self.tudo()
        self.altera_so_aquecimento("lastro-1b-L10-curto-quente")
        r = self.roda()["series"]["L10"]
        self.assertTrue(r["divergencias_5c_serie"] and all(x["aquecimento"] for x in r["divergencias_5c_serie"]))
        self.assertEqual(r["controle"]["classe"], "SEM PROVA")
        self.assertTrue(all(x == "SEM PROVA" for x in r["hipoteses"]["medio"].values()))
        self.assertEqual(self.roda()["series"]["L100"]["hipoteses"]["medio"]["H1"], "dentro")
        # o tempo do aquecimento segue fora das medianas: 5 pedidos medidos no curto
        self.assertEqual(r["textos"]["curto"]["descricao"]["n_pedidos"], 5)

    def test_id_do_prompt_alterado_so_no_aquecimento_tambem_bloqueia(self):
        self.tudo()
        self.altera_so_aquecimento("ctrl-1b-L100-medio-02-L", alvo="ids")
        r = self.roda()["series"]["L100"]
        self.assertTrue(any(x["tipo"] == "ids do prompt" for x in r["divergencias_5c_serie"]))
        self.assertEqual(r["controle"]["classe"], "SEM PROVA")

    def test_referencia_ausente_bloqueia_com_igualdade_nunca_presumida(self):
        self.tudo()
        ref = os.path.join(self.refs, "ref5c-1b.json")
        grava(ref, {"casos": []})
        self.shas["ref5c-1b"] = E.sha_arq(ref)
        self.manifesto()                                   # manifesto coerente com o arquivo: o bloqueio vem da ausência, não do sha
        r = self.roda()
        for s in ("L10", "L100"):
            self.assertEqual(r["series"][s]["controle"]["classe"], "SEM PROVA")
            self.assertIn("referência 5c indisponível", r["series"][s]["controle"]["motivo"])
            self.assertTrue(all(x == "SEM PROVA" for x in r["series"][s]["hipoteses"]["medio"].values()))

    def test_referencia_com_sha_divergente_do_manifesto_bloqueia(self):
        self.tudo()
        ref = os.path.join(self.refs, "ref5c-1b.json")
        with open(ref, encoding="utf-8") as fh:
            j = json.load(fh)
        j["observacao"] = "arquivo diferente do registrado"
        grava(ref, j)
        r = self.roda()
        self.assertTrue(r["referencia_problemas"])
        self.assertIn("sha", r["referencia_problemas"][0])
        for s in ("L10", "L100"):
            self.assertEqual(r["series"][s]["controle"]["classe"], "SEM PROVA")
            self.assertTrue(all(x == "SEM PROVA" for x in r["series"][s]["hipoteses"]["medio"].values()))

    def test_referencia_integra_e_ids_iguais_mantem_o_resultado_positivo(self):
        self.tudo()
        r = self.roda()
        self.assertEqual(r["referencia_problemas"], [])
        self.assertEqual(r["series"]["L100"]["controle"]["classe"], "comprovada")
        self.assertEqual(r["series"]["L100"]["hipoteses"]["medio"]["H1"], "dentro")

    def test_perfil_invalido_de_um_L_do_controle_so_marca_a_celula_medio(self):
        for s in ("L10", "L100"):
            for nn, b in enumerate("DLLDDL", 1):
                self.proc(s, nn, b, ms_pos=5.0, valido=(nn != 3))        # o L 03 do controle traz um pedido inválido do binário
            self.registra_ordem(s)
            for t in ("curto", "longo"):
                self.principal(s, t)
        r = self.roda()["series"]["L100"]
        self.assertEqual(r["controle"]["classe"], "comprovada", r["controle"])
        proc3 = [p for p in r["controle"]["processos"] if p["nn"] == 3][0]
        self.assertEqual(proc3["status_temporal"], "OK")
        self.assertEqual(proc3["status_perfil"], "SEM PROVA")
        self.assertEqual(r["textos"]["medio"]["status"], "SEM PROVA")
        self.assertTrue(all(x == "SEM PROVA" for x in r["hipoteses"]["medio"].values()))
        self.assertTrue(any("inválido pelo próprio binário" in x for x in r["textos"]["medio"]["motivos"]))
        self.assertEqual(r["hipoteses"]["curto"]["H1"], "dentro")        # curto e longo classificados (B médio vem dos processos L de perfil válido)
        self.assertEqual(r["hipoteses"]["longo"]["H1"], "dentro")
        self.assertIn("perfil SEM PROVA", E.relatorio(self.roda()))

    def test_h2_usa_o_passo_de_decode_do_braco_D(self):
        for s in ("L10", "L100"):
            for nn, b in enumerate("DLLDDL", 1):
                self.proc(s, nn, b, ms_pos=5.0, decode_d=20.0, decode_l=8.0)
            self.registra_ordem(s)
            for t in ("curto", "longo"):
                self.principal(s, t)
        r = self.roda()["series"]["L100"]
        self.assertAlmostEqual(r["passo_d_ms"], 20.0)
        self.assertAlmostEqual(r["textos"]["medio"]["f3_passo"], 2.4 / 20.0)      # não 2,4/8 (passo do L)
        self.assertNotAlmostEqual(r["textos"]["medio"]["f3_passo"], 2.4 / 8.0)
        self.assertEqual(r["hipoteses"]["medio"]["H2"], "fora")                   # F3/passo(D) = 0,12 < 1,5


    # ---- rodada 140
    def medidas_json(self, pasta):
        return os.path.join(self.saida, "celulas", pasta, "p1", "saida", "medidas-1b.json")

    def tira_aquecimento(self, pasta):
        p = self.medidas_json(pasta)
        with open(p, encoding="utf-8") as fh:
            j = json.load(fh)
        antes = len(j["medidas"])
        j["medidas"] = [e for e in j["medidas"] if not e.get("descartada")]
        self.assertEqual(len(j["medidas"]), antes - 1)
        grava(p, j)

    def test_sem_aquecimento_so_num_d_e_controle_incompleto_mesmo_com_perfil_no_l(self):
        self.tudo()
        self.tira_aquecimento("ctrl-1b-L100-medio-01-D")
        r = self.roda()["series"]["L100"]
        self.assertEqual(r["controle"]["classe"], "controle incompleto")
        self.assertEqual(self.roda()["series"]["L10"]["controle"]["classe"], "comprovada")

    def test_sem_aquecimento_so_num_l_com_perfil_presente_e_controle_incompleto(self):
        self.tudo()
        self.assertTrue(os.path.exists(os.path.join(self.saida, "celulas", "ctrl-1b-L10-medio-02-L", "p1", "saida", "perfil-bloco.json")))
        self.tira_aquecimento("ctrl-1b-L10-medio-02-L")
        r = self.roda()["series"]["L10"]
        self.assertEqual(r["controle"]["classe"], "controle incompleto")
        self.assertTrue(any("aquecimento" in x for x in r["controle"]["processos"][1]["motivos_temporais"]))

    def test_medianas_do_controle_usam_so_as_n_medidas(self):
        self.tudo()
        p = self.medidas_json("ctrl-1b-L100-medio-01-D")
        with open(p, encoding="utf-8") as fh:
            j = json.load(fh)
        meds = [e["prefill_ms"] for e in j["medidas"] if not e.get("descartada")]
        for e in j["medidas"]:
            if e.get("descartada"):
                e["prefill_ms"] = 99999.0
        grava(p, j)
        r = self.roda()["series"]["L100"]["controle"]
        self.assertEqual(len(meds), 3)
        self.assertAlmostEqual(r["processos"][0]["prefill_ms_mediana"], sorted(meds)[1])

    def test_lc_com_manifesto_falha_fica_sem_prova_com_medidas_preservadas(self):
        self.tudo()
        self.principal("LC", "curto", n=3)
        self.manifesto(estado="FALHA")
        r = self.roda()["series"]["LC"]
        self.assertTrue(all(x == "SEM PROVA" for x in r["hipoteses"]["curto"].values()))
        self.assertTrue(any("integridade" in x for x in r["bloqueios"]))
        self.assertTrue(r["textos"]["curto"]["medidas"])
        self.assertNotIn("tudo SEM PROVA", E.relatorio(self.roda()))

    def test_lc_com_smoke_reprovado_fica_sem_prova(self):
        self.tudo()
        self.principal("LC", "curto", n=3)
        self.smoke("LC", False)
        r = self.roda()["series"]
        self.assertTrue(all(x == "SEM PROVA" for x in r["LC"]["hipoteses"]["curto"].values()))
        self.assertTrue(r["LC"]["textos"]["curto"]["medidas"])
        self.assertNotEqual(r["L10"]["hipoteses"]["medio"]["H5"], "SEM PROVA")

    def test_lc_integra_com_ids_divergentes_segue_descritiva_com_divergencia_reportada(self):
        self.tudo()
        d = self.principal("LC", "curto", n=3)
        p = os.path.join(d, "p1", "saida", "medidas-1b.json")
        with open(p, encoding="utf-8") as fh:
            j = json.load(fh)
        for e in j["medidas"]:
            e["gerados"] = [x + 1 for x in e["gerados"]]
        grava(p, j)
        r = self.roda()["series"]["LC"]
        self.assertEqual(r["hipoteses"]["curto"]["H1"], "descritivo")
        self.assertTrue(r["divergencias_5c_serie"])
        self.assertEqual(r["bloqueios"], [])

    def test_referencia_so_do_texto_curto_ausente_bloqueia_a_serie_com_o_texto_no_motivo(self):
        self.tudo()
        ref = os.path.join(self.refs, "ref5c-1b.json")
        with open(ref, encoding="utf-8") as fh:
            j = json.load(fh)
        j["casos"] = [c for c in j["casos"] if c.get("nome") != "curto"]
        grava(ref, j)
        self.shas["ref5c-1b"] = E.sha_arq(ref)
        self.manifesto()
        r = self.roda()["series"]
        for s in ("L10", "L100"):
            self.assertEqual(r[s]["controle"]["classe"], "SEM PROVA")
            self.assertTrue(any("curto" in x and "referência 5c indisponível" in x for x in r[s]["bloqueios"]))
            self.assertTrue(all(x == "SEM PROVA" for x in r[s]["hipoteses"]["medio"].values()))

    def test_celula_curta_nao_rodada_por_orcamento_nao_bloqueia_o_medio(self):
        self.tudo()
        for s in ("L10", "L100"):
            os.rename(os.path.join(self.saida, "celulas", "lastro-1b-%s-curto-quente" % s), os.path.join(self.saida, "nao-rodou-%s-curto" % s))
        r = self.roda()["series"]
        for s in ("L10", "L100"):
            self.assertEqual(r[s]["controle"]["classe"], "comprovada")
            self.assertEqual(r[s]["bloqueios"], [])
        self.assertEqual(r["L100"]["hipoteses"]["medio"]["H1"], "dentro")


class TestRecusa(Base):
    def test_potencia_mudou_recusa(self):
        self.tudo()
        self.manifesto(potencia="potencia inicio=370.00_W fim=300.00_W FALHA\n")
        with self.assertRaises(E.Recusa) as cm:
            self.roda()
        self.assertIn("potência", str(cm.exception))

    def test_potencia_ausente_ou_ilegivel_recusa(self):
        self.tudo()
        self.manifesto(potencia="")
        with self.assertRaises(E.Recusa):
            self.roda()
        self.manifesto(potencia="potencia 370.00_W ilegivel OK\n")
        with self.assertRaises(E.Recusa):
            self.roda()

    def test_ambiente_sem_potencia_fim_recusa(self):
        self.tudo()
        grava(os.path.join(self.saida, "ambiente.txt"), "modelo=1b\npotencia_inicio=370.00_W\npotencia_fim=300.00_W\n")
        with self.assertRaises(E.Recusa):
            self.roda()

    def test_sem_a_linha_perfil_bloco_em_nenhum_processo_recusa(self):
        self.tudo()
        for raiz, _, arqs in os.walk(os.path.join(self.saida, "celulas")):
            if "stdout.txt" in arqs:
                grava(os.path.join(raiz, "stdout.txt"), "só integridade\n")
        with self.assertRaises(E.Recusa) as cm:
            self.roda()
        self.assertIn("perfil-bloco", str(cm.exception))

    def test_main_recusa_com_codigo_2_e_registra(self):
        self.tudo()
        self.manifesto(potencia="")
        err = io.StringIO()
        with redirect_stderr(err):
            rc = E.main(["--saida", self.saida, "--ids", self.ids, "--refs", self.refs, "--textos", self.textos])
        self.assertEqual(rc, 2)
        with open(os.path.join(self.saida, "instrumento-resultado.json"), encoding="utf-8") as fh:
            self.assertTrue(json.load(fh)["recusado"])
        with open(os.path.join(self.saida, "instrumento-resultado.md"), encoding="utf-8") as fh:
            self.assertIn("RECUSADO", fh.read())

    def test_orcamento_sem_modelo_unico_recusa(self):
        self.tudo()
        grava(os.path.join(self.saida, "orcamento.json"), {"modelos": ["1b", "3b"]})
        with self.assertRaises(E.Recusa):
            self.roda()

    def test_pasta_inexistente(self):
        self.assertEqual(E.main(["--saida", os.path.join(self.d, "nao-existe")]), 2)

    def test_main_grava_json_estrito_e_md(self):
        self.tudo()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = E.main(["--saida", self.saida, "--ids", self.ids, "--refs", self.refs, "--textos", self.textos])
        self.assertEqual(rc, 0, buf.getvalue())
        with open(os.path.join(self.saida, "instrumento-resultado.json"), encoding="utf-8") as fh:
            j = json.load(fh, parse_constant=lambda c: self.fail("constante não finita: " + c))
        self.assertTrue(j["descritivo"])
        self.assertIn("controle L10 comprovada", buf.getvalue())
        with open(os.path.join(self.saida, "instrumento-resultado.md"), encoding="utf-8") as fh:
            self.assertIn("| medio |", fh.read())


if __name__ == "__main__":
    unittest.main(verbosity=1)
