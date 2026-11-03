"""Teste do analise.py com uma árvore de resultados sintética montada a partir de saídas reais do A1.4 (formato idêntico ao do harness N).
Cada caso monta a árvore com os números desejados, roda o analise.py como processo e confere o resultado por célula e os controles."""
import os, re, shutil, subprocess, sys, tempfile, unittest

AQUI = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(AQUI, "..", ".."))
TPL = os.path.join(ROOT, "qual-runs", "A1-MEDIDA-20261101", "runs", "rep1")
PY = sys.executable
CFGS = {1: ["C", "N0", "I"], 2: ["N0", "A1w", "A1m"]}


def serie(path):
    with open(path) as f:
        return [float(x) for x in f.read().split()]


def escreve(path, texto):
    with open(path, "w") as f:
        f.write(texto)


def le(path):
    with open(path) as f:
        return f.read()


def monta(raiz, etapa, ajuste, estragos=None, integridade="inicio OK\nfim OK\n"):
    """ajuste[(cfg)] = dict(L1=delta, L2=delta, L3=delta, L4=delta) em ms somados à série modelo; estragos[(mod,rep,cfg,bl)] = função (texto)->texto."""
    estragos = estragos or {}
    tent, amo = [], []
    t0, pid = 1000.0, 5000
    for rep in range(1, 6):
        for mod in ("1b", "3b"):
            for cfg in CFGS[etapa]:
                for bl in ("m2a-b4", "a1"):
                    d = f"{raiz}/e{etapa}/rep{rep}/{mod}/{cfg}/{bl}"
                    os.makedirs(f"{d}/ms")
                    t = le(f"{TPL}/{mod}/{bl}/saida.txt")
                    if cfg == "A1w":
                        pols_ = ("resident 100", "lfu_cpu 10") if bl == "m2a-b4" else ("motor_cheio 100", "motor_evento 10")
                        n_ = lambda pol: 1 if pol.startswith("resident") else 8
                        t += "".join(f"CONTROLE A1w {pl}%: {n_(pl)} participantes, núcleos físicos distintos: ok\n" for pl in pols_)
                    if cfg == "A1m" and bl == "a1":
                        t += "CONTROLE A1m motor_cheio 100%: thread principal em 17\nCONTROLE A1m motor_evento 10%: thread principal em 17\n"
                    t = estragos.get((mod, rep, cfg, bl), lambda x: x)(t)
                    escreve(f"{d}/saida.txt", t)
                    aj = ajuste.get(cfg, {})
                    for f in os.listdir(f"{TPL}/{mod}/{bl}/ms"):
                        m = re.match(r"(?:m2a-ms|a1-motor-ms)-(\w+)-(\d+)\.txt", f)
                        pol = m[1]
                        k = {"motor_evento": "L1", "lfu_cpu": "L2", "resident": "L3" if bl == "m2a-b4" else "L1res", "motor_cheio": "L4"}[pol]
                        delta = aj.get(k, 0.0)
                        escreve(f"{d}/ms/{f}", "".join(f"{x + delta:.4f}\n" for x in serie(f"{TPL}/{mod}/{bl}/ms/{f}")))
                    with open(f"{d}/freq.txt", "w") as fq:
                        for s in range(5):
                            fq.write(f"{s}.0 chamador 0 5500000\n{s}.0 trabalhadores 2 5500000\n")
                    pid += 1
                    tent.append((d, pid, t0, t0 + 10))
                    amo.append((t0 + 5, pid, f"{ROOT}/qual-runs/A1-DIAG10-20261103/bin/N/a1-motor"))
                    t0 += 20
    with open(f"{raiz}/carga-e{etapa}.log", "w") as f:
        for d, pid, a, b in tent:
            f.write(f"TENTATIVA {d} n=1 pid={pid} inicio={a} fim={b} rc_harness=0 rc=0\n")
    with open(f"{raiz}/gpu-monitor-e{etapa}.log", "w") as f:
        for ts, pid, nm in amo:
            f.write(f"{ts} ok GPU-x, {pid}, {nm}, 1000 MiB;\n")
        f.write(f"{t0 + 100} ok \n")
    escreve(f"{raiz}/integridade-e{etapa}.txt", integridade)
    return tent


def roda(raiz):
    r = subprocess.run(["taskset", "-c", "16-27", PY, "-I", "-B", f"{AQUI}/analise.py", raiz], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def celula(txt, mod):
    m = re.search(rf"^  {mod} 10%: (.*)$", txt, re.M)
    return m[1] if m else None


class Analise(unittest.TestCase):
    def setUp(self):
        self.raiz = tempfile.mkdtemp(prefix="analise-", dir="/tmp/claude-1000/-home-brenoperucchi-Devs-lastro-rust/46cdbe1b-904f-422b-9a0c-ab7628b69d27/scratchpad")

    PAI = "/tmp/claude-1000/-home-brenoperucchi-Devs-lastro-rust/46cdbe1b-904f-422b-9a0c-ab7628b69d27/scratchpad"

    def limpa(self, raiz):
        """Apaga só uma pasta analise-* criada por este teste, pelo nome relativo a partir do pai literal."""
        base = os.path.basename(raiz)
        assert base.startswith("analise-") and os.path.dirname(raiz) == self.PAI
        os.chdir(self.PAI)
        shutil.rmtree(base)

    def tearDown(self):
        self.limpa(self.raiz)

    def e1(self, aj, **kw):
        monta(self.raiz, 1, aj, **kw)
        return roda(self.raiz)

    def test_etapa1_reproduz(self):
        rc, txt = self.e1({"C": {}, "N0": {}, "I": {}})
        self.assertEqual(rc, 0, txt)
        self.assertIn("PASSO 0 OK", celula(txt, "1b") or "", txt)
        self.assertIn("PASSO 0 OK", celula(txt, "3b") or "", txt)
        self.assertIn("reprova", txt)   # controles negativos presentes

    def test_etapa1_c_sem_excesso(self):
        rc, txt = self.e1({"C": {"L1": -0.6}, "N0": {}, "I": {}})
        self.assertEqual(celula(txt, "1b"), "NAO_REPRODUZIDO", txt)

    def test_etapa1_n_perde_o_excesso(self):
        rc, txt = self.e1({"C": {}, "N0": {"L1": -0.45}, "I": {}})
        self.assertEqual(celula(txt, "1b"), "SEM_CONCLUSAO_BUILD", txt)

    def test_logits_reprovados_suspendem_a_celula(self):
        ruim = lambda t: re.sub(r"(motor_evento\s+10%.*\|\s*)\d\.\de-\d", r"\g<1>2.0e-4", t, count=1)
        rc, txt = self.e1({}, estragos={("1b", 3, "N0", "a1"): ruim})
        self.assertIn("SEM CONCLUSÃO", celula(txt, "1b") or "", txt)
        self.assertIn("PASSO 0 OK", celula(txt, "3b") or "", txt)

    def test_processo_alheio_na_gpu_descarta_o_bloco(self):
        raiz = self.raiz
        tent = monta(raiz, 1, {})
        d, pid, a, b = tent[7]
        with open(f"{raiz}/gpu-monitor-e1.log", "a") as f:
            f.write(f"{(a + b) / 2} ok GPU-x, 999999, /usr/bin/outro, 500 MiB;\n")
        rc, txt = roda(raiz)
        self.assertIn("descartadas por carga: 1", txt)
        self.assertTrue(any(celula(txt, m) and "SEM CONCLUSÃO" in celula(txt, m) for m in ("1b", "3b")), txt)

    def test_frequencia_fora_de_2_por_cento_reprova(self):
        raiz = self.raiz
        monta(raiz, 1, {})
        with open(f"{raiz}/e1/rep2/1b/N0/a1/freq.txt", "w") as f:
            for s in range(5):
                f.write(f"{s}.0 chamador 0 5000000\n{s}.0 trabalhadores 2 5500000\n")
        rc, txt = roda(raiz)
        self.assertIn("0c reprova 1b rep2 N0", txt)
        self.assertIn("SEM CONCLUSÃO", celula(txt, "1b") or "")

    def test_contadores_fora_do_limite_na_etapa1(self):
        mexe = lambda t: re.sub(r"(motor_evento\s+10% \| needed\s+\d+ hits\s+\d+ cpu\s+\d+ cópias\s+)(\d+)", lambda m: m[1] + str(int(m[2]) + 300), t, count=1)
        rc, txt = self.e1({}, estragos={("1b", 4, "N0", "a1"): mexe})
        self.assertIn("SEM_CONCLUSAO_TRABALHO", celula(txt, "1b") or "", txt)

    def test_etapa2_sustentada_e_nao_sustentada(self):
        raiz = self.raiz
        monta(raiz, 2, {"N0": {}, "A1w": {"L1": -0.4}, "A1m": {}})
        for mod in ("1b", "3b"):
            escreve(f"{raiz}/d-etapa1-{mod}.txt", "status OK\nd_mediana 0.5\n")
        rc, txt = roda(raiz)
        self.assertEqual(rc, 0, txt)
        self.assertEqual(celula(txt, "1b"), "{'w': 'SUSTENTADA', 'm': 'NAO_SUSTENTADA'}", txt)

    def test_etapa2_controle_de_afinidade_falhou_invalida_o_bloco(self):
        raiz = self.raiz
        monta(raiz, 2, {"N0": {}, "A1w": {"L1": -0.4}, "A1m": {}},
              estragos={("1b", 2, "A1w", "a1"): lambda t: t.replace("distintos: ok", "FALHOU", 1)})
        for mod in ("1b", "3b"):
            escreve(f"{raiz}/d-etapa1-{mod}.txt", "status OK\nd_mediana 0.5\n")
        rc, txt = roda(raiz)
        self.assertIn("controle da A1w", txt)
        self.assertIn("SEM CONCLUSÃO", celula(txt, "1b") or "", txt)

    def test_integridade_invalida_ou_ausente_nao_conclui(self):
        for integ in ("inicio OK\nfim INVALIDA\n", "inicio OK\n", ""):
            raiz = tempfile.mkdtemp(prefix="analise-", dir=os.path.dirname(self.raiz))
            monta(raiz, 1, {}, integridade=integ)
            rc, txt = roda(raiz)
            self.assertIn("SEM CONCLUSÃO (integridade", celula(txt, "1b") or "", (integ, txt))
            self.assertNotIn("PASSO 0 OK", txt)
            self.assertTrue(le(f"{raiz}/d-etapa1-1b.txt").startswith("status INVÁLIDA"))
            self.addCleanup(self.limpa, raiz)

    def test_etapa1_reprovada_nao_libera_a_etapa2(self):
        # C sem excesso (NAO_REPRODUZIDO) e N0 com excesso: o registro fica "status NAO_REPRODUZIDO"; a etapa 2 com A1w que reduz L1 não pode sustentar H1
        raiz = self.raiz
        monta(raiz, 1, {"C": {"L1": -0.6}, "N0": {}, "I": {}})
        rc, txt = roda(raiz)
        self.assertEqual(le(f"{raiz}/d-etapa1-1b.txt").strip(), "status NAO_REPRODUZIDO")
        shutil.rmtree(f"{raiz}/e1")
        monta(raiz, 2, {"N0": {}, "A1w": {"L1": -0.4}, "A1m": {}})
        rc, txt = roda(raiz)
        self.assertIn("a etapa 1 não aprovou", celula(txt, "1b") or "", txt)
        self.assertNotIn("SUSTENTADA", (celula(txt, "1b") or "") + (celula(txt, "3b") or ""))

    def test_etapa2_sem_registro_da_etapa1_nao_conclui(self):
        raiz = self.raiz
        monta(raiz, 2, {"N0": {}, "A1w": {"L1": -0.4}, "A1m": {}})
        rc, txt = roda(raiz)     # nenhum registro da etapa 1 foi escrito
        self.assertIn("a etapa 1 não aprovou", celula(txt, "1b") or "", txt)

    def test_referencia_antiga_e_apagada_ao_reanalisar_a_etapa1(self):
        raiz = self.raiz
        monta(raiz, 1, {})
        roda(raiz)
        self.assertTrue(le(f"{raiz}/d-etapa1-1b.txt").startswith("status OK"))
        shutil.rmtree(f"{raiz}/e1")
        monta(raiz, 1, {"C": {"L1": -0.6}})
        roda(raiz)
        self.assertFalse(le(f"{raiz}/d-etapa1-1b.txt").startswith("status OK"))

    def test_etapa2_l4_movido_e_base(self):
        raiz = self.raiz
        monta(raiz, 2, {"N0": {}, "A1w": {}, "A1m": {"L4": 0.09, "L1": 0.09}})
        for mod in ("1b", "3b"):
            escreve(f"{raiz}/d-etapa1-{mod}.txt", "status OK\nd_mediana 0.5\n")
        rc, txt = roda(raiz)
        self.assertIn("SEM_CONCLUSAO_BASE", celula(txt, "1b") or "", txt)


if __name__ == "__main__":
    unittest.main()
