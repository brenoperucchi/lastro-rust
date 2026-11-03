"""Regressão (janela de 3/11): no binário C do A1.4 a thread de serviço do a1-motor não tem nome e herda o do executável; a regra 0c nunca achava o chamador."""
import io, os, sys, unittest
from unittest import mock
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import amostrador as A


def stat(ticks, cpu):
    # campos após ")": estado, ..., utime no índice 11, stime no 12, processador no 36
    f = ["S"] + ["0"] * 40
    f[11], f[12], f[36] = str(ticks), "0", str(cpu)
    return "1 (a1-motor) " + " ".join(f)


def amostra_com(threads, pid=100, motor=True):
    # threads: tid -> (comm, ticks, cpu)
    def fake_open(path, *a, **k):
        for tid, (comm, ticks, cpu) in threads.items():
            if path == f"/proc/{pid}/task/{tid}/comm":
                return io.StringIO(comm + "\n")
            if path == f"/proc/{pid}/task/{tid}/stat":
                return io.StringIO(stat(ticks, cpu))
        if path.startswith("/sys/devices/system/cpu/cpu"):
            return io.StringIO("5500000\n")
        raise OSError(path)
    with mock.patch("os.listdir", return_value=[str(t) for t in threads]), mock.patch("builtins.open", fake_open):
        return A.amostra(pid, motor)


class T(unittest.TestCase):
    def test_binario_c_acha_o_chamador_sem_nome(self):
        r = amostra_com({100: ("a1-motor", 5000, 2), 101: ("a1-motor", 0, 3), 102: ("a1-motor", 160, 4), 103: ("moe-cpu-1", 150, 6)})
        self.assertIn(("chamador", 4, 5500000), [x[:3] for x in r])
        self.assertEqual(sum(1 for p, *_ in r if p == "chamador"), 1)
        self.assertIn(("principal", 2, 5500000), [x[:3] for x in r])

    def test_binario_n_com_nome_nao_duplica(self):
        r = amostra_com({100: ("a1-motor", 5000, 2), 101: ("a1-motor", 90, 3), 102: ("moe-servico", 160, 4)})
        self.assertEqual([c for p, c, *_ in r if p == "chamador"], [4])

    def test_leitura_da_nomeada_falha_nao_promove_auxiliar(self):
        # moe-servico existe, mas a leitura do stat dele falha: o auxiliar sem nome não pode ocupar o papel
        threads = {100: ("a1-motor", 5000, 2), 101: ("a1-motor", 90, 3), 102: ("moe-servico", 160, 4)}
        orig = amostra_com.__globals__["stat"]
        def stat_quebrado(ticks, cpu):
            if ticks == 160:
                raise OSError("stat")
            return orig(ticks, cpu)
        with mock.patch.dict(amostra_com.__globals__, {"stat": stat_quebrado}):
            r = amostra_com(threads)
        self.assertFalse(any(p == "chamador" for p, *_ in r))

    def test_c_ambiguo_deixa_o_chamador_ausente(self):
        r = amostra_com({100: ("a1-motor", 5000, 2), 101: ("a1-motor", 500, 3), 102: ("a1-motor", 160, 4)})
        self.assertFalse(any(p == "chamador" for p, *_ in r))

    def test_nome_ilegivel_nao_promove_auxiliar(self):
        threads = {100: ("a1-motor", 5000, 2), 101: ("a1-motor", 90, 3), 102: ("moe-servico", 160, 4)}
        base = amostra_com.__globals__["stat"]
        r = None
        orig_open = io.StringIO
        def fake_stat(ticks, cpu):
            return base(ticks, cpu)
        # a leitura do comm de 102 falha: simula-se removendo-o do mapa de leitura mas mantendo-o na listagem
        def run():
            def fake_open(path, *a, **k):
                if path == "/proc/100/task/102/comm":
                    raise OSError(path)
                for tid, (comm, ticks, cpu) in threads.items():
                    if path == f"/proc/100/task/{tid}/comm":
                        return io.StringIO(comm + "\n")
                    if path == f"/proc/100/task/{tid}/stat":
                        return io.StringIO(stat(ticks, cpu))
                if path.startswith("/sys/devices/system/cpu/cpu"):
                    return io.StringIO("5500000\n")
                raise OSError(path)
            with mock.patch("os.listdir", return_value=[str(t) for t in threads]), mock.patch("builtins.open", fake_open):
                return A.amostra(100, True)
        r = run()
        self.assertFalse(any(p == "chamador" for p, *_ in r))

    def test_c_com_candidato_ilegivel_nao_promove_o_outro(self):
        threads = {100: ("a1-motor", 5000, 2), 101: ("a1-motor", 90, 3), 102: ("a1-motor", 160, 4)}
        def stat_quebrado(ticks, cpu):
            if ticks == 160:
                raise OSError("stat")
            return stat.__wrapped__(ticks, cpu) if hasattr(stat, "__wrapped__") else orig(ticks, cpu)
        orig = stat
        with mock.patch.dict(amostra_com.__globals__, {"stat": stat_quebrado}):
            r = amostra_com(threads)
        self.assertFalse(any(p == "chamador" for p, *_ in r))

    def test_sem_ticks_nao_inventa_chamador(self):
        r = amostra_com({100: ("a1-motor", 5000, 2), 101: ("a1-motor", 0, 3)})
        self.assertFalse(any(p == "chamador" for p, *_ in r))


class Ativas(unittest.TestCase):
    """Opção (a) do Breno, 3/11: a média de frequência da 0c só conta amostras em que a thread consumiu CPU."""
    def arq(self, linhas):
        import tempfile
        d = tempfile.mkdtemp()
        p = os.path.join(d, "freq.txt")
        open(p, "w").write("".join(linhas))
        self.addCleanup(lambda: (os.remove(p), os.rmdir(d)))
        return p

    def test_amostras_ociosas_nao_entram_na_media(self):
        # trabalhador ocioso a 3000 MHz (ticks parados) e ativo a 5500 MHz: só o ativo conta
        p = self.arq(["0.0 trabalhadores 2 5500000 7 10\n", "1.0 trabalhadores 2 3000000 7 10\n", "2.0 trabalhadores 2 3000000 7 10\n",
                      "3.0 trabalhadores 2 5500000 7 15\n", "4.0 trabalhadores 2 5400000 7 20\n"])
        self.assertAlmostEqual(A.medias(p)["trabalhadores"], 5450)

    def test_thread_sempre_ociosa_reprova_a_0c_de_modo_conservador(self):
        a = self.arq(["0.0 chamador 0 5500000 1 5\n", "1.0 chamador 0 5500000 1 9\n", "0.0 trabalhadores 2 3000000 7 10\n", "1.0 trabalhadores 2 3000000 7 10\n"])
        self.assertIn("sem amostras ativas de trabalhadores", A.compara(a, a))

    def test_ocioso_diferente_entre_binarios_nao_reprova_se_o_ativo_bate(self):
        a = self.arq(["0.0 chamador 0 5500000 1 5\n", "1.0 chamador 0 5500000 1 9\n", "0.0 trabalhadores 2 5500000 7 1\n", "1.0 trabalhadores 2 5500000 7 4\n", "2.0 trabalhadores 2 3000000 7 4\n"])
        b = self.arq(["0.0 chamador 0 5500000 1 5\n", "1.0 chamador 0 5500000 1 9\n", "0.0 trabalhadores 2 5500000 7 1\n", "1.0 trabalhadores 2 5500000 7 4\n", "2.0 trabalhadores 2 5500000 7 4\n"])
        self.assertEqual(A.compara(a, b), [])

    def test_ativo_fora_de_2_por_cento_continua_reprovando(self):
        a = self.arq(["0.0 chamador 0 5000000 1 5\n", "1.0 chamador 0 5000000 1 9\n", "0.0 trabalhadores 2 5500000 7 1\n", "1.0 trabalhadores 2 5500000 7 4\n"])
        b = self.arq(["0.0 chamador 0 5500000 1 5\n", "1.0 chamador 0 5500000 1 9\n", "0.0 trabalhadores 2 5500000 7 1\n", "1.0 trabalhadores 2 5500000 7 4\n"])
        self.assertTrue(any(x.startswith("chamador") for x in A.compara(a, b)))


if __name__ == "__main__":
    unittest.main()
