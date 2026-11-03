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
        self.assertIn(("chamador", 4, 5500000), r)
        self.assertEqual(sum(1 for p, _, _ in r if p == "chamador"), 1)
        self.assertIn(("principal", 2, 5500000), r)

    def test_binario_n_com_nome_nao_duplica(self):
        r = amostra_com({100: ("a1-motor", 5000, 2), 101: ("a1-motor", 90, 3), 102: ("moe-servico", 160, 4)})
        self.assertEqual([c for p, c, _ in r if p == "chamador"], [4])

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
        self.assertFalse(any(p == "chamador" for p, _, _ in r))

    def test_c_ambiguo_deixa_o_chamador_ausente(self):
        r = amostra_com({100: ("a1-motor", 5000, 2), 101: ("a1-motor", 500, 3), 102: ("a1-motor", 160, 4)})
        self.assertFalse(any(p == "chamador" for p, _, _ in r))

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
        self.assertFalse(any(p == "chamador" for p, _, _ in r))

    def test_c_com_candidato_ilegivel_nao_promove_o_outro(self):
        threads = {100: ("a1-motor", 5000, 2), 101: ("a1-motor", 90, 3), 102: ("a1-motor", 160, 4)}
        def stat_quebrado(ticks, cpu):
            if ticks == 160:
                raise OSError("stat")
            return stat.__wrapped__(ticks, cpu) if hasattr(stat, "__wrapped__") else orig(ticks, cpu)
        orig = stat
        with mock.patch.dict(amostra_com.__globals__, {"stat": stat_quebrado}):
            r = amostra_com(threads)
        self.assertFalse(any(p == "chamador" for p, _, _ in r))

    def test_sem_ticks_nao_inventa_chamador(self):
        r = amostra_com({100: ("a1-motor", 5000, 2), 101: ("a1-motor", 0, 3)})
        self.assertFalse(any(p == "chamador" for p, _, _ in r))


if __name__ == "__main__":
    unittest.main()
