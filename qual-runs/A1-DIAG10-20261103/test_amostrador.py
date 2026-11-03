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

    def test_sem_ticks_nao_inventa_chamador(self):
        r = amostra_com({100: ("a1-motor", 5000, 2), 101: ("a1-motor", 0, 3)})
        self.assertFalse(any(p == "chamador" for p, _, _ in r))


if __name__ == "__main__":
    unittest.main()
