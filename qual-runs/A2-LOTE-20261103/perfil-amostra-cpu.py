#!/usr/bin/env python3
"""Amostrador de %CPU do processo medido (P4 do criterio-perfil-prefill-20261109.md), por /proc/<pid>/stat a cada ~100 ms. Só registra; nada derruba a célula.

    python3 -I perfil-amostra-cpu.py --raiz PID --saida ARQ [--intervalo 0.1] [--comm primeira-palavra] [--proc /proc]

PID é o da árvore da célula (o roteiro roda-cmp-lastro.sh): o amostrador procura entre seus descendentes o processo cujo nome (campo comm do stat, até 15 caracteres) começa
por --comm e passa a lê-lo direto (sem varrer o /proc a cada volta). Cada linha: `epoch_ms,pid,cpu_pct` (cpu_pct = (utime+stime) consumidos entre duas leituras / tempo de parede
x 100; 100 = um núcleo cheio). Termina com SIGTERM/SIGINT ou quando o PID raiz some. Só biblioteca padrão.
"""
import argparse
import os
import signal
import sys
import time

PARAR = False


def _para(*_):
    global PARAR
    PARAR = True


def le_stat(proc, pid):
    """(comm, ppid, utime+stime em ticks) de /proc/<pid>/stat; None se sumiu ou ilegível (o comm pode ter espaços e parênteses: corta pelo último ')')."""
    try:
        with open("%s/%d/stat" % (proc, pid), "rb") as fh:
            b = fh.read().decode("utf-8", "replace")
    except OSError:
        return None
    i, j = b.find("("), b.rfind(")")
    if i < 0 or j < i:
        return None
    campos = b[j + 2:].split()
    try:
        return b[i + 1:j], int(campos[1]), int(campos[11]) + int(campos[12])
    except (IndexError, ValueError):
        return None


def descendentes(proc, raiz):
    """PIDs que descendem de `raiz` (varredura do /proc; só usada enquanto o alvo não foi achado)."""
    filhos = {}
    try:
        nomes = os.listdir(proc)
    except OSError:
        return []
    for n in nomes:
        if n.isdigit():
            st = le_stat(proc, int(n))
            if st:
                filhos.setdefault(st[1], []).append(int(n))
    achados, fila = [], [raiz]
    while fila:
        p = fila.pop()
        for c in filhos.get(p, []):
            achados.append(c)
            fila.append(c)
    return achados


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raiz", type=int, required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--intervalo", type=float, default=0.1)
    ap.add_argument("--comm", default="primeira-palavra")
    ap.add_argument("--proc", default="/proc")
    a = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, _para)
    signal.signal(signal.SIGINT, _para)
    hz = os.sysconf("SC_CLK_TCK")
    alvo, ant = None, None      # alvo = PID medido; ant = (instante monotônico, ticks)
    with open(a.saida, "w", encoding="utf-8", buffering=1) as fo:
        while not PARAR:
            if le_stat(a.proc, a.raiz) is None:
                break
            if alvo is None:
                for p in descendentes(a.proc, a.raiz):
                    st = le_stat(a.proc, p)
                    if st and st[0].startswith(a.comm[:15]):
                        alvo, ant = p, None
                        break
            if alvo is not None:
                st = le_stat(a.proc, alvo)
                agora = time.monotonic()
                if st is None:
                    alvo, ant = None, None
                else:
                    if ant is not None and agora > ant[0]:
                        pct = 100.0 * ((st[2] - ant[1]) / hz) / (agora - ant[0])
                        fo.write("%d,%d,%.1f\n" % (int(time.time() * 1000), alvo, max(0.0, pct)))
                    ant = (agora, st[2])
            time.sleep(a.intervalo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
