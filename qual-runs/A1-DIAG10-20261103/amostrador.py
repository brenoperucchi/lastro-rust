#!/usr/bin/env python3
"""A1-DIAG10: telemetria de frequência por papel, EXTERNA ao processo medido (criterio.txt v6, seção 2). Roda nos E-cores (taskset 16-27, por quem chama).
Uso:  amostrador.py PID ARQUIVO     amostra a cada 1 s, enquanto o PID existir:  <t> <papel> <cpu> <kHz>
      amostrador.py --medias ARQUIVO   papéis e médias em MHz
      amostrador.py --compara A B      0c: chamador e trabalhadores dentro de 2% entre os dois blocos da mesma configuração (código 0 = ok)
Papéis: chamador (a1-motor: moe-servico; m2a-cache: a thread principal), trabalhadores (moe-cpu-*), principal (a1-motor: a thread que gira na stream)."""
import os, sys, time

TOL = 0.02


def papel_de(tid, pid, comm, motor):
    if comm.startswith("moe-cpu-"):
        return "trabalhadores"
    if comm == "moe-servico":
        return "chamador"
    if tid == pid:
        return "principal" if motor else "chamador"
    return None


def amostra(pid, motor):
    out = []
    cand = []
    nomeada = False
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return None
    for t in tids:
        try:
            comm = open(f"/proc/{pid}/task/{t}/comm").read().strip()
            nomeada = nomeada or comm == "moe-servico"      # visto pelo nome, mesmo que a leitura seguinte falhe
            stat = open(f"/proc/{pid}/task/{t}/stat").read()
            cpu = int(stat[stat.rfind(")") + 2:].split()[36])
            khz = int(open(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq").read())
        except (OSError, ValueError, IndexError):
            continue
        p = papel_de(int(t), pid, comm, motor)
        if p:
            out.append((p, cpu, khz))
        elif motor and comm == "a1-motor" and int(t) != pid:
            # binário C do A1.4: a thread de serviço não tem nome e herda o do executável; as outras com esse nome (CUDA) ficam em 0 ticks.
            f = stat[stat.rfind(")") + 2:].split()
            cand.append((int(f[11]) + int(f[12]), cpu, khz))
    # só sem moe-servico (C) e só com UM candidato com CPU consumida; ambíguo ou nenhum: chamador ausente (a 0c falha de modo conservador)
    ativos = [c for c in cand if c[0] > 0]
    if not nomeada and len(ativos) == 1:
        out.append(("chamador", ativos[0][1], ativos[0][2]))
    return out


def coleta(pid, arq):
    t0 = time.time()
    motor = "a1-motor" in open(f"/proc/{pid}/cmdline").read()
    with open(arq, "w") as f:
        while True:
            s = amostra(pid, motor)
            if s is None:
                return
            for p, c, k in s:
                f.write(f"{time.time() - t0:.1f} {p} {c} {k}\n")
            f.flush()
            time.sleep(1.0)


def medias(arq):
    """papel -> média em MHz sobre as amostras do papel."""
    acc = {}
    for ln in open(arq):
        f = ln.split()
        if len(f) == 4:
            acc.setdefault(f[1], []).append(int(f[3]))
    return {p: sum(v) / len(v) / 1000 for p, v in acc.items()}


def compara(a, b):
    """Lista de problemas da regra 0c: o mesmo papel (chamador, trabalhadores) a mais de 2% de diferença entre as duas linhas."""
    ma, mb = medias(a), medias(b)
    pr = []
    for p in ("chamador", "trabalhadores"):
        if p not in ma or p not in mb:
            pr.append(f"sem amostras de {p}")
            continue
        m = (ma[p] + mb[p]) / 2
        if abs(ma[p] - mb[p]) > TOL * m:
            pr.append(f"{p}: {ma[p]:.0f} MHz contra {mb[p]:.0f} MHz")
    return pr


if __name__ == "__main__":
    if sys.argv[1] == "--medias":
        for p, m in sorted(medias(sys.argv[2]).items()):
            print(f"{p}: {m:.0f} MHz")
    elif sys.argv[1] == "--compara":
        pr = compara(sys.argv[2], sys.argv[3])
        print("; ".join(pr) if pr else "ok")
        sys.exit(1 if pr else 0)
    else:
        coleta(int(sys.argv[1]), sys.argv[2])
