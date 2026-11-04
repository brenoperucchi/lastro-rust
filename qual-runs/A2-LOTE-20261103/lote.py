#!/usr/bin/env python3
"""A2-LOTE: aplica o custo de lote MEDIDO do expert (lote_gpu e lote-cpu) ao SimM do A2-MULTIPOS-SIM. Critério: criterio.txt (v1).
Uso: tools/.venv-ref/bin/python -I -B qual-runs/A2-LOTE-20261103/lote.py <gpu.txt> <cpu.txt>     (sem GPU; só lê as duas saídas)
A calibração em M=1 é a do A2-MULTIPOS-SIM (w(1)=1 por definição, nada muda em M=1) e não se repete aqui."""
import os, re, sys
R = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(R, "..", "A2-MULTIPOS-SIM-20261103"))
import multipos as MP  # noqa: E402

NS = (1, 2, 4, 8, 16, 32)
TOL_GPU_ERR, TOL_CPU_ERR = 1e-3, 1e-5          # portão de correção (criterio.txt, seção 1)
TOL_GPU_T1, TOL_CPU_T1 = 0.40, 0.30            # portão de representatividade em n=1 (criterio.txt, seção 1)
RE_GPU = re.compile(r"^GPU (1b|3b) n=(\d+) lote_us=([\d.]+) serial_us=([\d.]+) err_lote_serial=([\d.e+-]+) err_ref=([\d.e+-]+)\s*$")
RE_CPU = re.compile(r"^CPU (1b|3b) n=(\d+) us_por_expert=([\d.]+) max_err_rel=([\d.e+-]+)\s*$")


def parse(texto, lado):
    """Devolve {(modelo, n): dict}. Linha repetida para o mesmo (modelo, n) é erro (nada de escolher a melhor)."""
    out = {}
    for ln in texto.splitlines():
        m = (RE_GPU if lado == "gpu" else RE_CPU).match(ln)
        if not m:
            continue
        mod, n = m.group(1), int(m.group(2))
        assert (mod, n) not in out, f"{lado}: linha repetida para {mod} n={n}"
        if lado == "gpu":
            out[(mod, n)] = {"lote": float(m.group(3)), "serial": float(m.group(4)), "err": max(float(m.group(5)), float(m.group(6)))}
        else:
            out[(mod, n)] = {"lote": float(m.group(3)), "err": float(m.group(4))}
    return out


def portoes(gpu, cpu):
    """Lista de motivos de reprovação (vazia = passou): tabela completa, correção e representatividade em n=1."""
    mot = []
    for lado, tab, tol in (("GPU", gpu, TOL_GPU_ERR), ("CPU", cpu, TOL_CPU_ERR)):
        for mod in ("1b", "3b"):
            for n in NS:
                r = tab.get((mod, n))
                if r is None:
                    mot.append(f"{lado} {mod} n={n}: ausente")
                elif not (r["err"] <= tol):
                    mot.append(f"{lado} {mod} n={n}: correção {r['err']:.2e} > {tol:.0e}")
    if mot:
        return mot
    for mod in ("1b", "3b"):
        p = MP.moe_sim.PARAMS_RUNTIME[mod]
        for lado, tab, ref, tol in (("GPU", gpu, p["h"] * 1000, TOL_GPU_T1), ("CPU", cpu, p["c_cpu_quiet"] * 1000, TOL_CPU_T1)):
            t1 = tab[(mod, 1)]["lote"]
            if abs(t1 / ref - 1) > tol:
                mot.append(f"{lado} {mod} n=1: {t1:.2f} us contra {ref:.2f} us do motor ({t1 / ref - 1:+.0%}, limite ±{tol:.0%})")
    return mot


def tabela_w(tab, mod, campo="lote"):
    """w(n) = t(n)/t(1), interpolação linear em n entre os pontos medidos; n > 32 não ocorre."""
    pts = [(n, tab[(mod, n)][campo] / tab[(mod, 1)][campo]) for n in NS]

    def w(n):
        assert 1 <= n <= NS[-1], n
        for (n0, w0), (n1, w1) in zip(pts, pts[1:]):
            if n <= n1:
                return w0 + (w1 - w0) * (n - n0) / (n1 - n0)
    return w, pts


class SimMed(MP.SimM):
    """SimM com o custo do expert medido (GPU e CPU) e g_M = g ('g1') ou M·g ('gM')."""
    def __init__(self, mod, frac, L, E, k, wg, wc, gm_modo, bw=MP.BW_GBS):
        super().__init__(mod, frac, L, E, k, "otimista", bw)
        self._wg, self._wc, self.gm_modo = wg, wc, gm_modo

    def wg(self, n):
        return self._wg(n)

    def wc(self, n):
        return self._wc(n)

    def gm(self):
        return self.M if self.gm_modo == "gM" else 1


def roda(mod, frac, M, wg, wc, gm_modo, treino, teste):
    L, _, k = treino[0].shape
    E = int(max(a.max() for a in treino + teste)) + 1
    s = SimMed(mod, frac, L, E, k, wg, wc, gm_modo)
    s.run_blocks(treino, M, False)
    s.run_blocks(teste, M, True)
    return s.stats["ms"] / s.stats["tokens"]


def decide(ganhos, portao_ok):
    """ganhos[(mod, gm_modo, M)]; critério seção 3."""
    if not portao_ok:
        return "SEM DECISÃO (portão reprovado)"
    def ok(modo):
        return all(max(ganhos[(m, modo, M)] for M in MP.MS if M <= MP.M_MAX) >= MP.GANHO_MIN for m in ("1b", "3b"))
    if ok("gM"):
        return "VALE PROTOTIPAR NO MOTOR (sob as hipóteses)"
    if ok("g1"):
        return "DEPENDE DO g_M (medir atenção e roteador em lote)"
    return "NÃO VALE AGORA"


def principal(gpu_txt, cpu_txt):
    gpu, cpu = parse(open(gpu_txt).read(), "gpu"), parse(open(cpu_txt).read(), "cpu")
    print("=== Tempo medido por expert (us) e w(n) = t(n)/t(1) ===")
    for mod in ("1b", "3b"):
        for lado, tab in (("GPU", gpu), ("CPU", cpu)):
            if all((mod, n) in tab for n in NS):
                print(f"  {lado} {mod}: " + "  ".join(f"n={n}: {tab[(mod, n)]['lote']:.2f} (w {tab[(mod, n)]['lote'] / tab[(mod, 1)]['lote']:.2f})" for n in NS))
        if all((mod, n) in gpu for n in NS):
            print(f"  GPU {mod} serial (informativo): " + "  ".join(f"n={n}: {gpu[(mod, n)]['serial']:.2f}" for n in NS))
    mot = portoes(gpu, cpu)
    print("\n=== Portões ===")
    print("  " + ("passaram (tabela completa, correção, representatividade em n=1)" if not mot else "REPROVADOS:"))
    for m in mot:
        print("   -", m)
    if mot:
        print("\n  DECISÃO:", decide({}, False))
        return 1
    ganhos = {}
    print("\n=== Ganho contra M=1 (10% de slots, conversas de teste) ===")
    for mod in ("1b", "3b"):
        tr, te = MP.carrega(mod)
        wg, _ = tabela_w(gpu, mod)
        wc, _ = tabela_w(cpu, mod)
        for modo in ("g1", "gM"):
            print(f"\n{mod} {modo}\n    M   ms/pos    ganho    a_eq   folga ms/passo")
            t1 = None
            for M in MP.MS:
                t = roda(mod, 0.10, M, wg, wc, modo, tr, te)
                t1 = t if M == 1 else t1
                g = 1 - t / t1
                ganhos[(mod, modo, M)] = g
                aeq, folga = MP.metricas(t1, t, M)
                print(f"  {M:>3}  {t:7.3f}  {g:7.1%}  {aeq:6.2f}  {folga:12.2f}")
    print(f"\n=== Regra (criterio.txt seção 3; M ≤ {MP.M_MAX}, ganho mínimo {MP.GANHO_MIN:.0%}) ===")
    for modo in ("gM", "g1"):
        for mod in ("1b", "3b"):
            g = max(ganhos[(mod, modo, M)] for M in MP.MS if M <= MP.M_MAX)
            print(f"  {mod} {modo}: maior ganho com M ≤ {MP.M_MAX} = {g:.1%} -> {'≥' if g >= MP.GANHO_MIN else '<'} {MP.GANHO_MIN:.0%}")
    print("  DECISÃO:", decide(ganhos, True))
    return 0


if __name__ == "__main__":
    sys.exit(principal(sys.argv[1], sys.argv[2]))
