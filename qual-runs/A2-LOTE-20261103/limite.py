#!/usr/bin/env python3
"""A2-LOTE-LIMITE (criterio-limite.txt): CPU medida, GPU no pior caso (w=n), g_M = g·(1+a·(M-1)). Sem GPU."""
import os, sys
R = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, R)
import lote as L  # noqa: E402
MP = L.MP
AS_DEC = (0.0, 1.0)
GRADE = [i / 20 for i in range(21)]


class SimA(L.SimMed):
    def __init__(self, *a, fator_a=1.0, **k):
        super().__init__(*a, **k)
        self.a = fator_a

    def gm(self):
        return 1 + self.a * (self.M - 1)


def roda(mod, frac, M, wg, wc, a, tr, te):
    Ln, _, k = tr[0].shape
    E = int(max(x.max() for x in tr + te)) + 1
    s = SimA(mod, frac, Ln, E, k, wg, wc, "g1", fator_a=a)
    s.run_blocks(tr, M, False)
    s.run_blocks(te, M, True)
    return s.stats["ms"] / s.stats["tokens"]


def ganhos(wg_por_mod, cpu, fracs, avals, carrega=MP.carrega):
    out = {}
    for mod in ("1b", "3b"):
        tr, te = carrega(mod)
        wc, _ = L.tabela_w(cpu, mod)
        for frac in fracs:
            for a in avals:
                t = {M: roda(mod, frac, M, wg_por_mod[mod], wc, a, tr, te) for M in MP.MS}
                for M in MP.MS:
                    out[(mod, frac, a, M)] = 1 - t[M] / t[1]
    return out


def melhor(g, mod, frac, a):
    return max(g[(mod, frac, a, M)] for M in MP.MS if M <= MP.M_MAX)


def passa(g, frac, a):
    return all(melhor(g, m, frac, a) >= MP.GANHO_MIN for m in ("1b", "3b"))


def decide(g):
    if passa(g, 0.10, 1.0):
        return "VALE PROTOTIPAR NO MOTOR mesmo no pior caso da GPU e do g_M", None
    if passa(g, 0.10, 0.0):
        astar = max(a for a in GRADE if passa(g, 0.10, a))
        return "DEPENDE DO g_M", astar
    return "NÃO VALE AGORA mesmo com g constante e GPU no pior caso (A2 fecha sem mais GPU)", None


def imprime(g, titulo, avals, frac):
    print(f"\n--- {titulo} ({frac:.0%} de slots) ---")
    for mod in ("1b", "3b"):
        for a in avals:
            print(f"  {mod} a={a:<4}: " + "  ".join(f"M={M}: {g[(mod, frac, a, M)]:6.1%}" for M in MP.MS) + f"   melhor M≤8: {melhor(g, mod, frac, a):.1%}")


def main(gpu_txt, cpu_txt):
    with open(cpu_txt) as h:
        cpu = L.parse(h.read(), "cpu")
    with open(gpu_txt) as f:
        gpu = L.parse(f.read(), "gpu")
    pior = {m: (lambda n: float(n)) for m in ("1b", "3b")}
    med = {m: L.tabela_w(gpu, m)[0] for m in ("1b", "3b")}
    print("=== A2-LOTE-LIMITE: CPU medida (cpu.txt), GPU w=n, g_M = g·(1+a·(M-1)) ===")
    g = ganhos(pior, cpu, (0.10,), GRADE)
    g.update(ganhos(pior, cpu, (0.25,), AS_DEC))
    imprime(g, "DECISÃO: GPU no pior caso", AS_DEC, 0.10)
    d, astar = decide(g)
    print("\n  DECISÃO:", d)
    if astar is not None:
        print(f"  a* (maior a da grade com 15% nos dois modelos): {astar:.2f}  -> g_M tolerável = g·(1+{astar:.2f}·(M-1)); em M=8: {1 + astar * 7:.2f}·g")
    imprime(g, "descritivo: grade de a, GPU no pior caso", (0.0, 0.25, 0.5, 0.75, 1.0), 0.10)
    imprime(g, "descritivo: 25% de slots, GPU no pior caso", AS_DEC, 0.25)
    gm_ = ganhos(med, cpu, (0.10,), AS_DEC)
    imprime(gm_, "descritivo: w_G medido (v1.2)", AS_DEC, 0.10)
    print("\n  Não calculado (previsto na seção 4 do critério): fração de passos de camada com GPU ou CPU no caminho crítico (o SimM não expõe).")


if __name__ == "__main__":
    main(os.path.join(R, "gpu.txt"), os.path.join(R, "cpu.txt"))
