#!/usr/bin/env python3
"""A2-PASSO E1 (criterio-e1.txt): caminho crítico, fração útil, folga e sensibilidade do custo de lote da CPU. Só simulador."""
import math, os, sys
R = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, R)
import lote as L  # noqa: E402
import limite as X  # noqa: E402
MP = L.MP
MS4 = (1, 2, 4, 8)
BETAS = (0.0, 0.03, 0.1, 0.2, 0.3, 0.45, 0.65, 1.0)
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


class SimC(X.SimA):
    """Cópia de SimM.layer_m (multipos.py) com contadores do lado dominante; o resultado de tempo tem de ser idêntico (testado)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.cont = {"camadas": 0, "cpu_dom": 0, "tempo": 0.0, "cpu_acima": 0.0}

    def layer_m(self, l, es, ns):
        p, M = self.p, self.M
        self.cur_layer = l
        if l == 0:
            self.begin_token()
        t_ini = self.now
        self.now += p["g"] * self.gm()
        if l > 0:
            self.release_deferred(l - 1)
        start = self.now
        keys = [(l, int(e)) for e in es]
        pinned = set(keys)
        for key, n in zip(keys, ns):
            for _ in range(int(n)):
                self.touch_freq(key)
        hsum, cpu_t, cpu_w = 0.0, start, 0.0
        busy0 = any(start < f < math.inf for f in self.inflight.values())
        copies0 = self.started
        c_cpu = p["c_cpu"] if busy0 else p["c_cpu_quiet"]
        adm = []
        for key, n in zip(keys, ns):
            n = int(n)
            self.stats["needed"] += n
            if key in self.table:
                if key in self.resident:
                    self.resident.move_to_end(key)
                    self.push(key)
                hsum += p["h"] * self.wg(n)
                self.stats["hits"] += n
                self.reserved.pop(key, None)
                continue
            cpu_t += c_cpu * self.wc(n)
            cpu_w += self.wc(n)
            self.stats["cpu"] += n
            if key in self.inflight:
                self.requested[key] = self.step
                continue
            adm.append(key)
        traffic = busy0 or self.started > copies0
        if traffic and not busy0:
            cpu_t += cpu_w * (p["c_cpu"] - c_cpu)
        if adm:
            self.now = cpu_t
            for key in adm:
                self.admit_bg(key, pinned)
            self.now = start
        gpu_end = start + hsum if hsum else start
        end = max(gpu_end, cpu_t)
        if traffic and cpu_t > start:
            end += p["o_layer"]
        self.now = end
        if l == self.L - 1:
            self.release_deferred(l)
        self.step += M
        c = self.cont
        c["camadas"] += 1
        c["cpu_dom"] += cpu_t > gpu_end
        c["tempo"] += end - t_ini
        c["cpu_acima"] += max(0.0, cpu_t - gpu_end)


def roda_c(mod, frac, M, wg, wc, a, tr, te):
    Ln, _, k = tr[0].shape
    E = int(max(x.max() for x in tr + te)) + 1
    s = SimC(mod, frac, Ln, E, k, wg, wc, "g1", fator_a=a)
    s.run_blocks(tr, M, False)
    for key in s.cont:
        s.cont[key] = 0 if key != "tempo" and key != "cpu_acima" else 0.0
    s.run_blocks(te, M, True)
    return s.stats["ms"] / s.stats["tokens"], dict(s.cont)


def ne_dist(mod, wc, M=8):
    _, te = MP.carrega(mod)
    tot = u4 = u8 = 0
    soma_w = 0.0
    for a in te:
        for b in range(a.shape[1] // M):
            blk = a[:, b * M:(b + 1) * M, :]
            for l in range(blk.shape[0]):
                es, ns = MP.primeira_aparicao(blk[l])
                for n in ns:
                    n = int(n)
                    tot += 1
                    u4 += n <= 4
                    u8 += n == 8
                    soma_w += wc(n)
    return u4 / tot, u8 / tot, soma_w / tot


def main():
    with open(os.path.join(R, "cpu.txt")) as h:
        cpu = L.parse(h.read(), "cpu")
    pior = lambda n: float(n)  # noqa: E731
    print("=== A2-PASSO E1 (criterio-e1.txt): GPU w=n, 10% de slots ===")
    # Passo 0 (seção 5): o instrumento reproduz o limite.py
    for mod in ("1b", "3b"):
        tr, te = MP.carrega(mod)
        wc, _ = L.tabela_w(cpu, mod)
        for a in (0.0, 1.0):
            t_ref = X.roda(mod, 0.10, 8, pior, wc, a, tr, te)
            t_i, _ = roda_c(mod, 0.10, 8, pior, wc, a, tr, te)
            assert abs(t_ref - t_i) < 1e-9, (mod, a, t_ref, t_i)
    print("instrumento reproduz o limite.py (M=8, a em {0,1}, os dois modelos): ok", flush=True)
    res = {}
    for mod in ("1b", "3b"):
        tr, te = MP.carrega(mod)
        wc, _ = L.tabela_w(cpu, mod)
        print(f"\n##### {mod} #####")
        print("P5 (M=8): n_e<=4 %.0f%%, n_e=8 %.1f%%, w_C efetivo %.3f" % tuple(x * (100 if i < 2 else 1) for i, x in enumerate(ne_dist(mod, wc))), flush=True)
        for a in (0.0, 1.0):
            t, cont = {}, {}
            for M in MP.MS:
                t[M], cont[M] = roda_c(mod, 0.10, M, pior, wc, a, tr, te)
            res[(mod, "base", a)] = t
            print(f"\nbase a={a} (CPU medida)")
            print("   M   ganho  CPU dom.(camadas)  CPU dom.(tempo)  a_eq  folga ms/passo")
            for M in MP.MS:
                c = cont[M]
                g = 1 - t[M] / t[1]
                aeq = (t[M] / t[1]) / 0.85
                fol = M * (0.85 * t[1] - t[M])
                print(f"  {M:>2}  {g:6.1%}  {c['cpu_dom'] / c['camadas']:15.1%}  {c['cpu_acima'] / c['tempo']:14.1%}  {aeq:5.2f}  {fol:9.2f}")
        for nome, grade, mk in (("beta", BETAS, lambda b: (lambda n: 1 + b * (n - 1))),
                                ("lambda", LAMBDAS, lambda lam: (lambda n, w=wc, l=lam: w(n) + l * (n - w(n))))):
            print(f"\nsensibilidade {nome} (a=1, GPU w=n): melhor ganho com M<=8")
            ok = []
            for v in grade:
                w = mk(v)
                t = {M: X.roda(mod, 0.10, M, pior, w, 1.0, tr, te) for M in MS4}
                best = max(1 - t[M] / t[1] for M in MS4)
                print(f"  {nome}={v:<5}: " + "  ".join(f"M={M}: {1 - t[M] / t[1]:6.1%}" for M in MS4) + f"   melhor {best:.1%}", flush=True)
                if best >= MP.GANHO_MIN:
                    ok.append(v)
            res[(mod, nome)] = max(ok) if ok else None
            print(f"  {nome}* (maior da grade com >=15%): {res[(mod, nome)]}")
        print("beta medido (w_med(n)-1)/(n-1): " + "  ".join(f"n={n}: {(wc(n) - 1) / (n - 1):.3f}" for n in (2, 4, 8, 16, 32)))
    print("\n=== Leitura (regra da seção 2) ===")
    lam = [res[(m, "lambda")] for m in ("1b", "3b")]
    bet = [res[(m, "beta")] for m in ("1b", "3b")]
    print(f"  lambda* 1b/3b: {lam}; beta* 1b/3b: {bet}")
    if any(x is None or x < 0.25 for x in lam) or any(x is None or x < 0.10 for x in bet):
        v = "FRÁGIL"
    elif all(x >= 0.5 for x in lam) and all(x >= 0.30 for x in bet):
        v = "ROBUSTO"
    else:
        v = "INTERMEDIÁRIO"
    print("  CLASSIFICAÇÃO:", v)


if __name__ == "__main__":
    main()
