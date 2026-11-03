#!/usr/bin/env python3
"""A2-MULTIPOS-SIM: tempo por token do passo multiposição do motor D1, só simulador (criterio.txt v1 desta pasta). Sem GPU.
Uso:  taskset -c 16-27 tools/.venv-ref/bin/python -I -B qual-runs/A2-MULTIPOS-SIM-20261103/multipos.py --calibra     (só M=1, 10%)
      taskset -c 16-27 tools/.venv-ref/bin/python -I -B qual-runs/A2-MULTIPOS-SIM-20261103/multipos.py                 (calibra; só se passar, varre)
O modelo e os dois limites (otimista, pessimista) estão no criterio.txt; aqui só a conta."""
import math, os, sys
R = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(R, "..", "..", "tools"))
import numpy as np  # noqa: E402
import moe_sim  # noqa: E402

TR = {"1b": "granite-3.1-1b-a400m-instruct", "3b": "granite-3.1-3b-a800m-instruct"}
BW_GBS = 12.2            # banda de cópia com a CPU calculando, medida no x8 (qual-runs/PCIE-X8-20261031/resultado.txt); o moe_sim.py ainda traz 14,1
BACKLOG = 4              # cópias em voo por camada (como o ExpertCache/D1)
HL = moe_sim.HALF_LIFE   # meia-vida da frequência, em tokens
MS = (1, 2, 4, 8, 16, 32)
FRACS = (0.10, 0.25)
LIMITES = ("otimista", "pessimista")
MEDIDO_L1 = {"1b": 8.0, "3b": 14.4}   # ms/token a 10%, binário C do A1.4 (mediana das 5 repetições, qual-runs/A1-DIAG10-20261103/analise-e1.txt)
TOL_CAL = 0.10                          # fixada no criterio.txt antes de rodar
GANHO_MIN, M_MAX = 0.15, 8              # fixados no criterio.txt antes de rodar


class Passo:
    """Estado do cache de experts e o custo de um passo de M posições (protocolo token: a tabela é a do começo do passo).
    A fila do PCIe é única e contínua entre passos: uma cópia custa c_copy, começa quando a fila libera e só vale como residente no passo seguinte
    ao seu fim; camada com tráfego = há cópia em voo no começo dela ou ela enfileira uma (moe_sim.py)."""
    def __init__(self, mod, frac, L, E, k, limite):
        assert limite in LIMITES
        self.p = moe_sim.PARAMS_RUNTIME[mod]
        self.L, self.E, self.k, self.limite = L, E, k, limite
        self.cap = int(frac * L * E)
        self.c_copy = self.p["expert_mb"] / BW_GBS      # MB / (GB/s) = ms
        self.resident = set()
        self.emvoo = []                                  # (chave, instante em que a cópia termina)
        self.pcie_livre = 0.0
        self.clock = 0.0
        self.freq = {}                                   # chave -> (valor, token da última atualização)
        self.tok = 0
        self.stats = {"ms": 0.0, "tokens": 0, "hits": 0, "needed": 0, "cop": 0}

    def score(self, key):
        v, t0 = self.freq.get(key, (0.0, self.tok))
        return v * 2.0 ** (-(self.tok - t0) / HL)

    def toca(self, key, n):
        self.freq[key] = (self.score(key) + n, self.tok)

    def _mais_fraco(self, saindo):
        cands = [k_ for k_ in self.resident if k_ not in saindo]
        return min(cands, key=self.score) if cands else None

    def passo(self, blk, medir):
        """blk: array (L, M, k) de experts pedidos. Devolve o tempo do passo em ms."""
        p, M = self.p, blk.shape[1]
        pess = self.limite == "pessimista"
        gM = p["g"] * (M if pess else 1)
        w = (lambda n: n) if pess else (lambda n: 1)
        # começo do passo: assenta as cópias terminadas e publica a tabela
        for key, fim in [x for x in self.emvoo if x[1] <= self.clock]:
            self.resident.add(key)
        self.emvoo = [x for x in self.emvoo if x[1] > self.clock]
        saindo, total, copias = set(), 0.0, 0
        for l in range(self.L):
            t = self.clock + total
            es, ns = np.unique(blk[l], return_counts=True)
            assert len(es) <= min(self.E, M * self.k)
            lst = list(zip(es.tolist(), ns.tolist()))
            r = [(e, n) for e, n in lst if (l, e) in self.resident]
            f = [(e, n) for e, n in lst if (l, e) not in self.resident]
            # admissões desta camada: no máximo BACKLOG cópias em voo (as de passos anteriores contam)
            voo = sum(1 for _, fim in self.emvoo if fim > t)
            novos = []
            for e, n in sorted(f, key=lambda en: -self.score((l, en[0]))):
                if voo + len(novos) >= BACKLOG:
                    break
                if len(self.resident) - len(saindo) + len(self.emvoo) + len(novos) < self.cap:
                    novos.append(e)
                    continue
                fraco = self._mais_fraco(saindo)
                if fraco is None or self.score((l, e)) <= self.score(fraco):
                    break
                saindo.add(fraco)
                novos.append(e)
            trafego = self.pcie_livre > t or bool(novos)
            c = p["c_cpu"] if trafego else p["c_cpu_quiet"]
            gpu = sum(p["h"] * w(n) for _, n in r)           # os experts residentes, depois do g da camada (atenção e roteador)
            cpu = sum(c * w(n) for _, n in f)                # a CPU só começa quando o roteador termina (moe_sim.py: "g antes dos experts")
            tl = gM + max(gpu, cpu) + (p["o_layer"] if trafego and f else 0.0)
            total += tl
            ini = max(self.pcie_livre, t + tl)               # as admissões são decididas quando a CPU da camada termina
            for e in novos:
                ini += self.c_copy
                self.emvoo.append(((l, e), ini))
            if novos:
                self.pcie_livre = ini
            copias += len(novos)
            for e, n in lst:
                self.toca((l, e), n)
            if medir:
                self.stats["hits"] += sum(n for _, n in r)
                self.stats["needed"] += int(ns.sum())
        self.resident -= saindo
        self.clock += total
        self.tok += M
        if medir:
            self.stats["ms"] += total
            self.stats["tokens"] += M
            self.stats["cop"] += copias
        return total


def roda(mod, frac, M, limite, treino, teste):
    L, _, k = treino[0].shape
    E = int(max(a.max() for a in treino + teste)) + 1
    s = Passo(mod, frac, L, E, k, limite)
    for medir, convs in ((False, treino), (True, teste)):
        for a in convs:
            T = a.shape[1]
            for b in range(T // M):
                s.passo(a[:, b * M:(b + 1) * M, :], medir)
    return s.stats["ms"] / s.stats["tokens"], s.stats["hits"] / s.stats["needed"], s.stats["cop"] / s.stats["tokens"]


def carrega(mod):
    d = os.path.join(R, "..", "TRACOS-20261029", TR[mod])
    return moe_sim.load(d, "treino"), moe_sim.load(d, "teste")


def calibra():
    ok = True
    print(f"=== Calibração (M=1, 10% de slots; banda de cópia {BW_GBS} GB/s; tolerância ±{TOL_CAL:.0%} contra o L1 medido do binário C) ===")
    for mod in ("1b", "3b"):
        tr, te = carrega(mod)
        t, hit, cop = roda(mod, 0.10, 1, "otimista", tr, te)
        tp, _, _ = roda(mod, 0.10, 1, "pessimista", tr, te)
        assert abs(t - tp) < 1e-9, "no M=1 os dois limites têm de coincidir"
        erro = (t - MEDIDO_L1[mod]) / MEDIDO_L1[mod]
        bom = abs(erro) <= TOL_CAL
        ok &= bom
        p = moe_sim.PARAMS_RUNTIME[mod]
        ideal = 24 if mod == "1b" else 32
        ideal = ideal * (p["g"] + 8 * p["h"])
        print(f"  {mod}: simulado {t:.3f} ms/token, medido L1 {MEDIDO_L1[mod]:.1f} ms, erro {erro:+.1%} -> {'ok' if bom else 'FORA'}; acerto {hit:.1%}, cópias {cop:.1f}/token; [informativo] ideal tudo residente {ideal:.2f} ms")
    print("  calibração:", "PASSOU" if ok else "REPROVOU: SIMULADOR NÃO SERVE (nenhuma previsão de M > 1)")
    return ok


def decide(res):
    """Regra do criterio.txt v1, seção 4: res[(mod, frac, limite, M)] = ganho contra M=1. Decide pelos dois modelos a 10%."""
    def ok(lim):
        return all(max(res[(m, 0.10, lim, M)] for M in MS if M <= M_MAX) >= GANHO_MIN for m in ("1b", "3b"))
    if ok("pessimista"):
        return "VALE IMPLEMENTAR NO MOTOR"
    if ok("otimista"):
        return "VALE MEDIR NA GPU ANTES (custo de lote do cálculo)"
    return "NÃO VALE AGORA"


def varredura():
    print("\n=== Varredura: tempo por token (ms) e ganho contra M=1, nas conversas de teste ===")
    res = {}
    for mod in ("1b", "3b"):
        tr, te = carrega(mod)
        for frac in FRACS:
            for lim in LIMITES:
                t1 = None
                print(f"\n{mod} {frac:.0%} {lim}\n    M   ms/token   ganho   acerto   cópias/token")
                for M in MS:
                    t, hit, cop = roda(mod, frac, M, lim, tr, te)
                    t1 = t if M == 1 else t1
                    res[(mod, frac, lim, M)] = 1 - t / t1
                    print(f"  {M:>3}   {t:8.3f}  {1 - t / t1:6.1%}  {hit:7.1%}  {cop:10.1f}")
                g = [res[(mod, frac, lim, M)] for M in MS]
                achata = next(M for M, x in zip(MS, g) if g[-1] - x < 0.02)
                print(f"  a curva achata (a menos de 2 pontos do ganho em M=32) em M={achata}")
    print("\n=== Regra de decisão (criterio.txt v1, seção 4; M ≤ %d, ganho mínimo %.0f%%, 10%% de slots) ===" % (M_MAX, 100 * GANHO_MIN))
    for lim in LIMITES:
        for mod in ("1b", "3b"):
            g = max(res[(mod, 0.10, lim, M)] for M in MS if M <= M_MAX)
            print(f"  {mod} {lim}: maior ganho com M ≤ {M_MAX} = {g:.1%} -> {'≥' if g >= GANHO_MIN else '<'} {GANHO_MIN:.0%}")
    print("  DECISÃO:", decide(res))


if __name__ == "__main__":
    if not calibra():
        sys.exit(1)
    if "--calibra" not in sys.argv:
        varredura()
