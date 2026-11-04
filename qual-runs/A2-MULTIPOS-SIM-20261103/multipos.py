#!/usr/bin/env python3
"""A2-MULTIPOS-SIM: tempo por posição do passo multiposição do motor D1, só simulador (criterio.txt v1.1 desta pasta). Sem GPU.
O cache é o do próprio tools/moe_sim.py (protocolo "token", lfu_cpu: tabela do começo do passo, backlog, cópia adiada até a GPU passar da camada
da vítima, fila de PCIe única); aqui só se generaliza a camada para um passo de M posições (união de experts com contagem n_e). Em M=1 o
resultado é IGUAL ao de Sim(protocol="token", policy="lfu_cpu"): teste de equivalência em test_multipos.py.
Uso:  taskset -c 16-27 tools/.venv-ref/bin/python -I -B qual-runs/A2-MULTIPOS-SIM-20261103/multipos.py --calibra     (só M=1, 10%)
      taskset -c 16-27 tools/.venv-ref/bin/python -I -B qual-runs/A2-MULTIPOS-SIM-20261103/multipos.py                 (calibra; só se passar, varre)"""
import math, os, sys
R = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(R, "..", "..", "tools"))
import numpy as np  # noqa: E402
import moe_sim  # noqa: E402

TR = {"1b": "granite-3.1-1b-a400m-instruct", "3b": "granite-3.1-3b-a800m-instruct"}
BW_GBS = 12.2            # banda de cópia com a CPU calculando, medida no x8 (qual-runs/PCIE-X8-20261031/resultado.txt); o moe_sim.py ainda traz 14,1
MS = (1, 2, 4, 8, 16, 32)
FRACS = (0.10, 0.25)
LIMITES = ("otimista", "pessimista")
# Calibração em M=1 a 10% (fixada no criterio.txt v1.1 antes de rodar esta versão). Alvos medidos no A1.4 (binário C, 5 repetições; analise-e1.txt e os
# contadores de cópias do rodar do A1.4): ms/token L1, cópias de fundo por token e, no 1B, acerto.
MEDIDO_L1 = {"1b": 8.0, "3b": 14.4}
MEDIDO_COP = {"1b": 1.17, "3b": 1.84}
MEDIDO_ACERTO = {"1b": 0.250}
TOL_MS, TOL_COP, TOL_ACERTO = 0.10, 0.15, 0.02
GANHO_MIN, M_MAX = 0.15, 8              # fixados no criterio.txt antes de rodar


class SimM(moe_sim.Sim):
    """Sim(protocol="token", policy="lfu_cpu") com camada de M posições. limite "otimista": o peso do expert é lido uma vez (custo do expert
    independe de n_e) e g não cresce com M; "pessimista": cada posição paga o expert (w = n_e) e g_M = M·g. Em M=1 os dois coincidem."""
    def __init__(self, mod, frac, L, E, k, limite, bw=BW_GBS):
        assert limite in LIMITES
        p = dict(moe_sim.PARAMS_RUNTIME[mod])
        p["pcie_gbs_cpu"] = bw
        super().__init__("lfu_cpu", int(frac * L * E), L, E, k, p, protocol="token")
        self.limite, self.M = limite, 1

    def layer_m(self, l, es, ns):
        p, M = self.p, self.M
        pess = self.limite == "pessimista"
        w = (lambda n: n) if pess else (lambda n: 1)
        self.cur_layer = l
        if l == 0:
            self.begin_token()
        self.now += p["g"] * (M if pess else 1)
        if l > 0:
            self.release_deferred(l - 1)
        start = self.now
        keys = [(l, int(e)) for e in es]
        pinned = set(keys)
        for key, n in zip(keys, ns):
            for _ in range(int(n)):                    # a frequência conta cada pedido, antes de qualquer decisão da camada
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
                hsum += p["h"] * w(n)
                self.stats["hits"] += n
                self.reserved.pop(key, None)
                continue
            cpu_t += c_cpu * w(n)
            cpu_w += w(n)
            self.stats["cpu"] += n
            if key in self.inflight:
                self.requested[key] = self.step        # falta com cópia em voo: CPU, sem nova admissão
                continue
            adm.append(key)
        traffic = busy0 or self.started > copies0
        if traffic and not busy0:
            cpu_t += cpu_w * (p["c_cpu"] - c_cpu)
        if adm:
            self.now = cpu_t                            # as admissões vêm depois da flag, com o backlog desse instante
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

    def run_blocks(self, convs, M, measure):
        """Blocos de M posições consecutivas, sem atravessar o fim da conversa (o resto incompleto é descartado)."""
        self.M = M
        if measure:
            for k_ in self.stats:
                self.stats[k_] = 0 if k_ != "wait_ms" else 0.0
            t0 = self.now
        for a in convs:
            for b in range(a.shape[1] // M):
                blk = a[:, b * M:(b + 1) * M, :]
                for l in range(self.L):
                    # ordem da primeira aparição (posição por posição): em M=1 é a ordem do roteador, como no moe_sim (a ordem decide quem é admitido primeiro)
                    es, ns = primeira_aparicao(blk[l])
                    assert len(es) <= min(self.E, M * self.k)
                    self.layer_m(l, es, ns)
                self.stats["tokens"] += M
        if measure:
            self.stats["ms"] = self.now - t0


def primeira_aparicao(x):
    flat = x.reshape(-1).tolist()
    es = list(dict.fromkeys(flat))
    return es, [flat.count(e) for e in es]


def roda(mod, frac, M, limite, treino, teste, bw=BW_GBS):
    L, _, k = treino[0].shape
    E = int(max(a.max() for a in treino + teste)) + 1
    s = SimM(mod, frac, L, E, k, limite, bw)
    s.run_blocks(treino, M, False)                      # aquecimento da frequência (conversas 0 a 3)
    s.run_blocks(teste, M, True)                        # medida (conversas 4 a 7)
    st = s.stats
    return {"ms": st["ms"] / st["tokens"], "acerto": st["hits"] / st["needed"], "cop": st["copies_bg"] / st["tokens"], "tokens": st["tokens"]}


def carrega(mod):
    d = os.path.join(R, "..", "TRACOS-20261029", TR[mod])
    return moe_sim.load(d, "treino"), moe_sim.load(d, "teste")


def avalia_calibracao(mod, r):
    """Confere um resultado de M=1 contra os alvos medidos. Devolve (ok, linha)."""
    e_ms = (r["ms"] - MEDIDO_L1[mod]) / MEDIDO_L1[mod]
    e_cop = (r["cop"] - MEDIDO_COP[mod]) / MEDIDO_COP[mod]
    b_ms, b_cop = abs(e_ms) <= TOL_MS, abs(e_cop) <= TOL_COP
    linha = f"  {mod}: {r['ms']:.3f} ms/token (medido {MEDIDO_L1[mod]:.1f}, {e_ms:+.1%}, {'ok' if b_ms else 'FORA'}); cópias {r['cop']:.2f}/token (medido {MEDIDO_COP[mod]:.2f}, {e_cop:+.1%}, {'ok' if b_cop else 'FORA'}); acerto {r['acerto']:.1%}"
    ok = b_ms and b_cop
    if mod in MEDIDO_ACERTO:
        b_ac = abs(r["acerto"] - MEDIDO_ACERTO[mod]) <= TOL_ACERTO
        ok = ok and b_ac
        linha += f" (medido {MEDIDO_ACERTO[mod]:.1%}, {'ok' if b_ac else 'FORA'})"
    return ok, linha


def calibra():
    ok = True
    print(f"=== Calibração (M=1, 10% de slots; banda de cópia {BW_GBS} GB/s) ===")
    print(f"  tolerâncias (criterio.txt v1.1): ms/token ±{TOL_MS:.0%}, cópias/token ±{TOL_COP:.0%}, acerto ±{100 * TOL_ACERTO:.0f} pontos (só 1B)")
    for mod in ("1b", "3b"):
        tr, te = carrega(mod)
        r = roda(mod, 0.10, 1, "otimista", tr, te)
        rp = roda(mod, 0.10, 1, "pessimista", tr, te)
        assert r == rp, "no M=1 os dois limites têm de coincidir"
        bom, linha = avalia_calibracao(mod, r)
        print(linha)
        ok &= bom
    print("  [informativo, não decide] medido no A1.4: acerto do 3B a 10% 29,06%; cópias adiadas por token 0,64 (1B) e 1,03 (3B); 1B a 25%: 1,95 cópias/token e acerto 50,3%")
    print("  calibração:", "PASSOU" if ok else "REPROVOU: SIMULADOR NÃO SERVE (nenhuma previsão de M > 1)")
    return ok


def metricas(t1, tm, M):
    """a_eq: menor fração de posições úteis que mantém ganho de GANHO_MIN; folga: custo extra por passo (ms) que ainda deixa esse ganho."""
    return tm / t1 / (1 - GANHO_MIN), M * ((1 - GANHO_MIN) * t1 - tm)


def achata(gs):
    """Menor M cujo ganho está a menos de 2 pontos (em valor absoluto) do ganho em M=32."""
    return next(M for M, x in zip(MS, gs) if abs(gs[-1] - x) < 0.02)


def decide(res):
    """Regra do criterio.txt, seção 4: res[(mod, frac, limite, M)] = ganho contra M=1. Decide pelos dois modelos a 10%."""
    def ok(lim):
        return all(max(res[(m, 0.10, lim, M)] for M in MS if M <= M_MAX) >= GANHO_MIN for m in ("1b", "3b"))
    if ok("pessimista"):
        return "VALE PROTOTIPAR NO MOTOR (sob as hipóteses do modelo)"
    if ok("otimista"):
        return "VALE MEDIR NA GPU ANTES (custo de lote do cálculo)"
    return "NÃO VALE AGORA"


def varredura():
    print("\n=== Varredura: tempo por posição processada (ms), ganho contra M=1, nas conversas de teste (todas as posições do passo contam como úteis) ===")
    print("  a_eq = menor fração de posições úteis que mantém ganho de 15% (t(M)/t(1) / 0,85); folga = custo extra por passo (ms) que ainda deixa 15% de ganho: M·(0,85·t(1) − t(M))")
    res = {}
    for mod in ("1b", "3b"):
        tr, te = carrega(mod)
        for frac in FRACS:
            for lim in LIMITES:
                print(f"\n{mod} {frac:.0%} {lim}\n    M   ms/pos    ganho   acerto   cópias/pos    a_eq   folga ms/passo")
                t1 = None
                for M in MS:
                    r = roda(mod, frac, M, lim, tr, te)
                    t1 = r["ms"] if M == 1 else t1
                    g = 1 - r["ms"] / t1
                    res[(mod, frac, lim, M)] = g
                    aeq, folga = metricas(t1, r["ms"], M)
                    print(f"  {M:>3}  {r['ms']:7.3f}  {g:7.1%}  {r['acerto']:7.1%}  {r['cop']:10.2f}  {aeq:6.2f}  {folga:12.2f}")
                gs = [res[(mod, frac, lim, M)] for M in MS]
                print(f"  a curva achata (a menos de 2 pontos do ganho em M=32) em M={achata(gs)}")
    print("\n=== Regra de decisão (criterio.txt v1.1, seção 4; M ≤ %d, ganho mínimo %.0f%%, 10%% de slots) ===" % (M_MAX, 100 * GANHO_MIN))
    for lim in LIMITES:
        for mod in ("1b", "3b"):
            g = max(res[(mod, 0.10, lim, M)] for M in MS if M <= M_MAX)
            print(f"  {mod} {lim}: maior ganho com M ≤ {M_MAX} = {g:.1%} -> {'≥' if g >= GANHO_MIN else '<'} {GANHO_MIN:.0%}")
    print("  DECISÃO:", decide(res))


def principal(argv):
    if not calibra():
        return 1
    if "--calibra" not in argv:
        varredura()
    return 0


if __name__ == "__main__":
    sys.exit(principal(sys.argv))
