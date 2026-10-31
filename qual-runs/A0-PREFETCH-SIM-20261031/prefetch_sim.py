"""A0a: teto da busca antecipada no simulador, pelo criterio.txt desta pasta. Rodar a partir da raiz do projeto:
  taskset -c 16-27 tools/.venv-ref/bin/python -I -B qual-runs/A0-PREFETCH-SIM-20261031/prefetch_sim.py > .../resultado.txt
Determinístico: a mesma árvore dá a mesma saída."""
import hashlib
import itertools
import os
import sys

sys.path.insert(0, "tools")
import moe_sim as m  # noqa: E402

T = "qual-runs/TRACOS-20261029"
MODS = {"1b": f"{T}/granite-3.1-1b-a400m-instruct", "3b": f"{T}/granite-3.1-3b-a800m-instruct"}
CAPS = (0.10, 0.25, 0.50)
PREDS = ("anterior", "frequente")
DS = (1, 2, 4, 8)
ADMS = ("livre", "lfu")
ESPERAS = (False, True)


class SoEspera(m.Sim):
    """lfu_cpu com a espera do evento e sem busca: o ganho do protocolo sozinho."""

    def fetch_ahead(self, pred, pinned):
        pass


def sim(L, E, k, p, cap, warm, meas, pf=None, so_espera=False):
    if so_espera:
        s = SoEspera("lfu_cpu", cap, L, E, k, p, prefetch={"pred": "oraculo", "d": 1, "admissao": "livre", "espera": True})
    else:
        s = m.Sim("lfu_cpu", cap, L, E, k, p, prefetch=pf)
    s.run(warm, False)
    s.run(meas, True)
    return s.stats


def base(L, E, k, p, cap, warm, meas, espera):
    return sim(L, E, k, p, cap, warm, meas, so_espera=espera)


def red(st, b):
    return 1 - st["wait_ms"] / b["wait_ms"]


def linha(nome, st, b):
    n = st["tokens"]
    return (f"  {nome:34s} espera {st['wait_ms'] / n:6.3f} ms/token ({red(st, b) * 100:+6.2f}%)  acerto {st['hits'] / st['needed']:6.1%}"
            f"  cópias bg {st['copies_bg'] / n:5.1f} + busca {st['copies_pf'] / n:5.1f}/token, úteis {st['pf_useful'] / n:5.1f}"
            f"  camadas com tráfego {st['layers_traffic'] / (n * L_[0]):5.1%}")


L_ = [0]
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]  # noqa: E731
print(f"simulador {sha('tools/moe_sim.py')}, critério {sha('qual-runs/A0-PREFETCH-SIM-20261031/criterio.txt')}")
gate = []
for mod, d in MODS.items():
    p = m.PARAMS_RUNTIME[mod]
    train, test = m.load(d, "treino"), m.load(d, "teste")
    L, _, k = train[0].shape
    L_[0] = L
    E = int(max(a.max() for a in train + test)) + 1
    print(f"\n== {mod}: {L} camadas × {E} experts, k {k}")

    # Escolha no treino (aquecimento 0–1, medida 2–3).
    tw, tm = train[:2], train[2:]
    bases_t = {(c, e): base(L, E, k, p, int(c * L * E), tw, tm, e) for c in CAPS for e in ESPERAS}
    score = {}
    for pr, dd, ad, es in itertools.product(PREDS, DS, ADMS, ESPERAS):
        pf = {"pred": pr, "d": dd, "admissao": ad, "espera": es}
        rs = [red(sim(L, E, k, p, int(c * L * E), tw, tm, pf), bases_t[(c, es)]) for c in CAPS]
        score[(pr, dd, ad, es)] = sum(rs) / len(rs)
    best = max(score, key=score.get)
    print("escolha no treino (redução média da espera nos 3 orçamentos; as 5 melhores):")
    for cfg in sorted(score, key=score.get, reverse=True)[:5]:
        print(f"  {cfg[0]:9s} d={cfg[1]} admissão {cfg[2]:5s} espera {'sim' if cfg[3] else 'não'}: {score[cfg] * 100:+6.2f}%")
    pf_best = {"pred": best[0], "d": best[1], "admissao": best[2], "espera": best[3]}

    # Teste (aquecimento 3, medida 4–7).
    warm = [train[3]]
    for c in CAPS:
        cap = int(c * L * E)
        b0 = base(L, E, k, p, cap, warm, test, False)
        b1 = base(L, E, k, p, cap, warm, test, True)
        bb = b1 if best[3] else b0
        print(f"\n{mod} {c:.0%} ({cap} slots): lfu_cpu espera {b0['wait_ms'] / b0['tokens']:.3f} ms/token")
        print(linha("lfu_cpu com espera do evento", b1, b0))
        st = sim(L, E, k, p, cap, warm, test, pf_best)
        r = red(st, bb)
        gate.append((mod, c, r))
        print(linha(f"escolhida: {best[0]} d={best[1]} {best[2]}", st, bb))
        for dd, ad in itertools.product(DS, ADMS):
            o = sim(L, E, k, p, cap, warm, test, {"pred": "oraculo", "d": dd, "admissao": ad, "espera": True})
            print(linha(f"oráculo d={dd} {ad} (limite)", o, b1))

print("\n== Portão (≥ 5% de redução da espera sobre a base em pelo menos uma célula do teste)")
for mod, c, r in gate:
    print(f"  {mod} {c:.0%}: {r * 100:+.2f}%")
ok = any(r >= 0.05 for _, _, r in gate)
print("portão:", "PASSA (o A2 entra depois do A1)" if ok else "NÃO PASSA (a busca antecipada sai do caminho; o A1 segue sem ela)")
