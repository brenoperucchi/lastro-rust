"""A0c: o custo da residência publicada por token no simulador, pelo criterio.txt desta pasta. Rodar da raiz do projeto:
  taskset -c 16-27 tools/.venv-ref/bin/python -I -B qual-runs/A0C-TOKEN-SIM-20261031/token_sim.py > .../resultado.txt
Determinístico: a mesma árvore dá a mesma saída. Rodada 51: imprime também o tempo total por token e a parada da barreira
(que passou a entrar na espera), e o teto do que a D2 ganharia (as cópias em voo na virada, cada uma a um expert na CPU)."""
import hashlib
import sys

sys.path.insert(0, "tools")
import moe_sim as m  # noqa: E402

T = "qual-runs/TRACOS-20261029"
MODS = {"1b": f"{T}/granite-3.1-1b-a400m-instruct", "3b": f"{T}/granite-3.1-3b-a800m-instruct"}
CAPS = (0.10, 0.25, 0.50)


def sim(L, E, k, p, cap, warm, meas, **kw):
    s = m.Sim("lfu_cpu", cap, L, E, k, p, **kw)
    s.run(warm, False)
    virada = [0]
    if kw.get("protocol") == "token":
        ob = s.begin_token

        def bt(_ob=ob, _s=s):
            _ob()
            virada[0] += sum(f < float("inf") for f in _s.inflight.values())

        s.begin_token = bt
    s.run(meas, True)
    return s.stats, virada[0]


sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]  # noqa: E731
print(f"simulador {sha('tools/moe_sim.py')}, critério {sha('qual-runs/A0C-TOKEN-SIM-20261031/criterio.txt')}")
gate = []
for mod, d in MODS.items():
    p = m.PARAMS_RUNTIME[mod]
    train, test = m.load(d, "treino"), m.load(d, "teste")
    L, _, k = train[0].shape
    E = int(max(a.max() for a in train + test)) + 1
    warm = [train[3]]
    print(f"\n== {mod}: {L} camadas × {E} experts, k {k}")
    for c in CAPS:
        cap = int(c * L * E)
        b, _ = sim(L, E, k, p, cap, warm, test)
        t, vir = sim(L, E, k, p, cap, warm, test, protocol="token")
        tb, _ = sim(L, E, k, p, cap, warm, test, protocol="token", token_barreira=True)
        n = b["tokens"]
        r = t["wait_ms"] / b["wait_ms"] - 1
        gate.append((mod, c, r))
        print(f"\n{mod} {c:.0%} ({cap} slots)")
        for nome, st in (("lfu_cpu, protocolo de hoje", b), ("lfu_cpu, protocolo token", t), ("lfu_cpu, token com barreira", tb)):
            print(f"  {nome:28s} espera {st['wait_ms'] / n:6.3f} ms/token (barreira {st['barrier_ms'] / n:5.3f})  total {st['ms'] / n:7.4f} ms/token"
                  f"  acerto {st['hits'] / st['needed']:6.2%}"
                  f"  CPU {st['cpu'] / n:6.2f}/token  cópias {st['copies_bg'] / n:5.2f}/token (adiadas {st['copies_deferred'] / n:5.2f})"
                  f"  descartes {st['dropped'] / n:6.2f}/token")
        print(f"  token contra hoje: espera {r * 100:+.2f}%; cópias em voo na virada do token {vir} de {t['copies_bg']}"
              f" ({vir / max(1, t['copies_bg']):.1%}); adiadas {t['copies_deferred'] / max(1, t['copies_bg']):.1%} das cópias")
        teto = vir / n * p["c_cpu"]
        print(f"  teto da D2 (em voo na virada × c_cpu): {teto:.4f} ms/token, {teto / (t['wait_ms'] / n):.2%} da espera do token;"
              f" barreira contra token no total: {tb['ms'] / t['ms'] - 1:+.2%}")

print("\n== Portão (espera do protocolo token ≤ 1,02 × a de hoje em todas as células)")
for mod, c, r in gate:
    print(f"  {mod} {c:.0%}: {r * 100:+.2f}%")
ok = all(r <= 0.02 for _, _, r in gate)
print("portão:", "PASSA (a D1 fica como desenho do A1)" if ok else "NÃO PASSA (a D2 volta à mesa antes do A1)")
