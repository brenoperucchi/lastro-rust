"""A0a, conferências depois do resultado (não decidem o portão): por que a busca com admissão lfu quase não copia, por que a
espera do evento sozinha dá zero, e quantas cópias ainda estão em voo na virada do token (para a A0c). Rodar da raiz:
  taskset -c 16-27 tools/.venv-ref/bin/python -I -B qual-runs/A0-PREFETCH-SIM-20261031/diagnostico.py > .../diagnostico.txt"""
import sys

sys.path.insert(0, "tools")
import moe_sim as m  # noqa: E402

T = "qual-runs/TRACOS-20261029"
MODS = {"1b": f"{T}/granite-3.1-1b-a400m-instruct", "3b": f"{T}/granite-3.1-3b-a800m-instruct"}


class EmVoo(dict):
    """Conta as consultas `chave in inflight` verdadeiras feitas durante a medida."""
    n = 0
    on = False

    def __contains__(self, k):
        r = dict.__contains__(self, k)
        if r and self.on:
            EmVoo.n += 1
        return r


for mod, d in MODS.items():
    p = m.PARAMS_RUNTIME[mod]
    train, test = m.load(d, "treino"), m.load(d, "teste")
    L, _, k = train[0].shape
    E = int(max(a.max() for a in train + test)) + 1
    print(f"== {mod}")
    for c in (0.10, 0.25, 0.50):
        cap = int(c * L * E)
        for pf in ({"pred": "anterior", "d": 8, "admissao": "lfu", "espera": True},
                   {"pred": "oraculo", "d": 8, "admissao": "lfu", "espera": True}):
            s = m.Sim("lfu_cpu", cap, L, E, k, p, prefetch=pf)
            s.run([train[3]], False)
            s.run(test, True)
            st = s.stats
            print(f"  {c:.0%} busca {pf['pred']} d=8 lfu: propostas descartadas {st['pf_dropped']}, cópias {st['copies_pf']}, úteis {st['pf_useful']}")
        s = m.Sim("lfu_cpu", cap, L, E, k, p)
        s.inflight = EmVoo(s.inflight)
        s.run([train[3]], False)
        EmVoo.n, s.inflight.on = 0, True
        virada = [0, 0]
        orig = s.layer

        def layer(l, experts, *a, _o=orig, _s=s):
            if l == 0:
                _s.inflight.on = False
                _s.settle()
                virada[0] += len(_s.inflight)
                virada[1] += 1
                _s.inflight.on = True
            return _o(l, experts, *a)

        s.layer = layer
        s.run(test, True)
        nb = s.stats["copies_bg"]
        print(f"  {c:.0%} lfu_cpu: chave pedida ainda em voo {EmVoo.n} vezes em {s.stats['needed']} pedidos;"
              f" cópias em voo na virada do token {virada[0]} de {nb} ({virada[0] / max(1, nb):.1%}) em {virada[1]} tokens")
