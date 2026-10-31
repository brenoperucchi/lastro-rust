"""A0c, conferência depois do resultado (não decide o portão): no protocolo de hoje, quantas vítimas de uma camada à
frente são pedidas de novo no mesmo token (viram falta na hora) e quantas são readmitidas. No protocolo token, essas
continuam acerto até o fim do token. Rodar da raiz:
  taskset -c 16-27 tools/.venv-ref/bin/python -I -B qual-runs/A0C-TOKEN-SIM-20261031/diagnostico.py > .../diagnostico.txt"""
import sys

sys.path.insert(0, "tools")
import moe_sim as m  # noqa: E402

T = "qual-runs/TRACOS-20261029"
MODS = {"1b": f"{T}/granite-3.1-1b-a400m-instruct", "3b": f"{T}/granite-3.1-3b-a800m-instruct"}

for mod, d in MODS.items():
    p = m.PARAMS_RUNTIME[mod]
    train, test = m.load(d, "treino"), m.load(d, "teste")
    L, _, k = train[0].shape
    E = int(max(a.max() for a in train + test)) + 1
    for c in (0.10, 0.25, 0.50):
        s = m.Sim("lfu_cpu", int(c * L * E), L, E, k, p)
        s.run([train[3]], False)
        cont = {"vitimas": 0, "a_frente": 0, "pedidas_no_token": 0, "readmitidas_no_token": 0}
        frente = {}  # vítima de camada à frente -> token em que saiu
        tok = [0]
        om, oa, ol = s.make_room, s.admit_bg, s.layer

        def make_room(pinned, candidate_prio=None, _s=s):
            ok = om(pinned, candidate_prio)
            v = _s.last_victim
            if ok and v is not None:
                cont["vitimas"] += 1
                if v[0] >= _s.cur_layer:
                    cont["a_frente"] += 1
                    frente[v] = tok[0]
            return ok

        def admit_bg(key, pinned, _s=s):
            n0 = _s.stats["copies_bg"]
            oa(key, pinned)
            if _s.stats["copies_bg"] > n0 and frente.get(key) == tok[0]:
                cont["readmitidas_no_token"] += 1

        def layer(l, experts, *a, _s=s):
            if l == 0:
                tok[0] += 1
            for e in experts:
                if frente.get((l, int(e))) == tok[0]:
                    cont["pedidas_no_token"] += 1
            return ol(l, experts, *a)

        s.make_room, s.admit_bg, s.layer = make_room, admit_bg, layer
        s.run(test, True)
        n = s.stats["tokens"]
        print(f"{mod} {c:.0%}: vítimas {cont['vitimas'] / n:.2f}/token, de camada à frente {cont['a_frente'] / n:.2f};"
              f" pedidas de novo no mesmo token {cont['pedidas_no_token'] / n:.3f}/token, readmitidas {cont['readmitidas_no_token'] / n:.3f}/token;"
              f" cópias {s.stats['copies_bg'] / n:.2f}/token")
