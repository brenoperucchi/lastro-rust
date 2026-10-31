"""A0a, rodada 51 (achado 5 do rev-2 na rodada 50): qual termo do custo do tráfego decide o resultado, e o limiar de
disputa da CPU que reabriria a busca antecipada. Não muda o portão da A0a. Rodar da raiz:
  taskset -c 24-27 tools/.venv-ref/bin/python -I -B qual-runs/A0-PREFETCH-SIM-20261031/sensibilidade.py > .../sensibilidade.txt

Protocolo de teste da A0a (aquecimento na conversa 3, medida nas 4–7), base lfu_cpu sem busca e sem espera, busca com
admissão livre e d = 8 (a melhor distância no treino), sem espera (a espera não muda nada, A0a). A admissão lfu fica de
fora: ela quase não copia (diagnostico.txt). Variantes dos parâmetros "runtime":
  calibrado           os da A0a;
  o_layer=0           sem a sobra por camada com tráfego;
  o_layer=0, sem disputa   e com c_cpu = c_cpu_quiet (a CPU não perde nada com cópia em voo);
  disputa f           o_layer calibrado e c_cpu = c_cpu_quiet + f·(c_cpu − c_cpu_quiet), f = 0, 0,25, 0,5 e 0,75
                      (f = 1 é o calibrado). O limiar de reabertura é o maior f em que um preditor realista passa de 5%
                      em alguma célula."""
import hashlib
import sys

sys.path.insert(0, "tools")
import moe_sim as m  # noqa: E402

T = "qual-runs/TRACOS-20261029"
MODS = {"1b": f"{T}/granite-3.1-1b-a400m-instruct", "3b": f"{T}/granite-3.1-3b-a800m-instruct"}
CAPS = (0.10, 0.25, 0.50)
PREDS = ("oraculo", "anterior", "frequente")


def variantes(p):
    q = p["c_cpu_quiet"]
    yield "calibrado", dict(p)
    yield "o_layer=0", dict(p, o_layer=0.0)
    yield "o_layer=0, sem disputa", dict(p, o_layer=0.0, c_cpu=q)
    for f in (0.0, 0.25, 0.5, 0.75):
        yield f"disputa f={f:.2f}", dict(p, c_cpu=q + f * (p["c_cpu"] - q))


def sim(L, E, k, p, cap, warm, meas, pf=None):
    s = m.Sim("lfu_cpu", cap, L, E, k, p, prefetch=pf)
    s.run(warm, False)
    s.run(meas, True)
    return s.stats["wait_ms"]


sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]  # noqa: E731
print(f"simulador {sha('tools/moe_sim.py')}")
passa = {}
for mod, d in MODS.items():
    p0 = m.PARAMS_RUNTIME[mod]
    train, test = m.load(d, "treino"), m.load(d, "teste")
    L, _, k = train[0].shape
    E = int(max(a.max() for a in train + test)) + 1
    warm = [train[3]]
    print(f"\n== {mod} (c_cpu {p0['c_cpu']}, c_cpu_quiet {p0['c_cpu_quiet']}, o_layer {p0['o_layer']})")
    print("  redução da espera sobre o lfu_cpu, 10 / 25 / 50%, busca livre d = 8:")
    for nome, p in variantes(p0):
        linha = []
        for pred in PREDS:
            rs = []
            for c in CAPS:
                cap = int(c * L * E)
                b = sim(L, E, k, p, cap, warm, test)
                w = sim(L, E, k, p, cap, warm, test, {"pred": pred, "d": 8, "admissao": "livre", "espera": False})
                r = 1 - w / b
                rs.append(r)
                if pred != "oraculo" and r >= 0.05:
                    passa.setdefault(nome, []).append(f"{mod} {c:.0%} {pred} {r:+.1%}")
            linha.append(f"{pred} " + " / ".join(f"{r * 100:+6.1f}" for r in rs))
        print(f"  {nome:24s} " + "   ".join(linha))

print("\n== Preditores realistas com ≥ 5% (o que reabriria o A2)")
for nome, _ in variantes(m.PARAMS_RUNTIME["1b"]):
    print(f"  {nome:24s} " + (", ".join(passa[nome]) if nome in passa else "nenhum"))
