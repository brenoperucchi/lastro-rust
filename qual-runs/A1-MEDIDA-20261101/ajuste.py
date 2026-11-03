#!/usr/bin/env python3
"""A1.4d, seção 5 do criterio.txt: refaz c_cpu, c_cpu_quiet e o_layer com as execuções lfu_cpu backlog 4 e backlog 0 da janela.

Para cada modelo e cada repetição i (5 ajustes completos; a incerteza é o intervalo deles):
  - g vem do residente do mesmo bloco (plano, passo 3: "a GPU pelo residente da sessão"): g = ms_res / L - k*h;
  - c_cpu e c_cpu_quiet: mínimos quadrados sem pesos de  CPU ms medido por token = c_cpu * experts_na_CPU_com_tráfego
    + c_cpu_quiet * experts_na_CPU_sem_tráfego  nas 6 linhas (3 orçamentos x backlog 4 e 0); as contagens vêm do Sim
    (lfu_cpu, protocolo runtime, aquecimento nas conversas de treino 0-3, medida nas de teste 4-7) e dependem do próprio
    c_cpu, então itera até o ponto fixo (até 5 iterações, parada com mudança < 1e-5 ms);
  - o_layer: com os c fixos, mínimos quadrados pela origem de (ms medido - ms simulado com o_layer=0) contra o número de
    camadas que pagam o_layer (obtido da diferença entre a simulação com 0 e com 0,0367), só nas linhas backlog 4;
  - (rodada 26) o ponto fixo é conjunto nos três parâmetros; não convergir em 12 voltas é inconclusivo;
  - aceitação: resíduo de ms/token <= 10% em cada uma das 6 células, com os três parâmetros ajustados da repetição.
Uso: taskset -c 16-27 tools/.venv-ref/bin/python -I -B qual-runs/A1-MEDIDA-20261101/ajuste.py
"""
import os, re, statistics as st, sys
from multiprocessing import Pool

R = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(R, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np  # noqa: E402
import moe_sim  # noqa: E402

TR = {"1b": "granite-3.1-1b-a400m-instruct", "3b": "granite-3.1-3b-a800m-instruct"}
BUD = [10, 25, 50]


def mean(v): return sum(v) / len(v)


def ms_file(rep, mod, bl, pref, pol, b):
    f = f"{R}/runs/rep{rep}/{mod}/{bl}/ms/{pref}-{pol}-{b}.txt"
    return mean([float(x) for x in open(f).read().split()])


def cpu_ms(rep, mod, bl):
    """CPU ms por token (coluna da tabela principal do m2a-cache; duas casas) das linhas lfu_cpu, por orçamento."""
    out = {}
    for ln in open(f"{R}/runs/rep{rep}/{mod}/{bl}/saida.txt"):
        if ln.startswith("lfu_cpu"):
            cols = ln.split("|")
            b = int(re.match(r"lfu_cpu\s+(\d+)%", ln).group(1))
            if b not in out:  # a tabela principal vem primeiro
                out[b] = float(cols[3].split()[1])
    return out


def sim(mod, cap, p, backlog, tr, te, L, E, k):
    s = moe_sim.Sim("lfu_cpu", cap, L, E, k, p, protocol="runtime", backlog=backlog)
    s.run(tr, measure=False)
    s.run(te, measure=True)
    st, n = s.stats, s.stats["tokens"]
    return st["ms"] / n, st["cpu_traffic"] / n, st["cpu_quiet"] / n


def job(args):
    mod, rep = args
    d = os.path.join(ROOT, "qual-runs", "TRACOS-20261029", TR[mod])
    tr, te = moe_sim.load(d, "treino"), moe_sim.load(d, "teste")
    L, _, k = tr[0].shape
    E = int(max(a.max() for a in tr + te)) + 1
    base = dict(moe_sim.PARAMS_RUNTIME[mod])
    caps = {b: int(b / 100 * L * E) for b in BUD}
    meas = {}
    for bl, bk in (("m2a-b4", 4), ("m2a-b0", 0)):
        res = ms_file(rep, mod, bl, "m2a-ms", "resident", 100)
        cm = cpu_ms(rep, mod, bl)
        for b in BUD:
            meas[(bk, b)] = (ms_file(rep, mod, bl, "m2a-ms", "lfu_cpu", b), cm[b], res)

    def params(c, cq, o, res):
        p = dict(base)
        p.update(c_cpu=c, c_cpu_quiet=cq, o_layer=o, g=res / L - k * p["h"])
        return p

    # Ponto fixo conjunto de (c_cpu, c_cpu_quiet, o_layer): as contagens dependem dos três (rodada 26). Em cada volta, com o_layer
    # corrente: (1) contagens das 6 linhas e mínimos quadrados de c_cpu e c_cpu_quiet; (2) com esses c, o_layer pelos mínimos
    # quadrados pela origem de (medido - simulado) contra o número de camadas que o pagam (inclinação da simulação em o_layer,
    # em torno do valor corrente), só nas linhas backlog 4. Para quando os três mudam menos que 1e-5 / 1e-5 / 1e-4 ms.
    c, cq, o = base["c_cpu"], base["c_cpu_quiet"], 0.0
    convergiu = False
    hist = []
    for it in range(12):
        rows, y = [], []
        for (bk, b), (m, cmeas, res) in meas.items():
            _, tt, qq = sim(mod, caps[b], params(c, cq, o, res), bk, tr, te, L, E, k)
            rows.append([tt, qq]); y.append(cmeas)
        sol, *_ = np.linalg.lstsq(np.array(rows), np.array(y), rcond=None)
        c_new, cq_new = float(sol[0]), float(sol[1])
        num = den = 0.0
        for b in BUD:
            m, _, res = meas[(4, b)]
            s0, _, _ = sim(mod, caps[b], params(c_new, cq_new, o, res), 4, tr, te, L, E, k)
            s1, _, _ = sim(mod, caps[b], params(c_new, cq_new, o + 0.01, res), 4, tr, te, L, E, k)
            n_elig = (s1 - s0) / 0.01
            num += n_elig * (m - s0)
            den += n_elig * n_elig
        o_new = o + num / den
        mud = (abs(c_new - c), abs(cq_new - cq), abs(o_new - o))
        c, cq, o = c_new, cq_new, o_new
        hist.append((c, cq, o))
        if mud[0] < 1e-5 and mud[1] < 1e-5 and mud[2] < 1e-4:
            convergiu = True
            break
    o_fit = o
    # resíduos com os três parâmetros ajustados
    resid = {}
    for (bk, b), (m, cmeas, res) in meas.items():
        s, _, _ = sim(mod, caps[b], params(c, cq, o_fit, res), bk, tr, te, L, E, k)
        resid[(bk, b)] = s / m - 1
    return mod, rep, c, cq, o_fit, (it + 1) if convergiu else -(it + 1), resid, hist[-2:]


if __name__ == "__main__":
    jobs = [(m, r) for m in ("1b", "3b") for r in range(1, 6)]
    with Pool(10) as pool:
        out = pool.map(job, jobs)
    lines = []
    def p(*a): lines.append(" ".join(str(x) for x in a))
    p("=== 5. Ajuste de c_cpu, c_cpu_quiet e o_layer com a janela (ms; resíduo = simulado/medido - 1) ===")
    allpass = True
    for mod in ("1b", "3b"):
        rs = [x for x in out if x[0] == mod]
        base = moe_sim.PARAMS_RUNTIME[mod]
        p(f"\n{mod}: valores em uso c_cpu {base['c_cpu']:.4f} c_cpu_quiet {base['c_cpu_quiet']:.4f} o_layer {base['o_layer']:.4f}")
        for _, rep, c, cq, o, it, resid, ult in rs:
            worst = max(abs(v) for v in resid.values())
            p(f"  rep{rep}: c_cpu {c:.4f} c_cpu_quiet {cq:.4f} o_layer {o:+.4f} ({"convergiu em " + str(it) + " voltas" if it > 0 else "NÃO CONVERGIU em " + str(-it) + " voltas; últimas duas voltas o_layer " + f"{ult[0][2]:.4f} e {ult[1][2]:.4f}, c_cpu {ult[0][0]:.4f} e {ult[1][0]:.4f}"})  resíduo máx {100*worst:.1f}%  "
              + " ".join(f"b{bk}/{b}%:{100*v:+.1f}" for (bk, b), v in sorted(resid.items())))
        conv = [x for x in rs if x[5] > 0]
        for nome, g in (("5 repetições (o do critério)", rs), (f"{len(conv)} que convergiram (regra da rodada 26, depois dos dados)", conv)):
            cs = [x[2] for x in g]; cqs = [x[3] for x in g]; os_ = [x[4] for x in g]
            p(f"  {nome}: c_cpu [{min(cs):.4f},{max(cs):.4f}] mediana {st.median(cs):.4f}  c_cpu_quiet [{min(cqs):.4f},{max(cqs):.4f}] mediana {st.median(cqs):.4f}  "
              f"o_layer [{min(os_):+.4f},{max(os_):+.4f}] mediana {st.median(os_):+.4f}")
        os_ = [x[4] for x in rs]
        wmax = max(max(abs(v) for v in x[6].values()) for x in rs)
        inconcl = min(os_) <= 0 and max(os_) >= moe_sim.O_LAYER_RUNTIME
        todas = len(conv) == len(rs)
        p(f"  resíduo: {'passa' if wmax <= 0.10 else 'NÃO passa'} (pior {100*wmax:.1f}%); convergência: {len(conv)} de {len(rs)} repetições; "
          f"o_layer: {'inconclusivo (o intervalo cobre 0 e 0,0367)' if inconcl else 'distingue de 0 e de 0,0367'}")
        p(f"  ajuste deste modelo: {'PASSA' if (wmax <= 0.10 and todas) else 'INCONCLUSIVO (repetições sem convergência)' if wmax <= 0.10 else 'NÃO PASSA'}")
        allpass &= (wmax <= 0.10 and todas)
    p("\nos três ajustes passam e convergem em todos os modelos:", "sim" if allpass else "não (ver o ajuste de cada modelo acima)")
    open(f"{R}/ajuste.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
