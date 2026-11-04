#!/usr/bin/env python3
"""A2-LOTE: aplica o custo de lote MEDIDO do expert (lote_gpu e lote-cpu) ao SimM do A2-MULTIPOS-SIM. Critério: criterio.txt (v1.2).
Uso: tools/.venv-ref/bin/python -I -B qual-runs/A2-LOTE-20261103/lote.py <gpu.txt> <cpu.txt>     (sem GPU; só lê as duas saídas)
A calibração em M=1 é a do A2-MULTIPOS-SIM (w(1)=1 por definição, nada muda em M=1) e não se repete aqui."""
import math, os, re, sys
R = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(R, "..", "A2-MULTIPOS-SIM-20261103"))
import multipos as MP  # noqa: E402

NS = (1, 2, 4, 8, 16, 32)
TOL_GPU_ERR, TOL_CPU_ERR = 1e-3, 1e-5          # portão de correção (criterio.txt, seção 1)
TOL_GPU_T1, TOL_CPU_T1 = 0.40, 0.30            # portão de representatividade em n=1 (criterio.txt, seção 1)
UUID_4090 = "GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd"
REPS = 15
RE_GPU = re.compile(r"^GPU (1b|3b) n=(\d+) lote_us=(\S+) serial_us=(\S+) err_lote_serial=(\S+) err_ref=(\S+)\s*$")
RE_CPU = re.compile(r"^CPU (1b|3b) n=(\d+) us_por_expert=(\S+) max_err_rel=(\S+)\s*$")


def pior(*erros):
    """Maior erro; qualquer erro NaN, infinito ou negativo contamina (o max ignoraria o NaN e esconderia um campo inválido menor)."""
    return float("nan") if any(not math.isfinite(e) or e < 0 for e in erros) else max(erros)


def parse(texto, lado):
    """Devolve {(modelo, n): dict}. Linha repetida para o mesmo (modelo, n) é erro (nada de escolher a melhor)."""
    out = {}
    for ln in texto.splitlines():
        m = (RE_GPU if lado == "gpu" else RE_CPU).match(ln)
        if not m:
            continue
        mod, n = m.group(1), int(m.group(2))
        assert (mod, n) not in out, f"{lado}: linha repetida para {mod} n={n}"
        if lado == "gpu":
            out[(mod, n)] = {"lote": float(m.group(3)), "serial": float(m.group(4)), "err": pior(float(m.group(5)), float(m.group(6)))}
        else:
            out[(mod, n)] = {"lote": float(m.group(3)), "err": float(m.group(4))}
    return out


def cabecalho(texto, lado):
    """Motivos de reprovação do cabeçalho: modo completo com 15 repetições e (GPU) a 4090 pelo UUID."""
    mot = []
    m = re.search(r"^# modo=(\S+) reps=(\d+)", texto, re.M)
    if not m or m.group(1) != "completo" or int(m.group(2)) != REPS:
        mot.append(f"{lado}: sem a linha '# modo=completo reps={REPS}' (teste rápido ou saída sem cabeçalho não decide)")
    if lado == "GPU":
        u = re.search(r"^# dispositivo: .* uuid=(\S+)", texto, re.M)
        if not re.search(r"^# variante=[012]\s*$", texto, re.M):
            mot.append("GPU: sem a linha '# variante=0|1|2'")
        if not u or u.group(1) != UUID_4090:
            mot.append(f"GPU: sem a linha '# dispositivo: ... uuid={UUID_4090}' (não provado que rodou na 4090)")
    return mot


def validos(tab, lado):
    """Tempos finitos e > 0 e erros finitos e ≥ 0 em todos os pontos (antes de qualquer razão)."""
    mot = []
    for (mod, n), r in sorted(tab.items()):
        tempos = [r["lote"]] + ([r["serial"]] if "serial" in r else [])
        if not all(math.isfinite(t) and t > 0 for t in tempos):
            mot.append(f"{lado} {mod} n={n}: tempo inválido {tempos}")
        if not (math.isfinite(r["err"]) and r["err"] >= 0):
            mot.append(f"{lado} {mod} n={n}: erro inválido {r['err']}")
    return mot


CAB_SEL = re.compile(r"^# modo=so-n1 reps=15 replays=10\s*$", re.M)


def escolhe_variante(textos):
    """Seleção da variante do kernel de GPU (criterio.txt v1.2, seção 6): textos = {variante: saída de `lote_gpu --so-n1 --var=V`}, EXATAMENTE
    as variantes 0, 1 e 2, uma saída cada (ausência, duplicata ou variante fora de {0,1,2} é erro: ValueError). Cada saída tem de ter
    `# modo=so-n1 reps=15 replays=10`, o UUID da 4090 e a própria variante. Vale a de menor max sobre os dois modelos de |t(1)/h − 1| (h do
    motor); empate, a de menor número; variante com saída inválida, correção reprovada ou tempo inválido não concorre (continua registrada).
    Só olha n=1 e a correção: não usa o w(n). Devolve (variante, {variante: desvio ou None se excluída})."""
    if sorted(textos) != [0, 1, 2]:
        raise ValueError(f"a seleção exige exatamente as variantes 0, 1 e 2, uma saída cada (vieram {sorted(textos)})")
    desv = {}
    for v, t in sorted(textos.items()):
        desv[v] = None
        tab = parse(t, "gpu")
        ok_cab = CAB_SEL.search(t) and re.search(rf"^# variante={v}\s*$", t, re.M) and re.search(rf"^# dispositivo: .* uuid={re.escape(UUID_4090)}\s*$", t, re.M)
        if not ok_cab or set(tab) != {("1b", 1), ("3b", 1)} or validos(tab, "GPU"):
            continue
        if not all(tab[(mod, 1)]["err"] <= TOL_GPU_ERR for mod in ("1b", "3b")):
            continue
        desv[v] = max(abs(tab[(mod, 1)]["lote"] / (MP.moe_sim.PARAMS_RUNTIME[mod]["h"] * 1000) - 1) for mod in ("1b", "3b"))
    vivas = {v: d for v, d in desv.items() if d is not None}
    if not vivas:
        return None, desv
    return min(vivas, key=lambda v: (vivas[v], v)), desv


def le_escolha(caminho):
    """Variante escrita no escolha.txt (linha 'VARIANTE ESCOLHIDA: V') ou None."""
    try:
        with open(caminho) as f:
            m = re.search(r"^VARIANTE ESCOLHIDA: ([012])\s*$", f.read(), re.M)
    except OSError:
        return None
    return int(m.group(1)) if m else None


def portoes(gpu, cpu):
    """Lista de motivos de reprovação (vazia = passou): tabela completa, valores válidos, correção e representatividade em n=1."""
    mot = []
    for lado, tab, tol in (("GPU", gpu, TOL_GPU_ERR), ("CPU", cpu, TOL_CPU_ERR)):
        for mod in ("1b", "3b"):
            for n in NS:
                r = tab.get((mod, n))
                if r is None:
                    mot.append(f"{lado} {mod} n={n}: ausente")
                elif not (r["err"] <= tol):
                    mot.append(f"{lado} {mod} n={n}: correção {r['err']:.2e} > {tol:.0e}")
    if mot:
        return mot
    mot = validos(gpu, "GPU") + validos(cpu, "CPU")
    if mot:
        return mot
    for mod in ("1b", "3b"):
        p = MP.moe_sim.PARAMS_RUNTIME[mod]
        for lado, tab, ref, tol in (("GPU", gpu, p["h"] * 1000, TOL_GPU_T1), ("CPU", cpu, p["c_cpu_quiet"] * 1000, TOL_CPU_T1)):
            t1 = tab[(mod, 1)]["lote"]
            if abs(t1 / ref - 1) > tol:
                mot.append(f"{lado} {mod} n=1: {t1:.2f} us contra {ref:.2f} us do motor ({t1 / ref - 1:+.0%}, limite ±{tol:.0%})")
    return mot


def tabela_w(tab, mod, campo="lote"):
    """w(n) = t(n)/t(1), interpolação linear em n entre os pontos medidos; n > 32 não ocorre."""
    pts = [(n, tab[(mod, n)][campo] / tab[(mod, 1)][campo]) for n in NS]

    def w(n):
        assert 1 <= n <= NS[-1], n
        for (n0, w0), (n1, w1) in zip(pts, pts[1:]):
            if n <= n1:
                return w0 + (w1 - w0) * (n - n0) / (n1 - n0)
    return w, pts


class SimMed(MP.SimM):
    """SimM com o custo do expert medido (GPU e CPU) e g_M = g ('g1') ou M·g ('gM')."""
    def __init__(self, mod, frac, L, E, k, wg, wc, gm_modo, bw=MP.BW_GBS):
        super().__init__(mod, frac, L, E, k, "otimista", bw)
        self._wg, self._wc, self.gm_modo = wg, wc, gm_modo

    def wg(self, n):
        return self._wg(n)

    def wc(self, n):
        return self._wc(n)

    def gm(self):
        return self.M if self.gm_modo == "gM" else 1


def roda(mod, frac, M, wg, wc, gm_modo, treino, teste):
    L, _, k = treino[0].shape
    E = int(max(a.max() for a in treino + teste)) + 1
    s = SimMed(mod, frac, L, E, k, wg, wc, gm_modo)
    s.run_blocks(treino, M, False)
    s.run_blocks(teste, M, True)
    return s.stats["ms"] / s.stats["tokens"]


def decide(ganhos, portao_ok):
    """ganhos[(mod, gm_modo, M)]; critério seção 3."""
    if not portao_ok:
        return "SEM DECISÃO (portão reprovado)"
    def ok(modo):
        return all(max(ganhos[(m, modo, M)] for M in MP.MS if M <= MP.M_MAX) >= MP.GANHO_MIN for m in ("1b", "3b"))
    if ok("gM"):
        return "VALE PROTOTIPAR NO MOTOR (sob as hipóteses)"
    if ok("g1"):
        return "DEPENDE DO g_M (medir atenção e roteador em lote)"
    return "NÃO VALE AGORA"


def calcula(gpu, cpu, carrega=MP.carrega, fracs=(0.10,)):
    """ganhos[(mod, frac, modo, M)] e tempos[(mod, frac, modo, M)] (ms/pos)."""
    ganhos, tempos = {}, {}
    for mod in ("1b", "3b"):
        tr, te = carrega(mod)
        wg, _ = tabela_w(gpu, mod)
        wc, _ = tabela_w(cpu, mod)
        for frac in fracs:
            for modo in ("g1", "gM"):
                t1 = None
                for M in MP.MS:
                    t = roda(mod, frac, M, wg, wc, modo, tr, te)
                    t1 = t if M == 1 else t1
                    ganhos[(mod, frac, modo, M)], tempos[(mod, frac, modo, M)] = 1 - t / t1, t
    return ganhos, tempos


def principal(gpu_txt, cpu_txt, carrega=MP.carrega, escolha=None):
    with open(gpu_txt) as f, open(cpu_txt) as h:
        tg, tc = f.read(), h.read()
    gpu, cpu = parse(tg, "gpu"), parse(tc, "cpu")
    mot = cabecalho(tg, "GPU") + cabecalho(tc, "CPU") + portoes(gpu, cpu)
    v = le_escolha(escolha) if escolha else None
    m = re.search(r"^# variante=([012])\s*$", tg, re.M)
    if v is None:
        mot.append("sem escolha.txt válido (a variante da execução completa tem de ser a escolhida pela seleção)")
    elif not m or int(m.group(1)) != v:
        mot.append(f"a variante da GPU ({m.group(1) if m else '?'}) não é a escolhida pela seleção ({v})")
    if not mot:
        print("=== Tempo medido por expert (us) e w(n) = t(n)/t(1) ===")
        for mod in ("1b", "3b"):
            for lado, tab in (("GPU", gpu), ("CPU", cpu)):
                print(f"  {lado} {mod}: " + "  ".join(f"n={n}: {tab[(mod, n)]['lote']:.2f} (w {tab[(mod, n)]['lote'] / tab[(mod, 1)]['lote']:.2f})" for n in NS))
            print(f"  GPU {mod} serial (informativo; da 2ª execução em diante os pesos vêm da L2, não é o w=n da DRAM): " + "  ".join(f"n={n}: {gpu[(mod, n)]['serial']:.2f}" for n in NS))
    print("\n=== Portões ===")
    print("  " + ("passaram (cabeçalho, tabela completa, valores válidos, correção, representatividade em n=1)" if not mot else "REPROVADOS:"))
    for m in mot:
        print("   -", m)
    if mot:
        print("\n  DECISÃO:", decide({}, False))
        return 1
    ganhos, tempos = calcula(gpu, cpu, carrega, (0.10, 0.25))
    for frac in (0.10, 0.25):
        print(f"\n=== Ganho contra M=1 ({frac:.0%} de slots, conversas de teste){'' if frac == 0.10 else '  [descritivo, não decide]'} ===")
        for mod in ("1b", "3b"):
            for modo in ("g1", "gM"):
                print(f"\n{mod} {modo}\n    M   ms/pos    ganho    a_eq   folga ms/passo")
                t1 = tempos[(mod, frac, modo, 1)]
                for M in MP.MS:
                    t = tempos[(mod, frac, modo, M)]
                    aeq, folga = MP.metricas(t1, t, M)
                    print(f"  {M:>3}  {t:7.3f}  {ganhos[(mod, frac, modo, M)]:7.1%}  {aeq:6.2f}  {folga:12.2f}")
    g10 = {(m, modo, M): v for (m, f, modo, M), v in ganhos.items() if f == 0.10}
    print(f"\n=== Regra (criterio.txt seção 3; 10% de slots; M ≤ {MP.M_MAX}, ganho mínimo {MP.GANHO_MIN:.0%}) ===")
    for modo in ("gM", "g1"):
        for mod in ("1b", "3b"):
            g = max(g10[(mod, modo, M)] for M in MP.MS if M <= MP.M_MAX)
            print(f"  {mod} {modo}: maior ganho com M ≤ {MP.M_MAX} = {g:.1%} -> {'≥' if g >= MP.GANHO_MIN else '<'} {MP.GANHO_MIN:.0%}")
    print("  DECISÃO:", decide(g10, True))
    print("  Ressalva: o kernel de GPU não tem x em memória compartilhada nem pipeline de posições; para n ≥ 16 o tempo medido é conservador (a regra só olha M ≤ 8).")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "--escolhe":
        import hashlib
        txt, shas = {}, {}
        for f in sys.argv[2:]:
            with open(f, "rb") as h:
                raw = h.read()
            m = re.search(rb"^# variante=(\d+)", raw, re.M)
            if not m or int(m.group(1)) in txt:
                sys.exit(f"erro: {f}: sem variante ou variante repetida")
            txt[int(m.group(1))] = raw.decode()
            shas[int(m.group(1))] = (f, hashlib.sha256(raw).hexdigest())
        try:
            v, d = escolhe_variante(txt)
        except ValueError as e:
            sys.exit(f"erro: {e}")
        saida = ["desvios |t(1)/h-1| (pior dos dois modelos; None = excluída):"] + [f"  variante {k}: {x if x is None else round(x, 4)}  {shas[k][0]} sha256={shas[k][1]}" for k, x in sorted(d.items())]
        saida.append(f"VARIANTE ESCOLHIDA: {v}")
        print("\n".join(saida))
        sys.exit(0 if v is not None else 1)
    sys.exit(principal(sys.argv[1], sys.argv[2], escolha=sys.argv[3] if len(sys.argv) > 3 else None))
