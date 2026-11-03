#!/usr/bin/env python3
"""A1-DIAG10: classificador da árvore de decisão do criterio.txt (versão 6, aprovada pelos dois revisores na rodada 34), seção 5.
Só aritmética sobre números por repetição; não lê arquivos de medida e não toca a GPU. Os testes sintéticos estão em test_decide.py.
Uso: taskset -c 16-27 tools/.venv-ref/bin/python -I -B -m unittest qual-runs/A1-DIAG10-20261103/test_decide.py   (da raiz do projeto)

Entrada: uma configuração é uma lista de 5 repetições; cada repetição é um dict {"L1","L2","L3","L4"} em ms por token.
  d_i = (L1 - L4) - (L2 - L3).
Os limiares são os do criterio.txt e não mudam depois de ver número."""
from statistics import median

# --- limiares do criterio.txt (seção 5) ---
D_MIN = 0.3        # 0a: mediana de d >= 0,3 ms
BUILD_TOL = 0.1    # 0b: |mediana d(N) - mediana d(C)| <= 0,1 ms
BASE_TOL = 0.05    # condição de base: |dL3|, |dL4| <= 0,05 ms
R_LOW, R_HIGH = 0.3, 0.7
OPP = 0.1          # efeitos opostos: cada um por mais de 0,1 ms
NPAIRS = 5
EPS = 1e-9        # as fronteiras do criterio.txt são inclusivas (>= 0,3; >= 0,7; <= 0,05; <= 0,1); EPS só absorve o erro de ponto flutuante

# saídas
NAO_REPRODUZIDO = "NAO_REPRODUZIDO"
SEM_CONCLUSAO_BUILD = "SEM_CONCLUSAO_BUILD"
SEM_CONCLUSAO_TRABALHO = "SEM_CONCLUSAO_TRABALHO"
SEM_CONCLUSAO_BASE = "SEM_CONCLUSAO_BASE"
SEM_CONCLUSAO_NAO_MOTOR = "SEM_CONCLUSAO_NAO_MOTOR"
NAO_SUSTENTADA = "NAO_SUSTENTADA"
PARCIAL = "PARCIAL"
SUSTENTADA = "SUSTENTADA"
INCONCLUSIVO_OPOSTOS = "INCONCLUSIVO_OPOSTOS"
SAIDAS = {NAO_REPRODUZIDO, SEM_CONCLUSAO_BUILD, SEM_CONCLUSAO_TRABALHO, SEM_CONCLUSAO_BASE, SEM_CONCLUSAO_NAO_MOTOR,
          NAO_SUSTENTADA, PARCIAL, SUSTENTADA, INCONCLUSIVO_OPOSTOS}


def d_list(cfg):
    return [(r["L1"] - r["L4"]) - (r["L2"] - r["L3"]) for r in cfg]


def reproduz(d):
    """0a: mediana >= 0,3 ms e os 5 d_i > 0."""
    return len(d) == NPAIRS and median(d) >= D_MIN - EPS and all(x > 0 for x in d)


def passo0_etapa1(C, N):
    """Passo 0 com os binários congelados C e o novo sem instrumentação N@A0."""
    dC, dN = d_list(C), d_list(N)
    if not reproduz(dC):
        return NAO_REPRODUZIDO
    if not reproduz(dN) or abs(median(dN) - median(dC)) > BUILD_TOL + EPS:
        return SEM_CONCLUSAO_BUILD
    return "OK"


def par_valido(ref, alt, tokens, c_copy_cpu):
    """Controle de trabalho de um par: uma inequação por contador, sem compensação. Contadores por execução: hits, cpu, dropped, copies
    (deferred é subconjunto das cópias, só reportado). Acertos, experts na CPU e descartes dentro de +-0,1% da referência; cópias pelo custo
    convencional |dif| x c_copy_cpu / tokens <= 0,01 ms por token."""
    for k in ("hits", "cpu", "dropped"):
        if abs(alt[k] - ref[k]) > 0.001 * ref[k] + EPS:
            return False
    return abs(alt["copies"] - ref["copies"]) * c_copy_cpu / tokens <= 0.01 + EPS


def pares_validos(refs, alts, tokens, c_copy_cpu):
    return [par_valido(r, a, tokens, c_copy_cpu) for r, a in zip(refs, alts)]


def fator(N0, N1, d_etapa1_med):
    """Regra (a) para um fator x (N1 = N@A1x) contra a referência contemporânea N0 = N@A0 da etapa 2.
    Devolve (saida, detalhes). O controle de trabalho é aplicado antes por quem chama (pares inválidos saem das listas)."""
    d0 = d_list(N0)
    # portão reaplicado à referência da etapa 2: 0a e 0b contra a mediana de d(N) da etapa 1
    if not reproduz(d0):
        return NAO_REPRODUZIDO, {}
    if abs(median(d0) - d_etapa1_med) > BUILD_TOL + EPS:
        return SEM_CONCLUSAO_BUILD, {}
    n = len(N0)
    dl = lambda k: median(N1[i][k] - N0[i][k] for i in range(n))  # dLk = mediana( Lk(A1x) - Lk(A0) )
    det = {"dL1": dl("L1"), "dL2": dl("L2"), "dL3": dl("L3"), "dL4": dl("L4")}
    if abs(det["dL4"]) > BASE_TOL + EPS or abs(det["dL3"]) > BASE_TOL + EPS:
        return SEM_CONCLUSAO_BASE, det
    m = [((N0[i]["L1"] - N0[i]["L4"]) - (N1[i]["L1"] - N1[i]["L4"])) for i in range(n)]   # redução da espera do motor
    c = [((N0[i]["L2"] - N0[i]["L3"]) - (N1[i]["L2"] - N1[i]["L3"])) for i in range(n)]   # redução da espera do comparador
    r = [m[i] - c[i] for i in range(n)]
    R = median(r) / median(d0)
    Mfr = median(m) / median(d0)
    det.update(R=R, Mfr=Mfr, r=r, m=m, c=c, r_med=median(r))
    if R < R_LOW - EPS:
        return NAO_SUSTENTADA, det
    if Mfr < R_LOW - EPS:
        return SEM_CONCLUSAO_NAO_MOTOR, det
    if R >= R_HIGH - EPS and Mfr >= R_HIGH - EPS and all(x > EPS for x in r) and all(x > EPS for x in m):
        return SUSTENTADA, det
    return PARCIAL, det


def combina(res_w, res_m):
    """Combina A1w e A1m numa célula. Efeitos opostos (cada um move d em sentido contrário por mais de 0,1 ms): INCONCLUSIVO, nunca 'sem disputa'.
    res_* = (saida, detalhes). Sem 'r_med' nos detalhes (portão ou base falharam) não há efeito para comparar."""
    (sw, dw), (sm, dm) = res_w, res_m
    if "r_med" in dw and "r_med" in dm:
        a, b = dw["r_med"], dm["r_med"]
        if (a > OPP + EPS and b < -OPP - EPS) or (b > OPP + EPS and a < -OPP - EPS):
            return INCONCLUSIVO_OPOSTOS
    return {"w": sw, "m": sm}


def celula(C, N0_e1, e2, tokens, c_copy_cpu):
    """Pipeline completo de uma célula (1B 10% ou 3B 10%): passo 0 -> controle de trabalho -> condição de base -> regra (a).
    e2 = {"N0": [...], "A1w": [...], "A1m": [...], "cont": {"N0": [...], "A1w": [...], "A1m": [...]}} com contadores por repetição
    (ou "valid": {"A1w": [bool x5], "A1m": [bool x5]} no lugar de "cont")."""
    p0 = passo0_etapa1(C, N0_e1)
    if p0 != "OK":
        return p0
    d_e1 = median(d_list(N0_e1))
    resultados = {}
    for x in ("A1w", "A1m"):
        # "valid" (opcional): a validade dos pares já calculada por quem lê os arquivos (duas linhas por par, L1 e L2); sem ele, os contadores de uma linha.
        v = e2["valid"][x] if "valid" in e2 else pares_validos(e2["cont"]["N0"], e2["cont"][x], tokens, c_copy_cpu)
        if sum(v) < NPAIRS:
            resultados[x] = (SEM_CONCLUSAO_TRABALHO, {})
            continue
        resultados[x] = fator(e2["N0"], e2[x], d_e1)
    return combina(resultados["A1w"], resultados["A1m"])
