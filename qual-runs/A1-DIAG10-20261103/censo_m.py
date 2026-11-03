#!/usr/bin/env python3
"""Censo da união de experts por camada em M posições consecutivas (passo multiposição, primeiro passo; plano, nota de 02/11; consulta de 03/11).
Só contagem nos traços de roteamento gravados (nada de simulador de tempo, nada de GPU). Cada conversa é percorrida em blocos de M tokens consecutivos
SEM atravessar o fim da conversa; o último bloco incompleto é descartado e contado. Treino (conversas 0 a 3) e teste (4 a 7) ficam separados.
Ressalva (corrigida na rodada 29): os traços vêm de UMA passada causal sobre um texto fixo por conversa (meta.json: text_file, forward_seconds), não de
decode. Num modelo causal, o roteamento por posição dessa passada é o mesmo que o prefill do mesmo texto daria. Logo o censo vale para posições
consecutivas de um único texto como estes. Em lote de sequências DIFERENTES a união é outra; na decodificação especulativa só os tokens aceitos seguem o
texto. O resultado não prevê velocidade.
Uso: taskset -c 16-27 tools/.venv-ref/bin/python -I -B qual-runs/A1-DIAG10-20261103/censo_m.py > qual-runs/A1-DIAG10-20261103/censo_m.txt"""
import os, sys
R = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(R, "..", "..", "tools"))
import numpy as np  # noqa: E402
import moe_sim  # noqa: E402

TR = {"1b": "granite-3.1-1b-a400m-instruct", "3b": "granite-3.1-3b-a800m-instruct"}
MS = (1, 2, 4, 8, 16, 32)


def censo(convs, E, k, M):
    """Por camada e bloco: |U|. Devolve array (n_blocos_total*L,) de tamanhos, e número de tokens descartados."""
    tam, desc = [], 0
    for a in convs:
        L, T, kk = a.shape
        assert kk == k
        n = T // M
        desc += T - n * M
        for b in range(n):
            blk = a[:, b * M:(b + 1) * M, :]
            for l in range(L):
                u = np.unique(blk[l])
                assert k <= len(u) <= min(E, M * k)
                assert M > 1 or len(u) == k
                tam.append(len(u))
    return np.array(tam), desc


print("=== Censo da união de experts por camada em M tokens consecutivos (rodada 03/11; só contagem nos traços) ===")
for mod in ("1b", "3b"):
    d = os.path.join(R, "..", "TRACOS-20261029", TR[mod])
    tr, te = moe_sim.load(d, "treino"), moe_sim.load(d, "teste")
    L, _, k = tr[0].shape
    E = int(max(a.max() for a in tr + te)) + 1
    print(f"\n{mod}: {L} camadas, {E} experts por camada, k={k}; treino {len(tr)} conversas ({sum(a.shape[1] for a in tr)} tokens), teste {len(te)} conversas ({sum(a.shape[1] for a in te)} tokens)")
    print("  M   conj   blocos(camada x bloco)  descartados  |U| média  mediana  p95  máx   |U|/E média  p95   reuso M*k/|U| média  min(E,M*k)")
    for M in MS:
        for nome, cs in (("treino", tr), ("teste", te)):
            tam, desc = censo(cs, E, k, M)
            u = tam / E
            reuso = M * k / tam
            teto = min(E, M * k)
            assert tam.max() <= teto
            print(f"  {M:>2}  {nome:<6} {len(tam):>9}  {desc:>11}  {tam.mean():8.2f} {np.median(tam):7.1f} {np.percentile(tam,95):5.1f} {tam.max():4d}   {u.mean():8.3f} {np.percentile(u,95):6.3f}   {reuso.mean():12.2f}   {teto}")
    # M=1 reproduz o top-k
    tam1, _ = censo(te, E, k, 1)
    assert (tam1 == k).all(), "M=1 tem de reproduzir o top-k exato"
print("\nChecagens: M=1 reproduz k experts distintos por camada; |U| <= min(E, M*k) em todos os blocos; blocos não atravessam o fim da conversa.")
