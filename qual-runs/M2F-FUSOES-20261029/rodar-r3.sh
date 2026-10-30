#!/bin/bash
# M2f, terceira medida (rodada 32): variância do lru_cpu entre execuções do mesmo binário e efeito das cópias em segundo plano.
#   - P2-1 do rev-1: o bin/depois do 3B (que deu +19% e +43% a 25% e 50% em rodar.sh) rodado de novo, intercalado com o
#     bin/final, 3 vezes cada, só lru_cpu 25% e 50%;
#   - P3-2 do rev-2: lru_cpu com --backlog 0 (nenhuma admissão: nenhuma cópia em segundo plano, todos os experts frios na
#     CPU), para ver o CPU por expert sem cópias, contra o backlog 4 de sempre.
# Binários os mesmos de bin/SHA256SUMS; nada compilou durante a medida.
cd /home/brenoperucchi/Devs/strata-opencode
R=qual-runs/M2F-FUSOES-20261029
B=$R/bin
T=qual-runs/TRACOS-20261029
M=/home/brenoperucchi/models
P="taskset -c 0-15"
for m in 3b 1b; do
  case $m in 1b) d=granite-3.1-1b-a400m-instruct;; 3b) d=granite-3.1-3b-a800m-instruct;; esac
  for n in 1 2 3; do
    for v in depois final; do
      $P $B/$v/m2a-cache --model $M/$d --traces $T/$d --ref qual-runs/REF-granite-$m-texto-a --policies lru_cpu --budgets 0.25,0.5 > $R/m2a-$m-$v-r3-$n.txt 2>&1; echo "m2a $m $v $n $?"
    done
  done
  for n in 1 2; do
    $P $B/final/m2a-cache --model $M/$d --traces $T/$d --ref qual-runs/REF-granite-$m-texto-a --policies lru_cpu --budgets 0.25,0.5 --backlog 0 > $R/m2a-$m-final-backlog0-$n.txt 2>&1; echo "m2a $m backlog0 $n $?"
  done
done
