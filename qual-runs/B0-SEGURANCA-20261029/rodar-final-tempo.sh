#!/bin/bash
# B0, rodada 34: os critérios de tempo da medida 2 (residente e lru_sync ≤ 1%) com os binários bin/final, que acrescentaram
# as conferências de instância e da vítima. 5 pares antes/final intercalados por modelo, admissão por evento (o padrão).
cd /home/brenoperucchi/Devs/strata-opencode
R=qual-runs/B0-SEGURANCA-20261029
B=$R/bin
T=qual-runs/TRACOS-20261029
M=/home/brenoperucchi/models
P="taskset -c 0-15"
dir() { case $1 in 1b) echo granite-3.1-1b-a400m-instruct;; 3b) echo granite-3.1-3b-a800m-instruct;; esac; }
for n in 1 2 3 4 5; do
  for v in antes final; do
    for m in 1b 3b; do
      d=$(dir $m)
      $P $B/$v/m2a-cache --model $M/$d --traces $T/$d --ref qual-runs/REF-granite-$m-texto-a --budgets 0.25,0.5 --policies resident,lru_sync --admit evento > $R/m2a-$m-$v-tempo-$n.txt 2>&1
      echo "m2a $v $m tempo $n $?"
    done
  done
done
