#!/bin/bash
# B0 depois da rodada 33 (binários bin/final): instância única Slots+Epoch, vítima com geração, testes novos.
# 1. Barreira, todas as políticas, uma vez por modelo: contagens exatas e hash iguais aos do antes.
# 2. lru_cpu em barreira (mesmo trabalho nos dois binários), 3 pares antes/final intercalados por modelo: mede o viés de
#    tempo do lru_cpu apontado na rodada 33 sem a variação da admissão por evento.
cd /home/brenoperucchi/Devs/strata-opencode
R=qual-runs/B0-SEGURANCA-20261029
B=$R/bin
T=qual-runs/TRACOS-20261029
M=/home/brenoperucchi/models
P="taskset -c 0-15"
dir() { case $1 in 1b) echo granite-3.1-1b-a400m-instruct;; 3b) echo granite-3.1-3b-a800m-instruct;; esac; }
run() { # versão modelo políticas saída
  local d; d=$(dir $2)
  $P $B/$1/m2a-cache --model $M/$d --traces $T/$d --ref qual-runs/REF-granite-$2-texto-a --budgets 0.25,0.5 --policies $3 --admit barreira > $4 2>&1
  echo "m2a $1 $2 $(basename $4) $?"
}
for m in 1b 3b; do
  run final $m resident,cache_full,lru_sync,lru_cpu $R/m2a-$m-final-barreira-1.txt
done
for n in 1 2 3; do
  for v in antes final; do
    for m in 1b 3b; do
      run $v $m lru_cpu $R/m2a-$m-$v-barreira-lrucpu-$n.txt
    done
  done
done
