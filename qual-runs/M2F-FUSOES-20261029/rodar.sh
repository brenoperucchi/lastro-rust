#!/bin/bash
# M2f (fusões locais do MoE: gemv_swiglu e gemv_axpby): antes contra depois na mesma sessão, nos P-cores (0-15).
#
# Binários (sha256 em bin/SHA256SUMS), da árvore de 29/10 sobre o HEAD a67d25f, em rust/moe-rt, com
# CUDA_TOOLKIT_PATH=/opt/cuda PATH=/opt/cuda/bin:$PATH cargo build --release e cópia de target/release/{m1b-modelo,
# m1b-gulosa,m2a-cache,m2b-custo-fixo,m2c-kernels}:
#   antes:  a árvore do fim da rodada 30 (M2e: 4 lançamentos por expert, down por gemv_down);
#   depois: a árvore com gemv_swiglu e gemv_axpby (2 lançamentos por expert).
cd /home/brenoperucchi/Devs/strata-opencode
R=qual-runs/M2F-FUSOES-20261029
B=$R/bin
T=qual-runs/TRACOS-20261029
M=/home/brenoperucchi/models
P="taskset -c 0-15"
for m in 1b 3b; do
  case $m in 1b) d=granite-3.1-1b-a400m-instruct;; 3b) d=granite-3.1-3b-a800m-instruct;; esac
  for v in antes depois; do
    taskset -c 4 $B/$v/m2c-kernels --model $M/$d > $R/m2c-$m-$v.txt 2>&1; echo "m2c $m $v $?"
  done
  for n in 1 2; do
    for v in antes depois; do
      $P $B/$v/m2b-custo-fixo --model $M/$d --ref qual-runs/REF-granite-$m-texto-a --repeat 4 > $R/m2b-$m-longo-$v-$n.txt 2>&1; echo "m2b $m $v $n $?"
      $P $B/$v/m1b-gulosa --model $M/$d --ref qual-runs/REF-greedy-granite-$m/greedy.json --paths gpu > $R/gulosa-$m-$v-$n.txt 2>&1; echo "gulosa $m $v $n $?"
    done
  done
  $P $B/depois/m1b-modelo --model $M/$d --ref qual-runs/REF-granite-$m-texto-a --paths gpu > $R/m1b-$m.txt 2>&1; echo "m1b $m $?"
  runs="1"; [ $m = 1b ] && runs="1 2"
  for n in $runs; do
    for v in antes depois; do
      $P $B/$v/m2a-cache --model $M/$d --traces $T/$d --ref qual-runs/REF-granite-$m-texto-a --sim $T/sim-$m.json > $R/m2a-$m-$v-$n.txt 2>&1; echo "m2a $m $v $n $?"
    done
  done
done
