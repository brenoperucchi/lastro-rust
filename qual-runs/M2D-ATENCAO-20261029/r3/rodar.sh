#!/bin/bash
# M2e (gemv_down): antes contra depois na mesma sessão, nos P-cores (0-15) do i7-14700KF.
#
# Binários (sha256 em bin/SHA256SUMS), da árvore de 29/10 sobre o HEAD a67d25f, em rust/moe-rt:
#   depois: CUDA_TOOLKIT_PATH=/opt/cuda PATH=/opt/cuda/bin:$PATH cargo build --release, e cópia de target/release/{m1b-modelo,
#           m1b-gulosa,m2a-cache,m2b-custo-fixo,m2c-kernels} para bin/depois;
#   antes:  a mesma árvore com o down em kernels::gemv (em src/gpu.rs, `kernels::gemv_down(sc.o, &self.down[e], &aa)` →
#           `kernels::gemv(...)`; em src/cache.rs, `kernels::gemv_down(sc.o, &slot.down, &aa)` → `kernels::gemv(...)`), o mesmo
#           build e cópia para bin/antes (sem o m2c-kernels); depois o down voltou a gemv_down e o build bate com bin/depois.
# O m2a de antes rodou depois deste script, à parte (comando no fim, comentado).
cd /home/brenoperucchi/Devs/strata-opencode
R=qual-runs/M2D-ATENCAO-20261029/r3
B=$R/bin
T=qual-runs/TRACOS-20261029
M=/home/brenoperucchi/models
P="taskset -c 0-15"
for m in 1b 3b; do
  case $m in 1b) d=granite-3.1-1b-a400m-instruct;; 3b) d=granite-3.1-3b-a800m-instruct;; esac
  taskset -c 4 $B/depois/m2c-kernels --model $M/$d > $R/m2c-$m.txt 2>&1; echo "m2c $m $?"
  for v in antes depois antes depois; do
    n=1; [ -e $R/m2b-$m-longo-$v-1.txt ] && n=2
    $P $B/$v/m2b-custo-fixo --model $M/$d --ref qual-runs/REF-granite-$m-texto-a --repeat 4 > $R/m2b-$m-longo-$v-$n.txt 2>&1; echo "m2b $m $v $n $?"
    $P $B/$v/m1b-gulosa --model $M/$d --ref qual-runs/REF-greedy-granite-$m/greedy.json --paths gpu > $R/gulosa-$m-$v-$n.txt 2>&1; echo "gulosa $m $v $n $?"
  done
  $P $B/depois/m1b-modelo --model $M/$d --ref qual-runs/REF-granite-$m-texto-a --paths gpu > $R/m1b-$m.txt 2>&1; echo "m1b $m $?"
  $P $B/depois/m2a-cache --model $M/$d --traces $T/$d --ref qual-runs/REF-granite-$m-texto-a --sim $T/sim-$m.json > $R/m2a-$m.txt 2>&1; echo "m2a $m $?"
done
# m2a de antes (rodado depois do script, mesmos P-cores):
# for m in 1b 3b; do ...; taskset -c 0-15 $B/antes/m2a-cache --model $M/$d --traces $T/$d --ref qual-runs/REF-granite-$m-texto-a \
#   --sim $T/sim-$m.json > $R/m2a-$m-antes.txt; done
