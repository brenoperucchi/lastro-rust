#!/bin/bash
# M2f, segunda medida: gemv_swiglu com BM = 1 (I blocos, `SWIGLU_BM`) contra o BM = 2 da primeira versão.
#   depois: o mesmo bin/depois de rodar.sh (BM = 2);
#   final:  a árvore de 29/10 ~21:40 com SWIGLU_BM = 1 e as correções da rodada 31 (fontes em bin/FONTES-final.sha256),
#           cargo build --release e cópia de target/release/{m1b-modelo,m1b-gulosa,m2a-cache,m2b-custo-fixo,m2c-kernels}.
# cargo test (37 testes) terminou antes desta medida; nada compilou durante ela.
# O m2a roda só no final (1 vez por modelo): compara com o m2a-*-depois de rodar.sh, que é de uma hora antes.
cd /home/brenoperucchi/Devs/strata-opencode
R=qual-runs/M2F-FUSOES-20261029
B=$R/bin
T=qual-runs/TRACOS-20261029
M=/home/brenoperucchi/models
P="taskset -c 0-15"
for m in 1b 3b; do
  case $m in 1b) d=granite-3.1-1b-a400m-instruct;; 3b) d=granite-3.1-3b-a800m-instruct;; esac
  taskset -c 4 $B/final/m2c-kernels --model $M/$d > $R/m2c-$m-final.txt 2>&1; echo "m2c $m final $?"
  for n in 1 2; do
    for v in depois final; do
      $P $B/$v/m2b-custo-fixo --model $M/$d --ref qual-runs/REF-granite-$m-texto-a --repeat 4 > $R/m2b-$m-longo-$v-r2-$n.txt 2>&1; echo "m2b $m $v $n $?"
      $P $B/$v/m1b-gulosa --model $M/$d --ref qual-runs/REF-greedy-granite-$m/greedy.json --paths gpu > $R/gulosa-$m-$v-r2-$n.txt 2>&1; echo "gulosa $m $v $n $?"
    done
  done
  $P $B/final/m1b-modelo --model $M/$d --ref qual-runs/REF-granite-$m-texto-a --paths gpu > $R/m1b-$m-final.txt 2>&1; echo "m1b $m final $?"
  $P $B/final/m2a-cache --model $M/$d --traces $T/$d --ref qual-runs/REF-granite-$m-texto-a --sim $T/sim-$m.json > $R/m2a-$m-final-1.txt 2>&1; echo "m2a $m final $?"
done
