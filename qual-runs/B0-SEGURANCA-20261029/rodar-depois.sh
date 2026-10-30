#!/bin/bash
# B0, passo 7: o "depois" (núcleo tipado em cache/core.rs).
# 1. Admissão em barreira, uma execução por modelo: contagens exatas e hash dos logits têm de ser iguais aos do antes.
# 2. Tempo com a admissão por evento (o padrão), 5 pares antes/depois intercalados por modelo: residente e lru_sync ≤ 1%
#    nos dois modelos; lru_cpu só no 3B, mediana das 5 dentro da faixa do antes.
# Binários em bin/{antes,depois} (sha256 em bin/SHA256SUMS; fontes em bin/FONTES-{antes,depois}.sha256). cargo test antes.
cd /home/brenoperucchi/Devs/strata-opencode
R=qual-runs/B0-SEGURANCA-20261029
B=$R/bin
T=qual-runs/TRACOS-20261029
M=/home/brenoperucchi/models
P="taskset -c 0-15"
dir() { case $1 in 1b) echo granite-3.1-1b-a400m-instruct;; 3b) echo granite-3.1-3b-a800m-instruct;; esac; }
run() { # versão modelo políticas admissão saída
  local d; d=$(dir $2)
  $P $B/$1/m2a-cache --model $M/$d --traces $T/$d --ref qual-runs/REF-granite-$2-texto-a --budgets 0.25,0.5 --policies $3 --admit $4 > $5 2>&1
  echo "m2a $1 $2 $4 $(basename $5) $?"
}
for m in 1b 3b; do
  run depois $m resident,cache_full,lru_sync,lru_cpu barreira $R/m2a-$m-depois-barreira-1.txt
done
for n in 1 2 3 4 5; do
  for v in antes depois; do
    run $v 1b resident,lru_sync evento $R/m2a-1b-$v-evento-$n.txt
    run $v 3b resident,lru_sync,lru_cpu evento $R/m2a-3b-$v-evento-$n.txt
  done
done
