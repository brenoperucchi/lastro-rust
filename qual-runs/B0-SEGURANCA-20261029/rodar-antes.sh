#!/bin/bash
# B0, passo 1: o "antes" do B0 com a admissão em barreira (AdmitMode::Barrier), duas execuções por modelo, para conferir que
# contagens exatas e hash dos logits se repetem (o critério bit a bit do B0) e guardar a referência de tempo.
# Binários em bin/antes (sha256 em bin/SHA256SUMS; fontes em bin/FONTES-antes.sha256). cargo test antes da medida.
cd /home/brenoperucchi/Devs/strata-opencode
R=qual-runs/B0-SEGURANCA-20261029
B=$R/bin
T=qual-runs/TRACOS-20261029
M=/home/brenoperucchi/models
P="taskset -c 0-15"
for m in 1b 3b; do
  case $m in 1b) d=granite-3.1-1b-a400m-instruct;; 3b) d=granite-3.1-3b-a800m-instruct;; esac
  for n in 1 2; do
    $P $B/antes/m2a-cache --model $M/$d --traces $T/$d --ref qual-runs/REF-granite-$m-texto-a --budgets 0.25,0.5 --admit barreira > $R/m2a-$m-antes-barreira-$n.txt 2>&1; echo "m2a $m barreira $n $?"
  done
done
