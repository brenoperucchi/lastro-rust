#!/bin/bash
# C: o binário com o lfu_cpu, admissão em barreira, políticas antigas, uma vez por modelo. As contagens exatas e o hash dos
# logits têm de ser iguais aos de qual-runs/B0-SEGURANCA-20261029/m2a-*-final-barreira-1.txt (binário final do B0).
# Depois, lfu_cpu em barreira com aquecimento 0-3, duas vezes por modelo: contagens exatas e hash iguais entre as duas.
# Versão 3 (rodada 39): binário refeito depois da trava dos testes de unidade (só cfg(test), mas o sha muda pelos hashes de
# símbolo do LLVM); a v2 está em v2/ (binário em bin/v2/), a v1 em rodar-barreira-v1.log.
cd /home/brenoperucchi/Devs/strata-opencode
R=qual-runs/C-SEGURANCA-20261030
T=qual-runs/TRACOS-20261029
M=/home/brenoperucchi/models
for m in 1b 3b; do
  case $m in 1b) d=granite-3.1-1b-a400m-instruct;; 3b) d=granite-3.1-3b-a800m-instruct;; esac
  taskset -c 0-15 $R/bin/m2a-cache --model $M/$d --traces $T/$d --ref qual-runs/REF-granite-$m-texto-a --budgets 0.25,0.5 --policies resident,cache_full,lru_sync,lru_cpu --admit barreira > $R/m2a-$m-barreira-1.txt 2>&1
  echo "m2a $m $?"
done
for m in 1b 3b; do
  case $m in 1b) d=granite-3.1-1b-a400m-instruct;; 3b) d=granite-3.1-3b-a800m-instruct;; esac
  for n in 1 2; do
    taskset -c 0-15 $R/bin/m2a-cache --model $M/$d --traces $T/$d --ref qual-runs/REF-granite-$m-texto-a --budgets 0.25,0.5 --policies resident,lfu_cpu --admit barreira --warmup 0,1,2,3 > $R/m2a-$m-lfu-barreira-$n.txt 2>&1
    echo "m2a $m lfu_cpu $n $?"
  done
done
for m in 1b 3b; do
  a=$(sed -n '/^contagens exatas/,$p' qual-runs/B0-SEGURANCA-20261029/m2a-$m-final-barreira-1.txt | sha256sum)
  b=$(sed -n '/^contagens exatas/,$p' $R/m2a-$m-barreira-1.txt | sha256sum)
  [ "$a" = "$b" ] && echo "$m: contagens e hash iguais ao B0 final" || echo "$m: DIFERENTE do B0 final"
done
for m in 1b 3b; do
  a=$(sed -n '/^contagens exatas/,$p' $R/m2a-$m-lfu-barreira-1.txt | sha256sum)
  b=$(sed -n '/^contagens exatas/,$p' $R/m2a-$m-lfu-barreira-2.txt | sha256sum)
  [ "$a" = "$b" ] && echo "$m lfu_cpu: contagens e hash iguais nas duas execuções" || echo "$m lfu_cpu: DIFERENTES entre as duas execuções"
done
