#!/bin/bash
# Passo 3, medida pedida pelo limite da recalibração (30/10, aprovada pelo Breno): o lfu_cpu do simulador vive com 12–19% das
# camadas com tráfego de cópia, entre o backlog 0 (nenhuma) e o lru_cpu de sempre (63–100%). Aqui o lru_cpu com backlog 1 e 2
# cai no meio; o backlog 0 e o 4 entram intercalados como controle da mesma sessão (o driver foi atualizado às 03:22 de 30/10
# sem recarregar o módulo: módulo 610.57, libcuda 615.78; o cuInit funciona, mas o ambiente não é mais o das referências). Binário: o mesmo m2a-cache das medidas de
# referência (qual-runs/M2F-FUSOES-20261029/bin/final, SHA256SUMS conferido antes); nada compila durante a medida.
cd /home/brenoperucchi/Devs/strata-opencode
R=qual-runs/BACKLOG-INTERMEDIARIO-20261030
B=qual-runs/M2F-FUSOES-20261029/bin/final
T=qual-runs/TRACOS-20261029
M=/home/brenoperucchi/models
P="taskset -c 0-15"
(cd qual-runs/M2F-FUSOES-20261029/bin && sha256sum -c SHA256SUMS 2>&1 | grep '^final/m2a-cache') > $R/sha-binario.txt
{ cat /proc/driver/nvidia/version | head -1; ls -l /usr/lib/libcuda.so.1; uptime; } > $R/ambiente.txt
for m in 1b 3b; do
  case $m in 1b) d=granite-3.1-1b-a400m-instruct;; 3b) d=granite-3.1-3b-a800m-instruct;; esac
  for n in 1 2; do
    for b in 0 1 2 4; do
      $P $B/m2a-cache --model $M/$d --traces $T/$d --ref qual-runs/REF-granite-$m-texto-a --policies lru_cpu --budgets 0.1,0.25,0.5 --backlog $b > $R/m2a-$m-backlog$b-$n.txt 2>&1; echo "m2a $m backlog $b $n $?"
    done
  done
done
