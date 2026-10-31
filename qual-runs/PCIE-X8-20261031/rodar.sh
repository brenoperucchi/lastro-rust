#!/bin/bash
# roda da raiz do repo: bash qual-runs/PCIE-X8-20261031/rodar.sh
cd /home/brenoperucchi/Devs/lastro-rust || exit 1
R=qual-runs/PCIE-X8-20261031
B=qual-runs/LLAMACPP-COMPARA-20261030/bin
T=qual-runs/TRACOS-20261029
M=/home/brenoperucchi/models
export CUDA_VISIBLE_DEVICES=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd
(cd $B && sha256sum -c SHA256SUMS) > $R/sha-binarios.txt 2>&1 || { echo "sha dos binários não confere"; exit 1; }
dir() { case $1 in 1b) echo granite-3.1-1b-a400m-instruct;; 3b) echo granite-3.1-3b-a800m-instruct;; esac; }
{ date; head -1 /proc/driver/nvidia/version; uptime
  nvidia-smi --query-gpu=index,name,pci.bus_id,pcie.link.gen.max,pcie.link.width.max,memory.used --format=csv
  ps -eo pid,pcpu,comm --sort=-pcpu | awk 'NR == 1 || $2 > 5'; } > $R/ambiente.txt
for rep in 1 2; do for mod in 1b 3b; do
  out=$R/x8-$mod-nosso-$rep.txt
  ( while sleep 20; do nvidia-smi --id=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd --query-gpu=pcie.link.gen.current,pcie.link.width.current,utilization.gpu --format=csv,noheader; done ) > $R/link-$mod-$rep.log 2>&1 &
  amostra=$!
  taskset -c 0-15 $B/m2a-cache --model $M/$(dir $mod) --traces $T/$(dir $mod) --ref qual-runs/REF-granite-$mod-texto-a \
    --budgets 0.1,0.25,0.5 --policies resident,lru_cpu,lfu_cpu --warmup 3 > $out
  rc=$?
  kill $amostra
  echo "$(date +%T) $out $rc $(cut -d' ' -f1-3 /proc/loadavg)"
done; done
