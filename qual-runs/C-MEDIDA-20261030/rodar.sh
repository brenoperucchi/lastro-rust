#!/bin/bash
# Medida do C: comparação aquecida (5 repetições) e validação fria em duas fases (3 repetições). Critério em criterio.txt,
# gravado com as previsões em previsao.txt antes da primeira execução; previsao-fase-b.txt é gravado pelo ajuste_fase_a.py
# antes da fase B. Binário em bin/ (sha conferido antes); nada compila durante a medida; Python só nos E-cores.
# Qualquer falha (execução do m2a-cache, previsão, gravação) interrompe o roteiro; ele recusa rodar com outra instância
# segurando a trava ou com resultados e previsões de uma sessão anterior na pasta (rodada 40), sem apagar nada.
# MEDIDA_DIR, MEDIDA_BIN e MEDIDA_PY só servem ao teste do próprio roteiro (cópia da pasta, binário falso, Python falso);
# a medida roda sem elas.
set -euo pipefail
cd /home/brenoperucchi/Devs/strata-opencode
R=${MEDIDA_DIR:-qual-runs/C-MEDIDA-20261030}
exec 9> "$R/.trava"
flock -n 9 || { echo "outra instância do rodar.sh está com a trava"; exit 1; }
for f in "$R"/previsao.txt "$R"/previsao-fase-b.txt "$R"/ambiente.txt "$R"/carga.log "$R"/sha-binario.txt "$R"/quente-* "$R"/faseA-* "$R"/faseB-*; do
  [ -e "$f" ] && { echo "já existe $f: sessão anterior na pasta; nada foi alterado"; exit 1; }
done
B=${MEDIDA_BIN:-$R/bin/m2a-cache}
T=qual-runs/TRACOS-20261029
M=/home/brenoperucchi/models
P="taskset -c 0-15"
PY="taskset -c 16-27 tools/.venv-ref/bin/python -I -B"
(cd $R/bin && sha256sum -c SHA256SUMS) > $R/sha-binario.txt 2>&1 || { echo "sha do binário não confere"; exit 1; }
PY=${MEDIDA_PY:-$PY}
{ date; head -1 /proc/driver/nvidia/version; ls -l /usr/lib/libcuda.so.1; uptime; sha256sum tools/moe_sim.py $R/moe_sim_s.py; } > $R/ambiente.txt
$PY $R/previsao.py > $R/previsao.tmp || exit 1
{ cat $R/criterio.txt; echo; echo "Previsões gravadas em $(date '+%d/%m %H:%M:%S'), antes da primeira execução."; cat $R/previsao.tmp; } > $R/previsao.txt
rm $R/previsao.tmp
dir() { case $1 in 1b) echo granite-3.1-1b-a400m-instruct;; 3b) echo granite-3.1-3b-a800m-instruct;; esac; }
run() { # saída modelo argumentos...; devolve o código do m2a-cache, e o set -e para o roteiro se não for zero
  local out=$1 mod=$2 rc=0; shift 2
  $P $B --model $M/$(dir $mod) --traces $T/$(dir $mod) --ref qual-runs/REF-granite-$mod-texto-a --budgets 0.1,0.25,0.5 "$@" > $R/$out 2>&1 || rc=$?
  echo "$(date +%T) $out $rc"
  echo "$(date +%T) $out $(cut -d' ' -f1-3 /proc/loadavg)" >> $R/carga.log
  return $rc
}
for n in 1 2 3 4 5; do
  if [ $((n % 2)) = 1 ]; then pols="lru_cpu lfu_cpu"; else pols="lfu_cpu lru_cpu"; fi
  for mod in 1b 3b; do
    for pol in $pols; do
      run quente-$mod-$pol-$n.txt $mod --policies resident,$pol --warmup 0,1,2,3
    done
  done
done
for n in 1 2 3; do
  for mod in 1b 3b; do
    for b in 0 4; do
      run faseA-$mod-b$b-$n.txt $mod --policies resident,lru_cpu --backlog $b
    done
  done
done
$PY $R/ajuste_fase_a.py > $R/previsao-fase-b.tmp || exit 1
{ echo "Parâmetros reajustados e previsões do lfu_cpu frio, gravados em $(date '+%d/%m %H:%M:%S'), antes da fase B."; cat $R/previsao-fase-b.tmp; } > $R/previsao-fase-b.txt
rm $R/previsao-fase-b.tmp
for n in 1 2 3; do
  for mod in 1b 3b; do
    run faseB-$mod-lfu-$n.txt $mod --policies resident,lfu_cpu
    run faseB-$mod-ctl-$n.txt $mod --policies resident,lru_cpu --backlog 4
  done
done
echo "$(date +%T) fim"
