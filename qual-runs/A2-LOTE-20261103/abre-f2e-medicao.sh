#!/usr/bin/env bash
# Janelas de medida da F2E (plano da O3 na RTX 5090), NA maquina 5090: 1B e depois 3B, dentro de UMA chamada do ~/janela-gpu-strata.sh.
# Uso (externo, é o que se roda):  abre-f2e-medicao.sh <pasta-da-coleta-1b> <pasta-da-coleta-3b>
# Recusa (código 2, sem tocar na GPU nem no Strata) se a coleta de algum modelo não estiver aprovada: HC1b dentro e teste de GPU passou com procedência completa.
set -u
. "$(dirname "$0")/abre-f2e-lib.sh"
if [ "${1:-}" = --interno ]; then
  C1B=$2; C3B=$3; T=$4
  for M in 1b 3b; do
    S=$AQ/s5090-f2e-$M-real-$T
    if [ $M = 1b ]; then COLETA=$C1B; else COLETA=$C3B; fi
    echo "inicio f2e $M $(date +%T)"
    PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID bash "$AQ/roda-f2e-janela.sh" --modelos $M --saida "$S" --coleta-dir "$COLETA" > "$S.log" 2>&1
    rc=$?; echo "fim f2e $M rc=$rc $(date +%T) $S"
    [ $rc -eq 0 ] || exit $rc
  done
  exit 0
fi
C1B=${1:?uso: abre-f2e-medicao.sh <pasta-da-coleta-1b> <pasta-da-coleta-3b>}; C3B=${2:?uso: abre-f2e-medicao.sh <pasta-da-coleta-1b> <pasta-da-coleta-3b>}
T=$(date +%Y%m%d-%H%M%S)
verifica_fixos
logs_livres "$AQ/s5090-f2e-1b-real-$T" "$AQ/s5090-f2e-3b-real-$T"
C1B=$(realpath -e -- "$C1B" 2> /dev/null) || { echo "pasta da coleta 1B não existe: recusado (nada foi tocado)"; exit 2; }
C3B=$(realpath -e -- "$C3B" 2> /dev/null) || { echo "pasta da coleta 3B não existe: recusado (nada foi tocado)"; exit 2; }
confere_coleta 1b "$C1B" || { echo "coleta do 1B não aprovada: recusado (nada foi tocado)"; exit 2; }
confere_coleta 3b "$C3B" || { echo "coleta do 3B não aprovada: recusado (nada foi tocado)"; exit 2; }
echo "medição f2e: carimbo $T; Strata e GPU entregues ao janela-gpu-strata.sh (teto 2700 s)"
exec "$JANELA_GPU" --teto 2700 -- bash "$AQ/abre-f2e-medicao.sh" --interno "$C1B" "$C3B" "$T"
