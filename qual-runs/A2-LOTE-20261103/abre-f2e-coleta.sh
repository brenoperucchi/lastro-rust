#!/usr/bin/env bash
# Coleta da F2E (HC1b) e teste de erro em GPU (motor_reforma_gpu, motor_passo_gpu, o3_kernels_gpu) NA maquina 5090, nos dois modelos, ANTES da medição.
# Uso (externo, é o que se roda):  abre-f2e-coleta.sh <worktree-cand>     [worktree do commit candidato, limpo, com os 3 testes já compilados em <wt>/target-f2e]
# Faz os portões sem tocar na GPU e entrega a execução ao ~/janela-gpu-strata.sh (para o Strata, confirma a GPU livre, roda o modo --interno, religa no EXIT).
set -u
. "$(dirname "$0")/abre-f2e-lib.sh"
if [ "${1:-}" = --interno ]; then
  WT=$2; T=$3
  for M in 1b 3b; do
    S=$AQ/s5090-f2e-coleta-$M-real-$T
    echo "inicio coleta f2e $M $(date +%T)"
    PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID bash "$AQ/roda-f2e-janela.sh" --modelos $M --saida "$S" --coleta --teste-gpu --worktree-cand "$WT" > "$S.log" 2>&1
    rc=$?; echo "fim coleta f2e $M rc=$rc $(date +%T) $S"
    [ $rc -eq 0 ] || exit $rc
  done
  exit 0
fi
WT=${1:?uso: abre-f2e-coleta.sh <worktree-cand>}
WT=$(realpath -e -- "$WT" 2> /dev/null) || { echo "worktree do candidato não existe: recusado (nada foi tocado)"; exit 2; }
T=$(date +%Y%m%d-%H%M%S)
verifica_fixos
logs_livres "$AQ/s5090-f2e-coleta-1b-real-$T" "$AQ/s5090-f2e-coleta-3b-real-$T"
for T_GPU in motor_reforma_gpu motor_passo_gpu o3_kernels_gpu; do
  EXE=$(ls -t "$WT"/target-f2e/release/deps/$T_GPU-* 2> /dev/null | grep -v '\.d$' | head -n 1)
  { [ -n "$EXE" ] && [ -x "$EXE" ]; } || { echo "executável do teste de GPU $T_GPU ausente em $WT/target-f2e/release/deps: compile antes (--no-run); nada foi tocado"; exit 2; }
done
echo "coleta f2e: carimbo $T; Strata e GPU entregues ao janela-gpu-strata.sh (teto 2700 s)"
exec "$JANELA_GPU" --teto 2700 -- bash "$AQ/abre-f2e-coleta.sh" --interno "$WT" "$T"
