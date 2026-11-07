#!/usr/bin/env bash
# A2, etapa 4(c): as 12 repetições FRIAS da leitura em blocos (1b depois 3b; cenários C e A; 3 repetições cada), uma por processo NOVO do
# binário leitura-velocidade, mesmo ambiente, e no fim a consolidação (sem GPU). Pensado para ser o COMANDO de UMA janela sob o roteiro de limite:
#
#   roda-com-limite.sh --limite 1800 -- env CUDA_VISIBLE_DEVICES=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd LEITURA_VELOCIDADE_AUTORIZADA=1 \
#       /home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/roda-frio.sh --saida <pasta nova>
#
# O roteiro do limite não repassa o ambiente por conta própria: o `env` acima leva o portão e o UUID até aqui, e daqui para cada processo.
# Este script NÃO tira o portão nem o UUID do lugar, NÃO chama nvidia-smi, NÃO mexe em timers e NÃO repete nada: cada repetição roda uma vez, na
# ordem fixa. Um processo que grava falha (código 1) não interrompe a sequência (a repetição fica SEM PROVA na consolidação); recusa (código 2:
# portão, UUID, hash, argumentos) interrompe tudo na hora e sai 2, sem consolidar. A pasta de saída tem de ser nova ou vazia.
#
#   roda-frio.sh --saida DIR [--bin BINARIO]
# Variáveis opcionais (só para teste com binário de mentira): MODELO_1B, MODELO_3B, TRACOS_1B, TRACOS_3B.
# Código de saída: o da consolidação (0), ou 2 em recusa/uso, ou 3 se alguma repetição gravou falha (a consolidação ainda roda e decide).
set -u
AQUI=$(cd "$(dirname "$0")" && pwd)
BIN=$AQUI/../../rust/moe-rt/target/release/leitura-velocidade
SAIDA=
usa() { echo "uso: roda-frio.sh --saida DIR [--bin BINARIO]" >&2; }
while [ $# -gt 0 ]; do
  case $1 in
    --saida) [ $# -ge 2 ] || { usa; exit 2; }; SAIDA=$2; shift 2;;
    --bin) [ $# -ge 2 ] || { usa; exit 2; }; BIN=$2; shift 2;;
    *) usa; exit 2;;
  esac
done
[ -n "$SAIDA" ] || { usa; exit 2; }
[ -x "$BIN" ] || { echo "binário não executável: $BIN" >&2; exit 2; }
if [ -e "$SAIDA" ]; then
  [ -d "$SAIDA" ] && [ -z "$(ls -A "$SAIDA")" ] || { echo "--saida $SAIDA já existe e não está vazia: uma janela nova pede pasta nova" >&2; exit 2; }
fi
mkdir -p "$SAIDA" || exit 2
M1=${MODELO_1B:-/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct}
M3=${MODELO_3B:-/home/brenoperucchi/models/granite-3.1-3b-a800m-instruct}
T1=${TRACOS_1B:-/home/brenoperucchi/Devs/lastro-rust/qual-runs/TRACOS-20261029/granite-3.1-1b-a400m-instruct}
T3=${TRACOS_3B:-/home/brenoperucchi/Devs/lastro-rust/qual-runs/TRACOS-20261029/granite-3.1-3b-a800m-instruct}
LOG=$SAIDA/roda-frio.log
log() { echo "$(date '+%d/%m %H:%M:%S') $*" | tee -a "$LOG"; }

FALHOU=0
for modelo in 1b 3b; do
  if [ "$modelo" = 1b ]; then MD=$M1; TR=$T1; else MD=$M3; TR=$T3; fi
  for cen in C A; do
    for rep in 1 2 3; do
      log "início: $modelo cenário $cen repetição $rep"
      "$BIN" --estado frio --celula "$modelo:$MD:$TR" --cenario "$cen" --rep "$rep" --saida "$SAIDA" 2>&1 | tee -a "$LOG"
      rc=${PIPESTATUS[0]}
      log "fim: $modelo cenário $cen repetição $rep, código $rc"
      if [ "$rc" = 2 ]; then
        log "RECUSA (código 2) antes de qualquer trabalho: sequência interrompida, sem consolidar"
        exit 2
      fi
      [ "$rc" = 0 ] || FALHOU=1
    done
  done
done
log "consolidação (sem GPU)"
"$BIN" --consolida "$SAIDA" --estado frio 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || exit "$rc"
[ "$FALHOU" = 0 ] || exit 3
exit 0
