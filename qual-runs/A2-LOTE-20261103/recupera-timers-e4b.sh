#!/bin/bash
# A2-PASSO E4b: recuperação MANUAL dos timers quando janela-e4b.sh não pôde restaurar (SIGKILL, queda, reboot, falha no start).
# Uso: recupera-timers-e4b.sh [PASTA_DE_ESTADO]   (sem argumento, usa o ponteiro PENDENTE)
# Lê ativos.txt (os .timer que estavam ativos ANTES da pausa), inicia só os que não estão ativos e confere. Não mata nenhum processo:
# se o comando da janela ainda existe, só avisa (filho.pgid) e a decisão é do Breno.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
SYSTEMCTL=${SYSTEMCTL:-systemctl}
PENDENTE=${PENDENTE:-$AQUI/.janela-e4b-pendente}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
ESTADO=${1:-}
if [ -z "$ESTADO" ]; then
  [ -s "$PENDENTE" ] || { echo "não há janela pendente (falta $PENDENTE) e nenhuma pasta foi dada"; exit 2; }
  ESTADO=$(cat "$PENDENTE")
fi
[ -s "$ESTADO/ativos.txt" ] || [ -e "$ESTADO/ativos.txt" ] || { echo "falta $ESTADO/ativos.txt: nada a restaurar com segurança"; exit 2; }
ativo() { $SYSTEMCTL --user is-active "$1" 2>/dev/null || true; }
falhou=0
while IFS= read -r t; do
  [ -n "$t" ] || continue
  if [ "$(ativo "$t")" != active ]; then
    $SYSTEMCTL --user start "$t" || { echo "FALHA ao iniciar $t"; falhou=1; continue; }
  fi
  if [ "$(ativo "$t")" = active ]; then echo "$t: active (ok)"; else echo "$t: NÃO está active"; falhou=1; fi
done < "$ESTADO/ativos.txt"
if [ -s "$ESTADO/filho.pgid" ]; then
  pg=$(cat "$ESTADO/filho.pgid")
  if ps -o stat= -s "$pg" 2>/dev/null | grep -qv '^Z'; then echo "ATENÇÃO: o grupo de processos do comando (PGID $pg) ainda existe; não foi tocado. Decida se encerra: kill -TERM -- -$pg"; fi
fi
if [ "$falhou" = 0 ]; then
  if [ -s "$PENDENTE" ] && [ "$(cat "$PENDENTE")" = "$ESTADO" ]; then rm -f "$PENDENTE"; fi
  echo "RECUPERAÇÃO OK" | tee -a "$ESTADO/recuperacao.txt"
  exit 0
fi
echo "RECUPERAÇÃO INCOMPLETA" | tee -a "$ESTADO/recuperacao.txt"
exit 1
