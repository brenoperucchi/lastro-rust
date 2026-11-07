#!/bin/bash
# A2-PASSO E4b: recuperação MANUAL dos timers quando janela-e4b.sh não pôde restaurar (SIGKILL, queda, reboot, falha no start).
# Uso: recupera-timers-e4b.sh [PASTA_DE_ESTADO]   (sem argumento, usa o ponteiro PENDENTE)
# Lê ativos.txt (os .timer que estavam ativos ANTES da pausa), inicia só os que não estão ativos e confere. Não mata nenhum processo.
#
# Saídas: 0 recuperação completa e ponteiro apagado; 1 restauração incompleta (ponteiro mantido); 2 recusa (sem estado, ou a janela
# ainda está viva: o supervisor dela ainda existe); 3 timers restaurados, MAS o comando da janela anterior ainda está vivo: o ponteiro
# fica, e uma nova janela continua recusada até o comando terminar (a decisão de encerrá-lo é do Breno; nada é morto aqui).
# Identidade: o comando e o supervisor só são tratados como "os da janela" se boot_id e starttime gravados baterem com o processo atual
# (depois de reboot, ou com PID reaproveitado, o número pode ser de outro processo: nesse caso nada é dito sobre ele).
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
[ -e "$ESTADO/ativos.txt" ] || { echo "falta $ESTADO/ativos.txt: nada a restaurar com segurança"; exit 2; }
ident_pid() { # "boot_id starttime" do processo $1; falha se ele não existe
  local b st
  [ -r "/proc/$1/stat" ] || return 1
  b=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null) || return 1
  st=$(sed 's/^.*) //' "/proc/$1/stat" 2>/dev/null | cut -d' ' -f20)
  [ -n "$st" ] || return 1
  echo "$b $st"
}
ativo() { $SYSTEMCTL --user is-active "$1" 2>/dev/null || true; }

# A janela ainda está viva? (supervisor.id = "pid boot_id starttime")
if [ -s "$ESTADO/supervisor.id" ]; then
  read -r spid sboot sstart < "$ESTADO/supervisor.id"
  if [ "$(ident_pid "$spid" 2>/dev/null)" = "$sboot $sstart" ]; then
    echo "RECUSADO: a janela ainda está viva (supervisor PID $spid). Recuperar agora religaria os timers no meio da medida."
    exit 2
  fi
fi

falhou=0
while IFS= read -r t; do
  [ -n "$t" ] || continue
  if [ "$(ativo "$t")" != active ]; then
    $SYSTEMCTL --user start "$t" || { echo "FALHA ao iniciar $t"; falhou=1; continue; }
  fi
  if [ "$(ativo "$t")" = active ]; then echo "$t: active (ok)"; else echo "$t: NÃO está active"; falhou=1; fi
done < "$ESTADO/ativos.txt"

viva=0
if [ -s "$ESTADO/filho.pgid" ]; then
  pg=$(cat "$ESTADO/filho.pgid")
  if ps -o stat= -s "$pg" 2>/dev/null | grep -qv '^Z'; then
    if [ -s "$ESTADO/filho.id" ]; then
      read -r fboot fstart < "$ESTADO/filho.id"
      atual_boot=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || true)
      if [ "$fboot" != "$atual_boot" ]; then
        echo "o número $pg existe, mas é de outro boot: não é o comando da janela; nada a fazer com ele"
      elif [ -r "/proc/$pg/stat" ] && [ "$(ident_pid "$pg")" != "$fboot $fstart" ]; then
        echo "o número $pg existe, mas o líder tem outro início: PID reaproveitado, não é o comando da janela; nada a fazer com ele"
      else
        viva=1
      fi
    else
      viva=1   # sem identidade gravada: não dá para provar que é outro processo; conservador
    fi
    [ "$viva" = 0 ] || echo "ATENÇÃO: o comando da janela anterior ainda está vivo (grupo $pg); não foi tocado. Se decidir encerrá-lo: kill -TERM -- -$pg"
  fi
fi
if [ "$falhou" != 0 ]; then
  echo "RECUPERAÇÃO INCOMPLETA" | tee -a "$ESTADO/recuperacao.txt"
  exit 1
fi
if [ "$viva" = 1 ]; then
  echo "RECUPERAÇÃO PARCIAL: timers restaurados; ponteiro mantido porque o comando anterior ainda está vivo" | tee -a "$ESTADO/recuperacao.txt"
  exit 3
fi
if [ -s "$PENDENTE" ] && [ "$(cat "$PENDENTE")" = "$ESTADO" ]; then rm -f "$PENDENTE"; fi
echo "RECUPERAÇÃO OK" | tee -a "$ESTADO/recuperacao.txt"
exit 0
