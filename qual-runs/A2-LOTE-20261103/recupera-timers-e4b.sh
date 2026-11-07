#!/bin/bash
# A2-PASSO E4b: recuperação MANUAL dos timers quando janela-e4b.sh não pôde restaurar (SIGKILL, queda, reboot, falha no start).
# Uso: recupera-timers-e4b.sh [PASTA_DE_ESTADO]   (sem argumento, usa o ponteiro PENDENTE)
# Lê ativos.txt (os .timer que estavam ativos ANTES da pausa), inicia só os que não estão ativos e confere. Não mata nenhum processo.
#
# Saídas: 0 recuperação completa e ponteiro apagado; 1 restauração incompleta ou divergência em timer que estava parado (ponteiro mantido);
# 2 recusa (a pasta não é a janela pendente atual, outra recuperação em curso, ou a janela ainda está viva); 3 o comando da janela
# anterior ainda está vivo: NADA é feito (timers não religados, ponteiro mantido); rode de novo depois que ele terminar. Nada é morto aqui.
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
ESTADO=$(cd "$ESTADO" 2>/dev/null && pwd -P) || { echo "a pasta de estado não existe"; exit 2; }
# Só recupera a janela que é DONA do ponteiro atual: uma pasta antiga nunca mexe nos timers de outra janela em curso.
[ -s "$PENDENTE" ] && [ "$(cat "$PENDENTE")" = "$ESTADO" ] || { echo "RECUSADO: $ESTADO não é a janela pendente atual (ponteiro: $(cat "$PENDENTE" 2>/dev/null || echo ausente)). Nada foi tocado."; exit 2; }
# Duas recuperações ao mesmo tempo não se atropelam.
exec 8> "$PENDENTE.rec.lock"
flock -n 8 || { echo "RECUSADO: outra recuperação está em curso"; exit 2; }
# Reconfere o dono dentro da trava (a outra recuperação pode ter acabado de apagar o ponteiro).
[ -s "$PENDENTE" ] && [ "$(cat "$PENDENTE")" = "$ESTADO" ] || { echo "RECUSADO: a janela já foi recuperada ou substituída. Nada foi tocado."; exit 2; }
ident_pid() { # "boot_id starttime" do processo $1; falha se ele não existe
  local b st
  [ -r "/proc/$1/stat" ] || return 1
  b=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null) || return 1
  st=$(sed 's/^.*) //' "/proc/$1/stat" 2>/dev/null | cut -d' ' -f20)
  [ -n "$st" ] || return 1
  echo "$b $st"
}
ativo() { $SYSTEMCTL --user is-active "$1" 2>/dev/null || true; }
conhecido() { case "$1" in active|inactive|failed|activating|deactivating|reloading|maintenance) return 0;; *) return 1;; esac; }

# A janela ainda está viva? (supervisor.id = "pid boot_id starttime")
if [ -s "$ESTADO/supervisor.id" ]; then
  read -r spid sboot sstart < "$ESTADO/supervisor.id"
  if [ "$(ident_pid "$spid" 2>/dev/null)" = "$sboot $sstart" ]; then
    echo "RECUSADO: a janela ainda está viva (supervisor PID $spid). Recuperar agora religaria os timers no meio da medida."
    exit 2
  fi
fi
# Sem ativos.txt nenhum `stop` chegou a acontecer (ele é gravado antes do primeiro): só havia a reserva da janela.
if [ ! -e "$ESTADO/ativos.txt" ]; then
  rm -f "$PENDENTE"
  echo "nada foi parado nesta janela (falta ativos.txt, supervisor encerrado); ponteiro apagado" | tee -a "$ESTADO/recuperacao.txt"
  echo "RECUPERAÇÃO OK" | tee -a "$ESTADO/recuperacao.txt"
  exit 0
fi

# O comando da janela ainda vive? Se viver, os timers NÃO são religados: a medida segue protegida e o vigia aplica o limite.
viva=0
if [ -s "$ESTADO/filho.pgid" ]; then
  pg=$(cat "$ESTADO/filho.pgid")
  anc=$(cat "$ESTADO/ancora.pid" 2>/dev/null || echo 0)
  if ps -o pid=,stat= -s "$pg" 2>/dev/null | awk -v a="$anc" '$1!=a && $2!~/^Z/ {n++} END{exit !n}'; then
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
  fi
fi
# Comando morto: se a âncora (sleep da janela, que sobra de propósito) ainda vive na mesma sessão com a identidade gravada, ela é encerrada.
if [ "$viva" = 0 ] && [ -s "$ESTADO/ancora.id" ] && [ -n "${anc:-}" ] && [ "$anc" != 0 ] \
   && [ "$(ident_pid "$anc" 2>/dev/null)" = "$(cat "$ESTADO/ancora.id")" ] && [ "$(ps -o sid= -p "$anc" 2>/dev/null | tr -d ' ')" = "$pg" ]; then
  kill -KILL "$anc" 2>/dev/null
fi
if [ "$viva" = 1 ]; then
  echo "O comando da janela anterior ainda está vivo (grupo $pg); nada foi tocado e os timers NÃO foram religados (a medida ainda roda; o vigia encerra no limite)."
  echo "Rode de novo depois que ele terminar. Se decidir encerrá-lo antes: kill -TERM -- -$pg (a medida então fica sem conclusão)."
  echo "RECUPERAÇÃO ADIADA: comando vivo; ponteiro mantido" | tee -a "$ESTADO/recuperacao.txt"
  exit 3
fi

falhou=0
while IFS= read -r t; do
  [ -n "$t" ] || continue
  if [ "$(ativo "$t")" != active ]; then
    $SYSTEMCTL --user start "$t" || { echo "FALHA ao iniciar $t"; falhou=1; continue; }
  fi
  if [ "$(ativo "$t")" = active ]; then echo "$t: active (ok)"; else echo "$t: NÃO está active"; falhou=1; fi
done < "$ESTADO/ativos.txt"
# Os timers que estavam PARADOS na captura têm de continuar parados: divergência ou consulta inconclusiva = incompleto (nada é parado nem iniciado aqui).
while read -r t resto; do
  case "$t" in *.timer) ;; *) continue;; esac
  ini=${resto#is-active=}; ini=${ini%% *}
  [ "$ini" = active ] && continue
  v=$(ativo "$t")
  if ! conhecido "$v"; then echo "$t: estado inconclusivo ('$v')"; falhou=1
  elif [ "$v" = active ]; then echo "$t: estava $ini na captura e está active: DIVERGÊNCIA (não foi alterado)"; falhou=1
  else echo "$t: $v (estava $ini): ok"; fi
done < "$ESTADO/estado.txt"
if [ "$falhou" != 0 ]; then
  echo "RECUPERAÇÃO INCOMPLETA" | tee -a "$ESTADO/recuperacao.txt"
  exit 1
fi
rm -f "$PENDENTE"
echo "RECUPERAÇÃO OK" | tee -a "$ESTADO/recuperacao.txt"
exit 0
