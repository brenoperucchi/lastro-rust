#!/usr/bin/env bash
# Roteiro de LIMITE DE TEMPO para a janela de correção da leitura (A2, etapa 3). Escopo próprio e novo; não é o roteiro de pausa dos timers
# (janela-e4b.sh) e não mexe em timers, serviços nem backups.
#
#   roda-com-limite-f2e.sh [--limite SEGUNDOS] -- COMANDO [ARGS...]
#
# Roda o comando numa unidade TRANSITÓRIA `systemd --user` (systemd-run --wait --collect), com RuntimeMaxSec = limite (padrão 1800 s = 30 min),
# KillMode=control-group (o limite atinge todos os processos do cgroup, inclusive filhos em outra sessão) e TimeoutStopSec curto (depois dele
# vem SIGKILL). A unidade some sozinha ao fim (--collect); nada é instalado, habilitado nem deixado em execução.
# Ambiente: a unidade transitória NÃO herda o ambiente, então só o que está em LIMITE_PASSA (nomes separados por espaço; padrão abaixo) vai,
# por --setenv, e só se estiver definido. O diretório de trabalho é o atual.
#
# F2E: copia de roda-com-limite.sh (que fica intacto, congelado nas outras fases) com CANCELAMENTO: em TERM/HUP/INT, para a unidade (systemctl --user stop) e
# espera o systemd-run terminar; depois confere o cgroup como no fluxo normal e sai 130 (ou 5 se sobrar processo). Assim o chamador (o envelope que religa o Strata)
# so recebe de volta o controle com a unidade encerrada e confirmada.
#
# Depois do fim, confere o cgroup da unidade: sumiu ou vazio = encerrado. Se ainda houver processo, relata o PID e o estado (D = espera do driver,
# que nenhum sinal mata) e sai 5. Não promete matar espera ininterruptível.
#
# Códigos: 0 ou o código do comando; 124 limite de tempo estourado (processos encerrados); 2 recusa de uso/ambiente; 5 processos remanescentes
# depois do fim (ou "não morreu"); 7 resultado do systemd não reconhecido (inconclusivo); 125 o systemd-run não conseguiu iniciar.
set -u
LIMITE=1800
SD_RUN=${SYSTEMD_RUN:-systemd-run}
SYSCTL=${SYSTEMCTL:-systemctl}
CG_RAIZ=${CGROUP_RAIZ:-/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/app.slice}
PASSA=${LIMITE_PASSA:-PATH HOME CUDA_VISIBLE_DEVICES LEITURA_CORRECAO_AUTORIZADA}
ESPERA_CG=${LIMITE_ESPERA_CGROUP:-15}
TIMEOUT_STOP=${LIMITE_TIMEOUT_STOP:-5}

usa() { echo "uso: roda-com-limite-f2e.sh [--limite SEGUNDOS] -- COMANDO [ARGS...]" >&2; }
if [ "${1:-}" = "--limite" ]; then
  [ $# -ge 2 ] || { usa; exit 2; }
  LIMITE=$2; shift 2
fi
[ "${1:-}" = "--" ] || { usa; exit 2; }
shift
[ $# -ge 1 ] || { usa; exit 2; }
case $LIMITE in ''|*[!0-9]*) echo "limite inválido: $LIMITE" >&2; exit 2;; esac
[ "$LIMITE" -ge 1 ] && [ "$LIMITE" -le 1800 ] || { echo "limite fora de 1..1800 s: $LIMITE" >&2; exit 2; }
command -v "$SD_RUN" >/dev/null 2>&1 || { echo "systemd-run não encontrado: sem como garantir o limite; recusando" >&2; exit 2; }
[ -d "$CG_RAIZ" ] || { echo "cgroup esperado não existe ($CG_RAIZ): sem como conferir o encerramento; recusando" >&2; exit 2; }
# O comando é resolvido aqui (a unidade tem PATH próprio só se o passarmos): caminho absoluto evita surpresa.
CMD0=$1; shift
CMD=$CMD0
case $CMD in /*|./*) ;; *) CMD=$(command -v -- "$CMD0") || { echo "comando não encontrado: $CMD0" >&2; exit 2; };; esac

UNIDADE="lastro-limite-$$-$(date +%s%N)"
ERR=$(mktemp) || exit 2
SETENV=()
for n in $PASSA; do
  if [ -n "${!n+x}" ]; then SETENV+=("--setenv=$n=${!n}"); fi
done
CANCELADO=0
cancela() {
  CANCELADO=1
  "$SYSCTL" --user stop "$UNIDADE" > /dev/null 2>&1
}
trap cancela TERM HUP INT
"$SD_RUN" --user --wait --collect --pipe --unit="$UNIDADE" --slice=app.slice \
  -p RuntimeMaxSec="$LIMITE" -p KillMode=control-group -p TimeoutStopSec="$TIMEOUT_STOP" \
  -p WorkingDirectory="$PWD" "${SETENV[@]}" -- "$CMD" "$@" 2>"$ERR" &
SRPID=$!
wait "$SRPID"; RC=$?
# um sinal interrompe o wait: espera o systemd-run de fato terminar (a unidade ja foi mandada parar em cancela)
while kill -0 "$SRPID" 2> /dev/null; do
  wait "$SRPID"; RC=$?
  if [ "$CANCELADO" = 1 ]; then "$SYSCTL" --user stop "$UNIDADE" > /dev/null 2>&1; sleep 1; fi
done
trap - TERM HUP INT
cat "$ERR" >&2
RESULT=$(sed -n 's/^ *Finished with result: *//p' "$ERR" | tail -n1)
INICIOU=$(grep -c "Running as unit: $UNIDADE" "$ERR")

# Confere o cgroup: espera ele sumir ou esvaziar por até ESPERA_CG s.
CG="$CG_RAIZ/$UNIDADE.service"
REMANESCENTE=0
t=0
while [ -e "$CG/cgroup.procs" ]; do
  if [ -z "$(cat "$CG/cgroup.procs" 2>/dev/null)" ]; then break; fi
  if [ "$t" -ge "$ESPERA_CG" ]; then REMANESCENTE=1; break; fi
  sleep 1; t=$((t + 1))
done
if [ "$REMANESCENTE" = 1 ]; then
  echo "ATENÇÃO: o cgroup $CG ainda tem processos ${ESPERA_CG}s depois do fim da unidade:" >&2
  while read -r pid; do
    [ -n "$pid" ] || continue
    est=$(awk '/^State:/{print $2, $3}' "/proc/$pid/status" 2>/dev/null)
    echo "  pid $pid estado ${est:-?} ($(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-120))" >&2
  done < "$CG/cgroup.procs"
  echo "  estado D = espera ininterruptível do driver: nenhum sinal encerra; o roteiro não promete isso. Não rode outra janela antes de resolver." >&2
  rm -f "$ERR"; exit 5
fi
rm -f "$ERR"

if [ "$CANCELADO" = 1 ]; then
  echo "cancelado por sinal: unidade $UNIDADE encerrada, cgroup vazio" >&2; exit 130
fi
if [ "$INICIOU" -lt 1 ]; then
  echo "o systemd-run não iniciou a unidade (código $RC)" >&2; exit 125
fi
case $RESULT in
  timeout) echo "LIMITE de ${LIMITE}s estourado: unidade encerrada, cgroup vazio" >&2; exit 124;;
  success|exit-code) exit "$RC";;
  signal|core-dump) echo "o comando morreu por sinal (resultado $RESULT, código $RC)" >&2; exit "$RC";;
  *) echo "resultado do systemd não reconhecido: '${RESULT:-vazio}' (código $RC): inconclusivo" >&2; exit 7;;
esac
