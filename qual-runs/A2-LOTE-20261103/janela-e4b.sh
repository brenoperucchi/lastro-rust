#!/bin/bash
# A2-PASSO E4b: pausa TEMPORÁRIA de timers de usuário, execução de UM comando com limite de tempo e restauração conferida.
#
# Uso: janela-e4b.sh --estado DIR [--timers ARQ] [--limite S] [--kill-apos S] [--espera S] -- COMANDO [ARGS...]
#   --estado    pasta NOVA ou vazia onde o estado inicial e o resultado ficam gravados (obrigatória)
#   --timers    lista de .timer de usuário (padrão: timers-e4b.txt ao lado deste arquivo)
#   --limite    segundos até o TERM no grupo do comando (padrão 1800)
#   --kill-apos segundos de espera depois do TERM antes do KILL (padrão 30)
#   --espera    segundos esperando serviços dos timers terminarem antes de pausar o comando (padrão 600); NÃO mata serviço nenhum
#
# O que faz, nesta ordem:
#   1. Falha fechada ANTES de mexer em qualquer coisa: sem lista, sem pasta nova, sem comando, janela anterior não recuperada
#      (arquivo PENDENTE), unidade inexistente ou estado que o systemd não devolve (por exemplo sem XDG_RUNTIME_DIR).
#   2. Captura is-active e is-enabled de cada .timer e .service e grava ativos.txt (os .timer que estavam ativos) e o ponteiro PENDENTE.
#   3. Arma a restauração (EXIT, INT, TERM, HUP) e só então faz `stop` DOS .timer que estavam ativos. Nunca disable, nunca mask.
#   4. Espera os serviços dos timers em execução terminarem. Se não terminarem no prazo, NÃO os mata: não roda o comando, restaura e sai 3.
#   5. Roda o comando numa sessão própria (setsid), com vigia: no limite manda TERM ao grupo, depois KILL; se o processo ainda assim
#      não morrer (estado D, por exemplo), o vigia desiste de esperar e derruba o próprio roteiro para que a restauração rode.
#   6. Restaura: `start` só dos timers que estavam ativos e confere is-active; só então apaga o PENDENTE.
#
# O que NÃO cobre (declarado): SIGKILL deste roteiro, queda da máquina e reboot não rodam o trap. Nesses casos o estado inicial já está em
# ativos.txt e o PENDENTE aponta para a pasta; a recuperação é manual e conferida com recupera-timers-e4b.sh. Um processo que não morre
# nem com KILL continua existindo: o roteiro avisa (saída 5) e restaura os timers mesmo assim.
#
# Saídas: o código do comando; 124 limite de tempo; 2 recusa antes de mexer; 3 serviço de timer não terminou; 4 restauração incompleta
# (prevalece sobre os outros); 5 o comando não morreu (ou o grupo não foi reconhecido como o do comando, e então nenhum sinal foi enviado);
# 6 `stop` falhou ou timer continuou ativo; 7 consulta ao systemd inconclusiva depois do preflight; 125 o comando não deixou código de saída.
# Se o comando terminar no mesmo segundo do limite, o vigia confere o grupo antes de marcar o limite: grupo já morto não vira 124.
# Identidade: filho.id guarda boot_id e starttime do líder do grupo; nenhum sinal é enviado a um grupo cujo líder tenha outra identidade
# (número de PID reaproveitado). Um comando só começa depois do aval do supervisor (arquivo vai): sinal na largada cancela sem rodar nada.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
SYSTEMCTL=${SYSTEMCTL:-systemctl}
PENDENTE=${PENDENTE:-$AQUI/.janela-e4b-pendente}
PASSO=${JANELA_PASSO_ESPERA:-5}
TIMERS_ARQ=$AQUI/timers-e4b.txt
LIMITE=1800; KILL_APOS=30; ESPERA=600; ESTADO=
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}

recusa() { echo "janela-e4b: $*" >&2; exit 2; }
inteiro() { case "$2" in ''|*[!0-9]*) recusa "$1 precisa ser um inteiro positivo (veio '$2')";; esac; [ "$2" -ge 1 ] || recusa "$1 precisa ser maior que zero"; }

while [ $# -gt 0 ]; do
  case $1 in
    --estado) ESTADO=${2:-}; shift 2;;
    --timers) TIMERS_ARQ=${2:-}; shift 2;;
    --limite) LIMITE=${2:-}; shift 2;;
    --kill-apos) KILL_APOS=${2:-}; shift 2;;
    --espera) ESPERA=${2:-}; shift 2;;
    --) shift; break;;
    *) recusa "opção desconhecida: $1";;
  esac
done
[ $# -ge 1 ] || recusa "falta o comando depois de --"
[ -n "$ESTADO" ] || recusa "falta --estado"
inteiro --limite "$LIMITE"; inteiro --kill-apos "$KILL_APOS"; inteiro --espera "$ESPERA"
[ -f "$TIMERS_ARQ" ] || recusa "falta a lista de timers $TIMERS_ARQ"
mapfile -t TIMERS < <(grep -v '^[[:space:]]*\(#\|$\)' "$TIMERS_ARQ")
[ "${#TIMERS[@]}" -ge 1 ] || recusa "a lista $TIMERS_ARQ não tem nenhum timer"
for t in "${TIMERS[@]}"; do
  [[ $t =~ ^[A-Za-z0-9@._-]+\.timer$ ]] || recusa "linha inválida na lista de timers: '$t'"
done
[ ! -e "$PENDENTE" ] || recusa "há uma janela anterior não recuperada: $(cat "$PENDENTE" 2>/dev/null). Rode recupera-timers-e4b.sh antes de começar outra."
if [ -e "$ESTADO" ]; then
  [ -d "$ESTADO" ] || recusa "--estado existe e não é pasta: $ESTADO"
  [ -z "$(ls -A "$ESTADO" 2>/dev/null)" ] || recusa "--estado precisa ser pasta nova ou vazia: $ESTADO"
fi
mkdir -p "$ESTADO" || recusa "não consegui criar $ESTADO"
ESTADO=$(cd "$ESTADO" && pwd -P)

ativo() { $SYSTEMCTL --user is-active "$1" 2>/dev/null || true; }
estado_conhecido() { case "$1" in active|inactive|failed|activating|deactivating|reloading|maintenance) return 0;; *) return 1;; esac; }
# Leitura VALIDADA: saída vazia ou desconhecida (consulta que falhou) nunca vira "inactive".
le() { local v; v=$(ativo "$1"); estado_conhecido "$v" || return 1; printf '%s\n' "$v"; }
ident_pid() { # "boot_id starttime" do processo $1; falha se ele não existe
  local b st
  [ -r "/proc/$1/stat" ] || return 1
  b=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null) || return 1
  st=$(sed 's/^.*) //' "/proc/$1/stat" 2>/dev/null | cut -d' ' -f20)
  [ -n "$st" ] || return 1
  echo "$b $st"
}
abandona() { rm -f "$PENDENTE" "$PENDENTE.$$.tmp"; recusa "$*"; }

# 1b. Falha fechada sobre o que o systemd devolve: a unidade existe.
for t in "${TIMERS[@]}"; do
  $SYSTEMCTL --user cat "$t" >/dev/null 2>&1 || recusa "o systemd não conhece a unidade $t (escopo --user, XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR)"
done

# 1c. Reserva ATÔMICA da janela (ln falha se o ponteiro já existe): duas janelas simultâneas nunca passam juntas.
printf '%s\n' "$ESTADO" > "$PENDENTE.$$.tmp" && sync "$PENDENTE.$$.tmp" && ln "$PENDENTE.$$.tmp" "$PENDENTE" 2>/dev/null; rc=$?
rm -f "$PENDENTE.$$.tmp"
[ "$rc" = 0 ] || recusa "não consegui reservar a janela: há outra janela em curso ou não recuperada ($(cat "$PENDENTE" 2>/dev/null)). Rode recupera-timers-e4b.sh antes de começar outra."

# Até o `stop` nada foi pausado: um sinal ou erro nesta fase só precisa soltar a reserva (o trap definitivo vem depois).
trap 'rm -f "$PENDENTE" "$PENDENTE.$$.tmp"' EXIT
trap 'exit 130' INT; trap 'exit 143' TERM; trap 'exit 129' HUP

# 2. Captura VALIDADA e estado inicial em arquivo (antes de qualquer stop). Falha de leitura = recusa, sem pausar nada.
PAI=$$
PAI_ID=$(ident_pid "$PAI") || abandona "não consegui identificar este processo em /proc"
echo "$PAI $PAI_ID" > "$ESTADO/supervisor.id"
{ echo "início: $(date -Is) pid=$$ XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"; } > "$ESTADO/estado.txt"
: > "$ESTADO/ativos.tmp"
for t in "${TIMERS[@]}"; do
  s=${t%.timer}.service
  vt=$(le "$t") || abandona "estado de $t não reconhecido: o systemd de usuário respondeu?"
  vs=$(le "$s") || abandona "estado de $s não reconhecido: o systemd de usuário respondeu?"
  echo "$t is-active=$vt is-enabled=$($SYSTEMCTL --user is-enabled "$t" 2>/dev/null || true)" >> "$ESTADO/estado.txt"
  echo "$s is-active=$vs is-enabled=$($SYSTEMCTL --user is-enabled "$s" 2>/dev/null || true)" >> "$ESTADO/estado.txt"
  [ "$vt" = active ] && echo "$t" >> "$ESTADO/ativos.tmp"
done
sync "$ESTADO/estado.txt" "$ESTADO/ativos.tmp" "$ESTADO/supervisor.id" && mv "$ESTADO/ativos.tmp" "$ESTADO/ativos.txt" || abandona "não consegui gravar o estado inicial"

RESTAURADO=0; RESTAURA_RC=0; PGID=; SPID=; VIGIA=; NAOMORREU=0; LIMITE_ATINGIDO=0; INT_PEND=; ANC=
# O comando roda numa sessão própria (setsid): a sessão tem o id do líder. Dentro dela há uma ÂNCORA: um `sleep` que ignora TERM, criado antes do
# comando, com identidade (boot_id+starttime) gravada em ancora.id. Enquanto a âncora vive na sessão o número do grupo não pode ser reaproveitado,
# mesmo que o líder já tenha morrido e o bash o tenha colhido; só com a âncora verificada se envia sinal ao grupo. A âncora não conta como "grupo vivo".
grupo_vivo() { [ -n "$PGID" ] && ps -o pid=,stat= -s "$PGID" 2>/dev/null | awk -v a="${ANC:-0}" '$1!=a && $2!~/^Z/ {n++} END{exit !n}'; }
ancora_ok() { # a âncora existe, tem a identidade gravada e continua na sessão do grupo
  [ -n "$ANC" ] && [ -n "$PGID" ] && [ -s "$ESTADO/ancora.id" ] || return 1
  [ "$(ident_pid "$ANC" 2>/dev/null)" = "$(cat "$ESTADO/ancora.id")" ] || return 1
  [ "$(ps -o sid= -p "$ANC" 2>/dev/null | tr -d ' ')" = "$PGID" ]
}
mata_ancora() { ancora_ok && kill -KILL "$ANC" 2>/dev/null; return 0; }
# 0 = grupo vivo e verificado pela âncora; 1 = não existe mais; 2 = existe, mas sem âncora verificável (número possivelmente reaproveitado): NÃO sinalizar.
grupo_confere() {
  grupo_vivo || return 1
  ancora_ok || return 2
  return 0
}
mata_grupo() { # TERM, espera $KILL_APOS, KILL, espera 5 s; 0 = o grupo morreu, 1 = não morreu, 2 = identidade divergente (nada enviado)
  grupo_confere; local c=$?
  [ "$c" = 1 ] && return 0
  [ "$c" = 2 ] && return 2
  kill -TERM -- "-$PGID" 2>/dev/null
  for _ in $(seq 1 "$KILL_APOS"); do grupo_vivo || return 0; sleep 1; done
  grupo_confere; c=$?
  [ "$c" = 1 ] && return 0
  [ "$c" = 2 ] && return 2
  kill -KILL -- "-$PGID" 2>/dev/null
  for _ in 1 2 3 4 5; do grupo_vivo || return 0; sleep 1; done
  return 1
}
restaura() {
  [ "$RESTAURADO" = 0 ] || return
  RESTAURADO=1
  touch "$ESTADO/fim.flag"
  [ -n "$VIGIA" ] && kill "$VIGIA" 2>/dev/null
  # Um sinal pode chegar antes da leitura do PGID: recupera pelo arquivo e, no limite, pelo PID do setsid (filho nosso, ainda não colhido).
  [ -n "$PGID" ] || PGID=$(cat "$ESTADO/filho.pgid" 2>/dev/null || true)
  [ -n "$PGID" ] || { [ -n "$SPID" ] && { PGID=$SPID; touch "$ESTADO/vai.cancela"; kill -TERM "$SPID" 2>/dev/null; }; }
  touch "$ESTADO/vai.cancela"
  [ -n "$ANC" ] || { ANC=$(cat "$ESTADO/ancora.pid" 2>/dev/null || true); }
  if grupo_vivo; then
    mata_grupo; r=$?
    [ "$r" = 0 ] || NAOMORREU=1
  fi
  grupo_vivo || mata_ancora
  falha_r=0
  : > "$ESTADO/restauracao.txt"
  while IFS= read -r t; do
    [ -n "$t" ] || continue
    if [ "$(ativo "$t")" != active ]; then
      $SYSTEMCTL --user start "$t" 2>>"$ESTADO/restauracao.txt" || { echo "FALHA ao iniciar $t" | tee -a "$ESTADO/restauracao.txt" >&2; falha_r=1; }
    fi
  done < "$ESTADO/ativos.txt"
  for t in "${TIMERS[@]}"; do
    v=$(le "$t") || { echo "$t: estado inconclusivo na conferência final" | tee -a "$ESTADO/restauracao.txt" >&2; falha_r=1; continue; }
    if grep -qx "$t" "$ESTADO/ativos.txt"; then
      if [ "$v" = active ]; then echo "$t is-active=active (era active): ok" >> "$ESTADO/restauracao.txt"; else echo "$t NÃO voltou a active ($v)" | tee -a "$ESTADO/restauracao.txt" >&2; falha_r=1; fi
    else
      if [ "$v" = active ]; then echo "$t estava parado e está active: NÃO devia" | tee -a "$ESTADO/restauracao.txt" >&2; falha_r=1; else echo "$t is-active=$v (não estava ativo; não foi ativado): ok" >> "$ESTADO/restauracao.txt"; fi
    fi
  done
  [ -e "$ESTADO/nao-morreu.flag" ] && NAOMORREU=1
  if [ "$falha_r" = 0 ]; then
    echo "RESTAURAÇÃO OK" >> "$ESTADO/restauracao.txt"
    if [ "$NAOMORREU" = 1 ] || grupo_vivo; then
      echo "O comando da janela não terminou: ponteiro MANTIDO (nova janela recusada até ele terminar e a recuperação concluir)" | tee -a "$ESTADO/restauracao.txt" >&2
    else
      # só apaga o ponteiro se ele ainda aponta para ESTA janela
      [ "$(cat "$PENDENTE" 2>/dev/null)" = "$ESTADO" ] && rm -f "$PENDENTE"
    fi
  else
    echo "RESTAURAÇÃO INCOMPLETA: rode recupera-timers-e4b.sh $ESTADO" | tee -a "$ESTADO/restauracao.txt" >&2
    RESTAURA_RC=1
  fi
  sync "$ESTADO/restauracao.txt" 2>/dev/null
}
finaliza() {
  rc=$?
  trap '' INT TERM HUP   # um segundo sinal não pode interromper a restauração
  restaura
  [ -e "$ESTADO/limite.flag" ] && LIMITE_ATINGIDO=1
  [ -e "$ESTADO/nao-morreu.flag" ] && NAOMORREU=1
  if [ "$RESTAURA_RC" != 0 ]; then rc=4
  elif [ "$NAOMORREU" = 1 ]; then rc=5
  elif [ "$LIMITE_ATINGIDO" = 1 ]; then rc=124; fi
  { echo "fim: $(date -Is)"; echo "saída do roteiro: $rc"; echo "limite atingido: $LIMITE_ATINGIDO"; echo "processo não morreu: $NAOMORREU"; echo "restauração com falha: $RESTAURA_RC"; } > "$ESTADO/resultado.txt"
  exit "$rc"
}
trap finaliza EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# 3. Pausa: só os .timer que estavam ativos.
while IFS= read -r t; do
  [ -n "$t" ] || continue
  $SYSTEMCTL --user stop "$t" || { echo "janela-e4b: falha ao parar $t" >&2; exit 6; }
  v=$(le "$t") || { echo "janela-e4b: não consegui confirmar o stop de $t" >&2; exit 7; }
  [ "$v" != active ] || { echo "janela-e4b: $t continua active depois do stop" >&2; exit 6; }
done < "$ESTADO/ativos.txt"

# 4. Espera serviços dos timers em execução; nunca os mata.
fim_espera=$(( $(date +%s) + ESPERA ))
while :; do
  ocupado=0
  for t in "${TIMERS[@]}"; do
    sv=$(le "${t%.timer}.service") || { echo "janela-e4b: consulta inconclusiva sobre ${t%.timer}.service; o comando não rodou" >&2; exit 7; }
    case "$sv" in active|activating|deactivating|reloading) ocupado=1;; esac
  done
  [ $ocupado = 0 ] && break
  [ "$(date +%s)" -lt "$fim_espera" ] || { echo "janela-e4b: serviço de timer ainda em execução depois de ${ESPERA}s; nada foi morto, o comando não rodou" >&2; exit 3; }
  sleep "$PASSO"
done

# 5. O comando, numa sessão própria, com vigia de tempo. O lançamento é uma fase PROTEGIDA: sinais só são anotados (INT_PEND) até o
#    comando estar registrado e com vigia; e o comando só começa depois do aval (arquivo vai), então um sinal na largada cancela sem rodar nada.
echo "comando: $*" > "$ESTADO/comando.txt"; echo "gancho_atraso_lancamento=${JANELA_ATRASO_LANCAMENTO:-}" >> "$ESTADO/comando.txt"
trap 'INT_PEND=130' INT; trap 'INT_PEND=143' TERM; trap 'INT_PEND=129' HUP
read -r -d '' EMB <<'EOS'
echo $$ > "$1"; V=$2; A=$3; L=$4; shift 4
bash -c 'trap "" TERM INT HUP; exec sleep "$1"' _ "$L" & echo $! > "$A"
n=0; while [ ! -e "$V" ]; do [ -e "$V.cancela" ] && exit 125; n=$((n+1)); [ "$n" -le 600 ] || exit 125; sleep 0.05; done # GATE
"$@"; echo $? > "$RCF"
EOS
RCF="$ESTADO/rc-comando.txt" setsid bash -c "$EMB" _ "$ESTADO/filho.pgid" "$ESTADO/vai" "$ESTADO/ancora.pid" "$((LIMITE + KILL_APOS + 600))" "$@" &
[ -z "${JANELA_ATRASO_LANCAMENTO:-}" ] || sleep "$JANELA_ATRASO_LANCAMENTO"   # gancho de teste: alarga a fronteira do lançamento
SPID=$!
for _ in $(seq 1 50); do [ -s "$ESTADO/filho.pgid" ] && [ -s "$ESTADO/ancora.pid" ] && break; sleep 0.1; done
PGID=$(cat "$ESTADO/filho.pgid" 2>/dev/null || true)
ANC=$(cat "$ESTADO/ancora.pid" 2>/dev/null || true)
# Sem identidade do líder e da âncora o comando não é liberado (falha fechada).
{ [ -n "$PGID" ] && [ -n "$ANC" ] && ident_pid "$PGID" > "$ESTADO/filho.id" 2>/dev/null && ident_pid "$ANC" > "$ESTADO/ancora.id" 2>/dev/null && [ -s "$ESTADO/filho.id" ] && [ -s "$ESTADO/ancora.id" ]; } || INT_PEND=125
trap 'exit 130' INT; trap 'exit 143' TERM; trap 'exit 129' HUP
(
  echo "$BASHPID" > "$ESTADO/vigia.pid"
  for _ in $(seq 1 "$LIMITE"); do
    [ -e "$ESTADO/fim.flag" ] && exit 0
    grupo_vivo || { mata_ancora; exit 0; }   # comando já acabou (ou o supervisor morreu e o comando também): este vigia não deve sobrar
    sleep 1
  done
  [ -e "$ESTADO/fim.flag" ] && exit 0
  grupo_vivo || { mata_ancora; exit 0; }
  echo 1 > "$ESTADO/limite.flag"
  mata_grupo; r=$?
  if [ "$r" != 0 ]; then
    echo "$r" > "$ESTADO/nao-morreu.flag"
    # só sinaliza o supervisor se ele ainda é o mesmo processo (PID não reaproveitado)
    [ "$(ident_pid "$PAI" 2>/dev/null)" = "$PAI_ID" ] && kill -TERM "$PAI" 2>/dev/null
  fi
) &
VIGIA=$!
[ -z "$INT_PEND" ] || { echo "janela-e4b: interrompido ou sem identidade na largada; o comando não rodou" >&2; exit "$INT_PEND"; }
date -Is > "$ESTADO/comando-inicio.txt"
touch "$ESTADO/vai"
# A âncora (e não o líder, que o bash colhe sozinho) mantém o número do grupo reservado enquanto houver descendentes; ela é morta ao fim.
while grupo_vivo; do sleep 0.2; done
mata_ancora
wait "$SPID" 2>/dev/null
touch "$ESTADO/fim.flag"
wait "$VIGIA" 2>/dev/null
VIGIA=
date -Is > "$ESTADO/comando-fim.txt"
[ -e "$ESTADO/limite.flag" ] && exit 124
[ -s "$ESTADO/rc-comando.txt" ] || exit 125
exit "$(cat "$ESTADO/rc-comando.txt")"
