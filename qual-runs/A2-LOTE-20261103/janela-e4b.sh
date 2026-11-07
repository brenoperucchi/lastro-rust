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
# (prevalece sobre os outros); 5 processo do comando não morreu; 6 `stop` falhou; 125 o comando não deixou código de saída.
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

# 1b. Falha fechada sobre o que o systemd devolve: unidade existe e o estado é um dos conhecidos.
for t in "${TIMERS[@]}"; do
  s=${t%.timer}.service
  $SYSTEMCTL --user cat "$t" >/dev/null 2>&1 || recusa "o systemd não conhece a unidade $t (escopo --user, XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR)"
  for u in "$t" "$s"; do
    estado_conhecido "$(ativo "$u")" || recusa "estado de $u não reconhecido ('$(ativo "$u")'): o systemd de usuário respondeu?"
  done
done

# 2. Captura e estado inicial em arquivo (antes de qualquer stop).
{
  echo "início: $(date -Is) pid=$$ XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
  for t in "${TIMERS[@]}"; do
    s=${t%.timer}.service
    echo "$t is-active=$(ativo "$t") is-enabled=$($SYSTEMCTL --user is-enabled "$t" 2>/dev/null || true)"
    echo "$s is-active=$(ativo "$s") is-enabled=$($SYSTEMCTL --user is-enabled "$s" 2>/dev/null || true)"
  done
} > "$ESTADO/estado.txt"
: > "$ESTADO/ativos.txt"
for t in "${TIMERS[@]}"; do
  [ "$(ativo "$t")" = active ] && echo "$t" >> "$ESTADO/ativos.txt"
done
printf '%s\n' "$ESTADO" > "$PENDENTE.tmp" && sync "$ESTADO/estado.txt" "$ESTADO/ativos.txt" "$PENDENTE.tmp" && mv "$PENDENTE.tmp" "$PENDENTE" || recusa "não consegui gravar o estado inicial e o ponteiro"

RESTAURADO=0; RESTAURA_RC=0; PGID=; SPID=; VIGIA=; NAOMORREU=0; LIMITE_ATINGIDO=0
# O comando roda numa sessão própria (setsid): a sessão tem o id do líder. Zumbi (Z) não conta como vivo: o líder só é colhido pelo `wait`.
grupo_vivo() { [ -n "$PGID" ] && ps -o stat= -s "$PGID" 2>/dev/null | grep -qv '^Z'; }
mata_grupo() { # TERM, espera $KILL_APOS, KILL, espera 5 s; devolve 0 se o grupo morreu
  grupo_vivo || return 0
  kill -TERM -- "-$PGID" 2>/dev/null
  for _ in $(seq 1 "$KILL_APOS"); do grupo_vivo || return 0; sleep 1; done
  kill -KILL -- "-$PGID" 2>/dev/null
  for _ in 1 2 3 4 5; do grupo_vivo || return 0; sleep 1; done
  return 1
}
restaura() {
  [ "$RESTAURADO" = 0 ] || return
  RESTAURADO=1
  touch "$ESTADO/fim.flag"
  [ -n "$VIGIA" ] && kill "$VIGIA" 2>/dev/null
  # Um sinal pode chegar entre o início do comando e a leitura do PGID: recupera pelo arquivo e, no limite, pelo PID do setsid.
  [ -n "$PGID" ] || PGID=$(cat "$ESTADO/filho.pgid" 2>/dev/null || true)
  [ -n "$PGID" ] || { [ -n "$SPID" ] && { PGID=$SPID; kill -TERM "$SPID" 2>/dev/null; }; }
  if grupo_vivo; then
    mata_grupo || NAOMORREU=1
  fi
  falhou=0
  : > "$ESTADO/restauracao.txt"
  while IFS= read -r t; do
    [ -n "$t" ] || continue
    if [ "$(ativo "$t")" != active ]; then
      $SYSTEMCTL --user start "$t" 2>>"$ESTADO/restauracao.txt" || { echo "FALHA ao iniciar $t" | tee -a "$ESTADO/restauracao.txt" >&2; falhou=1; }
    fi
  done < "$ESTADO/ativos.txt"
  for t in "${TIMERS[@]}"; do
    if grep -qx "$t" "$ESTADO/ativos.txt"; then
      [ "$(ativo "$t")" = active ] && echo "$t is-active=active (era active): ok" >> "$ESTADO/restauracao.txt" || { echo "$t NÃO voltou a active" | tee -a "$ESTADO/restauracao.txt" >&2; falhou=1; }
    else
      [ "$(ativo "$t")" = active ] && { echo "$t estava parado e está active: NÃO devia" | tee -a "$ESTADO/restauracao.txt" >&2; falhou=1; } || echo "$t is-active=$(ativo "$t") (não estava ativo; não foi ativado): ok" >> "$ESTADO/restauracao.txt"
    fi
  done
  if [ "$falhou" = 0 ]; then
    echo "RESTAURAÇÃO OK" >> "$ESTADO/restauracao.txt"
    rm -f "$PENDENTE"
  else
    echo "RESTAURAÇÃO INCOMPLETA: rode recupera-timers-e4b.sh $ESTADO" | tee -a "$ESTADO/restauracao.txt" >&2
    RESTAURA_RC=1
  fi
  sync "$ESTADO/restauracao.txt" 2>/dev/null
}
finaliza() {
  rc=$?
  restaura
  { echo "fim: $(date -Is)"; echo "saída do roteiro: $rc"; echo "limite atingido: $LIMITE_ATINGIDO"; echo "processo não morreu: $NAOMORREU"; echo "restauração com falha: $RESTAURA_RC"; } > "$ESTADO/resultado.txt"
  [ "$RESTAURA_RC" = 0 ] || exit 4
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
  [ "$(ativo "$t")" != active ] || { echo "janela-e4b: $t continua active depois do stop" >&2; exit 6; }
done < "$ESTADO/ativos.txt"

# 4. Espera serviços dos timers em execução; nunca os mata.
fim_espera=$(( $(date +%s) + ESPERA ))
while :; do
  ocupado=0
  for t in "${TIMERS[@]}"; do
    case "$(ativo "${t%.timer}.service")" in active|activating|deactivating|reloading) ocupado=1;; esac
  done
  [ $ocupado = 0 ] && break
  [ "$(date +%s)" -lt "$fim_espera" ] || { echo "janela-e4b: serviço de timer ainda em execução depois de ${ESPERA}s; nada foi morto, o comando não rodou" >&2; exit 3; }
  sleep "$PASSO"
done

# 5. O comando, numa sessão própria, com vigia de tempo.
echo "comando: $*" > "$ESTADO/comando.txt"; date -Is > "$ESTADO/comando-inicio.txt"
RCF="$ESTADO/rc-comando.txt" setsid bash -c 'echo $$ > "$1"; shift; "$@"; echo $? > "$RCF"' _ "$ESTADO/filho.pgid" "$@" &
SPID=$!
for _ in $(seq 1 50); do [ -s "$ESTADO/filho.pgid" ] && break; sleep 0.1; done
PGID=$(cat "$ESTADO/filho.pgid" 2>/dev/null || true)
[ -n "$PGID" ] || { echo "janela-e4b: o comando não chegou a iniciar" >&2; exit 125; }
PAI=$$
(
  for _ in $(seq 1 "$LIMITE"); do [ -e "$ESTADO/fim.flag" ] && exit 0; sleep 1; done
  [ -e "$ESTADO/fim.flag" ] && exit 0
  echo 1 > "$ESTADO/limite.flag"
  if ! mata_grupo; then echo 1 > "$ESTADO/nao-morreu.flag"; kill -TERM "$PAI" 2>/dev/null; fi
) &
VIGIA=$!
wait "$SPID"
# setsid pode ter feito fork: espera o grupo inteiro acabar (o vigia o derruba no limite).
while grupo_vivo; do sleep 1 & wait $!; done
touch "$ESTADO/fim.flag"
wait "$VIGIA" 2>/dev/null
date -Is > "$ESTADO/comando-fim.txt"
[ -e "$ESTADO/nao-morreu.flag" ] && NAOMORREU=1
if [ -e "$ESTADO/limite.flag" ]; then LIMITE_ATINGIDO=1; exit 124; fi
[ -s "$ESTADO/rc-comando.txt" ] || exit 125
exit "$(cat "$ESTADO/rc-comando.txt")"
