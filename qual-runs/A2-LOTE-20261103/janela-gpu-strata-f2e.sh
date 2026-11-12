#!/usr/bin/env bash
# Roda UM comando com a GPU liberada do Strata, no computador da 5090. Rode como o usuario normal (sem root).
# Uso:   janela-gpu-strata.sh [--dry-run] [--teto SEG] -- <comando> [args...]
# Ordem: confere que e o usuario certo e que o Strata esta ativo; ARMA a obrigacao de religar e so entao para o Strata e CONFIRMA
#        (inactive, GPU livre); roda o comando; no EXIT (sucesso, falha ou sinal) ENCERRA o comando e qualquer unidade transitoria
#        lastro-limite-* (confirmando cgroup vazio) e SO ENTAO religa o Strata (se ele estava ativo e foi parado aqui).
# Nunca religa por cima de um estado inicial desconhecido. Log em ~/lastro-janelas, nunca sobrescrito.
# Codigos: o do comando (ou 124 teto); 2 recusa de uso/ambiente; 3 falha ao parar/GPU nao livre; 130 sinal;
#          5 ficou processo/unidade lastro-limite-* depois do cancelamento: o Strata NAO foi religado (decisao humana);
#          6 o Strata nao voltou a active depois do start (religar a mao). 5 e 6 so substituem o codigo 0; um codigo != 0 do comando e preservado
#          e a falha de religacao fica no log.
# Variaveis JGS_* so para teste (servicos simulados); o uso normal nao define nenhuma.
set -u
UNIDADE=strata.service
SYSCTL=${JGS_SYSTEMCTL:-systemctl}
SMI=${JGS_NVIDIA_SMI:-nvidia-smi}
CG_RAIZ=${JGS_CGROUP_RAIZ:-/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/app.slice}
ESPERA_START=${JGS_ESPERA_START:-60}      # s ate o Strata confirmar active depois do start
ESPERA_GPU=${JGS_ESPERA_GPU:-60}          # s esperando a GPU esvaziar antes do start
ESPERA_UNID=${JGS_ESPERA_UNIDADE:-20}     # s esperando as unidades lastro-limite-* esvaziarem depois do stop delas
ESPERA_CMD=${JGS_ESPERA_COMANDO:-45}      # s entre TERM e KILL no comando
PASSO=${JGS_PASSO:-1}
LOGDIR="$HOME/lastro-janelas"
TETO=1200
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --teto) TETO="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "argumento desconhecido: $1" >&2; exit 2 ;;
  esac
done
[ $# -gt 0 ] || { echo "faltou o comando depois de --" >&2; exit 2; }
[ "$(id -u)" -ne 0 ] || { echo "recuso rodar como root" >&2; exit 2; }
mkdir -p "$LOGDIR" || exit 2
LOG="$LOGDIR/janela-$(date +%Y%m%d-%H%M%S).log"
set -o noclobber
: > "$LOG" || { echo "log ja existe: $LOG" >&2; exit 2; }
set +o noclobber
say() { echo "$(date +%T) $*" | tee -a "$LOG"; }

INICIAL=$("$SYSCTL" --user is-active "$UNIDADE" 2>/dev/null)
say "estado inicial do Strata: ${INICIAL:-ilegivel}"
case "$INICIAL" in active|inactive) ;; *) say "estado inicial ilegivel ou incomum, recuso"; exit 2 ;; esac

PAROU=0      # 1 = a obrigacao de religar esta armada (armada ANTES de pedir o stop)
CPID=        # pid do timeout que roda o comando, enquanto vivo
FINAL=0

# unidades transitorias do Lastro (roda-com-limite*.sh) ainda vivas ou com processo no cgroup
unidades_lastro() { "$SYSCTL" --user list-units --all --plain --no-legend 'lastro-limite-*.service' 2>/dev/null | grep -o 'lastro-limite-[^ ]*\.service' | sort -u; }
unidades_vivas() {
  local u st p
  for u in $(unidades_lastro); do
    st=$("$SYSCTL" --user is-active "$u" 2>/dev/null)
    p=$(cat "$CG_RAIZ/$u/cgroup.procs" 2>/dev/null)
    case "$st" in active|activating|deactivating|reloading) echo "$u"; continue ;; esac
    [ -z "$p" ] || echo "$u"
  done
}
# encerra o comando e as unidades; devolve 0 so se nada sobrou
encerra_tudo() {
  local t=0 v
  if [ -n "$CPID" ] && kill -0 "$CPID" 2>/dev/null; then
    say "encerrando o comando (TERM no grupo $CPID)"
    kill -TERM "$CPID" 2>/dev/null
    while kill -0 "$CPID" 2>/dev/null && [ "$t" -lt "$ESPERA_CMD" ]; do sleep "$PASSO"; t=$((t + 1)); done
    if kill -0 "$CPID" 2>/dev/null; then say "comando nao encerrou em ${ESPERA_CMD}s: KILL no grupo"; kill -KILL -- "-$CPID" 2>/dev/null; kill -KILL "$CPID" 2>/dev/null; sleep "$PASSO"; fi
    wait "$CPID" 2>/dev/null
  fi
  CPID=
  for v in $(unidades_vivas); do
    say "parando unidade transitoria $v"
    timeout 40 "$SYSCTL" --user stop "$v" > /dev/null 2>&1
  done
  t=0
  while [ -n "$(unidades_vivas)" ] && [ "$t" -lt "$ESPERA_UNID" ]; do sleep "$PASSO"; t=$((t + 1)); done
  v=$(unidades_vivas)
  if [ -n "$v" ]; then
    say "ATENCAO: ficou unidade/processo do Lastro depois do cancelamento: $(echo $v)"
    return 1
  fi
  return 0
}
gpu_vazia() {
  local n m
  n=$("$SMI" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  m=$("$SMI" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [ "$n" -eq 0 ] && [ "${m:-99999}" -lt 1500 ]
}
religa() {
  FINAL=$?
  trap - EXIT
  trap '' INT TERM HUP     # nenhum sinal interrompe a restauracao
  local limpo=1 t=0 ok=0 tent
  encerra_tudo || limpo=0
  if [ "$PAROU" -eq 1 ]; then
    if [ "$limpo" -eq 0 ]; then
      say "Strata NAO religado: ha processo/unidade do Lastro ainda ativo (pode estar usando a GPU). Resolver a mao e religar com: systemctl --user start $UNIDADE"
      [ "$FINAL" -ne 0 ] || FINAL=5
    else
      while ! gpu_vazia && [ "$t" -lt "$ESPERA_GPU" ]; do sleep "$PASSO"; t=$((t + 1)); done
      gpu_vazia && say "GPU vazia antes de religar" || say "ATENCAO: GPU ainda com processo/memoria depois de ${ESPERA_GPU}s; religo mesmo assim (producao parada e pior)"
      for tent in 1 2 3; do
        "$SYSCTL" --user start "$UNIDADE" 2>&1 | tee -a "$LOG" > /dev/null
        [ "${PIPESTATUS[0]}" -eq 0 ] || say "start do Strata falhou (tentativa $tent)"
        t=0
        while [ "$t" -lt "$ESPERA_START" ]; do
          [ "$("$SYSCTL" --user is-active "$UNIDADE" 2>/dev/null)" = "active" ] && { ok=1; break; }
          sleep "$PASSO"; t=$((t + 1))
        done
        [ "$ok" -eq 1 ] && break
      done
      if [ "$ok" -eq 1 ]; then say "Strata religado: active"
      else
        say "ATENCAO: Strata NAO voltou a active, religar a mao: systemctl --user start $UNIDADE"
        [ "$FINAL" -ne 0 ] || FINAL=6
      fi
    fi
  elif [ "$limpo" -eq 0 ]; then
    [ "$FINAL" -ne 0 ] || FINAL=5
  fi
  say "envelope termina rc=$FINAL"
  exit "$FINAL"
}
trap religa EXIT
trap 'exit 130' INT TERM HUP

if [ "$DRY" -eq 1 ]; then
  say "dry-run: nao paro nada. Pararia $UNIDADE, esperaria a GPU liberar, rodaria: $* (teto ${TETO}s) e religaria."
  trap - EXIT; exit 0
fi

if [ "$INICIAL" = "active" ]; then
  PAROU=1    # obrigacao de religar armada ANTES do stop: um sinal durante ele nao deixa a producao parada
  say "parando $UNIDADE (pode levar ate ~5 min, o Strata fecha com calma)"
  "$SYSCTL" --user stop "$UNIDADE" || { say "falha ao parar, abortando"; exit 3; }
  [ "$("$SYSCTL" --user is-active "$UNIDADE")" = "inactive" ] || { say "Strata nao ficou inactive, abortando"; exit 3; }
fi

# GPU livre: nenhum processo de compute e memoria usada baixa (limite 1500 MiB)
for i in $(seq 1 30); do
  PROCS=$("$SMI" --query-compute-apps=pid --format=csv,noheader | wc -l)
  MEM=$("$SMI" --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$PROCS" -eq 0 ] && [ "${MEM:-99999}" -lt 1500 ] && break
  sleep "$PASSO" ; sleep "$PASSO"
done
say "GPU: processos=$PROCS memoria usada=${MEM} MiB"
[ "$PROCS" -eq 0 ] && [ "${MEM:-99999}" -lt 1500 ] || { say "GPU nao ficou livre, abortando (o Strata sera religado)"; exit 3; }

say "rodando: $* (teto ${TETO}s)"
timeout -k 30 "$TETO" "$@" >> "$LOG" 2>&1 &
CPID=$!
wait "$CPID"
RC=$?
CPID=
say "comando terminou rc=$RC"
exit "$RC"
