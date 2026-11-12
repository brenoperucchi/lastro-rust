#!/usr/bin/env bash
# Roda UM comando com a GPU liberada do Strata, no computador da 5090. Rode como o usuario normal (sem root).
# Uso:   janela-gpu-strata.sh [--dry-run] [--teto SEG] -- <comando> [args...]
# Ordem: confere que e o usuario certo e que o Strata esta ativo; ARMA a obrigacao de religar e so entao para o Strata e CONFIRMA
#        (inactive, GPU livre); roda o comando; no EXIT (sucesso, falha ou sinal) ENCERRA o comando e qualquer unidade transitoria
#        lastro-limite-* (confirmando cgroup vazio) e SO ENTAO religa o Strata (se ele estava ativo e foi parado aqui).
# Nunca religa por cima de um estado inicial desconhecido. Log em ~/lastro-janelas, nunca sobrescrito.
# Codigos: o do comando (ou 124 teto); 2 recusa de uso/ambiente; 3 falha ao parar/GPU nao livre; 130 sinal;
#          5 ficou processo/unidade lastro-limite-* depois do cancelamento, o termino nao pode ser confirmado (consulta falhou/ilegivel) ou a GPU nao
#            foi confirmada vazia: o Strata NAO foi religado, ha aviso em log/stdout/stderr e ATENCAO-STRATA-PARADO-*.txt em ~/lastro-janelas (decisao humana);
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
ESPERA_UNID=${JGS_ESPERA_UNIDADE:-60}     # s esperando as unidades lastro-limite-* esvaziarem depois do stop delas
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

# Confirmacao de termino FALHA FECHADA: toda consulta tem saida e codigo conferidos; erro, resposta ilegivel ou cgroup ilegivel = INCERTO, e incerteza nunca vira "nada vivo".
VIVAS=""     # unidades lastro-limite-* ainda vivas (ou com processo no cgroup) na ultima verificacao
INCERTO=0    # 1 = a ultima verificacao nao conseguiu provar nada (consulta falhou ou ilegivel)
DIAG=""      # motivo do ultimo INCERTO
verifica_unidades() {
  local out rc u st p lista
  VIVAS=""; INCERTO=0; DIAG=""
  out=$("$SYSCTL" --user list-units --all --plain --no-legend 'lastro-limite-*.service' 2>/dev/null); rc=$?
  if [ "$rc" -ne 0 ]; then INCERTO=1; DIAG="list-units falhou (rc $rc)"; return 0; fi
  lista=$(printf '%s\n' "$out" | grep -o 'lastro-limite-[^ ]*\.service' | sort -u)
  for u in $lista; do
    st=$("$SYSCTL" --user is-active "$u" 2>/dev/null)
    case "$st" in
      active|activating|deactivating|reloading) VIVAS="$VIVAS $u"; continue ;;
      inactive|failed) ;;
      *) INCERTO=1; DIAG="is-active de $u ilegivel ('$st')"; continue ;;
    esac
    if [ -d "$CG_RAIZ/$u" ]; then
      if [ ! -r "$CG_RAIZ/$u/cgroup.procs" ]; then INCERTO=1; DIAG="cgroup de $u existe mas cgroup.procs nao e legivel"; continue; fi
      p=$(cat "$CG_RAIZ/$u/cgroup.procs" 2>/dev/null) || { INCERTO=1; DIAG="leitura do cgroup de $u falhou"; continue; }
      [ -z "$p" ] || VIVAS="$VIVAS $u"
    fi
  done
  return 0
}
# alerta bem visivel (stdout, stderr, log e arquivo de estado) de que o Strata ficou parado; a decisao de religar e humana
alerta() {
  local f="$LOGDIR/ATENCAO-STRATA-PARADO-$(date +%Y%m%d-%H%M%S).txt"
  { echo "*** ATENCAO: strata.service de PRODUCAO PARADO e NAO religado automaticamente ***"
    echo "motivo: $*"
    echo "log da janela: $LOG"
    echo "depois de conferir a GPU e as unidades lastro-limite-* (systemctl --user list-units 'lastro-limite-*'; nvidia-smi), religar a mao: systemctl --user start $UNIDADE"
  } | tee -a "$LOG" "$f" >&2
  say "arquivo de estado: $f"
}
# encerra o comando e as unidades; devolve 0 so se a ULTIMA verificacao foi bem-sucedida e nao achou nada vivo
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
  verifica_unidades
  for v in $VIVAS; do
    say "parando unidade transitoria $v"
    timeout 40 "$SYSCTL" --user stop "$v" > /dev/null 2>&1
  done
  t=0
  while :; do
    verifica_unidades
    [ "$INCERTO" -eq 0 ] && [ -z "$VIVAS" ] && break
    [ "$t" -lt "$ESPERA_UNID" ] || break
    sleep "$PASSO"; t=$((t + 1))
  done
  if [ "$INCERTO" -ne 0 ]; then say "ATENCAO: nao foi possivel confirmar o termino das unidades do Lastro: $DIAG"; return 1; fi
  if [ -n "$VIVAS" ]; then say "ATENCAO: ficou unidade/processo do Lastro depois do cancelamento:$VIVAS"; return 1; fi
  return 0
}
# 0 = GPU vazia confirmada; 1 = ocupada; 2 = nao deu para ler (nvidia-smi falhou ou resposta ilegivel)
gpu_estado() {
  local n m rn rmem
  n=$("$SMI" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); rn=$?
  m=$("$SMI" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null); rmem=$?
  m=${m%%$'\n'*}
  [ "$rn" -eq 0 ] && [ "$rmem" -eq 0 ] || return 2
  case "$m" in ''|*[!0-9]*) return 2 ;; esac
  [ -z "$n" ] && [ "$m" -lt 1500 ]
}
religa() {
  FINAL=$?
  trap - EXIT
  trap '' INT TERM HUP     # nenhum sinal interrompe a restauracao
  local limpo=1 t=0 ok=0 tent g
  encerra_tudo || limpo=0
  if [ "$PAROU" -eq 1 ]; then
    if [ "$limpo" -eq 0 ]; then
      alerta "processo/unidade do Lastro ainda ativo ou termino nao confirmado (pode estar usando a GPU): ${DIAG:-$VIVAS}"
      [ "$FINAL" -ne 0 ] || FINAL=5
    else
      gpu_estado; g=$?
      while [ "$g" -ne 0 ] && [ "$t" -lt "$ESPERA_GPU" ]; do sleep "$PASSO"; t=$((t + 1)); gpu_estado; g=$?; done
      if [ "$g" -ne 0 ]; then
        # decisao documentada: GPU nao confirmada vazia (ocupada ou ilegivel) NAO segue para o start; producao parada com aviso, decisao humana
        alerta "GPU nao confirmada vazia depois de ${ESPERA_GPU}s (estado $g: 1 ocupada, 2 ilegivel); nao religo por cima"
        [ "$FINAL" -ne 0 ] || FINAL=5
      else
        say "GPU vazia confirmada antes de religar"
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
          alerta "o Strata nao voltou a active depois do start (3 tentativas)"
          [ "$FINAL" -ne 0 ] || FINAL=6
        fi
      fi
    fi
  elif [ "$limpo" -eq 0 ]; then
    say "ATENCAO: Strata nao foi parado por este envelope, mas ha unidade do Lastro nao confirmada encerrada"
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
