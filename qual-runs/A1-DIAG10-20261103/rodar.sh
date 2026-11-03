#!/bin/bash
# A1-DIAG10: a janela do diagnóstico dos 10% (criterio.txt v6, seções 2 a 4). Uso: rodar.sh 1   (etapa 1: C, N@A0, I)   ou   rodar.sh 2   (etapa 2: N@A0, A1w, A1m).
# Cada etapa precisa do aval do Breno, dado à parte. Nada compila durante a medida; Python só nos E-cores (na análise, não aqui).
#
# Entradas que o Breno aprova antes (arquivos ao lado deste script):
#   timers.txt   uma unidade .timer de usuário por linha, os que serão parados durante a janela (e restaurados no fim);
#   teto.txt     opcional, um número inteiro: o teto X de temperatura do pacote em graus C; sem o arquivo, a temperatura é só registrada.
#
# Layout dos resultados:  $R/e<etapa>/rep<n>/<modelo>/<config>/{m2a-b4,a1}/{saida.txt,ms/,diag/}
#   configs da etapa 1: C (binários congelados do A1.4), N0 (N com --afin a0), I (a1-motor com --features diag, m2a-cache do N)
#   configs da etapa 2: N0, A1w (a1-motor e m2a-cache com --afin a1w), A1m (a1-motor com --afin a1m; o m2a-cache não tem equivalente e roda a0)
# Cada bloco: 60 s ociosa antes, temperatura do pacote no começo e no fim em carga.log; tentativa com falha refaz uma vez (a primeira saída fica como .falha1).
# MEDIDA_REPS, MEDIDA_MODS, MEDIDA_CONVS, MEDIDA_WARMUP, MEDIDA_OCIOSA (fumaça curta), MEDIDA_DIR, MEDIDA_SECO=1 (imprime os comandos e não executa), MEDIDA_SO_TIMERS=1, MEDIDA_SEGURA, MEDIDA_ESPERA_TIMERS, MEDIDA_PASSO_TIMERS, TIMERS_FILE, SYSTEMCTL e TEMP_FILE só servem ao teste deste roteiro.
set -euo pipefail
ETAPA=${1:?uso: rodar.sh 1|2}
case $ETAPA in 1|2) ;; *) echo "etapa 1 ou 2"; exit 1;; esac
cd /home/brenoperucchi/Devs/lastro-rust
Q=qual-runs/A1-DIAG10-20261103
R=${MEDIDA_DIR:-$Q}
SECO=${MEDIDA_SECO:-0}
SYSTEMCTL=${SYSTEMCTL:-systemctl}
UUID=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd
OCIOSA=${MEDIDA_OCIOSA:-60}
RUNS=$R/e$ETAPA
mkdir -p "$R"
exec 9> "$R/.trava-e$ETAPA"
flock -n 9 || { echo "outra instância do rodar.sh está com a trava"; exit 1; }
for f in "$RUNS" "$R"/ambiente-e$ETAPA.txt "$R"/ambiente-fim-e$ETAPA.txt "$R"/carga-e$ETAPA.log "$R"/gpu-monitor-e$ETAPA.log "$R"/suspensas-e$ETAPA.txt "$R"/timers-estado-e$ETAPA.txt; do
  [ -e "$f" ] && { echo "já existe $f: sessão anterior na pasta; nada foi alterado"; exit 1; }
done
BC=qual-runs/A1-MEDIDA-20261101/bin     # C: congelados do A1.4
BN=$Q/bin/N                             # N: sem instrumentação, com as opções de afinidade
BI=$Q/bin/I                             # I: a1-motor com --features diag
T=qual-runs/TRACOS-20261029
M=/home/brenoperucchi/models
PY=tools/.venv-ref/bin/python
P="taskset -c 0-15"
export CUDA_VISIBLE_DEVICES=$UUID
dir() { case $1 in 1b) echo granite-3.1-1b-a400m-instruct;; 3b) echo granite-3.1-3b-a800m-instruct;; esac; }
temp() { # temperatura do pacote em graus C com uma casa, ou n/d
  if [ -n "${TEMP_FILE:-}" ]; then cat "$TEMP_FILE"; return; fi
  for h in /sys/class/hwmon/hwmon*; do
    [ "$(cat $h/name 2>/dev/null)" = coretemp ] && { awk '{printf "%.1f", $1/1000}' "$h/temp1_input"; return; }
  done; echo n/d
}
gpu() { nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu --format=csv,noheader | grep "$UUID"; }
procs() { ps -eo pid,pcpu,comm --sort=-pcpu | awk 'NR == 1 || $2 > 5'; }

TETO=""
[ -s "$Q/teto.txt" ] && TETO=$(tr -dc 0-9 < "$Q/teto.txt")
TIMERS=()
TIMERS_FILE=${TIMERS_FILE:-$Q/timers.txt}
[ -s "$TIMERS_FILE" ] && mapfile -t TIMERS < <(grep -v '^[[:space:]]*\(#\|$\)' "$TIMERS_FILE")

# --- 0. pré-condições (fora do modo seco) ---
if [ "$SECO" = 0 ]; then
  g=$(gpu) || { echo "4090 não encontrada pelo UUID"; exit 1; }
  util=$(echo "$g" | cut -d, -f2 | tr -dc 0-9); mem=$(echo "$g" | cut -d, -f3 | tr -dc 0-9)
  { [ "$util" -le 2 ] && [ "$mem" -le 500 ]; } || { echo "4090 ocupada ($g); nada foi alterado"; exit 1; }
  (cd "$BC" && sha256sum -c SHA256SUMS 2>&1 | grep -E "^(a1-motor|m2a-cache):") > "$R/sha-binarios-e$ETAPA.txt" || { echo "sha dos binários C não confere"; exit 1; }
  (cd "$Q/bin" && sha256sum -c SHA256SUMS) >> "$R/sha-binarios-e$ETAPA.txt" 2>&1 || { echo "sha dos binários N e I não confere"; exit 1; }
fi

# --- 1. timers: captura, parada e restauração em trap (só o que estava ativo volta a ser ativado) ---
ESTADO=$R/timers-estado-e$ETAPA.txt
captura_timers() {
  for t in "${TIMERS[@]}"; do
    s=${t%.timer}.service
    echo "$t timer: is-active=$($SYSTEMCTL --user is-active "$t" 2>/dev/null || true) is-enabled=$($SYSTEMCTL --user is-enabled "$t" 2>/dev/null || true)"
    echo "$s service: is-active=$($SYSTEMCTL --user is-active "$s" 2>/dev/null || true) is-enabled=$($SYSTEMCTL --user is-enabled "$s" 2>/dev/null || true)"
  done
}
PARADOS=()
restaura_timers() {
  for t in "${PARADOS[@]}"; do
    $SYSTEMCTL --user start "$t" || echo "FALHA ao reativar $t" >&2
  done
  PARADOS=()
}
para_timers() {
  captura_timers > "$ESTADO"
  for t in "${TIMERS[@]}"; do
    if [ "$($SYSTEMCTL --user is-active "$t" 2>/dev/null || true)" = active ]; then
      PARADOS+=("$t")
      $SYSTEMCTL --user stop "$t"
    fi
  done
  # espera os serviços em execução terminarem (até 10 min)
  for k in $(seq 1 ${MEDIDA_ESPERA_TIMERS:-120}); do
    ocupado=0
    for t in "${TIMERS[@]}"; do
      case "$($SYSTEMCTL --user is-active "${t%.timer}.service" 2>/dev/null || true)" in active|activating) ocupado=1;; esac
    done
    [ $ocupado = 0 ] && break
    sleep ${MEDIDA_PASSO_TIMERS:-5}
  done
  [ "${ocupado:-0}" = 0 ] || { echo "serviço de timer ainda em execução depois de 10 min"; exit 1; }
}
if [ "${MEDIDA_SO_TIMERS:-0}" = 1 ]; then      # teste do próprio mecanismo: para, mostra o estado e restaura
  trap restaura_timers EXIT
  trap 'exit 130' INT TERM
  para_timers
  cat "$ESTADO"
  [ "${MEDIDA_SEGURA:-0}" = 1 ] && { echo PARADOS; while :; do sleep 0.2; done; }      # teste: fica parado até receber um sinal
  exit 0
fi

if [ "$SECO" = 0 ]; then
  PAI=$$
  ( while kill -0 $PAI 2>/dev/null; do
      if o=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>&1); then
        echo "$(date +%s.%N) ok $(echo "$o" | grep "$UUID" | tr '\n' ';')"
      else echo "$(date +%s.%N) ERRO-DE-CONSULTA $(echo "$o" | tr '\n' ' ')"; fi
      sleep 5
    done ) >> "$R/gpu-monitor-e$ETAPA.log" 9>&- &
  MON=$!
  FILHO=
  trap 'kill $FILHO $MON 2>/dev/null; restaura_timers' EXIT
  trap 'exit 130' INT TERM
  para_timers
  {
    date; head -1 /proc/driver/nvidia/version; uptime
    echo "git: $(git rev-parse HEAD) ($(git status --porcelain | grep -v 'qual-runs/A1-DIAG10-20261103/' | grep -v 'A1-MEDIDA-20261101/.trava' | wc -l) arquivos alterados fora desta pasta)"
    sha256sum "$Q/criterio.txt" "$Q/rodar.sh" "$Q/analise.py" "$Q/decide.py"
    echo "timers parados: ${PARADOS[*]:-nenhum}"; echo "timers monitorados (estado antes):"; cat "$ESTADO"
    echo "teto de temperatura: ${TETO:-sem teto (só registro)}"
    echo "GPU no começo: $g"; echo "temperatura do pacote: $(temp) C"
    echo "processos acima de 5% de CPU no começo:"; procs
  } > "$R/ambiente-e$ETAPA.txt"
fi

run() { if [ "$SECO" = 1 ]; then echo "$*"; else "$@"; fi; }
fim() { # nome rc
  echo "$(date +%T) $1 $2"
  [ "$SECO" = 1 ] || echo "$(date +%T) $1 $(cut -d' ' -f1-3 /proc/loadavg) temp:$(temp)C gpu:$(gpu | cut -d, -f2-6 | tr -d ' ')" >> "$R/carga-e$ETAPA.log"
}
rot_ok() { # arquivo: roteamento das linhas do m2a-cache (no máximo 0,01% dos pares fora do traço)
  awk '/^(resident|lfu_cpu)/ { for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+\/[0-9]+$/) { split($i, a, "/"); if ((a[2] - a[1]) * 10000 > a[2]) ruim = 1; achou = 1 } }
       END { exit (ruim || !achou) }' "$1"
}
espera_gpu() {
  local k g util mem
  for k in $(seq 1 90); do
    g=$(gpu) || return 1
    util=$(echo "$g" | cut -d, -f2 | tr -dc 0-9); mem=$(echo "$g" | cut -d, -f3 | tr -dc 0-9)
    { [ "$util" -le 2 ] && [ "$mem" -le 500 ]; } && return 0
    echo "$(date +%T) GPU ocupada ($g), esperando" | tee -a "$R/carga-e$ETAPA.log"; sleep 20
  done
  return 1
}
ociosa() { # 60 s ociosa e, com teto, espera o pacote esfriar até ele (até 30 min)
  [ "$SECO" = 1 ] && return 0
  sleep "$OCIOSA"
  if [ -n "$TETO" ]; then
    for k in $(seq 1 90); do
      t=$(temp); [ "$t" = n/d ] && break
      awk -v t="$t" -v x="$TETO" 'BEGIN { exit !(t <= x) }' && break
      sleep 20
    done
  fi
}
bloco() { # nome checa_rot saída comando...
  local nome=$1 checa=$2 out=$3 tentativa rc a t0 pid; shift 3
  for tentativa in 1 2; do
    rc=0
    if [ "$SECO" = 1 ]; then echo "$* > $out"; fim "$nome" 0; return 0; fi
    espera_gpu || { echo "$(date +%T) GPU não liberou antes de $nome: janela interrompida" | tee -a "$R/carga-e$ETAPA.log"; exit 1; }
    ociosa
    echo "$(date +%T) INICIO $nome temp:$(temp)C" >> "$R/carga-e$ETAPA.log"
    t0=$(date +%s.%N)
    "$@" > "$out" 2>&1 9>&- &
    pid=$!; FILHO=$pid
    taskset -c 16-27 "$PY" -I -B "$Q/amostrador.py" $pid "${out%/saida.txt}/freq.txt" 9>&- &
    wait $pid || rc=$?; FILHO=
    rcb=$rc
    if [ $rc = 0 ] && [ "$checa" = rot ] && ! rot_ok "$out"; then rc=97; fi
    echo "TENTATIVA $nome n=$tentativa pid=$pid inicio=$t0 fim=$(date +%s.%N) rc_harness=$rcb rc=$rc" >> "$R/carga-e$ETAPA.log"
    fim "$nome" $rc
    [ $rc = 0 ] && return 0
    mv "$out" "$out.falha$tentativa"
    for a in ${ARQ:-}; do [ -e "$a" ] && mv "$a" "$a.falha$tentativa"; done
  done
  echo "SUSPENSA $nome (rc $rc, duas tentativas; as saídas ficam como .falha1 e .falha2)" | tee -a "$R/suspensas-e$ETAPA.txt"
}

# Um bloco de uma configuração: cfg modelo repetição bloco(m2a-b4|a1)
budgets() { # modelo: etapa 1 tem o 1B a 25% descritivo; o resto roda só 10%
  if [ "$ETAPA" = 1 ] && [ "$1" = 1b ]; then echo 0.1,0.25; else echo 0.1; fi
}
um_bloco() {
  local cfg=$1 mod=$2 n=$3 bl=$4 d bin afin extra="" pols
  d=$RUNS/rep$n/$mod/$cfg/$bl
  run mkdir -p "$d"
  local com="--model $M/$(dir $mod) --traces $T/$(dir $mod) --ref qual-runs/REF-granite-$mod-texto-a --budgets $(budgets $mod) --backlog 4 --warmup ${MEDIDA_WARMUP:-0,1,2,3} --threads 8"
  [ -n "${MEDIDA_CONVS:-}" ] && com="$com --convs $MEDIDA_CONVS"
  if [ $bl = m2a-b4 ]; then
    case $cfg in C) bin=$BC/m2a-cache; afin="";; N0|I|A1m) bin=$BN/m2a-cache; afin="--afin a0";; A1w) bin=$BN/m2a-cache; afin="--afin a1w";; esac
    ARQ="$d/ms" bloco "$d" rot "$d/saida.txt" $P $bin $com --policies resident,lfu_cpu $afin --metrics /dev/null --ms-dir "$d/ms"
  else
    case $cfg in
      C) bin=$BC/a1-motor; afin="";;
      N0) bin=$BN/a1-motor; afin="--afin a0";;
      I) bin=$BI/a1-motor; afin="--afin a0"; extra="--diag-dir $d/diag";;
      A1w) bin=$BN/a1-motor; afin="--afin a1w";;
      A1m) bin=$BN/a1-motor; afin="--afin a1m";;
    esac
    ARQ="$d/ms $d/diag" bloco "$d" norot "$d/saida.txt" $P $bin $com --policies resident,motor_cheio,motor_evento $afin $extra --ms-dir "$d/ms"
  fi
}

if [ $ETAPA = 1 ]; then CFGS=(C N0 I); else CFGS=(N0 A1w A1m); fi
for n in $(seq 1 ${MEDIDA_REPS:-5}); do
  if [ $((n % 2)) = 1 ]; then mods="1b 3b"; else mods="3b 1b"; fi
  [ -n "${MEDIDA_MODS:-}" ] && mods=$MEDIDA_MODS
  for mod in $mods; do
    for i in 0 1 2; do
      cfg=${CFGS[$(((i + n - 1) % 3))]}
      if [ $((n % 2)) = 1 ]; then bls="m2a-b4 a1"; else bls="a1 m2a-b4"; fi    # a ordem dos dois blocos também alterna
      for tent in 1 2; do
        for bl in $bls; do
          um_bloco $cfg $mod $n $bl
        done
        [ "$SECO" = 1 ] && break
        # 0c (criterio.txt): chamador e trabalhadores a menos de 2% entre as duas linhas da mesma configuração; senão repete uma vez
        d1=$RUNS/rep$n/$mod/$cfg/m2a-b4; d2=$RUNS/rep$n/$mod/$cfg/a1
        if r0c=$(taskset -c 16-27 "$PY" -I -B "$Q/amostrador.py" --compara "$d1/freq.txt" "$d2/freq.txt"); then break; fi
        echo "$(date +%T) 0c $cfg $mod rep$n tentativa $tent: $r0c" | tee -a "$R/carga-e$ETAPA.log"
        if [ $tent = 1 ]; then
          for d in $d1 $d2; do mv "$d" "$d.freq1"; done
        else
          echo "SUSPENSA 0c $cfg $mod rep$n ($r0c)" | tee -a "$R/suspensas-e$ETAPA.txt"
        fi
      done
    done
  done
done

if [ "$SECO" = 0 ]; then
  {
    date; uptime; echo "GPU no fim: $(gpu)"; echo "temperatura do pacote: $(temp) C"
    cat "$R/sha-binarios-e$ETAPA.txt"
    (cd "$BC" && sha256sum -c SHA256SUMS 2>&1 | grep -E "^(a1-motor|m2a-cache):") || echo "INVALIDA A JANELA: sha dos binários C mudou"
    (cd "$Q/bin" && sha256sum -c SHA256SUMS) 2>&1 || echo "INVALIDA A JANELA: sha dos binários N ou I mudou"
    echo "processos acima de 5% de CPU no fim:"; procs
    [ -e "$R/suspensas-e$ETAPA.txt" ] && { echo "blocos suspensos:"; cat "$R/suspensas-e$ETAPA.txt"; } || echo "nenhum bloco suspenso"
  } > "$R/ambiente-fim-e$ETAPA.txt"
fi
echo "$(date +%T) fim da etapa $ETAPA"
