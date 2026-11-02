#!/bin/bash
# A1.4c: a janela de medida do motor D1 contra o lfu_cpu de hoje e o llama.cpp. Critério em criterio.txt (gravado antes do
# harness; emendas datadas no fim). Binários em bin/ (sha conferido antes); nada compila durante a medida; Python só nos E-cores.
#
# O que o roteiro faz, na ordem:
#   0. recusa rodar com outra instância na trava, com resultados de uma sessão anterior na pasta, com a 4090 ocupada ou fora do
#      UUID; confere o sha dos binários; grava ambiente.txt (data, driver, git, sha do critério, GPU, processos acima de 5%);
#   1. 5 repetições intercaladas, 1B e 3B (a ordem dos modelos e dos três blocos gira a cada repetição). Cada bloco roda com
#      `taskset -c 0-15`, aquecimento nas conversas 0 a 3 e medida nas 4 a 7, e grava o log, o ms de cada token (--ms-dir) e a carga:
#        m2a-b4   m2a-cache  resident,lfu_cpu  backlog 4   (L2, L3 e a base dos ajustes)
#        a1       a1-motor   resident,motor_cheio,motor_evento  (L1, L4)
#        m2a-b0   m2a-cache  resident,lfu_cpu  backlog 0   (cópia desligada, para c_cpu_quiet; pareado com m2a-b4)
#   2. L6: motor_barreira, 1 execução por modelo;
#   3. L5: llama.cpp (forcado) em 3 repetições intercaladas entre si: tudo na GPU F32, cache 10/25/50% e -ncmoe 22/18/12 (1B) e
#      29/24/16 (3B), com o argmax gravado na 1ª repetição para o portão de correção;
#   4. ambiente-fim.txt (GPU, processos, carga).
# Falha de um bloco (código diferente de zero, ou roteamento fora da regra do critério nas linhas do m2a-cache): refaz uma vez
# (a primeira saída fica como .falha1) e, se repetir, registra a CÉLULA em suspensas.txt e segue (critério, seções 2 e 8).
# Interromper com Ctrl+C deixa o que já foi gravado; para recomeçar é preciso mover a pasta de resultados (o roteiro nunca apaga).
# MEDIDA_DIR e MEDIDA_SECO só servem ao teste do próprio roteiro (cópia da pasta; SECO=1 imprime os comandos e não executa).
set -euo pipefail
cd /home/brenoperucchi/Devs/lastro-rust
R=${MEDIDA_DIR:-qual-runs/A1-MEDIDA-20261101}
SECO=${MEDIDA_SECO:-0}
UUID=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd
mkdir -p "$R"
exec 9> "$R/.trava"
flock -n 9 || { echo "outra instância do rodar.sh está com a trava"; exit 1; }
for f in "$R"/ambiente.txt "$R"/ambiente-fim.txt "$R"/carga.log "$R"/sha-binarios.txt "$R"/suspensas.txt "$R"/runs "$R"/llama "$R"/gpu-monitor.log "$R"/libs.sha256; do
  [ -e "$f" ] && { echo "já existe $f: sessão anterior na pasta; nada foi alterado"; exit 1; }
done
B=${MEDIDA_BIN:-$R/bin}
T=qual-runs/TRACOS-20261029
M=/home/brenoperucchi/models
G=$M/gguf
TOK=qual-runs/LLAMACPP-COMPARA-20261030/tokens
P="taskset -c 0-15"
export CUDA_VISIBLE_DEVICES=$UUID
dir() { case $1 in 1b) echo granite-3.1-1b-a400m-instruct;; 3b) echo granite-3.1-3b-a800m-instruct;; esac; }
procs() { ps -eo pid,pcpu,comm --sort=-pcpu | awk 'NR == 1 || $2 > 5'; }
gpu() { nvidia-smi --query-gpu=uuid,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu --format=csv,noheader | grep "$UUID"; }
if [ "$SECO" = 0 ]; then
  g=$(gpu) || { echo "4090 não encontrada pelo UUID"; exit 1; }
  util=$(echo "$g" | cut -d, -f2 | tr -dc 0-9); mem=$(echo "$g" | cut -d, -f3 | tr -dc 0-9)
  { [ "$util" -le 2 ] && [ "$mem" -le 500 ]; } || { echo "4090 ocupada ($g); nada foi alterado"; exit 1; }
  (cd "$B" && sha256sum -c SHA256SUMS) > "$R/sha-binarios.txt" 2>&1 || { echo "sha dos binários não confere"; exit 1; }
  {
    date; head -1 /proc/driver/nvidia/version; ls -l /usr/lib/libcuda.so.1; uptime
    echo "git: $(git rev-parse HEAD) ($(git status --porcelain | grep -v 'qual-runs/A1-MEDIDA-20261101/' | wc -l) arquivos alterados fora desta pasta)"
    echo "fonte dos binários:"; cat "$B/fonte.txt"
    sha256sum "$R/criterio.txt" "$R/criterio-v1.txt" "$R/rodar.sh" tools/moe_sim.py
    echo "llama.cpp: $(git -C /home/brenoperucchi/Devs/GitHub/llama.cpp rev-parse HEAD)"
    sha256sum $G/granite-3.1-1b-a400m-instruct-bf16.gguf $G/granite-3.1-3b-a800m-instruct-bf16.gguf
    echo "bibliotecas do forcado (sha) em libs.sha256"
    echo "GPU no começo: $g"
    echo "processos acima de 5% de CPU no começo:"; procs
  } > "$R/ambiente.txt"
  ldd "$B/forcado" | awk '/=> \// {print $3}' | grep -E "llama|ggml" | xargs sha256sum > "$R/libs.sha256" || { echo "ldd/sha das bibliotecas falhou"; exit 1; }
  [ "$(wc -l < "$R/libs.sha256")" -ge 4 ] || { echo "poucas bibliotecas em libs.sha256"; exit 1; }
fi

if [ "$SECO" = 0 ]; then
  PAI=$$
  ( while kill -0 $PAI 2>/dev/null; do
      if o=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>&1); then
        echo "$(date +%s.%N) ok $(echo "$o" | grep "$UUID" | tr '\n' ';')"
      else echo "$(date +%s.%N) ERRO-DE-CONSULTA $(echo "$o" | tr '\n' ' ')"; fi
      sleep 5
    done ) >> "$R/gpu-monitor.log" 9>&- &
  MON=$!
  FILHO=
  trap 'kill $FILHO $MON 2>/dev/null' EXIT
  trap 'exit 130' INT TERM
fi
run() { # comando...; no modo SECO só imprime
  if [ "$SECO" = 1 ]; then echo "$*"; return 0; fi
  "$@"
}
# Roteamento das linhas do m2a-cache: no máximo 0,01% dos pares (token, camada) fora do traço (critério, seção 2).
rot_ok() { # arquivo
  awk '/^(resident|lfu_cpu|lru_cpu)/ { for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+\/[0-9]+$/) { split($i, a, "/"); if ((a[2] - a[1]) * 10000 > a[2]) ruim = 1; achou = 1 } }
       END { exit (ruim || !achou) }' "$1"
}
fim() { # nome rc
  echo "$(date +%T) $1 $2"
  [ "$SECO" = 1 ] || echo "$(date +%T) $1 $(cut -d' ' -f1-3 /proc/loadavg) gpu:$(gpu | cut -d, -f2-6 | tr -d ' ')" >> "$R/carga.log"
}
# Espera a 4090 ficar livre antes de cada bloco (até 30 min); carga alheia no meio do bloco fica no gpu-monitor.log (amostra a
# cada 5 s dos processos de computação) e a execução afetada é descartada na análise (critério, seção 8).
espera_gpu() {
  local k g util mem
  for k in $(seq 1 90); do
    g=$(gpu) || return 1
    util=$(echo "$g" | cut -d, -f2 | tr -dc 0-9); mem=$(echo "$g" | cut -d, -f3 | tr -dc 0-9)
    { [ "$util" -le 2 ] && [ "$mem" -le 500 ]; } && return 0
    echo "$(date +%T) GPU ocupada ($g), esperando" | tee -a "$R/carga.log"; sleep 20
  done
  return 1
}
# ARQ: arquivos ou pastas que o bloco grava além da saída (ms/, argmax); cada tentativa os arquiva inteiros em .falhaN.
# ERRSEP=1: o stderr vai para "<saída>.log" (como na comparação de 30/10); senão vai junto da saída.
bloco() { # nome checa_rot saída comando...   (refaz uma vez; depois registra a suspensão)
  local nome=$1 checa=$2 out=$3 tentativa rc a t0 pid; shift 3
  for tentativa in 1 2; do
    rc=0
    if [ "$SECO" = 1 ]; then echo "$* > $out"; fim "$nome" 0; return 0; fi
    espera_gpu || { echo "$(date +%T) GPU não liberou antes de $nome: janela interrompida, nada foi executado neste bloco" | tee -a "$R/carga.log"; exit 1; }
    t0=$(date +%s.%N)
    if [ "${ERRSEP:-0}" = 1 ]; then "$@" > "$out" 2> "$out.log" 9>&- & else "$@" > "$out" 2>&1 9>&- & fi
    pid=$!; FILHO=$pid; wait $pid || rc=$?; FILHO=
    rcb=$rc
    if [ $rc = 0 ] && [ "$checa" = rot ] && ! rot_ok "$out"; then rc=97; fi
    echo "TENTATIVA $nome n=$tentativa pid=$pid inicio=$t0 fim=$(date +%s.%N) rc_harness=$rcb rc=$rc" >> "$R/carga.log"
    fim "$nome" $rc
    [ $rc = 0 ] && return 0
    mv "$out" "$out.falha$tentativa"
    [ -e "$out.log" ] && mv "$out.log" "$out.log.falha$tentativa"
    for a in ${ARQ:-}; do [ -e "$a" ] && mv "$a" "$a.falha$tentativa"; done
  done
  echo "SUSPENSA $nome (rc $rc, duas tentativas; células afetadas: todas as linhas e orçamentos do bloco; as duas saídas ficam como .falha1 e .falha2 e a análise decide por célula)" | tee -a "$R/suspensas.txt"
}
m2a() { # saída-dir modelo backlog
  local d=$1 mod=$2 b=$3
  mkdir -p "$d"
  ARQ="$d/ms" bloco "$d" rot "$d/saida.txt" $P $B/m2a-cache --model $M/$(dir $mod) --traces $T/$(dir $mod) --ref qual-runs/REF-granite-$mod-texto-a \
    --budgets 0.1,0.25,0.5 --policies resident,lfu_cpu --backlog $b --warmup 0,1,2,3 --threads 8 --metrics /dev/null --ms-dir "$d/ms"
}
a1() { # saída-dir modelo políticas
  local d=$1 mod=$2 pols=$3
  mkdir -p "$d"
  ARQ="$d/ms" bloco "$d" norot "$d/saida.txt" $P $B/a1-motor --model $M/$(dir $mod) --traces $T/$(dir $mod) --ref qual-runs/REF-granite-$mod-texto-a \
    --budgets 0.1,0.25,0.5 --policies $pols --backlog 4 --warmup 0,1,2,3 --threads 8 --ms-dir "$d/ms"
}

BLOCOS=(m2a-b4 a1 m2a-b0)
for n in 1 2 3 4 5; do
  if [ $((n % 2)) = 1 ]; then mods="1b 3b"; else mods="3b 1b"; fi
  for mod in $mods; do
    for i in 0 1 2; do
      bl=${BLOCOS[$(((i + n - 1) % 3))]}
      d=$R/runs/rep$n/$mod/$bl
      case $bl in
        m2a-b4) m2a $d $mod 4;;
        m2a-b0) m2a $d $mod 0;;
        a1) a1 $d $mod resident,motor_cheio,motor_evento;;
      esac
    done
  done
done
# L6: o motor em TokenBarrier, uma execução por modelo, só nos contadores.
for mod in 1b 3b; do
  a1 $R/runs/barreira/$mod/a1 $mod resident,motor_barreira
done

# L5: llama.cpp, 3 repetições intercaladas entre si (forcado; mesmas flags da comparação de 30/10).
COMUM="-ngl 99 -c 2048 --fit off --no-warmup -t 8"
F32="-ctk f32 -ctv f32 -fa off"
mib() { case $1 in 1b-10) echo 231;; 1b-25) echo 579;; 1b-50) echo 1155;; 3b-10) echo 581;; 3b-25) echo 1445;; 3b-50) echo 2885;; esac; }
ncm() { case $1 in 1b-10) echo 22;; 1b-25) echo 18;; 1b-50) echo 12;; 3b-10) echo 29;; 3b-25) echo 24;; 3b-50) echo 16;; esac; }
flags() { case $2 in
  gpu32) echo "$F32";;
  c10|c25|c50) echo "$F32 --cpu-moe --moe-cache-mib $(mib $1-${2#c})";;
  ncm10|ncm25|ncm50) echo "$F32 -ncmoe $(ncm $1-${2#ncm})";;
esac; }
LINHAS=(gpu32 c10 c25 c50 ncm10 ncm25 ncm50)
for n in 1 2 3; do
  if [ $((n % 2)) = 1 ]; then mods="1b 3b"; else mods="3b 1b"; fi
  for mod in $mods; do
    for i in 0 1 2 3 4 5 6; do
      l=${LINHAS[$(((i + n - 1) % 7))]}
      d=$R/llama/rep$n/$mod
      mkdir -p "$d"
      extra=""; [ $n = 1 ] && extra="--argmax $d/argmax-$l.txt"
      ARQ="$d/argmax-$l.txt" ERRSEP=1 bloco "llama-$mod-$l-$n" norot "$d/$l.txt" $P $B/forcado --tokens $TOK/$mod --warm 0,1,2,3 --convs 4,5,6,7 $extra \
        -m $G/$(dir $mod)-bf16.gguf $COMUM $(flags $mod $l)
    done
  done
done

if [ "$SECO" = 0 ]; then
  {
    date; uptime; echo "GPU no fim: $(gpu)"
    (cd "$B" && sha256sum -c SHA256SUMS) 2>&1 || echo "INVALIDA A JANELA: sha dos binários mudou"
    sha256sum -c "$R/libs.sha256" 2>&1 || echo "INVALIDA A JANELA: sha das bibliotecas mudou"
    echo "processos acima de 5% de CPU no fim:"; procs
    [ -e "$R/suspensas.txt" ] && { echo "células suspensas:"; cat "$R/suspensas.txt"; } || echo "nenhuma célula suspensa"
  } > "$R/ambiente-fim.txt"
fi
echo "$(date +%T) fim"
