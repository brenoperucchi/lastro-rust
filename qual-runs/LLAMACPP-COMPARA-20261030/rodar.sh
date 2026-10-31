#!/bin/bash
# Comparação com o llama.cpp nos Granite 1B e 3B (plano, seção "Comparação com o llama.cpp nos Granite 1B e 3B"). Critério em
# criterio.txt, gravado com a previsão em previsao.txt antes da primeira execução. Núcleo com 3 repetições, linhas de base
# uma vez, linha do servidor no fim. Binários em bin/ (sha conferido antes); nada compila durante a medida; Python só nos
# E-cores. Qualquer falha interrompe o roteiro; ele recusa rodar com outra instância segurando a trava ou com resultados de
# uma sessão anterior na pasta, sem apagar nada. MEDIDA_DIR, MEDIDA_BIN, MEDIDA_LLAMA e MEDIDA_PY só servem ao teste do
# próprio roteiro; a medida roda sem elas.
set -euo pipefail
cd /home/brenoperucchi/Devs/strata-opencode
R=${MEDIDA_DIR:-qual-runs/LLAMACPP-COMPARA-20261030}
exec 9> "$R/.trava"
flock -n 9 || { echo "outra instância do rodar.sh está com a trava"; exit 1; }
for f in "$R"/previsao.txt "$R"/ambiente.txt "$R"/carga.log "$R"/sha-binarios.txt "$R"/nucleo-* "$R"/base-* "$R"/servidor-*; do
  [ -e "$f" ] && { echo "já existe $f: sessão anterior na pasta; nada foi alterado"; exit 1; }
done
B=${MEDIDA_BIN:-$R/bin}
LL=/home/brenoperucchi/Devs/GitHub/llama.cpp
SRV=${MEDIDA_LLAMA:-$LL/build/bin/llama-server}
T=qual-runs/TRACOS-20261029
M=/home/brenoperucchi/models
G=$M/gguf
P="taskset -c 0-15"
PY="taskset -c 16-27 tools/.venv-ref/bin/python -I -B"
PY=${MEDIDA_PY:-$PY}
PORTA=18431
if curl -s --max-time 2 http://127.0.0.1:$PORTA/health > /dev/null; then echo "porta $PORTA já ocupada; nada foi alterado"; exit 1; fi
(cd $B && sha256sum -c SHA256SUMS) > $R/sha-binarios.txt 2>&1 || { echo "sha dos binários não confere"; exit 1; }
dir() { case $1 in 1b) echo granite-3.1-1b-a400m-instruct;; 3b) echo granite-3.1-3b-a800m-instruct;; esac; }
procs() { ps -eo pid,pcpu,comm --sort=-pcpu | awk 'NR == 1 || $2 > 5'; }
{
  date; head -1 /proc/driver/nvidia/version; ls -l /usr/lib/libcuda.so.1; uptime
  echo "llama.cpp: $(git -C $LL rev-parse HEAD) $(git -C $LL status --porcelain | wc -l) arquivos alterados"
  sha256sum $LL/build/bin/libllama.so.0.8.0 $LL/build/bin/libllama-common.so.0.8.0 $LL/build/bin/libggml*.so.0.28.0 $SRV
  sha256sum $G/granite-3.1-1b-a400m-instruct-bf16.gguf $G/granite-3.1-3b-a800m-instruct-bf16.gguf tools/moe_sim.py
  echo "processos acima de 5% de CPU no começo:"; procs
} > $R/ambiente.txt
$PY $R/previsao.py > $R/previsao.tmp || exit 1
{ cat $R/criterio.txt; echo; echo "Previsões gravadas em $(date '+%d/%m %H:%M:%S'), antes da primeira execução."; cat $R/previsao.tmp; } > $R/previsao.txt
rm $R/previsao.tmp

COMUM="-ngl 99 -c 2048 --fit off --no-warmup -t 8"
F32="-ctk f32 -ctv f32 -fa off"  # com -fa on o CUDA converte K/V F32 para F16 antes de calcular (rodada 43)
mib() { case $1 in 1b-10) echo 231;; 1b-25) echo 579;; 1b-50) echo 1155;; 1b-cheio) echo 2307;;
                    3b-10) echo 581;; 3b-25) echo 1445;; 3b-50) echo 2885;; 3b-cheio) echo 5765;; esac; }
ncm() { case $1 in 1b-10) echo 22;; 1b-25) echo 18;; 1b-50) echo 12;; 3b-10) echo 29;; 3b-25) echo 24;; 3b-50) echo 16;; esac; }
flags() { # modelo linha -> flags do llama.cpp além de COMUM
  case $2 in
    gpu32) echo "$F32";;
    gpu16) echo "";;
    cpumoe) echo "$F32 --cpu-moe";;
    c10|c25|c50) echo "$F32 --cpu-moe --moe-cache-mib $(mib $1-${2#c})";;
    cheio) echo "$F32 --cpu-moe --moe-cache-mib $(mib $1-cheio)";;
    ncm10|ncm25|ncm50) echo "$F32 -ncmoe $(ncm $1-${2#ncm})";;
  esac
}
fim() { # saída rc: registra e devolve o código
  echo "$(date +%T) $1 $2"
  echo "$(date +%T) $1 $(cut -d' ' -f1-3 /proc/loadavg)" >> $R/carga.log
  return $2
}
llama() { # saída modelo linha [extra...]: decode forçado; o stderr (log do llama.cpp, hit rate do cache) vai para .log
  local out=$1 mod=$2 linha=$3 rc=0; shift 3
  $P $B/forcado --tokens $R/tokens/$mod --warm 3 --convs 4,5,6,7 "$@" -m $G/$(dir $mod)-bf16.gguf $COMUM $(flags $mod $linha) \
    > $R/$out.txt 2> $R/$out.log || rc=$?
  fim $out.txt $rc
}
nosso() { # saída modelo políticas
  local out=$1 mod=$2 pols=$3 rc=0
  $P $B/m2a-cache --model $M/$(dir $mod) --traces $T/$(dir $mod) --ref qual-runs/REF-granite-$mod-texto-a --budgets 0.1,0.25,0.5 \
    --policies $pols --warmup 3 > $R/$out.txt 2>&1 || rc=$?
  fim $out.txt $rc
}

# Núcleo: 3 repetições; a ordem dos modelos, das linhas do llama.cpp e do bloco nosso/llama.cpp gira a cada repetição. Na 1ª
# repetição o forcado grava o argmax de cada posição para a conferência contra a passada tudo na GPU.
LINHAS=(gpu32 c10 c25 c50)
for n in 1 2 3; do
  if [ $((n % 2)) = 1 ]; then mods="1b 3b"; else mods="3b 1b"; fi
  for mod in $mods; do
    blocos="llama nosso"; [ $((n % 2)) = 0 ] && blocos="nosso llama"
    for bl in $blocos; do
      if [ $bl = nosso ]; then
        nosso nucleo-$mod-nosso-$n $mod resident,lru_cpu,lfu_cpu
      else
        for i in 0 1 2 3; do
          l=${LINHAS[$(((i + n - 1) % 4))]}
          if [ $n = 1 ]; then llama nucleo-$mod-$l-$n $mod $l --argmax $R/argmax-$mod-$l.txt; else llama nucleo-$mod-$l-$n $mod $l; fi
        done
      fi
    done
  done
done

# Linhas de base, uma vez cada, com o argmax gravado para o portão de correção contra a passada tudo na GPU F32.
for mod in 1b 3b; do
  for l in gpu16 cpumoe ncm10 ncm25 ncm50 cheio; do
    llama base-$mod-$l $mod $l --argmax $R/argmax-$mod-$l.txt
  done
  nosso base-$mod-nosso $mod resident,lru_sync,cache_full
done

# Linha do servidor: uso normal, uma vez por configuração do núcleo do llama.cpp. O servidor não herda a trava (9>&-) e é
# encerrado pelo PID exato, também quando o roteiro falha ou é interrompido (trap); com a porta já ocupada ou o servidor
# morto antes de ficar pronto, o roteiro falha sem medir e sem tocar no que ocupa a porta (rodada 43).
pid=
limpa() { if [ -n "$pid" ]; then kill $pid 2> /dev/null || true; wait $pid 2> /dev/null || true; pid=; fi; }
trap limpa EXIT
trap 'limpa; exit 130' INT TERM
for mod in 1b 3b; do
  for l in gpu32 c10 c25 c50; do
    out=servidor-$mod-$l
    if curl -s --max-time 2 http://127.0.0.1:$PORTA/health > /dev/null; then echo "porta $PORTA já ocupada; a linha do servidor não foi medida"; exit 1; fi
    $P $SRV -m $G/$(dir $mod)-bf16.gguf $COMUM $(flags $mod $l) -np 1 --host 127.0.0.1 --port $PORTA > $R/$out.log 2>&1 9>&- &
    pid=$!
    pronto=0
    for i in $(seq 120); do
      kill -0 $pid 2> /dev/null || break
      if curl -sf http://127.0.0.1:$PORTA/health > /dev/null; then pronto=1; break; fi
      sleep 1
    done
    rc=0
    if [ $pronto = 1 ]; then $PY $R/servidor.py $R/tokens/$mod $M/$(dir $mod) $PORTA > $R/$out.txt 2>&1 || rc=$?; else rc=9; fi
    limpa
    fim $out.txt $rc
  done
done
{ echo "processos acima de 5% de CPU no fim:"; procs; } >> $R/ambiente.txt
echo "$(date +%T) fim"
