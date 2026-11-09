#!/usr/bin/env bash
# Comparação Lastro x llama.cpp (criterio-comparacao-llamacpp-20261108.md, E1, E4, emenda 4): UMA célula do llama-server (modelo, texto, estado, série), com GPU.
#
#   roda-cmp-llamacpp.sh --modelo 1b|3b --texto curto|medio|longo --estado frio|quente --serie S1|S2|S3 --n N --saida DIR [--ids cmp-ids-prompt.json]
#
# Séries (todas com -t 8 -c 2048 -np 1 -b 2048 -ub 512 --host 127.0.0.1 --no-webui, GPU = a 4090; --metrics desligado; --no-op-offload NÃO usado):
#   S1 (principal): -ngl 99 -fa on, KV f16 (--cache-type-k f16 --cache-type-v f16 explícitos, o padrão do llama.cpp)
#   S2 (descarregamento de ordem de grandeza parecida): -ngl 99 -fa on --n-cpu-moe 22 (1B, 24 camadas) ou 29 (3B, 32 camadas), KV f16
#   S3 (opcional, só se sobrar orçamento): -ngl 99 -fa off, KV f32 (--cache-type-k f32 --cache-type-v f32)
# Frio = processo NOVO por medida, `--no-warmup` na linha de comando e SEM pedido de aquecimento nosso: p0 = servidor descartado + p1..pN medidas (cada um: partida,
# marcador de pronto, 1 pedido, encerramento). Quente = UM servidor com o aquecimento padrão, 1 pedido de aquecimento descartado nosso (resp-0) e as medidas resp-1..N.
# Pedido: POST /completion com "prompt" = ARRAY DE IDS de cmp-ids-prompt.json (modelo, texto), n_predict 64, temperature 0, top_k 1, seed 0, cache_prompt false,
# return_tokens true, ignore_eos false, stream false. Marcador de pronto: 'model loaded' no log do servidor ou GET /health 200 (a linha 'server is listening' NÃO existe
# neste build). Carga = da partida do processo até o marcador (carga-K.ms). Grava a linha de comando (cmdline.txt), `llama-server --version` (version.txt), o log de cada
# servidor (servidor-K.log), as linhas de KV/flash attention do log (kv-fa.txt), a resposta de cada pedido (resp-K.json) e curl/HTTP (resp-K.rc "curl http"), ms (resp-K.ms).
# SEMPRE encerra o servidor (TERM, depois KILL) e confere que não sobrou processo com a porta usada. Rede só em 127.0.0.1.
#
# Falha FECHADA (2) sem PRIMEIRA_PALAVRA_AUTORIZADA=1 (o "autorizo" do Breno) e sem CUDA_VISIBLE_DEVICES igual ao UUID da 4090 (a 5090 nunca), ou com DIR existente.
# Portão e monitor como em roda-cmp-lastro.sh. Variáveis só para teste: LLAMA_SERVER, GGUF_DIR, CURL, MONITOR_SH, NVIDIA_SMI, ESPERA_PRONTO (s, padrão 300).
# Códigos: 0 a célula rodou até o fim, 1 algum passo falhou (partida, pedido ou encerramento), 2 recusa de uso/ambiente.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
UUID=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd
LLAMA=${LLAMA_SERVER:-/home/brenoperucchi/Devs/GitHub/llama.cpp/build/bin/llama-server}
GGUFS=${GGUF_DIR:-/home/brenoperucchi/models/gguf}
CURL=${CURL:-curl}
MONITOR=${MONITOR_SH:-$AQUI/roda-cmp-monitor.sh}
IDS=$AQUI/cmp-ids-prompt.json
ESPERA=${ESPERA_PRONTO:-300}
uso() { echo "uso: roda-cmp-llamacpp.sh --modelo 1b|3b --texto curto|medio|longo --estado frio|quente --serie S1|S2|S3 --n N --saida DIR [--ids ARQ]" >&2; exit 2; }
recusa() { echo "roda-cmp-llamacpp: $*" >&2; exit 2; }
MODELO=; TEXTO=; ESTADO=; SERIE=; N=; SAIDA=
while [ $# -gt 0 ]; do
  case $1 in
    --modelo) [ $# -ge 2 ] || uso; MODELO=$2; shift 2;;
    --texto) [ $# -ge 2 ] || uso; TEXTO=$2; shift 2;;
    --estado) [ $# -ge 2 ] || uso; ESTADO=$2; shift 2;;
    --serie) [ $# -ge 2 ] || uso; SERIE=$2; shift 2;;
    --n) [ $# -ge 2 ] || uso; N=$2; shift 2;;
    --saida) [ $# -ge 2 ] || uso; SAIDA=$2; shift 2;;
    --ids) [ $# -ge 2 ] || uso; IDS=$2; shift 2;;
    *) uso;;
  esac
done
case $MODELO in 1b) GGUF=$GGUFS/granite-3.1-1b-a400m-instruct-bf16.gguf; NCMOE=22;; 3b) GGUF=$GGUFS/granite-3.1-3b-a800m-instruct-bf16.gguf; NCMOE=29;; *) uso;; esac
case $TEXTO in curto|medio|longo) ;; *) uso;; esac
case $ESTADO in frio|quente) ;; *) uso;; esac
case $SERIE in S1|S2|S3) ;; *) uso;; esac
case $N in ''|*[!0-9]*) uso;; esac
[ "$N" -ge 1 ] && [ "$N" -le 50 ] || recusa "--n fora de 1..50: $N"
[ -n "$SAIDA" ] || uso
[ "${PRIMEIRA_PALAVRA_AUTORIZADA:-}" = 1 ] || recusa "portão fechado: sem PRIMEIRA_PALAVRA_AUTORIZADA=1"
[ "${CUDA_VISIBLE_DEVICES:-}" = "$UUID" ] || recusa "CUDA_VISIBLE_DEVICES tem de ser $UUID (a 4090); veio '${CUDA_VISIBLE_DEVICES:-}'"
[ -x "$LLAMA" ] || recusa "llama-server não executável: $LLAMA"
[ -f "$GGUF" ] || recusa "GGUF inexistente: $GGUF"
[ -f "$IDS" ] || recusa "sem $IDS"
[ -x "$MONITOR" ] || recusa "monitor não executável: $MONITOR"
if [ -e "$SAIDA" ]; then recusa "$SAIDA já existe: pasta de saída nova"; fi
mkdir -p "$SAIDA" || recusa "não consegui criar $SAIDA"
SAIDA=$(cd "$SAIDA" && pwd -P)
ms_agora() { echo $(( $(date +%s%N) / 1000000 )); }
porta_livre() { python3 -I -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }

# corpo do pedido (ids do cmp-ids-prompt.json)
python3 -I - "$IDS" "$MODELO" "$TEXTO" "$SAIDA/pedido.json" <<'PY' || recusa "ids de prompt de $MODELO/$TEXTO indisponíveis em $IDS"
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
ids = d["modelos"][sys.argv[2]][sys.argv[3]]["ids"]
assert ids and all(isinstance(i, int) for i in ids)
json.dump({"prompt": ids, "n_predict": 64, "temperature": 0, "top_k": 1, "seed": 0, "cache_prompt": False, "return_tokens": True, "ignore_eos": False, "stream": False},
          open(sys.argv[4], "w"))
PY

ARGS_BASE=(-m "$GGUF" -c 2048 -np 1 -b 2048 -ub 512 -t 8 --host 127.0.0.1 --no-webui)
case $SERIE in
  S1) ARGS_SERIE=(-ngl 99 -fa on --cache-type-k f16 --cache-type-v f16);;
  S2) ARGS_SERIE=(-ngl 99 -fa on --n-cpu-moe "$NCMOE" --cache-type-k f16 --cache-type-v f16);;
  S3) ARGS_SERIE=(-ngl 99 -fa off --cache-type-k f32 --cache-type-v f32);;
esac
[ "$ESTADO" = frio ] && ARGS_AQUEC=(--no-warmup) || ARGS_AQUEC=()

SRV_PID=; PORTA=; FALHOU=0; MOTIVOS=()
derruba() { # derruba o servidor e confere ausência de processo; retorna 1 se sobrou algo
  local p=$SRV_PID i
  if [ -n "$p" ]; then
    if kill -0 "$p" 2> /dev/null; then
      kill -TERM "$p" 2> /dev/null
      for i in $(seq 40); do kill -0 "$p" 2> /dev/null || break; sleep 0.25; done
      kill -0 "$p" 2> /dev/null && kill -KILL "$p" 2> /dev/null
    fi
    wait "$p" 2> /dev/null
    SRV_PID=
  fi
  if [ -n "$PORTA" ]; then
    # padrão restrito (r111, rev-2): só processo de llama-server com ESTA porta (a porta veio do bind efêmero), nunca qualquer "--port N" da máquina
    if pgrep -f -- "llama-server .*--port $PORTA( |\$)" > /dev/null 2>&1; then pkill -KILL -f -- "llama-server .*--port $PORTA( |\$)" 2> /dev/null; sleep 0.2; PORTA=; return 1; fi
    PORTA=
  fi
  return 0
}
cleanup() { derruba; "$MONITOR" para --pid-arq "$SAIDA/monitor.pid" 2> /dev/null; r_para=$?; }
trap cleanup EXIT
trap 'cleanup; exit 143' INT TERM HUP

# linha de comando completa (porta como <PORTA>); `--version` roda DEPOIS da célula (não toca a GPU durante a medida)
printf '%s ' "$LLAMA" "${ARGS_BASE[@]}" "${ARGS_SERIE[@]}" "${ARGS_AQUEC[@]}" --port '<PORTA>' > "$SAIDA/cmdline.txt"; echo >> "$SAIDA/cmdline.txt"

INI=$(date +%s)
"$MONITOR" portao --saida "$SAIDA/portao.txt"; PORTAO_RC=$?
PORTAO_MOTIVO=$(grep '^portao=FALHA' "$SAIDA/portao.txt" 2> /dev/null | head -n 1 | cut -d' ' -f2- | tr -d '"\\' | tr '\n' ' ')
MON_FALHAS=()
"$MONITOR" inicia --csv "$SAIDA/monitor.csv" --pid-arq "$SAIDA/monitor.pid"; r=$?
[ $r = 0 ] || MON_FALHAS+=("inicia rc=$r")
INI_COLETA=$(date +%s)   # emenda 7: marco depois do portão e logo antes da 1ª partida; o extrator exige a 1ª amostra a <= 5 s dele

# sobe um servidor (K = número do servidor): define SRV_PID e PORTA; grava servidor-K.log e carga-K.ms; retorna 1 se não ficou pronto
sobe() {
  local k=$1 t0 t1 t code
  PORTA=$(porta_livre) || { MOTIVOS+=("sem porta livre"); return 1; }
  t0=$(ms_agora)
  "$LLAMA" "${ARGS_BASE[@]}" "${ARGS_SERIE[@]}" "${ARGS_AQUEC[@]}" --port "$PORTA" > "$SAIDA/servidor-$k.log" 2>&1 < /dev/null &
  SRV_PID=$!
  echo "servidor $k pid=$SRV_PID porta=$PORTA" >> "$SAIDA/execucoes.txt"
  for t in $(seq $((ESPERA * 5))); do
    if grep -q 'model loaded' "$SAIDA/servidor-$k.log" 2> /dev/null; then break; fi
    code=$("$CURL" -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$PORTA/health" 2> /dev/null)
    [ "$code" = 200 ] && break
    kill -0 "$SRV_PID" 2> /dev/null || { MOTIVOS+=("servidor $k saiu antes de ficar pronto"); return 1; }
    sleep 0.2
  done
  kill -0 "$SRV_PID" 2> /dev/null || { MOTIVOS+=("servidor $k saiu antes de ficar pronto"); return 1; }
  if ! grep -q 'model loaded' "$SAIDA/servidor-$k.log" 2> /dev/null && [ "${code:-}" != 200 ]; then MOTIVOS+=("servidor $k sem marcador de pronto em ${ESPERA} s"); return 1; fi
  t1=$(ms_agora)
  echo $((t1 - t0)) > "$SAIDA/carga-$k.ms"
  # linhas de KV e flash attention do log (descritivo)
  grep -iE 'flash|type_k|type_v|kv' "$SAIDA/servidor-$k.log" 2> /dev/null | head -n 40 > "$SAIDA/kv-fa-$k.txt"
  [ -s "$SAIDA/kv-fa.txt" ] || cp "$SAIDA/kv-fa-$k.txt" "$SAIDA/kv-fa.txt" 2> /dev/null
  return 0
}
# pede K: POST /completion; grava resp-K.json, resp-K.rc ("curl http") e resp-K.ms
pede() {
  local k=$1 t0 out rc
  t0=$(ms_agora)
  out=$("$CURL" -sS --max-time 600 -X POST -H 'Expect:' -H 'Content-Type: application/json' --data-binary "@$SAIDA/pedido.json" -o "$SAIDA/resp-$k.json" -w '%{http_code}' \
        "http://127.0.0.1:$PORTA/completion" 2> "$SAIDA/resp-$k.err"); rc=$?
  echo $(( $(ms_agora) - t0 )) > "$SAIDA/resp-$k.ms"
  echo "$rc ${out:-000}" > "$SAIDA/resp-$k.rc"
  [ $rc = 0 ] && [ "${out:-000}" = 200 ]
}
termina_servidor() { # confere ausência de processo; marca falha se sobrou
  derruba || { MOTIVOS+=("sobrou processo com a porta do servidor (derrubado por KILL)"); FALHOU=1; }
}

if [ "$ESTADO" = frio ]; then
  for k in $(seq 0 "$N"); do
    if sobe "$k"; then pede "$k" || { MOTIVOS+=("pedido $k falhou"); FALHOU=1; }; else FALHOU=1; fi
    termina_servidor
    [ $FALHOU = 0 ] || break
  done
else
  if sobe 0; then
    for k in $(seq 0 "$N"); do
      pede "$k" || { MOTIVOS+=("pedido $k falhou"); FALHOU=1; break; }
    done
  else FALHOU=1; fi
  termina_servidor
fi
# a coleta tem de estar viva até o fim da célula (r111, rev-1 P2); falha de inicia/para/avalia fica em celula.json ("monitor_falha") e o extrator trata como SEM PROVA
MON_PID=$(cat "$SAIDA/monitor.pid" 2> /dev/null)
{ [ -n "$MON_PID" ] && kill -0 "$MON_PID" 2> /dev/null; } || MON_FALHAS+=("coleta encerrada antes do fim da célula")
FIM=$(date +%s)
r_para=0
cleanup
[ "$r_para" = 0 ] || MON_FALHAS+=("para rc=$r_para")
"$MONITOR" avalia --csv "$SAIDA/monitor.csv" > "$SAIDA/monitor-resumo.txt" 2>&1; r=$?
if [ $r -gt 1 ] || grep -q '^sem amostras' "$SAIDA/monitor-resumo.txt" 2> /dev/null; then MON_FALHAS+=("avalia rc=$r"); fi
MON_FALHA=$(printf '%s; ' "${MON_FALHAS[@]:-}" | tr -d '"\\' | sed -e 's/; $//')
"$LLAMA" --version > "$SAIDA/version.txt" 2>&1 < /dev/null || echo "llama-server --version saiu com erro" >> "$SAIDA/version.txt"
{
  printf '{"lado": "llamacpp", "serie": "%s", "modelo": "%s", "texto": "%s", "estado": "%s", "n_medidas": %s, "portao_ok": %s, "portao_motivo": "%s", "rc": %s, "inicio_epoch": %s, "inicio_coleta_epoch": %s, "fim_epoch": %s, "falhas": "%s", "monitor_falha": "%s"}\n' \
    "$SERIE" "$MODELO" "$TEXTO" "$ESTADO" "$N" "$([ $PORTAO_RC = 0 ] && echo true || echo false)" "$PORTAO_MOTIVO" "$FALHOU" "$INI" "$INI_COLETA" "$FIM" "$(printf '%s; ' "${MOTIVOS[@]:-}" | tr -d '"\\')" "$MON_FALHA"
} > "$SAIDA/celula.json"
[ $FALHOU = 0 ] || exit 1
exit 0
