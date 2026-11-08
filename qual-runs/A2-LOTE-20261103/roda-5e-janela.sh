#!/usr/bin/env bash
# A2, etapa 5e: a prova com GPU do servidor local `lastro-servidor` (criterio-5e-servidor-20261108.md), em UMA janela, na composição já validada: janela-e4b.sh (pausa
# e restauração conferida dos timers) por fora, roda-com-limite.sh (limite de tempo) por dentro, e este mesmo script em modo interno.
#
#   roda-5e-janela.sh --saida DIR [--estado-janela DIR] [--bin BINARIO] [--limite S]
#
# Dentro da janela: (1) roda-5e-recusas.sh no binário (sem GPU); (2) por modelo, 1B e depois 3B, um servidor por vez: porta livre, servidor em segundo plano (PID guardado),
# espera da linha `escutando em 127.0.0.1:<porta>` (até 120 s), `ss -ltnp` (socket só em 127.0.0.1, nunca 0.0.0.0, [::] ou *), GET /saude, POST /gerar com curto, médio e
# longo (data=November 07, 2026, novos=64) comparando o corpo byte a byte com texto_gerado + "\n" de 5c-referencia/ref5c-<modelo>.json, o médio repetido no fim (igual à
# referência e ao primeiro), recusa de corpo vazio (400) com o motor ligado e um pedido correto depois, POST /encerrar, código 0 em até 10 s, porta fechada e nenhum processo
# remanescente (se não encerrar: TERM, depois KILL, e o rótulo é REPROVA); (3) grava DIR/decisao.txt: PASSA / REPROVA / SEM PROVA pelas regras do critério.
# Uma TRAVA derruba o servidor em qualquer saída do roteiro (erro, sinal, limite). Integridade: sha256 do binário, dos critérios 5e e 5c, dos 3 textos, das 2 referências e do
# tokenizer de cada modelo, antes e depois (e contra os valores congelados); quebrada = SEM PROVA em tudo. REPROVA vence SEM PROVA, SEM PROVA vence PASSA.
# Descritivo (sem rótulo): tempo da partida até `escutando` e tempo de cada POST.
#
# Falha FECHADA antes de tocar em timers ou na GPU: PRIMEIRA_PALAVRA_AUTORIZADA=1 e CUDA_VISIBLE_DEVICES igual ao UUID da 4090 têm de estar no ambiente de quem chama
# (o "autorizo" do Breno); --saida e --estado-janela têm de ser pastas novas ou vazias (padrão do estado: DIR-estado-janela). Este roteiro NÃO é usado na entrega.
# Limite padrão 1800 s (máx. 1800); o de fora é o de dentro mais 120 s. Variáveis só para teste: JANELA_SH, LIMITE_SH, JANELA_TIMERS, MODELO_1B, MODELO_3B, TEXTOS_5C, REF_5C,
# CRITERIO_5E, CRITERIO_5C, RECUSAS_SH, SEM_SHA_FIXO=1 (pula a conferência contra os shas congelados), NVIDIA_SMI, LOADAVG_ARQ, CURL, SS, ESPERA_ESCUTANDO (s, padrão 120).
# Códigos: 0 se decisao.txt foi gravada (o rótulo está nela), o do comando (124 limite, etc.) ou 2 recusa de uso, portão ou ambiente.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
UUID=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd
JANELA=${JANELA_SH:-$AQUI/janela-e4b.sh}
LIMITE_SH=${LIMITE_SH:-$AQUI/roda-com-limite.sh}
RECUSAS=${RECUSAS_SH:-$AQUI/roda-5e-recusas.sh}
BIN=$AQUI/../../rust/moe-rt/target/release/lastro-servidor
M1=${MODELO_1B:-/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct}
M3=${MODELO_3B:-/home/brenoperucchi/models/granite-3.1-3b-a800m-instruct}
TEXTOS=${TEXTOS_5C:-$AQUI/5c-textos}
REFS=${REF_5C:-$AQUI/5c-referencia}
C5E=${CRITERIO_5E:-$AQUI/criterio-5e-servidor-20261108.md}
C5C=${CRITERIO_5C:-$AQUI/criterio-5c-geracao-livre-20261108.md}
CURL=${CURL:-curl}
SS=${SS:-ss}
ESPERA_ESC=${ESPERA_ESCUTANDO:-120}
DATA_Q="November%2007%2C%202026"   # "November 07, 2026" urlencoded
SAIDA=; ESTADO=; LIMITE=1800; INTERNO=0
usa() { echo "uso: roda-5e-janela.sh --saida DIR [--estado-janela DIR] [--bin BINARIO] [--limite S]" >&2; }
recusa() { echo "roda-5e-janela: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
  case $1 in
    --saida) [ $# -ge 2 ] || { usa; exit 2; }; SAIDA=$2; shift 2;;
    --estado-janela) [ $# -ge 2 ] || { usa; exit 2; }; ESTADO=$2; shift 2;;
    --bin) [ $# -ge 2 ] || { usa; exit 2; }; BIN=$2; shift 2;;
    --limite) [ $# -ge 2 ] || { usa; exit 2; }; LIMITE=$2; shift 2;;
    --interno) INTERNO=1; shift;;
    *) usa; exit 2;;
  esac
done
[ -n "$SAIDA" ] || { usa; exit 2; }

# SHAs congelados (critério 5c, textos e referências do 5c, tokenizer do Granite 3.1, critério 5e).
declare -A FIXO=(
  [criterio5e]=afeb7e2d09f29e767f753dc11f793d3bee459ce5176ecbfa106cecdcfb427ac8
  [criterio5c]=9e3299fd9bfaac4b3d8ec2bb5c2ad941c24e2e7124070feb7dc511079a831731
  [texto-curto]=5b6fad6f3825d430bad4f924d227105032581906739d68c726ac8e4827df2d4d
  [texto-medio]=bb0a067ae8839616f33b21ba41b362bdf16353a57d70321252a2d47851f1c32a
  [texto-longo]=821e0f15139b58173b5c17f5e9428527bb27b03dc10b35e38cbeed044defbaea
  [ref5c-1b]=9584fcabe7d8b2a1821207267651504c72d317ce9b566cd967d390a1dbbad259
  [ref5c-3b]=968d9b391324a99fbaf5c772a351018b90105f572b32ee2db3140f79da123124
  [tokenizer-1b]=ed944c0b3d71e2d651a5ba0f8e7b350faff30031c8e3353de70b12b9c1bf5654
  [tokenizer-3b]=ed944c0b3d71e2d651a5ba0f8e7b350faff30031c8e3353de70b12b9c1bf5654
)
declare -A ARQ=(
  [binario]=$BIN [criterio5e]=$C5E [criterio5c]=$C5C
  [texto-curto]=$TEXTOS/texto-curto.txt [texto-medio]=$TEXTOS/texto-medio.txt [texto-longo]=$TEXTOS/texto-longo.txt
  [ref5c-1b]=$REFS/ref5c-1b.json [ref5c-3b]=$REFS/ref5c-3b.json [tokenizer-1b]=$M1/tokenizer.json [tokenizer-3b]=$M3/tokenizer.json
)
CHAVES=(binario criterio5e criterio5c texto-curto texto-medio texto-longo ref5c-1b ref5c-3b tokenizer-1b tokenizer-3b)
sha() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
# coleta_shas ARQUIVO_DE_SAIDA: uma linha "chave sha" por item (vazio = ilegível).
coleta_shas() { : > "$1"; for k in "${CHAVES[@]}"; do echo "$k $(sha "${ARQ[$k]}")" >> "$1"; done; }

# ---------------------------------------------------------------- modo interno (dentro da janela, com o portão e o UUID no ambiente)
if [ $INTERNO = 1 ]; then
  [ "${PRIMEIRA_PALAVRA_AUTORIZADA:-}" = 1 ] && [ "${CUDA_VISIBLE_DEVICES:-}" = "$UUID" ] || recusa "modo interno sem portão ou UUID"
  mkdir -p "$SAIDA" || exit 2
  # TRAVA: em QUALQUER saída (erro, sinal, limite) o servidor ainda vivo é derrubado (TERM, espera curta, KILL).
  SERVIDOR_PID=
  derruba_servidor() {
    local p=$SERVIDOR_PID i
    [ -n "$p" ] || return 0
    if kill -0 "$p" 2> /dev/null; then
      kill -TERM "$p" 2> /dev/null
      for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do kill -0 "$p" 2> /dev/null || break; sleep 0.25; done
      kill -0 "$p" 2> /dev/null && kill -KILL "$p" 2> /dev/null
    fi
    wait "$p" 2> /dev/null
    SERVIDOR_PID=
  }
  trap 'derruba_servidor' EXIT
  trap 'derruba_servidor; exit 143' INT TERM HUP
  coleta_shas "$SAIDA/integridade-antes.txt"
  # 1. recusas (sem GPU)
  "$RECUSAS" --bin "$BIN" > "$SAIDA/recusas.txt" 2>&1; RC_RECUSAS=$?
  declare -A MDIR=([1b]=$M1 [3b]=$M3)
  REPROVA=0; SEMPROVA=0; LINHAS=(); TEMPOS=(); : > "$SAIDA/execucoes.txt"
  NOK=0
  ok() { NOK=$((NOK+1)); LINHAS+=("$M: $*: PASSA"); }
  reprova() { REPROVA=1; LINHAS+=("$M: $*: REPROVA"); }
  sem_prova() { SEMPROVA=1; LINHAS+=("$M: $*: SEM PROVA"); }
  ms_agora() { echo $(( $(date +%s%N) / 1000000 )); }
  porta_livre() { python3 -I -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
  escutando() { "$SS" -H -ltn "sport = :$1" 2> /dev/null | grep -q .; }
  restante() { pgrep -f -- "--porta $PORTA( |\$)" > /dev/null 2>&1; }
  # esperado MODELO NOME -> grava ARQ = texto_gerado + "\n" da referência
  esperado() {
    python3 -I - "$REFS/ref5c-$1.json" "$2" "$3" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
c = [x for x in d["casos"] if x["nome"] == sys.argv[2]]
assert len(c) == 1
open(sys.argv[3], "wb").write((c[0]["texto_gerado"] + "\n").encode("utf-8"))
PY
  }
  # pede ROTULO METODO CAMINHO ARQUIVO_CORPO: grava $SAIDA/$M-ROTULO.{body,err}; define HC (status; 000 = sem resposta), MS, T0 (ms do envio), RC_CURL (código de
  # transporte do curl, preservado) e ENVIADO (1 = a conexão foi aberta e o pedido seguiu; 0 = falha de CONECTAR). MAXT troca o --max-time do POST (padrão 600 s).
  pede() {
    local pref="$SAIDA/$M-$1" out tc
    T0=$(ms_agora)
    if [ "$2" = GET ]; then
      out=$("$CURL" -sS --max-time 60 -o "$pref.body" -w '%{http_code} %{time_connect}' "http://127.0.0.1:$PORTA$3" 2> "$pref.err"); RC_CURL=$?
    else
      out=$("$CURL" -sS --max-time "${MAXT:-600}" -X POST -H 'Expect:' -H 'Content-Type: text/plain; charset=utf-8' --data-binary "@$4" -o "$pref.body" -w '%{http_code} %{time_connect}' "http://127.0.0.1:$PORTA$3" 2> "$pref.err"); RC_CURL=$?
    fi
    MS=$(( $(ms_agora) - T0 ))
    HC=${out%% *}; tc=
    case $out in *" "*) tc=${out#* };; esac
    [ $RC_CURL = 0 ] || HC=000
    [ -n "$HC" ] || HC=000
    # Conectou = sucesso, ou time_connect > 0 (resposta vazia, reset, timeout depois de conectar). Código 5/6/7 ou time_connect 0 = não conectou.
    ENVIADO=0
    if [ $RC_CURL = 0 ]; then ENVIADO=1
    else case $RC_CURL in 5|6|7) ;; *) awk -v t="$tc" 'BEGIN { exit !(t + 0 > 0) }' && ENVIADO=1;; esac; fi
  }
  # gera ROTULO NOME_DO_TEXTO: POST /gerar e compara o corpo com o esperado (status 200 e igual byte a byte). Define GERA_OK=1/0 (0 também sem resposta).
  gera() {
    local rot=$1 nome=$2 pref="$SAIDA/$M-$1"
    GERA_OK=0
    pede "$rot" POST "/gerar?novos=64&data=$DATA_Q" "$TEXTOS/texto-$nome.txt"
    TEMPOS+=("$M POST /gerar $rot: HTTP $HC, $MS ms")
    if [ "$HC" = 000 ]; then sem_prova "POST /gerar $rot (sem resposta do servidor; $(head -c 200 "$pref.err" | tr '\n' ' '))"; return; fi
    if [ "$HC" != 200 ]; then reprova "POST /gerar $rot (status $HC, esperado 200)"; return; fi
    if [ ! -s "$SAIDA/$M-esperado-$nome" ]; then sem_prova "POST /gerar $rot (referência ilegível)"; return; fi
    if cmp -s "$pref.body" "$SAIDA/$M-esperado-$nome"; then GERA_OK=1; ok "POST /gerar $rot (200, corpo igual byte a byte à referência, $MS ms)"
    else reprova "POST /gerar $rot (corpo diferente da referência; ver $M-$rot.body)"; fi
  }
  roda_modelo() {
    M=$1; local mdir=${MDIR[$1]} nome ini fim i vivo rc=
    NOK=0
    for nome in curto medio longo; do esperado "$M" "$nome" "$SAIDA/$M-esperado-$nome" 2> "$SAIDA/$M-esperado-$nome.err" || : > "$SAIDA/$M-esperado-$nome"; done
    PORTA=$(porta_livre) || { sem_prova "sem porta livre"; return; }
    ini=$(ms_agora)
    "$BIN" --modelo "$mdir" --porta "$PORTA" > "$SAIDA/$M-servidor.out" 2> "$SAIDA/$M-servidor.err" &
    SERVIDOR_PID=$!
    echo "$M pid=$SERVIDOR_PID porta=$PORTA" >> "$SAIDA/execucoes.txt"
    # espera a linha `escutando` (limite ESPERA_ESC s); servidor que morre antes = sem resultado
    local pronto=0 t
    for t in $(seq $((ESPERA_ESC * 5))); do
      if grep -qxF "escutando em 127.0.0.1:$PORTA" "$SAIDA/$M-servidor.err" 2> /dev/null; then pronto=1; break; fi
      kill -0 "$SERVIDOR_PID" 2> /dev/null || break
      sleep 0.2
    done
    if [ $pronto = 0 ]; then
      if kill -0 "$SERVIDOR_PID" 2> /dev/null; then sem_prova "partida (sem a linha 'escutando' em $ESPERA_ESC s)"; else wait "$SERVIDOR_PID"; rc=$?; sem_prova "partida (o servidor saiu com código $rc antes de escutar: $(tail -n 1 "$SAIDA/$M-servidor.err" | head -c 300))"; fi
      derruba_servidor; return
    fi
    fim=$(ms_agora); TEMPOS+=("$M partida até 'escutando': $((fim - ini)) ms")
    # socket: só em 127.0.0.1
    "$SS" -ltnp "sport = :$PORTA" > "$SAIDA/$M-ss.txt" 2>&1
    local locais; locais=$("$SS" -H -ltn "sport = :$PORTA" 2> /dev/null | awk '{print $4}')
    if [ -z "$locais" ]; then sem_prova "socket (o ss não mostrou LISTEN na porta $PORTA)"
    elif [ "$(echo "$locais" | grep -vxF "127.0.0.1:$PORTA" | wc -l)" != 0 ]; then reprova "socket (escuta fora de 127.0.0.1: $(echo "$locais" | tr '\n' ' '))"
    else ok "socket (só 127.0.0.1:$PORTA)"; fi
    # saúde
    pede saude GET /saude ""
    TEMPOS+=("$M GET /saude: HTTP $HC, $MS ms")
    if [ "$HC" = 000 ]; then sem_prova "GET /saude (sem resposta)"; elif [ "$HC" = 200 ] && [ "$(cat "$SAIDA/$M-saude.body"; printf x)" = "$(printf 'ok\nx')" ]; then ok "GET /saude (200, ok)"; else reprova "GET /saude (status $HC ou corpo errado)"; fi
    # 3 textos, médio repetido, recusa e pedido correto depois
    gera curto curto
    gera medio medio
    GERA_MEDIO=$GERA_OK
    gera longo longo
    gera medio-repetido medio
    if [ $GERA_OK = 1 ] && [ $GERA_MEDIO = 1 ]; then
      if cmp -s "$SAIDA/$M-medio.body" "$SAIDA/$M-medio-repetido.body"; then ok "médio repetido idêntico ao primeiro"; else reprova "médio repetido diferente do primeiro"; fi
    fi
    : > "$SAIDA/vazio.txt"
    pede vazio POST "/gerar?novos=64&data=$DATA_Q" "$SAIDA/vazio.txt"
    TEMPOS+=("$M POST /gerar vazio: HTTP $HC, $MS ms")
    if [ "$HC" = 000 ]; then sem_prova "recusa de corpo vazio (sem resposta)"; elif [ "$HC" = 400 ]; then ok "recusa de corpo vazio (400)"; else reprova "recusa de corpo vazio (status $HC, esperado 400)"; fi
    gera curto-depois-da-recusa curto
    # encerrar
    : > "$SAIDA/vazio.txt"
    # Orçamento de 10 s DESDE O ENVIO, incluindo a espera HTTP e a saída do processo (curl com --max-time 10).
    MAXT=10 pede encerrar POST /encerrar "$SAIDA/vazio.txt"
    local t_envio=$T0
    TEMPOS+=("$M POST /encerrar: HTTP $HC, $MS ms (curl código $RC_CURL, enviado=$ENVIADO)")
    ENC_SEM_ENTREGA=0
    if [ "$HC" = 000 ] && [ $ENVIADO = 0 ]; then sem_prova "POST /encerrar (falha de CONECTAR, curl código $RC_CURL: o pedido não foi entregue)"; ENC_SEM_ENTREGA=1
    elif [ "$HC" = 000 ]; then reprova "POST /encerrar (pedido ENVIADO, curl código $RC_CURL, sem resposta HTTP)"
    elif [ "$HC" = 200 ] && [ "$(cat "$SAIDA/$M-encerrar.body"; printf x)" = "$(printf 'ate logo\nx')" ]; then ok "POST /encerrar (200, ate logo)"
    else reprova "POST /encerrar (status $HC ou corpo errado)"; fi
    # saída em até 10 s com código 0
    vivo=1
    if [ $ENC_SEM_ENTREGA = 1 ]; then
      # Demonstrado que o /encerrar não pôde ser entregue: não dá para julgar o encerramento; só derruba (a trava também faria isso).
      sem_prova "encerramento (o pedido /encerrar não foi entregue; servidor derrubado por TERM/KILL)"
      derruba_servidor
    else
    while :; do
      kill -0 "$SERVIDOR_PID" 2> /dev/null || { vivo=0; break; }
      [ $(( $(ms_agora) - t_envio )) -lt 10000 ] || break
      sleep 0.1
    done
    if [ $vivo = 0 ]; then
      wait "$SERVIDOR_PID"; rc=$?; SERVIDOR_PID=
      if [ "$rc" = 0 ]; then ok "encerrou com código 0 em até 10 s"; else reprova "encerrou com código $rc (esperado 0)"; fi
    else
      reprova "o processo NÃO encerrou em 10 s depois do /encerrar enviado (derrubado por TERM e depois KILL)"
      derruba_servidor
    fi
    fi
    # porta fechada e nenhum processo remanescente
    for i in $(seq 30); do escutando "$PORTA" || break; sleep 0.1; done
    if escutando "$PORTA"; then reprova "a porta $PORTA continua escutando depois do fim"; else ok "porta $PORTA fechada"; fi
    sleep 0.2
    if restante; then reprova "processo remanescente com --porta $PORTA"; pkill -KILL -f -- "--porta $PORTA( |\$)" 2> /dev/null; else ok "nenhum processo remanescente"; fi
    echo "$M checks_ok=$NOK" >> "$SAIDA/execucoes.txt"
  }
  declare -A OKS
  for m in 1b 3b; do roda_modelo "$m"; OKS[$m]=$NOK; derruba_servidor; done
  coleta_shas "$SAIDA/integridade-depois.txt"
  # 3. decisão
  INTEGRO=1; NOTAS=()
  if ! cmp -s "$SAIDA/integridade-antes.txt" "$SAIDA/integridade-depois.txt"; then INTEGRO=0; NOTAS+=("integridade: sha mudou entre antes e depois"); fi
  while read -r k h; do
    [ -n "$h" ] || { INTEGRO=0; NOTAS+=("integridade: $k ilegível"); continue; }
    if [ "${SEM_SHA_FIXO:-0}" != 1 ] && [ -n "${FIXO[$k]:-}" ] && [ "$h" != "${FIXO[$k]}" ]; then INTEGRO=0; NOTAS+=("integridade: $k com sha $h, não o congelado"); fi
  done < "$SAIDA/integridade-antes.txt"
  RECLINHA=
  if [ $RC_RECUSAS != 0 ]; then
    if [ $RC_RECUSAS = 1 ]; then REPROVA=1; RECLINHA="recusas de partida: FALHARAM (roda-5e-recusas.sh saiu 1): REPROVA"; else SEMPROVA=1; RECLINHA="recusas de partida: sem resultado (roda-5e-recusas.sh saiu $RC_RECUSAS): SEM PROVA"; fi
  else RECLINHA="recusas de partida: OK"; fi
  NOTA_SYSTEMD="unidades systemd: a única unidade criada pelo roteiro além da pausa e restauração dos backups é a transitória do limite de tempo (roda-com-limite.sh, systemd-run --collect, sem enable e sem boot); listas antes e depois ao lado da pasta de saída (unidades-antes/depois.txt, timers-antes/depois.txt)"
  NOTA_500="status 500 do motor conta como status errado e vira REPROVA"

  ESPERADOS=13
  COMPLETO=1
  for m in 1b 3b; do [ "${OKS[$m]:-0}" = $ESPERADOS ] || COMPLETO=0; done
  if [ $INTEGRO = 0 ]; then ROTULO="SEM PROVA"
  elif [ $REPROVA = 1 ]; then ROTULO=REPROVA
  elif [ $SEMPROVA = 1 ] || [ $COMPLETO = 0 ]; then ROTULO="SEM PROVA"
  else ROTULO=PASSA; fi
  {
    echo "etapa 5e, servidor local (critério sha256 $(sha "$C5E"))"
    echo "$RECLINHA"
    echo "$NOTA_SYSTEMD"
    echo "$NOTA_500"
    for l in "${LINHAS[@]}"; do echo "$l"; done
    echo "verificações que passaram: 1b ${OKS[1b]:-0} de $ESPERADOS, 3b ${OKS[3b]:-0} de $ESPERADOS"
    if [ $INTEGRO = 1 ]; then echo "integridade: íntegra (binário, critérios, textos, referências e tokenizers iguais antes e depois e aos congelados)"
    else echo "integridade: QUEBRADA: tudo SEM PROVA"; for n in "${NOTAS[@]}"; do echo "  $n"; done; fi
    echo "tempos (descritivos, sem rótulo; o 1º pedido de cada modelo contra os seguintes):"
    for t in "${TEMPOS[@]}"; do echo "  $t"; done
    echo "rótulo: $ROTULO"
  } > "$SAIDA/decisao.txt"
  cat "$SAIDA/decisao.txt"
  exit 0
fi

# ---------------------------------------------------------------- modo externo (portões, janela e limite)
case $LIMITE in ''|*[!0-9]*) recusa "limite inválido: $LIMITE";; esac
[ "$LIMITE" -ge 1 ] && [ "$LIMITE" -le 1800 ] || recusa "limite fora de 1..1800 s: $LIMITE"
[ "${PRIMEIRA_PALAVRA_AUTORIZADA:-}" = 1 ] || recusa "portão fechado: sem PRIMEIRA_PALAVRA_AUTORIZADA=1 (o 'autorizo' do Breno que nomeia a janela)"
[ "${CUDA_VISIBLE_DEVICES:-}" = "$UUID" ] || recusa "CUDA_VISIBLE_DEVICES tem de ser $UUID (a 4090 combinada); veio '${CUDA_VISIBLE_DEVICES:-}'"
[ -x "$BIN" ] || recusa "binário não executável: $BIN"
[ -x "$JANELA" ] && [ -x "$LIMITE_SH" ] && [ -x "$RECUSAS" ] || recusa "faltam janela-e4b.sh, roda-com-limite.sh ou roda-5e-recusas.sh executáveis"
[ -n "$ESTADO" ] || ESTADO=$SAIDA-estado-janela
for d in "$SAIDA" "$ESTADO"; do
  if [ -e "$d" ]; then
    [ -d "$d" ] && [ -z "$(ls -A "$d")" ] || recusa "$d já existe e não está vazio: uma janela nova pede pasta nova"
  fi
done
SAIDA=$(realpath -m "$SAIDA"); BIN=$(realpath "$BIN")
TIMERS=()
[ -z "${JANELA_TIMERS:-}" ] || TIMERS=(--timers "$JANELA_TIMERS")
coleta() {
  {
    echo "$1 $(date '+%d/%m %H:%M:%S')"
    "${NVIDIA_SMI:-nvidia-smi}" --query-gpu=uuid,memory.used,utilization.gpu --format=csv 2>&1 || echo "nvidia-smi indisponível"
    "${NVIDIA_SMI:-nvidia-smi}" --query-compute-apps=pid,used_memory,name --format=csv,noheader 2>&1 || true
    echo "loadavg $(cat "${LOADAVG_ARQ:-/proc/loadavg}" 2>&1)"
    echo "binário $(sha "$BIN")"
  } > "$2" 2>&1 || true
}
# As variáveis de teste atravessam as duas camadas por `env`.
PASSA=("CUDA_VISIBLE_DEVICES=$UUID" PRIMEIRA_PALAVRA_AUTORIZADA=1)
for v in MODELO_1B MODELO_3B TEXTOS_5C REF_5C CRITERIO_5E CRITERIO_5C RECUSAS_SH SEM_SHA_FIXO CURL SS ESPERA_ESCUTANDO; do
  [ -z "${!v+x}" ] || PASSA+=("$v=${!v}")
done
# systemd (descritivo): unidades e timers antes e depois, ao lado da pasta de saída.
lista_systemd() {
  "${SYSTEMCTL:-systemctl}" --user list-unit-files --type=service,timer > "$SAIDA.unidades-$1.txt" 2>&1 || true
  "${SYSTEMCTL:-systemctl}" --user list-timers --all > "$SAIDA.timers-$1.txt" 2>&1 || true
}
lista_systemd antes
coleta início "$SAIDA.carga-antes.txt"
"$JANELA" --estado "$ESTADO" "${TIMERS[@]}" --limite $((LIMITE + 120)) --kill-apos 30 -- \
  "$LIMITE_SH" --limite "$LIMITE" -- env "${PASSA[@]}" "$AQUI/roda-5e-janela.sh" --interno --saida "$SAIDA" --bin "$BIN"
RC=$?
coleta fim "$SAIDA.carga-depois.txt"
lista_systemd depois
exit $RC
