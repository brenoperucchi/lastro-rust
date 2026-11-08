#!/usr/bin/env bash
# A2, etapa 5e: as recusas de PARTIDA do critério (criterio-5e-servidor-20261108.md) no binário REAL `lastro-servidor`, SEM GPU: portão PRIMEIRA_PALAVRA_AUTORIZADA
# fora do ambiente e CUDA_VISIBLE_DEVICES vazio. Cada caso confere código de saída 2 (antes do device), saída padrão vazia, mensagem na saída de erro, tempo < 5 s
# e, por `ss`, que nada ficou escutando na porta usada (a exceção é o caso da porta ocupada, onde só o ocupante de teste escuta e, depois de derrubado, a porta fecha).
#
#   roda-5e-recusas.sh [--bin BINARIO]      (padrão: rust/moe-rt/target/release/lastro-servidor; MODELO_1B troca o modelo real; SS troca o `ss`)
#
# Código: 0 todas OK; 1 alguma falhou; 2 uso. Pasta com mktemp -d; remove só os arquivos que criou (rm de cada um e rmdir), nunca rm -rf.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
BIN=$AQUI/../../rust/moe-rt/target/release/lastro-servidor
if [ $# -ge 1 ]; then
  [ "$1" = --bin ] && [ $# -eq 2 ] || { echo "uso: roda-5e-recusas.sh [--bin BINARIO]" >&2; exit 2; }
  BIN=$2
fi
[ -x "$BIN" ] || { echo "binário não executável: $BIN" >&2; exit 2; }
SS=${SS:-ss}
command -v "$SS" > /dev/null || { echo "sem ss: $SS" >&2; exit 2; }
M1=${MODELO_1B:-/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct}
D=$(mktemp -d -p "${TMPDIR:-/tmp}" recusas-5e.XXXXXX) || exit 2
falhas=0
PRAZO=${RECUSAS_PRAZO:-5}   # prazo individual por caso, em s (o critério fixa 5; a variável existe só para o teste do prazo)
OCUPANTE=
limpa() { [ -z "$OCUPANTE" ] || { kill "$OCUPANTE" 2> /dev/null; wait "$OCUPANTE" 2> /dev/null; OCUPANTE=; }; }
trap 'limpa' EXIT
trap 'limpa; exit 2' INT TERM HUP
porta_livre() { python3 -I -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
escutando() { "$SS" -H -ltn "sport = :$1" 2> /dev/null | grep -q .; }   # há LISTEN nessa porta?
# caso NOME 'trecho da mensagem' PORTA_A_CONFERIR -- ARGS...   (PORTA_A_CONFERIR vazio = não confere o ss; "-" idem)
caso() {
  local nome=$1 msg=$2 porta=$3; shift 3
  [ "$1" = -- ] && shift
  local t0 t1 rc ms ok=1 motivo=
  t0=$(date +%s%N)
  # timeout põe o caso em grupo próprio e mata o grupo inteiro no prazo; --kill-after encerra quem ignorar o TERM.
  if [ -n "${CASO_CUDA:-}" ]; then
    # Variante com o portão ABERTO e outra GPU (caso 4b): PRIMEIRA_PALAVRA_AUTORIZADA=1 e CUDA_VISIBLE_DEVICES=$CASO_CUDA.
    env PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES="$CASO_CUDA" timeout --kill-after=1 "$PRAZO" "$BIN" "$@" < /dev/null > "$D/out" 2> "$D/err"
  else
    env -u PRIMEIRA_PALAVRA_AUTORIZADA CUDA_VISIBLE_DEVICES= timeout --kill-after=1 "$PRAZO" "$BIN" "$@" < /dev/null > "$D/out" 2> "$D/err"
  fi
  rc=$?
  t1=$(date +%s%N); ms=$(( (t1 - t0) / 1000000 ))
  case $rc in 124|137) ok=0; motivo="$motivo prazo excedido (${PRAZO} s, código $rc);";; esac
  [ $rc = 2 ] || [ $ok = 0 ] || { ok=0; motivo="$motivo código $rc;"; }
  [ ! -s "$D/out" ] || { ok=0; motivo="$motivo saída padrão não vazia;"; }
  grep -qF -- "$msg" "$D/err" || { ok=0; motivo="$motivo mensagem sem '$msg';"; }
  [ $ms -lt $((PRAZO * 1000)) ] || { ok=0; motivo="$motivo ${ms} ms;"; }
  if [ -n "$porta" ] && [ "$porta" != - ] && escutando "$porta"; then ok=0; motivo="$motivo algo escuta em 127.0.0.1:$porta depois da recusa;"; fi
  if [ $ok = 1 ]; then echo "ok: $nome (código 2, ${ms} ms): $(head -n1 "$D/err")"; else echo "FALHOU: $nome:$motivo erro=[$(head -n1 "$D/err")]"; falhas=$((falhas+1)); fi
}
P=$(porta_livre) || { echo "sem porta livre" >&2; exit 2; }
# 1. argumentos
caso "1a sem --modelo" "falta --modelo" "$P" -- --porta "$P"
caso "1b sem --porta" "falta --porta" - -- --modelo "$M1"
caso "1c porta 0" "--porta fora de 1..65535" - -- --modelo "$M1" --porta 0
caso "1d porta 65536" "--porta fora de 1..65535" - -- --modelo "$M1" --porta 65536
caso "1e porta não numérica" "--porta fora de 1..65535" - -- --modelo "$M1" --porta abc
caso "1f --porta duplicado" "--porta duplicado" "$P" -- --modelo "$M1" --porta "$P" --porta "$P"
caso "1g --modelo duplicado" "--modelo duplicado" "$P" -- --modelo "$M1" --modelo "$M1" --porta "$P"
caso "1h opção de endereço não existe" "opção desconhecida: --endereco" "$P" -- --modelo "$M1" --porta "$P" --endereco 0.0.0.0
caso "1i argumento posicional" "argumento inesperado" "$P" -- --modelo "$M1" --porta "$P" oi
# 2. modelo inexistente, sem tokenizer.json, sem pesos
caso "2a modelo inexistente" "modelo inexistente" "$P" -- --modelo "$D/nao-existe" --porta "$P"
mkdir "$D/m"; echo '{}' > "$D/m/config.json"
caso "2b modelo sem tokenizer.json" "sem tokenizer.json" "$P" -- --modelo "$D/m" --porta "$P"
echo '{}' > "$D/m/tokenizer.json"
caso "2c modelo sem pesos" "sem pesos" "$P" -- --modelo "$D/m" --porta "$P"
# 3. porta ocupada (modelo real, antes dos portões e do device): um ocupante de teste escuta em 127.0.0.1:Q
Q=$(porta_livre) || { echo "sem porta livre" >&2; exit 2; }
python3 -I -c 'import socket, sys, time
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", int(sys.argv[1]))); s.listen(1); time.sleep(120)' "$Q" &
OCUPANTE=$!
for _ in $(seq 50); do escutando "$Q" && break; sleep 0.1; done
escutando "$Q" || { echo "FALHOU: ocupante de teste não escutou em $Q"; falhas=$((falhas+1)); }
caso "3a porta ocupada" "indisponível em 127.0.0.1" - -- --modelo "$M1" --porta "$Q"
limpa
for _ in $(seq 30); do escutando "$Q" || break; sleep 0.1; done
if escutando "$Q"; then echo "FALHOU: 3b porta $Q ainda escuta depois de derrubar o ocupante"; falhas=$((falhas+1)); else echo "ok: 3b nada escuta em $Q depois do caso"; fi
# 4. modelo e porta válidos, mas SEM portão: recusa (código 2) antes de abrir o device; a porta não fica escutando
caso "4a sem portão (PRIMEIRA_PALAVRA_AUTORIZADA)" "PRIMEIRA_PALAVRA_AUTORIZADA" "$P" -- --modelo "$M1" --porta "$P"
# 4b. portão ABERTO mas UUID de outra GPU: recusa (código 2) antes do device, mensagem do UUID, porta livre
CASO_CUDA=GPU-00000000-0000-0000-0000-000000000000 caso "4b UUID de outra GPU (portão aberto)" "CUDA_VISIBLE_DEVICES tem de ser" "$P" -- --modelo "$M1" --porta "$P"
# nenhum servidor remanescente
if pgrep -f -- "$BIN --modelo" > /dev/null; then echo "FALHOU: processo $(basename "$BIN") remanescente"; falhas=$((falhas+1)); else echo "ok: nenhum processo $(basename "$BIN") remanescente"; fi
rm -f "$D/out" "$D/err" "$D/m/config.json" "$D/m/tokenizer.json"; rmdir "$D/m" "$D"
if [ $falhas = 0 ]; then echo "RECUSAS OK (todas sem GPU)"; exit 0; else echo "$falhas FALHA(S)"; exit 1; fi
