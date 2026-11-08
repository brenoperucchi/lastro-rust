#!/usr/bin/env bash
# A2, etapa 5d: as 4 recusas do critério (criterio-5d-cli-20261108.md) no binário REAL `lastro`, SEM GPU: portão PRIMEIRA_PALAVRA_AUTORIZADA fora do
# ambiente e CUDA_VISIBLE_DEVICES vazio. Cada caso confere código de saída (2), saída padrão vazia, mensagem na saída de erro e tempo < 5 s.
#
#   roda-5d-recusas.sh [--bin BINARIO]      (padrão: rust/moe-rt/target/release/lastro; MODELO_1B troca o modelo real usado no caso do texto grande)
#
# Código: 0 todas OK; 1 alguma falhou; 2 uso. Cria uma pasta com mktemp -d e remove só os arquivos que criou (rm de cada um e rmdir), nunca rm -rf.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
BIN=$AQUI/../../rust/moe-rt/target/release/lastro
if [ $# -ge 1 ]; then
  [ "$1" = --bin ] && [ $# -eq 2 ] || { echo "uso: roda-5d-recusas.sh [--bin BINARIO]" >&2; exit 2; }
  BIN=$2
fi
[ -x "$BIN" ] || { echo "binário não executável: $BIN" >&2; exit 2; }
M1=${MODELO_1B:-/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct}
D=$(mktemp -d -p "${TMPDIR:-/tmp}" recusas-5d.XXXXXX) || exit 2
falhas=0
PRAZO=${RECUSAS_PRAZO:-5}   # prazo individual por caso, em s (o critério fixa 5; a variável existe só para o teste do prazo)
# caso NOME 'trecho da mensagem' [--stdin ARQUIVO] -- ARGS...
caso() {
  local nome=$1 msg=$2 entrada=/dev/null; shift 2
  if [ "$1" = --stdin ]; then entrada=$2; shift 2; fi
  [ "$1" = -- ] && shift
  local t0 t1 rc ms
  t0=$(date +%s%N)
  # timeout põe o caso em grupo de processos próprio e mata o grupo inteiro (descendentes) no prazo; --kill-after encerra quem ignorar o TERM.
  if [ -p "$entrada" ]; then
    env -u PRIMEIRA_PALAVRA_AUTORIZADA CUDA_VISIBLE_DEVICES= timeout --kill-after=1 "$PRAZO" "$BIN" "$@" 0<> "$entrada" > "$D/out" 2> "$D/err"
  else
    env -u PRIMEIRA_PALAVRA_AUTORIZADA CUDA_VISIBLE_DEVICES= timeout --kill-after=1 "$PRAZO" "$BIN" "$@" < "$entrada" > "$D/out" 2> "$D/err"
  fi
  rc=$?
  t1=$(date +%s%N); ms=$(( (t1 - t0) / 1000000 ))
  local ok=1 motivo=
  case $rc in 124|137) ok=0; motivo="$motivo prazo excedido (${PRAZO} s, código $rc);";; esac
  [ $rc = 2 ] || [ $ok = 0 ] || { ok=0; motivo="$motivo código $rc;"; }
  [ ! -s "$D/out" ] || { ok=0; motivo="$motivo saída padrão não vazia;"; }
  grep -qF -- "$msg" "$D/err" || { ok=0; motivo="$motivo mensagem sem '$msg';"; }
  [ $ms -lt 5000 ] || { ok=0; motivo="$motivo ${ms} ms;"; }
  if [ $ok = 1 ]; then echo "ok: $nome (código 2, ${ms} ms): $(head -n1 "$D/err")"; else echo "FALHOU: $nome:$motivo erro=[$(head -n1 "$D/err")]"; falhas=$((falhas+1)); fi
}
# 1. entrada vazia ou só espaços (argumento e entrada padrão)
caso "1a vazio pelo argumento" "entrada vazia ou só espaços" -- --modelo "$M1" ""
caso "1b só espaços pelo argumento" "entrada vazia ou só espaços" -- --modelo "$M1" "   "
printf ' \n\t\n' > "$D/espacos"; : > "$D/vazio"
caso "1c só espaços pela entrada padrão" "entrada vazia ou só espaços" --stdin "$D/espacos" -- --modelo "$M1"
caso "1d entrada padrão vazia" "entrada vazia ou só espaços" --stdin "$D/vazio" -- --modelo "$M1"
# 2. prompt + novos > 2048 (conta pelo tokenizador)
for i in $(seq 3000); do printf 'word '; done > "$D/grande"
caso "2a texto grande, 64 novos" "passam de 2048 posições" --stdin "$D/grande" -- --modelo "$M1"
caso "2b texto grande, 1 novo" "passam de 2048 posições" --stdin "$D/grande" -- --modelo "$M1" --novos 1
# 3. modelo inexistente, sem tokenizer.json, sem pesos
caso "3a modelo inexistente" "modelo inexistente" -- --modelo "$D/nao-existe" oi
mkdir "$D/m"; echo '{}' > "$D/m/config.json"
# 3d: modelo inexistente SEM texto e com a entrada padrão aberta (FIFO sem escritor nem EOF): a recusa vem antes de ler a entrada.
mkfifo "$D/aberta"
caso "3d modelo inexistente, sem texto, entrada aberta" "modelo inexistente" --stdin "$D/aberta" -- --modelo "$D/nao-existe"
caso "3b modelo sem tokenizer.json" "sem tokenizer.json" -- --modelo "$D/m" oi
echo '{}' > "$D/m/tokenizer.json"
caso "3c modelo sem pesos" "sem pesos" -- --modelo "$D/m" oi
# 4. --novos fora de 1..64, opção desconhecida, argumento duplicado
caso "4a --novos 0" "--novos fora de 1..64" -- --modelo "$M1" --novos 0 oi
caso "4b --novos 65" "--novos fora de 1..64" -- --modelo "$M1" --novos 65 oi
caso "4c opção desconhecida" "opção desconhecida: --x" -- --modelo "$M1" --x oi
caso "4d --modelo duplicado" "--modelo duplicado" -- --modelo "$M1" --modelo "$M1" oi
caso "4e --novos duplicado" "--novos duplicado" -- --modelo "$M1" --novos 3 --novos 4 oi
caso "4f dois textos" "texto duplicado" -- --modelo "$M1" um dois
caso "4g sem --modelo" "falta --modelo" -- oi
caso "4h --data inválida" "Month DD, YYYY" -- --modelo "$M1" --data 2026-11-07 oi
rm -f "$D/out" "$D/err" "$D/espacos" "$D/vazio" "$D/grande" "$D/aberta" "$D/m/config.json" "$D/m/tokenizer.json"; rmdir "$D/m" "$D"
if [ $falhas = 0 ]; then echo "RECUSAS OK (todas sem GPU)"; exit 0; else echo "$falhas FALHA(S)"; exit 1; fi
