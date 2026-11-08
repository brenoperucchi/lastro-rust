#!/usr/bin/env bash
# A2, etapa 5d: a prova com GPU do comando de linha `lastro` (criterio-5d-cli-20261108.md), em UMA janela, na composição já validada: janela-e4b.sh (pausa e
# restauração conferida dos timers) por fora, roda-com-limite.sh (limite de tempo) por dentro, e este mesmo script em modo interno.
#
#   roda-5d-janela.sh --saida DIR [--estado-janela DIR] [--bin BINARIO] [--limite S]
#
# Dentro da janela: (1) roda-5d-recusas.sh no binário (sem GPU); (2) 6 execuções, processo novo cada, sem repetição, com --data "November 07, 2026" e --novos 64:
# 1B e 3B, cada um com curto pelo argumento, médio pela entrada padrão e longo pelo argumento; compara a saída padrão, byte a byte, com texto_gerado + "\n" da
# referência HF do 5c (5c-referencia/ref5c-<modelo>.json) e confere o código 0; (3) grava DIR/decisao.txt: PASSA / REPROVA / SEM PROVA pelas regras do critério.
# Integridade: sha256 do binário, dos critérios 5d e 5c, dos 3 textos, das 2 referências e do tokenizer de cada modelo, antes e depois (e contra os valores
# congelados); quebrada = SEM PROVA em tudo. REPROVA vence SEM PROVA, SEM PROVA vence PASSA. Descritivo (sem rótulo): tempo total e tempo até o primeiro byte.
#
# Falha FECHADA antes de tocar em timers ou na GPU: PRIMEIRA_PALAVRA_AUTORIZADA=1 e CUDA_VISIBLE_DEVICES igual ao UUID da 4090 têm de estar no ambiente de quem chama
# (o "autorizo" do Breno); --saida e --estado-janela têm de ser pastas novas ou vazias (padrão do estado: DIR-estado-janela). Este roteiro NÃO é usado na entrega.
# Limite padrão 1800 s (máx. 1800); o de fora é o de dentro mais 120 s. Variáveis só para teste: JANELA_SH, LIMITE_SH, JANELA_TIMERS, MODELO_1B, MODELO_3B,
# TEXTOS_5C, REF_5C, CRITERIO_5D, CRITERIO_5C, RECUSAS_SH, SEM_SHA_FIXO=1 (pula a conferência contra os shas congelados), NVIDIA_SMI, LOADAVG_ARQ.
# Códigos: 0 se decisao.txt foi gravada (o rótulo está nela), o do comando (124 limite, etc.) ou 2 recusa de uso, portão ou ambiente.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
UUID=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd
JANELA=${JANELA_SH:-$AQUI/janela-e4b.sh}
LIMITE_SH=${LIMITE_SH:-$AQUI/roda-com-limite.sh}
RECUSAS=${RECUSAS_SH:-$AQUI/roda-5d-recusas.sh}
BIN=$AQUI/../../rust/moe-rt/target/release/lastro
M1=${MODELO_1B:-/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct}
M3=${MODELO_3B:-/home/brenoperucchi/models/granite-3.1-3b-a800m-instruct}
TEXTOS=${TEXTOS_5C:-$AQUI/5c-textos}
REFS=${REF_5C:-$AQUI/5c-referencia}
C5D=${CRITERIO_5D:-$AQUI/criterio-5d-cli-20261108.md}
C5C=${CRITERIO_5C:-$AQUI/criterio-5c-geracao-livre-20261108.md}
DATA="November 07, 2026"
SAIDA=; ESTADO=; LIMITE=1800; INTERNO=0
usa() { echo "uso: roda-5d-janela.sh --saida DIR [--estado-janela DIR] [--bin BINARIO] [--limite S]" >&2; }
recusa() { echo "roda-5d-janela: $*" >&2; exit 2; }
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

# SHAs congelados (criterio-5c, textos e referências do 5c, tokenizer do Granite 3.1, critério 5d).
declare -A FIXO=(
  [criterio5d]=acf84e6ce79b8ff7a200c6734e1fab96060c2c3b73911faa198e85980950180a
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
  [binario]=$BIN [criterio5d]=$C5D [criterio5c]=$C5C
  [texto-curto]=$TEXTOS/texto-curto.txt [texto-medio]=$TEXTOS/texto-medio.txt [texto-longo]=$TEXTOS/texto-longo.txt
  [ref5c-1b]=$REFS/ref5c-1b.json [ref5c-3b]=$REFS/ref5c-3b.json [tokenizer-1b]=$M1/tokenizer.json [tokenizer-3b]=$M3/tokenizer.json
)
CHAVES=(binario criterio5d criterio5c texto-curto texto-medio texto-longo ref5c-1b ref5c-3b tokenizer-1b tokenizer-3b)
sha() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
# coleta_shas ARQUIVO_DE_SAIDA: uma linha "chave sha" por item (vazio = ilegível).
coleta_shas() { : > "$1"; for k in "${CHAVES[@]}"; do echo "$k $(sha "${ARQ[$k]}")" >> "$1"; done; }

# ---------------------------------------------------------------- modo interno (dentro da janela, com o portão e o UUID no ambiente)
if [ $INTERNO = 1 ]; then
  [ "${PRIMEIRA_PALAVRA_AUTORIZADA:-}" = 1 ] && [ "${CUDA_VISIBLE_DEVICES:-}" = "$UUID" ] || recusa "modo interno sem portão ou UUID"
  mkdir -p "$SAIDA" || exit 2
  coleta_shas "$SAIDA/integridade-antes.txt"
  # 1. recusas (sem GPU)
  "$RECUSAS" --bin "$BIN" > "$SAIDA/recusas.txt" 2>&1; RC_RECUSAS=$?
  # 2. as 6 execuções
  # roda_uma MODELO_DIR NOME(não usado) MODO(arg|stdin) ARQUIVO_TEXTO PREFIXO -> grava PREFIXO.{out,err,info}; info = "rc total_ms primeiro_ms"
  roda_uma() {
    python3 -I - "$BIN" "$1" "$3" "$4" "$5" "$DATA" <<'PY'
import os, subprocess, sys, time
bin_, modelo, modo, arq, pref, data = sys.argv[1:7]
texto = open(arq, "rb").read()
cmd = [bin_, "--modelo", modelo, "--data", data, "--novos", "64"]
if modo == "arg":
    cmd.append(texto.decode("utf-8"))
out = open(pref + ".out", "wb"); err = open(pref + ".err", "wb")
t0 = time.monotonic()
try:
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE if modo == "stdin" else subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err)
except OSError as e:
    open(pref + ".info", "w").write("NOSTART 0 0\n"); sys.exit(0)
if modo == "stdin":
    p.stdin.write(texto); p.stdin.close()
primeiro = None
while True:
    b = os.read(p.stdout.fileno(), 65536)
    if not b: break
    if primeiro is None: primeiro = (time.monotonic() - t0) * 1000
    out.write(b)
rc = p.wait()
total = (time.monotonic() - t0) * 1000
open(pref + ".info", "w").write("%d %.1f %.1f\n" % (rc, total, primeiro if primeiro is not None else -1))
PY
  }
  # esperado MODELO NOME -> escreve PREFIXO.esperado = texto_gerado + "\n" da referência
  esperado() {
    python3 -I - "$REFS/ref5c-$1.json" "$2" "$3" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
c = [x for x in d["casos"] if x["nome"] == sys.argv[2]]
assert len(c) == 1
open(sys.argv[3], "wb").write((c[0]["texto_gerado"] + "\n").encode("utf-8"))
PY
  }
  declare -A MDIR=([1b]=$M1 [3b]=$M3)
  PEDIDOS=("curto arg" "medio stdin" "longo arg")
  : > "$SAIDA/execucoes.txt"
  for m in 1b 3b; do
    for p in "${PEDIDOS[@]}"; do
      set -- $p; nome=$1; modo=$2
      pref="$SAIDA/$m-$nome"
      esperado "$m" "$nome" "$pref.esperado" 2> "$pref.esperado.err" || echo "referência ilegível" >> "$pref.esperado.err"
      roda_uma "${MDIR[$m]}" "$nome" "$modo" "$TEXTOS/texto-$nome.txt" "$pref" 2>> "$pref.err" || true
      echo "$m $nome $modo $(cat "$pref.info" 2>/dev/null || echo 'SEM_INFO 0 0')" >> "$SAIDA/execucoes.txt"
    done
  done
  coleta_shas "$SAIDA/integridade-depois.txt"
  # 3. decisão
  INTEGRO=1; NOTAS=()
  if ! cmp -s "$SAIDA/integridade-antes.txt" "$SAIDA/integridade-depois.txt"; then INTEGRO=0; NOTAS+=("integridade: sha mudou entre antes e depois"); fi
  while read -r k h; do
    [ -n "$h" ] || { INTEGRO=0; NOTAS+=("integridade: $k ilegível"); continue; }
    if [ "${SEM_SHA_FIXO:-0}" != 1 ] && [ -n "${FIXO[$k]:-}" ] && [ "$h" != "${FIXO[$k]}" ]; then INTEGRO=0; NOTAS+=("integridade: $k com sha $h, não o congelado"); fi
  done < "$SAIDA/integridade-antes.txt"
  REPROVA=0; SEMPROVA=0; LINHAS=()
  if [ $RC_RECUSAS != 0 ]; then
    if [ $RC_RECUSAS = 1 ]; then REPROVA=1; LINHAS+=("recusas: FALHARAM (roda-5d-recusas.sh saiu 1)"); else SEMPROVA=1; LINHAS+=("recusas: sem resultado (roda-5d-recusas.sh saiu $RC_RECUSAS)"); fi
  else LINHAS+=("recusas: OK"); fi
  IGUAIS=0
  while read -r m nome modo rc total primeiro; do
    pref="$SAIDA/$m-$nome"
    if [ "$rc" = NOSTART ] || [ "$rc" = SEM_INFO ] || [ ! -s "$pref.esperado" ]; then
      SEMPROVA=1; LINHAS+=("$m $nome ($modo): SEM PROVA (sem resultado: $rc)"); continue
    fi
    if [ "$rc" -lt 0 ] 2>/dev/null || [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
      SEMPROVA=1; LINHAS+=("$m $nome ($modo): SEM PROVA (encerrado por limite ou sinal, código $rc)"); continue
    fi
    ok=1; mot=
    [ "$rc" = 0 ] || { ok=0; mot="código $rc;"; }
    cmp -s "$pref.out" "$pref.esperado" || { ok=0; mot="$mot saída diferente da referência;"; }
    if [ $ok = 1 ]; then IGUAIS=$((IGUAIS+1)); LINHAS+=("$m $nome ($modo): PASSA (igual byte a byte, código 0; total ${total} ms, primeiro byte ${primeiro} ms)")
    else REPROVA=1; LINHAS+=("$m $nome ($modo): REPROVA ($mot total ${total} ms)"); fi
  done < "$SAIDA/execucoes.txt"
  [ "$(wc -l < "$SAIDA/execucoes.txt")" = 6 ] || { SEMPROVA=1; LINHAS+=("execuções: esperadas 6, registradas $(wc -l < "$SAIDA/execucoes.txt")"); }
  if [ $INTEGRO = 0 ]; then ROTULO="SEM PROVA"
  elif [ $REPROVA = 1 ]; then ROTULO=REPROVA
  elif [ $SEMPROVA = 1 ] || [ $IGUAIS != 6 ]; then ROTULO="SEM PROVA"
  else ROTULO=PASSA; fi
  {
    echo "etapa 5d, comando de linha (critério sha256 $(sha "$C5D"))"
    for l in "${LINHAS[@]}"; do echo "$l"; done
    echo "saídas iguais: $IGUAIS de 6"
    if [ $INTEGRO = 1 ]; then echo "integridade: íntegra (binário, critérios, textos, referências e tokenizers iguais antes e depois e aos congelados)"
    else echo "integridade: QUEBRADA: tudo SEM PROVA"; for n in "${NOTAS[@]}"; do echo "  $n"; done; fi
    echo "código 1 (falha de execução do binário) conta como 'código errado' e vira REPROVA"
    echo "o tempo até o primeiro byte sai igual ao total porque a saída é escrita de uma vez no fim; não é latência do primeiro token"
    echo "tempos (descritivos, sem rótulo): ver as linhas acima e execucoes.txt (rc, total ms, primeiro byte ms)"
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
[ -x "$JANELA" ] && [ -x "$LIMITE_SH" ] && [ -x "$RECUSAS" ] || recusa "faltam janela-e4b.sh, roda-com-limite.sh ou roda-5d-recusas.sh executáveis"
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
for v in MODELO_1B MODELO_3B TEXTOS_5C REF_5C CRITERIO_5D CRITERIO_5C RECUSAS_SH SEM_SHA_FIXO; do
  [ -z "${!v+x}" ] || PASSA+=("$v=${!v}")
done
coleta início "$SAIDA.carga-antes.txt"
"$JANELA" --estado "$ESTADO" "${TIMERS[@]}" --limite $((LIMITE + 120)) --kill-apos 30 -- \
  "$LIMITE_SH" --limite "$LIMITE" -- env "${PASSA[@]}" "$AQUI/roda-5d-janela.sh" --interno --saida "$SAIDA" --bin "$BIN"
RC=$?
coleta fim "$SAIDA.carga-depois.txt"
exit $RC
