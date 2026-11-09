#!/usr/bin/env bash
# Comparação Lastro x llama.cpp (criterio-comparacao-llamacpp-20261108.md, emenda 1 e E9): UMA célula do Lastro (modelo, texto, estado), com GPU.
#
#   roda-cmp-lastro.sh --modelo 1b|3b --texto curto|medio|longo --estado frio|quente --n N --saida DIR [--bin BINARIO]
#
#   frio:   `primeira-palavra --gera --so-texto T` em processo NOVO por medida: p0 = 1 processo DESCARTADO (o "aquecimento" do frio) + p1..pN medidas.
#   quente: UM processo `--gera --so-texto T --repete N` (1 aquecimento descartado dentro do processo + N medidas `quente rep 1..N`): p1.
# Cada processo grava a própria pasta DIR/pK/saida (nova; o harness recusa pasta existente), stdout, stderr, o tempo de parede (wall_ms.txt, para a
# carga_lastro_limite) e o código de saída (rc.txt). O medidas-<modelo>.json de cada saída é a fonte do extrator (media_por_passo_ms = soma/len).
# Antes da célula: portão do roda-cmp-monitor.sh (CPU < 10% com espera de até 20 s, GPU sem outro PID de compute, carga da GPU <= 5%); falha do portão NÃO adia nem
# repete: a célula roda e fica marcada SEM PROVA em celula.json. Durante a célula o monitor só registra (monitor.csv). Sem repetição automática: o 1º processo com
# erro encerra a célula (celula.json grava rc != 0). Parâmetros 64 novos e data 2026-11-07 NÃO são passados: o harness `--gera --so-texto` os fixa (CfgGera::prova, data_fixa_texto) e recusa
# --novos/--data fora do uso normal com --texto (primeira-palavra.rs, parse_args); r111, rev-1 P1.
# Monitor (r111, rev-1 P2): falha de inicia/para/avalia ou coleta morta antes do fim da célula fica em celula.json ("monitor_falha"); o extrator trata como SEM PROVA.
#
# Falha FECHADA (código 2) sem PRIMEIRA_PALAVRA_AUTORIZADA=1 e sem CUDA_VISIBLE_DEVICES igual ao UUID da 4090 (a 5090 nunca), ou com DIR já existente.
# Variáveis só para teste: MODELO_1B, MODELO_3B, MONITOR_SH, NVIDIA_SMI.
# Códigos: 0 a célula rodou até o fim (a validade fica em celula.json e no extrator), 1 algum processo saiu com erro, 2 recusa de uso/ambiente, inclusive recusa (código 2)
# do binário (emenda 5, item 11: o envelope interrompe a janela e não inicia células posteriores); celula.json é gravado mesmo assim.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
UUID=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd
BIN=$AQUI/../../rust/moe-rt/target/release/primeira-palavra
MONITOR=${MONITOR_SH:-$AQUI/roda-cmp-monitor.sh}
M1=${MODELO_1B:-/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct}
M3=${MODELO_3B:-/home/brenoperucchi/models/granite-3.1-3b-a800m-instruct}
uso() { echo "uso: roda-cmp-lastro.sh --modelo 1b|3b --texto curto|medio|longo --estado frio|quente --n N --saida DIR [--bin BINARIO]" >&2; exit 2; }
recusa() { echo "roda-cmp-lastro: $*" >&2; exit 2; }
MODELO=; TEXTO=; ESTADO=; N=; SAIDA=
while [ $# -gt 0 ]; do
  case $1 in
    --modelo) [ $# -ge 2 ] || uso; MODELO=$2; shift 2;;
    --texto) [ $# -ge 2 ] || uso; TEXTO=$2; shift 2;;
    --estado) [ $# -ge 2 ] || uso; ESTADO=$2; shift 2;;
    --n) [ $# -ge 2 ] || uso; N=$2; shift 2;;
    --saida) [ $# -ge 2 ] || uso; SAIDA=$2; shift 2;;
    --bin) [ $# -ge 2 ] || uso; BIN=$2; shift 2;;
    *) uso;;
  esac
done
case $MODELO in 1b) MDIR=$M1;; 3b) MDIR=$M3;; *) uso;; esac
case $TEXTO in curto|medio|longo) ;; *) uso;; esac
case $ESTADO in frio|quente) ;; *) uso;; esac
case $N in ''|*[!0-9]*) uso;; esac
[ "$N" -ge 1 ] && [ "$N" -le 50 ] || recusa "--n fora de 1..50: $N"
[ -n "$SAIDA" ] || uso
[ "${PRIMEIRA_PALAVRA_AUTORIZADA:-}" = 1 ] || recusa "portão fechado: sem PRIMEIRA_PALAVRA_AUTORIZADA=1"
[ "${CUDA_VISIBLE_DEVICES:-}" = "$UUID" ] || recusa "CUDA_VISIBLE_DEVICES tem de ser $UUID (a 4090); veio '${CUDA_VISIBLE_DEVICES:-}'"
[ -x "$BIN" ] || recusa "binário não executável: $BIN"
[ -x "$MONITOR" ] || recusa "monitor não executável: $MONITOR"
if [ -e "$SAIDA" ]; then recusa "$SAIDA já existe: pasta de saída nova"; fi
mkdir -p "$SAIDA" || recusa "não consegui criar $SAIDA"
SAIDA=$(cd "$SAIDA" && pwd -P)
ms_agora() { echo $(( $(date +%s%N) / 1000000 )); }

# portão antes da célula (não adia, não repete)
INI=$(date +%s)
"$MONITOR" portao --saida "$SAIDA/portao.txt"; PORTAO_RC=$?
PORTAO_MOTIVO=$(grep '^portao=FALHA' "$SAIDA/portao.txt" 2> /dev/null | head -n 1 | cut -d' ' -f2- | tr -d '"\\' | tr '\n' ' ')
# amostragem a 1 Hz só registra
MON_FALHAS=()
"$MONITOR" inicia --csv "$SAIDA/monitor.csv" --pid-arq "$SAIDA/monitor.pid"; r=$?
[ $r = 0 ] || MON_FALHAS+=("inicia rc=$r")
SAI_RC=0
cleanup() { "$MONITOR" para --pid-arq "$SAIDA/monitor.pid" 2> /dev/null; r_para=$?; }
trap cleanup EXIT
trap 'cleanup; exit 143' INT TERM HUP

roda_processo() { # K [--repete N]  (K = nome da pasta pK)
  local k=$1; shift
  local pd="$SAIDA/p$k" t0 t1 rc
  mkdir -p "$pd"
  t0=$(ms_agora)
  "$BIN" --gera --celula "$MODELO:$MDIR" --so-texto "$TEXTO" "$@" --saida "$pd/saida" > "$pd/stdout.txt" 2> "$pd/stderr.txt" < /dev/null
  rc=$?
  t1=$(ms_agora)
  echo $((t1 - t0)) > "$pd/wall_ms.txt"
  echo "$rc" > "$pd/rc.txt"
  return $rc
}
if [ "$ESTADO" = frio ]; then
  for k in $(seq 0 "$N"); do
    roda_processo "$k" || { SAI_RC=$?; break; }
  done
else
  roda_processo 1 --repete "$N" || SAI_RC=$?
fi
# a coleta tem de estar viva até o fim da célula (uma amostra inicial seguida de morte do monitor não vale)
MON_PID=$(cat "$SAIDA/monitor.pid" 2> /dev/null)
{ [ -n "$MON_PID" ] && kill -0 "$MON_PID" 2> /dev/null; } || MON_FALHAS+=("coleta encerrada antes do fim da célula")
FIM=$(date +%s)
r_para=0
cleanup
[ "$r_para" = 0 ] || MON_FALHAS+=("para rc=$r_para")
"$MONITOR" avalia --csv "$SAIDA/monitor.csv" > "$SAIDA/monitor-resumo.txt" 2>&1; r=$?
# avalia sai 1 também para > 80 C (o extrator lê o csv); aqui só a falha da própria avaliação: uso (2) ou sem amostras
if [ $r -gt 1 ] || grep -q '^sem amostras' "$SAIDA/monitor-resumo.txt" 2> /dev/null; then MON_FALHAS+=("avalia rc=$r"); fi
MON_FALHA=$(printf '%s; ' "${MON_FALHAS[@]:-}" | tr -d '"\\' | sed -e 's/; $//')
{
  printf '{"lado": "lastro", "serie": "L", "modelo": "%s", "texto": "%s", "estado": "%s", "n_medidas": %s, "portao_ok": %s, "portao_motivo": "%s", "rc": %s, "inicio_epoch": %s, "fim_epoch": %s, "monitor_falha": "%s"}\n' \
    "$MODELO" "$TEXTO" "$ESTADO" "$N" "$([ $PORTAO_RC = 0 ] && echo true || echo false)" "$PORTAO_MOTIVO" "$SAI_RC" "$INI" "$FIM" "$MON_FALHA"
} > "$SAIDA/celula.json"
[ "$SAI_RC" != 2 ] || exit 2
[ "$SAI_RC" = 0 ] || exit 1
exit 0
