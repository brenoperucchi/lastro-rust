#!/usr/bin/env bash
# A2, etapa 5, degrau 5b: UMA janela de GPU do degrau, na composição já validada nas etapas 3 e 4: janela-e4b.sh (pausa e restauração conferida dos
# timers de backup) por fora, roda-com-limite.sh (limite de tempo por unidade systemd) por dentro, e o binário primeira-palavra (ou o roda-5b-frio.sh).
#
#   roda-5b-janela.sh --etapa correcao|quente|frio --saida DIR --estado-janela DIR [--bin BINARIO] [--limite S]
#
#   correcao  J1: primeira-palavra --modo correcao (1b e 3b)
#   quente    J2: primeira-palavra --modo velocidade --estado quente (1b e 3b)
#   frio      J3: roda-5b-frio.sh (12 processos novos e a consolidação)
#
# Falha FECHADA antes de tocar em qualquer timer ou na GPU: o portão PRIMEIRA_PALAVRA_AUTORIZADA=1 e CUDA_VISIBLE_DEVICES igual ao UUID da 4090 combinada
# têm de estar no ambiente de quem chama (o "autorizo" do Breno, que nomeia a janela); --saida e --estado-janela têm de ser pastas novas ou vazias. O ambiente
# atravessa as duas camadas por `env`. Este script NÃO repete nada e NÃO escolhe outra GPU. A carga externa (DESCRIÇÃO, não decide rótulo) é coletada FORA da janela, como na etapa 4: antes de pausar
# os timers e depois da restauração, em SAIDA.carga-antes.txt e SAIDA.carga-depois.txt (ao lado da pasta da janela): hora, nvidia-smi (uso da GPU e processos
# de computação), loadavg e o hash do binário. Falha de coleta nunca muda o código de saída. Variáveis só para teste: NVIDIA_SMI, LOADAVG_ARQ.
# O limite de fora (janela-e4b) é o de dentro mais 120 s, para o de dentro disparar primeiro. Padrão 1800 s (30 min), máximo 1800.
# Variáveis opcionais (só para teste com sistema falso): JANELA_SH, LIMITE_SH, JANELA_TIMERS, MODELO_1B, MODELO_3B (e as do janela-e4b: SYSTEMCTL, PENDENTE), NVIDIA_SMI, LOADAVG_ARQ.
# Códigos: o do comando (via janela-e4b.sh e roda-com-limite.sh: 124 limite, 3/4/5/6/7 do roteiro de timers), 2 recusa de uso, portão ou ambiente.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
UUID=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd
JANELA=${JANELA_SH:-$AQUI/janela-e4b.sh}
LIMITE_SH=${LIMITE_SH:-$AQUI/roda-com-limite.sh}
BIN=$AQUI/../../rust/moe-rt/target/release/primeira-palavra
FRIO=$AQUI/roda-5b-frio.sh
ETAPA=; SAIDA=; ESTADO=; LIMITE=1800
usa() { echo "uso: roda-5b-janela.sh --etapa correcao|quente|frio --saida DIR --estado-janela DIR [--bin BINARIO] [--limite S]" >&2; }
recusa() { echo "roda-5b-janela: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
  case $1 in
    --etapa) [ $# -ge 2 ] || { usa; exit 2; }; ETAPA=$2; shift 2;;
    --saida) [ $# -ge 2 ] || { usa; exit 2; }; SAIDA=$2; shift 2;;
    --estado-janela) [ $# -ge 2 ] || { usa; exit 2; }; ESTADO=$2; shift 2;;
    --bin) [ $# -ge 2 ] || { usa; exit 2; }; BIN=$2; shift 2;;
    --limite) [ $# -ge 2 ] || { usa; exit 2; }; LIMITE=$2; shift 2;;
    *) usa; exit 2;;
  esac
done
case $ETAPA in correcao|quente|frio) ;; *) usa; exit 2;; esac
[ -n "$SAIDA" ] && [ -n "$ESTADO" ] || { usa; exit 2; }
case $LIMITE in ''|*[!0-9]*) recusa "limite inválido: $LIMITE";; esac
[ "$LIMITE" -ge 1 ] && [ "$LIMITE" -le 1800 ] || recusa "limite fora de 1..1800 s: $LIMITE"
[ "${PRIMEIRA_PALAVRA_AUTORIZADA:-}" = 1 ] || recusa "portão fechado: sem PRIMEIRA_PALAVRA_AUTORIZADA=1 (o 'autorizo' do Breno que nomeia a janela)"
[ "${CUDA_VISIBLE_DEVICES:-}" = "$UUID" ] || recusa "CUDA_VISIBLE_DEVICES tem de ser $UUID (a 4090 combinada); veio '${CUDA_VISIBLE_DEVICES:-}'"
[ -x "$BIN" ] || recusa "binário não executável: $BIN"
[ -x "$JANELA" ] && [ -x "$LIMITE_SH" ] || recusa "faltam janela-e4b.sh ou roda-com-limite.sh executáveis"
for d in "$SAIDA" "$ESTADO"; do
  if [ -e "$d" ]; then
    [ -d "$d" ] && [ -z "$(ls -A "$d")" ] || recusa "$d já existe e não está vazio: uma janela nova pede pasta nova"
  fi
done
M1=${MODELO_1B:-/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct}
M3=${MODELO_3B:-/home/brenoperucchi/models/granite-3.1-3b-a800m-instruct}
case $ETAPA in
  correcao) CMD=("$BIN" --modo correcao --celula "1b:$M1" --celula "3b:$M3" --saida "$SAIDA");;
  quente) CMD=("$BIN" --modo velocidade --estado quente --celula "1b:$M1" --celula "3b:$M3" --saida "$SAIDA");;
  frio) CMD=("$FRIO" --saida "$SAIDA" --bin "$BIN");;
esac
TIMERS=()
[ -z "${JANELA_TIMERS:-}" ] || TIMERS=(--timers "$JANELA_TIMERS")
# Carga externa, só descrição: nunca interrompe nem muda o código de saída.
coleta() {
  {
    echo "$1 $(date '+%d/%m %H:%M:%S')"
    "${NVIDIA_SMI:-nvidia-smi}" --query-gpu=uuid,memory.used,utilization.gpu --format=csv 2>&1 || echo "nvidia-smi indisponível"
    "${NVIDIA_SMI:-nvidia-smi}" --query-compute-apps=pid,used_memory,name --format=csv,noheader 2>&1 || true
    echo "loadavg $(cat "${LOADAVG_ARQ:-/proc/loadavg}" 2>&1)"
    echo "binário $(sha256sum "$BIN" 2>&1 | cut -d' ' -f1)"
  } > "$2" 2>&1 || true
}
coleta início "$SAIDA.carga-antes.txt"
"$JANELA" --estado "$ESTADO" "${TIMERS[@]}" --limite $((LIMITE + 120)) --kill-apos 30 -- \
  "$LIMITE_SH" --limite "$LIMITE" -- env "CUDA_VISIBLE_DEVICES=$UUID" PRIMEIRA_PALAVRA_AUTORIZADA=1 "${CMD[@]}"
RC=$?
coleta fim "$SAIDA.carga-depois.txt"
exit $RC
