#!/usr/bin/env bash
# Comparação Lastro x llama.cpp (criterio-comparacao-llamacpp-20261108.md, emenda 1 e E9): UMA célula do Lastro (modelo, texto, estado), com GPU.
#
#   roda-cmp-lastro.sh --modelo 1b|3b --texto curto|medio|longo --estado frio|quente --n N --saida DIR [--bin BINARIO]
#
#   frio:   `primeira-palavra --gera --so-texto T` em processo NOVO por medida: p0 = 1 processo DESCARTADO (o "aquecimento" do frio) + p1..pN medidas.
#   quente: UM processo `--gera --so-texto T --repete N` (1 aquecimento descartado dentro do processo + N medidas `quente rep 1..N`): p1.
# Cada processo grava a própria pasta DIR/pK/saida (nova; o harness recusa pasta existente), stdout, stderr, o tempo de parede (wall_ms.txt, para a
# carga_lastro_limite) e o código de saída (rc.txt). O medidas-<modelo>.json de cada saída é a fonte do extrator (media_por_passo_ms = soma/len).
# Antes da célula: portão do roda-f2e-monitor.sh (CPU < 10% com espera de até 20 s, GPU sem outro PID de compute, carga da GPU <= 5%); falha do portão NÃO adia nem
# repete: a célula roda e fica marcada SEM PROVA em celula.json. Durante a célula o monitor só registra (monitor.csv). Sem repetição automática: o 1º processo com
# erro encerra a célula (celula.json grava rc != 0). Parâmetros 64 novos e data 2026-11-07 NÃO são passados: o harness `--gera --so-texto` os fixa (CfgGera::prova, data_fixa_texto) e recusa
# --novos/--data fora do uso normal com --texto (primeira-palavra.rs, parse_args); r111, rev-1 P1.
# Monitor (r111, rev-1 P2): falha de inicia/para/avalia ou coleta morta antes do fim da célula fica em celula.json ("monitor_falha"); o extrator trata como SEM PROVA.
#
# Falha FECHADA (código 2) sem PRIMEIRA_PALAVRA_AUTORIZADA=1 e sem CUDA_VISIBLE_DEVICES igual ao UUID da RTX 5090 (a RTX 5090; a 3080 Ti e a 4090 nunca), ou com DIR já existente.
# Variáveis só para teste: MODELO_1B, MODELO_3B, MONITOR_SH, NVIDIA_SMI.
# Degrau 1 (criterio-degrau1-100pct-20261109.md): CMP_FRACAO_SLOTS=1.0 repassa `--fracao-slots 1.0` ao binário (única fração aceita além de "não definida" = padrão 0,10 do binário,
# sem a flag); a célula passa a ser da série L100 (celula.json: "serie": "L100", "fracao_slots": "1.0") e a linha de comando de cada processo fica em DIR/cmdline.txt (uma linha por
# processo; o stdout/stderr de cada pK guarda os `slots` impressos pelo binário, se houver). Sem a variável o comportamento é o de antes (série "L", sem flag).
#   roda-cmp-lastro.sh --smoke --modelo 1b|3b --saida DIR [--bin BINARIO]   (seção 3 do critério do degrau 1; exige CMP_FRACAO_SLOTS=1.0)
# smoke = UMA geração curta (texto curto, 64 tokens, processo novo), sem portão nem monitor de célula e sem medida; amostra a memória da RTX 5090 a ~2 Hz em DIR/gpumem.csv
# ("epoch_ms, usada, total", MiB; só leituras ok), com falhas em gpumem-falhas.txt, tempos em smoke-tempos.txt e amostra pós-saída (só descritiva) em gpumem-pos-saida.csv; grava DIR/{cmdline.txt,rc.txt,p1/...}; quem julga (código 0, GPU <= 90%, cpu/needed == 0, ids) é `cmp-extrai.py --smoke`.
# Degrau 2 (criterio-degrau2-camadas-20261109.md): CMP_CAMADAS_CPU=N repassa `--camadas-cpu N` ao binário (experts das N primeiras camadas só na CPU, os das demais todos na GPU); aceita só o valor
# FIXO do modelo (22 no 1B, 29 no 3B = o --n-cpu-moe da S2), é exclusiva com CMP_FRACAO_SLOTS e a célula passa a ser da série LC (celula.json: "serie": "LC", "camadas_cpu": "N").
#   roda-cmp-lastro.sh --smoke ... com CMP_CAMADAS_CPU (smoke do degrau 2, seção 3 do critério; julgado por `cmp-extrai.py --smoke --degrau2`).
# O roda-cmp-janela.sh recusa CMP_CAMADAS_CPU no ambiente do modo real e só a define por dentro, no --degrau2.
# Códigos: 0 a célula rodou até o fim (a validade fica em celula.json e no extrator), 1 algum processo saiu com erro, 2 recusa de uso/ambiente, inclusive recusa (código 2)
# do binário (emenda 5, item 11: o envelope interrompe a janela e não inicia células posteriores); celula.json é gravado mesmo assim.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
UUID=GPU-00005090-0000-0000-0000-000000005090
BIN=$AQUI/../../rust/moe-rt/target/release/primeira-palavra
MONITOR=${MONITOR_SH:-$AQUI/roda-f2e-monitor.sh}
M1=${MODELO_1B:-$HOME/models/granite-3.1-1b-a400m-instruct}
M3=${MODELO_3B:-$HOME/models/granite-3.1-3b-a800m-instruct}
uso() { echo "uso: roda-cmp-lastro.sh --modelo 1b|3b --texto curto|medio|longo --estado frio|quente --n N --saida DIR [--bin BINARIO] | --smoke --modelo 1b|3b --saida DIR [--bin BINARIO]" >&2; exit 2; }
recusa() { echo "roda-cmp-lastro: $*" >&2; exit 2; }
MODELO=; TEXTO=; ESTADO=; N=; SAIDA=; SMOKE=0
while [ $# -gt 0 ]; do
  case $1 in
    --modelo) [ $# -ge 2 ] || uso; MODELO=$2; shift 2;;
    --texto) [ $# -ge 2 ] || uso; TEXTO=$2; shift 2;;
    --estado) [ $# -ge 2 ] || uso; ESTADO=$2; shift 2;;
    --n) [ $# -ge 2 ] || uso; N=$2; shift 2;;
    --saida) [ $# -ge 2 ] || uso; SAIDA=$2; shift 2;;
    --bin) [ $# -ge 2 ] || uso; BIN=$2; shift 2;;
    --smoke) SMOKE=1; shift;;
    *) uso;;
  esac
done
# degrau 1: fração de slots (única aceita: 1.0)
FRACAO_ARGS=(); SERIE=L; FRACAO_REG=
if [ "${CMP_FRACAO_SLOTS+x}" = x ]; then
  [ "$CMP_FRACAO_SLOTS" = 1.0 ] || recusa "CMP_FRACAO_SLOTS só aceita 1.0 (degrau 1); veio '$CMP_FRACAO_SLOTS'"
  FRACAO_ARGS=(--fracao-slots 1.0); SERIE=L100; FRACAO_REG=1.0
fi
if [ "${CMP_CAMADAS_CPU+x}" = x ]; then
  [ -z "$FRACAO_REG" ] || recusa "CMP_CAMADAS_CPU e CMP_FRACAO_SLOTS são exclusivas"
fi
if [ $SMOKE = 1 ]; then
  [ "$SERIE" = L100 ] || [ "${CMP_CAMADAS_CPU+x}" = x ] || recusa "--smoke exige CMP_FRACAO_SLOTS=1.0 (degrau 1) ou CMP_CAMADAS_CPU (degrau 2)"
  [ -z "$TEXTO$ESTADO$N" ] || uso
  TEXTO=curto; ESTADO=frio; N=1
fi
case $MODELO in 1b) MDIR=$M1;; 3b) MDIR=$M3;; *) uso;; esac
# degrau 2: camadas na CPU (valor fixo por modelo)
CAMADAS_REG=
if [ "${CMP_CAMADAS_CPU+x}" = x ]; then
  case $MODELO in 1b) NFIXO=22;; 3b) NFIXO=29;; esac
  [ "$CMP_CAMADAS_CPU" = "$NFIXO" ] || recusa "CMP_CAMADAS_CPU só aceita $NFIXO no modelo $MODELO (degrau 2); veio '$CMP_CAMADAS_CPU'"
  FRACAO_ARGS=(--camadas-cpu "$NFIXO"); SERIE=LC; CAMADAS_REG=$NFIXO
fi
case $TEXTO in curto|medio|longo) ;; *) uso;; esac
case $ESTADO in frio|quente) ;; *) uso;; esac
case $N in ''|*[!0-9]*) uso;; esac
[ "$N" -ge 1 ] && [ "$N" -le 50 ] || recusa "--n fora de 1..50: $N"
[ -n "$SAIDA" ] || uso
[ "${PRIMEIRA_PALAVRA_AUTORIZADA:-}" = 1 ] || recusa "portão fechado: sem PRIMEIRA_PALAVRA_AUTORIZADA=1"
[ "${CUDA_VISIBLE_DEVICES:-}" = "$UUID" ] || recusa "CUDA_VISIBLE_DEVICES tem de ser $UUID (a RTX 5090; a 3080 Ti e a 4090 nunca); veio '${CUDA_VISIBLE_DEVICES:-}'"
[ -x "$BIN" ] || recusa "binário não executável: $BIN"
[ -x "$MONITOR" ] || recusa "monitor não executável: $MONITOR"
if [ -e "$SAIDA" ]; then recusa "$SAIDA já existe: pasta de saída nova"; fi
mkdir -p "$SAIDA" || recusa "não consegui criar $SAIDA"
SAIDA=$(cd "$SAIDA" && pwd -P)
ms_agora() { echo $(( $(date +%s%N) / 1000000 )); }

if [ $SMOKE = 1 ]; then
  # smoke do degrau 1: uma geração curta com --fracao-slots 1.0; sem portão, sem monitor de célula, sem medida publicada
  SMI=${NVIDIA_SMI:-nvidia-smi}
  mkdir -p "$SAIDA/p1"
  CMD=("$BIN" --gera --celula "$MODELO:$MDIR" --so-texto curto "${FRACAO_ARGS[@]}" --saida "$SAIDA/p1/saida")
  printf '%q ' "${CMD[@]}" > "$SAIDA/cmdline.txt"; echo >> "$SAIDA/cmdline.txt"
  # memória a ~2 Hz com carimbo: gpumem.csv = "epoch_ms, usada, total" (só leituras com nvidia-smi rc 0); falhas da coleta em gpumem-falhas.txt ("epoch_ms rc=N");
  # smoke-tempos.txt = início/fim do binário e do coletor (epoch ms). A amostra depois da saída do binário vai para gpumem-pos-saida.csv (descritiva, o extrator nunca a usa para aprovar).
  : > "$SAIDA/gpumem.csv"; : > "$SAIDA/gpumem-falhas.txt"
  AMOSTRA=
  para_coletor() { [ -z "$AMOSTRA" ] || { kill "$AMOSTRA" 2> /dev/null; wait "$AMOSTRA" 2> /dev/null; AMOSTRA=; echo "coletor_fim_ms=$(ms_agora)" >> "$SAIDA/smoke-tempos.txt"; }; }
  trap para_coletor EXIT
  trap 'para_coletor; exit 143' INT TERM HUP
  echo "coletor_inicio_ms=$(ms_agora)" > "$SAIDA/smoke-tempos.txt"
  ( while :; do
      leit=$("$SMI" -i "$UUID" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2> /dev/null); r=$?
      agora=$(ms_agora)
      if [ $r = 0 ] && [ -n "$leit" ]; then echo "$agora, $leit" >> "$SAIDA/gpumem.csv"; else echo "$agora rc=$r" >> "$SAIDA/gpumem-falhas.txt"; fi
      sleep 0.5
    done ) &
  AMOSTRA=$!
  t0=$(ms_agora); echo "bin_inicio_ms=$t0" >> "$SAIDA/smoke-tempos.txt"
  "${CMD[@]}" > "$SAIDA/p1/stdout.txt" 2> "$SAIDA/p1/stderr.txt" < /dev/null; rc=$?
  t1=$(ms_agora); echo "bin_fim_ms=$t1" >> "$SAIDA/smoke-tempos.txt"
  para_coletor
  { echo "$(ms_agora), $("$SMI" -i "$UUID" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2> /dev/null)"; } > "$SAIDA/gpumem-pos-saida.csv"
  echo $((t1 - t0)) > "$SAIDA/p1/wall_ms.txt"; echo "$rc" > "$SAIDA/p1/rc.txt"; echo "$rc" > "$SAIDA/rc.txt"
  if [ -n "$CAMADAS_REG" ]; then printf '{"lado": "lastro", "serie": "LC", "smoke": true, "modelo": "%s", "camadas_cpu": "%s", "rc": %s}\n' "$MODELO" "$CAMADAS_REG" "$rc" > "$SAIDA/smoke.json"
  else printf '{"lado": "lastro", "serie": "L100", "smoke": true, "modelo": "%s", "fracao_slots": "1.0", "rc": %s}\n' "$MODELO" "$rc" > "$SAIDA/smoke.json"; fi
  [ "$rc" != 2 ] || exit 2
  [ "$rc" = 0 ] || exit 1
  exit 0
fi

# portão antes da célula (não adia, não repete)
INI=$(date +%s)
"$MONITOR" portao --saida "$SAIDA/portao.txt"; PORTAO_RC=$?
PORTAO_MOTIVO=$(grep '^portao=FALHA' "$SAIDA/portao.txt" 2> /dev/null | head -n 1 | cut -d' ' -f2- | tr -d '"\\' | tr '\n' ' ')
# amostragem a 1 Hz só registra
MON_FALHAS=()
"$MONITOR" inicia --csv "$SAIDA/monitor.csv" --pid-arq "$SAIDA/monitor.pid"; r=$?
[ $r = 0 ] || MON_FALHAS+=("inicia rc=$r")
INI_COLETA=$(date +%s)   # emenda 7: marco depois do portão e logo antes da 1ª partida; o extrator exige a 1ª amostra a <= 5 s dele
SAI_RC=0
cleanup() { "$MONITOR" para --pid-arq "$SAIDA/monitor.pid" 2> /dev/null; r_para=$?; }
trap cleanup EXIT
trap 'cleanup; exit 143' INT TERM HUP

roda_processo() { # K [--repete N]  (K = nome da pasta pK)
  local k=$1; shift
  local pd="$SAIDA/p$k" t0 t1 rc
  mkdir -p "$pd"
  t0=$(ms_agora)
  { printf 'p%s: ' "$k"; printf '%q ' "$BIN" --gera --celula "$MODELO:$MDIR" --so-texto "$TEXTO" "${FRACAO_ARGS[@]}" "$@" --saida "$pd/saida"; echo; } >> "$SAIDA/cmdline.txt"
  "$BIN" --gera --celula "$MODELO:$MDIR" --so-texto "$TEXTO" "${FRACAO_ARGS[@]}" "$@" --saida "$pd/saida" > "$pd/stdout.txt" 2> "$pd/stderr.txt" < /dev/null
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
# emenda 8: mesma amostra pmon depois da célula (só registra; o desktop divide a RTX 5090 nos dois lados)
"$MONITOR" portao-depois --saida "$SAIDA/portao-depois.txt" > /dev/null 2>&1 || true
[ "$r_para" = 0 ] || MON_FALHAS+=("para rc=$r_para")
"$MONITOR" avalia --csv "$SAIDA/monitor.csv" > "$SAIDA/monitor-resumo.txt" 2>&1; r=$?
# avalia sai 1 também para > 80 C (o extrator lê o csv); aqui só a falha da própria avaliação: uso (2) ou sem amostras
if [ $r -gt 1 ] || grep -q '^sem amostras' "$SAIDA/monitor-resumo.txt" 2> /dev/null; then MON_FALHAS+=("avalia rc=$r"); fi
MON_FALHA=$(printf '%s; ' "${MON_FALHAS[@]:-}" | tr -d '"\\' | sed -e 's/; $//')
{
  printf '{"lado": "lastro", "serie": "%s", "fracao_slots": "%s", "modelo": "%s", "texto": "%s", "estado": "%s", "n_medidas": %s, "portao_ok": %s, "portao_motivo": "%s", "rc": %s, "inicio_epoch": %s, "inicio_coleta_epoch": %s, "fim_epoch": %s, "monitor_falha": "%s"%s}\n' \
    "$SERIE" "$FRACAO_REG" "$MODELO" "$TEXTO" "$ESTADO" "$N" "$([ $PORTAO_RC = 0 ] && echo true || echo false)" "$PORTAO_MOTIVO" "$SAI_RC" "$INI" "$INI_COLETA" "$FIM" "$MON_FALHA" "$([ -z "$CAMADAS_REG" ] || printf ', "camadas_cpu": "%s"' "$CAMADAS_REG")"
} > "$SAIDA/celula.json"
# emenda 8/9: pico do pmon (soma de sm% dos PIDs de tipo G, desktop) antes e depois, mais o sm máximo do dmon antes (só informação), para o relatório (null = sem leitura)
desk_n() { local v; v=$(grep -h "^$3_$2_$4=" "$SAIDA/$1" 2> /dev/null | head -n 1 | cut -d= -f2); case $v in ''|*[!0-9]*) echo null;; *) echo "$v";; esac; }
sed -i "\$ s/}\$/, \"desktop_g_sm_antes\": $(desk_n portao.txt antes pmon g_soma_sm_max), \"desktop_g_sm_depois\": $(desk_n portao-depois.txt depois pmon g_soma_sm_max), \"dmon_sm_antes\": $(desk_n portao.txt antes dmon sm_max)}/" "$SAIDA/celula.json"
[ "$SAI_RC" != 2 ] || exit 2
[ "$SAI_RC" = 0 ] || exit 1
exit 0
