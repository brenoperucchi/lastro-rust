#!/usr/bin/env bash
# Comparação Lastro x llama.cpp (criterio-comparacao-llamacpp-20261108.md, E5 e emenda 3): portão ANTES de cada célula e amostragem nvidia-smi a 1 Hz DURANTE a célula.
#
#   roda-cmp-monitor.sh portao  --saida ARQ [--arvore PID]   portão antes da célula (código 0 = passou, 1 = a célula é SEM PROVA, 2 = uso)
#   roda-cmp-monitor.sh inicia  --csv ARQ --pid-arq ARQ      amostragem a 1 Hz em segundo plano (SÓ REGISTRA: nada derruba a célula por carga)
#   roda-cmp-monitor.sh para    --pid-arq ARQ                encerra a amostragem
#   roda-cmp-monitor.sh avalia  --csv ARQ                    temperatura máxima das amostras (código 1 se > 80 C: SEM PROVA); clocks só registrados
#
# Portão (emenda 3 > E5 > item (4)): (a) ocupação instantânea da CPU da máquina em janela de 3 s (/proc/stat) < 10%, com espera ativa de até 20 s; se não baixar,
# a célula é SEM PROVA (não adiada, não repetida); (b) `nvidia-smi --query-compute-apps` sem nenhum PID fora da árvore medida (--arvore PID: esse PID e seus
# descendentes são o lado medido; antes da partida da célula ninguém está medido); (c) carga da GPU <= 5%. O loadavg só é registrado.
# O portão grava o que mediu em ARQ (linhas chave=valor) e uma linha final `portao=OK` ou `portao=FALHA motivo`.
# Variáveis só para teste: NVIDIA_SMI, PROC_STAT, CMP_CPU_JANELA (s, padrão 3), CMP_CPU_ESPERA (s, padrão 20), LOADAVG_ARQ.
set -u
UUID=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd
SMI=${NVIDIA_SMI:-nvidia-smi}
STAT=${PROC_STAT:-/proc/stat}
JANELA_CPU=${CMP_CPU_JANELA:-3}
ESPERA_CPU=${CMP_CPU_ESPERA:-20}
CPU_MAX=10        # %
GPU_MAX=5         # %
uso() { echo "uso: roda-cmp-monitor.sh portao|inicia|para|avalia ... (veja o cabeçalho)" >&2; exit 2; }
[ $# -ge 1 ] || uso
CMD=$1; shift
SAIDA=; ARVORE=; CSV=; PIDARQ=
while [ $# -gt 0 ]; do
  case $1 in
    --saida) [ $# -ge 2 ] || uso; SAIDA=$2; shift 2;;
    --arvore) [ $# -ge 2 ] || uso; ARVORE=$2; shift 2;;
    --csv) [ $# -ge 2 ] || uso; CSV=$2; shift 2;;
    --pid-arq) [ $# -ge 2 ] || uso; PIDARQ=$2; shift 2;;
    *) uso;;
  esac
done

# "total ocioso" da linha agregada de /proc/stat: idle + iowait (campos 4 e 5 depois do rótulo) e total = soma de user..steal (campos 1 a 8).
le_cpu() { awk '/^cpu /{ t=0; for (i=2; i<=9; i++) t+=$i; print t, $5+$6; exit }' "$STAT" 2> /dev/null; }
# cpu_ocupada: imprime a % ocupada em uma janela de $JANELA_CPU s; falha (1) se a leitura for inválida.
cpu_ocupada() {
  local a b ta ia tb ib
  a=$(le_cpu); [ -n "$a" ] || return 1
  sleep "$JANELA_CPU"
  b=$(le_cpu); [ -n "$b" ] || return 1
  read -r ta ia <<< "$a"; read -r tb ib <<< "$b"
  awk -v ta="$ta" -v ia="$ia" -v tb="$tb" -v ib="$ib" 'BEGIN { dt = tb - ta; if (dt <= 0) exit 1; printf "%.2f\n", 100 * (dt - (ib - ia)) / dt }'
}
# descendentes de um PID (inclusive ele): um por linha
arvore_de() {
  local raiz=$1 fila novos p
  fila=$raiz
  while [ -n "$fila" ]; do
    for p in $fila; do echo "$p"; done
    novos=
    for p in $fila; do novos="$novos $(ps -o pid= --ppid "$p" 2> /dev/null | tr -d ' ' | tr '\n' ' ')"; done
    fila=$(echo "$novos" | xargs 2> /dev/null || true)
  done
}

case $CMD in
  portao)
    [ -n "$SAIDA" ] || uso
    : > "$SAIDA" || exit 2
    falhas=()
    # (a) CPU: espera ativa de até ESPERA_CPU s por janelas de JANELA_CPU s abaixo de 10%.
    t0=$(date +%s); ok_cpu=0; ult=
    while :; do
      if ult=$(cpu_ocupada); then
        echo "cpu_janela_pct=$ult" >> "$SAIDA"
        if awk -v x="$ult" -v m="$CPU_MAX" 'BEGIN { exit !(x < m) }'; then ok_cpu=1; break; fi
      else
        echo "cpu_janela_pct=ilegivel" >> "$SAIDA"; ult=ilegivel
      fi
      [ $(( $(date +%s) - t0 )) -lt "$ESPERA_CPU" ] || break
    done
    echo "cpu_espera_s=$(( $(date +%s) - t0 ))" >> "$SAIDA"
    [ $ok_cpu = 1 ] || falhas+=("CPU da máquina não baixou de ${CPU_MAX}% em ${ESPERA_CPU} s (última janela: ${ult:-?}%)")
    echo "loadavg_registrado=$(cat "${LOADAVG_ARQ:-/proc/loadavg}" 2> /dev/null | cut -d' ' -f1-3)" >> "$SAIDA"
    # (b) processos de compute fora da árvore medida
    mapfile -t apps < <("$SMI" -i "$UUID" --query-compute-apps=pid,used_memory,name --format=csv,noheader 2>&1)
    echo "compute_apps_n=${#apps[@]}" >> "$SAIDA"
    permitidos=
    [ -z "$ARVORE" ] || permitidos=$(arvore_de "$ARVORE")
    for l in "${apps[@]}"; do
      [ -n "${l// /}" ] || continue
      pid=${l%%,*}; pid=${pid// /}
      case $pid in
        ''|*[!0-9]*) falhas+=("nvidia-smi --query-compute-apps ilegível: $l"); continue;;
      esac
      if [ -z "$permitidos" ] || ! echo "$permitidos" | grep -qx "$pid"; then falhas+=("outro processo de compute na GPU: $l"); echo "compute_app_fora=$l" >> "$SAIDA"; fi
    done
    # (c) carga da GPU
    g=$("$SMI" -i "$UUID" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>&1 | head -n 1 | tr -d ' %')
    echo "gpu_util_pct=$g" >> "$SAIDA"
    case $g in
      ''|*[!0-9.]*) falhas+=("carga da GPU ilegível: '$g'");;
      *) awk -v x="$g" -v m="$GPU_MAX" 'BEGIN { exit !(x <= m) }' || falhas+=("carga da GPU ${g}% > ${GPU_MAX}%");;
    esac
    if [ ${#falhas[@]} = 0 ]; then echo "portao=OK" >> "$SAIDA"; exit 0; fi
    { printf 'portao=FALHA '; printf '%s; ' "${falhas[@]}"; echo; } >> "$SAIDA"
    exit 1;;
  inicia)
    [ -n "$CSV" ] && [ -n "$PIDARQ" ] || uso
    # sem -i na amostragem de clock? Mantém o -i: só a 4090 interessa.
    "$SMI" -i "$UUID" --query-gpu=timestamp,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,memory.used --format=csv,noheader,nounits -l 1 > "$CSV" 2> "$CSV.err" &
    echo $! > "$PIDARQ"
    exit 0;;
  para)
    [ -n "$PIDARQ" ] || uso
    p=$(cat "$PIDARQ" 2> /dev/null)
    if [ -n "$p" ] && kill -0 "$p" 2> /dev/null; then kill "$p" 2> /dev/null; wait "$p" 2> /dev/null; fi
    exit 0;;
  avalia)
    [ -n "$CSV" ] || uso
    [ -s "$CSV" ] || { echo "sem amostras em $CSV"; exit 1; }
    awk -F, -v max=80 '
      { n++; t=$5+0; if (n==1 || t>tm) tm=t; s=$3+0; if (n==1 || s>sm) sm=s; mm=$4+0; if (n==1 || mm>mmx) mmx=mm }
      END { printf "amostras=%d temp_max=%s clock_sm_max=%s clock_mem_max=%s (clock de memória: registrado, não decisivo)\n", n, tm, sm, mmx; exit (tm > max) }' "$CSV"
    exit $?;;
  *) uso;;
esac
