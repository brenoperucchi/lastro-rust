#!/usr/bin/env bash
# Comparação Lastro x llama.cpp (criterio-comparacao-llamacpp-20261108.md, E5 e emenda 3): portão ANTES de cada célula e amostragem nvidia-smi a 1 Hz DURANTE a célula.
#
#   roda-cmp-monitor.sh portao  --saida ARQ [--arvore PID]   portão antes da célula (código 0 = passou, 1 = a célula é SEM PROVA, 2 = uso)
#   roda-cmp-monitor.sh portao-depois --saida ARQ [--arvore PID]   mesma amostra pmon depois da célula (só registra, código 0)
#   roda-cmp-monitor.sh inicia  --csv ARQ --pid-arq ARQ      amostragem a 1 Hz em segundo plano (SÓ REGISTRA: nada derruba a célula por carga)
#   roda-cmp-monitor.sh para    --pid-arq ARQ                encerra a amostragem
#   roda-cmp-monitor.sh avalia  --csv ARQ                    temperatura máxima das amostras (código 1 se > 80 C: SEM PROVA); clocks só registrados
#
# Portão (emenda 3 > E5 > item (4)): (a) ocupação instantânea da CPU da máquina em janela de 3 s (/proc/stat) < 10%, com espera ativa de até 20 s; se não baixar,
# a célula é SEM PROVA (não adiada, não repetida); (b) `nvidia-smi --query-compute-apps` sem nenhum PID fora da árvore medida (--arvore PID: esse PID e seus
# descendentes são o lado medido; antes da partida da célula ninguém está medido); (c) carga da GPU <= 5%. O loadavg só é registrado.
# EMENDA 8 (vale onde conflitar): (b) e (c) foram substituídas por `nvidia-smi pmon -i UUID -c 3 -s um` (3 s): SEM PROVA se algum PID fora da árvore de tipo C ou C+G
# tem sm > 5% (ocioso '-' = 0) ou se a soma de sm dos PIDs de tipo G passa de 60%; pmon ilegível ou ausente = SEM PROVA. Utilização total, compute-apps e C+G ociosos só registrados.
# O portão grava o que mediu em ARQ (linhas chave=valor) e uma linha final `portao=OK` ou `portao=FALHA motivo`.
# Variáveis só para teste: NVIDIA_SMI, PROC_STAT, CMP_CPU_JANELA (s, padrão 3), CMP_CPU_ESPERA (s, padrão 20), LOADAVG_ARQ.
set -u
UUID=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd
SMI=${NVIDIA_SMI:-nvidia-smi}
STAT=${PROC_STAT:-/proc/stat}
JANELA_CPU=${CMP_CPU_JANELA:-3}
ESPERA_CPU=${CMP_CPU_ESPERA:-20}
CPU_MAX=10        # %
C_SM_MAX=5        # % (emenda 8: palpite declarado)
G_SOMA_MAX=60     # % (emenda 8: palpite declarado)
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

# pmon_amostra "PIDS_DA_ARVORE" ARQ ROTULO: roda `nvidia-smi pmon -i UUID -c 3 -s um`, grava no ARQ uma linha por PID (pmon_<rotulo>_pid=...) e o resumo;
# define PMON_VEREDITO (OK ou o motivo, falha fechada: pmon ausente, com erro, sem cabeçalho ou com linha ilegível) e PMON_C_MAX (maior sm% C/C+G fora da árvore) e PMON_G_SOMA.
# Amostra = cada volta de 1 s do pmon; uma nova volta começa quando um PID já visto nela se repete. sm '-' (ocioso) conta como 0. Por PID vale o maior sm das voltas;
# a soma dos PIDs G é calculada por volta e vale a maior. Valores 5, 60 e 3 s são palpites declarados (emenda 8).
PMON_VEREDITO=; PMON_C_MAX=; PMON_G_SOMA=
pmon_amostra() {
  local arv=$1 arq=$2 rot=$3 bruto rc resumo
  bruto=$(timeout 20 "$SMI" pmon -i "$UUID" -c 3 -s um 2>&1); rc=$?
  { echo "pmon_${rot}_rc=$rc"; echo "$bruto" | sed "s/^/pmon_${rot}_bruto: /"; } >> "$arq"
  if [ $rc != 0 ]; then PMON_VEREDITO="nvidia-smi pmon falhou (código $rc): sem prova do desktop"; return; fi
  resumo=$(echo "$bruto" | awk -v arv="$arv" -v rot="$rot" '
    BEGIN { n=split(arv, a, " "); for (i=1;i<=n;i++) dentro[a[i]]=1 }
    /^#/ { if ($0 ~ /^# *gpu/) cab=1; next }
    NF==0 { next }
    {
      if (NF<11 || $2 !~ /^[0-9]+$/ || $3 !~ /^(C|G|C\+G)$/ || ($4!="-" && $4 !~ /^[0-9]+$/) || ($5!="-" && $5 !~ /^[0-9]+$/) || ($10!="-" && $10 !~ /^[0-9]+$/)) { ruim++; next }
      pid=$2; if (pid in vis) { am++; split("", vis) } vis[pid]=1
      sm=($4=="-")?0:$4+0; mem=($5=="-")?0:$5+0; fb=($10=="-")?0:$10+0
      linhas++
      if (!(pid in tipo)) { ordem[++np]=pid; tipo[pid]=$3; cmd[pid]=$12 }
      if (sm>smx[pid]) smx[pid]=sm
      if (mem>mmx[pid]) mmx[pid]=mem
      if (fb>fbx[pid]) fbx[pid]=fb
      if (!(pid in dentro) && $3=="G") gs[am+1]+=sm
    }
    END {
      if (!cab) { print "ERRO sem cabeçalho do pmon"; exit }
      if (ruim>0) { print "ERRO " ruim " linha(s) ilegível(is) do pmon"; exit }
      cmax=0; gmax=0; ociosos=0; fbout=0
      for (k in gs) if (gs[k]>gmax) gmax=gs[k]
      for (i=1;i<=np;i++) {
        p=ordem[i]; d=(p in dentro)?1:0
        printf "PID pmon_%s_pid=%s tipo=%s sm=%d mem=%d fb_mb=%d dentro_arvore=%d cmd=%s\n", rot, p, tipo[p], smx[p], mmx[p], fbx[p], d, cmd[p]
        if (!d) { fbout+=fbx[p]; if (tipo[p]!="G") { if (smx[p]>cmax) cmax=smx[p]; if (smx[p]==0) ociosos++ } }
      }
      printf "RES c_fora_sm_max=%d g_soma_sm_max=%d cg_ociosos=%d fb_outros_mb=%d linhas=%d\n", cmax, gmax, ociosos, fbout, linhas+0
    }')
  echo "$resumo" | grep '^PID ' | sed 's/^PID //' >> "$arq"
  local res; res=$(echo "$resumo" | grep '^RES ' | head -n 1)
  if [ -z "$res" ]; then PMON_VEREDITO="pmon ilegível: $(echo "$resumo" | head -n 1)"; echo "pmon_${rot}_veredito=ILEGIVEL" >> "$arq"; return; fi
  echo "${res#RES }" | tr ' ' '\n' | sed "s/^/pmon_${rot}_/" >> "$arq"
  PMON_C_MAX=$(echo "$res" | sed 's/.*c_fora_sm_max=\([0-9]*\).*/\1/'); PMON_G_SOMA=$(echo "$res" | sed 's/.*g_soma_sm_max=\([0-9]*\).*/\1/')
  PMON_VEREDITO=OK
  if [ "$PMON_C_MAX" -gt "$C_SM_MAX" ]; then PMON_VEREDITO="PID fora da árvore medida com tipo C ou C+G usando sm ${PMON_C_MAX}% (> ${C_SM_MAX}%)"
  elif [ "$PMON_G_SOMA" -gt "$G_SOMA_MAX" ]; then PMON_VEREDITO="soma de sm dos PIDs de tipo G em ${PMON_G_SOMA}% (> ${G_SOMA_MAX}%): desktop disputando a placa de forma anormal"; fi
  echo "pmon_${rot}_veredito=$PMON_VEREDITO" >> "$arq"
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
    # (b) emenda 8: amostra pmon de 3 s; decisivo = PID fora da árvore com tipo C ou C+G e sm > 5, ou soma de sm dos PIDs de tipo G > 60
    permitidos=
    [ -z "$ARVORE" ] || permitidos=$(arvore_de "$ARVORE")
    pmon_amostra "$permitidos" "$SAIDA" antes
    case $PMON_VEREDITO in OK) ;; *) falhas+=("$PMON_VEREDITO");; esac
    # registrado, não decisivo: compute-apps e utilização total
    mapfile -t apps < <("$SMI" -i "$UUID" --query-compute-apps=pid,used_memory,name --format=csv,noheader 2>&1)
    echo "compute_apps_n_registrado=${#apps[@]}" >> "$SAIDA"
    g=$("$SMI" -i "$UUID" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>&1 | head -n 1 | tr -d ' %')
    echo "gpu_util_pct_registrado=$g" >> "$SAIDA"
    if [ ${#falhas[@]} = 0 ]; then echo "portao=OK" >> "$SAIDA"; exit 0; fi
    { printf 'portao=FALHA '; printf '%s; ' "${falhas[@]}"; echo; } >> "$SAIDA"
    exit 1;;
  portao-depois)
    # emenda 8: mesma amostra pmon DEPOIS da célula; só registra (código 0 sempre), grava no arquivo dado
    [ -n "$SAIDA" ] || uso
    : > "$SAIDA" || exit 2
    permitidos=
    [ -z "$ARVORE" ] || permitidos=$(arvore_de "$ARVORE")
    pmon_amostra "$permitidos" "$SAIDA" depois
    exit 0;;
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
