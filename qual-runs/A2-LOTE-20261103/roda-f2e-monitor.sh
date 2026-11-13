#!/usr/bin/env bash
# Comparação Lastro x llama.cpp (criterio-comparacao-llamacpp-20261108.md, E5 e emenda 3): portão ANTES de cada célula e amostragem nvidia-smi a 1 Hz DURANTE a célula.
#
#   roda-cmp-monitor.sh portao  --saida ARQ [--arvore PID]   portão antes da célula (código 0 = passou, 1 = a célula é SEM PROVA, 2 = uso)
#   roda-cmp-monitor.sh portao-depois --saida ARQ [--arvore PID]   mesma amostra pmon depois da célula (só registra, código 0)
#   roda-cmp-monitor.sh preflight                            (emenda 9) antes da janela: código 1 se houver PID C ou C+G na RTX 5090 (voxtype, GPU do Chrome...), imprime quais
#   roda-cmp-monitor.sh inicia  --csv ARQ --pid-arq ARQ      amostragem a 1 Hz em segundo plano (SÓ REGISTRA: nada derruba a célula por carga)
#   roda-cmp-monitor.sh para    --pid-arq ARQ                encerra a amostragem
#   roda-cmp-monitor.sh avalia  --csv ARQ                    temperatura máxima das amostras (código 1 se > 80 C: SEM PROVA); clocks só registrados
#
# Portão (emenda 3 > E5 > item (4)): (a) ocupação instantânea da CPU da máquina em janela de 3 s (/proc/stat) < 10%, com espera ativa de até 20 s; se não baixar,
# a célula é SEM PROVA (não adiada, não repetida); (b) `nvidia-smi --query-compute-apps` sem nenhum PID fora da árvore medida (--arvore PID: esse PID e seus
# descendentes são o lado medido; antes da partida da célula ninguém está medido); (c) carga da GPU <= 5%. O loadavg só é registrado.
# EMENDA 8 (vale onde conflitar): (b) e (c) foram substituídas por `nvidia-smi pmon -i UUID -c 3 -s um` (3 s): SEM PROVA se algum PID fora da árvore de tipo C ou C+G
# tem sm > 5% (ocioso '-' = 0) ou se a soma de sm dos PIDs de tipo G passa de 60%; pmon ilegível ou ausente = SEM PROVA. Utilização total, compute-apps e C+G ociosos só registrados.
# EMENDA 9 (r114; vale sobre a emenda 8): só PIDs de tipo G (desktop gráfico) são isentos. QUALQUER PID C ou C+G fora da árvore, no pmon OU em --query-compute-apps, ocioso ou não,
# é SEM PROVA (regra de presença); a soma G (60%) fica como salvaguarda e é pico de amostra esparsa; as 3 voltas do pmon são 3 capturas separadas (-c 1) com delimitador do roteiro;
# `dmon -s u -c 3` é gravado só como informação.
# O portão grava o que mediu em ARQ (linhas chave=valor) e uma linha final `portao=OK` ou `portao=FALHA motivo`.
# Variáveis só para teste: NVIDIA_SMI, PROC_STAT, CMP_CPU_JANELA (s, padrão 3), CMP_CPU_ESPERA (s, padrão 20), LOADAVG_ARQ.
set -u
GPU_ARQ=$(realpath -m -- "${F2E_GPU_ARQ:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/gpu-local.uuids}")   # UUID da GPU desta execução: arquivo LOCAL fora do git (um UUID por linha); nos resultados vai só o rótulo gpu_5090
UUID=$(grep -m1 -E '^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$' "$GPU_ARQ" 2> /dev/null) || UUID=
SHA_GPU_5090=88269ef39d73981809645a172b26d68c98ae0470e0a38c8fb1fed4772d17e1f2   # sha256 do UUID aprovado (rótulo gpu_5090; 128 bits aleatórios, sem recuperação prática do valor); o UUID em si fica só no arquivo local
if [ "${CMP_MODO_TESTE:-0}" = 1 ] || [ "${F2E_MODO_TESTE:-0}" = 1 ]; then SHA_GPU_5090=${F2E_SHA_GPU_ESPERADO:-$SHA_GPU_5090}; fi
if [ -n "$UUID" ] && [ "$(printf %s "$UUID" | sha256sum | cut -d' ' -f1)" != "$SHA_GPU_5090" ]; then UUID=; fi
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

# pmon_amostra "PIDS_DA_ARVORE" ARQ ROTULO (emenda 9): TRÊS capturas separadas de UMA volta (`nvidia-smi pmon -i UUID -c 1 -s um`); o próprio roteiro insere o delimitador
# de volta (linha `pmon_<rotulo>_volta=N` no ARQ e marcador @@VOLTA na entrada do awk): a fronteira NUNCA é inferida pela repetição de PID. Grava uma linha por PID
# (pmon_<rotulo>_pid=...) e o resumo; define PMON_VEREDITO (OK ou o motivo; falha fechada: pmon ausente, com erro, volta sem cabeçalho ou com linha ilegível), PMON_C_MAX
# (maior sm% C/C+G fora da árvore, só registro), PMON_C_N / PMON_C_LISTA (PIDs C ou C+G FORA da árvore, ociosos ou não: decisivo, emenda 9), PMON_G_SOMA (maior soma G por volta,
# pico de amostra esparsa, não fração da placa) e PMON_SO_G=1 se o único motivo é a soma G. sm '-' conta como 0. Por PID vale o maior sm entre as voltas.
PMON_VEREDITO=; PMON_C_MAX=; PMON_G_SOMA=; PMON_C_N=0; PMON_C_LISTA=; PMON_SO_G=0
pmon_amostra() {
  local arv=$1 arq=$2 rot=$3 bruto rc resumo todo= i falhou=
  PMON_SO_G=0; PMON_C_N=0; PMON_C_LISTA=
  for i in 1 2 3; do
    bruto=$(timeout 20 "$SMI" pmon -i "$UUID" -c 1 -s um 2>&1); rc=$?
    { echo "pmon_${rot}_volta=$i"; echo "pmon_${rot}_rc=$rc"; echo "$bruto" | sed "s/^/pmon_${rot}_bruto: /"; } >> "$arq"
    [ $rc = 0 ] || { falhou="nvidia-smi pmon falhou na volta $i (código $rc): sem prova do desktop"; break; }
    todo="${todo}@@VOLTA"$'\n'"${bruto}"$'\n'
  done
  if [ -n "$falhou" ]; then PMON_VEREDITO=$falhou; echo "pmon_${rot}_veredito=$falhou" >> "$arq"; return; fi
  resumo=$(printf '%s' "$todo" | awk -v arv="$arv" '
    BEGIN { n=split(arv, a, " "); for (i=1;i<=n;i++) dentro[a[i]]=1 }
    /^@@VOLTA/ { am++; next }
    /^#/ { if ($0 ~ /^# *gpu/) cab[am]=1; next }
    NF==0 { next }
    {
      # GPU sem processo: o pmon imprime UMA linha placeholder (idx numérico, todos os demais campos "-", pid "-"): é "nenhum processo", não é ilegível
      if (am>=1 && $1 ~ /^[0-9]+$/ && $2=="-") { ph=1; for (k=3;k<=NF;k++) if ($k!="-") ph=0; if (ph && NF>=11) { vazias++; next } }
      if (am<1 || NF<11 || $2 !~ /^[0-9]+$/ || $3 !~ /^(C|G|C\+G)$/ || ($4!="-" && $4 !~ /^[0-9]+$/) || ($5!="-" && $5 !~ /^[0-9]+$/) || ($10!="-" && $10 !~ /^[0-9]+$/)) { ruim++; next }
      pid=$2
      sm=($4=="-")?0:$4+0; mem=($5=="-")?0:$5+0; fb=($10=="-")?0:$10+0
      linhas++
      if (!(pid in tipo)) { ordem[++np]=pid; tipo[pid]=$3; cmd[pid]=$12 }
      if (sm>smx[pid]) smx[pid]=sm
      if (mem>mmx[pid]) mmx[pid]=mem
      if (fb>fbx[pid]) fbx[pid]=fb
      if (!(pid in dentro) && $3=="G") gs[am]+=sm
    }
    END {
      if (am!=3) { print "ERRO " am+0 " volta(s) em vez de 3"; exit }
      for (v=1; v<=3; v++) if (!cab[v]) { print "ERRO sem cabeçalho do pmon na volta " v; exit }
      if (ruim>0) { print "ERRO " ruim " linha(s) ilegível(is) do pmon"; exit }
      cmax=0; gmax=0; ociosos=0; fbout=0; cn=0; lista=""
      for (v=1; v<=3; v++) if (gs[v]>gmax) gmax=gs[v]
      for (i=1;i<=np;i++) {
        p=ordem[i]; d=(p in dentro)?1:0
        printf "PID pmon_%s_pid=%s tipo=%s sm=%d mem=%d fb_mb=%d dentro_arvore=%d cmd=%s\n", "'"$rot"'", p, tipo[p], smx[p], mmx[p], fbx[p], d, cmd[p]
        if (!d) { fbout+=fbx[p]; if (tipo[p]!="G") { cn++; lista=lista (lista==""?"":", ") p " (" cmd[p] ")"; if (smx[p]>cmax) cmax=smx[p]; if (smx[p]==0) ociosos++ } }
      }
      printf "RES c_fora_sm_max=%d g_soma_sm_max=%d cg_ociosos=%d c_fora_n=%d fb_outros_mb=%d linhas=%d\n", cmax, gmax, ociosos, cn, fbout, linhas+0
      print "LISTA " lista
    }')
  echo "$resumo" | grep '^PID ' | sed 's/^PID //' >> "$arq"
  local res; res=$(echo "$resumo" | grep '^RES ' | head -n 1)
  if [ -z "$res" ]; then PMON_VEREDITO="pmon ilegível: $(echo "$resumo" | head -n 1)"; echo "pmon_${rot}_veredito=ILEGIVEL" >> "$arq"; return; fi
  echo "${res#RES }" | tr ' ' '\n' | sed "s/^/pmon_${rot}_/" >> "$arq"
  PMON_C_MAX=$(echo "$res" | sed 's/.*c_fora_sm_max=\([0-9]*\).*/\1/'); PMON_G_SOMA=$(echo "$res" | sed 's/.*g_soma_sm_max=\([0-9]*\).*/\1/')
  PMON_C_N=$(echo "$res" | sed 's/.*c_fora_n=\([0-9]*\).*/\1/'); PMON_C_LISTA=$(echo "$resumo" | grep '^LISTA ' | head -n 1 | sed 's/^LISTA //')
  PMON_VEREDITO=OK
  if [ "$PMON_C_N" -gt 0 ]; then PMON_VEREDITO="processo(s) de compute (tipo C ou C+G) fora da árvore medida no pmon, ocioso ou não: ${PMON_C_LISTA}; feche-os antes da janela (voxtype, processo de GPU do Chrome)"
  elif [ "$PMON_G_SOMA" -gt "$G_SOMA_MAX" ]; then PMON_SO_G=1; PMON_VEREDITO="pico do pmon (soma G) em ${PMON_G_SOMA}% (> ${G_SOMA_MAX}%): desktop disputando a placa de forma anormal"; fi
  echo "pmon_${rot}_veredito=$PMON_VEREDITO" >> "$arq"
}
# apps_fora "PIDS_DA_ARVORE": `--query-compute-apps` da RTX 5090; define APPS_VEREDITO (OK, ou os PIDs/comandos fora da árvore, ou falha fechada se a consulta não for legível) e APPS_N (linhas lidas).
APPS_VEREDITO=; APPS_N=0
apps_fora() {
  local arv=" $(echo $1) " bruto rc l pid fora=
  bruto=$("$SMI" -i "$UUID" --query-compute-apps=pid,used_memory,name --format=csv,noheader 2>&1); rc=$?
  APPS_N=0
  if [ $rc != 0 ]; then APPS_VEREDITO="nvidia-smi --query-compute-apps falhou (código $rc): sem prova"; return; fi
  while IFS= read -r l; do
    [ -n "$l" ] || continue
    APPS_N=$((APPS_N + 1))
    pid=${l%%,*}; pid=${pid// /}
    case $pid in ''|*[!0-9]*) APPS_VEREDITO="compute-apps ilegível: $l"; return;; esac
    case $arv in *" $pid "*) ;; *) fora="${fora}${fora:+, }${pid} ($(echo "${l#*,}" | sed 's/^[^,]*, *//'))";; esac
  done <<< "$bruto"
  if [ -n "$fora" ]; then APPS_VEREDITO="processo(s) de compute fora da árvore medida em --query-compute-apps: ${fora}; feche-os antes da janela"; else APPS_VEREDITO=OK; fi
}
# dmon_registra ARQ ROTULO: só informação (emenda 9): `nvidia-smi dmon -i UUID -s u -c 3`; antes da célula, com nada medido rodando, o sm do dmon é toda a carga estranha.
dmon_registra() {
  local arq=$1 rot=$2 bruto rc mx
  bruto=$(timeout 20 "$SMI" dmon -i "$UUID" -s u -c 3 2>&1); rc=$?
  { echo "dmon_${rot}_rc=$rc"; echo "$bruto" | sed "s/^/dmon_${rot}_bruto: /"; } >> "$arq"
  mx=$(echo "$bruto" | awk '!/^#/ && NF>=2 && $2 ~ /^[0-9]+$/ { if (!n || $2+0>m) m=$2+0; n++ } END { if (n) print m }')
  [ $rc = 0 ] && [ -n "$mx" ] && echo "dmon_${rot}_sm_max=$mx" >> "$arq" || echo "dmon_${rot}_sm_max=ilegivel" >> "$arq"
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
    # emenda 9: qualquer PID C ou C+G fora da árvore em --query-compute-apps também é decisivo (presença, ocioso ou não); utilização total só registrada; dmon só informação
    apps_fora "$permitidos"
    echo "compute_apps_n_registrado=$APPS_N" >> "$SAIDA"
    echo "compute_apps_veredito=$APPS_VEREDITO" >> "$SAIDA"
    case $APPS_VEREDITO in OK) ;; *) falhas+=("$APPS_VEREDITO");; esac
    g=$("$SMI" -i "$UUID" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>&1 | head -n 1 | tr -d ' %')
    echo "gpu_util_pct_registrado=$g" >> "$SAIDA"
    dmon_registra "$SAIDA" antes
    if [ ${#falhas[@]} = 0 ]; then echo "portao=OK" >> "$SAIDA"; exit 0; fi
    { printf 'portao=FALHA '; printf '%s; ' "${falhas[@]}"; echo; } >> "$SAIDA"
    exit 1;;
  preflight)
    # emenda 9: usado pelo envelope da janela ANTES de tocar em timers ou na GPU: nada medido roda, então QUALQUER PID C ou C+G (pmon ou --query-compute-apps) é estranho. Só imprime; código 1 se houver.
    pmon_amostra "" /dev/null preflight
    apps_fora ""
    pf=()
    case $PMON_VEREDITO in OK) ;; *) [ "$PMON_SO_G" = 1 ] || pf+=("$PMON_VEREDITO");; esac
    case $APPS_VEREDITO in OK) ;; *) pf+=("$APPS_VEREDITO");; esac
    if [ ${#pf[@]} = 0 ]; then echo "preflight: nenhum PID C ou C+G na RTX 5090"; exit 0; fi
    printf 'preflight FALHOU: %s\n' "${pf[@]}"; exit 1;;
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
    # sem -i na amostragem de clock? Mantém o -i: só a placa de medida interessa.
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
