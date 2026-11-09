#!/usr/bin/env bash
# Teste do roda-cmp-monitor.sh com nvidia-smi FALSO e /proc/stat FALSO (um gerador em segundo plano com ocupação escolhida). SEM GPU. Pasta com mktemp -d, sem rm -rf.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
M=$AQUI/roda-cmp-monitor.sh
D=$(mktemp -d -p "${TMPDIR:-/tmp}" cmp-monitor-teste.XXXXXX) || exit 1
echo "pasta de teste: $D"
falhas=0
confere() { if eval "$2"; then echo "ok: $1"; else echo "FALHOU: $1"; falhas=$((falhas+1)); fi; }
GER=
para_gerador() { [ -z "$GER" ] || { kill "$GER" 2> /dev/null; wait "$GER" 2> /dev/null; GER=; }; }
trap 'para_gerador; [ -z "${ARV:-}" ] || kill "$ARV" 2> /dev/null' EXIT
# gerador de /proc/stat: ocupação ($1 %, inteiro) durante $2 s, depois $3 %
cat > "$D/gera-stat.sh" <<'G'
#!/usr/bin/env bash
arq=$1; p1=$2; dur=$3; p2=$4
t=0; i=0; ini=$(date +%s)
while :; do
  p=$p1; [ $(( $(date +%s) - ini )) -ge "$dur" ] && p=$p2
  t=$((t + 100)); i=$((i + 100 - p))
  printf 'cpu  %d 0 0 %d 0 0 0 0 0 0\n' $((t - i)) "$i" > "$arq.tmp" && mv "$arq.tmp" "$arq"
  sleep 0.1
done
G
chmod +x "$D/gera-stat.sh"
inicia_stat() { para_gerador; "$D/gera-stat.sh" "$D/stat" "$1" "$2" "$3" & GER=$!; sleep 0.3; }
# nvidia-smi falso: APPS (linhas de compute-apps), UTIL (carga), TEMP
cat > "$D/smi" <<'F'
#!/usr/bin/env bash
case $* in
  *pmon*) [ "${FAKE_PMON_RC:-0}" = 0 ] || { echo "erro falso" >&2; exit "$FAKE_PMON_RC"; }
          if [ -n "${FAKE_PMON_ARQ:-}" ]; then cat "$FAKE_PMON_ARQ"; else printf '# gpu         pid   type     sm    mem    enc    dec    jpg    ofa     fb   ccpm    command \n# Idx           #    C/G      %%      %%      %%      %%      %%      %%     MB     MB    name \n'; fi;;
  *query-compute-apps*) [ -z "${FAKE_APPS:-}" ] || printf '%b\n' "$FAKE_APPS";;
  *-l\ 1*) while :; do echo "2026/11/08 10:00:00.000, ${FAKE_UTIL:-0}, 2100, 10500, ${FAKE_TEMP:-60}, 4000"; sleep 1; done;;
  *query-gpu=utilization.gpu*) echo "${FAKE_UTIL:-0}";;
  *) echo "?";;
esac
F
chmod +x "$D/smi"
export NVIDIA_SMI=$D/smi PROC_STAT=$D/stat CMP_CPU_JANELA=1 CMP_CPU_ESPERA=3
echo "0.1 0.1 0.1 1/1 1" > "$D/loadavg"; export LOADAVG_ARQ=$D/loadavg

# 1. passa: CPU 2%, sem apps, GPU 0%
inicia_stat 2 999 2
FAKE_APPS= FAKE_UTIL=0 "$M" portao --saida "$D/p1.txt"; rc=$?
confere "portão passa com CPU ociosa, sem compute app e GPU 0%" "[ $rc = 0 ] && grep -q '^portao=OK' $D/p1.txt && grep -q '^cpu_janela_pct=' $D/p1.txt && grep -q '^loadavg_registrado=0.1' $D/p1.txt"
# 2. CPU ocupada o tempo todo: falha após a espera de até 20 s (aqui 3 s)
inicia_stat 80 999 80
t0=$(date +%s); "$M" portao --saida "$D/p2.txt"; rc=$?; t1=$(date +%s)
confere "CPU a 80% o tempo todo: falha (SEM PROVA) depois da espera ativa, sem adiar além dela" "[ $rc = 1 ] && grep -q 'portao=FALHA.*CPU da máquina' $D/p2.txt && [ $((t1 - t0)) -ge 3 ] && [ $((t1 - t0)) -le 8 ]"
# 3. espera ativa: CPU cai para 1% depois de ~3 s (dentro dos 8 s de espera deste caso): passa e o arquivo tem >= 2 janelas
inicia_stat 80 3 1
CMP_CPU_ESPERA=8 "$M" portao --saida "$D/p3.txt"; rc=$?
confere "espera ativa: CPU baixa durante a espera e o portão passa" "[ $rc = 0 ] && [ \$(grep -c '^cpu_janela_pct=' $D/p3.txt) -ge 2 ]"
# 4. limite exato: 10% não passa (< 10%)
inicia_stat 10 999 10
"$M" portao --saida "$D/p4.txt"; rc=$?
confere "CPU a 10% exatos não passa (< 10%)" "[ $rc = 1 ]"
inicia_stat 2 999 2
# pmon sintético no formato real lido em 09/11 (nvidia-smi pmon -i UUID -c 3 -s um): cabeçalho de 2 linhas, depois uma linha por PID e por volta (3 voltas)
CAB='# gpu         pid   type     sm    mem    enc    dec    jpg    ofa     fb   ccpm    command \n# Idx           #    C/G      %      %      %      %      %      %     MB     MB    name \n'
lin() { printf '    0 %7s %5s %6s %6s      -      -      -      -  %5s      0    %s       \n' "$1" "$2" "$3" "$4" "$5" "$6"; }
volta() { # $1 = sm do Hyprland (G); $2.. extras já formatados
  lin 41533 G "$1" "$1" 413 Hyprland; lin 41609 G 1 - 761 quickshell; lin 42406 C+G - - 364 'renderD128 --cr'; lin 1352017 C+G - - 997 voxtype; [ -z "${2:-}" ] || printf '%s\n' "$2"
}
pm() { { printf "%b" "$CAB"; for i in 1 2 3; do volta "$1" "${2:-}"; done; } > "$D/pmon.txt"; }
roda_pm() { FAKE_PMON_ARQ=$D/pmon.txt "$M" portao --saida "$1" "${@:2}"; }
# 5. só o desktop: Hyprland G 25%, chrome e voxtype C+G ociosos: VÁLIDO
pm 25
roda_pm "$D/p5.txt"; rc=$?
confere "só desktop (G 25%) com chrome e voxtype C+G ociosos: portão passa" "[ $rc = 0 ] && grep -q '^portao=OK' $D/p5.txt && grep -q '^pmon_antes_veredito=OK' $D/p5.txt && grep -q '^pmon_antes_g_soma_sm_max=26' $D/p5.txt && grep -q '^pmon_antes_cg_ociosos=2' $D/p5.txt && grep -q 'pmon_antes_pid=1352017 tipo=C+G' $D/p5.txt"
# 6. um PID C com sm 20 fora da árvore: SEM PROVA
pm 25 "$(lin 77777 C 20 5 500 outro)"
roda_pm "$D/p6.txt"; rc=$?
confere "PID C com sm 20% fora da árvore: falha e registra o motivo" "[ $rc = 1 ] && grep -q 'portao=FALHA.*tipo C ou C+G usando sm 20%' $D/p6.txt"
pm 25 "$(lin 77777 C+G 6 5 500 outro)"
roda_pm "$D/p6b.txt"; rc=$?
confere "PID C+G com sm 6% (> 5%) fora da árvore: falha" "[ $rc = 1 ]"
pm 25 "$(lin 77777 C 5 5 500 outro)"
roda_pm "$D/p6c.txt"; rc=$?
confere "PID C com sm 5% exatos passa (> 5% é o limite)" "[ $rc = 0 ]"
# 7. PID da árvore medida é ignorado (inclusive descendente)
sleep 60 & ARV=$!
pm 25 "$(lin "$ARV" C 90 50 3000 medido)"
roda_pm "$D/p7.txt" --arvore "$ARV"; rc=$?
confere "PID da árvore medida (C, sm 90%) é ignorado" "[ $rc = 0 ] && grep -q \"dentro_arvore=1 cmd=medido\" $D/p7.txt"
bash -c 'sleep 60 & echo $! > '"$D"'/filho.pid; wait' &
PAI=$!; sleep 0.5; FILHO=$(cat "$D/filho.pid")
pm 25 "$(lin "$FILHO" C 90 50 3000 filho; lin 99999 C 30 1 1 fora)"
roda_pm "$D/p7b.txt" --arvore "$PAI"; rc=$?
confere "descendente da árvore ignorado, mas outro PID C fora dela falha" "[ $rc = 1 ] && grep -q 'usando sm 30%' $D/p7b.txt"
kill "$PAI" "$FILHO" "$ARV" 2> /dev/null; wait "$PAI" "$ARV" 2> /dev/null; ARV=
# 8. soma dos PIDs G: 70% falha; 60% passa
pm 70
roda_pm "$D/p8.txt"; rc=$?
confere "PIDs G somando 71%: SEM PROVA" "[ $rc = 1 ] && grep -q 'soma de sm dos PIDs de tipo G em 71%' $D/p8.txt"
pm 59
roda_pm "$D/p8b.txt"; rc=$?
confere "PIDs G somando 60% exatos passa (> 60% é o limite)" "[ $rc = 0 ]"
# 9. utilização total só registrada (46% não decide)
FAKE_UTIL=46 roda_pm "$D/p9.txt"; rc=$?
confere "utilização total da GPU 46% só registrada, não decide" "[ $rc = 0 ] && grep -q '^gpu_util_pct_registrado=46' $D/p9.txt"
# 10. pmon ilegível ou ausente: SEM PROVA fechado
FAKE_PMON_RC=1 "$M" portao --saida "$D/p10.txt"; rc=$?
confere "pmon com erro (código 1): falha fechada" "[ $rc = 1 ] && grep -q 'pmon falhou' $D/p10.txt"
printf 'lixo sem cabecalho\n' > "$D/pmon.txt"; roda_pm "$D/p10b.txt"; rc=$?
confere "pmon sem cabeçalho: falha fechada" "[ $rc = 1 ] && grep -q 'pmon ilegível' $D/p10b.txt"
{ printf "%b" "$CAB"; printf '    0   41533     G    abc      -      -      -      -      -    413      0    Hyprland\n'; } > "$D/pmon.txt"; roda_pm "$D/p10c.txt"; rc=$?
confere "linha de PID ilegível: falha fechada" "[ $rc = 1 ] && grep -q 'ilegíve' $D/p10c.txt"
NVIDIA_SMI=$D/nao-existe "$M" portao --saida "$D/p10d.txt"; rc=$?
confere "nvidia-smi ausente: falha fechada" "[ $rc = 1 ] && grep -q 'pmon falhou' $D/p10d.txt"
# 11. depois da célula: só registra (código 0 mesmo com PID C), grava o veredito e a soma G
pm 25 "$(lin 77777 C 20 5 500 outro)"
FAKE_PMON_ARQ=$D/pmon.txt "$M" portao-depois --saida "$D/pd.txt"; rc=$?
confere "portao-depois: código 0, registra soma G e veredito, não decide" "[ $rc = 0 ] && grep -q '^pmon_depois_g_soma_sm_max=26' $D/pd.txt && grep -q '^pmon_depois_veredito=PID' $D/pd.txt"
FAKE_PMON_RC=1 "$M" portao-depois --saida "$D/pd2.txt"; rc=$?
confere "portao-depois com pmon quebrado: código 0 e registra" "[ $rc = 0 ] && grep -q '^pmon_depois_rc=1' $D/pd2.txt"
# 12. amostragem a 1 Hz só registra e avalia temperatura
FAKE_TEMP=60 "$M" inicia --csv "$D/mon.csv" --pid-arq "$D/mon.pid"; sleep 3.3
"$M" para --pid-arq "$D/mon.pid"
n=$(wc -l < "$D/mon.csv")
confere "amostragem a ~1 Hz (3 a 5 linhas em 3,3 s) e processo encerrado" "[ $n -ge 3 ] && [ $n -le 5 ] && ! kill -0 \$(cat $D/mon.pid) 2> /dev/null"
"$M" avalia --csv "$D/mon.csv" > "$D/av.txt"; rc=$?
confere "temperatura 60 C: ok, clock de memória 'registrado, não decisivo'" "[ $rc = 0 ] && grep -q 'temp_max=60' $D/av.txt && grep -q 'não decisivo' $D/av.txt"
FAKE_TEMP=85 "$M" inicia --csv "$D/mon2.csv" --pid-arq "$D/mon2.pid"; sleep 1.3; "$M" para --pid-arq "$D/mon2.pid"
"$M" avalia --csv "$D/mon2.csv" > "$D/av2.txt"; rc=$?
confere "temperatura 85 C: SEM PROVA (código 1)" "[ $rc = 1 ] && grep -q 'temp_max=85' $D/av2.txt"
# 13. uso
"$M" 2> /dev/null; confere "sem subcomando: código 2" "[ $? = 2 ]"
"$M" portao 2> /dev/null; confere "portão sem --saida: código 2" "[ $? = 2 ]"
para_gerador
if [ "$falhas" = 0 ]; then echo "TODOS OS TESTES PASSARAM"; exit 0; else echo "$falhas FALHA(S)"; exit 1; fi
