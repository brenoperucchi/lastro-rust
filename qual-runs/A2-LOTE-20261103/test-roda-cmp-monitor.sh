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
# 5. outro PID de compute
FAKE_APPS='12345, 500 MiB, /usr/bin/outro' "$M" portao --saida "$D/p5.txt"; rc=$?
confere "outro processo de compute: falha e registra a linha" "[ $rc = 1 ] && grep -q 'outro processo de compute na GPU: 12345' $D/p5.txt && grep -q '^compute_app_fora=12345' $D/p5.txt"
# 6. PID dentro da árvore medida passa; PID filho também
sleep 60 & ARV=$!
FAKE_APPS="$ARV, 500 MiB, /x" "$M" portao --saida "$D/p6.txt" --arvore "$ARV"; rc=$?
confere "PID da própria árvore medida não é acusado" "[ $rc = 0 ]"
bash -c 'sleep 60 & echo $! > '"$D"'/filho.pid; wait' &
PAI=$!; sleep 0.5
FILHO=$(cat "$D/filho.pid")
FAKE_APPS="$FILHO, 500 MiB, /x" "$M" portao --saida "$D/p6b.txt" --arvore "$PAI"; rc=$?
confere "PID descendente da árvore medida não é acusado" "[ $rc = 0 ]"
FAKE_APPS="$FILHO, 500 MiB, /x
99999, 1 MiB, /y" "$M" portao --saida "$D/p6c.txt" --arvore "$PAI"; rc=$?
confere "mas um PID fora da árvore, junto com o da árvore, falha" "[ $rc = 1 ] && grep -q '99999' $D/p6c.txt"
kill "$PAI" "$FILHO" "$ARV" 2> /dev/null; wait "$PAI" "$ARV" 2> /dev/null; ARV=
# 7. carga da GPU
FAKE_UTIL=50 "$M" portao --saida "$D/p7.txt"; rc=$?
confere "GPU a 50%: falha" "[ $rc = 1 ] && grep -q 'carga da GPU 50%' $D/p7.txt"
FAKE_UTIL=5 "$M" portao --saida "$D/p7b.txt"; rc=$?
confere "GPU a 5% passa (<= 5%)" "[ $rc = 0 ]"
# 8. amostragem a 1 Hz só registra e avalia temperatura
FAKE_TEMP=60 "$M" inicia --csv "$D/mon.csv" --pid-arq "$D/mon.pid"; sleep 3.3
"$M" para --pid-arq "$D/mon.pid"
n=$(wc -l < "$D/mon.csv")
confere "amostragem a ~1 Hz (3 a 5 linhas em 3,3 s) e processo encerrado" "[ $n -ge 3 ] && [ $n -le 5 ] && ! kill -0 \$(cat $D/mon.pid) 2> /dev/null"
"$M" avalia --csv "$D/mon.csv" > "$D/av.txt"; rc=$?
confere "temperatura 60 C: ok, clock de memória 'registrado, não decisivo'" "[ $rc = 0 ] && grep -q 'temp_max=60' $D/av.txt && grep -q 'não decisivo' $D/av.txt"
FAKE_TEMP=85 "$M" inicia --csv "$D/mon2.csv" --pid-arq "$D/mon2.pid"; sleep 1.3; "$M" para --pid-arq "$D/mon2.pid"
"$M" avalia --csv "$D/mon2.csv" > "$D/av2.txt"; rc=$?
confere "temperatura 85 C: SEM PROVA (código 1)" "[ $rc = 1 ] && grep -q 'temp_max=85' $D/av2.txt"
# 9. uso
"$M" 2> /dev/null; confere "sem subcomando: código 2" "[ $? = 2 ]"
"$M" portao 2> /dev/null; confere "portão sem --saida: código 2" "[ $? = 2 ]"
para_gerador
if [ "$falhas" = 0 ]; then echo "TODOS OS TESTES PASSARAM"; exit 0; else echo "$falhas FALHA(S)"; exit 1; fi
