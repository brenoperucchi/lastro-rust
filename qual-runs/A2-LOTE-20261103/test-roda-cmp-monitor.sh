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
  *dmon*) [ "${FAKE_DMON_RC:-0}" = 0 ] || exit "$FAKE_DMON_RC"
          printf '# gpu    sm   mem   enc   dec   jpg   ofa\n# Idx     %%   %%     %%     %%     %%     %%\n    0   %s     0     0     0     0     0\n    0   %s     0     0     0     0     0\n' "${FAKE_DMON_A:-3}" "${FAKE_DMON_B:-7}";;
  *pmon*) if [ -n "${FAKE_PMON_DIR:-}" ]; then n=$(( $(cat "$FAKE_PMON_DIR/cont" 2> /dev/null || echo 0) + 1 )); echo $n > "$FAKE_PMON_DIR/cont"; cat "$FAKE_PMON_DIR/volta.$n"; exit 0; fi
          [ "${FAKE_PMON_RC:-0}" = 0 ] || { echo "erro falso" >&2; exit "$FAKE_PMON_RC"; }
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
# cada chamada do pmon falso (-c 1) devolve UMA volta, igual a FAKE_PMON_ARQ; as 3 voltas do portão são 3 chamadas
pm() { { printf "%b" "$CAB"; volta "$1" "${2:-}"; } > "$D/pmon.txt"; }
# pmg: só o desktop (PIDs G)
pmg() { { printf "%b" "$CAB"; lin 41533 G "$1" "$1" 413 Hyprland; lin 41609 G 1 - 761 quickshell; } > "$D/pmon.txt"; }
# pmv: voltas diferentes por chamada; cada argumento é o conteúdo (linhas de PID) de uma volta
pmv() { mkdir -p "$D/pv"; echo 0 > "$D/pv/cont"; local n=0 v; for v in "$@"; do n=$((n+1)); { printf "%b" "$CAB"; printf '%b' "$v"; } > "$D/pv/volta.$n"; done; }
roda_pv() { FAKE_PMON_DIR=$D/pv "$M" portao --saida "$1" "${@:2}"; }
roda_pm() { FAKE_PMON_ARQ=$D/pmon.txt "$M" portao --saida "$1" "${@:2}"; }
# 5. emenda 9: só PIDs G (desktop) são isentos; chrome (C+G) e voxtype (C+G) ociosos FALHAM por presença, com a lista dos PIDs/comandos
pm 25
roda_pm "$D/p5.txt"; rc=$?
confere "voxtype e chrome C+G ociosos fora da árvore: SEM PROVA por presença, listando PIDs e comandos e pedindo para fechar" "[ $rc = 1 ] && grep -q '^pmon_antes_veredito=processo.*1352017 (voxtype)' $D/p5.txt && grep -q '42406 (renderD128)' $D/p5.txt && grep -q 'portao=FALHA.*feche-os antes da janela' $D/p5.txt && grep -q '^pmon_antes_g_soma_sm_max=26' $D/p5.txt && grep -q '^pmon_antes_cg_ociosos=2' $D/p5.txt && grep -q 'pmon_antes_pid=1352017 tipo=C+G' $D/p5.txt"
# só desktop G (sem C/C+G): passa; dmon só informação
cat > "$D/pmon.txt" <<P
$(printf "%b" "$CAB")
$(lin 41533 G 25 25 413 Hyprland)
$(lin 41609 G 1 - 761 quickshell)
P
FAKE_DMON_A=4 FAKE_DMON_B=9 roda_pm "$D/p5b.txt"; rc=$?
confere "só PIDs G (Hyprland 25%): portão passa; soma G 26, 3 voltas, dmon registrado (sm máx 9) só como informação" "[ $rc = 0 ] && grep -q '^portao=OK' $D/p5b.txt && grep -q '^pmon_antes_veredito=OK' $D/p5b.txt && grep -q '^pmon_antes_g_soma_sm_max=26' $D/p5b.txt && [ \$(grep -c '^pmon_antes_volta=' $D/p5b.txt) = 3 ] && grep -q '^compute_apps_veredito=OK' $D/p5b.txt && grep -q '^dmon_antes_sm_max=9' $D/p5b.txt"
FAKE_DMON_RC=1 roda_pm "$D/p5c.txt"; rc=$?
confere "dmon quebrado não decide: portão passa e grava ilegivel" "[ $rc = 0 ] && grep -q '^dmon_antes_sm_max=ilegivel' $D/p5c.txt"
# compute-apps com PID estranho (mesmo com o pmon só com G): SEM PROVA, e o da árvore é ignorado
FAKE_APPS='1352017, 997 MiB, /usr/bin/voxtype' roda_pm "$D/p5d.txt"; rc=$?
confere "compute-apps com voxtype fora da árvore (pmon só G): SEM PROVA" "[ $rc = 1 ] && grep -q '^compute_apps_veredito=processo.*1352017 (/usr/bin/voxtype)' $D/p5d.txt"
FAKE_APPS='garbage' roda_pm "$D/p5e.txt"; rc=$?
confere "compute-apps ilegível: falha fechada" "[ $rc = 1 ] && grep -q 'compute-apps ilegível' $D/p5e.txt"
# 6. um PID C com sm 20 fora da árvore: SEM PROVA
pm 25 "$(lin 77777 C 20 5 500 outro)"
roda_pm "$D/p6.txt"; rc=$?
confere "PID C com sm 20% fora da árvore: falha e registra o motivo" "[ $rc = 1 ] && grep -q 'portao=FALHA.*tipo C ou C+G.*77777 (outro)' $D/p6.txt"
pm 25 "$(lin 77777 C+G 6 5 500 outro)"
roda_pm "$D/p6b.txt"; rc=$?
confere "PID C+G com sm 6% fora da árvore: falha" "[ $rc = 1 ]"
pm 25 "$(lin 77777 C 5 5 500 outro)"
roda_pm "$D/p6c.txt"; rc=$?
confere "PID C com sm 5% fora da árvore: falha (emenda 9: presença, sem limite de sm)" "[ $rc = 1 ]"
# 7. PID da árvore medida é ignorado (inclusive descendente)
sleep 60 & ARV=$!
cat > "$D/pmon.txt" <<P
$(printf "%b" "$CAB")
$(lin 41533 G 25 25 413 Hyprland)
$(lin "$ARV" C 90 50 3000 medido)
P
roda_pm "$D/p7.txt" --arvore "$ARV"; rc=$?
confere "PID da árvore medida (C, sm 90%) é ignorado" "[ $rc = 0 ] && grep -q \"dentro_arvore=1 cmd=medido\" $D/p7.txt"
bash -c 'sleep 60 & echo $! > '"$D"'/filho.pid; wait' &
PAI=$!; sleep 0.5; FILHO=$(cat "$D/filho.pid")
cat > "$D/pmon.txt" <<P
$(printf "%b" "$CAB")
$(lin 41533 G 25 25 413 Hyprland)
$(lin "$FILHO" C 90 50 3000 filho)
$(lin 99999 C 30 1 1 fora)
P
roda_pm "$D/p7b.txt" --arvore "$PAI"; rc=$?
confere "descendente da árvore ignorado, mas outro PID C fora dela falha" "[ $rc = 1 ] && grep -q '99999 (fora)' $D/p7b.txt && ! grep -q \"pmon_antes_veredito=.*\$FILHO (\" $D/p7b.txt"
kill "$PAI" "$FILHO" "$ARV" 2> /dev/null; wait "$PAI" "$ARV" 2> /dev/null; ARV=
# 8. soma dos PIDs G (só G no pmon): 71% falha; 60% exatos passa; pico do pmon (soma G), por VOLTA
pmv "$(lin 41533 G 71 71 413 Hyprland)" "$(lin 41533 G 0 0 413 Hyprland)" "$(lin 41533 G 0 0 413 Hyprland)"
roda_pv "$D/p8.txt"; rc=$?
confere "PIDs G somando 71% numa volta: SEM PROVA (pico do pmon)" "[ $rc = 1 ] && grep -q 'pico do pmon (soma G) em 71%' $D/p8.txt"
pmv "$(lin 41533 G 59 59 413 Hyprland; lin 41609 G 1 - 761 quickshell)" "$(lin 41533 G 0 0 413 Hyprland)" "$(lin 41533 G 0 0 413 Hyprland)"
roda_pv "$D/p8b.txt"; rc=$?
confere "PIDs G somando 60% exatos passa (> 60% é o limite)" "[ $rc = 0 ] && grep -q '^pmon_antes_g_soma_sm_max=60' $D/p8b.txt"
# 8c. voltas por captura separada (r114, rev-1): fronteira de volta é do roteiro, não a repetição de PID
pmv "$(lin 101 G 25 25 1 a)" "$(lin 102 G 25 25 1 b)" "$(lin 103 G 25 25 1 c)"
roda_pv "$D/p8c.txt"; rc=$?
confere "voltas com conjuntos DISJUNTOS (PIDs 101/102/103 a 25%): soma por volta 25, máximo 25, passa" "[ $rc = 0 ] && grep -q '^pmon_antes_g_soma_sm_max=25' $D/p8c.txt && grep -q 'pmon_antes_pid=102 tipo=G sm=25' $D/p8c.txt"
pmv "$(lin 101 G 20 20 1 a; lin 102 G 10 10 1 b)" "$(lin 102 G 15 15 1 b; lin 103 G 30 30 1 c)" "$(lin 103 G 5 5 1 c; lin 101 G 25 25 1 a)"
roda_pv "$D/p8d.txt"; rc=$?
confere "voltas parcialmente sobrepostas: soma por volta 30/45/30, máximo 45; máximo por PID 25/15/30" "[ $rc = 0 ] && grep -q '^pmon_antes_g_soma_sm_max=45' $D/p8d.txt && grep -q 'pmon_antes_pid=101 tipo=G sm=25' $D/p8d.txt && grep -q 'pmon_antes_pid=102 tipo=G sm=15' $D/p8d.txt && grep -q 'pmon_antes_pid=103 tipo=G sm=30' $D/p8d.txt"
pmv "$(lin 201 G 20 20 1 n1; lin 101 G 10 10 1 a)" "$(lin 202 G 20 20 1 n2; lin 101 G 10 10 1 a)" "$(lin 101 G 10 10 1 a; lin 203 G 20 20 1 n3)"
roda_pv "$D/p8e.txt"; rc=$?
confere "PIDs novos antes dos persistentes, ordem trocada entre voltas: soma por volta 30 (e não 30+30+30), máximo 30" "[ $rc = 0 ] && grep -q '^pmon_antes_g_soma_sm_max=30' $D/p8e.txt"
pmv "$(lin 101 G 30 30 1 a; lin 102 G 31 31 1 b)" "$(lin 101 G 1 1 1 a)" "$(lin 102 G 1 1 1 b)"
roda_pv "$D/p8f.txt"; rc=$?
confere "uma volta efetiva de soma 61: máximo 61, falha" "[ $rc = 1 ] && grep -q '^pmon_antes_g_soma_sm_max=61' $D/p8f.txt && grep -q 'pico do pmon (soma G) em 61%' $D/p8f.txt"
# volta sem cabeçalho: falha fechada; uma das três chamadas com erro: falha fechada
mkdir -p "$D/pv"; echo 0 > "$D/pv/cont"; cp "$D/pv/volta.1" "$D/pv/volta.1.bak" 2> /dev/null
printf '%s\n' 'sem cabecalho' > "$D/pv/volta.2"
roda_pv "$D/p8g.txt"; rc=$?
confere "uma volta sem cabeçalho: falha fechada" "[ $rc = 1 ] && grep -q 'sem cabeçalho do pmon na volta 2' $D/p8g.txt"
echo 0 > "$D/pv/cont"; rm -f "$D/pv/volta.3"
roda_pv "$D/p8h.txt"; rc=$?
confere "uma das três capturas falha (arquivo ausente na 3ª): falha fechada" "[ $rc = 1 ]"
# 9. utilização total só registrada (46% não decide)
pmg 25
FAKE_UTIL=46 roda_pm "$D/p9.txt"; rc=$?
confere "utilização total da GPU 46% só registrada, não decide" "[ $rc = 0 ] && grep -q '^gpu_util_pct_registrado=46' $D/p9.txt"
# 10. pmon ilegível ou ausente: SEM PROVA fechado
FAKE_PMON_RC=1 "$M" portao --saida "$D/p10.txt"; rc=$?
confere "pmon com erro (código 1): falha fechada" "[ $rc = 1 ] && grep -q 'pmon falhou na volta 1' $D/p10.txt"
printf 'lixo sem cabecalho\n' > "$D/pmon.txt"; roda_pm "$D/p10b.txt"; rc=$?
confere "pmon sem cabeçalho: falha fechada" "[ $rc = 1 ] && grep -q 'pmon ilegível' $D/p10b.txt"
{ printf "%b" "$CAB"; printf '    0   41533     G    abc      -      -      -      -      -    413      0    Hyprland\n'; } > "$D/pmon.txt"; roda_pm "$D/p10c.txt"; rc=$?
confere "linha de PID ilegível: falha fechada" "[ $rc = 1 ] && grep -q 'ilegíve' $D/p10c.txt"
NVIDIA_SMI=$D/nao-existe "$M" portao --saida "$D/p10d.txt"; rc=$?
confere "nvidia-smi ausente: falha fechada" "[ $rc = 1 ] && grep -q 'pmon falhou' $D/p10d.txt"
# 11. depois da célula: só registra (código 0 mesmo com PID C), grava o veredito e a soma G
pm 25 "$(lin 77777 C 20 5 500 outro)"
FAKE_PMON_ARQ=$D/pmon.txt "$M" portao-depois --saida "$D/pd.txt"; rc=$?
confere "portao-depois: código 0, registra soma G e veredito, não decide" "[ $rc = 0 ] && grep -q '^pmon_depois_g_soma_sm_max=26' $D/pd.txt && grep -q '^pmon_depois_veredito=processo' $D/pd.txt"
FAKE_PMON_RC=1 "$M" portao-depois --saida "$D/pd2.txt"; rc=$?
confere "portao-depois com pmon quebrado: código 0 e registra" "[ $rc = 0 ] && grep -q '^pmon_depois_rc=1' $D/pd2.txt"
# 11b. preflight da janela (emenda 9): nada medido roda, qualquer C ou C+G falha; só G passa
pm 25 "$(lin 42406 C+G - - 364 'renderD128 --cr')"
FAKE_PMON_ARQ=$D/pmon.txt "$M" preflight > "$D/pf1.txt"; rc=$?
confere "preflight com C+G (chrome) presente: código 1 listando PID e comando" "[ $rc = 1 ] && grep -q 'preflight FALHOU.*42406 (renderD128)' $D/pf1.txt"
pmg 25
FAKE_PMON_ARQ=$D/pmon.txt "$M" preflight > "$D/pf2.txt"; rc=$?
confere "preflight com só PIDs G (voxtype e chrome fechados): código 0" "[ $rc = 0  ] && grep -q 'nenhum PID C ou C+G' $D/pf2.txt"
FAKE_PMON_ARQ=$D/pmon.txt FAKE_APPS='1352017, 997 MiB, /usr/bin/voxtype' "$M" preflight > "$D/pf3.txt"; rc=$?
confere "preflight: voxtype só em compute-apps: código 1" "[ $rc = 1 ] && grep -q '1352017 (/usr/bin/voxtype)' $D/pf3.txt"
FAKE_PMON_RC=1 "$M" preflight > "$D/pf4.txt"; rc=$?
confere "preflight com pmon quebrado: falha fechada" "[ $rc = 1 ]"
# 11c. GPU sem processo (3080 Ti sem desktop): cabeçalhos com fb/ccpm + UMA linha placeholder com tudo '-'; é "nenhum processo", não ilegível
PH='    0          -     -      -      -      -      -      -      -      -      -    -\n'
pmv "$PH" "$PH" "$PH"; roda_pv "$D/p11c.txt"; rc=$?
confere "GPU sem processo (placeholder nas 3 voltas): portão passa, sem PID C" "[ $rc = 0 ] && grep -q '^pmon_antes_veredito=OK' $D/p11c.txt && grep -q '^pmon_antes_c_fora_n=0' $D/p11c.txt && ! grep -q 'pmon_antes_pid=' $D/p11c.txt"
pmv "$PH" "$PH" "$PH"; FAKE_PMON_DIR=$D/pv "$M" preflight > "$D/pf5.txt"; rc=$?
confere "preflight com GPU sem processo: veredito passa (código 0)" "[ $rc = 0 ] && grep -q 'nenhum PID C ou C+G' $D/pf5.txt"
pmv "$PH" "$(lin 77777 C 0 0 500 outro)\n" "$PH"; roda_pv "$D/p11d.txt"; rc=$?
confere "placeholder numa volta e PID C real noutra: detecta o C" "[ $rc = 1 ] && grep -q '77777 (outro)' $D/p11d.txt"
pmv "$PH" "    0          -     -      -      -      -      -      -      -      -      x    -\n" "$PH"; roda_pv "$D/p11e.txt"; rc=$?
confere "linha com pid '-' mas campo estranho (não é o placeholder): ilegível, recusa" "[ $rc = 1 ] && grep -q 'pmon ilegível.*ilegível' $D/p11e.txt"
pmv "$PH" "    0          -     -      -\n" "$PH"; roda_pv "$D/p11f.txt"; rc=$?
confere "placeholder truncado (poucas colunas): ilegível, recusa" "[ $rc = 1 ] && grep -q 'ilegível' $D/p11f.txt"
mkdir -p "$D/pw"; echo 0 > "$D/pw/cont"; for n in 1 2 3; do printf '%b' "$PH" > "$D/pw/volta.$n"; done
FAKE_PMON_DIR=$D/pw "$M" portao --saida "$D/p11g.txt"; rc=$?
confere "placeholder sem cabeçalho: recusa (falha fechada)" "[ $rc = 1 ] && grep -q 'sem cabeçalho' $D/p11g.txt"
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
