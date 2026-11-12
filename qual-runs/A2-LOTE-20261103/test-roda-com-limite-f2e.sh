#!/usr/bin/env bash
# Teste do roda-com-limite-f2e.sh (copia com cancelamento por sinal) com COMANDOS DE MENTIRA. Sem GPU (nenhuma sonda CUDA/nvidia-smi), sem guardas de autorização, sem efeito
# persistente: só unidades transitórias de `systemd --user` que somem sozinhas (--collect), e o teste confere que nenhuma sobrou.
set -u
unset E4B_GPU_AUTORIZADO E4B_MEDIDA_AUTORIZADA MOE_RT_REQUIRE_GPU LEITURA_CORRECAO_AUTORIZADA
export CUDA_VISIBLE_DEVICES=
AQUI=$(cd "$(dirname "$0")" && pwd -P)
R="$AQUI/roda-com-limite-f2e.sh"
FALHAS=0
ok() { echo "ok   $1"; }
ruim() { echo "FALHA $1"; FALHAS=$((FALHAS + 1)); }
espera() { # nome esperado obtido
  if [ "$2" = "$3" ]; then ok "$1 (código $3)"; else ruim "$1: esperado $2, veio $3"; fi
}
T=$(mktemp -d) || exit 2
trap 'cd /tmp && rm -rf -- "$(basename "$T")"' EXIT

# 1. caminhos normais
"$R" --limite 20 -- sh -c 'exit 0' >/dev/null 2>&1; espera "termina com 0" 0 $?
"$R" --limite 20 -- sh -c 'exit 7' >/dev/null 2>&1; espera "propaga o código do comando" 7 $?
SAI=$(CUDA_VISIBLE_DEVICES=vazio-de-teste "$R" --limite 20 -- sh -c 'echo "cvd=$CUDA_VISIBLE_DEVICES"' 2>/dev/null)
[ "$SAI" = "cvd=vazio-de-teste" ] && ok "ambiente listado vai por --setenv" || ruim "ambiente não chegou: '$SAI'"
SAI=$(SEGREDO_QUALQUER=x "$R" --limite 20 -- sh -c 'echo "s=${SEGREDO_QUALQUER:-}"' 2>/dev/null)
[ "$SAI" = "s=" ] && ok "variável fora da lista não é herdada" || ruim "vazou variável fora da lista: '$SAI'"

# 2. estoura o limite
t0=$(date +%s)
"$R" --limite 2 -- sleep 30 >/dev/null 2>&1; espera "passa do limite" 124 $?
dt=$(( $(date +%s) - t0 ))
[ "$dt" -le 15 ] && ok "encerrou em ${dt}s" || ruim "demorou ${dt}s para encerrar"

# 3. ignora SIGTERM: precisa do SIGKILL depois do TimeoutStopSec
LIMITE_TIMEOUT_STOP=2 "$R" --limite 2 -- sh -c 'trap "" TERM; while :; do sleep 1; done' >/dev/null 2>&1; espera "ignora TERM (vai a SIGKILL)" 124 $?

# 4. filho em outra sessão (setsid) tem de morrer junto
"$R" --limite 2 -- sh -c 'setsid sleep 4242 & wait' >/dev/null 2>&1; espera "filho em outra sessão: limite" 124 $?
if pgrep -f "[s]leep 4242" >/dev/null; then ruim "sobrou 'sleep 4242' depois do limite"; pkill -x -f "sleep 4242"; else ok "nenhum filho remanescente"; fi

# 5. recusas
"$R" -- >/dev/null 2>&1; espera "sem comando" 2 $?
"$R" sleep 1 >/dev/null 2>&1; espera "sem --" 2 $?
"$R" --limite 0 -- true >/dev/null 2>&1; espera "limite 0" 2 $?
"$R" --limite 1801 -- true >/dev/null 2>&1; espera "limite acima de 30 min" 2 $?
"$R" --limite abc -- true >/dev/null 2>&1; espera "limite não numérico" 2 $?
"$R" --limite 5 -- comando-que-nao-existe-xyz >/dev/null 2>&1; espera "comando inexistente" 2 $?
SYSTEMD_RUN=systemd-run-que-nao-existe "$R" --limite 5 -- true >/dev/null 2>&1; espera "sem systemd-run" 2 $?

# 5b. o cgroup que o roteiro confere é o da unidade (systemd real): o comando imprime o próprio cgroup
CG_RAIZ_REAL=/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/app.slice
SAI=$("$R" --limite 20 -- sh -c 'cat /proc/self/cgroup' 2>/dev/null)
case $SAI in
  *"${CG_RAIZ_REAL#/sys/fs/cgroup}/lastro-limite-"*".service"*) ok "o cgroup da unidade fica sob o caminho conferido ($SAI)";;
  *) ruim "cgroup da unidade fora do caminho conferido: '$SAI'";;
esac
CGROUP_RAIZ=/caminho/que/nao/existe "$R" --limite 5 -- true >/dev/null 2>&1; espera "cgroup esperado inexistente: recusa antes de lançar" 2 $?

# 6. systemd-run de mentira: caminhos que o real não produz sob demanda
cat > "$T/falso.sh" <<'EOF'
#!/usr/bin/env bash
# imita o systemd-run: acha a unidade, escreve as mensagens do systemd em stderr e age conforme FALSO_MODO
for a in "$@"; do case $a in --unit=*) U=${a#--unit=};; esac; done
[ "${FALSO_MODO:-}" = nao-inicia ] && { echo "Failed to start transient service unit" >&2; exit 1; }
echo "Running as unit: $U.service" >&2
case ${FALSO_MODO:-} in
  timeout) echo "          Finished with result: timeout" >&2; exit 1;;
  estranho) echo "          Finished with result: resource" >&2; exit 1;;
  remanescente)
    mkdir -p "$CGROUP_RAIZ/$U.service"
    echo "$FALSO_PID" > "$CGROUP_RAIZ/$U.service/cgroup.procs"
    echo "          Finished with result: timeout" >&2; exit 1;;
  nao-inicia) exit 1;;
esac
EOF
chmod +x "$T/falso.sh"
mkdir -p "$T/cg"
FALSO_MODO=timeout SYSTEMD_RUN="$T/falso.sh" CGROUP_RAIZ="$T/cg" "$R" --limite 5 -- true >/dev/null 2>&1; espera "falso: timeout" 124 $?
FALSO_MODO=estranho SYSTEMD_RUN="$T/falso.sh" CGROUP_RAIZ="$T/cg" "$R" --limite 5 -- true >/dev/null 2>&1; espera "falso: resultado desconhecido" 7 $?
FALSO_MODO=nao-inicia SYSTEMD_RUN="$T/falso.sh" CGROUP_RAIZ="$T/cg" "$R" --limite 5 -- true >/dev/null 2>&1; espera "falso: unidade não iniciou" 125 $?
sleep 60 & VIVO=$!
SAI=$(FALSO_MODO=remanescente FALSO_PID=$VIVO LIMITE_ESPERA_CGROUP=1 SYSTEMD_RUN="$T/falso.sh" CGROUP_RAIZ="$T/cg" "$R" --limite 5 -- true 2>&1 >/dev/null); RC=$?
espera "falso: processo remanescente no cgroup" 5 $RC
case $SAI in *"pid $VIVO"*) ok "relata o PID remanescente";; *) ruim "não relatou o PID: $SAI";; esac
kill "$VIVO" 2>/dev/null; wait "$VIVO" 2>/dev/null

set -m   # jobs em segundo plano com INT padrao
# 6b. CANCELAMENTO por sinal (F2E): TERM, HUP e INT no roteiro encerram a unidade transitoria e so devolvem o controle com o cgroup vazio
for SIG in TERM HUP INT; do
  "$R" --limite 60 -- sh -c 'setsid sleep 4343 & wait' >/dev/null 2>&1 &
  RP=$!
  sleep 3
  pgrep -f "[s]leep 4343" >/dev/null || ruim "$SIG: o trabalhador nao subiu"
  t0=$(date +%s)
  kill -$SIG "$RP"; wait "$RP"; RC=$?
  dt=$(( $(date +%s) - t0 ))
  espera "sinal $SIG: sai 130 depois de cancelar" 130 $RC
  [ "$dt" -le 20 ] && ok "$SIG: cancelou em ${dt}s" || ruim "$SIG: demorou ${dt}s"
  if pgrep -f "[s]leep 4343" >/dev/null; then ruim "$SIG: sobrou 'sleep 4343' (religar o Strata seria inseguro)"; pkill -f "sleep 4343"; else ok "$SIG: nenhum processo remanescente ao devolver o controle"; fi
done

# 7. nenhum efeito persistente
RESTO=$(systemctl --user list-units --all --no-legend 'lastro-limite-*' 2>/dev/null | wc -l)
[ "$RESTO" = 0 ] && ok "nenhuma unidade lastro-limite-* sobrou" || ruim "$RESTO unidade(s) sobraram"
if grep -nE 'nvidia-smi|cuda|enable|install|timer' "$R" | grep -vE '^[0-9]+:#|CUDA_VISIBLE_DEVICES|LEITURA_CORRECAO|nada é instalado|timers|TIMEOUT|RuntimeMaxSec' >/dev/null; then ruim "o roteiro menciona algo fora do escopo"; else ok "roteiro sem GPU, sem enable/install"; fi

echo "---"
[ "$FALHAS" = 0 ] && echo "TUDO OK" || echo "$FALHAS FALHA(S)"
exit "$FALHAS"
