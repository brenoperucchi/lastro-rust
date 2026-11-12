#!/usr/bin/env bash
# Teste do envelope janela-gpu-strata (janela-gpu-strata-f2e.sh, copia versionada do ~/janela-gpu-strata.sh) com SERVICOS SIMULADOS:
# systemctl e nvidia-smi falsos (JGS_SYSTEMCTL, JGS_NVIDIA_SMI, JGS_CGROUP_RAIZ). Nao toca o strata.service real, nem GPU, nem systemd.
# Nao apaga nada: a pasta temporaria (sob TMPDIR) e impressa ao fim para limpeza manual com caminho literal.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
ENV_SH=${ENV_SH:-$AQUI/janela-gpu-strata-f2e.sh}
export TMPDIR=${TMPDIR:-$HOME/.cache/lastro-testes}; mkdir -p "$TMPDIR"
BASE=$(mktemp -d) || exit 2
FALHAS=0
ok() { echo "ok   $1"; }
ruim() { echo "FALHA $1"; FALHAS=$((FALHAS + 1)); }
espera() { if [ "$2" = "$3" ]; then ok "$1 (rc $3)"; else ruim "$1: esperado rc $2, veio $3"; fi; }
set -m   # jobs em segundo plano com INT padrao (o shell nao interativo os ignoraria)

cat > "$BASE/systemctl" <<'FAKE'
#!/usr/bin/env bash
# systemctl falso. Estado em $FS. Argumentos: --user <acao> [unidade]
acao=$2; alvo=${3:-}
envpid() { cat "$FS/env.pid" 2> /dev/null; }
case "$acao" in
  is-active)
    if [ "$alvo" = strata.service ]; then
      if [ -f "$FS/sinal_no_is_active" ] && grep -q '^stop$' "$FS/eventos" 2> /dev/null; then s=$(cat "$FS/sinal_no_is_active"); rm -f "$FS/sinal_no_is_active"; kill -"$s" "$(envpid)"; sleep 0.5; fi
      cat "$FS/strata"
    else
      [ -f "$FS/isactive_ilegivel" ] && { echo "weird"; exit 1; }
      cat "$FS/unit-$alvo" 2> /dev/null || echo inactive
    fi ;;
  stop)
    if [ "$alvo" = strata.service ]; then
      echo stop >> "$FS/eventos"
      [ -f "$FS/stop_falha" ] && exit 1
      echo inactive > "$FS/strata"
      if [ -f "$FS/sinal_no_stop" ]; then s=$(cat "$FS/sinal_no_stop"); kill -"$s" "$(envpid)"; sleep 0.7; fi
    else
      echo "stop-unidade $alvo" >> "$FS/eventos"
      [ -f "$FS/unidade_teimosa" ] && exit 0
      [ -f "$FS/worker.pid" ] && kill "$(cat "$FS/worker.pid")" 2> /dev/null
      sleep 0.3
      : > "$FS/cg/$alvo/cgroup.procs"; : > "$FS/units"; echo inactive > "$FS/unit-$alvo"
    fi ;;
  start)
    vivo=0; [ -f "$FS/worker.pid" ] && kill -0 "$(cat "$FS/worker.pid")" 2> /dev/null && vivo=1
    echo "start worker_vivo=$vivo" >> "$FS/eventos"
    [ -f "$FS/start_falha" ] && exit 1
    [ -f "$FS/start_sem_estado" ] && exit 0
    echo active > "$FS/strata" ;;
  list-units)
    [ -f "$FS/list_falha" ] && exit 1
    if [ -f "$FS/list_falha_2a" ]; then n=$(cat "$FS/listcount" 2> /dev/null || echo 0); n=$((n + 1)); echo $n > "$FS/listcount"; [ "$n" -ge 2 ] && exit 1; fi
    cat "$FS/units" 2> /dev/null ;;
esac
exit 0
FAKE
cat > "$BASE/nvidia-smi" <<'FAKE'
#!/usr/bin/env bash
[ -f "$FS/smi_falha" ] && exit 1
case "$*" in *compute-apps*) [ -f "$FS/smi_apps_falha" ] && exit 1; [ -f "$FS/gpu_ocupada" ] && echo 1234 ;; *memory.used*) [ -f "$FS/smi_mem_falha" ] && exit 1; echo 120 ;; esac
exit 0
FAKE
cat > "$BASE/cmd.sh" <<'FAKE'
#!/usr/bin/env bash
echo ran >> "$FS/cmd.ran"
if [ -n "${COM_UNIDADE:-}" ]; then
  U=lastro-limite-1.service
  mkdir -p "$FS/cg/$U"
  setsid sleep 7771 > /dev/null 2>&1 &
  echo $! > "$FS/worker.pid"; echo $! > "$FS/cg/$U/cgroup.procs"
  echo "$U loaded active running x" > "$FS/units"; echo active > "$FS/unit-$U"
  [ -f "$FS/cg_ilegivel" ] && chmod 000 "$FS/cg/$U/cgroup.procs"
fi
case "${COM_GPU:-}" in ocupada) touch "$FS/gpu_ocupada" ;; falha) touch "$FS/smi_falha" ;; esac
MODO=${COM_MODO:-rc0}
case "$MODO" in
  rc*) exit "${MODO#rc}" ;;
  espera) sleep 60 & wait ;;
esac
FAKE
chmod +x "$BASE/systemctl" "$BASE/nvidia-smi" "$BASE/cmd.sh"

prepara() {
  FS=$BASE/$1; mkdir -p "$FS/home" "$FS/cg"; export FS
  echo "${INICIAL:-active}" > "$FS/strata"
  export HOME=$FS/home JGS_SYSTEMCTL=$BASE/systemctl JGS_NVIDIA_SMI=$BASE/nvidia-smi JGS_CGROUP_RAIZ=$FS/cg
  export JGS_ESPERA_START=${JGS_ESPERA_START:-2} JGS_ESPERA_GPU=1 JGS_ESPERA_UNIDADE=3 JGS_ESPERA_COMANDO=3 JGS_PASSO=0.25
}
lanca() {
  bash -c 'echo $$ > "$FS/env.pid"; exec bash "$@"' _ "$ENV_SH" "$@" > "$FS/saida.txt" 2>&1 &
  EP=$!
}
limpa() { pkill -f "sleep 7771" 2> /dev/null; unset COM_UNIDADE COM_MODO COM_GPU INICIAL; }
alerta_existe() { ls "$FS"/home/lastro-janelas/ATENCAO-STRATA-PARADO-*.txt > /dev/null 2>&1; }
starts() { local n; n=$(grep -c '^start' "$FS/eventos" 2> /dev/null); echo "${n:-0}"; }
log() { cat "$FS"/home/lastro-janelas/*.log 2> /dev/null; }

# 1. caminho normal
prepara n1; lanca --teto 20 -- "$BASE/cmd.sh"; wait $EP; espera "normal: comando rc 0" 0 $?
[ "$(starts)" = 1 ] && [ "$(cat "$FS/strata")" = active ] && ok "normal: Strata religado uma vez" || ruim "normal: starts=$(starts) estado=$(cat "$FS/strata")"
# 2. comando falha: rc preservado, religa
prepara n2; COM_MODO=rc7 lanca --teto 20 -- "$BASE/cmd.sh"; wait $EP; espera "comando rc 7 preservado" 7 $?
[ "$(starts)" = 1 ] && ok "comando falho: religou" || ruim "comando falho: starts=$(starts)"
# 3. P1-1: sinal DURANTE o stop (TERM, HUP, INT): religa, nao inicia o comando, rc 130
for S in TERM HUP INT; do
  prepara s-$S; echo $S > "$FS/sinal_no_stop"; lanca --teto 20 -- "$BASE/cmd.sh"; wait $EP; espera "$S durante o stop" 130 $?
  [ "$(starts)" = 1 ] && [ "$(cat "$FS/strata")" = active ] && ok "$S durante o stop: Strata religado" || ruim "$S durante o stop: starts=$(starts) estado=$(cat "$FS/strata")"
  [ ! -e "$FS/cmd.ran" ] && ok "$S durante o stop: comando nao iniciou" || ruim "$S durante o stop: o comando rodou"
done
# 3b. sinal entre o retorno do stop e a confirmacao de inactive
for S in TERM HUP INT; do
  prepara c-$S; echo $S > "$FS/sinal_no_is_active"; lanca --teto 20 -- "$BASE/cmd.sh"; wait $EP; espera "$S entre o stop e a confirmacao" 130 $?
  { [ "$(starts)" = 1 ] && [ ! -e "$FS/cmd.ran" ]; } && ok "$S entre o stop e a confirmacao: religou, sem comando" || ruim "$S entre: starts=$(starts)"
done
# 3c. falha do stop: aborta rc 3 e religa
prepara f1; touch "$FS/stop_falha"; lanca --teto 20 -- "$BASE/cmd.sh"; wait $EP; espera "stop falha: aborta" 3 $?
[ "$(starts)" = 1 ] && ok "stop falho: tentou religar" || ruim "stop falho: starts=$(starts)"
# 4. estado inicial inactive: nenhum start, comando roda
INICIAL=inactive prepara i1; lanca --teto 20 -- "$BASE/cmd.sh"; wait $EP; espera "inicial inactive: comando roda" 0 $?
{ [ "$(starts)" = 0 ] && ! grep -q '^stop$' "$FS/eventos" 2> /dev/null && [ -e "$FS/cmd.ran" ]; } && ok "inicial inactive: nem stop nem start" || ruim "inicial inactive: starts=$(starts)"
INICIAL=activating prepara i2; lanca --teto 20 -- "$BASE/cmd.sh"; wait $EP; espera "inicial incomum: recusa" 2 $?
# 5. P1-2: start falho / sem estado active: rc 6, log ATENCAO; comando falho preserva
JGS_ESPERA_START=1 prepara r1; touch "$FS/start_falha"; lanca --teto 20 -- "$BASE/cmd.sh"; wait $EP; espera "start falha, comando 0: rc 6" 6 $?
log | grep -q "o Strata nao voltou a active" && ok "start falho: registrado no log" || ruim "start falho: sem ATENCAO no log"
JGS_ESPERA_START=1 prepara r2; touch "$FS/start_sem_estado"; lanca --teto 20 -- "$BASE/cmd.sh"; wait $EP; espera "start rc 0 mas inactive: rc 6" 6 $?
JGS_ESPERA_START=1 prepara r3; touch "$FS/start_falha"; COM_MODO=rc7 lanca --teto 20 -- "$BASE/cmd.sh"; wait $EP; espera "start falha, comando 7: preserva 7" 7 $?
log | grep -q "o Strata nao voltou a active" && ok "comando falho + start falho: falha de religacao no log" || ruim "sem ATENCAO"
# 6. P1-3: TERM/HUP/INT no envelope com a unidade lastro-limite viva: unidade encerrada ANTES do start
for S in TERM HUP INT; do
  prepara u-$S; COM_UNIDADE=1 COM_MODO=espera lanca --teto 30 -- "$BASE/cmd.sh"
  for i in $(seq 1 40); do [ -e "$FS/worker.pid" ] && break; sleep 0.25; done
  sleep 0.5; kill -$S $EP; wait $EP; espera "$S com unidade viva" 130 $?
  grep -q '^start worker_vivo=0' "$FS/eventos" && ok "$S: unidade encerrada e confirmada antes do start" || ruim "$S: $(tail -3 "$FS/eventos" | tr '\n' '|')"
  grep -q "^stop-unidade" "$FS/eventos" && ok "$S: unidade parada pelo envelope" || ruim "$S: unidade nao foi parada"
  limpa
done
# 6b. teto externo estourado com unidade viva
prepara t1; COM_UNIDADE=1 COM_MODO=espera lanca --teto 3 -- "$BASE/cmd.sh"; wait $EP; espera "teto externo estourado" 124 $?
grep -q '^start worker_vivo=0' "$FS/eventos" && ok "teto: unidade encerrada antes do start" || ruim "teto: $(tail -3 "$FS/eventos" | tr '\n' '|')"
limpa
# 7. unidade que nao encerra: NAO religa, rc 5 (ou o rc do comando se != 0), registra
prepara k1; touch "$FS/unidade_teimosa"; COM_UNIDADE=1 lanca --teto 20 -- "$BASE/cmd.sh"; wait $EP; espera "unidade teimosa, comando 0: rc 5" 5 $?
[ "$(starts)" = 0 ] && ok "unidade teimosa: Strata NAO religado" || ruim "unidade teimosa: religou"
log | grep -q "NAO religado automaticamente" && ok "unidade teimosa: registrado" || ruim "unidade teimosa: sem registro"
alerta_existe && ok "unidade teimosa: arquivo de estado criado" || ruim "unidade teimosa: sem arquivo de estado"
limpa
prepara k2; touch "$FS/unidade_teimosa"; COM_UNIDADE=1 COM_MODO=rc7 lanca --teto 20 -- "$BASE/cmd.sh"; wait $EP; espera "unidade teimosa, comando 7: preserva 7" 7 $?
[ "$(starts)" = 0 ] && ok "unidade teimosa + rc 7: nao religou" || ruim "religou"
limpa
# 7b. P1 da rodada 179: confirmacao de termino FALHA FECHADA. Nenhuma incerteza pode dar rc 0 nem start; ha aviso em arquivo de estado
fechada() { # nome, arquivo-flag, [COM_UNIDADE], [COM_GPU]
  prepara "$1"; [ -n "$2" ] && touch "$FS/$2"
  COM_UNIDADE=$3 COM_GPU=$4 lanca --teto 20 -- "$BASE/cmd.sh"; wait $EP; espera "$1: sem prova de termino" 5 $?
  [ "$(starts)" = 0 ] && ok "$1: Strata NAO religado" || ruim "$1: religou (starts=$(starts))"
  alerta_existe && ok "$1: arquivo de estado ATENCAO-STRATA-PARADO criado" || ruim "$1: sem arquivo de estado"
  log | grep -q "ATENCAO" && ok "$1: aviso no log" || ruim "$1: sem aviso no log"
  limpa
}
fechada lf-com-unidade list_falha 1 ""
fechada lf-sem-unidade list_falha "" ""
fechada isactive-ilegivel isactive_ilegivel 1 ""
fechada cgroup-ilegivel cg_ilegivel 1 ""
fechada lista-falha-na-2a list_falha_2a 1 ""
# unidades CONFIRMADAS encerradas: GPU ocupada/ilegivel ao fim NAO impede o start (producao parada e pior); aviso visivel, rc do comando preservado
for G in ocupada falha; do
  prepara gfim-$G; COM_GPU=$G lanca --teto 20 -- "$BASE/cmd.sh"; wait $EP; espera "GPU $G ao fim, unidades limpas: religa" 0 $?
  [ "$(starts)" = 1 ] && ok "GPU $G ao fim: Strata religado" || ruim "GPU $G ao fim: starts=$(starts)"
  log | grep -q "religo o Strata mesmo assim" && ok "GPU $G ao fim: aviso no log" || ruim "GPU $G ao fim: sem aviso"
  limpa
done
prepara gfim-start-falha; touch "$FS/start_falha"; JGS_ESPERA_START=1 COM_GPU=ocupada lanca --teto 20 -- "$BASE/cmd.sh"; wait $EP; espera "GPU ocupada ao fim + start falho: rc 6" 6 $?
limpa
# portao de inicio: GPU ocupada ou nvidia-smi falho DESDE ANTES do comando -> rc 3, comando nao roda, Strata RELIGADO (starts=1), aviso
for F in gpu_ocupada smi_falha smi_apps_falha smi_mem_falha; do
  prepara gini-$F; touch "$FS/$F"; lanca --teto 20 -- "$BASE/cmd.sh"; wait $EP; espera "portao de inicio ($F)" 3 $?
  [ "$(starts)" = 1 ] && [ "$(cat "$FS/strata")" = active ] && ok "$F no inicio: Strata religado" || ruim "$F no inicio: starts=$(starts)"
  [ ! -e "$FS/cmd.ran" ] && ok "$F no inicio: comando nao rodou" || ruim "$F no inicio: o comando rodou"
  log | grep -q "religo o Strata mesmo assim" && ok "$F no inicio: aviso no log" || ruim "$F no inicio: sem aviso"
  limpa
done
# controles: lista vazia consultada com sucesso e unidade encerrada antes do start religam (n1 e u-*); incerteza com comando falho preserva o rc
prepara lf-rc7; touch "$FS/list_falha"; COM_MODO=rc7 lanca --teto 20 -- "$BASE/cmd.sh"; wait $EP; espera "incerteza + comando rc 7: preserva 7" 7 $?
[ "$(starts)" = 0 ] && ok "incerteza + rc 7: nao religou" || ruim "religou"
alerta_existe && ok "incerteza + rc 7: aviso criado" || ruim "sem aviso"
limpa
# 8. dry-run
prepara d1; lanca --dry-run --teto 20 -- "$BASE/cmd.sh"; wait $EP; espera "dry-run" 0 $?
{ [ ! -e "$FS/eventos" ] && [ ! -e "$FS/cmd.ran" ]; } && ok "dry-run: nada parado nem rodado" || ruim "dry-run mexeu"
# 9. recusa de uso
bash "$ENV_SH" > /dev/null 2>&1; espera "sem comando" 2 $?
echo "---"
echo "pasta temporaria deste teste (apagar a mao, caminho literal): $BASE"
[ "$FALHAS" = 0 ] && echo "TODOS OS TESTES PASSARAM" || echo "$FALHAS FALHA(S)"
exit "$FALHAS"
