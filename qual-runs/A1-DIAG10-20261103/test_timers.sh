#!/bin/bash
# Teste do mecanismo de timers do rodar.sh com um systemctl falso (nada do sistema é tocado). Casos: saída normal com estado inicial ativo, parado e misto; saída por
# falha (serviço de timer que não termina); saída por sinal TERM no meio. Em todos o estado final tem de ser o inicial: o que estava ativo volta a ser ativado e o
# que estava parado NÃO é ativado. Um caso extra confere que o teste PEGA a falha: um rodar.sh sem o trap de restauração tem de reprovar o caso ativo.
set -u
cd /home/brenoperucchi/Devs/lastro-rust || exit 1
D=$(mktemp -d) || exit 1
[ -d "$D" ] || exit 1
R=qual-runs/A1-DIAG10-20261103/rodar.sh
cat > "$D/fake-systemctl" <<'F'
#!/bin/bash
# fake-systemctl --user <ação> <unidade>; o estado de cada unidade é um arquivo em $FAKE_DIR/<unidade>
[ "$1" = --user ] && shift
a=$1; u=${2:-}
case $a in
  is-active) [ -e "$FAKE_DIR/$u" ] && cat "$FAKE_DIR/$u" || echo inactive; [ "$(cat "$FAKE_DIR/$u" 2>/dev/null)" = active ];;
  is-enabled) echo enabled;;
  stop) echo inactive > "$FAKE_DIR/$u"; echo "stop $u" >> "$FAKE_DIR/log";;
  start) echo active > "$FAKE_DIR/$u"; echo "start $u" >> "$FAKE_DIR/log";;
esac
F
chmod +x "$D/fake-systemctl"
printf 'a.timer\nb.timer\n' > "$D/timers.txt"
ok=1
prepara() { # nome ativo_a ativo_b serviço_a
  export FAKE_DIR=$D/$1; mkdir -p "$FAKE_DIR"
  echo "$2" > "$FAKE_DIR/a.timer"; echo "$3" > "$FAKE_DIR/b.timer"; echo "${4:-inactive}" > "$FAKE_DIR/a.service"; echo inactive > "$FAKE_DIR/b.service"; : > "$FAKE_DIR/log"
}
confere() { # nome ativo_a ativo_b
  fa=$(cat "$FAKE_DIR/a.timer"); fb=$(cat "$FAKE_DIR/b.timer")
  if [ "$fa" = "$2" ] && [ "$fb" = "$3" ]; then echo "caso $1: estado final igual ao inicial ($fa, $fb): ok"; else echo "caso $1: FALHOU (inicial $2/$3, final $fa/$fb)"; ok=0; fi
  if [ "$2" = inactive ] && grep -q "start a.timer" "$FAKE_DIR/log"; then echo "caso $1: ativou um timer que estava parado: FALHOU"; ok=0; fi
}
env_comum() { SYSTEMCTL="$D/fake-systemctl" TIMERS_FILE="$D/timers.txt" MEDIDA_SO_TIMERS=1 MEDIDA_DIR="$D/out-$1" "${@:2}"; }
normal() { prepara "$1" "$2" "$3"; env_comum "$1" "$R" 1 > "$D/saida-$1.txt" 2>&1; confere "$1" "$2" "$3"; }
normal ativos active active
normal parados inactive inactive
normal misto active inactive
# saída por falha: o serviço do timer a nunca termina; a espera é encurtada
prepara falha active active active
MEDIDA_ESPERA_TIMERS=2 MEDIDA_PASSO_TIMERS=0.1 env_comum falha "$R" 1 > "$D/saida-falha.txt" 2>&1; rc=$?
[ $rc -ne 0 ] || { echo "caso falha: devia sair com erro"; ok=0; }
confere falha active active
# sinal TERM no meio
prepara sinal active inactive
# direto (env faz exec), para o PID ser o do próprio roteiro e o TERM chegar nele
env SYSTEMCTL="$D/fake-systemctl" TIMERS_FILE="$D/timers.txt" MEDIDA_SO_TIMERS=1 MEDIDA_SEGURA=1 MEDIDA_DIR="$D/out-sinal" "$R" 1 > "$D/saida-sinal.txt" 2>&1 &
pid=$!
for k in $(seq 1 100); do grep -q PARADOS "$D/saida-sinal.txt" 2>/dev/null && break; sleep 0.1; done
[ "$(cat "$FAKE_DIR/a.timer")" = inactive ] || { echo "caso sinal: o timer ativo não foi parado antes do sinal: FALHOU"; ok=0; }
kill -TERM $pid; wait $pid 2>/dev/null
confere sinal active inactive
# o teste pega a falta de restauração: uma cópia do roteiro com a função arma_restauracao sem o trap tem de reprovar o caso ativo (a função serve aos dois caminhos)
sed 's/^arma_restauracao() { trap .*$/arma_restauracao() { :; }/' "$R" > "$D/rodar-sem-trap.sh"
if [ "$(diff "$R" "$D/rodar-sem-trap.sh" | wc -l)" -gt 0 ]; then
  prepara mutante active active
  chmod +x "$D/rodar-sem-trap.sh"
  env_comum mutante "$D/rodar-sem-trap.sh" 1 > /dev/null 2>&1
  if [ "$(cat "$FAKE_DIR/a.timer")" = inactive ]; then echo "mutante sem trap: reprovado como esperado: ok"; else echo "mutante sem trap NÃO foi pego: FALHOU"; ok=0; fi
else
  echo "mutante: o sed não alterou o roteiro: FALHOU"; ok=0
fi
# teto de temperatura (rodada 36): libera só com leitura válida no teto ou abaixo; timeout e leitura ausente não liberam
teto() { # nome temperatura rc_esperado
  echo 70 > "$D/teto.txt"; echo "$2" > "$D/temp.txt"
  r=$(SYSTEMCTL="$D/fake-systemctl" TETO_FILE="$D/teto.txt" TEMP_FILE="$D/temp.txt" MEDIDA_SO_OCIOSA=1 MEDIDA_OCIOSA=0 MEDIDA_ESPERA_TETO=2 MEDIDA_PASSO_TETO=0.05 MEDIDA_DIR="$D/out-$1" TIMERS_FILE="$D/timers.txt" "$R" 1 2>&1 | grep '^rc=')
  if [ "$r" = "rc=$3" ]; then echo "teto $1 (temperatura $2): $r ok"; else echo "teto $1 (temperatura $2): FALHOU ($r, esperado rc=$3)"; ok=0; fi
}
teto abaixo 60.0 0
teto igual 70.0 0
teto acima 90.0 1
teto semleitura n/d 1
teto malformada 1bad 1
teto texto abc 1
teto nan nan 1
teto inf inf 1
# sem teto só registra
r=$(SYSTEMCTL="$D/fake-systemctl" TETO_FILE="$D/inexistente" TEMP_FILE="$D/temp.txt" MEDIDA_SO_OCIOSA=1 MEDIDA_OCIOSA=0 MEDIDA_DIR="$D/out-semteto" TIMERS_FILE="$D/timers.txt" "$R" 1 2>&1 | grep '^rc=')
[ "$r" = "rc=0" ] && echo "sem teto: libera ok" || { echo "sem teto: FALHOU ($r)"; ok=0; }
# a lista de timers é decisão do Breno: sem o arquivo a janela não roda (falha fechado), e a etapa 2 exige a etapa 1 aprovada
out=$(TIMERS_FILE="$D/nao-existe.txt" MEDIDA_DIR="$D/out-semlista" "$R" 1 2>&1); rc=$?
if [ $rc -ne 0 ] && echo "$out" | grep -q "falta .*nao-existe.txt" && [ ! -e "$D/out-semlista/e1" ]; then echo "sem timers.txt: sai antes de tocar em qualquer coisa: ok"; else echo "sem timers.txt: FALHOU (rc=$rc)"; ok=0; fi
out=$(TIMERS_FILE="$D/timers.txt" MEDIDA_DIR="$D/out-sem-e1" "$R" 2 2>&1); rc=$?
if [ $rc -ne 0 ] && echo "$out" | grep -q "a etapa 1 não aprovou" && [ ! -e "$D/out-sem-e1/e2" ]; then echo "etapa 2 sem a etapa 1 aprovada: sai antes de tocar em qualquer coisa: ok"; else echo "etapa 2 sem etapa 1: FALHOU (rc=$rc)"; ok=0; fi
[ $ok = 1 ] && echo "TODOS OS CASOS PASSAM" || { echo "HÁ FALHA"; exit 1; }
