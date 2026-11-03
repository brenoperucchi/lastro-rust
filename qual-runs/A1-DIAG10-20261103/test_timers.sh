#!/bin/bash
# Teste do mecanismo de timers do rodar.sh com um systemctl falso (nada do sistema é tocado): estado inicial ativo e inativo, e saída
# por falha. Em cada caso o estado final tem de ser o inicial: o que estava ativo volta a ser ativado, o que estava parado NÃO é ativado.
set -u
cd /home/brenoperucchi/Devs/lastro-rust
D=$(mktemp -d /tmp/claude-1000/-home-brenoperucchi-Devs-lastro-rust/46cdbe1b-904f-422b-9a0c-ab7628b69d27/scratchpad/timers.XXXXXX)
cat > "$D/fake-systemctl" <<'F'
#!/bin/bash
# uso: fake-systemctl --user <ação> <unidade>; o estado de cada unidade é um arquivo em $FAKE_DIR/<unidade>
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
caso() { # nome ativo_a ativo_b
  export FAKE_DIR=$D/$1; mkdir -p "$FAKE_DIR"; rm -f "$FAKE_DIR"/*
  echo "$2" > "$FAKE_DIR/a.timer"; echo "$3" > "$FAKE_DIR/b.timer"; echo inactive > "$FAKE_DIR/a.service"; echo inactive > "$FAKE_DIR/b.service"
  SYSTEMCTL="$D/fake-systemctl" TIMERS_FILE="$D/timers.txt" MEDIDA_SO_TIMERS=1 MEDIDA_DIR="$D/out-$1" qual-runs/A1-DIAG10-20261103/rodar.sh 1 > "$D/saida-$1.txt" 2>&1
  fa=$(cat "$FAKE_DIR/a.timer"); fb=$(cat "$FAKE_DIR/b.timer")
  if [ "$fa" = "$2" ] && [ "$fb" = "$3" ]; then echo "caso $1: estado final igual ao inicial ($fa, $fb): ok"; else echo "caso $1: FALHOU (inicial $2/$3, final $fa/$fb)"; ok=0; fi
  grep -q "stop a.timer" "$FAKE_DIR/log" 2>/dev/null && [ "$2" = inactive ] && { echo "caso $1: parou um timer que já estava parado: FALHOU"; ok=0; }
  grep -q "start" "$FAKE_DIR/log" 2>/dev/null && { [ "$2" = inactive ] && grep -q "start a.timer" "$FAKE_DIR/log"; } && { echo "caso $1: ativou um timer que estava parado: FALHOU"; ok=0; }
}
caso ativos active active
caso parados inactive inactive
caso misto active inactive
[ $ok = 1 ] && echo "TODOS OS CASOS PASSAM" || { echo "HÁ FALHA"; exit 1; }
