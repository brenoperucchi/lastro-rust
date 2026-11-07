#!/bin/bash
# Teste de janela-e4b.sh e recupera-timers-e4b.sh com um systemctl FALSO e comandos falsos: nada do sistema real é tocado
# (o systemctl real nunca é chamado: SYSTEMCTL aponta para o falso em todos os casos).
# Casos: estados iniciais ativo, parado e misto; comando que falha; serviço de timer ocupado; limite de tempo com TERM e com KILL
# (comando que ignora TERM); TERM no próprio roteiro; SIGKILL no roteiro + recuperação manual; restauração que falha; recusas antes de
# mexer em qualquer coisa; estado desconhecido (sem barramento); nada de disable/mask; mutantes que o teste TEM de pegar.
# NÃO coberto: processo em estado D que nem o KILL derruba (não há como simulá-lo sem privilégio); o caminho existe no código
# (o vigia desiste e derruba o roteiro, saída 5) e foi lido, não executado.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
J=$AQUI/janela-e4b.sh
R=$AQUI/recupera-timers-e4b.sh
D=$(mktemp -d -p "${TMPDIR:-/tmp}" janela-e4b-teste.XXXXXX) || exit 1
[ -d "$D" ] || exit 1
echo "pasta de teste: $D"
cat > "$D/fake-systemctl" <<'F'
#!/bin/bash
# fake-systemctl --user <ação> <unidade>; o estado de cada unidade é um arquivo em $FAKE_DIR/<unidade>
[ "$1" = --user ] && shift
a=$1; u=${2:-}
echo "$a $u XDG=${XDG_RUNTIME_DIR:-vazio}" >> "$FAKE_DIR/log"
case $a in
  is-active) if [ -n "${FAKE_SEM_BARRAMENTO:-}" ]; then exit 1; fi
             [ -e "$FAKE_DIR/$u" ] && cat "$FAKE_DIR/$u" || echo inactive; [ "$(cat "$FAKE_DIR/$u" 2>/dev/null)" = active ];;
  is-enabled) echo enabled;;
  cat) [ -e "$FAKE_DIR/$u" ];;
  stop) echo inactive > "$FAKE_DIR/$u";;
  start) if [ "${FAKE_START_FALHA:-}" = "$u" ]; then echo "falha simulada em $u" >&2; exit 1; fi; echo active > "$FAKE_DIR/$u";;
  *) echo "ação inesperada: $a" >> "$FAKE_DIR/log"; exit 9;;
esac
F
chmod +x "$D/fake-systemctl"
printf '# lista de teste\na.timer\nb.timer\n' > "$D/timers.txt"
ok=1
falha() { echo "FALHOU: $*"; ok=0; }
passou() { echo "ok: $*"; }

prepara() { # nome ativo_a ativo_b [serviço_a]
  N=$1; export FAKE_DIR=$D/$1; mkdir -p "$FAKE_DIR"
  echo "$2" > "$FAKE_DIR/a.timer"; echo "$3" > "$FAKE_DIR/b.timer"; echo "${4:-inactive}" > "$FAKE_DIR/a.service"; echo inactive > "$FAKE_DIR/b.service"; : > "$FAKE_DIR/log"
  PEND=$D/pend-$1; rm -f "$PEND"
}
roda() { # args da janela; usa N, FAKE_DIR, PEND
  SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" JANELA_PASSO_ESPERA=0.1 XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR_TESTE:-/run/user/fake}" "${JANELA:-$J}" --timers "$D/timers.txt" "$@"
}
roda_exec() { # como roda, mas com exec: em segundo plano o PID é o do próprio roteiro
  SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" JANELA_PASSO_ESPERA=0.1 XDG_RUNTIME_DIR=/run/user/fake exec "$J" --timers "$D/timers.txt" "$@"
}
confere() { # nome ativo_a ativo_b
  fa=$(cat "$FAKE_DIR/a.timer"); fb=$(cat "$FAKE_DIR/b.timer")
  if [ "$fa" = "$2" ] && [ "$fb" = "$3" ]; then passou "$1: estado final igual ao inicial ($fa, $fb)"; else falha "$1: inicial $2/$3, final $fa/$fb"; fi
  if [ "$2" = inactive ] && grep -q "^start a.timer" "$FAKE_DIR/log"; then falha "$1: iniciou um timer que estava parado"; fi
  if [ "$3" = inactive ] && grep -q "^start b.timer" "$FAKE_DIR/log"; then falha "$1: iniciou b.timer que estava parado"; fi
}
grupo_vivo() { ps -o stat= -s "$1" 2>/dev/null | grep -qv '^Z'; }
espera_pgid() { for _ in $(seq 1 100); do [ -s "$1/filho.pgid" ] && return 0; sleep 0.1; done; return 1; }

# 1. normal, ativo/ativo
prepara normal active active
roda --estado "$D/e-normal" -- bash -c 'exit 0'; rc=$?
[ $rc = 0 ] && passou "normal: saída 0" || falha "normal: saída $rc"
confere normal active active
[ "$(cat "$D/e-normal/ativos.txt" | tr '\n' ' ')" = "a.timer b.timer " ] && passou "normal: ativos.txt registra os dois" || falha "normal: ativos.txt"
grep -q "RESTAURAÇÃO OK" "$D/e-normal/restauracao.txt" && [ ! -e "$PEND" ] && passou "normal: restauração conferida e ponteiro apagado" || falha "normal: restauração ou ponteiro"
grep -q "^stop a.timer" "$FAKE_DIR/log" && grep -q "^stop b.timer" "$FAKE_DIR/log" && passou "normal: parou os dois .timer" || falha "normal: não parou os dois"

# 2. parados
prepara parados inactive inactive
roda --estado "$D/e-parados" -- bash -c 'exit 0'; rc=$?
[ $rc = 0 ] && passou "parados: saída 0" || falha "parados: saída $rc"
confere parados inactive inactive
grep -qE "^(stop|start) " "$FAKE_DIR/log" && falha "parados: houve stop/start" || passou "parados: nenhum stop/start"

# 3. misto
prepara misto active inactive
roda --estado "$D/e-misto" -- bash -c 'exit 0'; rc=$?
confere misto active inactive
grep -q "^stop b.timer" "$FAKE_DIR/log" && falha "misto: parou b.timer que estava parado"

# 4. comando que falha
prepara falha active active
roda --estado "$D/e-falha" -- bash -c 'exit 7'; rc=$?
[ $rc = 7 ] && passou "falha: devolve o código 7 do comando" || falha "falha: saída $rc"
confere falha active active

# 5. serviço de timer ocupado: não mata, não roda o comando, restaura
prepara ocupado active active active
rm -f "$D/marcador-ocupado"
roda --estado "$D/e-ocupado" --espera 2 -- bash -c "touch '$D/marcador-ocupado'"; rc=$?
[ $rc = 3 ] && passou "ocupado: saída 3" || falha "ocupado: saída $rc"
[ ! -e "$D/marcador-ocupado" ] && passou "ocupado: o comando não rodou" || falha "ocupado: o comando rodou"
[ "$(cat "$FAKE_DIR/a.service")" = active ] && ! grep -qE "^(stop|kill) a.service" "$FAKE_DIR/log" && passou "ocupado: o serviço em curso não foi tocado" || falha "ocupado: mexeu no serviço"
confere ocupado active active

# 6. limite de tempo, comando que atende ao TERM
prepara limite active active
t0=$(date +%s)
roda --estado "$D/e-limite" --limite 2 --kill-apos 2 -- bash -c 'sleep 60'; rc=$?
t1=$(date +%s)
[ $rc = 124 ] && passou "limite: saída 124" || falha "limite: saída $rc"
[ $((t1-t0)) -le 15 ] && passou "limite: terminou em $((t1-t0)) s" || falha "limite: demorou $((t1-t0)) s"
grupo_vivo "$(cat "$D/e-limite/filho.pgid")" && falha "limite: o grupo do comando sobreviveu" || passou "limite: o grupo do comando morreu"
confere limite active active

# 7. limite com comando que ignora TERM: só o KILL derruba
prepara kill active active
roda --estado "$D/e-kill" --limite 2 --kill-apos 2 -- bash -c 'trap "" TERM; sleep 60'; rc=$?
[ $rc = 124 ] && passou "kill: saída 124" || falha "kill: saída $rc"
grupo_vivo "$(cat "$D/e-kill/filho.pgid")" && falha "kill: o grupo que ignora TERM sobreviveu" || passou "kill: o KILL derrubou o grupo que ignora TERM"
confere kill active active

# 8. TERM no próprio roteiro no meio do comando
prepara term active active
roda_exec --estado "$D/e-term" -- bash -c 'sleep 60' & pid=$!
espera_pgid "$D/e-term" || falha "term: o comando não iniciou"
[ "$(cat "$FAKE_DIR/a.timer")" = inactive ] && passou "term: o timer estava parado durante o comando" || falha "term: o timer não foi parado"
kill -TERM $pid; wait $pid; rc=$?
[ $rc = 143 ] && passou "term: saída 143" || falha "term: saída $rc"
grupo_vivo "$(cat "$D/e-term/filho.pgid")" && falha "term: o comando sobreviveu ao TERM" || passou "term: o comando morreu junto"
confere term active active

# 8b. TERM em cinco instantes logo depois do stop (inclusive antes de o PGID ser lido): nenhum comando pode ficar órfão
for atraso in 0 0.05 0.1 0.2 0.35; do
  prepara term8b active active
  rm -f "$D/marcador-8b"
  roda_exec --estado "$D/e-term8b-$atraso" -- bash -c "echo \$\$ > '$D/marcador-8b'; sleep 60" & pid=$!
  for _ in $(seq 1 100); do grep -q "^stop a.timer" "$FAKE_DIR/log" && break; sleep 0.02; done
  sleep "$atraso"; kill -TERM $pid; wait $pid; rc=$?
  sleep 0.3
  if [ -e "$D/marcador-8b" ]; then
    mp=$(cat "$D/marcador-8b")
    if grupo_vivo "$mp"; then falha "term8b (atraso $atraso): o comando (PID $mp) ficou órfão"; kill -KILL -- "-$mp" 2>/dev/null; else passou "term8b (atraso $atraso): comando morto"; fi
  else passou "term8b (atraso $atraso): o comando nem chegou a rodar"; fi
  confere "term8b-$atraso" active active
done

# 9. SIGKILL no roteiro: nada restaura; o ponteiro e ativos.txt ficam; nova janela é recusada; a recuperação manual restaura
prepara sigkill active active
roda_exec --estado "$D/e-sigkill" -- bash -c 'sleep 60' & pid=$!
espera_pgid "$D/e-sigkill" || falha "sigkill: o comando não iniciou"
kill -KILL $pid; wait $pid 2>/dev/null
[ "$(cat "$FAKE_DIR/a.timer")" = inactive ] && [ -s "$PEND" ] && passou "sigkill: o timer ficou parado e o ponteiro ficou (como esperado: o trap não cobre SIGKILL)" || falha "sigkill: estado depois do SIGKILL"
roda --estado "$D/e-sigkill2" -- bash -c 'exit 0' > "$D/saida-sigkill2.txt" 2>&1; rc=$?
[ $rc = 2 ] && grep -q "não recuperada" "$D/saida-sigkill2.txt" && passou "sigkill: nova janela recusada enquanto houver janela pendente" || falha "sigkill: nova janela não foi recusada (rc=$rc)"
saida=$(SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" 2>&1); rc=$?
[ $rc = 0 ] && passou "sigkill: recuperação manual saiu 0" || falha "sigkill: recuperação saiu $rc"
echo "$saida" | grep -q "ATENÇÃO" && passou "sigkill: a recuperação avisou do processo órfão sem matá-lo" || falha "sigkill: não avisou do órfão"
confere sigkill active active
[ ! -e "$PEND" ] && passou "sigkill: ponteiro apagado depois da recuperação" || falha "sigkill: ponteiro ficou"
pg=$(cat "$D/e-sigkill/filho.pgid"); kill -TERM -- "-$pg" 2>/dev/null; sleep 0.3

# 10. restauração que falha: saída 4, ponteiro fica, recuperação depois funciona
prepara rfalha active active
FAKE_START_FALHA=a.timer roda --estado "$D/e-rfalha" -- bash -c 'exit 0' > "$D/saida-rfalha.txt" 2>&1; rc=$?
[ $rc = 4 ] && passou "rfalha: saída 4" || falha "rfalha: saída $rc"
[ -s "$PEND" ] && grep -q "INCOMPLETA" "$D/e-rfalha/restauracao.txt" && passou "rfalha: ponteiro mantido e restauração marcada como incompleta" || falha "rfalha: ponteiro ou marcação"
saida=$(SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" 2>&1); rc=$?
[ $rc = 0 ] && passou "rfalha: a recuperação manual restaura" || falha "rfalha: recuperação saiu $rc"
confere rfalha active active

# 11. recusas antes de mexer: pasta não vazia, sem lista, sem comando
prepara recusa1 active active
mkdir -p "$D/e-cheia"; echo x > "$D/e-cheia/arquivo"
roda --estado "$D/e-cheia" -- bash -c 'exit 0' > /dev/null 2>&1; rc=$?
[ $rc = 2 ] && ! grep -qE "^(stop|start) " "$FAKE_DIR/log" && passou "recusa: pasta de estado não vazia, nada tocado" || falha "recusa: pasta não vazia (rc=$rc)"
SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$J" --timers "$D/nao-existe.txt" --estado "$D/e-semlista" -- bash -c 'exit 0' > /dev/null 2>&1; rc=$?
[ $rc = 2 ] && ! grep -qE "^(stop|start) " "$FAKE_DIR/log" && [ ! -e "$D/e-semlista" ] && passou "recusa: sem lista de timers, nada tocado" || falha "recusa: sem lista (rc=$rc)"
roda --estado "$D/e-semcmd" > /dev/null 2>&1; rc=$?
[ $rc = 2 ] && ! grep -qE "^(stop|start) " "$FAKE_DIR/log" && passou "recusa: sem comando, nada tocado" || falha "recusa: sem comando (rc=$rc)"
roda --estado "$D/e-limite0" --limite 0 -- bash -c 'exit 0' > /dev/null 2>&1; rc=$?
[ $rc = 2 ] && passou "recusa: --limite 0" || falha "recusa: --limite 0 (rc=$rc)"

# 12. estado desconhecido (sem barramento de usuário): não assume "nada ativo"
prepara semBus active active
FAKE_SEM_BARRAMENTO=1 roda --estado "$D/e-sembus" -- bash -c 'exit 0' > "$D/saida-sembus.txt" 2>&1; rc=$?
[ $rc = 2 ] && grep -q "não reconhecido" "$D/saida-sembus.txt" && ! grep -qE "^(stop|start) " "$FAKE_DIR/log" && passou "sem barramento: recusa antes de parar qualquer coisa" || falha "sem barramento (rc=$rc)"

# 13. unidade inexistente
prepara semunid active active
rm -f "$FAKE_DIR/b.timer"
roda --estado "$D/e-semunid" -- bash -c 'exit 0' > "$D/saida-semunid.txt" 2>&1; rc=$?
[ $rc = 2 ] && grep -q "não conhece" "$D/saida-semunid.txt" && ! grep -qE "^(stop|start) " "$FAKE_DIR/log" && passou "unidade inexistente: recusa antes de parar qualquer coisa" || falha "unidade inexistente (rc=$rc)"

# 14. nunca disable/mask; XDG_RUNTIME_DIR chega ao systemctl
cat "$D"/*/log | grep -qE "^(disable|mask|enable|kill|restart|reload) " && falha "o roteiro usou disable/mask/enable/kill/restart/reload" || passou "só is-active, is-enabled, cat, stop e start foram usados"
grep -q "XDG=/run/user/fake" "$D/normal/log" && passou "XDG_RUNTIME_DIR chegou ao systemctl" || falha "XDG_RUNTIME_DIR não chegou"
XDG_RUNTIME_DIR_TESTE= prepara xdg active active
(unset XDG_RUNTIME_DIR; SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" JANELA_PASSO_ESPERA=0.1 "$J" --timers "$D/timers.txt" --estado "$D/e-xdg" -- bash -c 'exit 0') > /dev/null 2>&1
grep -q "XDG=/run/user/$(id -u)" "$FAKE_DIR/log" && passou "sem XDG_RUNTIME_DIR no ambiente: o roteiro usa /run/user/<uid>" || falha "XDG_RUNTIME_DIR padrão"

# 15. mutantes que o teste TEM de pegar
mutante() { # nome padrão_sed
  sed "$2" "$J" > "$D/mut-$1.sh"; chmod +x "$D/mut-$1.sh"
  if diff -q "$J" "$D/mut-$1.sh" > /dev/null; then falha "mutante $1: o sed não alterou o roteiro"; return 1; fi
}
# (a) sem a restauração armada: o caso normal tem de deixar o timer parado
if mutante sem-trap 's/^trap finaliza EXIT$/:/'; then
  prepara mut1 active active
  JANELA="$D/mut-sem-trap.sh" roda --estado "$D/e-mut1" -- bash -c 'exit 0' > /dev/null 2>&1
  [ "$(cat "$FAKE_DIR/a.timer")" = inactive ] && passou "mutante sem trap: pego (o timer ficou parado)" || falha "mutante sem trap NÃO foi pego"
fi
# (b) sem KILL: o comando que ignora TERM tem de sobreviver
if mutante sem-kill 's/kill -KILL -- "-\$PGID" 2>\/dev\/null/:/'; then
  prepara mut2 active active
  JANELA="$D/mut-sem-kill.sh" roda --estado "$D/e-mut2" --limite 2 --kill-apos 1 -- bash -c 'trap "" TERM; sleep 25' > /dev/null 2>&1
  if grupo_vivo "$(cat "$D/e-mut2/filho.pgid")"; then passou "mutante sem KILL: pego (o grupo que ignora TERM sobreviveu)"; else falha "mutante sem KILL NÃO foi pego"; fi
  pg=$(cat "$D/e-mut2/filho.pgid"); kill -KILL -- "-$pg" 2>/dev/null
fi
# (c) restaurando também os timers que estavam parados: o caso misto tem de acusar
sed 's/^\(.*\)< "\$ESTADO\/ativos.txt"$/\1< <(printf "%s\\n" "${TIMERS[@]}")/' "$J" > "$D/mut-ativa-tudo.sh"; chmod +x "$D/mut-ativa-tudo.sh"
if diff -q "$J" "$D/mut-ativa-tudo.sh" > /dev/null; then falha "mutante ativa-tudo: o sed não alterou o roteiro"; else
  prepara mut3 active inactive
  JANELA="$D/mut-ativa-tudo.sh" roda --estado "$D/e-mut3" -- bash -c 'exit 0' > /dev/null 2>&1
  [ "$(cat "$FAKE_DIR/b.timer")" = active ] && passou "mutante que religa tudo: pego (b.timer parado foi ativado)" || falha "mutante que religa tudo NÃO foi pego"
fi
# (d) sem a espera pelos serviços: o caso ocupado tem de deixar o comando rodar
if mutante sem-espera 's/^  \[ \$ocupado = 0 \] && break$/  break/'; then
  prepara mut4 active active active
  rm -f "$D/marcador-mut4"
  JANELA="$D/mut-sem-espera.sh" roda --estado "$D/e-mut4" --espera 2 -- bash -c "touch '$D/marcador-mut4'" > /dev/null 2>&1
  [ -e "$D/marcador-mut4" ] && passou "mutante sem espera: pego (o comando rodou com serviço em curso)" || falha "mutante sem espera NÃO foi pego"
fi

[ $ok = 1 ] && { echo "TODOS OS CASOS PASSAM"; exit 0; } || { echo "HÁ FALHA"; exit 1; }
