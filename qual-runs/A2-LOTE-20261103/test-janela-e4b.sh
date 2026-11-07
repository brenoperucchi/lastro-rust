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
             if [ -n "${FAKE_FALHA:-}" ] && [ "${FAKE_FALHA%%:*}" = "$u" ]; then # FAKE_FALHA=unidade:N -> a consulta N+1 em diante falha, sem saída
               c=$(( $(cat "$FAKE_DIR/cnt-$u" 2>/dev/null || echo 0) + 1 )); echo $c > "$FAKE_DIR/cnt-$u"; [ "$c" -gt "${FAKE_FALHA##*:}" ] && exit 1
             fi
             [ -e "$FAKE_DIR/$u" ] && cat "$FAKE_DIR/$u" || echo inactive; [ "$(cat "$FAKE_DIR/$u" 2>/dev/null)" = active ];;
  is-enabled) [ -n "${FAKE_LENTO:-}" ] && sleep "$FAKE_LENTO"; echo enabled;;
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
proc_vivo() { [ -r "/proc/$1/status" ] && ! grep -q '^State:[[:space:]]*Z' "/proc/$1/status"; } # o PROCESSO $1 existe e não é zumbi
espera_arq() { for _ in $(seq 1 150); do [ -e "$1" ] && return 0; sleep 0.1; done; return 1; }
espera_pgid() { for _ in $(seq 1 100); do [ -s "$1/filho.pgid" ] && return 0; sleep 0.1; done; return 1; }

# 1. normal, ativo/ativo
prepara normal active active
roda --estado "$D/e-normal" -- bash -c 'exit 0'; rc=$?
[ $rc = 0 ] && passou "normal: saída 0" || falha "normal: saída $rc"
confere normal active active
[ "$(cat "$D/e-normal/ativos.txt" | tr '\n' ' ')" = "a.timer b.timer " ] && passou "normal: ativos.txt registra os dois" || falha "normal: ativos.txt"
grep -q "RESTAURAÇÃO OK" "$D/e-normal/restauracao.txt" && [ ! -e "$PEND" ] && passou "normal: restauração conferida e ponteiro apagado" || falha "normal: restauração ou ponteiro"
grep -q "^stop a.timer" "$FAKE_DIR/log" && grep -q "^stop b.timer" "$FAKE_DIR/log" && passou "normal: parou os dois .timer" || falha "normal: não parou os dois"
proc_vivo "$(cat "$D/e-normal/ancora.pid")" && falha "normal: a âncora da sessão sobrou" || passou "normal: a âncora da sessão foi encerrada"
grep -q "^ganchos_de_teste: atraso_lancamento= reserva= vigia=$" "$D/e-normal/comando.txt" && passou "normal: comando.txt registra que o gancho de teste estava desligado" || falha "normal: comando.txt sem registro do gancho"

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
espera_arq "$D/e-term/vai" || falha "term: o comando não foi liberado"
saida=$(SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" 2>&1); rc=$?
[ $rc = 2 ] && grep -q "ainda está viva" <<<"$saida" && ! grep -q "^start " "$FAKE_DIR/log" && [ "$(cat "$FAKE_DIR/a.timer")" = inactive ] && passou "term: a recuperação recusa com a janela viva e não religa nada" || falha "term: a recuperação não recusou com a janela viva (rc=$rc)"
kill -TERM $pid; wait $pid; rc=$?
[ $rc = 143 ] && passou "term: saída 143" || falha "term: saída $rc"
grupo_vivo "$(cat "$D/e-term/filho.pgid")" && falha "term: o comando sobreviveu ao TERM" || passou "term: o comando morreu junto"
confere term active active

# 8b. TERM em cinco instantes logo depois do stop: nenhum comando pode ficar órfão (observa o PROCESSO marcador, não a sessão)
for atraso in 0 0.05 0.1 0.2 0.35; do
  prepara term8b active active
  rm -f "$D/marcador-8b"
  roda_exec --estado "$D/e-term8b-$atraso" -- bash -c "echo \$\$ > '$D/marcador-8b'; exec sleep 60" & pid=$!
  for _ in $(seq 1 100); do grep -q "^stop a.timer" "$FAKE_DIR/log" && break; sleep 0.02; done
  sleep "$atraso"; kill -TERM $pid; wait $pid; rc=$?
  sleep 0.3
  if [ -e "$D/marcador-8b" ]; then
    mp=$(cat "$D/marcador-8b")
    if proc_vivo "$mp"; then falha "term8b (atraso $atraso): o comando (PID $mp) ficou órfão"; kill -KILL "$mp" 2>/dev/null; else passou "term8b (atraso $atraso): comando morto"; fi
  else passou "term8b (atraso $atraso): o comando nem chegou a rodar"; fi
  confere "term8b-$atraso" active active
done

mutante() { # nome padrão_sed
  sed "$2" "$J" > "$D/mut-$1.sh"; chmod +x "$D/mut-$1.sh"
  if diff -q "$J" "$D/mut-$1.sh" > /dev/null; then falha "mutante $1: o sed não alterou o roteiro"; return 1; fi
}
# 8c. Sinal na fronteira do lançamento (gancho alarga a janela entre o setsid e o registro do filho): o comando NÃO pode nem começar
largada() { # nome roteiro
  prepara "$1" active active; rm -f "$D/marcador-largada"
  JANELA="$2" JANELA_ATRASO_LANCAMENTO=1 roda_exec_j "$2" --estado "$D/e-$1" -- bash -c "touch '$D/marcador-largada'; exec sleep 60" & pid=$!
  espera_pgid "$D/e-$1" || falha "$1: o filho não iniciou"
  sleep 0.2; kill -TERM $pid; wait $pid; rc=$?; sleep 0.4
}
roda_exec_j() { j=$1; shift; SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" JANELA_PASSO_ESPERA=0.1 XDG_RUNTIME_DIR=/run/user/fake exec "$j" --timers "$D/timers.txt" "$@"; }
largada largada "$J"
[ $rc = 143 ] && passou "largada: saída 143" || falha "largada: saída $rc"
[ ! -e "$D/marcador-largada" ] && passou "largada: o comando não chegou a rodar" || falha "largada: o comando rodou"
grupo_vivo "$(cat "$D/e-largada/filho.pgid")" && falha "largada: o grupo sobreviveu" || passou "largada: o grupo do filho morreu"
confere largada active active
if mutante sem-gate '/# GATE$/d'; then
  largada largada-mut "$D/mut-sem-gate.sh"
  if [ -e "$D/marcador-largada" ]; then passou "mutante sem aval de largada: pego (o comando rodou antes do supervisor)"; else falha "mutante sem aval de largada NÃO foi pego"; fi
  pg=$(cat "$D/e-largada-mut/filho.pgid" 2>/dev/null); [ -n "$pg" ] && kill -KILL -- "-$pg" 2>/dev/null
fi

# 8d. Flags do vigia: saída 5 (não morreu) e limite registrado; restauração incompleta prevalece (4); só limite = 124
sinaliza_flags() { # nome flags... ; env extra por fora
  n=$1; shift
  prepara "$n" active active
  roda_exec --estado "$D/e-$n" -- bash -c 'exec sleep 60' & pid=$!
  espera_arq "$D/e-$n/vai" || falha "$n: o comando não foi liberado"
  for f in "$@"; do touch "$D/e-$n/$f"; done
  kill -TERM $pid; wait $pid; rc=$?
}
sinaliza_flags flags nao-morreu.flag limite.flag
[ $rc = 5 ] && passou "flags: saída 5 (o comando não morreu)" || falha "flags: saída $rc"
grep -q "limite atingido: 1" "$D/e-flags/resultado.txt" && grep -q "processo não morreu: 1" "$D/e-flags/resultado.txt" && passou "flags: resultado.txt registra limite e processo persistente" || falha "flags: resultado.txt"
confere flags active active
FAKE_START_FALHA=a.timer sinaliza_flags flags4 nao-morreu.flag limite.flag
[ $rc = 4 ] && passou "flags+falha: restauração incompleta prevalece (4)" || falha "flags+falha: saída $rc"
SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" > /dev/null 2>&1
sinaliza_flags flags124 limite.flag
[ $rc = 124 ] && passou "flags: só o limite dá 124" || falha "flags124: saída $rc"

# 8e. Dois TERM seguidos: o segundo não pode interromper a restauração
prepara dois active active
roda_exec --estado "$D/e-dois" --kill-apos 2 -- bash -c 'trap "" TERM; sleep 60' & pid=$!
espera_arq "$D/e-dois/vai" || falha "dois: o comando não foi liberado"
dois_sinais() { kill -TERM $pid; sleep 0.5; kill -TERM $pid 2>/dev/null; wait $pid; rc=$?; sleep 0.3; }
dois_sinais
grep -q "RESTAURAÇÃO OK" "$D/e-dois/restauracao.txt" && [ -s "$D/e-dois/resultado.txt" ] && passou "dois TERM: restauração completa e resultado gravado" || falha "dois TERM: restauração interrompida"
confere dois active active
if mutante sem-protecao "/trap '' INT TERM HUP/d"; then
  prepara dois2 active active
  JANELA="$D/mut-sem-protecao.sh" roda_exec_j "$D/mut-sem-protecao.sh" --estado "$D/e-dois2" --kill-apos 2 -- bash -c 'trap "" TERM; sleep 60' & pid=$!
  espera_arq "$D/e-dois2/vai" || falha "dois2: o comando não foi liberado"
  dois_sinais
  if [ "$(cat "$FAKE_DIR/a.timer")" = inactive ]; then passou "mutante sem proteção contra segundo sinal: pego (restauração cortada)"; else falha "mutante sem proteção NÃO foi pego"; fi
  pg=$(cat "$D/e-dois2/filho.pgid" 2>/dev/null); [ -n "$pg" ] && kill -KILL -- "-$pg" 2>/dev/null
fi

# 8f. Identidade divergente do grupo (PID reaproveitado): nenhum sinal é enviado; saída 5
prepara ident active active
rm -f "$D/marcador-ident"
roda --estado "$D/e-ident" --limite 2 --kill-apos 1 -- bash -c "echo \$\$ > '$D/marcador-ident'; exec sleep 60" & pid=$!
espera_arq "$D/e-ident/ancora.id" || falha "ident: sem ancora.id"
echo "$(cut -d' ' -f1 "$D/e-ident/ancora.id") 1" > "$D/e-ident/ancora.id"
wait $pid; rc=$?
[ $rc = 5 ] && passou "ident: saída 5 (grupo não reconhecido)" || falha "ident: saída $rc"
pgi=$(cat "$D/e-ident/filho.pgid")
proc_vivo "$(cat "$D/marcador-ident")" && passou "ident: nenhum sinal foi enviado ao comando de identidade divergente" || falha "ident: o comando foi sinalizado"
confere ident active active
[ -s "$PEND" ] && passou "ident: ponteiro mantido com o comando vivo" || falha "ident: ponteiro apagado com o comando vivo"
roda --estado "$D/e-ident-b" -- bash -c 'exit 0' > /dev/null 2>&1; rc=$?
[ $rc = 2 ] && passou "ident: nova janela recusada com o comando vivo" || falha "ident: nova janela liberada (rc=$rc)"
: > "$FAKE_DIR/log"   # a restauração do roteiro principal já religou os timers; daqui em diante nenhum start é esperado
saida=$(SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" 2>&1); rc=$?
[ $rc = 3 ] && ! grep -q "^start " "$FAKE_DIR/log" && passou "ident: recuperação adiada (3) sem religar nada com o comando vivo" || falha "ident: recuperação rc=$rc"
kill -KILL -- "-$pgi" 2>/dev/null; sleep 0.3
saida=$(SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" 2>&1); rc=$?
[ $rc = 0 ] && [ ! -e "$PEND" ] && passou "ident: depois do fim do comando a recuperação conclui e apaga o ponteiro" || falha "ident: recuperação final rc=$rc"

# mutante: sem a verificação da âncora o grupo de identidade divergente seria sinalizado
if mutante sem-ancora 's/^  ancora_ok || return 2$/  :/'; then
  prepara identm active active
  rm -f "$D/marcador-identm"
  JANELA="$D/mut-sem-ancora.sh" roda --estado "$D/e-identm" --limite 2 --kill-apos 1 -- bash -c "echo \$\$ > '$D/marcador-identm'; exec sleep 60" & pid=$!
  espera_arq "$D/e-identm/ancora.id" || falha "identm: sem ancora.id"
  echo "$(cut -d' ' -f1 "$D/e-identm/ancora.id") 1" > "$D/e-identm/ancora.id"
  wait $pid; rc=$?
  if ! proc_vivo "$(cat "$D/marcador-identm")"; then passou "mutante sem âncora: pego (o comando foi sinalizado sem identidade verificada)"; else falha "mutante sem âncora NÃO foi pego"; fi
  kill -KILL -- "-$(cat "$D/e-identm/filho.pgid")" 2>/dev/null
  SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" > /dev/null 2>&1
fi

# 8i. Trava comum: recuperação não remove a reserva de uma janela ainda em preparação; janela não começa com a trava ocupada
prepara reserva active active
JANELA_GANCHO_RESERVA="$D/lib-reserva" roda_exec --estado "$D/e-reserva" -- bash -c 'exit 0' & pid=$!
espera_arq "$PEND" || falha "reserva: o ponteiro não apareceu"
saida=$(SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" 2>&1); rc=$?
[ $rc = 2 ] && [ -s "$PEND" ] && ! grep -qE "^(start|stop) " "$FAKE_DIR/log" && passou "reserva: com a janela em preparação a recuperação recusa e a reserva fica" || falha "reserva: rc=$rc ponteiro=$(cat "$PEND" 2>/dev/null)"
touch "$D/lib-reserva"; wait $pid; rc=$?
[ $rc = 0 ] && [ ! -e "$PEND" ] && passou "reserva: a janela retomada executou uma única vez e terminou" || falha "reserva: rc=$rc"
confere reserva active active
prepara trava active active
( exec 9> "$PEND.rec.lock"; flock 9; sleep 2 ) & sleep 0.4
roda --estado "$D/e-trava" -- bash -c 'exit 0' > /dev/null 2>&1; rc=$?
[ $rc = 2 ] && ! grep -qE "^(start|stop) " "$FAKE_DIR/log" && passou "trava: com a trava comum ocupada (recuperação em curso) a janela recusa sem mexer em nada" || falha "trava: rc=$rc"
wait

# 8j. Vigia desligado antes de a âncora sair: o vigia fica retido depois de validar o grupo, o comando termina, e a âncora continua viva
retido() { # nome etapa comando...
  n=$1; et=$2; shift 2
  prepara "$n" active active; rm -f "$D/gv-$n".*
  [ $et = kill ] && touch "$D/gv-$n.term"
  JANELA_GANCHO_VIGIA="$D/gv-$n" roda_exec --estado "$D/e-$n" --limite 1 --kill-apos 1 -- "$@" & pid=$!
  espera_arq "$D/e-$n/vigia-cancela.flag" || falha "$n: o supervisor não chegou ao fim do comando"
  sleep 0.6
  if proc_vivo "$(cat "$D/e-$n/ancora.pid")" && kill -0 $pid 2>/dev/null; then passou "$n: com o vigia retido a âncora e o supervisor continuam vivos"; else falha "$n: a âncora saiu antes de o vigia confirmar"; fi
  touch "$D/gv-$n.$et"; wait $pid; rc=$?
}
retido vigiaB term bash -c 'sleep 3'
[ $rc = 0 ] && [ ! -s "$D/e-vigiaB/sinais.log" ] && passou "vigiaB: depois de liberado o vigia não enviou sinal (o comando já tinha terminado) e a saída foi 0" || falha "vigiaB: rc=$rc sinais=$(cat "$D/e-vigiaB/sinais.log" 2>/dev/null)"
proc_vivo "$(cat "$D/e-vigiaB/ancora.pid")" && falha "vigiaB: a âncora sobrou" || passou "vigiaB: a âncora foi retirada depois do vigia"
confere vigiaB active active
retido vigiaK kill bash -c 'trap "" TERM; sleep 3'
grep -q '^TERM ' "$D/e-vigiaK/sinais.log" && ! grep -q '^KILL ' "$D/e-vigiaK/sinais.log" && passou "vigiaK: o KILL não foi enviado depois do fim do comando (só o TERM, legítimo)" || falha "vigiaK: sinais=$(cat "$D/e-vigiaK/sinais.log" 2>/dev/null)"
proc_vivo "$(cat "$D/e-vigiaK/ancora.pid")" && falha "vigiaK: a âncora sobrou" || passou "vigiaK: a âncora foi retirada depois do vigia"
confere vigiaK active active

# 8k. Falha ao gravar a identidade da âncora: o comando não roda e a âncora não fica órfã
prepara semid active active; rm -f "$D/marcador-semid"
JANELA_TESTE_FALHA_ID_ANCORA=1 roda --estado "$D/e-semid" -- bash -c "touch '$D/marcador-semid'" > /dev/null 2>&1; rc=$?
sleep 0.3
[ $rc = 125 ] && [ ! -e "$D/marcador-semid" ] && passou "semid: sem identidade da âncora o comando não rodou (125)" || falha "semid: rc=$rc"
proc_vivo "$(cat "$D/e-semid/ancora.pid")" && falha "semid: a âncora ficou órfã" || passou "semid: a âncora não ficou órfã"
confere semid active active

# 8f2. Identidade ausente ou vazia: falha fechada, nenhum sinal
for variante in ausente vazia; do
  prepara ident-$variante active active
  rm -f "$D/marcador-id-$variante"
  roda --estado "$D/e-id-$variante" --limite 2 --kill-apos 1 -- bash -c "echo \$\$ > '$D/marcador-id-$variante'; exec sleep 60" & pid=$!
  espera_arq "$D/e-id-$variante/ancora.id" || falha "id-$variante: sem ancora.id"
  if [ $variante = ausente ]; then rm -f "$D/e-id-$variante/ancora.id"; else : > "$D/e-id-$variante/ancora.id"; fi
  wait $pid; rc=$?
  pgi=$(cat "$D/e-id-$variante/filho.pgid")
  if [ $rc = 5 ] && proc_vivo "$(cat "$D/marcador-id-$variante")"; then passou "id-$variante: sem identidade verificável nada foi sinalizado (saída 5)"; else falha "id-$variante: rc=$rc ou o grupo foi sinalizado"; fi
  kill -KILL -- "-$pgi" 2>/dev/null; sleep 0.2
  saida=$(SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" 2>&1)
done

# 8f3. Líder termina antes dos descendentes: o líder zumbi ancora o número e o limite ainda derruba o grupo
prepara lider active active
roda --estado "$D/e-lider" --limite 2 --kill-apos 1 -- bash -c 'sleep 60 & exit 0'; rc=$?
[ $rc = 124 ] && passou "líder: saída 124 (o limite derrubou os descendentes)" || falha "líder: saída $rc"
grupo_vivo "$(cat "$D/e-lider/filho.pgid")" && falha "líder: descendente sobreviveu" || passou "líder: grupo inteiro morreu"
confere lider active active

# 8f4. Sinal durante a captura (antes de ativos.txt): a reserva é solta e nada foi parado; e a recuperação cobre o SIGKILL nesse trecho
prepara captura active active
FAKE_LENTO=0.4 roda_exec --estado "$D/e-captura" -- bash -c 'exit 0' & pid=$!
espera_arq "$D/e-captura/supervisor.id" || falha "captura: sem supervisor.id"
kill -TERM $pid; wait $pid; rc=$?
[ $rc = 143 ] && [ ! -e "$PEND" ] && ! grep -qE "^(stop|start) " "$FAKE_DIR/log" && passou "captura: TERM soltou a reserva sem parar nada" || falha "captura: rc=$rc ou reserva/stop"
prepara captura2 active active
mkdir -p "$D/e-captura2"; echo "$D/e-captura2" > "$PEND"
saida=$(SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" 2>&1); rc=$?
[ $rc = 0 ] && [ ! -e "$PEND" ] && ! grep -q "^start " "$FAKE_DIR/log" && passou "captura: sem ativos.txt a recuperação solta o ponteiro sem religar nada" || falha "captura2: rc=$rc"

# 8f5. Recuperar uma pasta ANTIGA durante outra janela não pode religar nada
prepara antiga active active
mkdir -p "$D/e-antiga"; printf 'a.timer\n' > "$D/e-antiga/ativos.txt"; printf 'a.timer is-active=active\n' > "$D/e-antiga/estado.txt"
roda_exec --estado "$D/e-atual" -- bash -c 'sleep 60' & pid=$!
espera_arq "$D/e-atual/vai" || falha "antiga: janela atual sem aval"
saida=$(SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" "$D/e-antiga" 2>&1); rc=$?
[ $rc = 2 ] && ! grep -q "^start " "$FAKE_DIR/log" && [ "$(cat "$FAKE_DIR/a.timer")" = inactive ] && passou "antiga: recuperação de pasta antiga recusada, janela atual intacta" || falha "antiga: rc=$rc"
kill -TERM $pid; wait $pid 2>/dev/null
confere antiga active active

# 8f6. Duas recuperações simultâneas: exatamente uma conclui
prepara duasrec active active
roda_exec --estado "$D/e-duasrec" -- bash -c 'sleep 60' & pid=$!
espera_arq "$D/e-duasrec/vai" || falha "duasrec: sem aval"
kill -KILL $pid; wait $pid 2>/dev/null
pg=$(cat "$D/e-duasrec/filho.pgid"); kill -KILL -- "-$pg" 2>/dev/null; sleep 0.3
( SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" > /dev/null 2>&1; echo $? > "$D/rc-rec1" ) &
( SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" > /dev/null 2>&1; echo $? > "$D/rc-rec2" ) &
wait
r=$(sort "$D/rc-rec1" "$D/rc-rec2" | tr '\n' ' ')
[ "$r" = "0 2 " ] && passou "duas recuperações: uma concluiu e a outra recusou" || falha "duas recuperações: saídas '$r'"
confere duasrec active active

# 8f7. Timer que estava PARADO e aparece ativo na recuperação: divergência, ponteiro mantido, nada alterado
prepara diverge active inactive
roda_exec --estado "$D/e-diverge" -- bash -c 'sleep 60' & pid=$!
espera_arq "$D/e-diverge/vai" || falha "diverge: sem aval"
kill -KILL $pid; wait $pid 2>/dev/null
pg=$(cat "$D/e-diverge/filho.pgid"); kill -KILL -- "-$pg" 2>/dev/null; sleep 0.3
echo active > "$FAKE_DIR/b.timer"; : > "$FAKE_DIR/log"
saida=$(SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" 2>&1); rc=$?
[ $rc = 1 ] && [ -s "$PEND" ] && grep -q DIVERGÊNCIA <<<"$saida" && ! grep -qE "^(start|stop) b.timer" "$FAKE_DIR/log" && passou "diverge: recuperação incompleta (1), ponteiro mantido, b.timer não foi mexido" || falha "diverge: rc=$rc"
echo inactive > "$FAKE_DIR/b.timer"
FAKE_FALHA="b.timer:0" SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" > /dev/null 2>&1; rc=$?
[ $rc = 1 ] && [ -s "$PEND" ] && passou "diverge: consulta inconclusiva em timer parado também é incompleta (1)" || falha "diverge: inconclusivo rc=$rc"
saida=$(SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" 2>&1); rc=$?
[ $rc = 0 ] && [ ! -e "$PEND" ] && passou "diverge: resolvida a divergência, a recuperação conclui (0)" || falha "diverge: final rc=$rc"
confere diverge active inactive

# 8g. Depois de SIGKILL no supervisor e do fim do comando, não sobra vigia
prepara vigia active active
roda_exec --estado "$D/e-vigia" --limite 600 -- bash -c 'sleep 3' & pid=$!
espera_arq "$D/e-vigia/vai" || falha "vigia: o comando não foi liberado"
kill -KILL $pid; wait $pid 2>/dev/null; sleep 6
vp=$(cat "$D/e-vigia/vigia.pid" 2>/dev/null)
if [ -n "$vp" ] && ! proc_vivo "$vp"; then passou "vigia: nenhum vigia sobrou depois do fim do comando"; else falha "vigia: o vigia (PID ${vp:-?}) sobrou"; kill -KILL "$vp" 2>/dev/null; fi
SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" > /dev/null 2>&1
confere vigia active active

# 8h. Duas janelas ao mesmo tempo: exatamente uma executa
for rep in 1 2 3; do
  prepara conc$rep active active
  ( roda --estado "$D/e-conc$rep-1" -- bash -c 'sleep 1' > /dev/null 2>&1; echo $? > "$D/rc-conc$rep-1" ) &
  ( roda --estado "$D/e-conc$rep-2" -- bash -c 'sleep 1' > /dev/null 2>&1; echo $? > "$D/rc-conc$rep-2" ) &
  wait
  r=$(sort "$D/rc-conc$rep-1" "$D/rc-conc$rep-2" | tr '\n' ' ')
  [ "$r" = "0 2 " ] && passou "concorrência $rep: uma janela executou e a outra recusou" || falha "concorrência $rep: saídas '$r'"
  confere conc$rep active active
done

# 9. SIGKILL no roteiro: nada restaura; ponteiro e ativos.txt ficam; nova janela é recusada; a recuperação restaura os timers MAS mantém
#    o ponteiro enquanto o comando órfão viver (nova janela segue recusada); só depois do fim do comando o ponteiro sai
prepara sigkill active active
roda_exec --estado "$D/e-sigkill" -- bash -c 'sleep 60' & pid=$!
espera_arq "$D/e-sigkill/vai" || falha "sigkill: o comando não foi liberado"
sleep 0.3
kill -KILL $pid; wait $pid 2>/dev/null
[ "$(cat "$FAKE_DIR/a.timer")" = inactive ] && [ -s "$PEND" ] && passou "sigkill: o timer ficou parado e o ponteiro ficou (como esperado: o trap não cobre SIGKILL)" || falha "sigkill: estado depois do SIGKILL"
roda --estado "$D/e-sigkill2" -- bash -c 'exit 0' > "$D/saida-sigkill2.txt" 2>&1; rc=$?
[ $rc = 2 ] && grep -q "reservar a janela\|não recuperada" "$D/saida-sigkill2.txt" && passou "sigkill: nova janela recusada enquanto houver janela pendente" || falha "sigkill: nova janela não foi recusada (rc=$rc)"
saida=$(SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" 2>&1); rc=$?
[ $rc = 3 ] && ! grep -q "^start " "$FAKE_DIR/log" && [ "$(cat "$FAKE_DIR/a.timer")" = inactive ] && passou "sigkill: com o comando órfão vivo a recuperação adia (3) e NÃO religa os timers" || falha "sigkill: recuperação saiu $rc"
echo "$saida" | grep -q "ainda está vivo" && passou "sigkill: a recuperação avisou do processo órfão sem matá-lo" || falha "sigkill: não avisou do órfão"
[ -s "$PEND" ] && passou "sigkill: ponteiro mantido com o comando órfão vivo" || falha "sigkill: ponteiro foi apagado com o órfão vivo"
roda --estado "$D/e-sigkill3" -- bash -c 'exit 0' > /dev/null 2>&1; rc=$?
[ $rc = 2 ] && passou "sigkill: nova janela segue recusada com o órfão vivo" || falha "sigkill: nova janela liberada com o órfão vivo (rc=$rc)"
pg=$(cat "$D/e-sigkill/filho.pgid"); kill -TERM -- "-$pg" 2>/dev/null; sleep 0.5
saida=$(SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" 2>&1); rc=$?
[ $rc = 0 ] && [ ! -e "$PEND" ] && passou "sigkill: depois do fim do comando a recuperação sai 0 e apaga o ponteiro" || falha "sigkill: recuperação final saiu $rc"
confere sigkill active active
roda --estado "$D/e-sigkill4" -- bash -c 'exit 0' > /dev/null 2>&1; rc=$?
[ $rc = 0 ] && passou "sigkill: nova janela liberada depois da recuperação" || falha "sigkill: nova janela recusada depois da recuperação (rc=$rc)"

# 9b. Recuperação com identidade divergente (reboot ou PID reaproveitado): não acusa nem sugere matar processo alheio
prepara ident2 inactive inactive
mkdir -p "$D/e-ident2"; : > "$D/e-ident2/ativos.txt"; : > "$D/e-ident2/estado.txt"
sleep 30 & alheio=$!
echo "$alheio" > "$D/e-ident2/filho.pgid"; echo "00000000-0000-0000-0000-000000000000 1" > "$D/e-ident2/filho.id"
echo "$D/e-ident2" > "$PEND"
setsid sleep 30 & alheio_sessao=$!
sleep 0.2
echo "$alheio_sessao" > "$D/e-ident2/filho.pgid"
saida=$(SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" 2>&1); rc=$?
if [ $rc = 0 ] && ! grep -q "ATENÇÃO" <<<"$saida" && grep -q "outro boot" <<<"$saida"; then passou "ident2: recuperação não tratou o processo alheio como o comando"; else falha "ident2: rc=$rc saída: $saida"; fi
kill "$alheio" "$alheio_sessao" 2>/dev/null

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
# (a) sem a restauração armada: o caso normal tem de deixar o timer parado
if mutante sem-trap 's/^trap finaliza EXIT$/:/'; then
  prepara mut1 active active
  JANELA="$D/mut-sem-trap.sh" roda --estado "$D/e-mut1" -- bash -c 'exit 0' > /dev/null 2>&1
  [ "$(cat "$FAKE_DIR/a.timer")" = inactive ] && passou "mutante sem trap: pego (o timer ficou parado)" || falha "mutante sem trap NÃO foi pego"
fi
# (b) sem KILL: o comando que ignora TERM tem de sobreviver
if mutante sem-kill 's/^  envia KILL$/  :/'; then
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

# 16. Consulta ao systemd que falha DEPOIS do preflight (snapshot, stop, espera, conferência final): nunca vira estado seguro
for u in a.timer a.service b.timer; do
  for n in 0 1 2 3 4 5 6 7 8; do
    nome="pf-${u//./_}-$n"; prepara "$nome" active active; rm -f "$D/marcador-$nome"
    FAKE_FALHA="$u:$n" roda --estado "$D/e-$nome" --espera 2 -- bash -c "cat '$FAKE_DIR/a.timer' '$FAKE_DIR/b.timer' > '$D/marcador-$nome'" > /dev/null 2>&1; rc=$?
    fa=$(cat "$FAKE_DIR/a.timer"); fb=$(cat "$FAKE_DIR/b.timer")
    if [ -e "$PEND" ]; then [ $rc = 4 ] || falha "falha $u:$n: ponteiro mantido com saída $rc (esperava 4)"
    else { [ "$fa" = active ] && [ "$fb" = active ]; } || falha "falha $u:$n: estado final $fa/$fb sem ponteiro (rc=$rc)"; fi
    if [ -e "$D/marcador-$nome" ]; then
      [ "$(tr '\n' ' ' < "$D/marcador-$nome")" = "inactive inactive " ] || falha "falha $u:$n: o comando rodou sem a pausa confirmada ($(tr '\n' ' ' < "$D/marcador-$nome"))"
    fi
    if [ -e "$PEND" ]; then
      SYSTEMCTL="$D/fake-systemctl" PENDENTE="$PEND" "$R" > /dev/null 2>&1 || falha "falha $u:$n: a recuperação manual não fechou"
      confere "falha-$u-$n" active active
    fi
  done
done
passou "falhas seletivas de consulta: invariantes mantidos (27 casos)"
# serviço ocupado + consulta que falha na espera: o comando NÃO roda (exit 7)
prepara ocfalha active active active; rm -f "$D/marcador-ocfalha"
FAKE_FALHA="a.service:1" roda --estado "$D/e-ocfalha" --espera 2 -- bash -c "touch '$D/marcador-ocfalha'" > /dev/null 2>&1; rc=$?
[ $rc = 7 ] && [ ! -e "$D/marcador-ocfalha" ] && passou "consulta inconclusiva na espera: saída 7 e o comando não rodou" || falha "ocfalha: rc=$rc"
confere ocfalha active active
if mutante sem-validacao 's/estado_conhecido "\$v" || return 1;/:;/'; then
  prepara ocfalha2 active active active; rm -f "$D/marcador-ocfalha"
  FAKE_FALHA="a.service:1" JANELA="$D/mut-sem-validacao.sh" roda --estado "$D/e-ocfalha2" --espera 2 -- bash -c "touch '$D/marcador-ocfalha'" > /dev/null 2>&1
  [ -e "$D/marcador-ocfalha" ] && passou "mutante sem validação: pego (o comando rodou com serviço em curso)" || falha "mutante sem validação NÃO foi pego"
fi

# higiene: nenhuma âncora de teste pode sobrar
for f in "$D"/e-*/ancora.pid; do [ -s "$f" ] && proc_vivo "$(cat "$f")" && { echo "âncora sobrou: $f"; kill -KILL "$(cat "$f")" 2>/dev/null; sobrou=1; }; done
[ "${sobrou:-0}" = 0 ] && passou "nenhuma âncora sobrou depois de todos os casos" || falha "âncoras sobraram"
[ $ok = 1 ] && { echo "TODOS OS CASOS PASSAM"; exit 0; } || { echo "HÁ FALHA"; exit 1; }
