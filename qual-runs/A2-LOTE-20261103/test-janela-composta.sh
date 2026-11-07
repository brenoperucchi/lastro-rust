#!/usr/bin/env bash
# Teste da COMPOSIÇÃO janela-e4b.sh (pausa e restauração dos timers de backup) + roda-com-limite.sh (limite por unidade systemd) para as janelas da
# etapa 4: systemctl FALSO para os timers (nada do sistema real é pausado), comandos de mentira, SEM GPU, sem guardas de autorização.
# O limite de fora (janela-e4b) é maior que o de dentro (roda-com-limite), então o de dentro dispara primeiro.
# A pasta temporária (mktemp) NÃO é apagada pelo teste: use TMPDIR para apontá-la a um lugar descartável.
set -u
unset E4B_GPU_AUTORIZADO E4B_MEDIDA_AUTORIZADA MOE_RT_REQUIRE_GPU LEITURA_CORRECAO_AUTORIZADA LEITURA_VELOCIDADE_AUTORIZADA
AQUI=$(cd "$(dirname "$0")" && pwd -P)
J=$AQUI/janela-e4b.sh
L=$AQUI/roda-com-limite.sh
D=$(mktemp -d -p "${TMPDIR:-/tmp}" janela-composta-teste.XXXXXX) || exit 1
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
FALHAS=0
ok() { echo "ok   $1"; }
ruim() { echo "FALHA $1"; FALHAS=$((FALHAS + 1)); }
janela() { # nome limite_interno comando...
  local n=$1 li=$2; shift 2
  export FAKE_DIR=$D/$n; mkdir -p "$FAKE_DIR"
  echo active > "$FAKE_DIR/a.timer"; echo active > "$FAKE_DIR/b.timer"; echo inactive > "$FAKE_DIR/a.service"; echo inactive > "$FAKE_DIR/b.service"; : > "$FAKE_DIR/log"
  SYSTEMCTL="$D/fake-systemctl" PENDENTE="$D/pend-$n" JANELA_PASSO_ESPERA=0.1 XDG_RUNTIME_DIR=/run/user/fake \
    "$J" --timers "$D/timers.txt" --estado "$D/estado-$n" --limite 120 --kill-apos 5 -- "$L" --limite "$li" -- "$@" > "$D/saida-$n.txt" 2>&1
  RC=$?
}
restaurado() { [ "$(cat "$D/$1/a.timer")" = active ] && [ "$(cat "$D/$1/b.timer")" = active ] && grep -q "RESTAURAÇÃO OK" "$D/estado-$1/restauracao.txt"; }

CUDA_VISIBLE_DEVICES=cvd-de-teste janela normal 20 sh -c 'echo "cvd=$CUDA_VISIBLE_DEVICES"; exit 0'
[ "$RC" = 0 ] && ok "composição: comando termina com 0" || ruim "esperava 0, veio $RC"
grep -q "cvd=cvd-de-teste" "$D/saida-normal.txt" && ok "ambiente atravessa a janela e a unidade" || ruim "ambiente não chegou: $(cat "$D/saida-normal.txt")"
restaurado normal && ok "timers pausados e restaurados (normal)" || ruim "timers não restaurados (normal)"
grep -q "^stop a.timer" "$D/normal/log" && ok "os timers foram de fato parados durante a janela" || ruim "nenhum stop registrado"

janela limite 2 sleep 30
[ "$RC" = 124 ] && ok "limite de dentro estourado: 124 chega até fora" || ruim "esperava 124, veio $RC"
restaurado limite && ok "timers restaurados depois do limite" || ruim "timers não restaurados depois do limite"
[ ! -e "$D/pend-limite" ] && ok "ponteiro pendente apagado" || ruim "ponteiro pendente ficou"

janela falha 20 sh -c 'exit 3'
[ "$RC" = 3 ] && ok "código do comando propagado (3)" || ruim "esperava 3, veio $RC"
restaurado falha && ok "timers restaurados depois da falha" || ruim "timers não restaurados depois da falha"
[ -z "$(systemctl --user list-units --all --no-legend 'lastro-limite-*' 2>/dev/null)" ] && ok "nenhuma unidade lastro-limite sobrou" || ruim "unidade sobrou"
echo ---
[ "$FALHAS" = 0 ] && echo "TUDO OK" || echo "$FALHAS FALHA(S)"
exit "$FALHAS"
