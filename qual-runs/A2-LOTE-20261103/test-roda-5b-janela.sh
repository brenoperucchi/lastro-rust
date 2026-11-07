#!/usr/bin/env bash
# Teste do roda-5b-janela.sh com systemctl FALSO (nenhum timer real é pausado), binário de MENTIRA e SEM GPU. A composição janela-e4b.sh + roda-com-limite.sh
# é a real (unidades transitórias de `systemd --user` que somem sozinhas); o binário falso só registra argumentos e ambiente. Cria pasta com mktemp -d e NÃO apaga nada.
set -u
unset E4B_GPU_AUTORIZADO E4B_MEDIDA_AUTORIZADA MOE_RT_REQUIRE_GPU LEITURA_CORRECAO_AUTORIZADA LEITURA_VELOCIDADE_AUTORIZADA PRIMEIRA_PALAVRA_AUTORIZADA
export CUDA_VISIBLE_DEVICES=
AQUI=$(cd "$(dirname "$0")" && pwd -P)
J=$AQUI/roda-5b-janela.sh
UUID=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd
D=$(mktemp -d -p "${TMPDIR:-/tmp}" janela-5b-teste.XXXXXX) || exit 1
echo "pasta de teste: $D"
falhas=0
confere() { if eval "$2"; then echo "ok: $1"; else echo "FALHOU: $1"; falhas=$((falhas+1)); fi; }
cat > "$D/fake-systemctl" <<'F'
#!/bin/bash
[ "$1" = --user ] && shift
a=$1; u=${2:-}
echo "$a $u" >> "$FAKE_DIR/log"
case $a in
  is-active) [ -e "$FAKE_DIR/$u" ] && cat "$FAKE_DIR/$u" || echo inactive; [ "$(cat "$FAKE_DIR/$u" 2>/dev/null)" = active ];;
  is-enabled) echo enabled;;
  cat) [ -e "$FAKE_DIR/$u" ];;
  stop) echo inactive > "$FAKE_DIR/$u";;
  start) echo active > "$FAKE_DIR/$u";;
  *) exit 9;;
esac
F
chmod +x "$D/fake-systemctl"
printf '# lista de teste\na.timer\nb.timer\n' > "$D/timers.txt"
# Binário falso: registra argumentos e ambiente; FALSO_RC controla o código; para --consolida grava decisao.txt; repetições frias gravam o arquivo.
cat > "$D/falso" <<'F'
#!/usr/bin/env bash
N=$(( $(cat "$FALSO_CONTA" 2>/dev/null || echo 0) + 1 )); echo "$N" > "$FALSO_CONTA"
echo "$N $*" >> "$FALSO_ARGS"
echo "env: portao=${PRIMEIRA_PALAVRA_AUTORIZADA:-} uuid=${CUDA_VISIBLE_DEVICES:-}" >> "$FALSO_ARGS"
SAIDA=; CEN=; REP=; MOD=
while [ $# -gt 0 ]; do case $1 in --saida) SAIDA=$2; shift 2;; --cenario) CEN=$2; shift 2;; --rep) REP=$2; shift 2;; --celula) MOD=${2%%:*}; shift 2;; --consolida) SAIDA=$2; shift 2;; *) shift;; esac; done
if [ -n "$CEN" ]; then echo ok > "$SAIDA/frio-$MOD-$CEN-rep$REP.json"; else mkdir -p "$SAIDA"; echo ok > "$SAIDA/decisao.txt"; fi
exit "${FALSO_RC:-0}"
F
chmod +x "$D/falso"
# nvidia-smi falso e loadavg falso: a coleta de carga externa nunca toca a GPU real nem depende da máquina.
cat > "$D/fake-smi" <<'F'
#!/usr/bin/env bash
case $* in *query-gpu*) echo "uuid, memory.used [MiB], utilization.gpu [%]"; echo "GPU-falsa, 1 MiB, 0 %";; *) echo "123, 5 MiB, /usr/bin/falso";; esac
F
chmod +x "$D/fake-smi"
echo "0.10 0.20 0.30 1/100 999" > "$D/loadavg"
export NVIDIA_SMI=$D/fake-smi LOADAVG_ARQ=$D/loadavg
roda() { # nome etapa [args extras...]; define RC
  local n=$1 e=$2; shift 2
  export FAKE_DIR=$D/$n FALSO_CONTA=$D/conta-$n FALSO_ARGS=$D/args-$n
  mkdir -p "$FAKE_DIR"
  echo active > "$FAKE_DIR/a.timer"; echo active > "$FAKE_DIR/b.timer"; echo inactive > "$FAKE_DIR/a.service"; echo inactive > "$FAKE_DIR/b.service"; : > "$FAKE_DIR/log"
  SYSTEMCTL="$D/fake-systemctl" PENDENTE="$D/pend-$n" JANELA_PASSO_ESPERA=0.1 XDG_RUNTIME_DIR=/run/user/fake JANELA_TIMERS="$D/timers.txt" \
    MODELO_1B=/m/1b MODELO_3B=/m/3b "$J" --etapa "$e" --bin "$D/falso" --saida "$D/saida-$n" --estado-janela "$D/estado-$n" --limite 30 "$@" > "$D/out-$n" 2>&1
  RC=$?
}
export LIMITE_PASSA="PATH HOME CUDA_VISIBLE_DEVICES FALSO_CONTA FALSO_ARGS FALSO_RC"

# 1. recusas ANTES de pausar timers ou chamar o binário
roda sem-portao correcao; confere "sem portão: recusa 2, binário e timers intocados" "[ $RC = 2 ] && [ ! -f $D/conta-sem-portao ] && ! grep -q '^stop' $D/sem-portao/log"
export PRIMEIRA_PALAVRA_AUTORIZADA=1
roda sem-uuid correcao; confere "sem UUID: recusa 2" "[ $RC = 2 ] && [ ! -f $D/conta-sem-uuid ]"
export CUDA_VISIBLE_DEVICES=GPU-00000000-0000-0000-0000-000000000000
roda uuid-errado correcao; confere "UUID errado (outra GPU): recusa 2" "[ $RC = 2 ] && [ ! -f $D/conta-uuid-errado ]"
export CUDA_VISIBLE_DEVICES=$UUID
export PRIMEIRA_PALAVRA_AUTORIZADA=0
roda portao-zero correcao; confere "portão com valor diferente de 1: recusa 2" "[ $RC = 2 ] && [ ! -f $D/conta-portao-zero ]"
export PRIMEIRA_PALAVRA_AUTORIZADA=1
roda etapa-ruim outra; confere "etapa desconhecida: recusa 2" "[ $RC = 2 ] && [ ! -f $D/conta-etapa-ruim ]"
mkdir -p "$D/saida-ocupada"; echo x > "$D/saida-ocupada/a"
roda ocupada correcao; confere "--saida não vazia: recusa 2 sem chamar o binário" "[ $RC = 2 ] && [ ! -f $D/conta-ocupada ]"
roda limite-alto correcao --limite 1801; confere "limite acima de 1800: recusa 2" "[ $RC = 2 ] && [ ! -f $D/conta-limite-alto ]"

# 2. correção: composição completa
roda corr correcao
confere "correção: sai 0" "[ $RC = 0 ]"
confere "correção: o binário recebeu --modo correcao, as duas células e a saída" "grep -q -- '--modo correcao --celula 1b:/m/1b --celula 3b:/m/3b --saida $D/saida-corr' $D/args-corr"
confere "correção: portão e UUID atravessam as duas camadas" "grep -q 'env: portao=1 uuid=$UUID' $D/args-corr"
confere "correção: timers parados e restaurados" "grep -q '^stop a.timer' $D/corr/log && [ \"\$(cat $D/corr/a.timer)\" = active ] && grep -q 'RESTAURAÇÃO OK' $D/estado-corr/restauracao.txt"

confere "carga externa: antes e depois ao lado da pasta da janela" "[ -f $D/saida-corr.carga-antes.txt ] && [ -f $D/saida-corr.carga-depois.txt ] && [ ! -e $D/saida-corr/carga-antes.txt ]"
confere "carga externa: conteúdo (hora, GPU, processos, loadavg, binário)" "grep -q '^início ' $D/saida-corr.carga-antes.txt && grep -q '^fim ' $D/saida-corr.carga-depois.txt && grep -q 'GPU-falsa' $D/saida-corr.carga-antes.txt && grep -q '^123, 5 MiB' $D/saida-corr.carga-depois.txt && grep -q '^loadavg 0.10' $D/saida-corr.carga-antes.txt && grep -q '^binário [0-9a-f]\\{64\\}' $D/saida-corr.carga-depois.txt"
confere "carga externa: recusas não criam arquivo de carga" "[ ! -e $D/saida-sem-portao.carga-antes.txt ] && [ ! -e $D/saida-ocupada.carga-antes.txt ]"

# 3. quente
roda qte quente
confere "quente: sai 0 e usa --modo velocidade --estado quente" "[ $RC = 0 ] && grep -q -- '--modo velocidade --estado quente --celula 1b:/m/1b --celula 3b:/m/3b --saida $D/saida-qte' $D/args-qte"

# 4. frio: 12 repetições e consolidação, cada uma em processo novo
roda fr frio
confere "frio: sai 0" "[ $RC = 0 ]"
confere "frio: 13 chamadas (12 repetições e a consolidação)" "[ \$(cat $D/conta-fr) = 13 ]"
confere "frio: 12 arquivos de repetição e a decisão" "[ \$(ls $D/saida-fr | grep -c '^frio-.*json') = 12 ] && [ -f $D/saida-fr/decisao.txt ]"
confere "frio: ambiente atravessa as 13 chamadas" "[ \$(grep -c 'env: portao=1 uuid=$UUID' $D/args-fr) = 13 ]"

# 5. o código do binário e o limite chegam até fora; timers restaurados
export FALSO_RC=1
roda rc1 correcao; confere "código do binário (1) propaga" "[ $RC = 1 ] && [ \"\$(cat $D/rc1/a.timer)\" = active ]"
confere "carga depois também com o binário falhando" "[ -f $D/saida-rc1.carga-depois.txt ]"
unset FALSO_RC
export NVIDIA_SMI=$D/nao-existe
roda smi-ausente correcao; confere "nvidia-smi ausente: coleta não muda o código (0)" "[ $RC = 0 ] && grep -q indisponível $D/saida-smi-ausente.carga-antes.txt"
export NVIDIA_SMI=$D/fake-smi
cat > "$D/lento" <<'F'
#!/usr/bin/env bash
sleep 30
F
chmod +x "$D/lento"
export FAKE_DIR=$D/lim FALSO_CONTA=$D/conta-lim FALSO_ARGS=$D/args-lim; mkdir -p "$FAKE_DIR"
echo active > "$FAKE_DIR/a.timer"; echo active > "$FAKE_DIR/b.timer"; echo inactive > "$FAKE_DIR/a.service"; echo inactive > "$FAKE_DIR/b.service"; : > "$FAKE_DIR/log"
SYSTEMCTL="$D/fake-systemctl" PENDENTE="$D/pend-lim" JANELA_PASSO_ESPERA=0.1 XDG_RUNTIME_DIR=/run/user/fake JANELA_TIMERS="$D/timers.txt" \
  "$J" --etapa correcao --bin "$D/lento" --saida "$D/saida-lim" --estado-janela "$D/estado-lim" --limite 2 > "$D/out-lim" 2>&1; rc=$?
confere "limite de tempo estourado: 124 e timers restaurados" "[ $rc = 124 ] && [ \"\$(cat $D/lim/a.timer)\" = active ]"
if [ "$falhas" = 0 ]; then echo "TODOS OS TESTES PASSARAM"; exit 0; else echo "$falhas FALHA(S)"; exit 1; fi
