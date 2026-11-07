#!/usr/bin/env bash
# Teste do roda-frio.sh com binário de MENTIRA (sem GPU, sem o programa real). Cria uma pasta de trabalho com mktemp -d e NÃO apaga nada.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d) || exit 1
echo "pasta de trabalho: $T"
falhas=0
confere() { if eval "$2"; then echo "ok: $1"; else echo "FALHOU: $1"; falhas=$((falhas+1)); fi; }
# Binário falso: grava o arquivo da repetição e registra os argumentos; FALSO_RC controla o código (por chamada: FALSO_RC_<n>), FALSO_RECUSA_NA = chamada que recusa.
cat > "$T/falso" <<'F'
#!/usr/bin/env bash
N=$(( $(cat "$FALSO_CONTA" 2>/dev/null || echo 0) + 1 )); echo "$N" > "$FALSO_CONTA"
echo "$N $*" >> "$FALSO_ARGS"
echo "env: portao=${LEITURA_VELOCIDADE_AUTORIZADA:-} uuid=${CUDA_VISIBLE_DEVICES:-}" >> "$FALSO_ARGS"
if [ "$1" = "--consolida" ]; then echo "consolidado" > "$2/decisao.txt"; exit 0; fi
if [ "${FALSO_RECUSA_NA:-0}" = "$N" ]; then echo "recusado" >&2; exit 2; fi
SAIDA=; CEN=; REP=; MOD=
while [ $# -gt 0 ]; do case $1 in --saida) SAIDA=$2; shift 2;; --cenario) CEN=$2; shift 2;; --rep) REP=$2; shift 2;; --celula) MOD=${2%%:*}; shift 2;; *) shift;; esac; done
echo ok > "$SAIDA/frio-$MOD-$CEN-rep$REP.txt"
[ "${FALSO_FALHA_NA:-0}" = "$N" ] && exit 1
exit 0
F
chmod +x "$T/falso"
export FALSO_CONTA=$T/conta1 FALSO_ARGS=$T/args1
LEITURA_VELOCIDADE_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=GPU-x "$AQUI/roda-frio.sh" --saida "$T/s1" --bin "$T/falso" > "$T/out1" 2>&1; rc=$?
confere "caminho feliz sai 0" "[ $rc = 0 ]"
confere "13 chamadas (12 repetições e a consolidação)" "[ \$(grep -c . $T/conta1 >/dev/null; cat $T/conta1) = 13 ]"
confere "12 arquivos de repetição" "[ \$(ls $T/s1 | grep -c '^frio-') = 12 ]"
confere "ordem: 1b C 1, depois 1b C 2" "sed -n 1p $T/args1 | grep -q -- '--celula 1b:.* --cenario C --rep 1 ' && sed -n 3p $T/args1 | grep -q -- '--cenario C --rep 2 '"
confere "ordem: 3b só depois do 1b" "[ \$(grep -n -- '--celula 3b' $T/args1 | head -1 | cut -d: -f1) -gt \$(grep -n -- '--celula 1b' $T/args1 | tail -1 | cut -d: -f1) ]"
confere "a última chamada é --consolida --estado frio" "grep -- '--consolida' $T/args1 | grep -q -- '--estado frio'"
confere "ambiente herdado em todas as chamadas" "[ \$(grep -c 'env: portao=1 uuid=GPU-x' $T/args1) = 13 ]"
confere "decisao gravada pela consolidação" "[ -f $T/s1/decisao.txt ]"
# Pasta existente e não vazia é recusada.
export FALSO_CONTA=$T/conta2 FALSO_ARGS=$T/args2
"$AQUI/roda-frio.sh" --saida "$T/s1" --bin "$T/falso" > "$T/out2" 2>&1; rc=$?
confere "pasta não vazia: recusa com 2 e nenhuma chamada" "[ $rc = 2 ] && [ ! -f $T/conta2 ]"
# Recusa do processo (código 2) interrompe tudo, sem consolidar.
export FALSO_CONTA=$T/conta3 FALSO_ARGS=$T/args3 FALSO_RECUSA_NA=2
"$AQUI/roda-frio.sh" --saida "$T/s3" --bin "$T/falso" > "$T/out3" 2>&1; rc=$?
confere "recusa na 2ª chamada: sai 2 e para" "[ $rc = 2 ] && [ \$(cat $T/conta3) = 2 ] && ! grep -q -- '--consolida' $T/args3"
unset FALSO_RECUSA_NA
# Falha gravada (código 1) não interrompe: consolida e sai 3.
export FALSO_CONTA=$T/conta4 FALSO_ARGS=$T/args4 FALSO_FALHA_NA=5
"$AQUI/roda-frio.sh" --saida "$T/s4" --bin "$T/falso" > "$T/out4" 2>&1; rc=$?
confere "falha na 5ª repetição: segue, consolida e sai 3" "[ $rc = 3 ] && [ \$(cat $T/conta4) = 13 ] && [ -f $T/s4/decisao.txt ]"
unset FALSO_FALHA_NA
# Uso errado.
"$AQUI/roda-frio.sh" --bin "$T/falso" > /dev/null 2>&1; confere "sem --saida: 2" "[ $? = 2 ]"
"$AQUI/roda-frio.sh" --saida "$T/s5" --bin "$T/nao-existe" > /dev/null 2>&1; confere "binário inexistente: 2" "[ $? = 2 ]"
"$AQUI/roda-frio.sh" --saida "$T/s6" --outra x > /dev/null 2>&1; confere "opção desconhecida: 2" "[ $? = 2 ]"
if [ "$falhas" = 0 ]; then echo "TODOS OS TESTES PASSARAM"; exit 0; else echo "$falhas FALHA(S)"; exit 1; fi
