#!/usr/bin/env bash
# Teste do roda-5d-janela.sh com systemctl FALSO (nenhum timer real é pausado), binário `lastro` de MENTIRA, textos, referências e modelos falsos e SEM GPU. A
# composição janela-e4b.sh + roda-com-limite.sh é a real (unidades transitórias de `systemd --user`). Cria pasta com mktemp -d e NÃO apaga nada.
set -u
unset E4B_GPU_AUTORIZADO E4B_MEDIDA_AUTORIZADA MOE_RT_REQUIRE_GPU LEITURA_CORRECAO_AUTORIZADA LEITURA_VELOCIDADE_AUTORIZADA PRIMEIRA_PALAVRA_AUTORIZADA
export CUDA_VISIBLE_DEVICES=
AQUI=$(cd "$(dirname "$0")" && pwd -P)
J=$AQUI/roda-5d-janela.sh
UUID=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd
D=$(mktemp -d -p "${TMPDIR:-/tmp}" janela-5d-teste.XXXXXX) || exit 1
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
# Textos e referências falsos (acentos, quebra de linha interna, sem newline final, como os reais).
mkdir -p "$D/textos" "$D/refs" "$D/m/1b" "$D/m/3b"
printf 'Pergunta curta com acento: ação?' > "$D/textos/texto-curto.txt"
printf 'Linha um\nLinha dois ñ' > "$D/textos/texto-medio.txt"
printf 'Texto longo '"'"'entre aspas'"'"' e "duplas" $HOME `x`' > "$D/textos/texto-longo.txt"
for m in 1b 3b; do
  echo '{"x":1}' > "$D/m/$m/tokenizer.json"
  python3 -I - "$D/refs/ref5c-$m.json" "$m" <<'PY'
import json, sys
m = sys.argv[2]
json.dump({"meta": {}, "casos": [{"nome": n, "texto_gerado": "resp %s %s: çã\n  fim" % (m, n)} for n in ("curto", "medio", "longo")]}, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
PY
done
echo criterio5d > "$D/c5d.md"; echo criterio5c > "$D/c5c.md"
# Binário falso: acha o texto (argumento ou entrada padrão), registra o modo e responde com o texto_gerado da referência + newline.
cat > "$D/falso" <<'F'
#!/usr/bin/env bash
M=; T=; NOVOS=; DATA=
while [ $# -gt 0 ]; do case $1 in --modelo) M=$2; shift 2;; --novos) NOVOS=$2; shift 2;; --data) DATA=$2; shift 2;; *) T=$1; ARG=1; shift;; esac; done
if [ -z "${ARG:-}" ]; then T=$(cat; printf x); T=${T%x}; MODO=stdin; else MODO=arg; fi
NOME=; for n in curto medio longo; do [ "$(cat "$TEXTOS_FALSOS/texto-$n.txt"; printf x)" = "$T"x ] && NOME=$n; done
MOD=$(basename "$M")
echo "$MOD $NOME $MODO novos=$NOVOS data=$DATA portao=${PRIMEIRA_PALAVRA_AUTORIZADA:-} uuid=${CUDA_VISIBLE_DEVICES:-}" >> "$FALSO_LOG"
[ -n "$NOME" ] || exit 7
case ${FALSO_MODO:-ok} in
  rc1) echo falha >&2; exit 1;;
  morre) kill -9 $$;;
esac
python3 -I - "$REFS_FALSAS/ref5c-$MOD.json" "$NOME" "${FALSO_MODO:-ok}" "$MOD" <<'PY'
import json, sys
c = [x for x in json.load(open(sys.argv[1], encoding="utf-8"))["casos"] if x["nome"] == sys.argv[2]][0]
t = c["texto_gerado"]
if sys.argv[3] == "diff" and sys.argv[4] == "3b" and sys.argv[2] == "longo": t += "!"
sys.stdout.buffer.write((t + ("" if sys.argv[3] == "semnl" else "\n")).encode("utf-8"))
PY
F
chmod +x "$D/falso"
cat > "$D/recusas-falso" <<F
#!/usr/bin/env bash
echo "recusas falsas"; exit \$(cat "$D/recusas.rc")
F
chmod +x "$D/recusas-falso"; echo 0 > "$D/recusas.rc"
cat > "$D/fake-smi" <<'F'
#!/usr/bin/env bash
case $* in *query-gpu*) echo "uuid, memory.used [MiB], utilization.gpu [%]"; echo "GPU-falsa, 1 MiB, 0 %";; *) echo "123, 5 MiB, /usr/bin/falso";; esac
F
chmod +x "$D/fake-smi"; echo "0.10 0.20 0.30 1/100 999" > "$D/loadavg"
export NVIDIA_SMI=$D/fake-smi LOADAVG_ARQ=$D/loadavg
export LIMITE_PASSA="PATH HOME CUDA_VISIBLE_DEVICES FALSO_MODO FALSO_LOG TEXTOS_FALSOS REFS_FALSAS"
export TEXTOS_FALSOS=$D/textos REFS_FALSAS=$D/refs
roda() { # nome [args extras...]; define RC; env extra via variáveis já exportadas
  local n=$1; shift
  export FAKE_DIR=$D/$n FALSO_LOG=$D/log-$n; : > "$FALSO_LOG"
  mkdir -p "$FAKE_DIR"
  echo active > "$FAKE_DIR/a.timer"; echo active > "$FAKE_DIR/b.timer"; echo inactive > "$FAKE_DIR/a.service"; echo inactive > "$FAKE_DIR/b.service"; : > "$FAKE_DIR/log"
  SYSTEMCTL="$D/fake-systemctl" PENDENTE="$D/pend-$n" JANELA_PASSO_ESPERA=0.1 XDG_RUNTIME_DIR=/run/user/fake JANELA_TIMERS="$D/timers.txt" \
    MODELO_1B=$D/m/1b MODELO_3B=$D/m/3b TEXTOS_5C=$D/textos REF_5C=$D/refs CRITERIO_5D=$D/c5d.md CRITERIO_5C=$D/c5c.md RECUSAS_SH=$D/recusas-falso SEM_SHA_FIXO=1 \
    "$J" --bin "$D/falso" --saida "$D/saida-$n" --limite 30 "$@" > "$D/out-$n" 2>&1
  RC=$?
}
rotulo() { grep '^rótulo:' "$D/saida-$1/decisao.txt" 2>/dev/null | cut -d' ' -f2-; }

# 1. recusas ANTES de pausar timers ou chamar o binário
roda sem-portao; confere "sem portão: recusa 2, binário e timers intocados" "[ $RC = 2 ] && [ ! -s $D/log-sem-portao ] && ! grep -q '^stop' $D/sem-portao/log"
export PRIMEIRA_PALAVRA_AUTORIZADA=1
roda sem-uuid; confere "sem UUID: recusa 2" "[ $RC = 2 ] && [ ! -s $D/log-sem-uuid ]"
export CUDA_VISIBLE_DEVICES=GPU-00000000-0000-0000-0000-000000000000
roda uuid-errado; confere "UUID errado: recusa 2" "[ $RC = 2 ] && [ ! -s $D/log-uuid-errado ]"
export CUDA_VISIBLE_DEVICES=$UUID
export PRIMEIRA_PALAVRA_AUTORIZADA=0
roda portao-zero; confere "portão diferente de 1: recusa 2" "[ $RC = 2 ] && [ ! -s $D/log-portao-zero ]"
export PRIMEIRA_PALAVRA_AUTORIZADA=1
mkdir -p "$D/saida-ocupada"; echo x > "$D/saida-ocupada/a"
roda ocupada; confere "--saida não vazia: recusa 2 sem chamar o binário" "[ $RC = 2 ] && [ ! -s $D/log-ocupada ]"
roda limite-alto --limite 1801; confere "limite acima de 1800: recusa 2" "[ $RC = 2 ] && [ ! -s $D/log-limite-alto ]"
roda opcao-ruim --nao-existe; confere "opção desconhecida: recusa 2" "[ $RC = 2 ] && [ ! -s $D/log-opcao-ruim ]"

# 2. caminho feliz: 6 execuções, PASSA
roda ok
confere "ok: sai 0 e PASSA" "[ $RC = 0 ] && [ \"\$(rotulo ok)\" = PASSA ]"
confere "ok: 6 execuções, 1b e 3b, curto/médio/longo, modos arg/stdin/arg" "[ \$(wc -l < $D/log-ok) = 6 ] && grep -q '^1b curto arg ' $D/log-ok && grep -q '^1b medio stdin ' $D/log-ok && grep -q '^1b longo arg ' $D/log-ok && grep -q '^3b curto arg ' $D/log-ok && grep -q '^3b medio stdin ' $D/log-ok && grep -q '^3b longo arg ' $D/log-ok"
confere "ok: --novos 64, --data do critério, portão e UUID atravessam" "[ \$(grep -c 'novos=64 data=November 07, 2026 portao=1 uuid=$UUID' $D/log-ok) = 6 ]"
confere "ok: saída igual byte a byte ao esperado, 6 de 6" "grep -q 'saídas iguais: 6 de 6' $D/saida-ok/decisao.txt && cmp -s $D/saida-ok/1b-medio.out $D/saida-ok/1b-medio.esperado"
confere "ok: tempos e integridade na decisão" "grep -q 'total .* ms, primeiro byte' $D/saida-ok/decisao.txt && grep -q 'integridade: íntegra' $D/saida-ok/decisao.txt && grep -q '^binario ' $D/saida-ok/integridade-antes.txt && grep -q 'recusas: OK' $D/saida-ok/decisao.txt"
confere "ok: timers parados e restaurados; carga ao lado da pasta" "grep -q '^stop a.timer' $D/ok/log && [ \"\$(cat $D/ok/a.timer)\" = active ] && grep -q 'RESTAURAÇÃO OK' $D/saida-ok-estado-janela/restauracao.txt && [ -f $D/saida-ok.carga-antes.txt ] && [ -f $D/saida-ok.carga-depois.txt ]"
confere "ok: texto longo com aspas e \$ passa intacto pelo argumento" "cmp -s $D/saida-ok/3b-longo.out $D/saida-ok/3b-longo.esperado"

# 3. diferença de texto: REPROVA
export FALSO_MODO=diff; roda dif
confere "uma saída diferente (3b longo): REPROVA, 5 de 6" "[ \"\$(rotulo dif)\" = REPROVA ] && grep -q 'saídas iguais: 5 de 6' $D/saida-dif/decisao.txt && grep -q '3b longo (arg): REPROVA' $D/saida-dif/decisao.txt"
export FALSO_MODO=semnl; roda semnl
confere "sem newline final: REPROVA em todas" "[ \"\$(rotulo semnl)\" = REPROVA ]"
export FALSO_MODO=rc1; roda rc1
confere "código 1 do binário: REPROVA" "[ \"\$(rotulo rc1)\" = REPROVA ] && [ \$RC = 0 ]"
export FALSO_MODO=morre; roda morre
confere "binário morto por sinal: SEM PROVA" "[ \"\$(rotulo morre)\" = 'SEM PROVA' ]"
unset FALSO_MODO

# 4. recusas do binário falhando
echo 1 > "$D/recusas.rc"; roda rec1
confere "recusas falhando (1): REPROVA" "[ \"\$(rotulo rec1)\" = REPROVA ] && grep -q 'recusas: FALHARAM' $D/saida-rec1/decisao.txt"
echo 3 > "$D/recusas.rc"; roda rec3
confere "recusas sem resultado (3): SEM PROVA" "[ \"\$(rotulo rec3)\" = 'SEM PROVA' ]"
echo 0 > "$D/recusas.rc"

# 5. integridade: referência ou texto que muda durante a janela derruba tudo
cat > "$D/falso-muda" <<'F'
#!/usr/bin/env bash
echo mudou >> "$TEXTOS_FALSOS/texto-curto.txt"; exit 0
F
chmod +x "$D/falso-muda"
cp "$D/textos/texto-curto.txt" "$D/curto.bak"
export FAKE_DIR=$D/mud FALSO_LOG=$D/log-mud; mkdir -p "$FAKE_DIR"; : > "$FALSO_LOG"
echo active > "$FAKE_DIR/a.timer"; echo active > "$FAKE_DIR/b.timer"; echo inactive > "$FAKE_DIR/a.service"; echo inactive > "$FAKE_DIR/b.service"; : > "$FAKE_DIR/log"
SYSTEMCTL="$D/fake-systemctl" PENDENTE="$D/pend-mud" JANELA_PASSO_ESPERA=0.1 XDG_RUNTIME_DIR=/run/user/fake JANELA_TIMERS="$D/timers.txt" \
  MODELO_1B=$D/m/1b MODELO_3B=$D/m/3b TEXTOS_5C=$D/textos REF_5C=$D/refs CRITERIO_5D=$D/c5d.md CRITERIO_5C=$D/c5c.md RECUSAS_SH=$D/recusas-falso SEM_SHA_FIXO=1 \
  "$J" --bin "$D/falso-muda" --saida "$D/saida-mud" --limite 30 > "$D/out-mud" 2>&1
confere "texto alterado durante a janela: integridade quebrada, SEM PROVA" "[ \"\$(rotulo mud)\" = 'SEM PROVA' ] && grep -q 'integridade: QUEBRADA' $D/saida-mud/decisao.txt"
cp "$D/curto.bak" "$D/textos/texto-curto.txt"

# 6. sha congelado diferente (sem SEM_SHA_FIXO): integridade quebrada
export FAKE_DIR=$D/fixo FALSO_LOG=$D/log-fixo; mkdir -p "$FAKE_DIR"; : > "$FALSO_LOG"
echo active > "$FAKE_DIR/a.timer"; echo active > "$FAKE_DIR/b.timer"; echo inactive > "$FAKE_DIR/a.service"; echo inactive > "$FAKE_DIR/b.service"; : > "$FAKE_DIR/log"
SYSTEMCTL="$D/fake-systemctl" PENDENTE="$D/pend-fixo" JANELA_PASSO_ESPERA=0.1 XDG_RUNTIME_DIR=/run/user/fake JANELA_TIMERS="$D/timers.txt" \
  MODELO_1B=$D/m/1b MODELO_3B=$D/m/3b TEXTOS_5C=$D/textos REF_5C=$D/refs CRITERIO_5D=$D/c5d.md CRITERIO_5C=$D/c5c.md RECUSAS_SH=$D/recusas-falso \
  "$J" --bin "$D/falso" --saida "$D/saida-fixo" --limite 30 > "$D/out-fixo" 2>&1
confere "sha diferente do congelado: SEM PROVA e a nota diz qual" "[ \"\$(rotulo fixo)\" = 'SEM PROVA' ] && grep -q 'não o congelado' $D/saida-fixo/decisao.txt"

# 7. limite de tempo estourado: 124 e timers restaurados
cat > "$D/lento" <<'F'
#!/usr/bin/env bash
sleep 30
F
chmod +x "$D/lento"
export FAKE_DIR=$D/lim FALSO_LOG=$D/log-lim; mkdir -p "$FAKE_DIR"; : > "$FALSO_LOG"
echo active > "$FAKE_DIR/a.timer"; echo active > "$FAKE_DIR/b.timer"; echo inactive > "$FAKE_DIR/a.service"; echo inactive > "$FAKE_DIR/b.service"; : > "$FAKE_DIR/log"
SYSTEMCTL="$D/fake-systemctl" PENDENTE="$D/pend-lim" JANELA_PASSO_ESPERA=0.1 XDG_RUNTIME_DIR=/run/user/fake JANELA_TIMERS="$D/timers.txt" \
  MODELO_1B=$D/m/1b MODELO_3B=$D/m/3b TEXTOS_5C=$D/textos REF_5C=$D/refs CRITERIO_5D=$D/c5d.md CRITERIO_5C=$D/c5c.md RECUSAS_SH=$D/recusas-falso SEM_SHA_FIXO=1 \
  "$J" --bin "$D/lento" --saida "$D/saida-lim" --limite 3 > "$D/out-lim" 2>&1; rc=$?
confere "limite estourado: 124, sem decisão gravada e timers restaurados" "[ $rc = 124 ] && [ ! -f $D/saida-lim/decisao.txt ] && [ \"\$(cat $D/lim/a.timer)\" = active ]"
if [ "$falhas" = 0 ]; then echo "TODOS OS TESTES PASSARAM"; exit 0; else echo "$falhas FALHA(S)"; exit 1; fi
