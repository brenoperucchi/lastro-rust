#!/usr/bin/env bash
# Teste do roda-f2b-janela.sh SEM GPU: célula FALSA (relógio virtual), nvidia-smi falso, systemctl falso. mktemp -d; este teste NÃO remove nada.
# O "binário" falso só delega os subcomandos de planejamento (--planeja-celula e --planeja-controle, CPU pura) ao binário congelado real; qualquer outra chamada sai 0 sem fazer nada.
# Parte 1 (modo --interno): ordem D A B2 B2 D A A B2 D, variáveis por braço, tetos de processo, planejamento (cinco, corte único, NÃO RODOU por célula, controle que não cabe), smoke, potência, extrator.
# Parte 2 (envelope): recusas, bloco obrigatório que não cabe, binário fora do FIXO, caminho feliz, --verifica-fixos.
set -u
unset E4B_GPU_AUTORIZADO E4B_MEDIDA_AUTORIZADA MOE_RT_REQUIRE_GPU PRIMEIRA_PALAVRA_AUTORIZADA LASTRO_PERFIL_BLOCO LASTRO_PERFIL_F2 LASTRO_PERFIL_F2B
export CUDA_VISIBLE_DEVICES=
AQUI=$(cd "$(dirname "$0")" && pwd -P)
J=$AQUI/roda-f2b-janela.sh
UUID=GPU-b22d9465-8552-9f97-812e-380b590645f8
PLAN_BIN=$AQUI/../../.herdr/binarios/primeira-palavra-f2b-22d0bbe0
D=$(mktemp -d -p "${TMPDIR:-/tmp}" f2-janela-teste.XXXXXX) || exit 1
echo "pasta de teste: $D"
falhas=0
confere() { if eval "$2"; then echo "ok: $1"; else echo "FALHOU: $1"; falhas=$((falhas+1)); fi; }
[ -x "$PLAN_BIN" ] || { echo "FALHOU: binário congelado ausente em $PLAN_BIN (preciso dos subcomandos de planejamento)"; exit 1; }
mkdir -p "$D/m/1b" "$D/m/3b" "$D/textos" "$D/refs"
for m in 1b 3b; do echo cfg > "$D/m/$m/config.json"; echo tok > "$D/m/$m/tokenizer.json"; done
echo st > "$D/m/1b/model.safetensors"; echo s1 > "$D/m/3b/model-00001-of-00002.safetensors"; echo s2 > "$D/m/3b/model-00002-of-00002.safetensors"; echo ix > "$D/m/3b/model.safetensors.index.json"
echo c > "$D/textos/texto-curto.txt"; echo m > "$D/textos/texto-medio.txt"; echo l > "$D/textos/texto-longo.txt"; echo '{"textos": []}' > "$D/textos/textos.json"
python3 -I - "$D" <<'PY'
import json, sys
d = sys.argv[1]
for m in ("1b", "3b"):
    json.dump({"casos": [{"nome": "curto", "ids": [1, 2], "gerados": list(range(64)), "gaps": [5.0] * 64}]}, open("%s/refs/ref5c-%s.json" % (d, m), "w"))
json.dump({"modelos": {"1b": {"curto": {"ids": [1, 2]}}, "3b": {"curto": {"ids": [1, 2]}}}}, open("%s/ids.json" % d, "w"))
PY
for f in crit crit-d1 crit-d2 crit-f2b plano; do echo "$f" > "$D/$f.md"; done
printf '#!/usr/bin/env bash\ncase "${1:-}" in --planeja-*) exec "%s" "$@";; esac\nexit 0\n' "$PLAN_BIN" > "$D/pp"; chmod +x "$D/pp"
: > "$D/proc-vazia"
cat > "$D/smi" <<'F'
#!/usr/bin/env bash
case $* in
  *"--query-gpu=power.limit --format"*) f=${0%/*}/pot-seq; if [ -s "$f" ]; then l=$(head -n 1 "$f"); tail -n +2 "$f" > "$f.t"; mv "$f.t" "$f"; [ "$l" = FALHA ] && exit 1; echo "$l"; else echo "370.00 W"; fi;;
  *"-i GPU-b22d9465-8552-9f97-812e-380b590645f8 --query-gpu=power.limit,pcie.link.width.current,pcie.link.gen.max"*) echo "370.00 W, 16, 4";;
  *query-gpu*) echo "uuid, memory.used [MiB], utilization.gpu [%]"; echo "GPU-falsa, 1 MiB, 0 %";;
  *pmon*) if [ -n "${FAKE_PMON_ARQ:-}" ]; then cat "$FAKE_PMON_ARQ"; else echo "# gpu pid type sm mem enc dec jpg ofa fb ccpm command"; echo "# Idx # C/G % % % % % % MB MB name"; fi;;
  *) echo;;
esac
F
chmod +x "$D/smi"
# célula falsa (roda-cmp-lastro.sh): imprime as linhas de estado segundo as variáveis do ambiente (a mesma regra do binário), grava os perfis, avança o relógio virtual em (n+1)*CUSTO s
cat > "$D/lastro-falso.sh" <<'F'
#!/usr/bin/env bash
modelo=; texto=; estado=; n=; saida=
while [ $# -gt 0 ]; do case $1 in --modelo) modelo=$2;; --texto) texto=$2;; --estado) estado=$2;; --n) n=$2;; --saida) saida=$2;; esac; shift; done
chave=$(basename "$saida")
echo "$chave n=$n texto=$texto FRAC=${CMP_FRACAO_SLOTS:-} CAMADAS=${CMP_CAMADAS_CPU:-} PB=${LASTRO_PERFIL_BLOCO-unset} F2=${LASTRO_PERFIL_F2-unset} F2B=${LASTRO_PERFIL_F2B-unset}" >> "$CMP_FAKE_LOG"
[ "${CMP_FAKE_RECUSA:-}" = "$chave" ] && { echo recusa >&2; exit 2; }
custo=$(( (n + 1) * ${CMP_FAKE_CUSTO:-10} )); [ -z "${CMP_FAKE_RELOGIO:-}" ] || echo $(( $(cat "$CMP_FAKE_RELOGIO") + custo )) > "$CMP_FAKE_RELOGIO"
[ "${CMP_FAKE_DORME:-}" != "$chave" ] || sleep 5
mkdir -p "$saida/p1/saida"
bl=desligado; f2=desligado; f2b=desligado
{ [ "${LASTRO_PERFIL_BLOCO-}" = 1 ] || [ -n "${CMP_FAKE_FORCE_LIGADO:-}" ]; } && bl=ligado
{ [ "${LASTRO_PERFIL_F2-}" = 1 ] || [ -n "${CMP_FAKE_FORCE_LIGADO:-}" ]; } && f2=ligado
{ [ "${LASTRO_PERFIL_F2B-}" = 1 ] || [ -n "${CMP_FAKE_FORCE_LIGADO:-}" ]; } && f2b=ligado
printf 'perfil-bloco: %s\nperfil-f2: %s\nperfil-f2b: %s\n' "$bl" "$f2" "$f2b" > "$saida/p1/stdout.txt"
[ $bl = ligado ] && echo '{"tipo": "perfil-bloco", "versao": 2, "pedidos": []}' > "$saida/p1/saida/perfil-bloco.json"
[ $f2 = ligado ] && echo '{"tipo": "perfil-f2", "versao": 1, "pedidos": []}' > "$saida/p1/saida/perfil-f2.json"
[ $f2b = ligado ] && echo '{"tipo": "perfil-f2b", "versao": 1, "pedidos": []}' > "$saida/p1/saida/perfil-f2b.json"
rc=0; [ "${CMP_FAKE_RC:-}" != "$chave" ] || rc=1
echo $rc > "$saida/p1/rc.txt"
printf '{"lado": "lastro", "serie": "L100", "modelo": "%s", "texto": "%s", "estado": "%s", "n_medidas": %s, "portao_ok": true, "portao_motivo": "", "rc": %s}\n' "$modelo" "$texto" "$estado" "$n" "$rc" > "$saida/celula.json"
exit $rc
F
chmod +x "$D/lastro-falso.sh"
export NVIDIA_SMI=$D/smi LASTRO_SH_CMP=$D/lastro-falso.sh MODELO_1B=$D/m/1b MODELO_3B=$D/m/3b PROC_LISTA=$D/proc-vazia
export TEXTOS_5C=$D/textos REF_5C=$D/refs IDS_CMP=$D/ids.json CRITERIO_CMP=$D/crit.md CRITERIO_D1=$D/crit-d1.md CRITERIO_D2=$D/crit-d2.md CRITERIO_F2B=$D/crit-f2b.md PLANO_F2B=$D/plano.md
export SEM_SHA_FIXO=1 CMP_SEM_MEMORIA=1 CMP_MODO_TESTE=1
interno() { # nome limite [modelo=1b]; define RC; relógio virtual zerado
  local n=$1 lim=$2 mod=${3:-1b}
  echo 0 > "$D/clock"; : > "$D/log-$n"
  PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID CMP_RELOGIO_ARQ=$D/clock CMP_FAKE_RELOGIO=$D/clock CMP_FAKE_LOG=$D/log-$n \
    "$J" --interno --modelos "$mod" --saida "$D/si-$n" --bin "$D/pp" --limite "$lim" > "$D/outi-$n" 2>&1
  RC=$?
}
nproc() { grep -c '^[0-9][0-9] ' "$D/si-$1/ordem.txt"; }

echo "== Parte 1: modo interno"
"$J" --interno --modelos 1b --saida "$D/si-sem" --bin "$D/pp" --limite 100 > "$D/outi-sem" 2>&1
confere "modo interno sem portão/UUID: recusa 2, nada criado" "[ $? = 2 ] && [ ! -e $D/si-sem/ordem.txt ]"
echo 0 > "$D/clock"
env -i PATH="$PATH" HOME="$HOME" SEM_SHA_FIXO=1 PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID "$J" --interno --modelos 1b --saida "$D/si-real1" --bin "$D/pp" --limite 100 > "$D/outi-real1" 2>&1
RC=$?
confere "interno + SEM_SHA_FIXO=1 sem CMP_MODO_TESTE: recusa 2, nada criado" "[ $RC = 2 ] && grep -q 'variável só de teste presente no modo real: SEM_SHA_FIXO' $D/outi-real1 && [ ! -e $D/si-real1 ]"
env -i PATH="$PATH" HOME="$HOME" PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID "$J" --interno --modelos 1b --saida "$D/si-real2" --bin "$D/pp" --limite 100 > "$D/outi-real2" 2>&1
RC=$?
confere "interno real com binário falso: recusa 2 por sha do FIXO, nada criado" "[ $RC = 2 ] && grep -q 'sha diferente do FIXO' $D/outi-real2 && [ ! -e $D/si-real2 ]"
for v in LASTRO_PERFIL_F2B LASTRO_PERFIL_F2 LASTRO_PERFIL_BLOCO F2B_TETO_FORCADO F2B_EXTRAI_PY F2_ANTERIOR_DIR PLANO_F2B CRITERIO_F2B; do
  env -i PATH="$PATH" HOME="$HOME" PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID "$v=1" "$J" --interno --modelos 1b --saida "$D/si-real-$v" --bin "$D/pp" --limite 100 > "$D/outi-real-$v" 2>&1
  confere "modo real com $v no ambiente: recusa 2 citando a variável, nada criado" "[ $? = 2 ] && grep -q 'variável só de teste presente no modo real: $v' $D/outi-real-$v && [ ! -e $D/si-real-$v ]"
done
# A. 3B folgado, com as DUAS variáveis exportadas no ambiente pai (os D e A continuam sem elas)
LASTRO_PERFIL_BLOCO=1 LASTRO_PERFIL_F2=1 interno A 1200 3b
confere "A: sai 0, decisao.txt, 12 processos (smoke, 9 de controle, curto e longo)" "[ $RC = 0 ] && [ -s $D/si-A/decisao.txt ] && [ \$(nproc A) = 12 ]"
confere "A: smoke primeiro (curto, n=1, braço B2); depois controle na ordem D A B2 B2 D A A B2 D (01..09)" "[ \"\$(sed -n 1p $D/si-A/ordem.txt | cut -d' ' -f2,3,4)\" = 'smoke-3b-L100-curto-B2 n=1 braco=B2' ] && [ \"\$(grep -o 'ctrl-3b-L100-medio-0[0-9]-[DAB]2*' $D/si-A/ordem.txt | sed 's/.*-//' | tr -d '\n')\" = DAB2B2DAAB2D ]"
confere "A: controle com n=3 no médio; curto e longo com n=5; células de dados em B" "[ \$(grep -c '^ctrl-.* n=3 texto=medio ' $D/log-A) = 9 ] && grep -q '^lastro-3b-L100-curto-quente n=5 texto=curto ' $D/log-A && grep -q '^lastro-3b-L100-longo-quente n=5 texto=longo ' $D/log-A && grep -q 'lastro-3b-L100-curto-quente braco=B2' $D/si-A/prova-variavel.txt"
confere "A: D roda sem NENHUMA variável (PB=unset F2=unset F2B=unset), mesmo com as duas no ambiente pai" "[ \$(grep -c 'ctrl-.*-D n=3 .*PB=unset F2=unset F2B=unset' $D/log-A) = 3 ] && ! grep -E 'ctrl-.*-D n=.*(PB=1|F2=1|F2B=1)' $D/log-A"
confere "A: A roda só com o bloco (PB=1 F2=unset F2B=unset); B com as duas (PB=1 F2=1 F2B=1), incluindo smoke, curto e longo" "[ \$(grep -c 'ctrl-.*-A n=3 .*PB=1 F2=unset F2B=unset' $D/log-A) = 3 ] && [ \$(grep -c '^ctrl-.*-B2 n=3 .*PB=1 F2=1 F2B=1' $D/log-A) = 3 ] && grep -q '^smoke-3b-L100-curto-B2 n=1 .*PB=1 F2=1 F2B=1' $D/log-A && [ \$(grep -c '^lastro-.* PB=1 F2=1 F2B=1' $D/log-A) = 2 ]"
confere "A: série L100 em todos (FRAC=1.0, CAMADAS vazio)" "[ \$(grep -c 'FRAC=1.0 CAMADAS= ' $D/log-A) = 12 ]"
confere "A: prova-variavel.txt com 12 linhas, nenhuma INCOERENTE, com '-u LASTRO_PERFIL_F2' nos D e A" "[ \$(wc -l < $D/si-A/prova-variavel.txt) = 12 ] && ! grep -q INCOERENTE $D/si-A/prova-variavel.txt && grep 'braco=D' $D/si-A/prova-variavel.txt | grep -q -- '-u LASTRO_PERFIL_F2' && grep 'braco=A' $D/si-A/prova-variavel.txt | grep -q -- '-u LASTRO_PERFIL_F2'"
confere "A: nenhum perfil nos D; só perfil-bloco nos A; os dois nos B (smoke + 3 + curto + longo)" "[ -z \"\$(find $D/si-A/celulas -path '*/celulas/ctrl-*-D/*' -name 'perfil-*.json')\" ] && [ -z \"\$(find $D/si-A/celulas -path '*/celulas/ctrl-*-A/*' -name perfil-f2.json)\" ] && [ \$(find $D/si-A/celulas -name perfil-f2.json | wc -l) = 6 ] && [ \$(find $D/si-A/celulas -name perfil-bloco.json | wc -l) = 9 ]"
confere "A: plano.txt traz a tabela de tetos do 3B (smoke 60, médio 90, curto 90, longo 120) e o do 1B está certo" "grep -q 'smoke 60 s; médio 90 s; curto 90 s; longo 120 s; teto global 1200 s; folga 180 s' $D/si-A/plano.txt"
interno A1 1200 1b
confere "A1: tetos do 1B (smoke 60, médio 60, curto 60, longo 75)" "grep -q 'smoke 60 s; médio 60 s; curto 60 s; longo 75 s' $D/si-A1/plano.txt"
confere "A: ordem.txt no formato do extrator (NN chave n= braco= inicio= fim= rc=), manifesto com potência OK, orçamento com o modelo, sem cortes, extrator rodou (resultado gravado)" "grep -qE '^02 ctrl-3b-L100-medio-01-D n=3 braco=D inicio=[0-9]+s fim=[0-9]+s rc=0\$' $D/si-A/ordem.txt && grep -q '^potencia 370.00_W 370.00_W OK' $D/si-A/integridade-fixos.txt && grep -q '\"modelos\": \[\"3b\"\]' $D/si-A/orcamento.json && grep -q '\"cortes\": \[\]' $D/si-A/orcamento.json && [ -s $D/si-A/f2b-host-resultado.json ] && [ -s $D/si-A/f2b-host-resultado.md ]"
confere "A: decisao.txt imprime o sha do roteiro" "grep -q \"roteiro sha256 \$(sha256sum $J | cut -d' ' -f1)\" $D/si-A/decisao.txt"
# B. corte único, permanente: custo 25 s por pedido (1B): smoke 50 + controle 900 = 950; curto N=5 não cabe, N=3 cabe (corte), longo já cortado: NÃO RODOU pela célula atual com N=3
CMP_FAKE_CUSTO=25 interno B 1200 1b
confere "B: corte 5->3 registrado uma vez; curto roda com n=3; longo NÃO RODOU (célula atual com N=3 não cabe), nunca vira cinco" "grep -q 'quente cai de 5 para 3' $D/si-B/orcamento.json && grep -q '\"cortado\": true' $D/si-B/orcamento.json && grep -q '^lastro-1b-L100-curto-quente n=3 ' $D/log-B && ! grep -q '^lastro-1b-L100-longo' $D/log-B && grep -q 'lastro-1b-L100-longo-quente: não cabe nem com 3' $D/si-B/orcamento.json && [ \$(grep -c 'quente cai' $D/si-B/orcamento-log.txt) = 1 ]"
confere "B: o controle não foi cortado (9 processos com n=3)" "[ \$(grep -c '^ctrl-.* n=3 ' $D/log-B) = 9 ]"
# C. controle que não cabe: limite 600 (smoke 20 + 9 x 60 + 180 > 600): nenhum processo de controle, os nove NÃO RODOU com o prefixo ctrl-; curto e longo ainda rodam
interno C 600 1b
confere "C: controle não cabe: nenhum ctrl rodou, 9 NÃO RODOU (prefixo ctrl-1b-L100-), células seguem sem corte do controle" "[ \$(grep -c '^ctrl-' $D/log-C) = 0 ] && [ \$(grep -o 'ctrl-1b-L100-medio-0[0-9]-[DAB]2*: o controle de 9 processos não cabe' $D/si-C/orcamento.json | wc -l) = 9 ] && grep -q '^lastro-1b-L100-curto-quente n=5 ' $D/log-C"
# D. smoke falha (rc 1): a janela segue; decisão cita; o controle roda inteiro
CMP_FAKE_RC=smoke-1b-L100-curto-B2 interno D 1200 1b
confere "D: smoke com rc 1: a janela segue (12 processos), decisão diz SMOKE FALHOU, extrator marca (resultado gravado)" "[ \$(nproc D) = 12 ] && grep -q 'SMOKE FALHOU' $D/si-D/decisao.txt && [ -s $D/si-D/f2b-host-resultado.json ]"
CMP_FAKE_RECUSA=smoke-1b-L100-curto-B2 interno D2 1200 1b
confere "D2: smoke recusa o ambiente (código 2): nenhum processo depois, janela interrompida, decisão gravada" "[ \$(nproc D2) = 1 ] && grep -q 'recusou o ambiente' $D/si-D2/orcamento-log.txt && [ -s $D/si-D2/decisao.txt ] && ! grep -q '^ctrl-' $D/log-D2"
# E. controle com erro: encerra sem repetição
CMP_FAKE_RC=ctrl-1b-L100-medio-03-B2 interno E 1200 1b
confere "E: processo de controle com rc 1: 04..09 NÃO RODOU, sem repetição, curto e longo seguem" "[ \$(grep -c '^ctrl-' $D/log-E) = 3 ] && grep -q 'ctrl-1b-L100-medio-04-B2: controle encerrado' $D/si-E/orcamento.json && grep -q '^lastro-1b-L100-curto' $D/log-E"
CMP_FAKE_RECUSA=ctrl-1b-L100-medio-02-A interno E2 1200 1b
confere "E2: controle recusa o ambiente (código 2): janela interrompida, o resto NÃO RODOU, células não rodam" "grep -q 'recusou o ambiente' $D/si-E2/orcamento-log.txt && ! grep -q '^lastro-1b-L100-curto' $D/log-E2"
# F. teto de processo: 1 s forçado, o processo dorme 5 s: morre com 124, registrado, inválido, sem repetição
F2B_TETO_FORCADO=1 CMP_FAKE_DORME=smoke-1b-L100-curto-B2 interno F 1200 1b
confere "F: processo acima do teto morre (rc 124), entra em teto-excedido.txt e no log, uma só tentativa" "grep -q '^smoke-1b-L100-curto-B2 teto=1s rc=124' $D/si-F/teto-excedido.txt && grep -q 'teto de processo excedido em smoke-1b-L100-curto-B2' $D/si-F/orcamento-log.txt && [ \$(grep -c '^smoke-1b' $D/log-F) = 1 ]"
# G. variável vazando para um D/A: INCOERENTE
CMP_FAKE_FORCE_LIGADO=1 interno G 1200 1b
confere "G: binário que liga em D e A: INCOERENTE nos 6 processos D e A e a decisão conta" "[ \$(grep -c 'braco=[DA].*INCOERENTE\$' $D/si-G/prova-variavel.txt) = 6 ] && grep -q '6 processo(s) INCOERENTE' $D/si-G/decisao.txt"
# H. potência
printf '370.00 W\n300.00 W\n' > "$D/pot-seq"; interno H 1200 1b
confere "H (370 no início, 300 ao fim): manifesto FALHA, decisão QUEBRADA, extrator recusa" "[ $RC = 0 ] && grep -qx 'potencia inicio=370.00_W fim=300.00_W FALHA' $D/si-H/integridade-fixos.txt && grep -q 'integridade: QUEBRADA' $D/si-H/decisao.txt && grep -q 'RECUSADO' $D/si-H/decisao.txt"
: > "$D/pot-seq"
# I. cargo test no modo interno; planejador indisponível (falha fechada)
printf '4242 cargo test --lib foo\n' > "$D/proc-cargo"
PROC_LISTA=$D/proc-cargo interno I 1200 1b
confere "I (cargo test em execução, modo interno): recusa 2, nada rodou" "[ $RC = 2 ] && [ ! -s $D/log-I ] && grep -q 'cargo test' $D/outi-I"
printf '#!/usr/bin/env bash\nexit 0\n' > "$D/pp-mudo"; chmod +x "$D/pp-mudo"
echo 0 > "$D/clock"; : > "$D/log-J"
PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID CMP_RELOGIO_ARQ=$D/clock CMP_FAKE_RELOGIO=$D/clock CMP_FAKE_LOG=$D/log-J "$J" --interno --modelos 1b --saida "$D/si-J" --bin "$D/pp-mudo" --limite 1200 > "$D/outi-J" 2>&1
confere "J (planejador sem resposta): falha fechada, controle e células NÃO RODOU, só o smoke roda" "[ \$(grep -c '' $D/log-J) = 1 ] && grep -q 'planejador indisponível' $D/si-J/orcamento.json && grep -q 'planejador: erro' $D/si-J/orcamento.json"
for casom in "" "5b" "1b,3b" "ambos"; do
  : > "$D/log-m"; if [ -z "$casom" ]; then args=(); else args=(--modelos "$casom"); fi
  PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID CMP_FAKE_LOG=$D/log-m "$J" --interno "${args[@]}" --saida "$D/si-m-$casom" --bin "$D/pp" --limite 100 > "$D/outi-m" 2>&1; rcm=$?
  confere "--modelos '${casom:-ausente}' no modo interno: recusa 2, nada rodou" "[ $rcm = 2 ] && [ ! -s $D/log-m ]"
done
interno K 1200 3b
confere "K (3B): safetensors do 3B e não os do 1B no manifesto" "grep -q '^safetensors-3b-2 ' $D/si-K/integridade-antes.txt && ! grep -q '^safetensors-1b' $D/si-K/integridade-antes.txt"

echo "== Parte 2: envelope"
cat > "$D/fake-systemctl" <<'F'
#!/bin/bash
[ "$1" = --user ] && shift
a=$1; u=${2:-}
echo "$a $u" >> "$FAKE_DIR/log"
case $a in
  is-active) [ -e "$FAKE_DIR/$u" ] && cat "$FAKE_DIR/$u" || echo inactive; [ "$(cat "$FAKE_DIR/$u" 2>/dev/null)" = active ];;
  is-enabled) echo enabled;;
  list-unit-files) echo "a.timer enabled"; echo "b.timer enabled";;
  list-timers) echo "NEXT a.timer"; echo "NEXT b.timer";;
  cat) [ -e "$FAKE_DIR/$u" ];;
  stop) echo inactive > "$FAKE_DIR/$u";;
  start) echo active > "$FAKE_DIR/$u";;
  *) exit 9;;
esac
F
chmod +x "$D/fake-systemctl"
printf '# lista de teste\na.timer\nb.timer\n' > "$D/timers.txt"
printf '#!/usr/bin/env bash\necho "$1" > "%s/aviso-chamado"\n' "$D" > "$D/aviso-cmd"; chmod +x "$D/aviso-cmd"
export LIMITE_PASSA="PATH HOME CUDA_VISIBLE_DEVICES"
export CMP_MODO_TESTE=1
unset CMP_RELOGIO_ARQ CMP_FAKE_RELOGIO
roda() { # nome args...; define RC
  local n=$1; shift
  export FAKE_DIR=$D/$n CMP_FAKE_LOG=$D/elog-$n; : > "$CMP_FAKE_LOG"
  mkdir -p "$FAKE_DIR"
  echo active > "$FAKE_DIR/a.timer"; echo active > "$FAKE_DIR/b.timer"; echo inactive > "$FAKE_DIR/a.service"; echo inactive > "$FAKE_DIR/b.service"; : > "$FAKE_DIR/log"
  SYSTEMCTL="$D/fake-systemctl" PENDENTE="$D/pend-$n" JANELA_PASSO_ESPERA=0.1 XDG_RUNTIME_DIR=/run/user/fake JANELA_TIMERS="$D/timers.txt" CMP_AVISO_CMD="$D/aviso-cmd" \
    "$J" --modelos "${MOD-1b}" --bin "$D/pp" --saida "$D/se-$n" --limite "${LIM:-1200}" "$@" > "$D/oute-$n" 2>&1
  RC=$?
}
roda sem-portao; confere "sem portão: recusa 2, nada rodou, timers intocados" "[ $RC = 2 ] && [ ! -s $D/elog-sem-portao ] && ! grep -q '^stop' $D/sem-portao/log && [ ! -e $D/se-sem-portao ]"
export PRIMEIRA_PALAVRA_AUTORIZADA=1
roda sem-uuid; confere "sem UUID: recusa 2" "[ $RC = 2 ] && [ ! -s $D/elog-sem-uuid ]"
for u in GPU-00000000-0000-0000-0000-00000000f090 GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd; do
  export CUDA_VISIBLE_DEVICES=$u; roda uuid-x; confere "UUID de outra GPU ($u): recusa 2, timers intocados" "[ $RC = 2 ] && [ ! -s $D/elog-uuid-x ] && ! grep -q '^stop' $D/uuid-x/log"
done
export CUDA_VISIBLE_DEVICES=$UUID
printf '300.00 W\n' > "$D/pot-seq"; roda pot-300; confere "power.limit 300 W no preflight: recusa 2, nada tocado" "[ $RC = 2 ] && grep -q \"'300.00 W'\" $D/oute-pot-300 && [ ! -s $D/elog-pot-300 ] && ! grep -q '^stop' $D/pot-300/log && [ ! -e $D/se-pot-300 ]"
: > "$D/pot-seq"
for caso in "cargo test --lib foo" "/usr/bin/cargo +nightly -q test" "cargo-nextest run" "/r/target/debug/deps/moe_rt-0123456789abcdef --nocapture" "/b/primeira-palavra --gera --celula 1b:/m" "llama-server -m x.gguf"; do
  printf '4242 %s\n' "$caso" > "$D/proc-x"
  PROC_LISTA=$D/proc-x roda cargo-x; confere "processo em execução '$caso': recusa 2 antes de timers, GPU e pasta" "[ $RC = 2 ] && grep -q 'processo que pode usar a 3080 Ti' $D/oute-cargo-x && [ ! -s $D/elog-cargo-x ] && ! grep -q '^stop' $D/cargo-x/log && [ ! -e $D/se-cargo-x ]"
done
printf '4242 cargo build --release --bin primeira-palavra\n4343 vim cargo.toml\n' > "$D/proc-x"
PROC_LISTA=$D/proc-x roda build-ok; confere "cargo build e editor NÃO recusam" "[ $RC = 0 ] && ! grep -q 'processo que pode usar' $D/oute-build-ok"
mkdir -p "$D/se-ocupada"; echo x > "$D/se-ocupada/a"
roda ocupada; confere "--saida não vazia: recusa 2" "[ $RC = 2 ] && [ ! -s $D/elog-ocupada ]"
printf '# gpu pid type sm mem enc dec jpg ofa fb ccpm command\n    0 1352017   C+G      -      -      -      -      -      -    997      0    voxtype\n' > "$D/pmon-cg.txt"
FAKE_PMON_ARQ=$D/pmon-cg.txt roda cg-aberto; confere "GPU ocupada (voxtype C+G): recusa 2 antes de timers, listando o PID" "[ $RC = 2 ] && grep -q 'FECHADOS' $D/oute-cg-aberto && grep -q '1352017 (voxtype)' $D/oute-cg-aberto && ! grep -q '^stop' $D/cg-aberto/log"
roda limite-alto --limite 1201; confere "limite acima de 1200: recusa 2" "[ $RC = 2 ] && [ ! -s $D/elog-limite-alto ]"
# bloco obrigatório: 1B = 60 + 9 x 60 + 180 = 780 s
LIM=700 roda nao-cabe
confere "bloco obrigatório não cabe (780 s contra 700): recusa 2, janela NÃO COMEÇA, 'SEM PROVA da unidade inteira', timers e células intocados" "[ $RC = 2 ] && grep -q 'SEM PROVA da unidade inteira' $D/se-nao-cabe/decisao.txt && grep -q 'NÃO COMEÇOU' $D/se-nao-cabe/decisao.txt && [ ! -s $D/elog-nao-cabe ] && ! grep -q '^stop' $D/nao-cabe/log"
MOD=3b LIM=1000 roda nao-cabe-3b
confere "3B: 60 + 9 x 90 + 180 = 1050 contra 1000: não começa" "[ $RC = 2 ] && grep -q 'necessário (tetos): 1050 s' $D/se-nao-cabe-3b/decisao.txt"
LIM=780 roda cabe-justo; confere "1B: bloco cabe justo (780 s): a janela começa" "[ $RC = 0 ] && [ -s $D/se-cabe-justo/decisao.txt ]"
MOD="" roda sem-modelos; confere "envelope sem --modelos: recusa 2, pasta não criada" "[ $RC = 2 ] && [ ! -e $D/se-sem-modelos ]"
for mm in 5b ambos 1b,3b; do MOD=$mm roda mod-ruim; confere "envelope com --modelos '$mm': recusa 2" "[ $RC = 2 ] && [ ! -e $D/se-mod-ruim ]"; done
unset MOD
real() { local n=$1; shift; env -i PATH="$PATH" HOME="$HOME" PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID "$@" "$J" --modelos 1b --bin "$D/pp" --saida "$D/se-real-$n" --limite 60 > "$D/oute-real-$n" 2>&1; RC=$?; }
for v in LASTRO_PERFIL_BLOCO LASTRO_PERFIL_F2 LASTRO_PERFIL_F2B F2B_TETO_FORCADO F2B_EXTRAI_PY F2_EXTRAI_PY F2_ANTERIOR_DIR CMP_FRACAO_SLOTS CMP_CAMADAS_CPU CRITERIO_F2B PLANO_F2B SEM_SHA_FIXO LASTRO_SH_CMP EXTRAI_PY NVIDIA_SMI MODELO_1B CMP_FOLGA PROC_LISTA CMP_FAKE_RECUSA JANELA_SH JANELA_TIMERS LIMITE_SH CMP_EST_QUENTE_1B; do
  real "v-$v" "$v=1"
  confere "modo real com $v no ambiente: recusa 2 citando a variável, nada criado" "[ $RC = 2 ] && grep -q 'variável só de teste presente no modo real: $v' $D/oute-real-v-$v && [ ! -e $D/se-real-v-$v ]"
done
LIM=1200 roda ok
confere "ok: sai 0, decisao.txt, 12 processos, timers parados e restaurados, resultado do extrator gravado" "[ $RC = 0 ] && [ -s $D/se-ok/decisao.txt ] && [ \$(ls $D/se-ok/celulas | grep -c '^ctrl-') = 9 ] && [ -s $D/se-ok/f2b-host-resultado.json ] && grep -q '^stop' $D/ok/log && grep -q '^start' $D/ok/log"
confere "ok: aviso no fim da janela e listas systemd ao lado da pasta" "grep -q 'terminou (código 0)' $D/se-ok.aviso.txt && grep -q 'terminou (código 0)' $D/aviso-chamado && [ -s $D/se-ok.unidades-antes.txt ]"
# portão do sha do binário (modo real, antes de qualquer envelope)
env -i PATH="$PATH" HOME="$HOME" PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID "$J" --modelos 1b --bin "$D/pp" --saida "$D/se-bin-errado" --limite 1200 > "$D/oute-bin-errado" 2>&1
confere "binário falso no modo real: recusa 2 por sha do FIXO, nada criado" "[ $? = 2 ] && grep -q 'binário com sha diferente do FIXO' $D/oute-bin-errado && [ ! -e $D/se-bin-errado ]"
# --verifica-fixos: ambiente limpo contra os arquivos REAIS
env -i PATH="$PATH" HOME="$HOME" "$J" --verifica-fixos > "$D/fixos.txt" 2>&1; rcf=$?
nok=$(grep -c ' OK$' "$D/fixos.txt"); nfal=$(grep -c ' FALHA$' "$D/fixos.txt"); naus=$(grep -c ' AUSENTE ' "$D/fixos.txt")
echo "congelados do FIXO: $nok OK, $nfal FALHA, $naus AUSENTE (código $rcf)"
confere "--verifica-fixos: código 0, nenhum FALHA/AUSENTE, itens essenciais presentes" "[ $rcf = 0 ] && [ $nfal = 0 ] && [ $naus = 0 ] && [ $nok -ge 20 ] && grep -q '^binario-f2b .* OK\$' $D/fixos.txt && grep -q '^criterio-f2b .* OK\$' $D/fixos.txt && grep -q '^script-f2b-extrai .* OK\$' $D/fixos.txt"
confere "--verifica-fixos: sha do binário é 22d0bbe0..." "grep -q '^binario-f2b 22d0bbe048d8c7407f73817b9879f5bc695604bc1bc519e332c9287c193adacf ' $D/fixos.txt"
echo alterado > "$D/crit-alterado.md"
env -i PATH="$PATH" HOME="$HOME" CRITERIO_F2B="$D/crit-alterado.md" "$J" --verifica-fixos > "$D/fixos-ruim.txt" 2>&1; rcr=$?
confere "--verifica-fixos com critério alterado: FALHA e código 1" "[ $rcr = 1 ] && grep -q '^criterio-f2b .* FALHA\$' $D/fixos-ruim.txt"
env -i PATH="$PATH" HOME="$HOME" F2B_EXTRAI_PY="$D/nao-existe.py" "$J" --verifica-fixos > "$D/fixos-aus.txt" 2>&1; rca=$?
confere "--verifica-fixos com extrator ausente: AUSENTE e código 1" "[ $rca = 1 ] && grep -q '^script-f2b-extrai .* AUSENTE' $D/fixos-aus.txt"
# rodada 154, P1: o verificador do FIXO no modo externo falha FECHADO, capturando o código de saída dele (sem pipe que o mascare), antes de envelope, monitor, timers ou GPU.
# Ambiente do FIXO real (sem os caminhos falsos); o binário é a cópia congelada real (--bin), que aqui só roda os subcomandos de planejamento; LIM=10 impede qualquer janela de começar.
gate() { # nome [VAR=valor...]; define RC e escreve em $D/oute-NOME
  local n=$1; shift
  ( unset TEXTOS_5C REF_5C IDS_CMP CRITERIO_CMP CRITERIO_D1 CRITERIO_D2 CRITERIO_F2B PLANO_F2B MODELO_1B MODELO_3B LASTRO_SH_CMP
    export "$@"; export SEM_SHA_FIXO=0
    export FAKE_DIR=$D/$n CMP_FAKE_LOG=$D/elog-$n; : > "$CMP_FAKE_LOG"; mkdir -p "$FAKE_DIR"; echo active > "$FAKE_DIR/a.timer"; echo active > "$FAKE_DIR/b.timer"; : > "$FAKE_DIR/log"
    SYSTEMCTL="$D/fake-systemctl" PENDENTE="$D/pend-$n" JANELA_TIMERS="$D/timers.txt" CMP_AVISO_CMD="$D/aviso-cmd" \
      "$J" --modelos 1b --bin "$PLAN_BIN" --saida "$D/se-$n" --limite 10 > "$D/oute-$n" 2>&1
    echo $? > "$D/rc-$n" )
  RC=$(cat "$D/rc-$n")
}
gate fixo-diverge "CRITERIO_F2B=$AQUI/plano-f2-nivel-b-pre-roteador-20261111.md"
confere "FIXO com arquivo EXISTENTE de hash divergente: recusa 2 pelo verificador, antes de envelope/timers/GPU (nada rodou, nenhum stop, pasta não criada)" "[ $RC = 2 ] && grep -q 'verifica-fixos falhou (código 1)' $D/oute-fixo-diverge && grep -q 'criterio-f2b' $D/oute-fixo-diverge && [ ! -s $D/elog-fixo-diverge ] && ! grep -q '^stop' $D/fixo-diverge/log && [ ! -e $D/se-fixo-diverge ]"
gate fixo-ausente "IDS_CMP=$D/nao-existe.json"
confere "FIXO com arquivo AUSENTE: recusa 2 pelo verificador, antes de tudo" "[ $RC = 2 ] && grep -q 'verifica-fixos falhou (código 1)' $D/oute-fixo-ausente && grep -q 'AUSENTE' $D/oute-fixo-ausente && ! grep -q '^stop' $D/fixo-ausente/log && [ ! -e $D/se-fixo-ausente ]"
gate fixo-ok "XPTO=1"
confere "FIXO íntegro: passa do verificador (recusa só adiante, pelo bloco obrigatório de 10 s), sem a mensagem do verificador e sem timers tocados" "[ $RC = 2 ] && ! grep -q 'verifica-fixos falhou' $D/oute-fixo-ok && grep -q 'NÃO COMEÇOU' $D/se-fixo-ok/decisao.txt && ! grep -q '^stop' $D/fixo-ok/log"
# rodada 154, P3-4/P3-5: lista vazia recusa; a tabela de tetos do roteiro (impressa em plano.txt) bate com a do Rust (teto_processo_s) nos dois modelos
"$PLAN_BIN" --planeja-celula --decorrido 1 --restantes "" > "$D/vazio.out" 2>&1; rcv=$?
confere "--planeja-celula com --restantes vazio: código 2, nunca 'cinco'" "[ $rcv = 2 ] && ! grep -q cinco $D/vazio.out"
for mm in 1b 3b; do
  rust=$("$PLAN_BIN" --planeja-tetos --modelo $mm); pl=$D/si-A1/plano.txt; [ $mm = 3b ] && pl=$D/si-A/plano.txt
  bash_=$(sed -n 's/^tetos de processo: smoke \([0-9]*\) s; médio \([0-9]*\) s; curto \([0-9]*\) s; longo \([0-9]*\) s;.*/\1 \2 \3 \4/p' "$pl")
  lit="60 60 60 75"; [ $mm = 3b ] && lit="60 90 90 120"
  confere "tabela de tetos $mm: Rust ($rust), roteiro ($bash_) e literal do critério ($lit) iguais" "[ \"$rust\" = \"$bash_\" ] && [ \"$rust\" = \"$lit\" ]"
done
if [ "$falhas" = 0 ]; then echo "TODOS OS TESTES PASSARAM"; exit 0; else echo "$falhas FALHA(S)"; exit 1; fi
