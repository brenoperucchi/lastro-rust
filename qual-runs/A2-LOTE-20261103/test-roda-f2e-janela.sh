#!/usr/bin/env bash
# Teste SEM GPU do roda-f2e-janela.sh: só recusas e consistência estática (não roda célula, não abre janela). mktemp -d; nada é removido.
set -u
unset E4B_GPU_AUTORIZADO E4B_MEDIDA_AUTORIZADA MOE_RT_REQUIRE_GPU PRIMEIRA_PALAVRA_AUTORIZADA LASTRO_PERFIL_BLOCO LASTRO_PERFIL_F2 LASTRO_PERFIL_F2B
export CUDA_VISIBLE_DEVICES=
AQUI=$(cd "$(dirname "$0")" && pwd -P)
J=$AQUI/roda-f2e-janela.sh
# TMPDIR fora do /tmp (regra do Breno): vazio ou sob /tmp vira ~/.cache/lastro-testes (o padrão dos testes deste projeto), nunca /tmp.
TMPBASE=${TMPDIR:-}; case "$TMPBASE" in ""|/tmp|/tmp/*) TMPBASE=$HOME/.cache/lastro-testes;; esac
mkdir -p "$TMPBASE" || { echo "não foi possível criar $TMPBASE"; exit 2; }; export TMPDIR=$TMPBASE
D=$(mktemp -d -p "$TMPBASE" f2e-janela-teste.XXXXXX) || exit 1
echo "pasta de teste: $D (apagada ao fim se tudo passar; --guarda mantém)"
falhas=0
confere() { if eval "$2"; then echo "ok: $1"; else echo "FALHOU: $1"; falhas=$((falhas+1)); fi; }
"$J" --verifica-fixos > "$D/vf.txt" 2>&1; rc=$?
# DEPOIS do congelamento da F2E: os shas do critério, do plano da O3, da decisão, dos dois binários e dos dois extratores estão preenchidos; --verifica-fixos sai 0 e nada pode estar PENDENTE, FALHA ou AUSENTE.
confere "--verifica-fixos depois do congelamento sai 0" '[ $rc = 0 ]'
confere "--verifica-fixos: nenhum PENDENTE, FALHA nem AUSENTE, e os 7 itens congelados OK" '! grep -qE " (PENDENTE|FALHA|AUSENTE)( |$)" "$D/vf.txt" && [ "$(grep -E "^(binario-base|binario-cand|criterio-f2e|decisao-f2e|plano-o3|script-f2b-extrai-v3|script-f2e-extrai) .* OK$" "$D/vf.txt" | wc -l)" = 7 ]'
"$J" > "$D/sa.txt" 2>&1; rc=$?
confere "sem argumentos sai 2" '[ $rc = 2 ]'
"$J" --modelos 1b --saida "$D/s" > "$D/pa.txt" 2>&1; rc=$?
confere "sem PRIMEIRA_PALAVRA_AUTORIZADA o portão recusa" 'grep -q "portão fechado" "$D/pa.txt" && [ $rc != 0 ]'
"$J" --modelos 7b --saida "$D/s" > "$D/m.txt" 2>&1; rc=$?
confere "modelo desconhecido recusado" '[ $rc != 0 ]'
confere "a pasta de saída não foi criada nas recusas" '[ ! -e "$D/s" ]'
# a ordem do roteiro é a que o extrator exige
ordem_sh=$(grep -m1 '^ORDEM_CTRL=' "$J" | sed 's/.*(\(.*\))/\1/')
ordem_py=$(python3 -I -c "
import importlib.util
s=importlib.util.spec_from_file_location('c','$AQUI/f2e-extrai.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
print(' '.join(m.ORDEM_CONTROLE), m.quadrado_latino_ok(m.ORDEM_CONTROLE))")
confere "ORDEM_CTRL do roteiro igual à do extrator e quadrado latino" '[ "$ordem_sh True" = "$ordem_py" ]'
confere "roteiro tem teto de 1200 s por janela" 'grep -q 1200 "$J"'
confere "roteiro usa --repete 3" 'grep -q -- "--repete 3" "$J"'

# ============================================================================================================================================
# Parte 2: modo --interno com CÉLULA FALSA (relógio virtual, nvidia-smi falso, binários falsos). Nenhuma GPU, nenhuma janela real.
# O binário candidato falso só delega os subcomandos --planeja-* (CPU pura) ao binário congelado real; o base falso não faz nada.
PLAN_BIN=$HOME/lastro-f2e/binarios/primeira-palavra-f2e-cand-1e809fd9
UUID=GPU-12345678-1234-1234-1234-123456789abc
printf "%s\n" "$UUID" > "$D/gpu-local.uuids"; export F2E_GPU_ARQ="$D/gpu-local.uuids"
if [ ! -x "$PLAN_BIN" ]; then echo "FALHOU: binário congelado ausente ($PLAN_BIN): preciso dos subcomandos de planejamento"; exit 1; fi
mkdir -p "$D/m/1b" "$D/m/3b" "$D/textos" "$D/refs"
for m in 1b 3b; do echo cfg > "$D/m/$m/config.json"; echo tok > "$D/m/$m/tokenizer.json"; done
echo st > "$D/m/1b/model.safetensors"; echo s1 > "$D/m/3b/model-00001-of-00002.safetensors"; echo s2 > "$D/m/3b/model-00002-of-00002.safetensors"; echo ix > "$D/m/3b/model.safetensors.index.json"
echo c > "$D/textos/texto-curto.txt"; echo m > "$D/textos/texto-medio.txt"; echo l > "$D/textos/texto-longo.txt"; echo '{"textos": []}' > "$D/textos/textos.json"
python3 -I - "$D" <<'PY'
import json, sys
d = sys.argv[1]
for m in ("1b", "3b"):
    json.dump({"casos": [{"nome": "curto", "ids": [1, 2], "gerados": list(range(64)), "gaps": [5.0] * 64}, {"nome": "medio", "ids": [1, 2, 3], "gerados": list(range(64)), "gaps": [5.0] * 64}]}, open("%s/refs/ref5c-%s.json" % (d, m), "w"))
json.dump({"modelos": {"1b": {"curto": {"ids": [1, 2]}, "medio": {"ids": [1, 2, 3]}}, "3b": {"curto": {"ids": [1, 2]}, "medio": {"ids": [1, 2, 3]}}}}, open("%s/ids.json" % d, "w"))
PY
for f in crit crit-d1 crit-d2 crit-f2e plano decisao; do echo "$f" > "$D/$f.md"; done
printf '#!/usr/bin/env bash\ncase "${1:-}" in --planeja-*) exec "%s" "$@";; esac\nexit 0\n' "$PLAN_BIN" > "$D/pp-cand"; chmod +x "$D/pp-cand"
printf '#!/usr/bin/env bash\n# binario base falso (base+gancho)\nexit 0\n' > "$D/pp-base"; chmod +x "$D/pp-base"
: > "$D/proc-vazia"
cat > "$D/smi" <<'F'
#!/usr/bin/env bash
case $* in
  *"--query-gpu=power.limit --format"*) f=${0%/*}/pot-seq; if [ -s "$f" ]; then l=$(head -n 1 "$f"); tail -n +2 "$f" > "$f.t"; mv "$f.t" "$f"; [ "$l" = FALHA ] && exit 1; echo "$l"; else echo "600.00 W"; fi;;
  *"-i GPU-12345678-1234-1234-1234-123456789abc --query-gpu=power.limit,pcie.link.width.current,pcie.link.gen.max"*) echo "600.00 W, 16, 4";;
  *query-gpu*) echo "uuid, memory.used [MiB], utilization.gpu [%]"; echo "GPU-falsa, 1 MiB, 0 %";;
  *) echo;;
esac
F
chmod +x "$D/smi"
# célula falsa: registra chave, n, binário e variáveis; avança o relógio virtual em (n+1)*CUSTO s; grava os arquivos que o extrator lê
cat > "$D/lastro-falso.sh" <<'F'
#!/usr/bin/env bash
modelo=; texto=; n=; saida=; bin=
while [ $# -gt 0 ]; do case $1 in --modelo) modelo=$2;; --texto) texto=$2;; --n) n=$2;; --saida) saida=$2;; --bin) bin=$2;; esac; shift; done
chave=$(basename "$saida")
echo "$chave n=$n texto=$texto BIN=$bin FRAC=${CMP_FRACAO_SLOTS:-} PB=${LASTRO_PERFIL_BLOCO-unset} F2=${LASTRO_PERFIL_F2-unset} F2B=${LASTRO_PERFIL_F2B-unset} GANCHO=${LASTRO_GANCHO_LOGITS-unset}" >> "$CMP_FAKE_LOG"
[ "${CMP_FAKE_RECUSA:-}" = "$chave" ] && { echo recusa >&2; exit 2; }
custo=$(( (n + 1) * ${CMP_FAKE_CUSTO:-10} )); [ -z "${CMP_FAKE_RELOGIO:-}" ] || echo $(( $(cat "$CMP_FAKE_RELOGIO") + custo )) > "$CMP_FAKE_RELOGIO"
[ "${CMP_FAKE_DORME:-}" != "$chave" ] || sleep 5
mkdir -p "$saida/p1/saida"
lig=desligado
case $chave in *B2O3) lig=ligado;; esac
[ -z "${CMP_FAKE_FORCE_LIGADO:-}" ] || lig=ligado
printf 'perfil-bloco: %s\nperfil-f2: %s\nperfil-f2b: %s\n' "$lig" "$lig" "$lig" > "$saida/p1/stdout.txt"
if [ "${LASTRO_GANCHO_LOGITS-}" = 1 ]; then
  python3 -I - "$saida/p1/saida/gancho-logits-$modelo.json" "$modelo" "$n" "${CMP_FAKE_ULP:-}" "$chave" <<'PY'
import hashlib, json, sys
arq, modelo, n, ulp, chave = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
H = lambda x: hashlib.sha256(str(x).encode()).hexdigest()
layers = {"1b": 24, "3b": 32}[modelo]
peds = [{"indice": i, "aquecimento": i == 0, "ids": [1, 2, 3], "gerados": list(range(64)), "logits": [H(("l", i, t)) for t in range(64)], "kv": [[H(("k", i, l)), H(("v", i, l))] for l in range(layers)]} for i in range(n + 1)]
if ulp and chave.endswith(ulp):
    peds[2]["logits"][5] = H("um ulp")
json.dump({"tipo": "gancho-logits", "versao": 1, "variavel": "LASTRO_GANCHO_LOGITS", "modelo": modelo, "pedidos": peds}, open(arq, "w"))
PY
fi
rc=0; [ "${CMP_FAKE_RC:-}" != "$chave" ] || rc=1
echo $rc > "$saida/p1/rc.txt"
printf '{"lado": "lastro", "serie": "L100", "modelo": "%s", "texto": "%s", "n_medidas": %s, "rc": %s}\n' "$modelo" "$texto" "$n" "$rc" > "$saida/celula.json"
exit $rc
F
chmod +x "$D/lastro-falso.sh"
export NVIDIA_SMI=$D/smi LASTRO_SH_CMP=$D/lastro-falso.sh MODELO_1B=$D/m/1b MODELO_3B=$D/m/3b PROC_LISTA=$D/proc-vazia
export TEXTOS_5C=$D/textos REF_5C=$D/refs IDS_CMP=$D/ids.json CRITERIO_CMP=$D/crit.md CRITERIO_D1=$D/crit-d1.md CRITERIO_D2=$D/crit-d2.md CRITERIO_F2E=$D/crit-f2e.md PLANO_O3=$D/plano.md DECISAO_O3=$D/decisao.md
export SEM_SHA_FIXO=1 CMP_SEM_MEMORIA=1 CMP_MODO_TESTE=1
export F2E_SHA_GPU_ESPERADO=$(printf %s "$UUID" | sha256sum | cut -d' ' -f1)   # UUID sintetico; honrado so com CMP_MODO_TESTE=1
interno() { # nome limite [modelo=1b] [cand=$D/pp-cand]; define RC; relógio virtual zerado
  local n=$1 lim=$2 mod=${3:-1b} cand=${4:-$D/pp-cand}
  echo 0 > "$D/clock"; : > "$D/log-$n"
  PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID CMP_RELOGIO_ARQ=$D/clock CMP_FAKE_RELOGIO=$D/clock CMP_FAKE_LOG=$D/log-$n \
    "$J" --interno --modelos "$mod" --saida "$D/si-$n" --bin-base "$D/pp-base" --bin-cand "$cand" --limite "$lim" > "$D/outi-$n" 2>&1
  RC=$?
}
nproc() { grep -c '^[0-9][0-9] ' "$D/si-$1/ordem.txt"; }
echo "== Parte 2: modo interno com célula falsa de 9 processos"
LASTRO_PERFIL_BLOCO=1 LASTRO_PERFIL_F2=1 LASTRO_PERFIL_F2B=1 LASTRO_GANCHO_LOGITS=1 interno A 1200 1b
confere "A: sai 0, decisao.txt gravada, exatamente 9 processos, cada um com n=3 no texto médio" "[ $RC = 0 ] && [ -s $D/si-A/decisao.txt ] && [ \$(nproc A) = 9 ] && [ \$(grep -c '^ctrl-1b-L100-medio-..-[A-Za-z0-9]* n=3 texto=medio ' $D/log-A) = 9 ] && [ \$(wc -l < $D/log-A) = 9 ]"
confere "A: quadrado latino respeitado na ordem efetiva (ordem.txt): DO1 DO3 B2O3, B2O3 DO1 DO3, DO3 B2O3 DO1" "[ \"\$(grep -o 'braco=[A-Za-z0-9]*' $D/si-A/ordem.txt | sed 's/braco=//' | tr '\n' ' ')\" = 'DO1 DO3 B2O3 B2O3 DO1 DO3 DO3 B2O3 DO1 ' ]"
confere "A: cada braço uma vez por linha e por coluna do quadrado (3 x 3), nomes das pastas na sequência 01..09" "[ \"\$(sed -n 1,3p $D/si-A/ordem.txt | grep -o 'braco=[A-Za-z0-9]*' | sort | tr '\n' ' ')\" = 'braco=B2O3 braco=DO1 braco=DO3 ' ] && [ \"\$(sed -n 4,6p $D/si-A/ordem.txt | grep -o 'braco=[A-Za-z0-9]*' | sort | tr '\n' ' ')\" = 'braco=B2O3 braco=DO1 braco=DO3 ' ] && [ \"\$(sed -n 7,9p $D/si-A/ordem.txt | grep -o 'braco=[A-Za-z0-9]*' | sort | tr '\n' ' ')\" = 'braco=B2O3 braco=DO1 braco=DO3 ' ] && [ \"\$(grep -o 'medio-0[0-9]' $D/si-A/ordem.txt | tr -d '\n')\" = medio-01medio-02medio-03medio-04medio-05medio-06medio-07medio-08medio-09 ]"
confere "A: binário por braço: DO1 roda o base+gancho; DO3 e B2O3 rodam o candidato" "[ \$(grep -c 'DO1 n=3 .*BIN=$D/pp-base ' $D/log-A) = 3 ] && [ \$(grep -c 'DO3 n=3 .*BIN=$D/pp-cand ' $D/log-A) = 3 ] && [ \$(grep -c 'B2O3 n=3 .*BIN=$D/pp-cand ' $D/log-A) = 3 ]"
confere "A: DO1 e DO3 sem NENHUMA variável do instrumento mesmo com as quatro no ambiente pai; B2O3 com as três ligadas; gancho de logits nunca" "[ \$(grep -cE '(DO1|DO3) n=3 .*PB=unset F2=unset F2B=unset GANCHO=unset' $D/log-A) = 6 ] && [ \$(grep -c 'B2O3 n=3 .*PB=1 F2=1 F2B=1 GANCHO=unset' $D/log-A) = 3 ] && [ \$(grep -c 'FRAC=1.0' $D/log-A) = 9 ]"
confere "A: prova-variavel.txt com 9 linhas, nenhuma INCOERENTE, sha do binário do braço registrado" "[ \$(wc -l < $D/si-A/prova-variavel.txt) = 9 ] && ! grep -q INCOERENTE $D/si-A/prova-variavel.txt && [ \$(grep -c \"braco=DO1 bin=\$(sha256sum $D/pp-base | cut -d' ' -f1) \" $D/si-A/prova-variavel.txt) = 3 ] && [ \$(grep -c \"braco=B2O3 bin=\$(sha256sum $D/pp-cand | cut -d' ' -f1) \" $D/si-A/prova-variavel.txt) = 3 ]"
confere "A: sem corte, nada NÃO RODOU, sem smoke/curto/longo (nenhuma célula fora dos 9), potência OK, extrator chamado (resultado gravado)" "grep -q '\"cortes\": \[\]' $D/si-A/orcamento.json && grep -q '\"nao_rodou\": \[\]' $D/si-A/orcamento.json && grep -q '^potencia 600.00_W 600.00_W OK' $D/si-A/integridade-fixos.txt && [ -s $D/si-A/f2e-resultado.json ] && [ \$(ls $D/si-A/celulas | wc -l) = 9 ] && ! grep -q 'smoke\|curto-quente\|longo-quente' $D/log-A"
confere "A: ambiente.txt traz os dois shas dos binários e o de contexto; plano.txt declara teto médio 60 s (1B)" "grep -q '^binario_base=' $D/si-A/ambiente.txt && grep -q '^binario_cand=' $D/si-A/ambiente.txt && grep -q 'médio 60 s; teto global 1200 s' $D/si-A/plano.txt"
# --coleta-dir (junção do HC1b na janela de medida): o roteiro encaminha a pasta ao extrator (--coleta); sem a opção o HC1b fica "pasta da coleta não fornecida"
confere "A: sem --coleta-dir, o resultado diz que a pasta da coleta não foi fornecida (comportamento que o abre-f2e deixa de usar)" "grep -q 'pasta da coleta não fornecida' $D/si-A/f2e-resultado.json"
mkdir -p "$D/colvazia"
echo 0 > "$D/clock"; : > "$D/log-C"
PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID CMP_RELOGIO_ARQ=$D/clock CMP_FAKE_RELOGIO=$D/clock CMP_FAKE_LOG=$D/log-C "$J" --interno --modelos 1b --saida "$D/si-C" --bin-base "$D/pp-base" --bin-cand "$D/pp-cand" --limite 1200 --coleta-dir "$D/colvazia" > "$D/outi-C" 2>&1; RC=$?
confere "C: com --coleta-dir o extrator recebe a pasta da coleta: o resultado deixa de dizer 'não fornecida' e aponta a coleta (vazia) como sem prova" "[ $RC = 0 ] && ! grep -q 'pasta da coleta não fornecida' $D/si-C/f2e-resultado.json && grep -q 'SEM PROVA' $D/si-C/f2e-resultado.json"
PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID "$J" --modelos 1b --saida "$D/si-C2" --coleta --coleta-dir "$D/colvazia" > "$D/outi-C2" 2>&1; RC=$?
confere "C2: --coleta-dir junto de --coleta: recusa 2, pasta de saída não criada" "[ $RC = 2 ] && grep -q 'só existe na janela de medida' $D/outi-C2 && [ ! -e $D/si-C2 ]"
PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID "$J" --modelos 1b --saida "$D/si-C3" --coleta-dir "$D/nao-existe" > "$D/outi-C3" 2>&1; RC=$?
confere "C3: --coleta-dir que não é pasta: recusa 2, pasta de saída não criada" "[ $RC = 2 ] && grep -q 'não é uma pasta' $D/outi-C3 && [ ! -e $D/si-C3 ]"
interno A3 1200 3b
confere "A3 (3B): plano com teto médio 90 s; 9 processos" "grep -q 'médio 90 s; teto global 1200 s' $D/si-A3/plano.txt && [ \$(nproc A3) = 9 ]"
# teto 1200 s: o bloco de 9 tetos + folga tem de caber no limite pedido; senão nenhum processo roda
interno B 700 1b
confere "B: limite 700 s (9 x 60 + 180 = 720 não cabe): nenhum processo rodou, os 9 NÃO RODOU com o prefixo ctrl-1b-L100-, nada cortado" "[ ! -s $D/log-B ] && [ \$(grep -o 'ctrl-1b-L100-medio-0[0-9]-[A-Za-z0-9]*: o controle de 9 processos não cabe' $D/si-B/orcamento.json | wc -l) = 9 ] && grep -q '\"cortes\": \[\]' $D/si-B/orcamento.json"
interno B2 720 1b
confere "B2: limite 720 s (cabe justo): os 9 rodam" "[ \$(nproc B2) = 9 ]"
interno B3 989 3b
confere "B3 (3B): 9 x 90 + 180 = 990 contra 989: não cabe, nenhum processo" "[ ! -s $D/log-B3 ]"
interno B4 990 3b
confere "B4 (3B): 990 contra 990 cabe" "[ \$(nproc B4) = 9 ]"
"$J" --modelos 1b --saida "$D/se-acima" --limite 1201 > "$D/oute-acima" 2>&1; rcl=$?
confere "envelope com --limite 1201: recusa 2 pelo teto de 1200 s, pasta não criada" "[ $rcl = 2 ] && grep -q 'limite fora de 1..1200' $D/oute-acima && [ ! -e $D/se-acima ]"
# controle nunca cortado: o planejamento é feito uma vez, no início; o relógio passa do limite durante o bloco (custo alto) e os 9 ainda rodam inteiros, cada um com n=3
CMP_FAKE_CUSTO=100 interno C 1200 1b
confere "C: custo alto (cada processo 400 s, relógio passa do limite): nenhum processo cortado nem pulado, todos com n=3, orçamento sem cortes" "[ \$(nproc C) = 9 ] && [ \$(grep -c ' n=3 ' $D/log-C) = 9 ] && grep -q '\"cortes\": \[\]' $D/si-C/orcamento.json && grep -q '\"cortado\": false' $D/si-C/orcamento.json && grep -q '\"nao_rodou\": \[\]' $D/si-C/orcamento.json"
# corte só em células NÃO iniciadas: um processo que falha encerra o controle; os não iniciados viram NÃO RODOU, o iniciado nunca é repetido
CMP_FAKE_RC=ctrl-1b-L100-medio-03-B2O3 interno D 1200 1b
confere "D: processo 03 com rc 1: só 3 rodaram, 04..09 NÃO RODOU como não iniciados ('controle encerrado'), sem repetição do 03" "[ \$(nproc D) = 3 ] && [ \$(grep -c '^ctrl-1b-L100-medio-03-B2O3' $D/log-D) = 1 ] && [ \$(grep -o 'ctrl-1b-L100-medio-0[4-9]-[A-Za-z0-9]*: controle encerrado' $D/si-D/orcamento.json | wc -l) = 6 ]"
CMP_FAKE_RECUSA=ctrl-1b-L100-medio-02-DO3 interno E 1200 1b
confere "E: processo 02 recusa o ambiente (código 2): janela interrompida, 03..09 NÃO RODOU, nada depois roda" "[ \$(nproc E) = 2 ] && grep -q 'recusou o ambiente' $D/si-E/orcamento-log.txt && [ \$(grep -c 'janela interrompida' $D/si-E/orcamento.json) -ge 1 ] && [ \$(wc -l < $D/log-E) = 2 ]"
F2E_TETO_FORCADO=1 CMP_FAKE_DORME=ctrl-1b-L100-medio-02-DO3 interno F 1200 1b
confere "F: processo acima do teto (1 s forçado, dorme 5 s): morre com 124, registrado em teto-excedido.txt, uma só tentativa, o resto NÃO RODOU" "grep -q '^ctrl-1b-L100-medio-02-DO3 teto=1s rc=124' $D/si-F/teto-excedido.txt && [ \$(grep -c '^ctrl-1b-L100-medio-02-DO3' $D/log-F) = 1 ] && [ \$(nproc F) = 2 ] && grep -q 'teto de processo excedido em ctrl-1b-L100-medio-02-DO3' $D/si-F/orcamento-log.txt"
CMP_FAKE_FORCE_LIGADO=1 interno G 1200 1b
confere "G: binário que liga o instrumento em DO1 e DO3: INCOERENTE nos 6 e a decisão conta" "[ \$(grep -c 'braco=D.*INCOERENTE\$' $D/si-G/prova-variavel.txt) = 6 ] && grep -q '6 processo(s) INCOERENTE' $D/si-G/decisao.txt"
printf '600.00 W\n300.00 W\n' > "$D/pot-seq"; interno H 1200 1b
confere "H (potência 300 W ao fim): manifesto FALHA e decisão QUEBRADA" "grep -qx 'potencia inicio=600.00_W fim=300.00_W FALHA' $D/si-H/integridade-fixos.txt && grep -q 'integridade: QUEBRADA' $D/si-H/decisao.txt"
: > "$D/pot-seq"
printf '#!/usr/bin/env bash\nexit 0\n' > "$D/pp-mudo"; chmod +x "$D/pp-mudo"
interno J 1200 1b "$D/pp-mudo"
confere "J (planejador sem resposta): falha fechada, os 9 NÃO RODOU, nenhum processo roda" "[ ! -s $D/log-J ] && grep -q 'planejador: erro' $D/si-J/orcamento.json"
printf '4242 cargo test --lib foo\n' > "$D/proc-cargo"
PROC_LISTA=$D/proc-cargo interno I 1200 1b
confere "I (cargo test em execução, modo interno): recusa 2, nada rodou" "[ $RC = 2 ] && [ ! -s $D/log-I ]"
for casom in "" "5b" "1b,3b"; do
  : > "$D/log-m"; if [ -z "$casom" ]; then args=(); else args=(--modelos "$casom"); fi
  PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID CMP_FAKE_LOG=$D/log-m "$J" --interno "${args[@]}" --saida "$D/si-m-$casom" --bin-base "$D/pp-base" --bin-cand "$D/pp-cand" --limite 100 > "$D/outi-m" 2>&1; rcm=$?
  confere "--modelos '${casom:-ausente}' no modo interno: recusa 2, nada rodou" "[ $rcm = 2 ] && [ ! -s $D/log-m ]"
done

echo "== Parte 3: execução de coleta (--coleta) com binários falsos"
coleta_run() { # nome limite [args extras...]; define RC
  local n=$1 lim=$2; shift 2
  echo 0 > "$D/clock"; : > "$D/log-$n"
  PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID CMP_RELOGIO_ARQ=$D/clock CMP_FAKE_RELOGIO=$D/clock CMP_FAKE_LOG=$D/log-$n \
    "$J" --interno --coleta --modelos 1b --saida "$D/sc-$n" --bin-base "$D/pp-base" --bin-cand "$D/pp-cand" --limite "$lim" "$@" > "$D/outc-$n" 2>&1
  RC=$?
}
congela_fixos() { # simula o FIXO congelado: o manifesto passa a trazer esperado = observado, estado OK (só em teste; o roteiro real compara com o FIXO)
  python3 -I - "$1/integridade-fixos.txt" <<'PY'
import sys
p = sys.argv[1]
out = []
for ln in open(p).read().splitlines():
    c = ln.split()
    if len(c) == 4 and c[3] == "NAO-CONFERIDO":
        c = [c[0], c[2], c[2], "OK"]
    out.append(" ".join(c))
open(p, "w").write("\n".join(out) + "\n")
PY
}
extrai_coleta() { python3 -I "$AQUI/f2e-extrai.py" --coleta-so --saida "$D/sc-$1" --ids "$D/ids.json" --refs "$D/refs" > "$D/outx-$1" 2>&1; }
LASTRO_PERFIL_BLOCO=1 LASTRO_PERFIL_F2=1 LASTRO_PERFIL_F2B=1 LASTRO_GANCHO_LOGITS=0 coleta_run K1 1200
confere "K1: coleta sai 0, exatamente 2 processos (original com o base, candidato com o candidato), n=3, mesmas entradas (texto médio)" "[ $RC = 0 ] && [ \$(wc -l < $D/log-K1) = 2 ] && grep -q '^coleta-1b-medio-original n=3 texto=medio BIN=$D/pp-base ' $D/log-K1 && grep -q '^coleta-1b-medio-candidato n=3 texto=medio BIN=$D/pp-cand ' $D/log-K1"
confere "K1: só LASTRO_GANCHO_LOGITS=1 ligada nos dois (instrumento desligado mesmo com as três no ambiente pai)" "[ \$(grep -c 'PB=unset F2=unset F2B=unset GANCHO=1' $D/log-K1) = 2 ]"
confere "K1: gancho-original e gancho-candidato gravados, manifesto com os shas dos binários, dos dois arquivos e das entradas" "[ -s $D/sc-K1/gancho-original/gancho-logits-1b.json ] && [ -s $D/sc-K1/gancho-candidato/gancho-logits-1b.json ] && grep -q \"^gancho_original_sha=\$(sha256sum $D/sc-K1/gancho-original/gancho-logits-1b.json | cut -d' ' -f1)\$\" $D/sc-K1/coleta-manifesto.txt && grep -q \"^binario_original=\$(sha256sum $D/pp-base | cut -d' ' -f1)\$\" $D/sc-K1/coleta-manifesto.txt && grep -q \"^binario_candidato=\$(sha256sum $D/pp-cand | cut -d' ' -f1)\$\" $D/sc-K1/coleta-manifesto.txt && grep -q '^ids_sha=[0-9a-f]\{64\}\$' $D/sc-K1/coleta-manifesto.txt && grep -q '^refs_sha=[0-9a-f]\{64\}\$' $D/sc-K1/coleta-manifesto.txt"
confere "K1: decisão declara execução própria de coleta, sem INCOERENTE" "grep -q 'EXECUÇÃO DE COLETA' $D/sc-K1/decisao.txt && ! grep -q INCOERENTE $D/sc-K1/prova-variavel.txt"
confere "K1: sem o FIXO congelado (SEM_SHA_FIXO), o extrator não dá prova: SEM PROVA" "grep -q 'HC1b SEM PROVA' $D/sc-K1/f2e-extrai.out"
congela_fixos "$D/sc-K1"; extrai_coleta K1
confere "K1 com FIXO simulado como congelado: HC1b dentro (hashes iguais, cobertura e procedência conferem)" "grep -q 'HC1b dentro' $D/outx-K1"
CMP_FAKE_ULP=candidato coleta_run K2 1200
congela_fixos "$D/sc-K2"; extrai_coleta K2
confere "K2: 1 ULP trocado num hash de logits do candidato (ids iguais): HC1b REGRESSÃO" "grep -q 'HC1b REGRESSÃO' $D/outx-K2"
CMP_FAKE_RC=coleta-1b-medio-candidato coleta_run K3 1200
confere "K3: candidato com rc 1: sem arquivo de gancho, NÃO RODOU, manifesto sem sha do candidato, extrator SEM PROVA" "grep -q 'coleta-1b-medio-candidato: sem arquivo de gancho' $D/sc-K3/orcamento.json && grep -q '^gancho_candidato_sha=\$' $D/sc-K3/coleta-manifesto.txt && grep -q 'HC1b SEM PROVA' $D/sc-K3/f2e-extrai.out"
CMP_FAKE_RECUSA=coleta-1b-medio-original coleta_run K4 1200
confere "K4: original recusa o ambiente (código 2): o candidato NÃO RODOU, nada depois" "[ \$(wc -l < $D/log-K4) = 1 ] && grep -q 'recusou o ambiente' $D/sc-K4/orcamento-log.txt"
printf '#!/usr/bin/env bash\necho "E4B=${E4B_GPU_AUTORIZADO:-} CUDA=${CUDA_VISIBLE_DEVICES:-}"\necho "TMPDIR=${TMPDIR:-}"\n[ -d "${TMPDIR:-}" ] && echo "TMPDIR existe"\nexit 0\n' > "$D/cargo-falso"; chmod +x "$D/cargo-falso"
# worktree candidato falso: repositório git de verdade com um commit, e o critério passa a trazer esse hash
mkdir -p "$D/wt/target-f2e/release/deps"; git -C "$D/wt" init -q; echo x > "$D/wt/a.txt"; git -C "$D/wt" add a.txt; git -C "$D/wt" -c user.name=t -c user.email=t@t.invalid commit -q -m teste
echo bin-de-teste > "$D/wt/target-f2e/release/deps/motor_reforma_gpu-0123456789abcdef"
echo bin-de-teste-passo > "$D/wt/target-f2e/release/deps/motor_passo_gpu-fedcba9876543210"
echo bin-de-teste-o3 > "$D/wt/target-f2e/release/deps/o3_kernels_gpu-0011223344556677"
WT_HASH=$(git -C "$D/wt" rev-parse HEAD)
cp "$D/crit-f2e.md" "$D/crit-f2e.orig"
printf -- '- hash do commit candidato: PREENCHER NO CONGELAMENTO\n' > "$D/crit-f2e.md"
F2E_CARGO_TESTE=$D/cargo-falso coleta_run K5a 1200 --teste-gpu --worktree-cand "$D/wt"
confere "K5a: critério com o hash do candidato ainda PREENCHER: o modo interno recusa o worktree (código 2), nada rodou" "[ $RC = 2 ] && grep -q 'ainda não foi preenchido' $D/outc-K5a && [ ! -s $D/log-K5a ]"
printf -- '- hash do commit candidato: %s\n' "0123456789012345678901234567890123456789" > "$D/crit-f2e.md"
F2E_CARGO_TESTE=$D/cargo-falso coleta_run K5b 1200 --teste-gpu --worktree-cand "$D/wt"
confere "K5b: HEAD do worktree diferente do commit candidato do critério: recusa 2, nada rodou" "[ $RC = 2 ] && grep -q 'diferente do commit candidato do critério' $D/outc-K5b && [ ! -s $D/log-K5b ]"
printf -- '- hash do commit candidato: %s\n' "$WT_HASH" > "$D/crit-f2e.md"
echo sujo > "$D/wt/a.txt"
F2E_CARGO_TESTE=$D/cargo-falso coleta_run K5c 1200 --teste-gpu --worktree-cand "$D/wt"
confere "K5c: árvore do worktree com modificação: recusa 2, nada rodou" "[ $RC = 2 ] && grep -q 'com modificações' $D/outc-K5c && [ ! -s $D/log-K5c ]"
git -C "$D/wt" checkout -q -- a.txt
# git status que FALHA (índice ilegível, HEAD correto, árvore na verdade limpa): falha fechada em vez de "árvore limpa"
GIT_INDEX_FILE=/dev/null F2E_CARGO_TESTE=$D/cargo-falso coleta_run K5d 1200 --teste-gpu --worktree-cand "$D/wt"
confere "K5d: HEAD correto mas git status falha (GIT_INDEX_FILE=/dev/null, stdout vazio, código 128): modo interno recusa 2, nenhum processo" "[ $RC = 2 ] && grep -q 'não foi possível verificar o estado do worktree' $D/outc-K5d && [ ! -s $D/log-K5d ]"
GIT_INDEX_FILE=/dev/null PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID "$J" --modelos 1b --coleta --teste-gpu --worktree-cand "$D/wt" --saida "$D/se-wt2" --bin-base "$D/pp-base" --bin-cand "$D/pp-cand" --limite 1200 > "$D/oute-wt2" 2>&1; rcw2=$?
confere "K5e: o mesmo com git status falhando no envelope: recusa 2 antes de timers e GPU, pasta de saída não criada" "[ $rcw2 = 2 ] && grep -q 'recusado antes de tocar em timers' $D/oute-wt2 && grep -q 'não foi possível verificar o estado' $D/oute-wt2 && [ ! -e $D/se-wt2 ]"
mkdir -p "$D/bin-git"; GIT_REAL=$(command -v git)
printf '#!/usr/bin/env bash\nfor a in "$@"; do [ "$a" = status ] && { echo " M a.txt"; echo "fatal: parcial" >&2; exit 1; }; done\nexec "%s" "$@"\n' "$GIT_REAL" > "$D/bin-git/git"; chmod +x "$D/bin-git/git"
PATH="$D/bin-git:$PATH" F2E_CARGO_TESTE=$D/cargo-falso coleta_run K5f 1200 --teste-gpu --worktree-cand "$D/wt"
confere "K5f: git status com saída parcial e código 1: recusa 2, nenhum processo" "[ $RC = 2 ] && grep -q 'não foi possível verificar o estado do worktree' $D/outc-K5f && [ ! -s $D/log-K5f ]"
F2E_CARGO_TESTE=$D/cargo-falso coleta_run K5g 1200 --teste-gpu --worktree-cand "$D/wt"
confere "K5g (controle): HEAD correto, árvore limpa e git status saudável: o modo interno segue e roda os 2 processos" "[ $RC = 0 ] && [ \$(wc -l < $D/log-K5g) = 2 ]"
F2E_CARGO_TESTE=$D/cargo-falso coleta_run K6 1200 --teste-gpu --worktree-cand "$D/wt"
confere "K6: --teste-gpu roda o teste depois dos dois processos com E4B_GPU_AUTORIZADO=1 no mesmo envelope, saída e rc gravados" "grep -q 'E4B=1 CUDA=$UUID' $D/sc-K6/teste-gpu.txt && [ \"\$(cat $D/sc-K6/teste-gpu.rc)\" = 0 ] && grep -q 'teste-gpu' $D/sc-K6/ordem.txt && [ \$(wc -l < $D/log-K6) = 2 ] && grep -q \"^teste_gpu_commit=$WT_HASH\$\" $D/sc-K6/coleta-manifesto.txt && grep -q \"^teste_gpu_exe_sha=\$(sha256sum $D/wt/target-f2e/release/deps/motor_passo_gpu-fedcba9876543210 | cut -d' ' -f1)\$\" $D/sc-K6/coleta-manifesto.txt && grep -q \"^teste_gpu_exe_sha_reforma=\$(sha256sum $D/wt/target-f2e/release/deps/motor_reforma_gpu-0123456789abcdef | cut -d' ' -f1)\$\" $D/sc-K6/coleta-manifesto.txt"
confere "K6b: o TMPDIR do teste de GPU existe durante o teste, não começa com /tmp, fica dentro de target-f2e do worktree candidato, é removido ao fim e o caminho está no manifesto" "TT=\$(sed -n 's/^TMPDIR=//p' $D/sc-K6/teste-gpu.txt); [ -n \"\$TT\" ] && case \"\$TT\" in /tmp/*) false;; *) true;; esac && case \"\$TT\" in $D/wt/target-f2e/tmp-teste-gpu.*) true;; *) false;; esac && grep -q 'TMPDIR existe' $D/sc-K6/teste-gpu.txt && [ ! -e \"\$TT\" ] && [ \"\$(sed -n 's/^teste_gpu_tmpdir=//p' $D/sc-K6/coleta-manifesto.txt)\" = \"\$TT\" ]"
ln -s "$D/wt" "$D/wtlink"
F2E_CARGO_TESTE=$D/cargo-falso coleta_run K6c 1200 --teste-gpu --worktree-cand "$D/wtlink"
confere "K6c: --worktree-cand por symlink é canonicalizado (realpath): o TMPDIR do teste de GPU fica sob o caminho real do worktree, não sob o symlink" "[ \$RC = 0 ] && case \"\$(sed -n 's/^TMPDIR=//p' $D/sc-K6c/teste-gpu.txt)\" in $D/wt/target-f2e/tmp-teste-gpu.*) true;; *) false;; esac"
F2E_CARGO_TESTE=$D/cargo-falso coleta_run K6d 1200 --teste-gpu --worktree-cand /tmp
confere "K6d: --worktree-cand sob /tmp: recusa 2 antes de qualquer mktemp, processo ou teste" "[ \$RC = 2 ] && grep -q 'está sob /tmp' $D/outc-K6d && [ ! -s $D/log-K6d ]"
printf '#!/usr/bin/env bash\nexit 1\n' > "$D/cargo-ruim"; chmod +x "$D/cargo-ruim"
F2E_CARGO_TESTE=$D/cargo-ruim coleta_run K7 1200 --teste-gpu --worktree-cand "$D/wt"
confere "K7: teste de GPU com rc 1 fica registrado como NÃO RODOU/falha, a coleta não é apagada" "grep -q 'teste de erro em GPU falhou' $D/sc-K7/orcamento.json && [ -s $D/sc-K7/gancho-original/gancho-logits-1b.json ]"
printf -- '- hash do commit candidato: %s\n' "0123456789012345678901234567890123456789" > "$D/crit-f2e.md"
PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID "$J" --modelos 1b --coleta --teste-gpu --worktree-cand "$D/wt" --saida "$D/se-wt" --bin-base "$D/pp-base" --bin-cand "$D/pp-cand" --limite 1200 > "$D/oute-wt" 2>&1; rcw=$?
confere "envelope: --worktree-cand com HEAD diferente do critério recusa 2 antes de timers e GPU, pasta não criada" "[ $rcw = 2 ] && grep -q 'recusado antes de tocar em timers' $D/oute-wt && [ ! -e $D/se-wt ]"
printf -- '- hash do commit candidato: %s\n' "$WT_HASH" > "$D/crit-f2e.md"
"$J" --modelos 1b --saida "$D/se-tg" --teste-gpu > "$D/oute-tg" 2>&1; rc1=$?
confere "--teste-gpu sem --coleta: recusa 2" "[ $rc1 = 2 ] && grep -q 'só existe junto de --coleta' $D/oute-tg"
"$J" --modelos 1b --saida "$D/se-tg2" --coleta --teste-gpu > "$D/oute-tg2" 2>&1; rc2=$?
confere "--teste-gpu sem --worktree-cand: recusa 2" "[ $rc2 = 2 ] && grep -q 'exige --worktree-cand' $D/oute-tg2"
env -i PATH="$PATH" HOME="$HOME" F2E_GPU_ARQ="$F2E_GPU_ARQ" PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID F2E_CARGO_TESTE=x "$J" --modelos 1b --coleta --saida "$D/se-real" --bin-base "$D/pp-base" --bin-cand "$D/pp-cand" --limite 600 > "$D/oute-real" 2>&1; rc3=$?
confere "coleta no modo real com F2E_CARGO_TESTE no ambiente: recusa 2, nada criado" "[ $rc3 = 2 ] && grep -q 'variável só de teste presente no modo real: F2E_CARGO_TESTE' $D/oute-real && [ ! -e $D/se-real ]"
if [ $falhas = 0 ]; then
  echo "TODOS OS TESTES PASSARAM"
  # D vem de mktemp -d, nunca foi reatribuída; só apaga se ainda é uma pasta (não symlink) com o prefixo do teste
  case "$D" in */f2e-janela-teste.*) if [ -d "$D" ] && [ ! -L "$D" ] && [ "${1:-}" != --guarda ]; then rm -rf -- "$D"; else echo "pasta de teste mantida: $D"; fi;; *) echo "pasta de teste mantida: $D";; esac
  exit 0
fi
echo "$falhas FALHA(S); pasta de teste mantida: $D"; exit 1
