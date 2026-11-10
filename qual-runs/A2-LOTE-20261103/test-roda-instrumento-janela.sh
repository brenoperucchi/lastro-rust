#!/usr/bin/env bash
# Teste do roda-instrumento-janela.sh SEM GPU: binário/célula FALSOS (relógio virtual), nvidia-smi falso, systemctl falso, modelos falsos. mktemp -d; este teste NÃO remove nada.
# Parte 1 (modo --interno): ordem D L L D D L, variável só nos L (nunca nos D, mesmo com a variável no ambiente pai), orçamento, smoke, controle abortado, potência, extrator.
# Parte 2 (envelope): recusas (cargo test, potência, GPU ocupada, pasta, modelos, variáveis de teste), bloco do par que não cabe, caminho feliz, --verifica-fixos.
set -u
unset E4B_GPU_AUTORIZADO E4B_MEDIDA_AUTORIZADA MOE_RT_REQUIRE_GPU PRIMEIRA_PALAVRA_AUTORIZADA LASTRO_PERFIL_BLOCO
export CUDA_VISIBLE_DEVICES=
AQUI=$(cd "$(dirname "$0")" && pwd -P)
J=$AQUI/roda-instrumento-janela.sh
UUID=GPU-b22d9465-8552-9f97-812e-380b590645f8
D=$(mktemp -d -p "${TMPDIR:-/tmp}" instr-janela-teste.XXXXXX) || exit 1
echo "pasta de teste: $D"
falhas=0
confere() { if eval "$2"; then echo "ok: $1"; else echo "FALHOU: $1"; falhas=$((falhas+1)); fi; }
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
for f in crit crit-d1 crit-d2 crit-instr plano; do echo "$f" > "$D/$f.md"; done
printf '#!/usr/bin/env bash\nexit 0\n' > "$D/pp"; chmod +x "$D/pp"
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
# célula e smoke falsos (roda-cmp-lastro.sh). O "binário" falso: imprime `perfil-bloco: ligado` só se LASTRO_PERFIL_BLOCO=1 (ou CMP_FAKE_FORCE_LIGADO), e grava perfil-bloco.json nesse caso.
cat > "$D/lastro-falso.sh" <<'F'
#!/usr/bin/env bash
if [ "$1" = --smoke ]; then
  shift; modelo=; saida=
  while [ $# -gt 0 ]; do case $1 in --modelo) modelo=$2;; --saida) saida=$2;; esac; shift; done
  echo "SMOKE $modelo FRAC=${CMP_FRACAO_SLOTS:-} CAMADAS=${CMP_CAMADAS_CPU:-} PB=${LASTRO_PERFIL_BLOCO-unset}" >> "$CMP_FAKE_LOG"
  [ "${FAKE_SMOKE_RC:-0}" != 2 ] || { echo recusa >&2; exit 2; }
  mkdir -p "$saida/p1/saida"; echo 0 > "$saida/rc.txt"; echo 0 > "$saida/p1/rc.txt"
  L=24; E=32; N=22; [ "$modelo" = 3b ] && { L=32; E=40; N=29; }
  if [ -n "${CMP_FRACAO_SLOTS:-}" ]; then
    echo "pp --gera --celula $modelo:/m --so-texto curto --fracao-slots 1.0 --saida x" > "$saida/cmdline.txt"
    echo "slots $((L * E)) de $((L * E))" > "$saida/p1/stderr.txt"; cpu=0; [ "${FAKE_SMOKE_FALHA:-}" != L100 ] || cpu=7
  else
    echo "pp --gera --celula $modelo:/m --so-texto curto --camadas-cpu $N --saida x" > "$saida/cmdline.txt"
    echo "camadas-cpu $N de $L; slots $(((L - N) * E)) de $((L * E))" > "$saida/p1/stderr.txt"; cpu=$((N * E * 152)); [ "${FAKE_SMOKE_FALHA:-}" != LC ] || cpu=$((cpu + 1))
  fi
  printf '1000500, 2000, 24000\n1001500, 2000, 24000\n1002500, 2000, 24000\n1003500, 2000, 24000\n' > "$saida/gpumem.csv"
  printf 'coletor_inicio_ms=999900\nbin_inicio_ms=1000000\nbin_fim_ms=1004000\ncoletor_fim_ms=1004100\n' > "$saida/smoke-tempos.txt"
  python3 -I - "$saida" "$modelo" "$L" "$E" "$cpu" <<'PY'
import json, os, sys
saida, m, L, E, cpu = sys.argv[1:6]
json.dump({"medidas": [{"descartada": False, "falha": None, "gerados": list(range(64)), "cont": {"needed": int(L) * int(E) * 152, "cpu": int(cpu), "hits": 100}}]}, open(os.path.join(saida, "p1", "saida", "medidas-%s.json" % m), "w"))
PY
  : > "$saida/p1/stdout.txt"; exit 0
fi
modelo=; texto=; estado=; n=; saida=
while [ $# -gt 0 ]; do case $1 in --modelo) modelo=$2;; --texto) texto=$2;; --estado) estado=$2;; --n) n=$2;; --saida) saida=$2;; esac; shift; done
chave=$(basename "$saida")
echo "$chave n=$n FRAC=${CMP_FRACAO_SLOTS:-} CAMADAS=${CMP_CAMADAS_CPU:-} PB=${LASTRO_PERFIL_BLOCO-unset}" >> "$CMP_FAKE_LOG"
[ "${CMP_FAKE_RECUSA:-}" = "$chave" ] && { echo recusa >&2; exit 2; }
custo=$(( (n + 1) * 10 )); [ -z "${CMP_FAKE_RELOGIO:-}" ] || echo $(( $(cat "$CMP_FAKE_RELOGIO") + custo )) > "$CMP_FAKE_RELOGIO"
mkdir -p "$saida/p1/saida"
if [ "${LASTRO_PERFIL_BLOCO-}" = 1 ] || [ -n "${CMP_FAKE_FORCE_LIGADO:-}" ]; then
  echo "perfil-bloco: ligado" > "$saida/p1/stdout.txt"; echo '{"tipo": "perfil-bloco", "versao": 2, "pedidos": []}' > "$saida/p1/saida/perfil-bloco.json"
else echo "perfil-bloco: desligado" > "$saida/p1/stdout.txt"; fi
rc=0; [ "${CMP_FAKE_RC:-}" != "$chave" ] || rc=1
echo $rc > "$saida/p1/rc.txt"
serie=L10; [ -z "${CMP_FRACAO_SLOTS:-}" ] || serie=L100; [ -z "${CMP_CAMADAS_CPU:-}" ] || serie=LC
printf '{"lado": "lastro", "serie": "%s", "modelo": "%s", "texto": "%s", "estado": "%s", "n_medidas": %s, "portao_ok": true, "portao_motivo": "", "rc": %s}\n' "$serie" "$modelo" "$texto" "$estado" "$n" "$rc" > "$saida/celula.json"
exit $rc
F
chmod +x "$D/lastro-falso.sh"
# estimativas: processo com n medidas custa (n+1)*10 s (n=3: 40; n=5: 60); portão, carga, smoke e folga zerados. Bloco do par = 12 x 40 = 480 s.
export CMP_EST_PORTAO=0 CMP_EST_CARGA_1B=0 CMP_EST_CARGA_3B=0 CMP_EST_QUENTE_1B=10 CMP_EST_QUENTE_3B=10 CMP_EST_SMOKE_1B=0 CMP_EST_SMOKE_3B=0
export NVIDIA_SMI=$D/smi LASTRO_SH_CMP=$D/lastro-falso.sh MODELO_1B=$D/m/1b MODELO_3B=$D/m/3b PROC_LISTA=$D/proc-vazia
export TEXTOS_5C=$D/textos REF_5C=$D/refs IDS_CMP=$D/ids.json CRITERIO_CMP=$D/crit.md CRITERIO_D1=$D/crit-d1.md CRITERIO_D2=$D/crit-d2.md CRITERIO_INSTR=$D/crit-instr.md PLANO_INSTR=$D/plano.md
export SEM_SHA_FIXO=1 CMP_SEM_MEMORIA=1 CMP_FOLGA=0 CMP_MODO_TESTE=1
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
# A0 (rodada 140): modo interno em modo real (sem CMP_MODO_TESTE) recusa variável de teste e binário fora do FIXO ANTES de criar saída, smoke ou célula
echo 0 > "$D/clock"
env -i PATH="$PATH" HOME="$HOME" SEM_SHA_FIXO=1 PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID "$J" --interno --modelos 1b --saida "$D/si-real1" --bin "$D/pp" --limite 100 > "$D/outi-real1" 2>&1
RC=$?
confere "interno + binário fora do FIXO + SEM_SHA_FIXO=1 sem CMP_MODO_TESTE: recusa 2, nada criado" "[ $RC = 2 ] && grep -q 'variável só de teste presente no modo real: SEM_SHA_FIXO' $D/outi-real1 && [ ! -e $D/si-real1 ]"
env -i PATH="$PATH" HOME="$HOME" PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID "$J" --interno --modelos 1b --saida "$D/si-real2" --bin "$D/pp" --limite 100 > "$D/outi-real2" 2>&1
RC=$?
confere "interno real sem SEM_SHA_FIXO e binário falso: recusa 2 por sha do FIXO, nada criado" "[ $RC = 2 ] && grep -q 'sha diferente do FIXO' $D/outi-real2 && [ ! -e $D/si-real2 ]"
BIN_REAL=/home/brenoperucchi/Devs/lastro-rust/.herdr/binarios/primeira-palavra-instr-ae14cbde
if [ -x "$BIN_REAL" ]; then
  cp "$BIN_REAL" "$D/copia-real"
  env -i PATH="$PATH" HOME="$HOME" PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID "$J" --interno --modelos 1b --saida /proc/nao-cria/si --bin "$D/copia-real" --limite 100 > "$D/outi-real3" 2>&1
  RC=$?
  confere "cópia congelada real passa o portão do modo real (a recusa vem só depois, ao criar a saída em /proc; sem GPU)" "[ $RC = 2 ] && ! grep -q 'sha diferente' $D/outi-real3 && ! grep -q 'variável só de teste' $D/outi-real3"
else echo "ok: (binário congelado ausente nesta máquina; teste da cópia real pulado)"; fi
# A. folgado, com a variável EXPORTADA no ambiente pai: os D continuam sem ela
LASTRO_PERFIL_BLOCO=1 interno A 100000
confere "A: sai 0, decisao.txt, 19 processos (12 de controle, 4 obrigatórias, 3 LC)" "[ $RC = 0 ] && [ -s $D/si-A/decisao.txt ] && [ \$(nproc A) = 19 ]"
confere "A: smoke L100 primeiro; depois controle L10 na ordem D L L D D L (01..06), depois controle L100" "[ \"\$(sed -n 1p $D/si-A/ordem.txt | cut -d' ' -f1)\" = smoke-L100 ] && [ \"\$(grep -o 'ctrl-1b-L10-medio-0[0-9]-[DL]' $D/si-A/ordem.txt | sed 's/.*-//' | tr -d '\n')\" = DLLDDL ] && [ \"\$(grep -o 'ctrl-1b-L100-medio-0[0-9]-[DL]' $D/si-A/ordem.txt | sed 's/.*-//' | tr -d '\n')\" = DLLDDL ] && [ \"\$(sed -n 2p $D/si-A/ordem.txt | cut -d' ' -f2)\" = ctrl-1b-L10-medio-01-D ]"
confere "A: braço D roda SEM a variável, mesmo com ela no ambiente pai (todas as linhas D do log com PB=unset)" "[ \$(grep -c 'ctrl-.*-D n=3 .*PB=unset' $D/log-A) = 6 ] && ! grep -E 'ctrl-.*-D n=.*PB=1' $D/log-A"
confere "A: braço L roda COM a variável (todas as 6 linhas L com PB=1) e n=3 nos controles" "[ \$(grep -c 'ctrl-.*-L n=3 .*PB=1' $D/log-A) = 6 ]"
confere "A: obrigatórias (4) e LC (3) com PB=1 e n=5; smokes sem a variável (PB=unset)" "[ \$(grep -c '^lastro-1b-.* n=5 .*PB=1' $D/log-A) = 7 ] && [ \$(grep -c '^SMOKE .*PB=unset' $D/log-A) = 2 ]"
confere "A: séries certas por processo (L10 sem variável de série; L100 FRAC=1.0; LC CAMADAS=22)" "grep -q '^ctrl-1b-L10-medio-01-D n=3 FRAC= CAMADAS= ' $D/log-A && grep -q '^ctrl-1b-L100-medio-02-L n=3 FRAC=1.0 CAMADAS= ' $D/log-A && grep -q '^lastro-1b-LC-medio-quente n=5 FRAC= CAMADAS=22 ' $D/log-A"
confere "A: prova-variavel.txt com 19 linhas, D com '-u LASTRO_PERFIL_BLOCO' e linha desligado, L com LASTRO_PERFIL_BLOCO=1 e linha ligado, nenhuma INCOERENTE" "[ \$(wc -l < $D/si-A/prova-variavel.txt) = 19 ] && grep -q 'ctrl-1b-L10-medio-01-D braco=D env=\".*-u LASTRO_PERFIL_BLOCO\" linha_do_binario=\"perfil-bloco: desligado\"\$' $D/si-A/prova-variavel.txt && grep -q 'ctrl-1b-L10-medio-02-L braco=L env=\".*LASTRO_PERFIL_BLOCO=1\" linha_do_binario=\"perfil-bloco: ligado\"\$' $D/si-A/prova-variavel.txt && ! grep -q INCOERENTE $D/si-A/prova-variavel.txt && grep -q '0 processo(s) INCOERENTE' $D/si-A/decisao.txt"
confere "A: nenhum perfil-bloco.json nos D; um em cada L; manifesto com potência OK; ambiente com início e fim" "[ -z \"\$(find $D/si-A/celulas -path '*-D/*' -name perfil-bloco.json)\" ] && [ \$(find $D/si-A/celulas -name perfil-bloco.json | wc -l) = 13 ] && grep -qx 'potencia 370.00_W 370.00_W OK' $D/si-A/integridade-fixos.txt && grep -qx 'potencia_inicio=370.00_W' $D/si-A/ambiente.txt && grep -qx 'potencia_fim=370.00_W' $D/si-A/ambiente.txt"
confere "A: orçamento.json com o modelo, controle-ordem.txt com 12 linhas e o extrator real rodou (resultado gravado, controle incompleto pelos dados falsos)" "grep -q '\"modelos\": \[\"1b\"\]' $D/si-A/orcamento.json && [ \$(wc -l < $D/si-A/controle-ordem.txt) = 12 ] && [ -s $D/si-A/instrumento-resultado.json ] && grep -q 'extrator:' $D/si-A/decisao.txt"
confere "A: decisao.txt imprime o sha256 do roteiro" "grep -q \"roteiro sha256 \$(sha256sum $J | cut -d' ' -f1)\" $D/si-A/decisao.txt"
# B. falha de isolamento: binário falso que liga mesmo em D -> INCOERENTE
CMP_FAKE_FORCE_LIGADO=1 interno B 100000
confere "B (variável vazando para um D): prova-variavel.txt marca INCOERENTE nos 6 D e a decisão conta" "[ \$(grep -c 'braco=D.*INCOERENTE\$' $D/si-B/prova-variavel.txt) = 6 ] && grep -q '6 processo(s) INCOERENTE' $D/si-B/decisao.txt"
# C. potência mudando: 370 no início, 300 ao fim -> integridade QUEBRADA, extrator RECUSA
printf '370.00 W\n300.00 W\n' > "$D/pot-seq"; interno C 100000
confere "C (370 no início, 300 ao fim): manifesto FALHA, decisão QUEBRADA, extrator recusa (código 2) sem classificar" "[ $RC = 0 ] && grep -qx 'potencia inicio=370.00_W fim=300.00_W FALHA' $D/si-C/integridade-fixos.txt && grep -q 'integridade: QUEBRADA' $D/si-C/decisao.txt && grep -q 'RECUSADO' $D/si-C/decisao.txt && grep -q '(código 2)' $D/si-C/decisao.txt && grep -q '\"recusado\": true' $D/si-C/instrumento-resultado.json"
printf '370.00 W\nFALHA\n' > "$D/pot-seq"; interno C2 100000
confere "C2 (leitura ausente ao fim): fim ilegivel, QUEBRADA, extrator recusa" "grep -qx 'potencia inicio=370.00_W fim=ilegivel FALHA' $D/si-C2/integridade-fixos.txt && grep -q 'RECUSADO' $D/si-C2/decisao.txt"
: > "$D/pot-seq"
# D. orçamento
interno E 1000
confere "E (limite 1000, nominal 900): nenhum corte, 19 processos" "[ \$(nproc E) = 19 ] && grep -q '\"cortes\": \[\]' $D/si-E/orcamento.json"
interno F 700
confere "F (limite 700): controle inteiro, corte 5->3 nas obrigatórias, só 1 opcional LC cabe, resto NÃO RODOU" "grep -q 'quente cai de 5 para 3' $D/si-F/orcamento.json && [ \$(grep -c '^ctrl-' $D/log-F) = 12 ] && [ \$(grep -c '^lastro-1b-L.*-quente n=3' $D/log-F) = 5 ] && grep -q 'NÃO RODOU: lastro-1b-LC-medio-quente' $D/si-F/orcamento-log.txt"
interno G 400
confere "G (limite 400, direto no interno): 10 processos de controle; os 2 últimos do L100 NÃO RODOU (prefixo ctrl-), nenhuma obrigatória" "[ \$(grep -c '^ctrl-' $D/log-G) = 10 ] && grep -q 'ctrl-1b-L100-medio-05-D' $D/si-G/orcamento.json && grep -q 'ctrl-1b-L100-medio-06-L' $D/si-G/orcamento.json && ! grep -q '^lastro-1b-L10-curto' $D/log-G"
# E. smoke falha
FAKE_SMOKE_FALHA=L100 interno H 100000
confere "H (smoke L100 falha): controle L100 e células L100 NÃO RODOU/ausentes; L10 e LC seguem; um só SMOKE L100, sem nova tentativa; decisão cita o smoke" "[ \$(grep -c 'FRAC=1.0' $D/log-H) = 1 ] && [ \$(grep -c '^ctrl-1b-L10-' $D/log-H) = 6 ] && [ \$(grep -c '^ctrl-1b-L100' $D/log-H) = 0 ] && [ \$(grep -c '^lastro-1b-L100' $D/log-H) = 0 ] && grep -q 'ctrl-1b-L100-medio-01-D' $D/si-H/orcamento.json && grep -q 'L100 NÃO SUSTENTA 100%' $D/si-H/decisao.txt && [ \$(grep -c '^lastro-1b-LC' $D/log-H) = 3 ]"
FAKE_SMOKE_FALHA=LC interno H2 100000
confere "H2 (smoke LC falha): só LC fica sem células; decisão cita o smoke de camadas-CPU" "[ \$(grep -c '^lastro-1b-LC' $D/log-H2) = 0 ] && [ \$(grep -c '^ctrl-' $D/log-H2) = 12 ] && grep -q 'SMOKE DE CAMADAS-CPU FALHOU' $D/si-H2/decisao.txt"
FAKE_SMOKE_RC=2 interno H3 100000
confere "H3 (smoke recusa o ambiente, código 2): nenhum processo, janela inteira NÃO RODOU, decisão gravada" "[ \$(nproc H3) = 0 ] && grep -q 'recusou o ambiente' $D/si-H3/orcamento-log.txt && [ -s $D/si-H3/decisao.txt ]"
# F. controle abortado
CMP_FAKE_RC=ctrl-1b-L10-medio-03-L interno I 100000
confere "I (processo de controle sai com código 1): controle L10 encerrado sem repetição (04..06 NÃO RODOU), controle L100 e obrigatórias seguem" "[ \$(grep -c '^ctrl-1b-L10-' $D/log-I) = 3 ] && grep -q 'ctrl-1b-L10-medio-04-D' $D/si-I/orcamento.json && grep -q 'ctrl-1b-L10-medio-06-L' $D/si-I/orcamento.json && [ \$(grep -c '^ctrl-1b-L100' $D/log-I) = 6 ] && [ \$(grep -c '^lastro-1b-L10-curto' $D/log-I) = 1 ]"
CMP_FAKE_RECUSA=ctrl-1b-L10-medio-02-L interno I2 100000
confere "I2 (processo de controle recusa o ambiente, código 2): janela interrompida, o resto NÃO RODOU" "[ \$(nproc I2) = 2 ] && grep -q 'recusou o ambiente' $D/si-I2/orcamento-log.txt && ! grep -q '^lastro-1b-L10-curto' $D/log-I2"
# G. cargo test no modo interno
printf '4242 cargo test --lib foo\n' > "$D/proc-cargo"
PROC_LISTA=$D/proc-cargo interno J 100000
confere "J (cargo test em execução, modo interno): recusa 2, nada rodou" "[ $RC = 2 ] && [ ! -s $D/log-J ] && grep -q 'cargo test' $D/outi-J"
for casom in "" "5b" "1b,3b" "ambos"; do
  : > "$D/log-m"; if [ -z "$casom" ]; then args=(); else args=(--modelos "$casom"); fi
  PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID CMP_FAKE_LOG=$D/log-m "$J" --interno "${args[@]}" --saida "$D/si-m-$casom" --bin "$D/pp" --limite 100 > "$D/outi-m" 2>&1; rcm=$?
  confere "--modelos '${casom:-ausente}' no modo interno: recusa 2, nada rodou" "[ $rcm = 2 ] && [ ! -s $D/log-m ]"
done
interno K 100000 3b
confere "K (3B): camadas 29, safetensors do 3B e não os do 1B" "grep -q 'lastro-3b-LC-medio-quente n=5 FRAC= CAMADAS=29 ' $D/log-K && grep -q '^safetensors-3b-2 ' $D/si-K/integridade-antes.txt && ! grep -q '^safetensors-1b' $D/si-K/integridade-antes.txt"

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
printf 'FALHA\n' > "$D/pot-seq"; roda pot-falha; confere "power.limit ilegível no preflight: recusa 2, nada tocado" "[ $RC = 2 ] && grep -q 'ilegível' $D/oute-pot-falha && ! grep -q '^stop' $D/pot-falha/log && [ ! -e $D/se-pot-falha ]"
: > "$D/pot-seq"
# cargo test e processos CUDA
for caso in "cargo test --lib foo" "/usr/bin/cargo +nightly -q test" "bash -c 'cargo nextest run'" "/r/target/debug/deps/moe_rt-0123456789abcdef --nocapture" "/b/primeira-palavra --gera --celula 1b:/m" "llama-server -m x.gguf"; do
  printf '4242 %s\n' "$caso" > "$D/proc-x"
  PROC_LISTA=$D/proc-x roda cargo-x; confere "processo em execução '$caso': recusa 2 antes de timers, GPU e pasta" "[ $RC = 2 ] && grep -q 'processo que pode usar a 3080 Ti' $D/oute-cargo-x && [ ! -s $D/elog-cargo-x ] && ! grep -q '^stop' $D/cargo-x/log && [ ! -e $D/se-cargo-x ]"
done
printf '4242 cargo build --release --bin primeira-palavra\n4343 vim cargo.toml\n' > "$D/proc-x"
PROC_LISTA=$D/proc-x LIM=1200 roda build-ok; confere "cargo build e editor NÃO recusam (a janela segue até o fim)" "[ $RC = 0 ] && ! grep -q 'processo que pode usar' $D/oute-build-ok"
mkdir -p "$D/se-ocupada"; echo x > "$D/se-ocupada/a"
roda ocupada; confere "--saida não vazia: recusa 2" "[ $RC = 2 ] && [ ! -s $D/elog-ocupada ]"
printf '# gpu pid type sm mem enc dec jpg ofa fb ccpm command\n    0 1352017   C+G      -      -      -      -      -      -    997      0    voxtype\n' > "$D/pmon-cg.txt"
FAKE_PMON_ARQ=$D/pmon-cg.txt roda cg-aberto; confere "GPU ocupada (voxtype C+G): recusa 2 antes de timers, listando o PID" "[ $RC = 2 ] && grep -q 'FECHADOS' $D/oute-cg-aberto && grep -q '1352017 (voxtype)' $D/oute-cg-aberto && ! grep -q '^stop' $D/cg-aberto/log && [ ! -e $D/se-cg-aberto ]"
roda limite-alto --limite 1201; confere "limite acima de 1200: recusa 2" "[ $RC = 2 ] && [ ! -s $D/elog-limite-alto ]"
# bloco do par que não cabe: 12 x 40 = 480 s necessários; limite 300 -> não começa, registrado
LIM=300 roda par-nao-cabe
confere "bloco do par não cabe (480 s contra 300): recusa 2, janela NÃO COMEÇA, decisao.txt 'SEM PROVA da unidade inteira', timers e células intocados" "[ $RC = 2 ] && grep -q 'SEM PROVA da unidade inteira' $D/se-par-nao-cabe/decisao.txt && grep -q 'NÃO COMEÇOU' $D/se-par-nao-cabe/decisao.txt && [ ! -s $D/elog-par-nao-cabe ] && ! grep -q '^stop' $D/par-nao-cabe/log && [ ! -e $D/se-par-nao-cabe.timers-antes.txt ]"
LIM=480 roda par-cabe-justo; confere "bloco do par cabe justo (480 s): a janela começa" "[ $RC = 0 ] && [ -s $D/se-par-cabe-justo/decisao.txt ]"
MOD="" roda sem-modelos; confere "envelope sem --modelos: recusa 2, pasta não criada" "[ $RC = 2 ] && [ ! -e $D/se-sem-modelos ]"
for mm in 5b ambos 1b,3b; do MOD=$mm roda mod-ruim; confere "envelope com --modelos '$mm': recusa 2" "[ $RC = 2 ] && [ ! -e $D/se-mod-ruim ]"; done
unset MOD
real() { local n=$1; shift; env -i PATH="$PATH" HOME="$HOME" PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID "$@" "$J" --modelos 1b --bin "$D/pp" --saida "$D/se-real-$n" --limite 60 > "$D/oute-real-$n" 2>&1; RC=$?; }
for v in LASTRO_PERFIL_BLOCO CMP_FRACAO_SLOTS CMP_CAMADAS_CPU CRITERIO_INSTR PLANO_INSTR CRITERIO_D1 CRITERIO_D2 SEM_SHA_FIXO MONITOR_SH LASTRO_SH_CMP EXTRAI_PY PERFIL_EXTRAI_PY INSTR_EXTRAI_PY CMP_RELOGIO_ARQ NVIDIA_SMI MODELO_1B MODELO_3B TEXTOS_5C REF_5C IDS_CMP CRITERIO_CMP \
         PROC_STAT CMP_FOLGA CMP_SEM_MEMORIA LOADAVG_ARQ SYSTEMCTL SYSTEMD_RUN PENDENTE CGROUP_RAIZ PROC_LISTA CMP_EST_QUENTE_1B CMP_FAKE_RECUSA JANELA_SH JANELA_TIMERS LIMITE_SH; do
  real "v-$v" "$v=1"
  confere "modo real com $v no ambiente: recusa 2 citando a variável, nada criado" "[ $RC = 2 ] && grep -q 'variável só de teste presente no modo real: $v' $D/oute-real-v-$v && [ ! -e $D/se-real-v-$v ]"
done
# caminho feliz (relógio real, células instantâneas)
LIM=1200 roda ok
confere "ok: sai 0, decisao.txt, 19 processos, timers parados e restaurados, resultado do extrator, prova da variável sem incoerência" "[ $RC = 0 ] && [ -s $D/se-ok/decisao.txt ] && [ \$(ls $D/se-ok/celulas | grep -c '^ctrl-') = 12 ] && [ -s $D/se-ok/instrumento-resultado.json ] && grep -q '^stop a.timer' $D/ok/log && [ \"\$(cat $D/ok/a.timer)\" = active ] && ! grep -q INCOERENTE $D/se-ok/prova-variavel.txt && grep -q 'resultado DESCRITIVO' $D/se-ok/decisao.txt"
confere "ok: aviso no fim da janela e listas systemd ao lado da pasta" "grep -q 'terminou (código 0)' $D/se-ok.aviso.txt && grep -q 'terminou (código 0)' $D/aviso-chamado && [ -s $D/se-ok.unidades-antes.txt ]"
# rodada 139: guarda de processos de teste em qualquer raiz de alvo, falha de leitura da lista, portão do sha do binário
for caso in "/tmp/target-instrumento/debug/deps/moe_rt-0123456789abcdef --nocapture" "target/debug/deps/moe_rt-0123456789abcdef" "./alvo-relativo/release/deps/leitura_regra-fedcba9876543210 --test-threads=1" "/r/target/debug/deps/moe_rt-0123456789abcdef"; do
  printf '4242 %s\n' "$caso" > "$D/proc-x"
  PROC_LISTA=$D/proc-x roda alvo-x; confere "executável de teste em raiz de alvo '${caso%% *}': recusa 2 antes do envelope" "[ $RC = 2 ] && grep -q 'processo que pode usar a 3080 Ti' $D/oute-alvo-x && [ ! -s $D/elog-alvo-x ] && ! grep -q '^stop' $D/alvo-x/log && [ ! -e $D/se-alvo-x ]"
done
PROC_LISTA=$D/nao-existe-lista roda ps-erro; confere "falha ao ler a lista de processos (ps com erro): recusa 2 com o motivo, nunca lista vazia, nada tocado" "[ $RC = 2 ] && grep -q 'ilegível' $D/oute-ps-erro && grep -q 'falha ao ler a lista de processos' $D/oute-ps-erro && [ ! -s $D/elog-ps-erro ] && ! grep -q '^stop' $D/ps-erro/log && [ ! -e $D/se-ps-erro ]"
printf '4242 cargo build --release --bin primeira-palavra\n4343 /x/target/release/deps/libmoe_rt-0123456789abcdef.rlib\n' > "$D/proc-x"
PROC_LISTA=$D/proc-x roda build-ok2; confere "cargo build continua permitido (e uma biblioteca .rlib em deps não é executável de teste)" "[ $RC = 0 ] && ! grep -q 'processo que pode usar' $D/oute-build-ok2"
REAL=$AQUI/../../.herdr/binarios/primeira-palavra-instr-ae14cbde
SEM_SHA_FIXO=0 roda bin-errado
confere "binário com sha diferente do FIXO (--bin falso, SEM_SHA_FIXO=0): recusa 2 sem envelope, timers, smoke nem célula, nada criado" "[ $RC = 2 ] && grep -q 'binário com sha diferente do FIXO' $D/oute-bin-errado && [ ! -s $D/elog-bin-errado ] && ! grep -q '^stop' $D/bin-errado/log && [ ! -e $D/se-bin-errado ] && [ ! -e $D/se-bin-errado.timers-antes.txt ]"
if [ -x "$REAL" ]; then
  SEM_SHA_FIXO=0 roda bin-congelado --bin "$REAL"
  confere "cópia congelada real (sha do FIXO) passa do portão do binário (as células são falsas; o binário real nunca roda aqui)" "! grep -q 'binário com sha diferente' $D/oute-bin-congelado && [ -s $D/se-bin-congelado/decisao.txt ] && grep -q 'ae14cbde3965493808818f3aa4a8e7665b6397531e36c2abf020c35fc4f3d969' $D/se-bin-congelado/integridade-antes.txt"
else echo "FALHOU: binário congelado ausente em $REAL"; falhas=$((falhas+1)); fi
# portão do binário também no modo interno (antes da primeira execução)
: > "$D/log-bi"
PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID CMP_FAKE_LOG=$D/log-bi SEM_SHA_FIXO=0 "$J" --interno --modelos 1b --saida "$D/si-bi" --bin "$D/pp" --limite 100 > "$D/outi-bi" 2>&1
confere "modo interno com binário fora do FIXO: recusa 2 antes de qualquer processo" "[ $? = 2 ] && [ ! -s $D/log-bi ] && grep -q 'binário com sha diferente do FIXO' $D/outi-bi"
# --verifica-fixos: ambiente limpo contra os arquivos REAIS
env -i PATH="$PATH" HOME="$HOME" "$J" --verifica-fixos > "$D/fixos.txt" 2>&1; rcf=$?
nok=$(grep -c ' OK$' "$D/fixos.txt"); nfal=$(grep -c ' FALHA$' "$D/fixos.txt"); naus=$(grep -c ' AUSENTE ' "$D/fixos.txt")
echo "congelados do FIXO: $nok OK, $nfal FALHA, $naus AUSENTE (código $rcf)"
confere "--verifica-fixos: código 0, nenhum FALHA/AUSENTE, itens essenciais presentes" "[ $rcf = 0 ] && [ $nfal = 0 ] && [ $naus = 0 ] && [ $nok -ge 20 ] && grep -q '^binario-instr .* OK\$' $D/fixos.txt && grep -q '^criterio-instrumento .* OK\$' $D/fixos.txt && grep -q '^script-instrumento-extrai .* OK\$' $D/fixos.txt && grep -q '^tokenizer-3b .* OK\$' $D/fixos.txt"
confere "--verifica-fixos: sha do binário congelado é ae14cbde... e o do critério é o do arquivo atual" "grep -q '^binario-instr ae14cbde3965493808818f3aa4a8e7665b6397531e36c2abf020c35fc4f3d969 ' $D/fixos.txt && [ \"\$(grep '^criterio-instrumento ' $D/fixos.txt | cut -d' ' -f2)\" = \"\$(sha256sum $AQUI/criterio-instrumento-prefill-20261110.md | cut -d' ' -f1)\" ]"
echo alterado > "$D/crit-alterado.md"
env -i PATH="$PATH" HOME="$HOME" CRITERIO_INSTR="$D/crit-alterado.md" "$J" --verifica-fixos > "$D/fixos-ruim.txt" 2>&1; rcr=$?
confere "--verifica-fixos com critério alterado: FALHA e código 1" "[ $rcr = 1 ] && grep -q '^criterio-instrumento .* FALHA\$' $D/fixos-ruim.txt"
env -i PATH="$PATH" HOME="$HOME" INSTR_EXTRAI_PY="$D/nao-existe.py" "$J" --verifica-fixos > "$D/fixos-aus.txt" 2>&1; rca=$?
confere "--verifica-fixos com extrator ausente: AUSENTE e código 1" "[ $rca = 1 ] && grep -q '^script-instrumento-extrai .* AUSENTE' $D/fixos-aus.txt"
if [ "$falhas" = 0 ]; then echo "TODOS OS TESTES PASSARAM"; exit 0; else echo "$falhas FALHA(S)"; exit 1; fi
