#!/usr/bin/env bash
# Teste do modo --degrau1 do roda-cmp-janela.sh (criterio-degrau1-100pct-20261109.md) SEM GPU: modo interno com relógio virtual, roteiros de célula e de smoke FALSOS
# (o julgamento do smoke é o do cmp-extrai.py REAL, sobre uma pasta de smoke sintética), nvidia-smi falso. mktemp -d, sem rm -rf.
set -u
unset E4B_GPU_AUTORIZADO E4B_MEDIDA_AUTORIZADA MOE_RT_REQUIRE_GPU PRIMEIRA_PALAVRA_AUTORIZADA LEITURA_CORRECAO_AUTORIZADA LEITURA_VELOCIDADE_AUTORIZADA
export CUDA_VISIBLE_DEVICES=
AQUI=$(cd "$(dirname "$0")" && pwd -P)
J=$AQUI/roda-cmp-janela.sh
UUID=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd
D=$(mktemp -d -p "${TMPDIR:-/tmp}" cmp-degrau1-teste.XXXXXX) || exit 1
echo "pasta de teste: $D"
falhas=0
confere() { if eval "$2"; then echo "ok: $1"; else echo "FALHOU: $1"; falhas=$((falhas+1)); fi; }
# ---- fixtures falsas
mkdir -p "$D/gguf" "$D/m/1b" "$D/m/3b" "$D/llama" "$D/textos" "$D/refs"
echo g1 > "$D/gguf/granite-3.1-1b-a400m-instruct-bf16.gguf"; echo g3 > "$D/gguf/granite-3.1-3b-a800m-instruct-bf16.gguf"
for m in 1b 3b; do echo cfg > "$D/m/$m/config.json"; echo tok > "$D/m/$m/tokenizer.json"; done
echo st > "$D/m/1b/model.safetensors"; echo s1 > "$D/m/3b/model-00001-of-00002.safetensors"; echo s2 > "$D/m/3b/model-00002-of-00002.safetensors"; echo ix > "$D/m/3b/model.safetensors.index.json"
echo srv > "$D/llama/llama-server"; echo lib > "$D/llama/libllama.so.0.8.0"
echo c > "$D/textos/texto-curto.txt"; echo m > "$D/textos/texto-medio.txt"; echo l > "$D/textos/texto-longo.txt"; echo '{"textos": []}' > "$D/textos/textos.json"
python3 -I - "$D" <<'PY'
import json, sys
d = sys.argv[1]
for m in ("1b", "3b"):
    json.dump({"casos": [{"nome": "curto", "ids": [1, 2], "gerados": list(range(64)), "gaps": [5.0] * 64}]}, open("%s/refs/ref5c-%s.json" % (d, m), "w"))
json.dump({"modelos": {"1b": {"curto": {"ids": [1, 2]}}, "3b": {"curto": {"ids": [1, 2]}}}}, open("%s/ids.json" % d, "w"))
PY
echo criterio > "$D/crit.md"; echo criterio-d1 > "$D/crit-d1.md"
printf '#!/usr/bin/env bash\nexit 0\n' > "$D/pp"; chmod +x "$D/pp"
printf '#!/usr/bin/env bash\ncase $* in *query-gpu*) echo "uuid, memory.used [MiB], utilization.gpu [%%]"; echo "GPU-falsa, 1 MiB, 0 %%";; *) echo;; esac\n' > "$D/smi"; chmod +x "$D/smi"
# célula falsa (Lastro e llama.cpp) e SMOKE falso, por variáveis FAKE_SMOKE_*: MEM ("usada, total"), CPU, IDS (igual|diferente), RC
cat > "$D/celula-falsa.sh" <<'F'
#!/usr/bin/env bash
if [ "$1" = --smoke ]; then
  shift; modelo=; saida=
  while [ $# -gt 0 ]; do case $1 in --modelo) modelo=$2;; --saida) saida=$2;; esac; shift; done
  echo "SMOKE $modelo CMP_FRACAO_SLOTS=${CMP_FRACAO_SLOTS:-}" >> "$CMP_FAKE_LOG"
  [ "${FAKE_SMOKE_RC:-0}" != 2 ] || { echo recusa >&2; exit 2; }
  mkdir -p "$saida/p1/saida"
  echo "${FAKE_SMOKE_RC:-0}" > "$saida/rc.txt"; echo "${FAKE_SMOKE_RC:-0}" > "$saida/p1/rc.txt"
  echo "pp --gera --celula $modelo:/m --so-texto curto --fracao-slots 1.0 --saida x" > "$saida/cmdline.txt"
  # leituras carimbadas dentro do intervalo do binário (1.000.000 a 1.004.000 ms); FAKE_SMOKE_MEM = "usada, total"
  m=${FAKE_SMOKE_MEM:-2000, 24000}
  if [ "${FAKE_SMOKE_SEM_LEITURA:-0}" = 1 ]; then : > "$saida/gpumem.csv"; echo "1004200, $m" > "$saida/gpumem-pos-saida.csv"   # só a amostra pós-saída (descritiva)
  else printf '1000500, %s\n1001500, %s\n1002500, %s\n1003500, %s\n' "$m" "$m" "$m" "$m" > "$saida/gpumem.csv"; fi
  printf 'coletor_inicio_ms=999900\nbin_inicio_ms=1000000\nbin_fim_ms=1004000\ncoletor_fim_ms=1004100\n' > "$saida/smoke-tempos.txt"
  python3 -I - "$saida" "$modelo" <<'PY'
import json, os, sys
saida, m = sys.argv[1:3]
g = list(range(64)) if os.environ.get("FAKE_SMOKE_IDS", "igual") == "igual" else list(range(1, 65))
json.dump({"medidas": [{"descartada": False, "falha": None, "gerados": g, "cont": {"needed": 100, "cpu": int(os.environ.get("FAKE_SMOKE_CPU", "0")), "hits": 100}}]}, open(os.path.join(saida, "p1", "saida", "medidas-%s.json" % m), "w"))
PY
  echo "slots 768 de 768" > "$saida/p1/stderr.txt"; : > "$saida/p1/stdout.txt"
  [ "${FAKE_SMOKE_RC:-0}" = 0 ] || exit 1
  exit 0
fi
# uso (lastro): --modelo M --texto T --estado E --n N --saida DIR --bin B ; (llama): ... --serie S --ids ARQ
modelo=; texto=; estado=; n=; saida=; serie=
while [ $# -gt 0 ]; do case $1 in --modelo) modelo=$2;; --texto) texto=$2;; --estado) estado=$2;; --n) n=$2;; --saida) saida=$2;; --serie) serie=$2;; esac; shift; done
if [ -z "$serie" ]; then lado=lastro; serie=L; [ "${CMP_FRACAO_SLOTS:-}" != 1.0 ] || serie=L100; else lado=llamacpp; fi
chave="$lado-$modelo-$serie-$texto-$estado"
echo "$chave n=$n frac=${CMP_FRACAO_SLOTS:-}" >> "$CMP_FAKE_LOG"
if [ "$lado" = lastro ]; then custo=$(( (n + 1) * 10 )); else custo=$(( (n + 1) * 5 )); fi
if [ -n "${CMP_FAKE_RELOGIO:-}" ]; then echo $(( $(cat "$CMP_FAKE_RELOGIO") + custo )) > "$CMP_FAKE_RELOGIO"; fi
mkdir -p "$saida"
printf '{"lado": "%s", "serie": "%s", "modelo": "%s", "texto": "%s", "estado": "%s", "n_medidas": %s, "portao_ok": true, "portao_motivo": "", "rc": 0}\n' "$lado" "$serie" "$modelo" "$texto" "$estado" "$n" > "$saida/celula.json"
exit 0
F
chmod +x "$D/celula-falsa.sh"
printf '#!/usr/bin/env bash\nexec %s "$@"\n' "$D/celula-falsa.sh" > "$D/lastro-falso"; cp "$D/lastro-falso" "$D/llama-falso"; chmod +x "$D/lastro-falso" "$D/llama-falso"
export CMP_EST_PORTAO=0 CMP_EST_L_CARGA_1B=0 CMP_EST_L_CARGA_3B=0 CMP_EST_L_FRIO_1B=10 CMP_EST_L_FRIO_3B=10 CMP_EST_L_QUENTE_1B=10 CMP_EST_L_QUENTE_3B=10
export CMP_EST_C_CARGA_1B=0 CMP_EST_C_CARGA_3B=0 CMP_EST_C_PED_1B=5 CMP_EST_C_PED_3B=5 CMP_EST_C_AQUEC=0 CMP_EST_C_FIXO=0
export NVIDIA_SMI=$D/smi LASTRO_SH_CMP=$D/lastro-falso LLAMA_SH_CMP=$D/llama-falso MODELO_1B=$D/m/1b MODELO_3B=$D/m/3b GGUF_DIR=$D/gguf LLAMA_DIR=$D/llama
export TEXTOS_5C=$D/textos REF_5C=$D/refs IDS_CMP=$D/ids.json CRITERIO_CMP=$D/crit.md CRITERIO_D1=$D/crit-d1.md SEM_SHA_FIXO=1 CMP_SEM_MEMORIA=1 CMP_FOLGA=0
interno() { # nome modelo limite [opções do roteiro...] (env FAKE_* e CMP_FRACAO_SLOTS opcionais); define RC; relógio virtual zerado
  local n=$1 m=$2 lim=$3; shift 3
  echo 0 > "$D/clock"; : > "$D/log-$n"
  PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID CMP_RELOGIO_ARQ=$D/clock CMP_FAKE_RELOGIO=$D/clock CMP_FAKE_LOG=$D/log-$n \
    "$J" --interno --modelos "$m" --saida "$D/si-$n" --bin "$D/pp" --limite "$lim" "$@" > "$D/outi-$n" 2>&1
  RC=$?
}
ordem() { grep -v '^smoke' "$D/si-$1/ordem.txt" | sed -n "${2}p" | cut -d' ' -f2; }

echo "== degrau 1: caminho feliz"
interno A 1b 100000 --degrau1
confere "A: sai 0; smoke julgado ANTES das células (1ª linha da ordem e do log); 12 células do 1B (6 L100 + 6 S1), nenhuma S2/S3" "[ $RC = 0 ] && head -n 1 $D/si-A/ordem.txt | grep -q '^smoke ' && head -n 1 $D/log-A | grep -q '^SMOKE 1b' && [ \$(grep -vc '^smoke' $D/si-A/ordem.txt) = 12 ] && [ \$(grep -c 'L100' $D/si-A/ordem.txt) = 6 ] && [ \$(grep -c 'S1' $D/si-A/ordem.txt) = 6 ] && ! grep -q 'S2\|S3' $D/si-A/ordem.txt && ! grep -q 'S2\|S3' $D/si-A/orcamento-log.txt"
confere "A: CMP_FRACAO_SLOTS=1.0 chega ao smoke e a toda célula do Lastro (série L100); o llama.cpp não é afetado" "grep -q '^SMOKE 1b CMP_FRACAO_SLOTS=1.0' $D/log-A && [ \$(grep -c '^lastro-1b-L100-.* frac=1.0$' $D/log-A) = 6 ] && ! grep '^lastro' $D/log-A | grep -vq 'L100'"
confere "A: veredito do smoke OK gravado e citado na decisão; rótulo exato da seção 4 na decisão e no relatório" "grep -q '^veredito=OK' $D/si-A/smoke/veredito.txt && grep -q '^smoke: veredito=OK' $D/si-A/decisao.txt && grep -q 'Lastro 100% experts GPU contra llama.cpp S1; KV f32 no Lastro contra f16 no llama.cpp' $D/si-A/decisao.txt && grep -q 'Degrau 1' $D/si-A/cmp-resultado.md && grep -q 'decode L100/S1 entre 2 e 6' $D/si-A/cmp-resultado.md"
confere "A: orcamento.json declara degrau1, S3 inativa, sem cortes; critério do degrau 1 na lista de integridade" "grep -q '\"degrau1\": true' $D/si-A/orcamento.json && grep -q '\"s3_ativa_no_fim\": false' $D/si-A/orcamento.json && grep -q '\"cortes\": \[\]' $D/si-A/orcamento.json && grep -q '^criterio-degrau1 ' $D/si-A/integridade-antes.txt && grep -q 'critério do degrau 1 sha256' $D/si-A/decisao.txt"
confere "A: lados ALTERNADOS (bloco 0 começa pelo Lastro L100, bloco 1 pelo S1); frio antes do quente" "[ \$(ordem A 1) = lastro-1b-L100-curto-frio ] && [ \$(ordem A 2) = llamacpp-1b-S1-curto-frio ] && [ \$(ordem A 5) = llamacpp-1b-S1-medio-frio ] && [ \$(ordem A 6) = lastro-1b-L100-medio-frio ]"
confere "A: o extrator foi chamado com --degrau1 (cmp-resultado.json degrau1)" "grep -q '\"degrau1\": true' $D/si-A/cmp-resultado.json"
interno B 3b 100000 --degrau1
confere "B: janela do 3B só com células do 3B e o smoke do 3B" "[ $RC = 0 ] && head -n 1 $D/log-B | grep -q '^SMOKE 3b' && ! grep -q '1b-' <(grep -v '^SMOKE' $D/log-B) && [ \$(grep -vc '^smoke' $D/si-B/ordem.txt) = 12 ]"
echo "== degrau 1: smoke que falha para a janela do modelo"
FAKE_SMOKE_CPU=7 interno C 1b 100000 --degrau1
confere "C: cpu/needed != 0: nenhuma célula roda, decisão 'L100 NÃO SUSTENTA 100%: cpu/needed', sem razão publicada" "[ $RC = 0 ] && [ \$(grep -vc '^SMOKE' $D/log-C) = 0 ] && [ \$(ls $D/si-C/celulas | wc -l) = 0 ] && grep -q '^L100 NÃO SUSTENTA 100%: cpu/needed' $D/si-C/decisao.txt && grep -q 'L100 NÃO SUSTENTA 100%' $D/si-C/orcamento.json && grep -q 'L100 NÃO SUSTENTA 100%' $D/si-C/cmp-resultado.md && ! grep -q '| OK |' $D/si-C/cmp-resultado.md"
FAKE_SMOKE_MEM="23000, 24000" interno D 1b 100000 --degrau1
confere "D: GPU > 90%: janela para, 'L100 NÃO SUSTENTA 100%: memória usada da GPU'" "[ \$(ls $D/si-D/celulas | wc -l) = 0 ] && grep -q '^L100 NÃO SUSTENTA 100%: memória usada da GPU' $D/si-D/decisao.txt"
FAKE_SMOKE_SEM_LEITURA=1 interno D2 1b 100000 --degrau1
confere "D2: só a amostra pós-saída (sem leitura durante a execução): janela para, nenhuma célula, motivo de memória" "[ \$(ls $D/si-D2/celulas | wc -l) = 0 ] && grep -q '^L100 NÃO SUSTENTA 100%: só 0 leituras válidas de memória' $D/si-D2/decisao.txt"
FAKE_SMOKE_RC=1 interno E 1b 100000 --degrau1
confere "E: processo do smoke com código != 0: janela para" "[ \$(ls $D/si-E/celulas | wc -l) = 0 ] && grep -q '^L100 NÃO SUSTENTA 100%: o processo do smoke' $D/si-E/decisao.txt"
FAKE_SMOKE_RC=2 interno F 1b 100000 --degrau1
confere "F: smoke com recusa de ambiente (2): janela interrompida, nada roda" "[ \$(ls $D/si-F/celulas | wc -l) = 0 ] && grep -q 'smoke recusou o ambiente' $D/si-F/orcamento-log.txt && grep -q 'L100 NÃO SUSTENTA 100%' $D/si-F/decisao.txt"
FAKE_SMOKE_IDS=diferente interno G 1b 100000 --degrau1
confere "G: ids do smoke diferentes da referência 5c: REPORTADO, a janela segue (12 células), decisão traz ids=DIVERGEM" "[ $RC = 0 ] && [ \$(grep -vc '^smoke' $D/si-G/ordem.txt) = 12 ] && grep -q 'ids=DIVERGEM' $D/si-G/decisao.txt && ! grep -q 'NÃO SUSTENTA' $D/si-G/decisao.txt"
echo "== degrau 1: orçamento (só os cortes 3 e 4; sem S2/S3)"
interno H 1b 400 --degrau1
confere "H (limite 400, projeção 450): corte (3) quente 5->3; nenhum corte de S3/S2" "grep -q '(3) quente cai' $D/si-H/orcamento.json && ! grep -q '(1) S3\|(2)' $D/si-H/orcamento.json && grep -c 'quente n=3' $D/si-H/ordem.txt | grep -q '^[1-9]'"
interno I 1b 200 --degrau1
confere "I (limite 200): cortes 3 e 4 esgotados, o que não cabe é NÃO RODOU (nunca estimado); nada de S2/S3 no registro" "grep -q '(3) quente' $D/si-I/orcamento.json && grep -q '(4) frio' $D/si-I/orcamento.json && grep -q 'NÃO RODOU: bloco' $D/si-I/orcamento-log.txt && ! grep -q 'S2\|S3' $D/si-I/orcamento-log.txt"
echo "== modo antigo intacto"
CMP_FRACAO_SLOTS=1.0 interno J 1b 100000 --sem-s3
confere "J: sem --degrau1 a janela NÃO repassa a fração (mesmo com CMP_FRACAO_SLOTS=1.0 no ambiente de quem chama): série L, sem smoke" "[ $RC = 0 ] && ! grep -q 'SMOKE\|L100' $D/log-J && grep -q '^lastro-1b-L-curto-frio n=3 frac=\$' $D/log-J && [ ! -e $D/si-J/smoke ] && ! grep -q '^criterio-degrau1 ' $D/si-J/integridade-antes.txt && grep -q '\"degrau1\": false' $D/si-J/orcamento.json"
confere "J: relatório do modo antigo segue com o rótulo de 10%" "grep -q 'Lastro 10% experts GPU contra' $D/si-J/cmp-resultado.md"
echo "== envelope e FIXO"
for v in CMP_FRACAO_SLOTS CRITERIO_D1; do
  env -i PATH="$PATH" HOME="$HOME" PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID "$v=1.0" "$J" --degrau1 --modelos 1b --bin "$D/pp" --saida "$D/se-real-$v" --limite 60 > "$D/oute-$v" 2>&1; RC=$?
  confere "modo real com $v no ambiente: recusa 2 citando a variável, nada criado" "[ $RC = 2 ] && grep -q 'variável só de teste presente no modo real: $v' $D/oute-$v && [ ! -e $D/se-real-$v ]"
done
env -i PATH="$PATH" HOME="$HOME" PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID "$J" --degrau1 --saida "$D/se-sem-mod" > "$D/oute-sem-mod" 2>&1
confere "--degrau1 sem --modelos: recusa 2 (uma janela por modelo)" "[ $? = 2 ] && [ ! -e $D/se-sem-mod ]"
env -i PATH="$PATH" HOME="$HOME" "$J" --verifica-fixos > "$D/fixos.txt" 2>&1; rcf=$?
confere "--verifica-fixos: 11 OK (os 10 antigos + criterio-degrau1), nenhum FALHA/AUSENTE, sha do novo critério = sha256sum direto" "[ $rcf = 0 ] && [ \$(grep -c ' OK\$' $D/fixos.txt) = 11 ] && ! grep -q 'FALHA\|AUSENTE' $D/fixos.txt && [ \"\$(grep '^criterio-degrau1 ' $D/fixos.txt | cut -d' ' -f2)\" = \"\$(sha256sum $AQUI/criterio-degrau1-100pct-20261109.md | cut -d' ' -f1)\" ] && [ \"\$(grep '^criterio-cmp ' $D/fixos.txt | cut -d' ' -f2)\" = f1bd1e0f35378f5efe16ea156546d8b96370b754f986c4dca56e2ecd191a5932 ]"
if [ "$falhas" = 0 ]; then echo "TODOS OS TESTES PASSARAM"; exit 0; else echo "$falhas FALHA(S)"; exit 1; fi
