#!/usr/bin/env bash
# Teste do modo --degrau2 do roda-cmp-janela.sh (criterio-degrau2-camadas-20261109.md) SEM GPU: modo interno com relógio virtual, roteiros de célula e de smoke FALSOS
# (o julgamento do smoke é o do cmp-extrai.py REAL, sobre uma pasta de smoke sintética), nvidia-smi falso. mktemp -d, sem rm -rf.
set -u
unset E4B_GPU_AUTORIZADO E4B_MEDIDA_AUTORIZADA MOE_RT_REQUIRE_GPU PRIMEIRA_PALAVRA_AUTORIZADA LEITURA_CORRECAO_AUTORIZADA LEITURA_VELOCIDADE_AUTORIZADA
export CUDA_VISIBLE_DEVICES=
AQUI=$(cd "$(dirname "$0")" && pwd -P)
J=$AQUI/roda-cmp-janela.sh
UUID=GPU-b22d9465-8552-9f97-812e-380b590645f8
D=$(mktemp -d -p "${TMPDIR:-/tmp}" cmp-degrau2-teste.XXXXXX) || exit 1
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
echo criterio > "$D/crit.md"; echo criterio-d1 > "$D/crit-d1.md"; echo criterio-d2 > "$D/crit-d2.md"
printf '#!/usr/bin/env bash\nexit 0\n' > "$D/pp"; chmod +x "$D/pp"
printf '#!/usr/bin/env bash\ncase $* in *power.limit*) case $* in *"query-gpu=power.limit --format"*) echo "370.00 W";; *) echo "370.00 W, 16, 4";; esac;; *query-gpu*) echo "uuid, memory.used [MiB], utilization.gpu [%%]"; echo "GPU-falsa, 1 MiB, 0 %%";; *) echo;; esac\n' > "$D/smi"; chmod +x "$D/smi"
# célula falsa (Lastro e llama.cpp) e SMOKE falso, por variáveis FAKE_SMOKE_*: MEM ("usada, total"), CPU, IDS (igual|diferente), RC
cat > "$D/celula-falsa.sh" <<'F'
#!/usr/bin/env bash
if [ "$1" = --smoke ]; then
  shift; modelo=; saida=
  while [ $# -gt 0 ]; do case $1 in --modelo) modelo=$2;; --saida) saida=$2;; esac; shift; done
  echo "SMOKE $modelo CMP_CAMADAS_CPU=${CMP_CAMADAS_CPU:-} FRAC=${CMP_FRACAO_SLOTS:-}" >> "$CMP_FAKE_LOG"
  [ "${FAKE_SMOKE_RC:-0}" != 2 ] || { echo recusa >&2; exit 2; }
  mkdir -p "$saida/p1/saida"
  echo "${FAKE_SMOKE_RC:-0}" > "$saida/rc.txt"; echo "${FAKE_SMOKE_RC:-0}" > "$saida/p1/rc.txt"
  echo "pp --gera --celula $modelo:/m --so-texto curto ${FAKE_FLAG:---camadas-cpu $CMP_CAMADAS_CPU} --saida x" > "$saida/cmdline.txt"
  # leituras carimbadas dentro do intervalo do binário (1.000.000 a 1.004.000 ms); FAKE_SMOKE_MEM = "usada, total"
  m=${FAKE_SMOKE_MEM:-2000, 24000}
  if [ "${FAKE_SMOKE_SEM_LEITURA:-0}" = 1 ]; then : > "$saida/gpumem.csv"; echo "1004200, $m" > "$saida/gpumem-pos-saida.csv"   # só a amostra pós-saída (descritiva)
  else printf '1000500, %s\n1001500, %s\n1002500, %s\n1003500, %s\n' "$m" "$m" "$m" "$m" > "$saida/gpumem.csv"; fi
  printf 'coletor_inicio_ms=999900\nbin_inicio_ms=1000000\nbin_fim_ms=1004000\ncoletor_fim_ms=1004100\n' > "$saida/smoke-tempos.txt"
  python3 -I - "$saida" "$modelo" <<'PY'
import json, os, sys
saida, m = sys.argv[1:3]
g = list(range(64)) if os.environ.get("FAKE_SMOKE_IDS", "igual") == "igual" else list(range(1, 65))
json.dump({"medidas": [{"descartada": False, "falha": None, "gerados": g, "cont": {"needed": (24 if m == "1b" else 32) * 8 * 152, "cpu": int(os.environ.get("FAKE_SMOKE_CPU", str((22 if m == "1b" else 29) * 8 * 152))), "hits": 100}}]}, open(os.path.join(saida, "p1", "saida", "medidas-%s.json" % m), "w"))
PY
  if [ -n "${FAKE_LINHA:-}" ]; then echo "$FAKE_LINHA" > "$saida/p1/stderr.txt"; elif [ "$modelo" = 1b ]; then echo "camadas-cpu 22 de 24; slots 64 de 768" > "$saida/p1/stderr.txt"; else echo "camadas-cpu 29 de 32; slots 120 de 1280" > "$saida/p1/stderr.txt"; fi; : > "$saida/p1/stdout.txt"
  [ "${FAKE_SMOKE_RC:-0}" = 0 ] || exit 1
  exit 0
fi
# uso (lastro): --modelo M --texto T --estado E --n N --saida DIR --bin B ; (llama): ... --serie S --ids ARQ
modelo=; texto=; estado=; n=; saida=; serie=
while [ $# -gt 0 ]; do case $1 in --modelo) modelo=$2;; --texto) texto=$2;; --estado) estado=$2;; --n) n=$2;; --saida) saida=$2;; --serie) serie=$2;; esac; shift; done
if [ -z "$serie" ]; then lado=lastro; serie=L; [ -z "${CMP_CAMADAS_CPU:-}" ] || serie=LC; else lado=llamacpp; fi
chave="$lado-$modelo-$serie-$texto-$estado"
echo "$chave n=$n cam=${CMP_CAMADAS_CPU:-} frac=${CMP_FRACAO_SLOTS:-}" >> "$CMP_FAKE_LOG"
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
export TEXTOS_5C=$D/textos REF_5C=$D/refs IDS_CMP=$D/ids.json CRITERIO_CMP=$D/crit.md CRITERIO_D1=$D/crit-d1.md CRITERIO_D2=$D/crit-d2.md SEM_SHA_FIXO=1 CMP_SEM_MEMORIA=1 CMP_FOLGA=0
interno() { # nome modelo limite [opções do roteiro...] (env FAKE_* e CMP_FRACAO_SLOTS opcionais); define RC; relógio virtual zerado
  local n=$1 m=$2 lim=$3; shift 3
  echo 0 > "$D/clock"; : > "$D/log-$n"
  PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID CMP_RELOGIO_ARQ=$D/clock CMP_FAKE_RELOGIO=$D/clock CMP_FAKE_LOG=$D/log-$n \
    "$J" --interno --modelos "$m" --saida "$D/si-$n" --bin "$D/pp" --limite "$lim" "$@" > "$D/outi-$n" 2>&1
  RC=$?
}
ordem() { grep -v '^smoke' "$D/si-$1/ordem.txt" | sed -n "${2}p" | cut -d' ' -f2; }

echo "== degrau 2: caminho feliz"
interno A 1b 100000 --degrau2
confere "A: sai 0; smoke julgado ANTES das células; 12 células do 1B (6 LC + 6 S2), nenhuma S1/S3" "[ $RC = 0 ] && head -n 1 $D/si-A/ordem.txt | grep -q '^smoke ' && head -n 1 $D/log-A | grep -q '^SMOKE 1b' && [ \$(grep -vc '^smoke' $D/si-A/ordem.txt) = 12 ] && [ \$(grep -c 'lastro-1b-LC' $D/si-A/ordem.txt) = 6 ] && [ \$(grep -c 'llamacpp-1b-S2' $D/si-A/ordem.txt) = 6 ] && ! grep -q 'S1\|S3\|L100' $D/si-A/ordem.txt && ! grep -q 'S1\|S3' $D/si-A/orcamento-log.txt"
confere "A: CMP_CAMADAS_CPU=22 chega ao smoke e a toda célula do Lastro (série LC), sem CMP_FRACAO_SLOTS; o llama.cpp roda sem a variável de camadas por célula" "grep -q '^SMOKE 1b CMP_CAMADAS_CPU=22 FRAC=\$' $D/log-A && [ \$(grep -c '^lastro-1b-LC-.* cam=22 frac=\$' $D/log-A) = 6 ]"
confere "A: veredito do smoke OK (cpu/needed alto NÃO reprova) citado na decisão; rótulo exato com N=22 na decisão e no relatório" "grep -q '^veredito=OK' $D/si-A/smoke/veredito.txt && grep -q '^smoke: veredito=OK' $D/si-A/decisao.txt && grep -q '^cpu_needed=0.9166' $D/si-A/smoke/veredito.txt && grep -q 'Lastro camadas inteiras (22 na CPU) contra llama.cpp S2 (--n-cpu-moe 22); KV f32 no Lastro contra f16 no llama.cpp; op offload ligado no llama.cpp, prefill da S2 não é leitura de experts na CPU' $D/si-A/decisao.txt && grep -q 'Degrau 2' $D/si-A/cmp-resultado.md && grep -q 'decode LC/S2 <= 1,25: equivale' $D/si-A/cmp-resultado.md && grep -q 'integridade: íntegra' $D/si-A/decisao.txt"
confere "A: orcamento.json declara degrau2 e não degrau1, S3 inativa, sem cortes; critério do degrau 2 na lista de integridade e na decisão" "grep -q '\"degrau2\": true' $D/si-A/orcamento.json && grep -q '\"degrau1\": false' $D/si-A/orcamento.json && grep -q '\"s3_ativa_no_fim\": false' $D/si-A/orcamento.json && grep -q '\"cortes\": \[\]' $D/si-A/orcamento.json && grep -q '^criterio-degrau2 ' $D/si-A/integridade-antes.txt && grep -q 'critério do degrau 2 sha256' $D/si-A/decisao.txt"
confere "A: lados ALTERNADOS (bloco 0 começa pelo Lastro LC, bloco 1 pela S2); frio antes do quente" "[ \$(ordem A 1) = lastro-1b-LC-curto-frio ] && [ \$(ordem A 2) = llamacpp-1b-S2-curto-frio ] && [ \$(ordem A 5) = llamacpp-1b-S2-medio-frio ] && [ \$(ordem A 6) = lastro-1b-LC-medio-frio ]"
confere "A: o extrator foi chamado com --degrau2 (cmp-resultado.json degrau2)" "grep -q '\"degrau2\": true' $D/si-A/cmp-resultado.json"
interno B 3b 100000 --degrau2
confere "B: janela do 3B só com células do 3B, smoke do 3B e N=29; rótulo com 29" "[ $RC = 0 ] && head -n 1 $D/log-B | grep -q '^SMOKE 3b CMP_CAMADAS_CPU=29' && ! grep -q '1b-' <(grep -v '^SMOKE' $D/log-B) && [ \$(grep -vc '^smoke' $D/si-B/ordem.txt) = 12 ] && [ \$(grep -c '^lastro-3b-LC-.* cam=29 ' $D/log-B) = 6 ] && grep -q 'Lastro camadas inteiras (29 na CPU) contra llama.cpp S2 (--n-cpu-moe 29)' $D/si-B/decisao.txt"
echo "== degrau 2: smoke que falha para a janela do modelo"
FAKE_LINHA="camadas-cpu 21 de 24; slots 96 de 768" interno C 1b 100000 --degrau2
confere "C: linha com N errado: nenhuma célula roda, 'SMOKE DO DEGRAU 2 FALHOU', sem razão publicada" "[ $RC = 0 ] && [ \$(grep -vc '^SMOKE' $D/log-C) = 0 ] && [ \$(ls $D/si-C/celulas | wc -l) = 0 ] && grep -q '^SMOKE DO DEGRAU 2 FALHOU: .*linha de camadas/slots diferente' $D/si-C/decisao.txt && grep -q 'SMOKE DO DEGRAU 2 FALHOU' $D/si-C/orcamento.json && grep -q 'SMOKE DO DEGRAU 2 FALHOU' $D/si-C/cmp-resultado.md && ! grep -q '| OK |' $D/si-C/cmp-resultado.md"
FAKE_LINHA="camadas-cpu 22 de 24; slots 768 de 768" interno C2 1b 100000 --degrau2
confere "C2: slots errados (768 em vez de 64): janela para" "[ \$(ls $D/si-C2/celulas | wc -l) = 0 ] && grep -q '^SMOKE DO DEGRAU 2 FALHOU' $D/si-C2/decisao.txt"
FAKE_LINHA="slots 768 de 768" interno C3 1b 100000 --degrau2
confere "C3: só a linha antiga 'slots N de M' (sem a de camadas): janela para" "[ \$(ls $D/si-C3/celulas | wc -l) = 0 ] && grep -q '^SMOKE DO DEGRAU 2 FALHOU: sem a linha' $D/si-C3/decisao.txt"
FAKE_FLAG="--fracao-slots 1.0" interno C4 1b 100000 --degrau2
confere "C4: linha de comando do smoke sem --camadas-cpu N: janela para" "[ \$(ls $D/si-C4/celulas | wc -l) = 0 ] && grep -q '^SMOKE DO DEGRAU 2 FALHOU: linha de comando do smoke sem --camadas-cpu 22' $D/si-C4/decisao.txt"
FAKE_SMOKE_MEM="23000, 24000" interno D 1b 100000 --degrau2
confere "D: GPU > 90%: janela para, 'SMOKE DO DEGRAU 2 FALHOU: memória usada da GPU'" "[ \$(ls $D/si-D/celulas | wc -l) = 0 ] && grep -q '^SMOKE DO DEGRAU 2 FALHOU: memória usada da GPU' $D/si-D/decisao.txt"
FAKE_SMOKE_SEM_LEITURA=1 interno D2 1b 100000 --degrau2
confere "D2: só a amostra pós-saída: janela para, nenhuma célula, motivo de memória" "[ \$(ls $D/si-D2/celulas | wc -l) = 0 ] && grep -q '^SMOKE DO DEGRAU 2 FALHOU: só 0 leituras válidas de memória' $D/si-D2/decisao.txt"
FAKE_SMOKE_RC=1 interno E 1b 100000 --degrau2
confere "E: processo do smoke com código != 0: janela para" "[ \$(ls $D/si-E/celulas | wc -l) = 0 ] && grep -q '^SMOKE DO DEGRAU 2 FALHOU: o processo do smoke' $D/si-E/decisao.txt"
FAKE_SMOKE_RC=2 interno F 1b 100000 --degrau2
confere "F: smoke com recusa de ambiente (2): janela interrompida, nada roda" "[ \$(ls $D/si-F/celulas | wc -l) = 0 ] && grep -q 'smoke recusou o ambiente' $D/si-F/orcamento-log.txt && grep -q 'SMOKE DO DEGRAU 2 FALHOU' $D/si-F/decisao.txt"
FAKE_SMOKE_CPU=0 interno F2 1b 100000 --degrau2
confere "F2: cpu = 0 no smoke: janela para, nenhuma célula (posicionamento não provado)" "[ \$(ls $D/si-F2/celulas | wc -l) = 0 ] && grep -q '^SMOKE DO DEGRAU 2 FALHOU: cpu/needed = 0.0, esperado 22/24 exato' $D/si-F2/decisao.txt"
FAKE_SMOKE_IDS=diferente interno G 1b 100000 --degrau2
confere "G: ids do smoke diferentes da referência 5c: REPORTADO, a janela segue (12 células), decisão traz ids=DIVERGEM" "[ $RC = 0 ] && [ \$(grep -vc '^smoke' $D/si-G/ordem.txt) = 12 ] && grep -q 'ids=DIVERGEM' $D/si-G/decisao.txt && ! grep -q 'FALHOU' $D/si-G/decisao.txt"
echo "== degrau 2: orçamento (só os cortes 3 e 4; sem S1/S3)"
interno H 1b 400 --degrau2
confere "H (limite 400): corte (3) quente 5->3; nenhum corte de S3/S2" "grep -q '(3) quente cai' $D/si-H/orcamento.json && ! grep -q '(1) S3\|(2)' $D/si-H/orcamento.json && grep -c 'quente n=3' $D/si-H/ordem.txt | grep -q '^[1-9]'"
interno I 1b 200 --degrau2
confere "I (limite 200): cortes 3 e 4 esgotados, o que não cabe é NÃO RODOU; nada de S1/S3 no registro" "grep -q '(3) quente' $D/si-I/orcamento.json && grep -q '(4) frio' $D/si-I/orcamento.json && grep -q 'NÃO RODOU: bloco' $D/si-I/orcamento-log.txt && ! grep -q 'S1\|S3' $D/si-I/orcamento-log.txt"
echo "== exclusividade e modos antigos intactos"
interno X 1b 100000 --degrau1 --degrau2
confere "X: --degrau1 com --degrau2: recusa 2, nada roda" "[ $RC = 2 ] && [ ! -s $D/log-X ] && grep -q 'exclusivos' $D/outi-X"
CMP_CAMADAS_CPU=22 interno J 1b 100000 --sem-s3
confere "J: sem --degrau2 a janela NÃO repassa CMP_CAMADAS_CPU (mesmo definida no ambiente de quem chama): série L, sem smoke" "[ $RC = 0 ] && ! grep -q 'SMOKE\|LC\|L100' $D/log-J && grep -q '^lastro-1b-L-curto-frio n=3 cam= frac=\$' $D/log-J && [ ! -e $D/si-J/smoke ] && ! grep -q '^criterio-degrau2 ' $D/si-J/integridade-antes.txt && grep -q '\"degrau2\": false' $D/si-J/orcamento.json"
FAKE_FLAG="--fracao-slots 1.0" FAKE_LINHA="slots 768 de 768" FAKE_SMOKE_CPU=0 CMP_CAMADAS_CPU=22 interno J1 1b 100000 --degrau1
confere "J1: degrau 1 não repassa CMP_CAMADAS_CPU (usa só a fração)" "[ $RC = 0 ] && grep -q '^SMOKE 1b CMP_CAMADAS_CPU= FRAC=1.0' $D/log-J1 && [ \$(grep -c '^lastro.* cam= frac=1.0' $D/log-J1) = 6 ] && ! grep -q 'cam=22' $D/log-J1"
echo "== envelope e FIXO"
for v in CMP_FRACAO_SLOTS CMP_CAMADAS_CPU CRITERIO_D1 CRITERIO_D2; do
  env -i PATH="$PATH" HOME="$HOME" PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID "$v=22" "$J" --degrau2 --modelos 1b --bin "$D/pp" --saida "$D/se-real-$v" --limite 60 > "$D/oute-$v" 2>&1; RC=$?
  confere "modo real com $v no ambiente: recusa 2 citando a variável, nada criado" "[ $RC = 2 ] && grep -q 'variável só de teste presente no modo real: $v' $D/oute-$v && [ ! -e $D/se-real-$v ]"
done
env -i PATH="$PATH" HOME="$HOME" PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID "$J" --degrau2 --saida "$D/se-sem-mod" > "$D/oute-sem-mod" 2>&1
confere "--degrau2 sem --modelos: recusa 2 (uma janela por modelo)" "[ $? = 2 ] && [ ! -e $D/se-sem-mod ]"
env -i PATH="$PATH" HOME="$HOME" "$J" --verifica-fixos > "$D/fixos.txt" 2>&1; rcf=$?
confere "--verifica-fixos: 12 OK (os 10 antigos + criterio-degrau1 + criterio-degrau2), nenhum FALHA/AUSENTE; sha do critério novo = sha256sum direto" "[ $rcf = 0 ] && [ \$(grep -c ' OK\$' $D/fixos.txt) = 12 ] && ! grep -q 'FALHA\|AUSENTE' $D/fixos.txt && [ \"\$(grep '^criterio-degrau2 ' $D/fixos.txt | cut -d' ' -f2)\" = \"\$(sha256sum $AQUI/criterio-degrau2-camadas-20261109.md | cut -d' ' -f1)\" ] && [ \"\$(grep '^criterio-degrau1 ' $D/fixos.txt | cut -d' ' -f2)\" = 8c3406a9fd3ba55133dfb2c2d3800076c6ee0a765735edf8be65f6ba1a72c7de ] && [ \"\$(grep '^criterio-cmp ' $D/fixos.txt | cut -d' ' -f2)\" = bf3ef09b9c1e915e248050bdf2a509c04aab08c7a5f11db1c335f424d3b2c64f ]"
if [ "$falhas" = 0 ]; then echo "TODOS OS TESTES PASSARAM"; exit 0; else echo "$falhas FALHA(S)"; exit 1; fi
