#!/usr/bin/env bash
# Teste do roda-perfil-janela.sh SEM GPU: roteiros de célula e de smoke FALSOS (avançam um RELÓGIO VIRTUAL, sem dormir), nvidia-smi falso, arquivos e modelos falsos.
# O julgamento do smoke é o do cmp-extrai.py REAL, sobre uma pasta de smoke sintética. mktemp -d, sem rm -rf.
# Parte 1 (modo --interno, sem systemd): ordem das células, ambiente por série, P4, orçamento (corte 5 -> 3 e NÃO RODOU), smoke falho, recusa de ambiente, integridade.
# Parte 2 (envelope real: janela-e4b.sh + roda-com-limite.sh em unidades transitórias de `systemd --user`, mas systemctl FALSO para os timers): recusas, pastas novas, aviso, limite, --verifica-fixos.
set -u
unset E4B_GPU_AUTORIZADO E4B_MEDIDA_AUTORIZADA MOE_RT_REQUIRE_GPU PRIMEIRA_PALAVRA_AUTORIZADA LEITURA_CORRECAO_AUTORIZADA LEITURA_VELOCIDADE_AUTORIZADA
export CUDA_VISIBLE_DEVICES=
AQUI=$(cd "$(dirname "$0")" && pwd -P)
J=$AQUI/roda-perfil-janela.sh
UUID=GPU-b22d9465-8552-9f97-812e-380b590645f8
D=$(mktemp -d -p "${TMPDIR:-/tmp}" perfil-janela-teste.XXXXXX) || exit 1
echo "pasta de teste: $D"
falhas=0
confere() { if eval "$2"; then echo "ok: $1"; else echo "FALHOU: $1"; falhas=$((falhas+1)); fi; }
# ---- fixtures falsas
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
echo criterio > "$D/crit.md"; echo c1 > "$D/crit-d1.md"; echo c2 > "$D/crit-d2.md"; echo perfil > "$D/crit-perfil.md"
printf '#!/usr/bin/env bash\nexit 0\n' > "$D/pp"; chmod +x "$D/pp"
# nvidia-smi falso: -lms imprime uma amostra e sai; registra os argumentos em $SMI_LOG se definido
cat > "$D/smi" <<'F'
#!/usr/bin/env bash
[ -z "${SMI_LOG:-}" ] || echo "$*" >> "$SMI_LOG"
case $* in
  *-lms*) echo "2026/11/09 04:00:00.000, 12 %";;
  *"--query-gpu=power.limit --format"*) f=${0%/*}/pot-seq; if [ -s "$f" ]; then l=$(head -n 1 "$f"); tail -n +2 "$f" > "$f.t"; mv "$f.t" "$f"; [ "$l" = FALHA ] && exit 1; echo "$l"; else echo "370.00 W"; fi;;
  *"-i GPU-b22d9465-8552-9f97-812e-380b590645f8 --query-gpu=power.limit,pcie.link.width.current,pcie.link.gen.max"*) echo "370.00 W, 16, 4";;
  *query-gpu*) echo "uuid, memory.used [MiB], utilization.gpu [%]"; echo "GPU-falsa, 1 MiB, 0 %";;
  *pmon*) if [ -n "${FAKE_PMON_ARQ:-}" ]; then cat "$FAKE_PMON_ARQ"; else echo "# gpu pid type sm mem enc dec jpg ofa fb ccpm command"; echo "# Idx # C/G % % % % % % MB MB name"; fi;;
  *) echo;;
esac
F
chmod +x "$D/smi"
# célula e smoke falsos (roda-cmp-lastro.sh): registram a chamada e as variáveis de série; avançam o relógio virtual pelo custo da célula
cat > "$D/lastro-falso.sh" <<'F'
#!/usr/bin/env bash
if [ "$1" = --smoke ]; then
  shift; modelo=; saida=
  while [ $# -gt 0 ]; do case $1 in --modelo) modelo=$2;; --saida) saida=$2;; esac; shift; done
  echo "SMOKE $modelo FRAC=${CMP_FRACAO_SLOTS:-} CAMADAS=${CMP_CAMADAS_CPU:-}" >> "$CMP_FAKE_LOG"
  [ "${FAKE_SMOKE_RC:-0}" != 2 ] || { echo recusa >&2; exit 2; }
  mkdir -p "$saida/p1/saida"
  echo 0 > "$saida/rc.txt"; echo 0 > "$saida/p1/rc.txt"
  L=24; E=32; N=22; [ "$modelo" = 3b ] && { L=32; E=40; N=29; }
  if [ -n "${CMP_FRACAO_SLOTS:-}" ]; then
    echo "pp --gera --celula $modelo:/m --so-texto curto --fracao-slots 1.0 --saida x" > "$saida/cmdline.txt"
    echo "slots $((L * E)) de $((L * E))" > "$saida/p1/stderr.txt"; cpu=${FAKE_SMOKE_CPU:-0}
    [ "${FAKE_SMOKE_FALHA:-}" != L100 ] || cpu=7
  else
    echo "pp --gera --celula $modelo:/m --so-texto curto --camadas-cpu $N --saida x" > "$saida/cmdline.txt"
    echo "camadas-cpu $N de $L; slots $(((L - N) * E)) de $((L * E))" > "$saida/p1/stderr.txt"; cpu=$((N * E * 152))
    [ "${FAKE_SMOKE_FALHA:-}" != LC ] || cpu=$((cpu + 1))
  fi
  printf '1000500, 2000, 24000\n1001500, 2000, 24000\n1002500, 2000, 24000\n1003500, 2000, 24000\n' > "$saida/gpumem.csv"
  printf 'coletor_inicio_ms=999900\nbin_inicio_ms=1000000\nbin_fim_ms=1004000\ncoletor_fim_ms=1004100\n' > "$saida/smoke-tempos.txt"
  python3 -I - "$saida" "$modelo" "$L" "$E" "$cpu" <<'PY'
import json, os, sys
saida, m, L, E, cpu = sys.argv[1:6]
json.dump({"medidas": [{"descartada": False, "falha": None, "gerados": list(range(64)), "cont": {"needed": int(L) * int(E) * 152, "cpu": int(cpu), "hits": 100}}]}, open(os.path.join(saida, "p1", "saida", "medidas-%s.json" % m), "w"))
PY
  : > "$saida/p1/stdout.txt"
  exit 0
fi
modelo=; texto=; estado=; n=; saida=
while [ $# -gt 0 ]; do case $1 in --modelo) modelo=$2;; --texto) texto=$2;; --estado) estado=$2;; --n) n=$2;; --saida) saida=$2;; esac; shift; done
serie=L10; [ -z "${CMP_FRACAO_SLOTS:-}" ] || serie=L100; [ -z "${CMP_CAMADAS_CPU:-}" ] || serie=LC
chave="lastro-$modelo-$serie-$texto-$estado"
echo "$chave n=$n FRAC=${CMP_FRACAO_SLOTS:-} CAMADAS=${CMP_CAMADAS_CPU:-} PID=$$" >> "$CMP_FAKE_LOG"
[ "${CMP_FAKE_RECUSA:-}" = "$chave" ] && { echo recusa >&2; exit 2; }
[ -z "${CMP_FAKE_MUDA:-}" ] || [ "$chave" != "${CMP_FAKE_MUDA_NA:-}" ] || echo mudou >> "$CMP_FAKE_MUDA"
custo=$(( (n + 1) * 10 * ${CMP_FAKE_MULT:-1} ))
if [ -n "${CMP_FAKE_RELOGIO:-}" ]; then echo $(( $(cat "$CMP_FAKE_RELOGIO") + custo )) > "$CMP_FAKE_RELOGIO"; fi
[ -z "${CMP_FAKE_SLEEP:-}" ] || sleep "$CMP_FAKE_SLEEP"
mkdir -p "$saida"
printf '{"lado": "lastro", "serie": "%s", "modelo": "%s", "texto": "%s", "estado": "%s", "n_medidas": %s, "portao_ok": true, "portao_motivo": "", "rc": 0}\n' "$serie" "$modelo" "$texto" "$estado" "$n" > "$saida/celula.json"
exit 0
F
chmod +x "$D/lastro-falso.sh"
# estimativas dos testes: custo estimado por célula = o das células falsas, (n+1)*10 (5 medidas = 60 s, 3 medidas = 40 s); portão, carga, smoke e folga zerados
export CMP_EST_PORTAO=0 CMP_EST_CARGA_1B=0 CMP_EST_CARGA_3B=0 CMP_EST_QUENTE_1B=10 CMP_EST_QUENTE_3B=10 CMP_EST_SMOKE_1B=0 CMP_EST_SMOKE_3B=0
export NVIDIA_SMI=$D/smi LASTRO_SH_CMP=$D/lastro-falso.sh MODELO_1B=$D/m/1b MODELO_3B=$D/m/3b
export TEXTOS_5C=$D/textos REF_5C=$D/refs IDS_CMP=$D/ids.json CRITERIO_CMP=$D/crit.md CRITERIO_D1=$D/crit-d1.md CRITERIO_D2=$D/crit-d2.md CRITERIO_PERFIL=$D/crit-perfil.md
export SEM_SHA_FIXO=1 CMP_SEM_MEMORIA=1 CMP_FOLGA=0
interno() { # nome limite [modelo=1b] [env extras via ambiente]; define RC; relógio virtual zerado
  local n=$1 lim=$2 mod=${3:-1b}
  echo 0 > "$D/clock"; : > "$D/log-$n"
  PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID CMP_RELOGIO_ARQ=$D/clock CMP_FAKE_RELOGIO=$D/clock CMP_FAKE_LOG=$D/log-$n SMI_LOG=$D/smilog-$n \
    "$J" --interno --modelos "$mod" --saida "$D/si-$n" --bin "$D/pp" --limite "$lim" > "$D/outi-$n" 2>&1
  RC=$?
}
nceluas() { grep -c '^[0-9][0-9] lastro-' "$D/si-$1/ordem.txt"; }
cortes() { python3 -I -c "import json; print(len(json.load(open('$D/si-$1/orcamento.json'))['cortes']))"; }
ESPERADA_1B="lastro-1b-L10-curto-quente lastro-1b-L10-medio-quente lastro-1b-L10-longo-quente lastro-1b-L100-curto-quente lastro-1b-L100-medio-quente lastro-1b-L100-longo-quente lastro-1b-LC-curto-quente lastro-1b-LC-medio-quente lastro-1b-LC-longo-quente"

echo "== Parte 1: modo interno (relógio virtual)"
"$J" --interno --modelos 1b --saida "$D/si-sem" --bin "$D/pp" --limite 100 > "$D/outi-sem" 2>&1
confere "modo interno sem portão/UUID: recusa 2, nada criado" "[ $? = 2 ] && [ ! -e $D/si-sem/ordem.txt ]"
# A. orçamento folgado: nada cortado, ordem fixa, 5 medidas, séries com as variáveis certas
CMP_FAKE_SLEEP=0.4 interno A 100000
confere "A (folgado, 1B): sai 0, decisao.txt, 9 células quentes, nenhum corte" "[ $RC = 0 ] && [ -s $D/si-A/decisao.txt ] && [ \$(nceluas A) = 9 ] && [ \$(cortes A) = 0 ] && grep -q 'cortes de orçamento: nenhum' $D/si-A/decisao.txt"
confere "A: ordem das células = plano gravado = L10, L100, LC, cada uma com curto, medio, longo" "[ \"\$(grep -o 'lastro-[^ ]*' $D/si-A/ordem.txt | tr '\n' ' ' | sed 's/ \$//')\" = \"$ESPERADA_1B\" ] && [ \"\$(grep -o 'lastro-[^ ]*' $D/si-A/plano.txt | tr '\n' ' ' | sed 's/ \$//')\" = \"$ESPERADA_1B\" ]"
confere "A: ambiente.txt registra power.limit, largura e geração do PCIe (nvidia-smi -i UUID)" "grep -qx 'gpu_limites=370.00 W, 16, 4' $D/si-A/ambiente.txt && grep -q -- '-i GPU-b22d9465-8552-9f97-812e-380b590645f8 --query-gpu=power.limit,pcie.link.width.current,pcie.link.gen.max' $D/smilog-A"
confere "A: smoke L100 e depois smoke LC, ANTES de qualquer célula (ordem.txt)" "[ \"\$(sed -n 1p $D/si-A/ordem.txt | cut -d' ' -f1)\" = smoke-L100 ] && [ \"\$(sed -n 2p $D/si-A/ordem.txt | cut -d' ' -f1)\" = smoke-LC ] && [ \"\$(sed -n 3p $D/si-A/ordem.txt | cut -d' ' -f2)\" = lastro-1b-L10-curto-quente ]"
confere "A: smokes com CMP_FRACAO_SLOTS=1.0 (L100) e CMP_CAMADAS_CPU=22 (LC), um só por série" "[ \"\$(grep -c '^SMOKE' $D/log-A)\" = 2 ] && grep -q '^SMOKE 1b FRAC=1.0 CAMADAS=\$' $D/log-A && grep -q '^SMOKE 1b FRAC= CAMADAS=22\$' $D/log-A && grep -q 'veredito=OK' $D/si-A/smoke-L100/veredito.txt && grep -q 'veredito=OK' $D/si-A/smoke-LC/veredito.txt"
confere "A: L10 sem variável de série; L100 com FRACAO 1.0; LC com CAMADAS 22; só --estado quente --n 5 (nenhuma célula fria)" "grep -c 'L10-.* FRAC= CAMADAS= ' $D/log-A | grep -q '^3\$' && grep -c 'L100-.* FRAC=1.0 CAMADAS= ' $D/log-A | grep -q '^3\$' && grep -c 'LC-.* FRAC= CAMADAS=22 ' $D/log-A | grep -q '^3\$' && [ \$(grep -c ' n=5 ' $D/log-A) = 9 ] && ! grep -q 'frio' $D/si-A/ordem.txt"
confere "A: pastas de célula novas (uma por célula) e integridade antes/depois gravadas e iguais; orcamento.json com o modelo e reps finais 5" "[ \$(ls $D/si-A/celulas | wc -l) = 9 ] && cmp -s $D/si-A/integridade-antes.txt $D/si-A/integridade-depois.txt && grep -q '^safetensors-1b ' $D/si-A/integridade-antes.txt && ! grep -q 'safetensors-3b\|config-3b' $D/si-A/integridade-antes.txt && grep -q '\"modelos\": \[\"1b\"\]' $D/si-A/orcamento.json && grep -q '\"reps_finais\": {\"quente\": 5}' $D/si-A/orcamento.json"
confere "A: P4 por célula: arquivos .gpu.csv e .cpu.csv de cada uma das 9 células; nvidia-smi chamado com -i <UUID da 3080 Ti> --query-gpu=timestamp,utilization.gpu --format=csv,noheader -lms 100" "[ \$(ls $D/si-A/p4/*.gpu.csv | wc -l) = 9 ] && [ \$(ls $D/si-A/p4/*.cpu.csv | wc -l) = 9 ] && grep -q -- \"-i $UUID --query-gpu=timestamp,utilization.gpu --format=csv,noheader -lms 100\" $D/smilog-A && grep -q '12 %' $D/si-A/p4/lastro-1b-L10-curto-quente.gpu.csv && ! grep -q dmon $D/smilog-A"
confere "A: o extrator do perfil rodou (perfil-resultado.json e .md)" "[ -s $D/si-A/perfil-resultado.json ] && [ -s $D/si-A/perfil-resultado.md ] && grep -q 'perfil-extrai: células' $D/si-A/decisao.txt"
confere "A: nenhuma amostragem órfã de nvidia-smi ou do amostrador depois da janela" "! pgrep -f \"$D/smi -i $UUID\" > /dev/null && ! pgrep -f \"perfil-amostra-cpu.py --raiz\" > /dev/null"
confere "A: manifesto integridade-fixos.txt gravado de uma vez (sem .tmp), uma entrada por chave congelada; com SEM_SHA_FIXO=1 o extrator o recusa (NAO-CONFERIDO) e nenhuma previsão sai" "[ ! -e $D/si-A/.integridade-fixos.tmp ] && [ \$(( \$(grep -c 'NAO-CONFERIDO\$' $D/si-A/integridade-fixos.txt) + 1 )) = \$(wc -l < $D/si-A/integridade-fixos.txt) ] && grep -qx 'potencia 370.00_W 370.00_W OK' $D/si-A/integridade-fixos.txt && grep -q '^criterio-perfil ' $D/si-A/integridade-fixos.txt && grep -q '^tokenizer-1b ' $D/si-A/integridade-fixos.txt && python3 -I -c \"import json; r=json.load(open('$D/si-A/perfil-resultado.json')); assert any('NAO-CONFERIDO' in x for x in r['integridade_problemas']); assert all(c['status']=='SEM PROVA' for c in r['celulas'].values()); assert all(s['previsoes']['P1']['resultado'].startswith('SEM PROVA') for s in r['series'].values())\""
# A3. janela do 3B: camadas 29
interno A3 100000 3b
confere "A3 (3B): células lastro-3b-..., smoke e LC com CMP_CAMADAS_CPU=29, safetensors do 3B e não os do 1B" "[ \$(nceluas A3) = 9 ] && grep -q '^SMOKE 3b FRAC= CAMADAS=29\$' $D/log-A3 && grep -c 'LC-.* CAMADAS=29 ' $D/log-A3 | grep -q '^3\$' && grep -q '^safetensors-3b-2 ' $D/si-A3/integridade-antes.txt && ! grep -q '^safetensors-1b' $D/si-A3/integridade-antes.txt"
# B. cortes: o ÚNICO corte é quente 5 -> 3; o que não couber é NÃO RODOU
# power.limit ao fim (Emenda 2/10): a janela roda, mas o limite mudou ou não pôde ser lido = integridade QUEBRADA (SEM PROVA), causa no manifesto, em ambiente.txt, na decisão e no extrator
rm -f "$D/pot-seq"; interno PE 100000
confere "PE (370 W estáveis): válido, manifesto com potencia OK, ambiente com potencia_inicio e potencia_fim, integridade íntegra" "[ $RC = 0 ] && grep -qx 'potencia 370.00_W 370.00_W OK' $D/si-PE/integridade-fixos.txt && grep -qx 'potencia_inicio=370.00_W' $D/si-PE/ambiente.txt && grep -qx 'potencia_fim=370.00_W' $D/si-PE/ambiente.txt && grep -q 'integridade: íntegra' $D/si-PE/decisao.txt"
printf '370.00 W\n300.00 W\n' > "$D/pot-seq"; interno PM 100000
confere "PM (370 no início, 300 ao fim): a janela roda, manifesto com FALHA, ambiente registra início e fim, decisão QUEBRADA citando power.limit, extrator reporta a causa" "[ $RC = 0 ] && grep -qx 'potencia inicio=370.00_W fim=300.00_W FALHA' $D/si-PM/integridade-fixos.txt && grep -qx 'potencia_fim=300.00_W' $D/si-PM/ambiente.txt && grep -q 'integridade: QUEBRADA' $D/si-PM/decisao.txt && grep -q 'power.limit' $D/si-PM/decisao.txt && grep -q 'potencia inicio=370.00_W fim=300.00_W FALHA\|power.limit fora de 370.00 W' $D/si-PM/perfil-resultado.json"
printf '370.00 W\nFALHA\n' > "$D/pot-seq"; interno PF 100000
confere "PF (leitura ausente ao fim): a janela roda, integridade QUEBRADA, fim ilegivel no ambiente e no manifesto, extrator reporta a causa" "[ $RC = 0 ] && grep -qx 'potencia inicio=370.00_W fim=ilegivel FALHA' $D/si-PF/integridade-fixos.txt && grep -qx 'potencia_fim=ilegivel' $D/si-PF/ambiente.txt && grep -q 'integridade: QUEBRADA' $D/si-PF/decisao.txt && grep -q 'ilegivel' $D/si-PF/perfil-resultado.json"
printf '370.00 W\nN/A\n' > "$D/pot-seq"; interno PI 100000
confere "PI (leitura ilegível 'N/A' ao fim): integridade QUEBRADA" "[ $RC = 0 ] && grep -q 'integridade: QUEBRADA' $D/si-PI/decisao.txt && grep -q 'FALHA' $D/si-PI/integridade-fixos.txt"
rm -f "$D/pot-seq"
interno B 540
confere "B (limite 540 = exatamente o nominal 9x60): sem corte, 9 células com n=5" "[ \$(cortes B) = 0 ] && [ \$(nceluas B) = 9 ] && [ \$(grep -c ' n=5 ' $D/log-B) = 9 ]"
interno C 539
confere "C (limite 539, nominal estoura por 1 s): corte quente 5->3 antes da 1ª célula; 9 células com n=3 (nada mais é cortado)" "[ \$(cortes C) = 1 ] && grep -q 'quente cai de 5 para 3' $D/si-C/orcamento.json && [ \$(nceluas C) = 9 ] && [ \$(grep -c ' n=3 ' $D/log-C) = 9 ] && [ \$(grep -c 'NÃO RODOU' $D/si-C/orcamento-log.txt) = 0 ]"
interno D 300
confere "D (limite 300): corte 5->3; cabem 7 células (40 s cada), as 2 últimas (LC medio e longo) ficam NÃO RODOU, nunca estimadas" "[ \$(cortes D) = 1 ] && [ \$(nceluas D) = 7 ] && [ \"\$(cat $D/clock)\" = 280 ] && grep -q 'NÃO RODOU: lastro-1b-LC-medio-quente' $D/si-D/orcamento-log.txt && grep -q 'NÃO RODOU: lastro-1b-LC-longo-quente' $D/si-D/orcamento-log.txt && [ ! -e $D/si-D/celulas/lastro-1b-LC-longo-quente ]"
confere "D: o extrator lista as células ausentes como NÃO RODOU, sem ms/posição" "python3 -I -c \"import json; r=json.load(open('$D/si-D/perfil-resultado.json')); c=r['celulas']['lastro-1b-LC-longo-quente']; assert c['status']=='NÃO RODOU' and c.get('ms_por_posicao') is None; assert r['series']['LC']['status']!='OK'\""
interno E 30
confere "E (limite 30, nem 1 célula de 40 s cabe): 0 células, tudo NÃO RODOU, decisão gravada" "[ \$(nceluas E) = 0 ] && [ \$(grep -c 'NÃO RODOU: lastro' $D/si-E/orcamento-log.txt) = 9 ] && [ -s $D/si-E/decisao.txt ]"
# F. células 2x mais lentas que o estimado: corte dinâmico pelo relógio no meio da janela
CMP_FAKE_MULT=2 interno F 700
confere "F (células 2x mais lentas, limite 700): corte no meio da janela (tempo > 0), as primeiras células com n=5 e as seguintes com n=3" "[ \$(cortes F) = 1 ] && grep -q '^corte em [1-9][0-9]* s: quente cai' $D/si-F/orcamento-log.txt && sed -n 3p $D/si-F/ordem.txt | grep -q 'n=5' && grep -q ' n=3 ' $D/si-F/ordem.txt"
# G. smoke por série: L100 falha por memória/CPU (código 0) e LC passa: SÓ L100 fica SEM PROVA, a janela segue com L10 e LC
FAKE_SMOKE_FALHA=L100 interno G 100000
confere "G (smoke L100 falha, código 0): um smoke de cada série, nenhuma célula L100, 3 células L10 e 3 LC, decisão com 'L100 NÃO SUSTENTA 100%'" "[ \$(nceluas G) = 6 ] && [ \$(grep -c '^SMOKE' $D/log-G) = 2 ] && [ \$(ls $D/si-G/celulas | grep -c L100) = 0 ] && [ \$(ls $D/si-G/celulas | grep -c -- '-L10-') = 3 ] && [ \$(ls $D/si-G/celulas | grep -c -- '-LC-') = 3 ] && grep -q 'L100 NÃO SUSTENTA 100%' $D/si-G/decisao.txt && grep -q 'SEM PROVA, série L100' $D/si-G/orcamento-log.txt && ! grep -q 'NÃO RODOU: lastro-1b-L100' $D/si-G/orcamento-log.txt && [ -s $D/si-G/perfil-resultado.json ]"
confere "G: sem nova tentativa (um único SMOKE L100 e nenhuma célula L100 no log) e orçamento sem cortes por causa da série perdida" "[ \$(grep -c 'FRAC=1.0' $D/log-G) = 1 ] && [ \$(cortes G) = 0 ]"
confere "G: o extrator marca L100 como SEM PROVA (não NÃO RODOU) com o motivo do smoke; L10 e LC seguem OK ou conforme as células" "python3 -I -c \"import json; r=json.load(open('$D/si-G/perfil-resultado.json')); assert r['smokes']['L100']['ok'] is False; assert r['series']['L100']['status']=='SEM PROVA'; assert all(r['celulas']['lastro-1b-L100-%s-quente'%t]['status']=='SEM PROVA' and r['celulas']['lastro-1b-L100-%s-quente'%t]['motivos'] for t in ('curto','medio','longo')); assert r['series']['L10']['status']!='NÃO RODOU' and r['series']['LC']['status']!='NÃO RODOU'\""
FAKE_SMOKE_FALHA=LC interno H 100000
confere "H (smoke LC falha, L100 ok): só LC fica SEM PROVA; 3 células L10 e 3 L100 rodam, decisão cita o smoke de camadas-CPU" "[ \$(nceluas H) = 6 ] && [ \$(grep -c '^SMOKE' $D/log-H) = 2 ] && grep -q 'SMOKE DE CAMADAS-CPU FALHOU' $D/si-H/decisao.txt && [ \$(ls $D/si-H/celulas | grep -c -- '-LC-') = 0 ] && python3 -I -c \"import json; r=json.load(open('$D/si-H/perfil-resultado.json')); assert r['series']['LC']['status']=='SEM PROVA' and r['smokes']['LC']['ok'] is False\""
# G2. código 2 num smoke (recusa de ambiente) interrompe a janela INTEIRA
FAKE_SMOKE_RC=2 interno G2 100000
confere "G2 (smoke recusa o ambiente, código 2): o smoke LC nem roda, nenhuma célula, a janela inteira NÃO RODOU, decisão gravada" "[ \$(nceluas G2) = 0 ] && [ \$(grep -c '^SMOKE' $D/log-G2) = 1 ] && [ \$(ls $D/si-G2/celulas | wc -l) = 0 ] && grep -q 'recusou o ambiente' $D/si-G2/orcamento-log.txt && grep -q 'NÃO RODOU: lastro-1b-LC-longo-quente' $D/si-G2/orcamento-log.txt && [ -s $D/si-G2/decisao.txt ]"
# I. recusa de ambiente (código 2) numa célula interrompe a janela
CMP_FAKE_RECUSA=lastro-1b-L100-curto-quente interno I 100000
confere "célula que recusa o ambiente (2): janela interrompida na 4ª célula, as seguintes NÃO RODOU, decisão gravada" "[ \$(nceluas I) = 4 ] && grep -q 'recusou o ambiente' $D/si-I/orcamento-log.txt && grep -q 'NÃO RODOU: lastro-1b-LC-longo-quente' $D/si-I/orcamento-log.txt && [ -s $D/si-I/decisao.txt ]"
# J. integridade: arquivo alterado durante a janela; sha diferente do congelado
CMP_FAKE_MUDA=$D/textos/texto-curto.txt CMP_FAKE_MUDA_NA=lastro-1b-L10-curto-quente interno K 100000
confere "texto alterado durante a janela: integridade QUEBRADA na decisão e no relatório do perfil" "grep -q 'integridade: QUEBRADA' $D/si-K/decisao.txt && grep -q 'sha mudou entre antes e depois' $D/si-K/decisao.txt && grep -q 'Integridade QUEBRADA' $D/si-K/perfil-resultado.md"
echo c > "$D/textos/texto-curto.txt"
SEM_SHA_FIXO=0 interno L 100000
confere "sem SEM_SHA_FIXO, arquivos falsos não batem com o congelado: QUEBRADA e a nota diz qual" "grep -q 'não o congelado' $D/si-L/decisao.txt && grep -q 'FALHA' $D/si-L/integridade-fixos.txt"
# --modelos: obrigatório e validado no modo interno (recusa 2, nada roda)
for casom in "" "5b" "1b,3b" "ambos"; do
  : > "$D/log-m"
  if [ -z "$casom" ]; then args=(); else args=(--modelos "$casom"); fi
  PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID CMP_FAKE_LOG=$D/log-m "$J" --interno "${args[@]}" --saida "$D/si-m-$casom" --bin "$D/pp" --limite 100 > "$D/outi-m" 2>&1
  rcm=$?
  confere "--modelos '${casom:-ausente}' no modo interno: recusa 2, nada rodou" "[ $rcm = 2 ] && [ ! -s $D/log-m ] && [ ! -e '$D/si-m-$casom/ordem.txt' ]"
done
# amostrador de %CPU: acha o processo filho pelo nome e grava epoch_ms,pid,cpu_pct (um 'sleep' de 2 s, sem carga)
bash -c 'sleep 2 & wait' & PAI=$!
sleep 0.3
timeout 1 python3 -I "$AQUI/perfil-amostra-cpu.py" --raiz "$PAI" --comm sleep --intervalo 0.1 --saida "$D/cpu-amostra.csv"
wait "$PAI"
confere "perfil-amostra-cpu.py: >= 3 amostras 'epoch_ms,pid,cpu_pct' do processo filho (comm sleep), pct numérico e >= 0" "[ \$(wc -l < $D/cpu-amostra.csv) -ge 3 ] && python3 -I -c \"
import sys
for l in open('$D/cpu-amostra.csv'):
    a = l.strip().split(','); assert len(a) == 3 and float(a[0]) > 1e12 and int(a[1]) > 0 and float(a[2]) >= 0, l
\""

echo "== Parte 2: envelope (portões, pastas novas, unidades transitórias, aviso, limite, --verifica-fixos)"
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
export CMP_MODO_TESTE=1     # modo de teste explícito: sem ele o envelope recusa as variáveis só de teste (testado abaixo com ambiente limpo)
unset CMP_RELOGIO_ARQ CMP_FAKE_RELOGIO
roda() { # nome args...; define RC
  local n=$1; shift
  export FAKE_DIR=$D/$n CMP_FAKE_LOG=$D/elog-$n; : > "$CMP_FAKE_LOG"
  mkdir -p "$FAKE_DIR"
  echo active > "$FAKE_DIR/a.timer"; echo active > "$FAKE_DIR/b.timer"; echo inactive > "$FAKE_DIR/a.service"; echo inactive > "$FAKE_DIR/b.service"; : > "$FAKE_DIR/log"
  SYSTEMCTL="$D/fake-systemctl" PENDENTE="$D/pend-$n" JANELA_PASSO_ESPERA=0.1 XDG_RUNTIME_DIR=/run/user/fake JANELA_TIMERS="$D/timers.txt" CMP_AVISO_CMD="$D/aviso-cmd" \
    "$J" --modelos "${MOD-1b}" --bin "$D/pp" --saida "$D/se-$n" --limite "${LIM:-60}" "$@" > "$D/oute-$n" 2>&1
  RC=$?
}
roda sem-portao; confere "sem portão: recusa 2, nada rodou, timers intocados" "[ $RC = 2 ] && [ ! -s $D/elog-sem-portao ] && ! grep -q '^stop' $D/sem-portao/log && [ ! -e $D/se-sem-portao ]"
export PRIMEIRA_PALAVRA_AUTORIZADA=1
roda sem-uuid; confere "sem UUID: recusa 2" "[ $RC = 2 ] && [ ! -s $D/elog-sem-uuid ]"
export CUDA_VISIBLE_DEVICES=GPU-00000000-0000-0000-0000-00000000f090
roda uuid-5090; confere "UUID de outra GPU (a 5090 nunca entra): recusa 2" "[ $RC = 2 ] && [ ! -s $D/elog-uuid-5090 ] && ! grep -q '^stop' $D/uuid-5090/log"
export CUDA_VISIBLE_DEVICES=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd
roda uuid-4090; confere "UUID da 4090 (placa antiga): recusa 2" "[ $RC = 2 ] && [ ! -s $D/elog-uuid-4090 ] && ! grep -q '^stop' $D/uuid-4090/log"
export CUDA_VISIBLE_DEVICES=$UUID
export PRIMEIRA_PALAVRA_AUTORIZADA=0
roda portao-zero; confere "portão diferente de 1: recusa 2" "[ $RC = 2 ] && [ ! -s $D/elog-portao-zero ]"
export PRIMEIRA_PALAVRA_AUTORIZADA=1
# power.limit (Emenda 2/10): diferente de 370.00 W no preflight = recusa 2 antes de tocar em timers; leitura ilegível ou ausente também
printf '300.00 W\n' > "$D/pot-seq"; roda pot-300; confere "power.limit 300 W no preflight: recusa 2 citando o limite, nada tocado (timers, pasta, células)" "[ $RC = 2 ] && grep -q '370.00 W' $D/oute-pot-300 && grep -q \"'300.00 W'\" $D/oute-pot-300 && [ ! -s $D/elog-pot-300 ] && ! grep -q '^stop' $D/pot-300/log && [ ! -e $D/se-pot-300 ] && [ ! -e $D/se-pot-300.timers-antes.txt ]"
printf 'FALHA\n' > "$D/pot-seq"; roda pot-falha; confere "power.limit ilegível/ausente no preflight (nvidia-smi falha): recusa 2, nada tocado" "[ $RC = 2 ] && grep -q 'ilegível' $D/oute-pot-falha && [ ! -s $D/elog-pot-falha ] && ! grep -q '^stop' $D/pot-falha/log && [ ! -e $D/se-pot-falha ]"
rm -f "$D/pot-seq"
mkdir -p "$D/se-ocupada"; echo x > "$D/se-ocupada/a"
roda ocupada; confere "--saida não vazia: recusa 2" "[ $RC = 2 ] && [ ! -s $D/elog-ocupada ]"
# GPU ocupada: voxtype e Chrome (C+G) abertos na 3080 Ti = recusa 2 ANTES de tocar em timers, com a lista; só G passa
printf '# gpu pid type sm mem enc dec jpg ofa fb ccpm command\n    0 1352017   C+G      -      -      -      -      -      -    997      0    voxtype\n    0   42406   C+G      -      -      -      -      -      -    364      0    renderD128 --cr\n    0   41533     G     25     25      -      -      -      -    413      0    Hyprland\n' > "$D/pmon-cg.txt"
FAKE_PMON_ARQ=$D/pmon-cg.txt roda cg-aberto; confere "GPU ocupada (voxtype/Chrome C+G): recusa 2 antes de tocar em timers, listando PIDs e comandos e mandando fechar" "[ $RC = 2 ] && grep -q 'FECHADOS' $D/oute-cg-aberto && grep -q '1352017 (voxtype)' $D/oute-cg-aberto && grep -q '42406 (renderD128)' $D/oute-cg-aberto && [ ! -s $D/elog-cg-aberto ] && ! grep -q '^stop' $D/cg-aberto/log && [ ! -e $D/se-cg-aberto ] && [ ! -e $D/se-cg-aberto.timers-antes.txt ]"
roda limite-alto --limite 1201; confere "limite acima de 1200: recusa 2" "[ $RC = 2 ] && [ ! -s $D/elog-limite-alto ]"
roda limite-zero --limite 0; confere "limite 0: recusa 2" "[ $RC = 2 ] && [ ! -s $D/elog-limite-zero ]"
roda opcao-ruim --nao-existe; confere "opção desconhecida: recusa 2" "[ $RC = 2 ] && [ ! -s $D/elog-opcao-ruim ]"
roda opcao-degrau --degrau1; confere "opção de outro roteiro (--degrau1): recusa 2" "[ $RC = 2 ] && [ ! -s $D/elog-opcao-degrau ]"
MOD="" roda sem-modelos; confere "envelope sem --modelos: recusa 2, nada rodou, timers intocados, pasta não criada" "[ $RC = 2 ] && grep -q 'modelos' $D/oute-sem-modelos && [ ! -s $D/elog-sem-modelos ] && ! grep -q '^stop' $D/sem-modelos/log && [ ! -e $D/se-sem-modelos ]"
unset MOD
for mm in 5b ambos 1b,3b; do MOD=$mm roda mod-ruim; confere "envelope com --modelos '$mm' (as duas de uma vez nunca): recusa 2" "[ $RC = 2 ] && [ ! -s $D/elog-mod-ruim ] && [ ! -e $D/se-mod-ruim ] && ! grep -q '^stop' $D/mod-ruim/log"; done
unset MOD
# variáveis só de teste no modo real: ambiente LIMPO (env -i) com a variável e SEM CMP_MODO_TESTE = recusa 2, nada criado
real() { # nome VAR=valor...
  local n=$1; shift
  env -i PATH="$PATH" HOME="$HOME" PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID "$@" "$J" --modelos 1b --bin "$D/pp" --saida "$D/se-real-$n" --limite 60 > "$D/oute-real-$n" 2>&1
  RC=$?
}
for v in CMP_FRACAO_SLOTS CMP_CAMADAS_CPU CRITERIO_PERFIL CRITERIO_D1 CRITERIO_D2 SEM_SHA_FIXO MONITOR_SH LASTRO_SH_CMP EXTRAI_PY PERFIL_EXTRAI_PY PERFIL_AMOSTRA_PY CMP_RELOGIO_ARQ NVIDIA_SMI MODELO_1B MODELO_3B TEXTOS_5C REF_5C IDS_CMP CRITERIO_CMP \
         PROC_STAT CMP_CPU_JANELA CMP_CPU_ESPERA CMP_FOLGA CMP_SEM_MEMORIA ESPERA_PRONTO LOADAVG_ARQ SYSTEMCTL SYSTEMD_RUN PENDENTE CGROUP_RAIZ \
         CMP_EST_QUENTE_1B CMP_FAKE_RECUSA JANELA_SH JANELA_TIMERS JANELA_PASSO_ESPERA LIMITE_SH LIMITE_PASSA; do
  real "v-$v" "$v=1"
  confere "modo real com $v no ambiente: recusa 2 citando a variável, nada criado" "[ $RC = 2 ] && grep -q 'variável só de teste presente no modo real: $v' $D/oute-real-v-$v && [ ! -e $D/se-real-v-$v ]"
done
mkdir -p "$D/se-real-limpo"; echo x > "$D/se-real-limpo/a"
real limpo
confere "ambiente limpo: a checagem de variáveis de teste passa (a recusa seguinte é a da pasta ocupada)" "[ $RC = 2 ] && ! grep -q 'variável só de teste' $D/oute-real-limpo && grep -q 'não está vazio' $D/oute-real-limpo"
real com-modo-teste CMP_MODO_TESTE=1 SEM_SHA_FIXO=1
confere "CMP_MODO_TESTE=1 (modo de teste explícito) aceita a variável de teste: passa da checagem" "! grep -q 'variável só de teste' $D/oute-real-com-modo-teste"
# --verifica-fixos: ambiente limpo, só leitura, contra os arquivos REAIS
env -i PATH="$PATH" HOME="$HOME" "$J" --verifica-fixos > "$D/fixos.txt" 2>&1; rcf=$?
nok=$(grep -c ' OK$' "$D/fixos.txt"); nfal=$(grep -c ' FALHA$' "$D/fixos.txt"); naus=$(grep -c ' AUSENTE ' "$D/fixos.txt")
echo "congelados do FIXO contra os arquivos reais: $nok OK, $nfal FALHA, $naus AUSENTE (código $rcf)"
confere "--verifica-fixos: código 0 só com tudo OK (nenhum FALHA nem AUSENTE) e os itens essenciais presentes" "[ $rcf = 0 ] && [ $nfal = 0 ] && [ $naus = 0 ] && [ $nok -ge 20 ] && grep -q '^criterio-perfil fd9b6190.*4e9e OK\$' $D/fixos.txt && grep -q '^binario-lastro .* OK\$' $D/fixos.txt && grep -q '^script-perfil-extrai .* OK\$' $D/fixos.txt && grep -q '^script-lastro .* OK\$' $D/fixos.txt && grep -q '^texto-longo .* OK\$' $D/fixos.txt && grep -q '^ref5c-3b .* OK\$' $D/fixos.txt && grep -q '^tokenizer-3b .* OK\$' $D/fixos.txt"
confere "--verifica-fixos: o sha congelado do critério do perfil é o do arquivo atual (sha256sum direto)" "[ \"\$(grep '^criterio-perfil ' $D/fixos.txt | cut -d' ' -f2)\" = \"\$(sha256sum $AQUI/criterio-perfil-prefill-20261109.md | cut -d' ' -f1)\" ]"
# arquivo de integridade alterado: FALHA e código 1; arquivo ausente: AUSENTE e código 1
echo alterado > "$D/crit-perfil-alterado.md"
env -i PATH="$PATH" HOME="$HOME" CRITERIO_PERFIL="$D/crit-perfil-alterado.md" "$J" --verifica-fixos > "$D/fixos-ruim.txt" 2>&1; rcr=$?
confere "--verifica-fixos com o critério alterado: FALHA e código 1" "[ $rcr = 1 ] && grep -q '^criterio-perfil .* FALHA\$' $D/fixos-ruim.txt"
env -i PATH="$PATH" HOME="$HOME" PERFIL_EXTRAI_PY="$D/nao-existe.py" "$J" --verifica-fixos > "$D/fixos-aus.txt" 2>&1; rca=$?
confere "--verifica-fixos com um script ausente: AUSENTE e código 1" "[ $rca = 1 ] && grep -q '^script-perfil-extrai .* AUSENTE' $D/fixos-aus.txt"
# caminho feliz no envelope (relógio real; células falsas instantâneas)
unset CMP_FAKE_RELOGIO
LIM=1200 roda ok
confere "ok: sai 0, decisao.txt, 9 células do 1B, timers parados e restaurados, perfil-resultado" "[ $RC = 0 ] && [ -s $D/se-ok/decisao.txt ] && [ \$(ls $D/se-ok/celulas | wc -l) = 9 ] && ! ls $D/se-ok/celulas | grep -q 3b && [ -s $D/se-ok/perfil-resultado.json ] && grep -q '^stop a.timer' $D/ok/log && [ \"\$(cat $D/ok/a.timer)\" = active ] && grep -q 'RESTAURAÇÃO OK' $D/se-ok-estado-janela/restauracao.txt"
confere "ok: decisão descritiva (sem PASSA/REPROVA), íntegra, sem cortes; cargas e listas systemd ao lado da pasta" "grep -q 'resultado DESCRITIVO' $D/se-ok/decisao.txt && grep -q 'integridade: íntegra' $D/se-ok/decisao.txt && grep -q 'cortes de orçamento: nenhum' $D/se-ok/decisao.txt && [ -f $D/se-ok.carga-antes.txt ] && [ -f $D/se-ok.carga-depois.txt ] && [ -s $D/se-ok.unidades-antes.txt ] && [ -s $D/se-ok.timers-depois.txt ]"
confere "ok: aviso no fim da janela (arquivo e comando de aviso)" "grep -q 'terminou (código 0)' $D/se-ok.aviso.txt && grep -q 'terminou (código 0)' $D/aviso-chamado"
# limite de tempo estourado: células lentas; 124, timers restaurados, aviso mesmo assim
rm -f "$D/aviso-chamado"
for v in $(compgen -v | grep '^CMP_EST_'); do export "$v=0"; done
CMP_FAKE_SLEEP=40 LIM=3 roda lim
confere "limite estourado: 124, sem decisão, timers restaurados, aviso dado" "[ $RC = 124 ] && [ ! -f $D/se-lim/decisao.txt ] && [ \"\$(cat $D/lim/a.timer)\" = active ] && grep -q 'terminou (código 124)' $D/aviso-chamado"
# composição com o binário REAL (sem GPU): o portão do próprio primeira-palavra recusa a 4090 ANTES de abrir device ou criar a pasta de saída (nunca roda com o UUID da 3080 Ti aqui: abriria a GPU)
BINREAL=$AQUI/../../rust/moe-rt/target/release/primeira-palavra
if [ -x "$BINREAL" ]; then
  env -u CMP_MODO_TESTE PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd "$BINREAL" --gera --celula 1b:/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct --so-texto curto --saida "$D/saida-binario-real" > "$D/out-binreal" 2>&1
  rcb=$?
  confere "binário real: com a UUID da 4090 recusa com código 2 citando a 3080 Ti, sem criar a pasta de saída" "[ $rcb = 2 ] && grep -q '3080 Ti' $D/out-binreal && [ ! -e $D/saida-binario-real ]"
else
  echo "FALHOU: binário real ausente em $BINREAL (cargo build --offline --release --bin primeira-palavra)"; falhas=$((falhas+1))
fi
if [ "$falhas" = 0 ]; then echo "TODOS OS TESTES PASSARAM"; exit 0; else echo "$falhas FALHA(S)"; exit 1; fi
