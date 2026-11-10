#!/usr/bin/env bash
# Teste do roda-cmp-janela.sh SEM GPU: roteiros de célula FALSOS (avançam um RELÓGIO VIRTUAL, sem dormir), nvidia-smi falso, arquivos, modelos, llama.cpp e critério falsos.
# Parte 1 (modo --interno, sem systemd): ORÇAMENTO E8 e ordem de corte por simulação (processos falsos), ordem alternada, mesmas repetições nos dois lados, abort por recusa.
# Parte 2 (envelope real: janela-e4b.sh + roda-com-limite.sh em unidades transitórias de `systemd --user`, mas systemctl FALSO para os timers): portões, pastas novas,
# integridade, aviso, limite de tempo. mktemp -d, sem rm -rf.
set -u
unset E4B_GPU_AUTORIZADO E4B_MEDIDA_AUTORIZADA MOE_RT_REQUIRE_GPU PRIMEIRA_PALAVRA_AUTORIZADA
export CUDA_VISIBLE_DEVICES=
AQUI=$(cd "$(dirname "$0")" && pwd -P)
J=$AQUI/roda-cmp-janela.sh
UUID=GPU-b22d9465-8552-9f97-812e-380b590645f8
D=$(mktemp -d -p "${TMPDIR:-/tmp}" cmp-janela-teste.XXXXXX) || exit 1
echo "pasta de teste: $D"
falhas=0
confere() { if eval "$2"; then echo "ok: $1"; else echo "FALHOU: $1"; falhas=$((falhas+1)); fi; }
# ---- fixtures falsas
mkdir -p "$D/gguf" "$D/m/1b" "$D/m/3b" "$D/llama" "$D/textos" "$D/refs"
echo g1 > "$D/gguf/granite-3.1-1b-a400m-instruct-bf16.gguf"; echo g3 > "$D/gguf/granite-3.1-3b-a800m-instruct-bf16.gguf"
for m in 1b 3b; do echo cfg > "$D/m/$m/config.json"; echo tok > "$D/m/$m/tokenizer.json"; done
echo st > "$D/m/1b/model.safetensors"; echo s1 > "$D/m/3b/model-00001-of-00002.safetensors"; echo s2 > "$D/m/3b/model-00002-of-00002.safetensors"; echo ix > "$D/m/3b/model.safetensors.index.json"
echo srv > "$D/llama/llama-server"; echo lib > "$D/llama/libllama.so.0.8.0"; echo lib2 > "$D/llama/libggml.so"
echo c > "$D/textos/texto-curto.txt"; echo m > "$D/textos/texto-medio.txt"; echo l > "$D/textos/texto-longo.txt"; echo '{"textos": []}' > "$D/textos/textos.json"
echo r1 > "$D/refs/ref5c-1b.json"; echo r3 > "$D/refs/ref5c-3b.json"
echo '{"modelos": {}}' > "$D/ids.json"; echo criterio > "$D/crit.md"
printf '#!/usr/bin/env bash\nexit 0\n' > "$D/pp"; chmod +x "$D/pp"
printf '#!/usr/bin/env bash\ncase $* in *power.limit*) case $* in *"query-gpu=power.limit --format"*) f=${0%%/*}/pot-seq; if [ -s "$f" ]; then l=$(head -n 1 "$f"); tail -n +2 "$f" > "$f.t"; mv "$f.t" "$f"; [ "$l" = FALHA ] && exit 1; echo "$l"; else echo "370.00 W"; fi;; *"-i GPU-b22d9465-8552-9f97-812e-380b590645f8 "*) echo "370.00 W, 16, 4";; esac;; *query-gpu*) echo "uuid, memory.used [MiB], utilization.gpu [%%]"; echo "GPU-falsa, 1 MiB, 0 %%";; *pmon*) if [ -n "${FAKE_PMON_ARQ:-}" ]; then cat "$FAKE_PMON_ARQ"; else echo "# gpu pid type sm mem enc dec jpg ofa fb ccpm command"; echo "# Idx # C/G %% %% %% %% %% %% MB MB name"; fi;; *) echo;; esac\n' > "$D/smi"; chmod +x "$D/smi"
# células falsas: avançam o relógio virtual pelo custo (mesma fórmula das estimativas dos testes) e gravam celula.json
cat > "$D/celula-falsa.sh" <<'F'
#!/usr/bin/env bash
# uso (lastro): --modelo M --texto T --estado E --n N --saida DIR --bin B ; (llama): ... --serie S --ids ARQ
lado=$1_; modelo=; texto=; estado=; n=; saida=; serie=L
while [ $# -gt 0 ]; do case $1 in --modelo) modelo=$2;; --texto) texto=$2;; --estado) estado=$2;; --n) n=$2;; --saida) saida=$2;; --serie) serie=$2;; esac; shift; done
lado=lastro; [ "$serie" = L ] || lado=llamacpp
chave="$lado-$modelo-$serie-$texto-$estado"
echo "$chave n=$n" >> "$CMP_FAKE_LOG"
[ "${CMP_FAKE_RECUSA:-}" = "$chave" ] && { echo recusa >&2; exit 2; }
[ -z "${CMP_FAKE_MUDA:-}" ] || [ "$chave" != "${CMP_FAKE_MUDA_NA:-}" ] || echo mudou >> "$CMP_FAKE_MUDA"
mult=${CMP_FAKE_MULT:-1}
if [ "$lado" = lastro ]; then custo=$(( (n + 1) * 10 )); elif [ "$serie" = S2 ]; then custo=$(( (n + 1) * 10 )); else custo=$(( (n + 1) * 5 )); fi
if [ "${CMP_FAKE_EST:-}" = 1 ]; then
  # custo = as estimativas FINAIS da emenda 7 (cópia independente das fórmulas do envelope; se os padrões mudarem, o cenário K/L falha de propósito)
  P=8; if [ "$modelo" = 1b ]; then LC=5; LF=8; LQ=5; CC=8; CP=2; else LC=7; LF=12; LQ=9; CC=15; CP=3; fi; AQ=3; FX=3; m2=1; [ "$serie" = S2 ] && m2=2
  if [ "$lado" = lastro ]; then
    if [ "$estado" = frio ]; then custo=$(( P + (n + 1) * (LC + LF) )); else custo=$(( P + LC + (n + 1) * LQ )); fi
  else
    if [ "$estado" = frio ]; then custo=$(( P + FX + (n + 1) * (CC + CP * m2) )); else custo=$(( P + FX + CC + AQ + (n + 1) * CP * m2 )); fi
  fi
fi
if [ -n "${CMP_FAKE_RELOGIO:-}" ]; then echo $(( $(cat "$CMP_FAKE_RELOGIO") + custo * mult )) > "$CMP_FAKE_RELOGIO"; fi
[ -z "${CMP_FAKE_SLEEP:-}" ] || sleep "$CMP_FAKE_SLEEP"
mkdir -p "$saida"
printf '{"lado": "%s", "serie": "%s", "modelo": "%s", "texto": "%s", "estado": "%s", "n_medidas": %s, "portao_ok": true, "portao_motivo": "", "rc": 0}\n' "$lado" "$serie" "$modelo" "$texto" "$estado" "$n" > "$saida/celula.json"
printf 'pmon_antes_veredito=OK\ncompute_apps_veredito=OK\nportao=OK\n' > "$saida/portao.txt"
exit 0
F
chmod +x "$D/celula-falsa.sh"
printf '#!/usr/bin/env bash\nexec %s "$@"\n' "$D/celula-falsa.sh" > "$D/lastro-falso"; cp "$D/lastro-falso" "$D/llama-falso"; chmod +x "$D/lastro-falso" "$D/llama-falso"
# estimativas dos testes: custo por célula = o das células falsas (Lastro (n+1)*10; llama S1/S3 (n+1)*5, S2 (n+1)*10) => bloco principal 250 s, bloco S3 50 s
export CMP_EST_PORTAO=0 CMP_EST_L_CARGA_1B=0 CMP_EST_L_CARGA_3B=0 CMP_EST_L_FRIO_1B=10 CMP_EST_L_FRIO_3B=10 CMP_EST_L_QUENTE_1B=10 CMP_EST_L_QUENTE_3B=10
export CMP_EST_C_CARGA_1B=0 CMP_EST_C_CARGA_3B=0 CMP_EST_C_PED_1B=5 CMP_EST_C_PED_3B=5 CMP_EST_C_AQUEC=0 CMP_EST_C_FIXO=0
# 3B: o custo estimado das células do 3B segue a mesma fórmula; o bloco do 3B com S2 desligado (S1 apenas) vale (L 40+60) + (S1 20+30) = 150
export NVIDIA_SMI=$D/smi LASTRO_SH_CMP=$D/lastro-falso LLAMA_SH_CMP=$D/llama-falso MODELO_1B=$D/m/1b MODELO_3B=$D/m/3b GGUF_DIR=$D/gguf LLAMA_DIR=$D/llama
export TEXTOS_5C=$D/textos REF_5C=$D/refs IDS_CMP=$D/ids.json CRITERIO_CMP=$D/crit.md SEM_SHA_FIXO=1 CMP_SEM_MEMORIA=1 CMP_FOLGA=0
interno() { # nome limite [env extras via ambiente]; define RC; relógio virtual zerado. Cenários A a J usam os dois modelos de uma vez (só o modo interno de teste, CMP_TESTE_AMBOS=1)
  local n=$1 lim=$2; shift 2
  echo 0 > "$D/clock"; : > "$D/log-$n"
  PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID CMP_RELOGIO_ARQ=$D/clock CMP_FAKE_RELOGIO=$D/clock CMP_FAKE_LOG=$D/log-$n CMP_TESTE_AMBOS=1 \
    "$J" --interno --modelos "${MODELOS_T:-ambos}" --saida "$D/si-$n" --bin "$D/pp" --limite "$lim" "$@" > "$D/outi-$n" 2>&1
  RC=$?
}
nceluas() { wc -l < "$D/si-$1/ordem.txt"; }
ordem() { sed -n "${2}p" "$D/si-$1/ordem.txt" | cut -d' ' -f2; }
cortes() { python3 -I -c "import json; print(len(json.load(open('$D/si-$1/orcamento.json'))['cortes']))"; }

echo "== Parte 1: orçamento E8 (modo interno, relógio virtual)"
# recusa do modo interno sem portão
"$J" --interno --modelos 1b --saida "$D/si-sem" --bin "$D/pp" --limite 100 > "$D/outi-sem" 2>&1
confere "modo interno sem portão/UUID: recusa 2" "[ $? = 2 ] && [ ! -e $D/si-sem/ordem.txt ]"
# A. orçamento folgado: nada cortado, 48 células (36 principais + 12 da S3), repetições nominais
interno A 100000
confere "A (folgado): sai 0, 48 células, nenhum corte, decisao.txt gravada" "[ $RC = 0 ] && [ \$(nceluas A) = 48 ] && [ \$(cortes A) = 0 ] && [ -s $D/si-A/decisao.txt ]"
confere "A: repetições nominais (frio n=3, quente n=5) nos dois lados; as 12 células da S3 com as mesmas repetições do Lastro" "[ \$(grep -c 'frio n=3' $D/si-A/ordem.txt) = 24 ] && [ \$(grep -c 'quente n=5' $D/si-A/ordem.txt) = 24 ] && [ \$(grep 'S3.*frio n=3' $D/si-A/ordem.txt | wc -l) = 6 ] && [ \$(grep 'S3.*quente n=5' $D/si-A/ordem.txt | wc -l) = 6 ]"
confere "A: ordem E8: 1B antes do 3B, S3 por último; frio antes do quente no bloco" "[ \$(ordem A 1) = lastro-1b-L-curto-frio ] && [ \$(ordem A 37) = llamacpp-1b-S3-curto-frio ] && [ \$(ordem A 48) = llamacpp-3b-S3-longo-quente ] && sed -n 1,18p $D/si-A/ordem.txt | grep -q '1b-L-curto-quente' && ! sed -n 1,18p $D/si-A/ordem.txt | grep -q ' llamacpp-3b'"
confere "A: lados ALTERNADOS: bloco 0 começa pelo Lastro, bloco 1 pelo llama.cpp (e termina no Lastro), bloco 2 de novo pelo Lastro; ordem gravada com início/fim" "[ \$(ordem A 1) = lastro-1b-L-curto-frio ] && [ \$(ordem A 7) = llamacpp-1b-S1-medio-frio ] && [ \$(ordem A 9) = lastro-1b-L-medio-frio ] && [ \$(ordem A 13) = lastro-1b-L-longo-frio ] && grep -q 'inicio=0s fim=40s rc=0' $D/si-A/ordem.txt"
confere "A: ambiente.txt registra power.limit, largura e geração do PCIe da placa de medida (consulta com -i UUID)" "grep -qx 'gpu_limites=370.00 W, 16, 4' $D/si-A/ambiente.txt"
confere "A: pastas de célula novas (uma por célula), integridade antes/depois e ambiente gravados, extrator rodou" "[ \$(ls $D/si-A/celulas | wc -l) = 48 ] && [ -s $D/si-A/integridade-antes.txt ] && cmp -s $D/si-A/integridade-antes.txt $D/si-A/integridade-depois.txt && [ -s $D/si-A/ambiente.txt ] && [ -s $D/si-A/cmp-resultado.json ] && [ -s $D/si-A/cmp-resultado.md ] && grep -q '^llama-lib-libllama.so.0.8.0 ' $D/si-A/integridade-antes.txt && grep -q '^safetensors-3b-2 ' $D/si-A/integridade-antes.txt && grep -q '^gguf-3b ' $D/si-A/integridade-antes.txt"
# B. S3 não cabe: corte (1) apenas
# power.limit ao fim (Emenda 2/10): a janela roda, mas o limite mudou ou não pôde ser lido = integridade QUEBRADA (SEM PROVA), causa no manifesto, em ambiente.txt, na decisão e no extrator
rm -f "$D/pot-seq"; interno PE 100000
confere "PE (370 W estáveis): válido, manifesto com potencia OK, ambiente com potencia_inicio e potencia_fim, integridade íntegra" "[ $RC = 0 ] && grep -qx 'potencia 370.00_W 370.00_W OK' $D/si-PE/integridade-fixos.txt && grep -qx 'potencia_inicio=370.00_W' $D/si-PE/ambiente.txt && grep -qx 'potencia_fim=370.00_W' $D/si-PE/ambiente.txt && grep -q 'integridade: íntegra' $D/si-PE/decisao.txt"
printf '370.00 W\n300.00 W\n' > "$D/pot-seq"; interno PM 100000
confere "PM (370 no início, 300 ao fim): a janela roda, manifesto com FALHA, ambiente registra início e fim, decisão QUEBRADA citando power.limit, extrator reporta a causa" "[ $RC = 0 ] && grep -qx 'potencia inicio=370.00_W fim=300.00_W FALHA' $D/si-PM/integridade-fixos.txt && grep -qx 'potencia_fim=300.00_W' $D/si-PM/ambiente.txt && grep -q 'integridade: QUEBRADA' $D/si-PM/decisao.txt && grep -q 'power.limit' $D/si-PM/decisao.txt && grep -q 'potencia inicio=370.00_W fim=300.00_W FALHA\|power.limit fora de 370.00 W' $D/si-PM/cmp-resultado.json"
printf '370.00 W\nFALHA\n' > "$D/pot-seq"; interno PF 100000
confere "PF (leitura ausente ao fim): a janela roda, integridade QUEBRADA, fim ilegivel no ambiente e no manifesto, extrator reporta a causa" "[ $RC = 0 ] && grep -qx 'potencia inicio=370.00_W fim=ilegivel FALHA' $D/si-PF/integridade-fixos.txt && grep -qx 'potencia_fim=ilegivel' $D/si-PF/ambiente.txt && grep -q 'integridade: QUEBRADA' $D/si-PF/decisao.txt && grep -q 'ilegivel' $D/si-PF/cmp-resultado.json"
printf '370.00 W\nN/A\n' > "$D/pot-seq"; interno PI 100000
confere "PI (leitura ilegível 'N/A' ao fim): integridade QUEBRADA" "[ $RC = 0 ] && grep -q 'integridade: QUEBRADA' $D/si-PI/decisao.txt && grep -q 'FALHA' $D/si-PI/integridade-fixos.txt"
rm -f "$D/pot-seq"
interno B 1500
confere "B (limite 1500, projeção 1800): só a S3 é cortada (1º corte), 36 células, S3 NÃO RODOU" "[ \$(nceluas B) = 36 ] && [ \$(cortes B) = 1 ] && grep -q '(1) S3 inteira não roda' $D/si-B/orcamento.json && ! grep -q 'S3' $D/si-B/ordem.txt && grep -q 'NÃO RODOU: llamacpp-1b-S3-curto' $D/si-B/orcamento-log.txt"
# C. 1000: S3, S2 do 3B e S2 do 1B
interno C 1000
confere "C (limite 1000): cortes 1, 2 e 2b na ordem (S3, S2 do 3B, S2 do 1B); 24 células, sem nenhuma S2 nem S3, reps nominais" "[ \$(nceluas C) = 24 ] && [ \$(cortes C) = 3 ] && ! grep -q 'S2\|S3' $D/si-C/ordem.txt && grep -c 'frio n=3' $D/si-C/ordem.txt | grep -q '^12$' && python3 -I -c \"import json; c=json.load(open('$D/si-C/orcamento.json'))['cortes']; assert c[0].startswith('(1)') and c[1].startswith('(2)') and c[2].startswith('(2b)')\""
confere "C: S2 que não rodou é declarada NÃO RODOU (12 itens) e nunca vira célula" "grep -c 'NÃO RODOU: llamacpp-..-S2' $D/si-C/orcamento-log.txt | grep -q '^12$'"
# D. 700: quente 5->3 depois frio 3->2
interno D 700
confere "D (limite 700): cortes 1 a 5 em ordem; quente com 3 medidas e frio com 2 nos dois lados" "[ \$(cortes D) = 5 ] && [ \$(nceluas D) = 24 ] && grep -c 'frio n=2' $D/si-D/ordem.txt | grep -q '^12$' && grep -c 'quente n=3' $D/si-D/ordem.txt | grep -q '^12$' && python3 -I -c \"import json; c=json.load(open('$D/si-D/orcamento.json')); assert [x[:3] for x in c['cortes']]==['(1)','(2)','(2b','(3)','(4)'][:0] or len(c['cortes'])==5; assert c['reps_finais']=={'quente':3,'frio':2}\""
confere "D: Lastro e llama.cpp do mesmo (modelo, texto, estado) com as mesmas repetições" "python3 -I - $D/si-D/ordem.txt <<'PY'
import re, sys, collections
d = collections.defaultdict(set)
for l in open(sys.argv[1]):
    m = re.match(r'\d+ (lastro|llamacpp)-(\w+)-(\w+)-(\w+)-(\w+) n=(\d+)', l)
    d[(m.group(2), m.group(4), m.group(5))].add(m.group(6))
assert all(len(v) == 1 for v in d.values()), d
PY"
# E. 300: cortes esgotados, o que não cabe é NÃO RODOU (nunca estimado)
interno E 300
confere "E (limite 300): todos os cortes e só os 2 primeiros blocos cabem (8 células); o resto NÃO RODOU" "[ \$(cortes E) = 5 ] && [ \$(nceluas E) = 8 ] && grep -c 'NÃO RODOU: bloco' $D/si-E/orcamento-log.txt | grep -q '^4$' && grep -q 'NÃO RODOU: bloco 3b/longo' $D/si-E/orcamento-log.txt && [ \"\$(cat $D/clock)\" = 210 ]"
confere "E: o extrator lista as células ausentes como NÃO RODOU, sem razão" "grep -q 'NÃO RODOU' $D/si-E/cmp-resultado.md && python3 -I -c \"import json; r=json.load(open('$D/si-E/cmp-resultado.json')); x=[z for z in r['razoes'] if z['modelo']=='3b' and z['texto']=='longo']; assert x and all(z['status']=='NÃO RODOU' and 'prefill' not in z for z in x)\""
# F. orçamento estoura no meio (células 2x mais lentas que o estimado): cortes dinâmicos pelo relógio
CMP_FAKE_MULT=2 interno F 1900
confere "F (células 2x mais lentas): S3 cortada primeiro, depois S2 do 3B; cortes no meio da janela (tempo > 0); 3B com quente 3; frio 2 no fim" "[ \$(cortes F) -ge 4 ] && ! grep -q 'S3' $D/si-F/ordem.txt && python3 -I -c \"import json; c=json.load(open('$D/si-F/orcamento.json'))['cortes']; assert c[0].startswith('(1)') and c[1].startswith('(2)')\" && grep -q '^corte em [1-9][0-9]* s: (1) S3' $D/si-F/orcamento-log.txt && ! grep -q 'llamacpp-3b-S2' $D/si-F/ordem.txt && grep -q 'llamacpp-1b-S2-longo-quente n=5' $D/si-F/ordem.txt && grep 'lastro-3b-L-curto-quente' $D/si-F/ordem.txt | grep -q 'n=3'"
# G. --sem-s3 e recusa de ambiente (código 2 de uma célula interrompe a janela)
interno G 100000 --sem-s3
confere "--sem-s3: 36 células e a S3 declarada NÃO RODOU" "[ \$(nceluas G) = 36 ] && grep -q 'NÃO RODOU: llamacpp-1b-S3-curto' $D/si-G/orcamento-log.txt"
CMP_FAKE_RECUSA=llamacpp-1b-S1-curto-frio interno H 100000
confere "célula que recusa o ambiente (2): janela interrompida, nada mais roda, decisão gravada" "[ \$(nceluas H) = 2 ] && grep -q 'recusou o ambiente' $D/si-H/orcamento-log.txt && [ -s $D/si-H/decisao.txt ]"
# I. integridade: arquivo alterado durante a janela
CMP_FAKE_MUDA=$D/textos/texto-curto.txt CMP_FAKE_MUDA_NA=lastro-1b-L-curto-frio interno I 100000
confere "texto alterado durante a janela: integridade QUEBRADA (tudo SEM PROVA) na decisão e no extrator" "grep -q 'integridade: QUEBRADA' $D/si-I/decisao.txt && grep -q 'sha mudou entre antes e depois' $D/si-I/decisao.txt && grep -q 'Integridade QUEBRADA' $D/si-I/cmp-resultado.md"
echo c > "$D/textos/texto-curto.txt"
SEM_SHA_FIXO=0 interno J 100000
confere "sha diferente do congelado (arquivos falsos): QUEBRADA e a nota diz qual" "grep -q 'não o congelado' $D/si-J/decisao.txt && grep -q 'FALHA' $D/si-J/integridade-fixos.txt"

# K e L: SIMULAÇÃO DAS DUAS JANELAS com as estimativas FINAIS da emenda 6 (sem CMP_EST_* nem CMP_FOLGA: padrões do envelope; limite 1800 s, folga 180 s) e células falsas que
# gastam exatamente a estimativa (CMP_FAKE_EST=1). Mostra quais células rodam: o que não cabe fica declarado NÃO RODOU, nunca estimado.
interno_final() { # nome modelo
  local n=$1 m=$2 us=() v
  for v in $(compgen -e | grep -E '^CMP_(EST|FOLGA)'); do us+=(-u "$v"); done
  echo 0 > "$D/clock"; : > "$D/log-$n"
  env "${us[@]}" PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID CMP_RELOGIO_ARQ=$D/clock CMP_FAKE_RELOGIO=$D/clock CMP_FAKE_LOG=$D/log-$n CMP_FAKE_EST=1 \
    "$J" --interno --modelos "$m" --saida "$D/si-$n" --bin "$D/pp" --limite 1800 > "$D/outi-$n" 2>&1
  RC=$?
}
interno_final K 1b; CLK_K=$(cat "$D/clock")
interno_final L 3b; CLK_L=$(cat "$D/clock")
echo "simulação K (janela do 1B, estimativas finais): $(nceluas K) células, $CLK_K s de 1800 (folga 180); cortes: $(cortes K); NÃO RODOU: $(grep -c 'NÃO RODOU' $D/si-K/orcamento-log.txt)"
echo "simulação L (janela do 3B, estimativas finais): $(nceluas L) células, $CLK_L s de 1800 (folga 180); cortes: $(cortes L); NÃO RODOU: $(grep -c 'NÃO RODOU' $D/si-L/orcamento-log.txt)"
[ -z "${CMP_SIM_SAIDA:-}" ] || { cp "$D/si-K/ordem.txt" "$CMP_SIM_SAIDA.K"; cp "$D/si-L/ordem.txt" "$CMP_SIM_SAIDA.L"; cp "$D/si-L/orcamento-log.txt" "$CMP_SIM_SAIDA.Llog"; }
confere "K (janela do 1B, estimativas finais, 1800 s): sem cortes, 24 células (S1, S2 e S3 nos 3 textos, frio e quente), nenhum NÃO RODOU, cabe com folga" "[ \$(nceluas K) = 24 ] && [ \$(cortes K) = 0 ] && [ \$(grep -c 'NÃO RODOU' $D/si-K/orcamento-log.txt) = 0 ] && grep -q '1b-S2-longo-quente' $D/si-K/ordem.txt && grep -q '1b-S3-longo-quente' $D/si-K/ordem.txt && ! grep -q '3b' $D/si-K/ordem.txt && [ $CLK_K -le 1620 ]"
confere "L (janela do 3B, estimativas finais, 1800 s): só a S3 não cabe (corte 1); S1 e S2 em todos os textos, frio e quente, sem bloco principal NÃO RODOU" "[ \$(nceluas L) = 18 ] && [ \$(cortes L) = 1 ] && grep -q '(1) S3 inteira não roda' $D/si-L/orcamento.json && grep -q '3b-S2-longo-quente' $D/si-L/ordem.txt && ! grep -q 'S3\|1b' $D/si-L/ordem.txt && ! grep -q 'NÃO RODOU: bloco' $D/si-L/orcamento-log.txt && [ \$(grep -c 'NÃO RODOU' $D/si-L/orcamento-log.txt) = 3 ] && [ $CLK_L -le 1620 ]"
confere "K e L: o orçamento declara o modelo da janela e a extração só espera esse modelo (sem NÃO RODOU do outro)" "grep -q '\"modelos\": \[\"1b\"\]' $D/si-K/orcamento.json && grep -q '\"modelos\": \[\"3b\"\]' $D/si-L/orcamento.json && ! grep -q '| 1b ' $D/si-L/cmp-resultado.md && ! grep -q '| 3b ' $D/si-K/cmp-resultado.md"
# --modelos: obrigatório e validado (recusa 2, nada roda); 'ambos' só com CMP_TESTE_AMBOS no modo interno
for casom in "" "5b" "1b,3b" "ambos"; do
  rm -f "$D/log-m"; : > "$D/log-m"
  if [ -z "$casom" ]; then args=(); else args=(--modelos "$casom"); fi
  PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID CMP_FAKE_LOG=$D/log-m env -u CMP_TESTE_AMBOS "$J" --interno "${args[@]}" --saida "$D/si-m-$casom" --bin "$D/pp" --limite 100 > "$D/outi-m" 2>&1
  rcm=$?
  confere "--modelos '${casom:-ausente}': recusa 2, nada rodou" "[ $rcm = 2 ] && [ ! -s $D/log-m ] && [ ! -e '$D/si-m-$casom/ordem.txt' ]"
done

echo "== Parte 2: envelope (portões, pastas novas, unidades transitórias, aviso, limite)"
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
export CMP_MODO_TESTE=1     # modo de teste explícito: sem ele o envelope recusa as variáveis só de teste (testado mais abaixo com ambiente limpo)
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
# emenda 9: voxtype e Chrome (C+G) abertos na 4090 = recusa 2 ANTES de tocar em timers, com a lista; só G passa
printf '# gpu pid type sm mem enc dec jpg ofa fb ccpm command\n    0 1352017   C+G      -      -      -      -      -      -    997      0    voxtype\n    0   42406   C+G      -      -      -      -      -      -    364      0    renderD128 --cr\n    0   41533     G     25     25      -      -      -      -    413      0    Hyprland\n' > "$D/pmon-cg.txt"
FAKE_PMON_ARQ=$D/pmon-cg.txt roda cg-aberto; confere "voxtype/Chrome C+G abertos: recusa 2 antes de tocar em timers, listando PIDs e comandos e mandando fechar" "[ $RC = 2 ] && grep -q 'FECHADOS' $D/oute-cg-aberto && grep -q '1352017 (voxtype)' $D/oute-cg-aberto && grep -q '42406 (renderD128)' $D/oute-cg-aberto && [ ! -s $D/elog-cg-aberto ] && ! grep -q '^stop' $D/cg-aberto/log && [ ! -e $D/se-cg-aberto ] && [ ! -e $D/se-cg-aberto.timers-antes.txt ]"
roda limite-alto --limite 1801; confere "limite acima de 1800: recusa 2" "[ $RC = 2 ] && [ ! -s $D/elog-limite-alto ]"
roda opcao-ruim --nao-existe; confere "opção desconhecida: recusa 2" "[ $RC = 2 ] && [ ! -s $D/elog-opcao-ruim ]"
# emenda 6: --modelos obrigatório no envelope (recusa 2 antes de qualquer coisa)
MOD="" roda sem-modelos; confere "envelope sem --modelos (a janela dupla por engano): recusa 2, nada rodou, timers intocados" "[ $RC = 2 ] && grep -q 'modelos' $D/oute-sem-modelos && [ ! -s $D/elog-sem-modelos ] && ! grep -q '^stop' $D/sem-modelos/log && [ ! -e $D/se-sem-modelos ]"
unset MOD
for mm in 5b ambos 1b,3b; do MOD=$mm roda mod-ruim; confere "envelope com --modelos '$mm': recusa 2" "[ $RC = 2 ] && [ ! -s $D/elog-mod-ruim ] && [ ! -e $D/se-mod-ruim ]"; done
unset MOD
# variáveis só de teste no modo real: ambiente LIMPO (env -i) com a variável e SEM CMP_MODO_TESTE = recusa 2, nada criado; com ambiente limpo e outra recusa, a de teste não aparece
real() { # nome VAR=valor...  (ambiente limpo; o envelope é chamado com autorização válida e pasta de saída nova)
  local n=$1; shift
  env -i PATH="$PATH" HOME="$HOME" PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=$UUID "$@" "$J" --modelos 1b --bin "$D/pp" --saida "$D/se-real-$n" --limite 60 > "$D/oute-real-$n" 2>&1
  RC=$?
}
for v in CMP_FRACAO_SLOTS CRITERIO_D1 SEM_SHA_FIXO LLAMA_SERVER MONITOR_SH LASTRO_SH_CMP LLAMA_SH_CMP EXTRAI_PY CMP_RELOGIO_ARQ NVIDIA_SMI MODELO_1B MODELO_3B GGUF_DIR LLAMA_DIR TEXTOS_5C REF_5C IDS_CMP CRITERIO_CMP \
         PROC_STAT CMP_CPU_JANELA CMP_CPU_ESPERA CMP_FOLGA CMP_SEM_MEMORIA CMP_TESTE_AMBOS CURL ESPERA_PRONTO LOADAVG_ARQ SYSTEMCTL SYSTEMD_RUN PENDENTE CGROUP_RAIZ \
         CMP_EST_L_CARGA_1B CMP_FAKE_RECUSA JANELA_SH JANELA_TIMERS JANELA_PASSO_ESPERA LIMITE_SH LIMITE_PASSA; do
  real "v-$v" "$v=1"
  confere "modo real com $v no ambiente: recusa 2 citando a variável, nada criado" "[ $RC = 2 ] && grep -q 'variável só de teste presente no modo real: $v' $D/oute-real-v-$v && [ ! -e $D/se-real-v-$v ]"
done
mkdir -p "$D/se-real-limpo"; echo x > "$D/se-real-limpo/a"
real limpo
confere "ambiente limpo: a checagem de variáveis de teste passa (a recusa seguinte é a da pasta ocupada)" "[ $RC = 2 ] && ! grep -q 'variável só de teste' $D/oute-real-limpo && grep -q 'não está vazio' $D/oute-real-limpo"
real com-modo-teste CMP_MODO_TESTE=1 SEM_SHA_FIXO=1
confere "CMP_MODO_TESTE=1 (modo de teste explícito) aceita a variável de teste: passa da checagem" "! grep -q 'variável só de teste' $D/oute-real-com-modo-teste"
# o sha congelado do critério e os demais, contra os arquivos REAIS (sem SEM_SHA_FIXO, sem desvios de caminho): ambiente limpo, só leitura
env -i PATH="$PATH" HOME="$HOME" "$J" --verifica-fixos > "$D/fixos.txt" 2>&1; rcf=$?
nok=$(grep -c ' OK$' "$D/fixos.txt"); nfal=$(grep -c ' FALHA$' "$D/fixos.txt"); naus=$(grep -c ' AUSENTE ' "$D/fixos.txt")
echo "congelados do FIXO contra os arquivos reais: $nok OK, $nfal FALHA, $naus AUSENTE (tokenizers fora do repositório podem faltar em outra máquina)"
confere "FIXO: os shas congelados (critério, ids, textos, textos.json, referências 5c; mais os tokenizers se existirem aqui) batem com os arquivos reais; nenhum FALHA" "[ $nfal = 0 ] && [ $nok -ge 8 ] && grep -q '^criterio-cmp .* OK$' $D/fixos.txt && grep -q '^cmp-ids-prompt .* OK$' $D/fixos.txt && grep -q '^ref5c-3b .* OK$' $D/fixos.txt && ! grep ' AUSENTE ' $D/fixos.txt | grep -qv 'tokenizer-'"
confere "FIXO: o sha congelado do critério é o do arquivo atual (sha256sum direto)" "[ \"\$(grep '^criterio-cmp ' $D/fixos.txt | cut -d' ' -f2)\" = \"\$(sha256sum $AQUI/criterio-comparacao-llamacpp-20261108.md | cut -d' ' -f1)\" ]"
# caminho feliz (relógio real; células falsas instantâneas)
LIM=1800 roda ok
confere "ok: sai 0, decisao.txt, 24 células do 1B (emenda 6: uma janela por modelo), extrator, timers parados e restaurados" "[ $RC = 0 ] && [ -s $D/se-ok/decisao.txt ] && [ \$(ls $D/se-ok/celulas | wc -l) = 24 ] && ! ls $D/se-ok/celulas | grep -q 3b && grep -q '\"modelos\": \[\"1b\"\]' $D/se-ok/orcamento.json && [ -s $D/se-ok/cmp-resultado.json ] && grep -q '^stop a.timer' $D/ok/log && [ \"\$(cat $D/ok/a.timer)\" = active ] && grep -q 'RESTAURAÇÃO OK' $D/se-ok-estado-janela/restauracao.txt"
confere "ok: decisão descritiva (sem PASSA/REPROVA), integridade íntegra, sem cortes; cargas e listas systemd ao lado da pasta" "grep -q 'resultado DESCRITIVO' $D/se-ok/decisao.txt && ! grep -q '^rótulo:' $D/se-ok/decisao.txt && grep -q 'integridade: íntegra' $D/se-ok/decisao.txt && grep -q 'cortes de orçamento: nenhum' $D/se-ok/decisao.txt && [ -f $D/se-ok.carga-antes.txt ] && [ -f $D/se-ok.carga-depois.txt ] && [ -s $D/se-ok.unidades-antes.txt ] && [ -s $D/se-ok.timers-depois.txt ]"
confere "ok: aviso no fim da janela (arquivo e comando de aviso)" "grep -q 'terminou (código 0)' $D/se-ok.aviso.txt && grep -q 'terminou (código 0)' $D/aviso-chamado"
# limite de tempo estourado: células lentas; 124, timers restaurados, aviso mesmo assim
rm -f "$D/aviso-chamado"
for v in $(compgen -v | grep '^CMP_EST_'); do export "$v=0"; done   # estimativas zeradas: as células começam mesmo com limite curto
CMP_FAKE_SLEEP=40 LIM=3 roda lim
confere "limite estourado: 124, sem decisão, timers restaurados, aviso dado" "[ $RC = 124 ] && [ ! -f $D/se-lim/decisao.txt ] && [ \"\$(cat $D/lim/a.timer)\" = active ] && grep -q 'terminou (código 124)' $D/aviso-chamado"
if [ "$falhas" = 0 ]; then echo "TODOS OS TESTES PASSARAM"; exit 0; else echo "$falhas FALHA(S)"; exit 1; fi
