#!/usr/bin/env bash
# Teste do roda-cmp-lastro.sh com `primeira-palavra` FALSO (grava medidas-<modelo>.json no formato do E9) e monitor FALSO. SEM GPU. mktemp -d, sem rm -rf.
set -u
unset PRIMEIRA_PALAVRA_AUTORIZADA
export CUDA_VISIBLE_DEVICES=
AQUI=$(cd "$(dirname "$0")" && pwd -P)
L=$AQUI/roda-cmp-lastro.sh
UUID=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd
D=$(mktemp -d -p "${TMPDIR:-/tmp}" cmp-lastro-teste.XXXXXX) || exit 1
echo "pasta de teste: $D"
falhas=0
confere() { if eval "$2"; then echo "ok: $1"; else echo "FALHOU: $1"; falhas=$((falhas+1)); fi; }
cat > "$D/pp.py" <<'PY'
import json, os, sys
a = sys.argv[1:]
log = os.environ["FAKE_LOG"]
n_call = len(open(log).read().splitlines()) + 1 if os.path.exists(log) else 1
open(log, "a").write(" ".join(a) + "\n")
v = lambda k: a[a.index(k) + 1] if k in a else None
saida = v("--saida"); nome, mdir = v("--celula").split(":", 1); texto = v("--so-texto"); rep = v("--repete")
# o parser real (primeira-palavra.rs, parse_args) recusa --novos/--data na prova --gera sem --texto: o falso recusa igual (código 2)
if "--so-texto" in a and "--texto" not in a and ("--novos" in a or "--data" in a):
    sys.stderr.write("primeira-palavra: recusado: a prova --gera tem data e 64 tokens fixos pelo critério\n"); sys.exit(2)
if os.environ.get("FAKE_RECUSA_NA_CHAMADA") == str(n_call):
    sys.stderr.write("primeira-palavra: recusado: ambiente\n"); sys.exit(2)
if os.path.exists(saida) and os.listdir(saida):
    sys.stderr.write("saida ja existe\n"); sys.exit(2)
os.makedirs(saida)
if os.environ.get("FAKE_FALHA_NA_CHAMADA") == str(n_call):
    sys.stderr.write("erro de mentira\n"); sys.exit(1)
def ent(rot, desc):
    return {"rotulo": rot, "descartada": desc, "ids": [1, 2], "gerados": list(range(64)), "passo_ms": [2.0] * 63, "prefill_ms": 10.0, "total_ms": 200.0, "T_sequencia_ms": 50.0,
            "media_por_passo_ms": 2.0, "cont": {"needed": 10, "hits": 1, "cpu": 9}, "fracao_cpu_needed": 0.9, "falha": None}
ms = [ent("frio", False)] if rep is None else [ent("aquecimento", True)] + [ent("quente rep %d" % i, False) for i in range(1, int(rep) + 1)]
json.dump({"tipo": "medidas-gera", "modelo": nome, "texto": texto, "medidas": ms}, open(os.path.join(saida, "medidas-%s.json" % nome), "w"))
json.dump({"falha": None, "integro": True}, open(os.path.join(saida, "gera-%s.json" % nome), "w"))
open(os.path.join(saida, "gera-final.txt"), "w").write("tipo=gera-final\nintegro=1\n")
print("ok falso")
PY
printf '#!/usr/bin/env bash\nexec python3 -I %s "$@"\n' "$D/pp.py" > "$D/pp"; chmod +x "$D/pp"
# monitor falso: portao devolve o conteúdo de $D/portao.rc; inicia/para controlam um sleep; avalia ok
cat > "$D/mon" <<F
#!/usr/bin/env bash
# comportamentos por arquivo: mon.inicia.rc (falha de inicia), mon.morre (a coleta morre logo), mon.para.rc, mon.avalia.rc e mon.avalia.txt
case \$1 in
  portao) { [ ! -e $D/desk.antes ] || printf 'pmon_antes_g_soma_sm_max=%s\\ndmon_antes_sm_max=%s\\n' "\$(cat $D/desk.antes)" "\$(cat $D/dmon.antes)"; echo "portao=\$( [ "\$(cat $D/portao.rc)" = 0 ] && echo OK || echo 'FALHA CPU ocupada' )"; } > "\$3"; exit \$(cat $D/portao.rc);;
  portao-depois) [ ! -e $D/desk.depois ] || printf 'pmon_depois_g_soma_sm_max=%s\\n' "\$(cat $D/desk.depois)" > "\$3"; exit 0;;
  inicia) [ ! -e $D/mon.inicia.rc ] || exit \$(cat $D/mon.inicia.rc)
          echo "t, 1, 2000, 10000, 60, 100" > "\$3"
          if [ -e $D/mon.morre ]; then true & p=\$!; wait \$p; echo \$p > "\$5"
          else sleep 100 & echo \$! > "\$5"; echo \$! >> $D/mon.pids; fi
          echo "inicia" >> $D/mon.log;;
  para) p=\$(cat "\$3" 2>/dev/null); [ -z "\$p" ] || kill "\$p" 2>/dev/null; echo "para" >> $D/mon.log; [ ! -e $D/mon.para.rc ] || exit \$(cat $D/mon.para.rc);;
  avalia) if [ -e $D/mon.avalia.txt ]; then cat $D/mon.avalia.txt; else echo "amostras=1 temp_max=60"; fi; [ ! -e $D/mon.avalia.rc ] || exit \$(cat $D/mon.avalia.rc); exit 0;;
esac
F
chmod +x "$D/mon"; echo 0 > "$D/portao.rc"
export MONITOR_SH=$D/mon FAKE_LOG=$D/log MODELO_1B=$D/m1b MODELO_3B=$D/m3b
roda() { # nome args...; define RC; log em $D/log-<nome>
  local n=$1; shift
  : > "$D/log"; "$L" --bin "$D/pp" --saida "$D/s-$n" "$@" > "$D/out-$n" 2>&1; RC=$?; cp "$D/log" "$D/log-$n"
}
# 1. falha fechada
roda sem-portao --modelo 1b --texto curto --estado frio --n 2
confere "sem autorização: recusa 2, nada executado, sem pasta" "[ $RC = 2 ] && [ ! -s $D/log-sem-portao ] && [ ! -e $D/s-sem-portao ]"
export PRIMEIRA_PALAVRA_AUTORIZADA=1
roda sem-uuid --modelo 1b --texto curto --estado frio --n 2
confere "sem UUID: recusa 2" "[ $RC = 2 ] && [ ! -s $D/log-sem-uuid ]"
export CUDA_VISIBLE_DEVICES=GPU-00000000-0000-0000-0000-000000005090
roda uuid-5090 --modelo 1b --texto curto --estado frio --n 2
confere "UUID de outra GPU (a 5090 nunca): recusa 2" "[ $RC = 2 ] && [ ! -s $D/log-uuid-5090 ]"
export CUDA_VISIBLE_DEVICES=$UUID
mkdir -p "$D/s-existe"
roda existe --modelo 1b --texto curto --estado frio --n 2
confere "pasta de saída existente: recusa 2" "[ $RC = 2 ] && [ ! -s $D/log-existe ]"
roda uso --modelo 5b --texto curto --estado frio --n 2
confere "modelo inválido: recusa 2" "[ $RC = 2 ]"
roda uso2 --modelo 1b --texto curto --estado frio --n 0
confere "n = 0: recusa 2" "[ $RC = 2 ]"

# 2. frio: 1 descartado + N medidas, processos NOVOS, cada um com saída nova
roda frio --modelo 1b --texto medio --estado frio --n 3
confere "frio: sai 0 e roda 4 processos (p0 descartado + p1..p3)" "[ $RC = 0 ] && [ \$(wc -l < $D/log-frio) = 4 ] && [ -d $D/s-frio/p0 ] && [ -d $D/s-frio/p3 ] && [ ! -d $D/s-frio/p4 ]"
confere "frio: cada processo é --gera --so-texto medio com saída própria, sem --repete, sem --novos nem --data (o harness os fixa; o parser real recusa)" "! grep -q -- --repete $D/log-frio && ! grep -q -- '--novos\|--data' $D/log-frio && [ \$(grep -c -- '--gera --celula 1b:$D/m1b --so-texto medio --saida $D/s-frio/p' $D/log-frio) = 4 ]"
confere "frio: cada pK tem saida/medidas-1b.json, wall_ms e rc" "[ -s $D/s-frio/p2/saida/medidas-1b.json ] && [ -s $D/s-frio/p2/wall_ms.txt ] && [ \"\$(cat $D/s-frio/p2/rc.txt)\" = 0 ]"
confere "frio: celula.json com n_medidas 3, portao_ok true, rc 0" "grep -q '\"estado\": \"frio\", \"n_medidas\": 3, \"portao_ok\": true' $D/s-frio/celula.json && grep -q '\"rc\": 0' $D/s-frio/celula.json"
confere "celula.json grava inicio_coleta_epoch entre inicio_epoch e fim_epoch (emenda 7)" "python3 -I -c \"import json,sys; m=json.load(open(sys.argv[1])); assert m['inicio_epoch']<=m['inicio_coleta_epoch']<=m['fim_epoch']\" $D/s-frio/celula.json"
confere "monitor: portão, amostragem a 1 Hz e parada na ordem" "[ \"\$(cat $D/mon.log | tr '\n' ' ')\" = 'inicia para para ' ] || [ \"\$(cat $D/mon.log | tr '\n' ' ')\" = 'inicia para ' ]"
# 3. quente: UM processo com --repete N
roda quente --modelo 3b --texto longo --estado quente --n 5
confere "quente: 1 processo com --repete 5 (3B)" "[ $RC = 0 ] && [ \$(wc -l < $D/log-quente) = 1 ] && grep -q -- '--celula 3b:$D/m3b --so-texto longo --repete 5 --saida' $D/log-quente && ! grep -q -- '--novos\|--data' $D/log-quente && [ -d $D/s-quente/p1 ]"
confere "quente: o medidas tem 1 aquecimento descartado + 5 medidas" "python3 -I -c \"import json; m=json.load(open('$D/s-quente/p1/saida/medidas-3b.json'))['medidas']; assert len(m)==6 and m[0]['descartada'] and not any(x['descartada'] for x in m[1:])\""
# 4. portão falho: roda mesmo assim (não adia), marca portao_ok false
echo 1 > "$D/portao.rc"
roda portao-falho --modelo 1b --texto curto --estado frio --n 2
confere "portão falho: a célula roda (sem adiar nem repetir) e fica marcada" "[ $RC = 0 ] && [ \$(wc -l < $D/log-portao-falho) = 3 ] && grep -q '\"portao_ok\": false' $D/s-portao-falho/celula.json && grep -q 'CPU ocupada' $D/s-portao-falho/celula.json"
echo 0 > "$D/portao.rc"
# 4b. emenda 9: caminho NUMÉRICO do sed que grava desktop_g_sm_* e dmon_sm_antes no celula.json (monitor falso com linhas pmon/dmon); sem leitura = null
confere "sem linhas pmon do monitor: desktop_g_sm_* e dmon_sm_antes = null, JSON válido" "python3 -I -c \"import json,sys; m=json.load(open(sys.argv[1])); assert m['desktop_g_sm_antes'] is None and m['desktop_g_sm_depois'] is None and m['dmon_sm_antes'] is None\" $D/s-portao-falho/celula.json"
echo 26 > "$D/desk.antes"; echo 9 > "$D/dmon.antes"; echo 31 > "$D/desk.depois"
roda desk --modelo 1b --texto curto --estado frio --n 2
confere "com linhas pmon/dmon: celula.json traz desktop_g_sm_antes 26, depois 31 e dmon_sm_antes 9 como números (JSON válido)" "python3 -I -c \"import json,sys; m=json.load(open(sys.argv[1])); assert (m['desktop_g_sm_antes'], m['desktop_g_sm_depois'], m['dmon_sm_antes'])==(26,31,9) and m['n_medidas']==2\" $D/s-desk/celula.json"
echo 'ilegivel' > "$D/dmon.antes"; echo 5x > "$D/desk.depois"
roda desk2 --modelo 1b --texto curto --estado frio --n 2
confere "valor não numérico (ilegivel, 5x) vira null, sem quebrar o JSON" "python3 -I -c \"import json,sys; m=json.load(open(sys.argv[1])); assert (m['desktop_g_sm_antes'], m['desktop_g_sm_depois'], m['dmon_sm_antes'])==(26,None,None)\" $D/s-desk2/celula.json"
rm -f "$D/desk.antes" "$D/dmon.antes" "$D/desk.depois"
# 5. processo com erro encerra a célula: sem repetição automática
export FAKE_FALHA_NA_CHAMADA=2
roda erro --modelo 1b --texto curto --estado frio --n 3
unset FAKE_FALHA_NA_CHAMADA
confere "erro no 2º processo: sai 1, não roda os seguintes, não repete" "[ $RC = 1 ] && [ \$(wc -l < $D/log-erro) = 2 ] && grep -q '\"rc\": 1' $D/s-erro/celula.json && [ \"\$(cat $D/s-erro/p1/rc.txt)\" = 1 ]"
vivos=0; for p in $(cat "$D/mon.pids"); do kill -0 "$p" 2> /dev/null && vivos=1; done
confere "o monitor (amostragem) é parado em todas as células, inclusive a com erro" "[ -s $D/mon.pids ] && [ $vivos = 0 ]"
# 6. recusa (código 2) do binário: propaga 2 e não inicia os processos seguintes (emenda 5, item 11)
export FAKE_RECUSA_NA_CHAMADA=2
roda recusa2 --modelo 1b --texto curto --estado frio --n 3
unset FAKE_RECUSA_NA_CHAMADA
confere "recusa 2 do binário no 2º processo: a célula sai 2 (não 1), não roda p2 em diante, celula.json grava rc 2" "[ $RC = 2 ] && [ \$(wc -l < $D/log-recusa2) = 2 ] && grep -q '\"rc\": 2' $D/s-recusa2/celula.json && [ \"\$(cat $D/s-recusa2/p1/rc.txt)\" = 2 ] && [ ! -d $D/s-recusa2/p2 ]"
export FAKE_RECUSA_NA_CHAMADA=1
roda recusa2q --modelo 1b --texto curto --estado quente --n 2
unset FAKE_RECUSA_NA_CHAMADA
confere "recusa 2 no quente: sai 2" "[ $RC = 2 ]"
# o fake recusa, como o parser real, --novos/--data com --so-texto: provado chamando o binário falso direto
"$D/pp" --gera --celula 1b:/x --so-texto curto --novos 64 --data 2026-11-07 --saida "$D/direto" > /dev/null 2>&1
confere "o binário falso recusa --novos/--data na prova --gera (como o parser real)" "[ $? = 2 ] && [ ! -e $D/direto ]"
# 7. monitor: falhas de inicia/para/avalia e coleta morta ficam no metadata (o extrator trata como SEM PROVA)
vazio() { grep -q '"monitor_falha": ""' "$1"; }
roda mon-ok --modelo 1b --texto curto --estado frio --n 1
confere "monitor saudável: monitor_falha vazio" "[ $RC = 0 ] && vazio $D/s-mon-ok/celula.json"
echo 3 > "$D/mon.inicia.rc"
roda mon-inicia --modelo 1b --texto curto --estado frio --n 1
rm -f "$D/mon.inicia.rc"
confere "falha de inicia registrada (inicia rc=3 e coleta ausente)" "[ $RC = 0 ] && grep -q 'inicia rc=3' $D/s-mon-inicia/celula.json"
touch "$D/mon.morre"
roda mon-morre --modelo 1b --texto curto --estado frio --n 1
rm -f "$D/mon.morre"
confere "coleta morta antes do fim da célula registrada" "[ $RC = 0 ] && grep -q 'coleta encerrada antes do fim' $D/s-mon-morre/celula.json"
echo 4 > "$D/mon.para.rc"
roda mon-para --modelo 1b --texto curto --estado frio --n 1
rm -f "$D/mon.para.rc"
confere "falha de para registrada" "grep -q 'para rc=4' $D/s-mon-para/celula.json"
echo 2 > "$D/mon.avalia.rc"
roda mon-avalia --modelo 1b --texto curto --estado frio --n 1
rm -f "$D/mon.avalia.rc"
confere "falha de avalia (código 2) registrada" "grep -q 'avalia rc=2' $D/s-mon-avalia/celula.json"
echo "sem amostras em x" > "$D/mon.avalia.txt"; echo 1 > "$D/mon.avalia.rc"
roda mon-semam --modelo 1b --texto curto --estado frio --n 1
rm -f "$D/mon.avalia.txt" "$D/mon.avalia.rc"
confere "avalia 'sem amostras' registrada" "grep -q 'avalia rc=1' $D/s-mon-semam/celula.json"
echo "amostras=3 temp_max=85" > "$D/mon.avalia.txt"; echo 1 > "$D/mon.avalia.rc"
roda mon-quente --modelo 1b --texto curto --estado frio --n 1
rm -f "$D/mon.avalia.txt" "$D/mon.avalia.rc"
confere "avalia com > 80 C (código 1, com amostras) NÃO é falha de coleta" "vazio $D/s-mon-quente/celula.json"
# 8. degrau 1: CMP_FRACAO_SLOTS=1.0 repassa --fracao-slots 1.0, grava a linha de comando e muda a série para L100; sem a variável nada muda
printf '#!/usr/bin/env bash\necho "2000, 24000"\n' > "$D/smi"; chmod +x "$D/smi"; export NVIDIA_SMI=$D/smi
CMP_FRACAO_SLOTS=1.0 roda frac --modelo 1b --texto curto --estado frio --n 2
confere "fração 1.0 (frio): todo processo (p0 a p2) recebe --fracao-slots 1.0 antes do --saida; o resto da linha é o de sempre" "[ $RC = 0 ] && [ \$(grep -c -- '--gera --celula 1b:$D/m1b --so-texto curto --fracao-slots 1.0 --saida $D/s-frac/p' $D/log-frac) = 3 ]"
confere "fração 1.0: cmdline.txt com uma linha por processo, cada uma com a flag" "[ \$(wc -l < $D/s-frac/cmdline.txt) = 3 ] && [ \$(grep -c -- '--fracao-slots 1.0' $D/s-frac/cmdline.txt) = 3 ] && grep -q '^p0: ' $D/s-frac/cmdline.txt"
confere "fração 1.0: celula.json com serie L100 e fracao_slots 1.0 (JSON válido)" "python3 -I -c \"import json,sys; m=json.load(open(sys.argv[1])); assert m['serie']=='L100' and m['fracao_slots']=='1.0' and m['lado']=='lastro'\" $D/s-frac/celula.json"
CMP_FRACAO_SLOTS=1.0 roda frac-q --modelo 3b --texto longo --estado quente --n 2
confere "fração 1.0 (quente): o processo único leva --fracao-slots 1.0 e --repete" "[ $RC = 0 ] && grep -q -- '--so-texto longo --fracao-slots 1.0 --repete 2 --saida' $D/log-frac-q"
CMP_FRACAO_SLOTS=0.5 roda frac-ruim --modelo 1b --texto curto --estado frio --n 2
confere "fração diferente de 1.0 (0.5): recusa 2, nada executado, sem pasta" "[ $RC = 2 ] && [ ! -s $D/log-frac-ruim ] && [ ! -e $D/s-frac-ruim ]"
roda sem-frac --modelo 1b --texto curto --estado frio --n 1
confere "sem a variável: sem --fracao-slots na linha, série L, fracao_slots vazio" "! grep -q -- '--fracao-slots' $D/log-sem-frac && python3 -I -c \"import json,sys; m=json.load(open(sys.argv[1])); assert m['serie']=='L' and m['fracao_slots']==''\" $D/s-sem-frac/celula.json"
# 9. smoke do degrau 1
: > "$D/mon.log"
CMP_FRACAO_SLOTS=1.0 roda smoke --smoke --modelo 1b
confere "smoke: sai 0, um processo com --so-texto curto --fracao-slots 1.0, sem monitor nem portão" "[ $RC = 0 ] && [ \$(wc -l < $D/log-smoke) = 1 ] && grep -q -- '--gera --celula 1b:$D/m1b --so-texto curto --fracao-slots 1.0 --saida $D/s-smoke/p1/saida' $D/log-smoke && [ ! -s $D/mon.log ] && [ ! -e $D/s-smoke/celula.json ]"
confere "smoke: grava rc.txt 0, cmdline.txt, gpumem.csv (epoch_ms, usada, total) e smoke.json" "[ \"\$(cat $D/s-smoke/rc.txt)\" = 0 ] && grep -q -- '--fracao-slots 1.0' $D/s-smoke/cmdline.txt && grep -q '^[0-9]*, 2000, 24000\$' $D/s-smoke/gpumem.csv && grep -q '\"smoke\": true' $D/s-smoke/smoke.json && [ -s $D/s-smoke/p1/saida/medidas-1b.json ]"
confere "smoke: smoke-tempos.txt com início/fim do coletor e do binário (epoch ms, ordem coerente) e amostra pós-saída em arquivo À PARTE" "python3 -I -c \"import sys; t=dict(l.strip().split('=') for l in open(sys.argv[1]) if '=' in l); v=[int(t[k]) for k in ('coletor_inicio_ms','bin_inicio_ms','bin_fim_ms','coletor_fim_ms')]; assert v==sorted(v) and v[0]>1700000000000\" $D/s-smoke/smoke-tempos.txt && grep -q '^[0-9]*, 2000, 24000\$' $D/s-smoke/gpumem-pos-saida.csv"
# monitor de memória falso que falha durante toda a execução do binário e só responde depois de ele sair (rev-1 P2 da r116): gpumem.csv não pode ter leitura, a falha é registrada e a pós-saída fica à parte
printf '#!/usr/bin/env bash\nif grep -q bin_fim_ms "%s/s-smoke-tardio/smoke-tempos.txt" 2> /dev/null; then echo "2000, 24000"; else exit 1; fi\n' "$D" > "$D/smi-tardio"; chmod +x "$D/smi-tardio"
NVIDIA_SMI=$D/smi-tardio CMP_FRACAO_SLOTS=1.0 roda smoke-tardio --smoke --modelo 1b
confere "smoke com monitor que só responde depois da saída: gpumem.csv sem leitura viva, falhas registradas, pós-saída à parte" "[ $RC = 0 ] && [ ! -s $D/s-smoke-tardio/gpumem.csv ] && [ -s $D/s-smoke-tardio/gpumem-falhas.txt ] && grep -q '^[0-9]*, 2000, 24000\$' $D/s-smoke-tardio/gpumem-pos-saida.csv"
roda smoke-sem-var --smoke --modelo 1b
confere "smoke sem CMP_FRACAO_SLOTS=1.0: recusa 2" "[ $RC = 2 ] && [ ! -s $D/log-smoke-sem-var ]"
CMP_FRACAO_SLOTS=1.0 roda smoke-uso --smoke --modelo 1b --texto longo
confere "smoke com --texto: recusa de uso (2)" "[ $RC = 2 ] && [ ! -s $D/log-smoke-uso ]"
export FAKE_FALHA_NA_CHAMADA=1
CMP_FRACAO_SLOTS=1.0 roda smoke-erro --smoke --modelo 1b
unset FAKE_FALHA_NA_CHAMADA
confere "smoke com erro do binário: sai 1 e rc.txt = 1 (quem julga é o extrator)" "[ $RC = 1 ] && [ \"\$(cat $D/s-smoke-erro/rc.txt)\" = 1 ]"
export FAKE_RECUSA_NA_CHAMADA=1
CMP_FRACAO_SLOTS=1.0 roda smoke-recusa --smoke --modelo 1b
unset FAKE_RECUSA_NA_CHAMADA
confere "smoke com recusa 2 do binário: sai 2" "[ $RC = 2 ]"
# 10. degrau 2: CMP_CAMADAS_CPU = valor FIXO do modelo (22 no 1B, 29 no 3B) repassa --camadas-cpu N, série LC, camadas_cpu em celula.json; exclusiva com CMP_FRACAO_SLOTS
CMP_CAMADAS_CPU=22 roda cam --modelo 1b --texto curto --estado frio --n 2
confere "camadas 22 (1B, frio): todo processo (p0 a p2) recebe --camadas-cpu 22 antes do --saida; sem --fracao-slots" "[ $RC = 0 ] && [ \$(grep -c -- '--gera --celula 1b:$D/m1b --so-texto curto --camadas-cpu 22 --saida $D/s-cam/p' $D/log-cam) = 3 ] && ! grep -q -- '--fracao-slots' $D/log-cam"
confere "camadas 22: cmdline.txt com uma linha por processo, cada uma com a flag; celula.json serie LC e camadas_cpu 22" "[ \$(wc -l < $D/s-cam/cmdline.txt) = 3 ] && [ \$(grep -c -- '--camadas-cpu 22' $D/s-cam/cmdline.txt) = 3 ] && python3 -I -c \"import json,sys; m=json.load(open(sys.argv[1])); assert m['serie']=='LC' and m['camadas_cpu']=='22' and m['fracao_slots']=='' and m['lado']=='lastro'\" $D/s-cam/celula.json"
CMP_CAMADAS_CPU=29 roda cam-q --modelo 3b --texto longo --estado quente --n 2
confere "camadas 29 (3B, quente): o processo único leva --camadas-cpu 29 e --repete" "[ $RC = 0 ] && grep -q -- '--so-texto longo --camadas-cpu 29 --repete 2 --saida' $D/log-cam-q"
CMP_CAMADAS_CPU=29 roda cam-errado --modelo 1b --texto curto --estado frio --n 1
confere "valor de outro modelo (29 no 1B): recusa 2, nada executado, sem pasta" "[ $RC = 2 ] && [ ! -s $D/log-cam-errado ] && [ ! -e $D/s-cam-errado ]"
CMP_CAMADAS_CPU=22 roda cam-errado3 --modelo 3b --texto curto --estado frio --n 1
confere "valor do 1B no 3B (22): recusa 2" "[ $RC = 2 ] && [ ! -s $D/log-cam-errado3 ] && [ ! -e $D/s-cam-errado3 ]"
for v in 0 21 '' x; do
  CMP_CAMADAS_CPU=$v roda cam-ruim --modelo 1b --texto curto --estado frio --n 1
  confere "CMP_CAMADAS_CPU='$v': recusa 2, nada executado" "[ $RC = 2 ] && [ ! -s $D/log-cam-ruim ] && [ ! -e $D/s-cam-ruim ]"
done
CMP_CAMADAS_CPU=22 CMP_FRACAO_SLOTS=1.0 roda cam-excl --modelo 1b --texto curto --estado frio --n 1
confere "CMP_CAMADAS_CPU junto com CMP_FRACAO_SLOTS: recusa 2" "[ $RC = 2 ] && [ ! -s $D/log-cam-excl ] && [ ! -e $D/s-cam-excl ]"
: > "$D/mon.log"
CMP_CAMADAS_CPU=22 roda smoke-cam --smoke --modelo 1b
confere "smoke do degrau 2: um processo com --so-texto curto --camadas-cpu 22, sem monitor, sem celula.json; smoke.json serie LC" "[ $RC = 0 ] && [ \$(wc -l < $D/log-smoke-cam) = 1 ] && grep -q -- '--gera --celula 1b:$D/m1b --so-texto curto --camadas-cpu 22 --saida $D/s-smoke-cam/p1/saida' $D/log-smoke-cam && [ ! -s $D/mon.log ] && [ ! -e $D/s-smoke-cam/celula.json ] && grep -q -- '--camadas-cpu 22' $D/s-smoke-cam/cmdline.txt && grep -q '\"serie\": \"LC\"' $D/s-smoke-cam/smoke.json && grep -q '\"camadas_cpu\": \"22\"' $D/s-smoke-cam/smoke.json"
CMP_CAMADAS_CPU=29 roda smoke-cam3 --smoke --modelo 3b
confere "smoke do degrau 2 no 3B leva --camadas-cpu 29" "[ $RC = 0 ] && grep -q -- '--camadas-cpu 29' $D/s-smoke-cam3/cmdline.txt"
if [ "$falhas" = 0 ]; then echo "TODOS OS TESTES PASSARAM"; exit 0; else echo "$falhas FALHA(S)"; exit 1; fi
