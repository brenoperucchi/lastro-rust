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
case \$1 in
  portao) echo "portao=\$( [ "\$(cat $D/portao.rc)" = 0 ] && echo OK || echo 'FALHA CPU ocupada' )" > "\$3"; exit \$(cat $D/portao.rc);;
  inicia) echo "t, 1, 2000, 10000, 60, 100" > "\$3"; sleep 100 & echo \$! > "\$5"; echo \$! >> $D/mon.pids; echo "inicia" >> $D/mon.log;;
  para) p=\$(cat "\$3" 2>/dev/null); [ -z "\$p" ] || kill "\$p" 2>/dev/null; echo "para" >> $D/mon.log;;
  avalia) echo "amostras=1 temp_max=60"; exit 0;;
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
confere "frio: cada processo é --gera --so-texto medio, sem --repete, --novos 64, --data 2026-11-07 e saída própria" "! grep -q -- --repete $D/log-frio && [ \$(grep -c -- '--gera --celula 1b:$D/m1b --so-texto medio --novos 64 --data 2026-11-07 --saida $D/s-frio/p' $D/log-frio) = 4 ]"
confere "frio: cada pK tem saida/medidas-1b.json, wall_ms e rc" "[ -s $D/s-frio/p2/saida/medidas-1b.json ] && [ -s $D/s-frio/p2/wall_ms.txt ] && [ \"\$(cat $D/s-frio/p2/rc.txt)\" = 0 ]"
confere "frio: celula.json com n_medidas 3, portao_ok true, rc 0" "grep -q '\"estado\": \"frio\", \"n_medidas\": 3, \"portao_ok\": true' $D/s-frio/celula.json && grep -q '\"rc\": 0' $D/s-frio/celula.json"
confere "monitor: portão, amostragem a 1 Hz e parada na ordem" "[ \"\$(cat $D/mon.log | tr '\n' ' ')\" = 'inicia para para ' ] || [ \"\$(cat $D/mon.log | tr '\n' ' ')\" = 'inicia para ' ]"
# 3. quente: UM processo com --repete N
roda quente --modelo 3b --texto longo --estado quente --n 5
confere "quente: 1 processo com --repete 5 (3B)" "[ $RC = 0 ] && [ \$(wc -l < $D/log-quente) = 1 ] && grep -q -- '--celula 3b:$D/m3b --so-texto longo --repete 5 --novos 64' $D/log-quente && [ -d $D/s-quente/p1 ]"
confere "quente: o medidas tem 1 aquecimento descartado + 5 medidas" "python3 -I -c \"import json; m=json.load(open('$D/s-quente/p1/saida/medidas-3b.json'))['medidas']; assert len(m)==6 and m[0]['descartada'] and not any(x['descartada'] for x in m[1:])\""
# 4. portão falho: roda mesmo assim (não adia), marca portao_ok false
echo 1 > "$D/portao.rc"
roda portao-falho --modelo 1b --texto curto --estado frio --n 2
confere "portão falho: a célula roda (sem adiar nem repetir) e fica marcada" "[ $RC = 0 ] && [ \$(wc -l < $D/log-portao-falho) = 3 ] && grep -q '\"portao_ok\": false' $D/s-portao-falho/celula.json && grep -q 'CPU ocupada' $D/s-portao-falho/celula.json"
echo 0 > "$D/portao.rc"
# 5. processo com erro encerra a célula: sem repetição automática
export FAKE_FALHA_NA_CHAMADA=2
roda erro --modelo 1b --texto curto --estado frio --n 3
unset FAKE_FALHA_NA_CHAMADA
confere "erro no 2º processo: sai 1, não roda os seguintes, não repete" "[ $RC = 1 ] && [ \$(wc -l < $D/log-erro) = 2 ] && grep -q '\"rc\": 1' $D/s-erro/celula.json && [ \"\$(cat $D/s-erro/p1/rc.txt)\" = 1 ]"
vivos=0; for p in $(cat "$D/mon.pids"); do kill -0 "$p" 2> /dev/null && vivos=1; done
confere "o monitor (amostragem) é parado em todas as células, inclusive a com erro" "[ -s $D/mon.pids ] && [ $vivos = 0 ]"
if [ "$falhas" = 0 ]; then echo "TODOS OS TESTES PASSARAM"; exit 0; else echo "$falhas FALHA(S)"; exit 1; fi
