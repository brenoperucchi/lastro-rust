#!/usr/bin/env bash
# Teste do roda-5e-janela.sh com systemctl FALSO (nenhum timer real é pausado), servidor `lastro-servidor` de MENTIRA (um HTTP em python que responde com a referência),
# textos, referências e modelos falsos e SEM GPU. curl e ss são os REAIS (a rede é o loopback da máquina); um caso usa curl falso. A composição janela-e4b.sh +
# roda-com-limite.sh é a real (unidades transitórias de `systemd --user`). Cria pasta com mktemp -d e NÃO apaga nada.
set -u
unset E4B_GPU_AUTORIZADO E4B_MEDIDA_AUTORIZADA MOE_RT_REQUIRE_GPU LEITURA_CORRECAO_AUTORIZADA LEITURA_VELOCIDADE_AUTORIZADA PRIMEIRA_PALAVRA_AUTORIZADA
export CUDA_VISIBLE_DEVICES=
AQUI=$(cd "$(dirname "$0")" && pwd -P)
J=$AQUI/roda-5e-janela.sh
UUID=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd
D=$(mktemp -d -p "${TMPDIR:-/tmp}" janela-5e-teste.XXXXXX) || exit 1
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
echo criterio5e > "$D/c5e.md"; echo criterio5c > "$D/c5c.md"
# Servidor falso (python): mesmos caminhos, status e corpos do real; modos em FALSO_MODO: ok, diff (3b longo com "!"), fora (escuta em 0.0.0.0), naoencerra (responde 200 ao
# /encerrar mas fica vivo e ignora TERM), rc1 (sai com 1 depois do /encerrar), muda (altera um texto congelado ao subir).
cat > "$D/falso.py" <<'PYF'
import http.server, json, os, signal, socket, sys, threading
args = sys.argv[1:]
modelo = os.path.basename(args[args.index("--modelo") + 1]); porta = int(args[args.index("--porta") + 1])
modo = os.environ.get("FALSO_MODO", "ok")
with open(os.environ["FALSO_LOG"], "a") as f: f.write("%s %s %d pid=%d\n" % (modo, modelo, porta, os.getpid()))
if modo == "muda":
    with open(os.environ["TEXTOS_FALSOS"] + "/texto-curto.txt", "a") as f: f.write("mudou")
if modo == "naoencerra": signal.signal(signal.SIGTERM, signal.SIG_IGN)
textos = {n: open("%s/texto-%s.txt" % (os.environ["TEXTOS_FALSOS"], n), "rb").read() for n in ("curto", "medio", "longo")}
refs = {c["nome"]: c["texto_gerado"] for c in json.load(open("%s/ref5c-%s.json" % (os.environ["REFS_FALSAS"], modelo), encoding="utf-8"))["casos"]}
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def resp(self, st, corpo):
        self.send_response(st); self.send_header("Content-Type", "text/plain; charset=utf-8"); self.send_header("Content-Length", str(len(corpo))); self.send_header("Connection", "close"); self.end_headers(); self.wfile.write(corpo); self.close_connection = True
    def do_GET(self):
        self.resp(200, b"ok\n") if self.path == "/saude" else self.resp(404, b"caminho desconhecido\n")
    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0")); corpo = self.rfile.read(n)
        if self.path == "/encerrar" and modo == "fecha":
            self.close_connection = True; return          # recebe, fecha SEM resposta e continua vivo
        if self.path == "/encerrar" and modo == "silencio":
            import time; time.sleep(60); return           # recebe e fica mudo, vivo
        if self.path == "/encerrar":
            self.resp(200, b"ate logo\n")
            if modo != "naoencerra": threading.Thread(target=lambda: os._exit(1 if modo == "rc1" else 0)).start()
            return
        if not self.path.startswith("/gerar?novos=64&data=November%2007%2C%202026"): return self.resp(400, b"parametros\n")
        if not corpo.strip(): return self.resp(400, "entrada vazia ou só espaços\n".encode())
        nome = [k for k, v in textos.items() if v == corpo]
        if not nome: return self.resp(400, b"texto desconhecido\n")
        t = refs[nome[0]]
        if modo == "diff" and modelo == "3b" and nome[0] == "longo": t += "!"
        self.resp(200, (t + "\n").encode())
class S(http.server.HTTPServer):
    allow_reuse_address = True
s = S(("0.0.0.0" if modo == "fora" else "127.0.0.1", porta), H)
sys.stderr.write("escutando em 127.0.0.1:%d\n" % porta); sys.stderr.flush()
s.serve_forever()
PYF
cat > "$D/falso" <<F
#!/usr/bin/env bash
exec python3 -I "$D/falso.py" "\$@"
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
roda() { # nome [args extras...]; define RC
  local n=$1; shift
  export FAKE_DIR=$D/$n FALSO_LOG=$D/log-$n; : > "$FALSO_LOG"
  mkdir -p "$FAKE_DIR"
  echo active > "$FAKE_DIR/a.timer"; echo active > "$FAKE_DIR/b.timer"; echo inactive > "$FAKE_DIR/a.service"; echo inactive > "$FAKE_DIR/b.service"; : > "$FAKE_DIR/log"
  SYSTEMCTL="$D/fake-systemctl" PENDENTE="$D/pend-$n" JANELA_PASSO_ESPERA=0.1 XDG_RUNTIME_DIR=/run/user/fake JANELA_TIMERS="$D/timers.txt" \
    MODELO_1B=$D/m/1b MODELO_3B=$D/m/3b TEXTOS_5C=$D/textos REF_5C=$D/refs CRITERIO_5E=$D/c5e.md CRITERIO_5C=$D/c5c.md RECUSAS_SH=$D/recusas-falso SEM_SHA_FIXO=${SEM_SHA:-1} \
    "$J" --bin "${BIN_FALSO:-$D/falso}" --saida "$D/saida-$n" --limite "${LIM:-60}" "$@" > "$D/out-$n" 2>&1
  RC=$?
}
rotulo() { grep '^rótulo:' "$D/saida-$1/decisao.txt" 2>/dev/null | cut -d' ' -f2-; }
# nenhum servidor falso vivo (pids do log)
sem_vivos() { local p; for p in $(grep -o 'pid=[0-9]*' "$D/log-$1" | cut -d= -f2); do kill -0 "$p" 2> /dev/null && return 1; done; return 0; }

# 1. recusas ANTES de pausar timers ou subir o servidor
roda sem-portao; confere "sem portão: recusa 2, servidor e timers intocados" "[ $RC = 2 ] && [ ! -s $D/log-sem-portao ] && ! grep -q '^stop' $D/sem-portao/log"
export PRIMEIRA_PALAVRA_AUTORIZADA=1
roda sem-uuid; confere "sem UUID: recusa 2" "[ $RC = 2 ] && [ ! -s $D/log-sem-uuid ]"
export CUDA_VISIBLE_DEVICES=GPU-00000000-0000-0000-0000-000000000000
roda uuid-errado; confere "UUID errado: recusa 2" "[ $RC = 2 ] && [ ! -s $D/log-uuid-errado ]"
export CUDA_VISIBLE_DEVICES=$UUID
export PRIMEIRA_PALAVRA_AUTORIZADA=0
roda portao-zero; confere "portão diferente de 1: recusa 2" "[ $RC = 2 ] && [ ! -s $D/log-portao-zero ]"
export PRIMEIRA_PALAVRA_AUTORIZADA=1
mkdir -p "$D/saida-ocupada"; echo x > "$D/saida-ocupada/a"
roda ocupada; confere "--saida não vazia: recusa 2 sem subir o servidor" "[ $RC = 2 ] && [ ! -s $D/log-ocupada ]"
roda limite-alto --limite 1801; confere "limite acima de 1800: recusa 2" "[ $RC = 2 ] && [ ! -s $D/log-limite-alto ]"
roda opcao-ruim --nao-existe; confere "opção desconhecida: recusa 2" "[ $RC = 2 ] && [ ! -s $D/log-opcao-ruim ]"

# 2. caminho feliz: PASSA
roda ok
confere "ok: sai 0 e PASSA" "[ $RC = 0 ] && [ \"\$(rotulo ok)\" = PASSA ]"
confere "ok: um servidor por modelo, 1b depois 3b, portas livres em 127.0.0.1" "[ \$(wc -l < $D/log-ok) = 2 ] && sed -n 1p $D/log-ok | grep -q '^ok 1b ' && sed -n 2p $D/log-ok | grep -q '^ok 3b '"
confere "ok: 13 verificações por modelo" "grep -q 'verificações que passaram: 1b 13 de 13, 3b 13 de 13' $D/saida-ok/decisao.txt"
confere "ok: corpo igual byte a byte (referência + newline), inclusive acentos e o médio repetido" "[ \"\$(cat $D/saida-ok/1b-medio.body; printf x)\" = \"\$(printf 'resp 1b medio: çã\n  fim\nx')\" ] && [ \"\$(cat $D/saida-ok/3b-longo.body; printf x)\" = \"\$(printf 'resp 3b longo: çã\n  fim\nx')\" ] && cmp -s $D/saida-ok/3b-medio.body $D/saida-ok/3b-medio-repetido.body"
confere "ok: socket só em 127.0.0.1 (ss gravado), saúde, recusa 400 e encerrar 200 na decisão" "grep -q '127.0.0.1' $D/saida-ok/1b-ss.txt && grep -q 'GET /saude (200, ok): PASSA' $D/saida-ok/decisao.txt && grep -q 'recusa de corpo vazio (400): PASSA' $D/saida-ok/decisao.txt && grep -q 'POST /encerrar (200, ate logo): PASSA' $D/saida-ok/decisao.txt"
confere "ok: encerrou com código 0, porta fechada, sem processo" "grep -q '1b: encerrou com código 0 em até 10 s: PASSA' $D/saida-ok/decisao.txt && grep -q '3b: porta .* fechada: PASSA' $D/saida-ok/decisao.txt && grep -q '3b: nenhum processo remanescente: PASSA' $D/saida-ok/decisao.txt"
confere "ok: tempos descritivos (partida e cada POST) e integridade na decisão" "grep -q \"1b partida até 'escutando': \" $D/saida-ok/decisao.txt && grep -q '3b POST /gerar medio-repetido: HTTP 200, ' $D/saida-ok/decisao.txt && grep -q 'integridade: íntegra' $D/saida-ok/decisao.txt && grep -q 'recusas de partida: OK' $D/saida-ok/decisao.txt"
confere "ok: sem servidor vivo, timers parados e restaurados, carga ao lado da pasta" "sem_vivos ok && grep -q '^stop a.timer' $D/ok/log && [ \"\$(cat $D/ok/a.timer)\" = active ] && grep -q 'RESTAURAÇÃO OK' $D/saida-ok-estado-janela/restauracao.txt && [ -f $D/saida-ok.carga-antes.txt ] && [ -f $D/saida-ok.carga-depois.txt ]"

# 3. REPROVA: corpo diferente
export FALSO_MODO=diff; roda dif
confere "corpo diferente (3b longo): REPROVA e o 1b segue 13 de 13" "[ \"\$(rotulo dif)\" = REPROVA ] && grep -q '3b: POST /gerar longo (corpo diferente da referência.*REPROVA' $D/saida-dif/decisao.txt && grep -q '1b 13 de 13' $D/saida-dif/decisao.txt && sem_vivos dif"
# 4. REPROVA: socket fora de 127.0.0.1
export FALSO_MODO=fora; roda fora
confere "escuta em 0.0.0.0: REPROVA (socket)" "[ \"\$(rotulo fora)\" = REPROVA ] && grep -q 'socket (escuta fora de 127.0.0.1: 0.0.0.0:' $D/saida-fora/decisao.txt && sem_vivos fora"
# 5. REPROVA: processo que não encerra (e não sobra processo)
export FALSO_MODO=naoencerra; roda nao
confere "não encerra: REPROVA, derrubado por TERM/KILL, nenhum servidor vivo" "[ \"\$(rotulo nao)\" = REPROVA ] && grep -q 'NÃO encerrou em 10 s' $D/saida-nao/decisao.txt && sem_vivos nao"
export FALSO_MODO=rc1; roda rc1
confere "encerra com código 1: REPROVA" "[ \"\$(rotulo rc1)\" = REPROVA ] && grep -q 'encerrou com código 1' $D/saida-rc1/decisao.txt"
unset FALSO_MODO

# 5b. /encerrar ENVIADO sem resposta e servidor vivo: REPROVA (processo e porta eliminados); mudo respeita o orçamento de 10 s; falha de CONECTAR: SEM PROVA
porta_viva() { local p; for p in $(grep -o ' [0-9]* pid=' "$D/log-$1" | tr -d ' pid='); do ss -H -ltn "sport = :$p" | grep -q . && return 0; done; return 1; }
export FALSO_MODO=fecha; roda fecha
confere "/encerrar recebido, conexão fechada sem resposta, servidor vivo: REPROVA (não SEM PROVA), processo e porta eliminados" "[ \"\$(rotulo fecha)\" = REPROVA ] && grep -q 'pedido ENVIADO' $D/saida-fecha/decisao.txt && grep -q 'NÃO encerrou em 10 s' $D/saida-fecha/decisao.txt && ! grep -q 'POST /encerrar.*SEM PROVA' $D/saida-fecha/decisao.txt && sem_vivos fecha && ! porta_viva fecha && grep -q '3b: nenhum processo remanescente: PASSA' $D/saida-fecha/decisao.txt"
export FALSO_MODO=silencio; t0=$(date +%s); roda silencio; t1=$(date +%s)
confere "/encerrar recebido e mudo: REPROVA dentro do orçamento de 10 s por modelo (não espera os 600 s)" "[ \"\$(rotulo silencio)\" = REPROVA ] && grep -q 'pedido ENVIADO' $D/saida-silencio/decisao.txt && [ \$((t1 - t0)) -ge 20 ] && [ \$((t1 - t0)) -lt 50 ] && sem_vivos silencio && ! porta_viva silencio"
unset FALSO_MODO
printf '#!/usr/bin/env bash\nfor a in "$@"; do case $a in */encerrar) printf "000 0.000000"; echo "curl: (7) Failed to connect" >&2; exit 7;; esac; done\nexec curl "$@"\n' > "$D/curl-sem-conectar"; chmod +x "$D/curl-sem-conectar"
CURL=$D/curl-sem-conectar roda semconectar
confere "falha real de CONECTAR no /encerrar: SEM PROVA (não REPROVA), servidor derrubado" "[ \"\$(rotulo semconectar)\" = 'SEM PROVA' ] && grep -q 'falha de CONECTAR' $D/saida-semconectar/decisao.txt && ! grep -q ': REPROVA$' $D/saida-semconectar/decisao.txt && sem_vivos semconectar"

# 5c. decisao.txt traz as linhas de systemd e de 500; listas do systemd antes e depois ao lado da pasta
confere "decisao.txt: unidade transitória do limite e 500 = REPROVA" "grep -q 'única unidade criada pelo roteiro além da pausa e restauração dos backups é a transitória do limite de tempo (roda-com-limite.sh, systemd-run --collect, sem enable e sem boot)' $D/saida-ok/decisao.txt && grep -q 'status 500 do motor conta como status errado e vira REPROVA' $D/saida-ok/decisao.txt"
confere "systemctl --user list-unit-files e list-timers gravados antes e depois ao lado da pasta" "[ -s $D/saida-ok.unidades-antes.txt ] && [ -s $D/saida-ok.unidades-depois.txt ] && [ -s $D/saida-ok.timers-antes.txt ] && [ -s $D/saida-ok.timers-depois.txt ]"

# 6. recusas de partida falhando
echo 1 > "$D/recusas.rc"; roda rec1
confere "recusas falhando (1): REPROVA" "[ \"\$(rotulo rec1)\" = REPROVA ] && grep -q 'recusas de partida: FALHARAM' $D/saida-rec1/decisao.txt"
echo 3 > "$D/recusas.rc"; roda rec3
confere "recusas sem resultado (3): SEM PROVA" "[ \"\$(rotulo rec3)\" = 'SEM PROVA' ]"
echo 0 > "$D/recusas.rc"

# 7. SEM PROVA: integridade (texto muda durante a janela) e sha congelado diferente
export FALSO_MODO=muda; roda mud
confere "texto alterado durante a janela: integridade quebrada, SEM PROVA" "[ \"\$(rotulo mud)\" = 'SEM PROVA' ] && grep -q 'integridade: QUEBRADA' $D/saida-mud/decisao.txt"
printf 'Pergunta curta com acento: ação?' > "$D/textos/texto-curto.txt"
unset FALSO_MODO
SEM_SHA=0 roda fixo
confere "sha diferente do congelado: SEM PROVA e a nota diz qual" "[ \"\$(rotulo fixo)\" = 'SEM PROVA' ] && grep -q 'não o congelado' $D/saida-fixo/decisao.txt"

# 8. SEM PROVA: curl falso que não conecta (sem resposta) e servidor que morre antes de escutar
printf '#!/usr/bin/env bash\nexit 7\n' > "$D/curl-ruim"; chmod +x "$D/curl-ruim"
CURL=$D/curl-ruim roda semcurl
confere "sem resposta do servidor (curl falha): SEM PROVA, sem servidor vivo" "[ \"\$(rotulo semcurl)\" = 'SEM PROVA' ] && sem_vivos semcurl"
printf '#!/usr/bin/env bash\necho "motor falhou" >&2; exit 1\n' > "$D/morre"; chmod +x "$D/morre"
BIN_FALSO=$D/morre roda saiu
confere "servidor que sai antes de escutar: SEM PROVA com o código" "[ \"\$(rotulo saiu)\" = 'SEM PROVA' ] && grep -q 'saiu com código 1 antes de escutar' $D/saida-saiu/decisao.txt"

# 9. TRAVA: limite de tempo estourado com o servidor no ar (curl falso que dorme): 124, nenhum servidor vivo, timers restaurados
printf '#!/usr/bin/env bash\nsleep 40\n' > "$D/curl-lento"; chmod +x "$D/curl-lento"
export LIMITE_PASSA="$LIMITE_PASSA CURL"
LIM=3 CURL=$D/curl-lento roda lim
confere "limite estourado: 124, sem decisão, servidor derrubado e timers restaurados" "[ $RC = 124 ] && [ ! -f $D/saida-lim/decisao.txt ] && [ -s $D/log-lim ] && sleep 1 && sem_vivos lim && [ \"\$(cat $D/lim/a.timer)\" = active ]"
if [ "$falhas" = 0 ]; then echo "TODOS OS TESTES PASSARAM"; exit 0; else echo "$falhas FALHA(S)"; exit 1; fi
