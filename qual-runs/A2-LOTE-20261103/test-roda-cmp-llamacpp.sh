#!/usr/bin/env bash
# Teste do roda-cmp-llamacpp.sh com `llama-server` FALSO (HTTP em python: /health, /completion com timings e tokens; log com 'warming up' e 'model loaded') e monitor FALSO.
# curl é o REAL (loopback). SEM GPU. mktemp -d, sem rm -rf.
set -u
unset PRIMEIRA_PALAVRA_AUTORIZADA
export CUDA_VISIBLE_DEVICES=
AQUI=$(cd "$(dirname "$0")" && pwd -P)
L=$AQUI/roda-cmp-llamacpp.sh
UUID=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd
D=$(mktemp -d -p "${TMPDIR:-/tmp}" cmp-llama-teste.XXXXXX) || exit 1
echo "pasta de teste: $D"
falhas=0
confere() { if eval "$2"; then echo "ok: $1"; else echo "FALHOU: $1"; falhas=$((falhas+1)); fi; }
cat > "$D/srv.py" <<'PY'
import http.server, json, os, signal, sys, threading, time
a = sys.argv[1:]
if "--version" in a:
    print("version: 1 (0893ec9)"); sys.exit(0)
modo = os.environ.get("FAKE_MODO", "ok")
host = a[a.index("--host") + 1]; porta = int(a[a.index("--port") + 1])
open(os.environ["FAKE_LOG"], "a").write("pid=%d host=%s porta=%d args=%s\n" % (os.getpid(), host, porta, " ".join(a)))
if modo == "naoencerra": signal.signal(signal.SIGTERM, signal.SIG_IGN)
if modo == "morre": sys.stderr.write("falha ao carregar\n"); sys.exit(1)
pronto = [False]
print("load_tensors: loading", flush=True)
if "--no-warmup" not in a: print("common_init_from_params: warming up the model with an empty run", flush=True)
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *x): pass
    def resp(self, st, corpo):
        b = corpo if isinstance(corpo, bytes) else corpo.encode()
        self.send_response(st); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.send_header("Connection", "close"); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health": self.resp(200 if pronto[0] else 503, "{}")
        else: self.resp(404, "{}")
    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0")); corpo = self.rfile.read(n)
        open(os.environ["FAKE_LOG"] + ".body", "ab").write(corpo + b"\n")
        if self.path != "/completion": return self.resp(404, "{}")
        if modo == "erro500": return self.resp(500, "{\"error\": \"x\"}")
        self.resp(200, json.dumps({"content": "txt", "tokens": list(range(64)), "timings": {"prompt_n": len(json.loads(corpo)["prompt"]), "prompt_ms": 10.0, "predicted_n": 64, "predicted_ms": 126.0, "predicted_per_token_ms": 2.0}}))
class S(http.server.HTTPServer): allow_reuse_address = True
s = S((host, porta), H)
def sobe():
    time.sleep(0.3)
    if modo != "semprontidao":
        pronto[0] = True; print("srv load_model: model loaded", flush=True)
threading.Thread(target=sobe, daemon=True).start()
s.serve_forever()
PY
printf '#!/usr/bin/env bash\nexec python3 -I %s "$@"\n' "$D/srv.py" > "$D/llama-server"; chmod +x "$D/llama-server"
mkdir -p "$D/gguf"; : > "$D/gguf/granite-3.1-1b-a400m-instruct-bf16.gguf"; : > "$D/gguf/granite-3.1-3b-a800m-instruct-bf16.gguf"
python3 -I - "$D/ids.json" <<'PY'
import json, sys
json.dump({"modelos": {m: {t: {"ids": list(range(n))} for t, n in (("curto", 89), ("medio", 170), ("longo", 364))} for m in ("1b", "3b")}}, open(sys.argv[1], "w"))
PY
cat > "$D/mon" <<F
#!/usr/bin/env bash
case \$1 in
  portao) echo "portao=OK" > "\$3"; exit 0;;
  inicia) echo "t, 1, 2000, 10000, 60, 100" > "\$3"; sleep 100 & echo \$! > "\$5"; echo \$! >> $D/mon.pids;;
  para) p=\$(cat "\$3" 2>/dev/null); [ -z "\$p" ] || kill "\$p" 2>/dev/null;;
  avalia) echo "amostras=1 temp_max=60"; exit 0;;
esac
F
chmod +x "$D/mon"
export MONITOR_SH=$D/mon FAKE_LOG=$D/log LLAMA_SERVER=$D/llama-server GGUF_DIR=$D/gguf ESPERA_PRONTO=20
roda() { local n=$1; shift; rm -f "$D/log" "$D/log.body"; : > "$D/log"; "$L" --ids "$D/ids.json" --saida "$D/s-$n" "$@" > "$D/out-$n" 2>&1; RC=$?; cp "$D/log" "$D/log-$n"; cp "$D/log.body" "$D/body-$n" 2> /dev/null; }
sem_vivos() { local p; for p in $(grep -o 'pid=[0-9]*' "$D/log-$1" | cut -d= -f2); do kill -0 "$p" 2> /dev/null && return 1; done; return 0; }
# 1. falha fechada
roda sem-portao --modelo 1b --texto curto --estado frio --serie S1 --n 1
confere "sem autorização: recusa 2, servidor não subiu, sem pasta" "[ $RC = 2 ] && [ ! -s $D/log-sem-portao ] && [ ! -e $D/s-sem-portao ]"
export PRIMEIRA_PALAVRA_AUTORIZADA=1
roda sem-uuid --modelo 1b --texto curto --estado frio --serie S1 --n 1
confere "sem UUID: recusa 2" "[ $RC = 2 ] && [ ! -s $D/log-sem-uuid ]"
export CUDA_VISIBLE_DEVICES=GPU-00000000-0000-0000-0000-000000005090
roda uuid-5090 --modelo 1b --texto curto --estado frio --serie S1 --n 1
confere "UUID de outra GPU (a 5090 nunca): recusa 2, servidor não subiu" "[ $RC = 2 ] && [ ! -s $D/log-uuid-5090 ]"
export CUDA_VISIBLE_DEVICES=$UUID
roda serie-ruim --modelo 1b --texto curto --estado frio --serie S9 --n 1
confere "série inválida: recusa 2" "[ $RC = 2 ] && [ ! -s $D/log-serie-ruim ]"

# 2. S1 frio 1B: N+1 servidores, cada um com --no-warmup, 1 pedido cada
roda s1f --modelo 1b --texto curto --estado frio --serie S1 --n 2
confere "S1 frio: sai 0, 3 servidores (p0 descartado + 2), 1 pedido em cada, nenhum vivo" "[ $RC = 0 ] && [ \$(wc -l < $D/log-s1f) = 3 ] && [ -s $D/s-s1f/resp-0.json ] && [ -s $D/s-s1f/resp-2.json ] && [ ! -e $D/s-s1f/resp-3.json ] && sem_vivos s1f"
confere "S1: -ngl 99 -fa on, KV f16, -t 8 -c 2048 -np 1 -b 2048 -ub 512, --host 127.0.0.1 --no-webui, sem --n-cpu-moe, sem --metrics, sem --no-op-offload" "grep -q -- '-ngl 99 -fa on --cache-type-k f16 --cache-type-v f16' $D/s-s1f/cmdline.txt && grep -q -- '-c 2048 -np 1 -b 2048 -ub 512 -t 8 --host 127.0.0.1 --no-webui' $D/s-s1f/cmdline.txt && ! grep -q -- 'n-cpu-moe\|--metrics\|no-op-offload' $D/s-s1f/cmdline.txt"
confere "frio: --no-warmup na linha de comando e o log não tem 'warming up'" "grep -q -- '--no-warmup' $D/s-s1f/cmdline.txt && ! grep -qi 'warming up' $D/s-s1f/servidor-0.log $D/s-s1f/servidor-1.log $D/s-s1f/servidor-2.log"
confere "servidores só em 127.0.0.1 (o falso registra o host recebido)" "! grep -q 'host=0.0.0.0' $D/log-s1f && [ \$(grep -c 'host=127.0.0.1' $D/log-s1f) = 3 ]"
confere "pedido: array de ids (89), n_predict 64, temperature 0, top_k 1, seed 0, cache_prompt false, return_tokens true, ignore_eos false, stream false" "python3 -I -c \"import json; b=json.load(open('$D/s-s1f/pedido.json')); assert b['prompt']==list(range(89)); assert (b['n_predict'],b['temperature'],b['top_k'],b['seed'],b['cache_prompt'],b['return_tokens'],b['ignore_eos'],b['stream'])==(64,0,1,0,False,True,False,False)\" && [ \$(wc -l < $D/body-s1f) = 3 ]"
confere "grava version.txt, carga por servidor, resp .rc/.ms e celula.json (n_medidas 2, portao_ok, rc 0)" "grep -q 0893ec9 $D/s-s1f/version.txt && [ -s $D/s-s1f/carga-0.ms ] && [ -s $D/s-s1f/carga-2.ms ] && [ \"\$(cat $D/s-s1f/resp-1.rc)\" = '0 200' ] && [ -s $D/s-s1f/resp-1.ms ] && grep -q '\"serie\": \"S1\", \"modelo\": \"1b\", \"texto\": \"curto\", \"estado\": \"frio\", \"n_medidas\": 2, \"portao_ok\": true' $D/s-s1f/celula.json && grep -q '\"rc\": 0' $D/s-s1f/celula.json"
confere "log de KV/flash attention separado (kv-fa.txt existe)" "[ -e $D/s-s1f/kv-fa.txt ]"

# 3. S1 quente: UM servidor com aquecimento padrão, resp-0 descartado + N medidas
roda s1q --modelo 1b --texto medio --estado quente --serie S1 --n 3
confere "S1 quente: 1 servidor, 4 pedidos (aquecimento nosso + 3), sem --no-warmup, log COM 'warming up'" "[ $RC = 0 ] && [ \$(wc -l < $D/log-s1q) = 1 ] && [ \$(wc -l < $D/body-s1q) = 4 ] && [ -s $D/s-s1q/resp-3.json ] && ! grep -q -- '--no-warmup' $D/s-s1q/cmdline.txt && grep -qi 'warming up' $D/s-s1q/servidor-0.log && sem_vivos s1q"
# 4. S2: --n-cpu-moe 22 no 1B e 29 no 3B; S3: -fa off, KV f32
roda s2-1b --modelo 1b --texto curto --estado quente --serie S2 --n 1
roda s2-3b --modelo 3b --texto longo --estado quente --serie S2 --n 1
confere "S2: --n-cpu-moe 22 (1B) e 29 (3B), com -ngl 99 -fa on e KV f16" "grep -q -- '-ngl 99 -fa on --n-cpu-moe 22 --cache-type-k f16' $D/s-s2-1b/cmdline.txt && grep -q -- '-ngl 99 -fa on --n-cpu-moe 29 --cache-type-k f16' $D/s-s2-3b/cmdline.txt && grep -q 'granite-3.1-3b-a800m-instruct-bf16.gguf' $D/s-s2-3b/cmdline.txt && [ \$(python3 -I -c \"import json; print(len(json.load(open('$D/s-s2-3b/pedido.json'))['prompt']))\") = 364 ]"
roda s3 --modelo 1b --texto curto --estado quente --serie S3 --n 1
confere "S3: -ngl 99 -fa off, KV f32 (k e v), sem --n-cpu-moe" "grep -q -- '-ngl 99 -fa off --cache-type-k f32 --cache-type-v f32' $D/s-s3/cmdline.txt && ! grep -q 'n-cpu-moe' $D/s-s3/cmdline.txt"

# 5. servidor sempre encerrado: pedido com erro HTTP 500
FAKE_MODO=erro500 roda e500 --modelo 1b --texto curto --estado quente --serie S1 --n 2
confere "HTTP 500: sai 1, célula marcada, servidor encerrado, sem processo com a porta" "[ $RC = 1 ] && grep -q '\"rc\": 1' $D/s-e500/celula.json && [ \"\$(cat $D/s-e500/resp-0.rc)\" = '0 500' ] && sem_vivos e500"
# 6. sem marcador de pronto: sai 1 e encerra
ESPERA_PRONTO=2 FAKE_MODO=semprontidao roda semp --modelo 1b --texto curto --estado frio --serie S1 --n 1
confere "sem 'model loaded' nem /health 200 em 2 s: sai 1, servidor derrubado, não repete" "[ $RC = 1 ] && [ \$(wc -l < $D/log-semp) = 1 ] && grep -q 'sem marcador de pronto' $D/s-semp/celula.json && sem_vivos semp"
FAKE_MODO=morre roda morre --modelo 1b --texto curto --estado frio --serie S1 --n 1
confere "servidor que morre na partida: sai 1, célula marcada" "[ $RC = 1 ] && grep -q 'saiu antes de ficar pronto' $D/s-morre/celula.json && sem_vivos morre"
# 7. servidor que ignora TERM: derrubado por KILL, sem processo remanescente
FAKE_MODO=naoencerra roda naoenc --modelo 1b --texto curto --estado quente --serie S1 --n 1
confere "servidor que ignora TERM: derrubado por KILL e nenhum processo vivo" "[ $RC = 0 ] && sem_vivos naoenc && ! pgrep -f 'srv.py.*--port' > /dev/null"
# 8. o monitor é parado em todas as células
vivos=0; for p in $(cat "$D/mon.pids"); do kill -0 "$p" 2> /dev/null && vivos=1; done
confere "amostragem parada ao fim de todas as células" "[ -s $D/mon.pids ] && [ $vivos = 0 ]"
if [ "$falhas" = 0 ]; then echo "TODOS OS TESTES PASSARAM"; exit 0; else echo "$falhas FALHA(S)"; exit 1; fi
