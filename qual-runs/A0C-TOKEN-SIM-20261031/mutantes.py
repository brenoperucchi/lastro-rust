"""A0c, rodadas 51 e 52: os testes do protocolo token contra 8 mutantes do simulador. Rodar da raiz, com uma pasta vazia de
mktemp -d como argumento (o roteiro escreve nela tools/moe_sim.py mutado e tests/test_moe_sim.py):
  M=$(mktemp -d) && mkdir $M/tools $M/tests && cp tests/test_moe_sim.py $M/tests/ &&
  taskset -c 16-27 tools/.venv-ref/bin/python -I -B qual-runs/A0C-TOKEN-SIM-20261031/mutantes.py $M > .../mutantes.txt"""
import subprocess, sys
src = open("tools/moe_sim.py").read()
M = sys.argv[1]
muts = {
    "M1 cópia adiada começa na hora": ('if self.protocol == "token" and vic is not None and vic[0] >= self.cur_layer:', 'if False:'),
    "M2 acerto pelo residente, não pela tabela": ('if (key in self.table) if self.protocol == "token" else (key in self.resident):', 'if key in self.resident:'),
    "M3 settle em toda camada": ('''            if l > 0:
                self.release_deferred(l - 1)  # o doorbell de l mostra que a GPU passou dos experts de l − 1''', '''            if l > 0:
                self.release_deferred(l - 1)  # o doorbell de l mostra que a GPU passou dos experts de l − 1
                self.settle(); self.table = frozenset(self.resident)'''),
    "M4 barreira com backlog pelo tempo": ('pend = len(self.inflight) if self.token_barreira else', 'pend = 0 if False else'),
    "M5 tabela publicada depois do g": ('''            if l == 0:
                self.begin_token()  # a tabela vai para a GPU antes do primeiro trabalho do token
            self.now += p["g"]''', '''            self.now += p["g"]
            if l == 0:
                self.begin_token()'''),
    "M6 parada da barreira fora da espera": ('''                self.stats["wait_ms"] += fim - self.now
''', ''),
    "M7 adiada liberada no fim da própria camada": ('''        if self.protocol == "token" and l == self.L - 1:
            self.release_deferred(l)  # o doorbell do fim do token''', '''        if self.protocol == "token":
            self.release_deferred(l)'''),
    "M8 admissão no começo da camada, antes da CPU": ('''            self.now = cpu_t
            for key in adm:''', '''            self.now = start
            for key in adm:'''),
}
for name, (a, b) in muts.items():
    assert src.count(a) == 1, name
    open(f"{M}/tools/moe_sim.py", "w").write(src.replace(a, b))
    r = subprocess.run([sys.executable, "-I", "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", f"{M}/tests/test_moe_sim.py",
                        "-k", "token or barreira or vitima or nao_muda or primeiro_g or ultima"], capture_output=True, text=True)
    falhas = [l.split("::")[1].split(" ")[0] for l in r.stdout.splitlines() if l.startswith("FAILED")]
    print(name, "->", falhas or "NENHUM TESTE FALHOU")
