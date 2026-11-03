"""Seção 5 do criterio.txt: roda de novo o prefetch_sim.py da A0a com c_cpu, c_cpu_quiet e o_layer da janela (medianas de ajuste.txt).
Nada mais muda. Dois conjuntos, para não esconder a escolha feita depois dos dados (rodadas 26 e 27):
  cinco       medianas das 5 repetições dos dois modelos (o que o criterio.txt manda); no 3B a rep5 oscila e entra com a volta 12.
  convergidas 1B: as 5 repetições; 3B: só rep1, rep3 e rep4, as que convergiram (regra da rodada 26, de depois dos dados).
Uso, da raiz do projeto (o padrão é "cinco"):
  taskset -c 16-27 tools/.venv-ref/bin/python -I -B qual-runs/A1-MEDIDA-20261101/prefetch_novo.py [cinco|convergidas] > saída"""
import runpy, sys
sys.path.insert(0, "tools")
import moe_sim as m  # noqa: E402
CONJUNTOS = {
    "cinco": {"1b": dict(c_cpu=0.0492, c_cpu_quiet=0.0393, o_layer=0.1314),
              "3b": dict(c_cpu=0.0684, c_cpu_quiet=0.0584, o_layer=0.0870)},
    "convergidas": {"1b": dict(c_cpu=0.0492, c_cpu_quiet=0.0393, o_layer=0.1314),
                    "3b": dict(c_cpu=0.0678, c_cpu_quiet=0.0586, o_layer=0.0870)},
}
escolha = sys.argv[1] if len(sys.argv) > 1 else "cinco"
NOVOS = CONJUNTOS[escolha]
for mod, v in NOVOS.items():
    m.PARAMS_RUNTIME[mod].update(v)
print("conjunto:", escolha, "parâmetros novos:", NOVOS)
runpy.run_path("qual-runs/A0-PREFETCH-SIM-20261031/prefetch_sim.py", run_name="__main__")
