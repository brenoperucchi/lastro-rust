# Funções comuns dos scripts de abertura da F2E (lidas por source), rodam NA maquina 5090. Sem GPU aqui.
AQ=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
UUID=GPU-00005090-0000-0000-0000-000000005090
JANELA_GPU=${F2E_JANELA_GPU:-$HOME/janela-gpu-strata.sh}
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
export CUDA_TOOLKIT_PATH=${CUDA_TOOLKIT_PATH:-/usr/local/cuda}
case ":$PATH:" in *":$CUDA_TOOLKIT_PATH/bin:"*) ;; *) export PATH="$PATH:$CUDA_TOOLKIT_PATH/bin";; esac
export TMPDIR=${F2E_TMPDIR:-$HOME/.cache/lastro-testes}; mkdir -p "$TMPDIR"
set -C   # noclobber: um log existente nunca é truncado
verifica_fixos() { bash "$AQ/roda-f2e-janela.sh" --verifica-fixos > /dev/null 2>&1 || { echo "FIXOS FALHARAM"; exit 2; }; }
logs_livres() { local s; for s in "$@"; do [ ! -e "$s.log" ] || { echo "log já existe, recusado (nada foi tocado): $s.log"; exit 2; }; done; }
[ -x "$JANELA_GPU" ] || { echo "janela-gpu-strata.sh ausente em $JANELA_GPU: recusado (nada foi tocado)"; exit 2; }
# Coleta aprovada do modelo M em DIR (pasta absoluta): f2e-coleta-resultado.json do MESMO modelo com HC1b "dentro" e o teste de erro em GPU executado, passou, com procedência completa.
confere_coleta() {
  python3 -I - "$1" "$2" <<'PY'
import json, os, sys
m, d = sys.argv[1], sys.argv[2]
arq = os.path.join(d, "f2e-coleta-resultado.json")
try:
    r = json.load(open(arq, encoding="utf-8"))
except Exception as e:
    print("coleta %s: %s ausente ou ilegível (%s)" % (m, arq, e)); sys.exit(1)
h = (r.get("HC1b") or {}) if isinstance(r, dict) else {}
t = (r.get("teste_gpu") or {}) if isinstance(r, dict) else {}
if r.get("modelo") != m:
    print("coleta %s: a pasta é do modelo %r" % (m, r.get("modelo"))); sys.exit(1)
if h.get("classe") != "dentro":
    print("coleta %s: HC1b %r (precisa de 'dentro')" % (m, h.get("classe"))); sys.exit(1)
if not (t.get("executado") is True and t.get("passou") is True and t.get("procedencia_completa") is True):
    print("coleta %s: teste de erro em GPU sem passou com procedência completa (%r)" % (m, t)); sys.exit(1)
PY
}
