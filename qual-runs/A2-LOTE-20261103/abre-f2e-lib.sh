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
# Procedência (nada foi tocado ainda): o clone onde isto roda tem de estar na árvore do commit e (se F2E_COMMIT vier) no commit aprovado; o envelope que para e religa
# o Strata tem de ser byte a byte o revisado (sha de janela-gpu-strata-f2e.sh, versionado ao lado). F2E_JANELA_GPU (envelope de teste) só com F2E_MODO_TESTE=1.
ENVELOPE_SHA_REVISADO=9236a0b21af559bd2c89d717fe4d629b13d6518e8f0865c4316c5300f4ece03f
[ "${F2E_MODO_TESTE:-0}" != 1 ] || ENVELOPE_SHA_REVISADO=${F2E_ENVELOPE_SHA_ESPERADO:-$ENVELOPE_SHA_REVISADO}   # só em teste
if [ -n "${F2E_JANELA_GPU:-}" ] && [ "${F2E_MODO_TESTE:-0}" != 1 ]; then echo "F2E_JANELA_GPU só vale com F2E_MODO_TESTE=1 (envelope de teste): recusado (nada foi tocado)"; exit 2; fi
CLONE_COMMIT=$(git -C "$AQ" rev-parse HEAD 2> /dev/null) || { echo "o clone não é um repositório git legível: recusado (nada foi tocado)"; exit 2; }
if [ "${F2E_MODO_TESTE:-0}" != 1 ]; then
  [ -z "$(git -C "$AQ" status --porcelain --untracked-files=no 2> /dev/null)" ] || { echo "clone com arquivo rastreado modificado: recusado (nada foi tocado)"; exit 2; }
  [ -z "${F2E_COMMIT:-}" ] || [ "$F2E_COMMIT" = "$CLONE_COMMIT" ] || { echo "o clone está em $CLONE_COMMIT, não em F2E_COMMIT=$F2E_COMMIT: recusado (nada foi tocado)"; exit 2; }
fi
ENV_OBS=$(sha256sum "$JANELA_GPU" | cut -d' ' -f1)
[ "$ENV_OBS" = "$ENVELOPE_SHA_REVISADO" ] || { echo "janela-gpu-strata.sh ($JANELA_GPU) com sha $ENV_OBS, o revisado é $ENVELOPE_SHA_REVISADO: recusado (nada foi tocado)"; exit 2; }
export F2E_PROC_ENVELOPE_SHA=$(sha256sum "$JANELA_GPU" | cut -d' ' -f1)
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
