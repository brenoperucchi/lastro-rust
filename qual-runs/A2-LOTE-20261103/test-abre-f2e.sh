#!/usr/bin/env bash
# Teste dos scripts de abertura da F2E (abre-f2e-*.sh) e das travas de procedência da abre-f2e-lib.sh, SEM GPU, SEM Strata: roteiro e envelope são de mentira,
# o clone é um repositório git temporário. Nada é apagado (a pasta temporária é impressa ao fim).
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
export TMPDIR=${TMPDIR:-$HOME/.cache/lastro-testes}; mkdir -p "$TMPDIR"
B=$(mktemp -d) || exit 2
FALHAS=0
ok() { echo "ok   $1"; }
ruim() { echo "FALHA $1"; FALHAS=$((FALHAS + 1)); }
espera() { if [ "$2" = "$3" ]; then ok "$1 (rc $3)"; else ruim "$1: esperado rc $2, veio $3"; fi; }
REPO=$B/clone; mkdir -p "$REPO"
cp "$AQUI/abre-f2e-lib.sh" "$AQUI/abre-f2e-coleta.sh" "$AQUI/abre-f2e-medicao.sh" "$REPO/"
printf '#!/usr/bin/env bash\nexit 0\n' > "$REPO/roda-f2e-janela.sh"
git -C "$REPO" init -q && git -C "$REPO" add -A && git -C "$REPO" -c user.name=t -c user.email=t@t commit -q -m base
COMMIT=$(git -C "$REPO" rev-parse HEAD)
# envelope de mentira: grava os argumentos e a sha vista
printf '#!/usr/bin/env bash\necho "chamado: $*" >> "%s/chamadas.txt"\necho "sha_vista=$F2E_PROC_ENVELOPE_SHA" >> "%s/chamadas.txt"\nexit 0\n' "$B" "$B" > "$B/janela-falsa.sh"; chmod +x "$B/janela-falsa.sh"
SHA_FALSA=$(sha256sum "$B/janela-falsa.sh" | cut -d' ' -f1)
# coletas aprovadas de mentira
for m in 1b 3b; do mkdir -p "$B/coleta-$m"; printf '{"modelo":"%s","HC1b":{"classe":"dentro"},"teste_gpu":{"executado":true,"passou":true,"procedencia_completa":true}}' $m > "$B/coleta-$m/f2e-coleta-resultado.json"; done
HFAKE=$B/home; mkdir -p "$HFAKE"; cp "$B/janela-falsa.sh" "$HFAKE/janela-gpu-strata.sh"
roda() { ( cd "$REPO" && env HOME="$HFAKE" F2E_TMPDIR="$B/tmp" "$@" ); }

# modo real: HOME de mentira com um janela de sha DIFERENTE do revisado -> recusa 2 sem chamar
roda F2E_COMMIT=$COMMIT bash abre-f2e-medicao.sh "$B/coleta-1b" "$B/coleta-3b" > "$B/o1.txt" 2>&1; espera "modo real, envelope com sha diferente do revisado" 2 $?
grep -q "o revisado é" "$B/o1.txt" && ok "mensagem de sha do envelope" || ruim "sem mensagem de sha: $(cat "$B/o1.txt")"
[ ! -e "$B/chamadas.txt" ] && ok "envelope nao foi chamado" || ruim "envelope chamado"
# arvore suja
echo x >> "$REPO/roda-f2e-janela.sh"
roda F2E_COMMIT=$COMMIT bash abre-f2e-medicao.sh "$B/coleta-1b" "$B/coleta-3b" > "$B/o2.txt" 2>&1; espera "arvore com arquivo rastreado modificado" 2 $?
grep -q "arquivo rastreado modificado" "$B/o2.txt" && ok "mensagem de arvore suja" || ruim "sem mensagem: $(cat "$B/o2.txt")"
git -C "$REPO" checkout -q -- roda-f2e-janela.sh
# F2E_COMMIT errado
roda F2E_COMMIT=0000000000000000000000000000000000000000 bash abre-f2e-medicao.sh "$B/coleta-1b" "$B/coleta-3b" > "$B/o3.txt" 2>&1; espera "F2E_COMMIT diferente do clone" 2 $?
grep -q "não em F2E_COMMIT" "$B/o3.txt" && ok "mensagem de commit" || ruim "sem mensagem: $(cat "$B/o3.txt")"
# F2E_COMMIT certo e arvore limpa chega ao teste do envelope (falha so no sha)
roda F2E_COMMIT=$COMMIT bash abre-f2e-medicao.sh "$B/coleta-1b" "$B/coleta-3b" > "$B/o4.txt" 2>&1; espera "commit certo, arvore limpa, mas envelope diferente" 2 $?
grep -q "o revisado é" "$B/o4.txt" && ok "passou pelas travas de clone e parou no envelope" || ruim "o4: $(cat "$B/o4.txt")"
# sem F2E_COMMIT: recusa 2 (obrigatorio fora do modo de teste)
roda bash abre-f2e-medicao.sh "$B/coleta-1b" "$B/coleta-3b" > "$B/o9.txt" 2>&1; espera "sem F2E_COMMIT" 2 $?
grep -q "falta F2E_COMMIT" "$B/o9.txt" && ok "mensagem de F2E_COMMIT obrigatorio" || ruim "o9: $(cat "$B/o9.txt")"
# git status que FALHA (rc 128, stdout vazio) nao passa por arvore limpa
mkdir -p "$B/bin"; GIT_REAL=$(command -v git)
printf '#!/usr/bin/env bash\ncase "$*" in *status*) echo "fatal: index file corrupt" >&2; exit 128;; esac\nexec %s "$@"\n' "$GIT_REAL" > "$B/bin/git"; chmod +x "$B/bin/git"
roda PATH="$B/bin:$PATH" F2E_COMMIT=$COMMIT bash abre-f2e-medicao.sh "$B/coleta-1b" "$B/coleta-3b" > "$B/o10.txt" 2>&1; espera "git status falha (rc 128, stdout vazio)" 2 $?
grep -q "git status falhou" "$B/o10.txt" && ok "mensagem de git status falho" || ruim "o10: $(cat "$B/o10.txt")"
[ ! -e "$B/chamadas.txt" ] && ok "sem chamada" || ruim "chamou"
# o roteiro real recusa as variaveis de teste da abertura no modo real
for V in F2E_MODO_TESTE F2E_ENVELOPE_SHA_ESPERADO F2E_JANELA_GPU; do
  env "$V=1" bash "$AQUI/roda-f2e-janela.sh" --modelos 1b --saida "$B/saida-$V" > "$B/o11.txt" 2>&1; espera "roteiro real recusa $V" 2 $?
  [ ! -e "$B/saida-$V" ] && ok "$V: nada criado" || ruim "$V: criou saida"
done
# F2E_JANELA_GPU sem modo de teste
roda F2E_JANELA_GPU="$B/janela-falsa.sh" bash abre-f2e-medicao.sh "$B/coleta-1b" "$B/coleta-3b" > "$B/o5.txt" 2>&1; espera "F2E_JANELA_GPU sem F2E_MODO_TESTE" 2 $?
[ ! -e "$B/chamadas.txt" ] && ok "sem chamada" || ruim "chamou"
# modo de teste: envelope esperado igual -> chama; diferente -> recusa
roda F2E_MODO_TESTE=1 F2E_JANELA_GPU="$B/janela-falsa.sh" F2E_ENVELOPE_SHA_ESPERADO=$SHA_FALSA bash abre-f2e-medicao.sh "$B/coleta-1b" "$B/coleta-3b" > "$B/o6.txt" 2>&1; espera "modo de teste com sha esperado" 0 $?
grep -q -- "--teto 2700 -- bash .*abre-f2e-medicao.sh --interno" "$B/chamadas.txt" && grep -q "sha_vista=$SHA_FALSA" "$B/chamadas.txt" && ok "chamou o envelope com --teto 2700 e a sha no ambiente" || ruim "chamadas: $(cat "$B/chamadas.txt" 2>&1)"
rm -f "$B/chamadas.txt"
roda F2E_MODO_TESTE=1 F2E_JANELA_GPU="$B/janela-falsa.sh" F2E_ENVELOPE_SHA_ESPERADO=$(printf 'x' | sha256sum | cut -d' ' -f1) bash abre-f2e-medicao.sh "$B/coleta-1b" "$B/coleta-3b" > "$B/o7.txt" 2>&1; espera "modo de teste com sha esperado diferente" 2 $?
[ ! -e "$B/chamadas.txt" ] && ok "sem chamada" || ruim "chamou"
# coleta aprovada ausente na medicao
echo '{"modelo":"1b","HC1b":{"classe":"REGRESSAO"}}' > "$B/coleta-1b/f2e-coleta-resultado.json"
roda F2E_MODO_TESTE=1 F2E_JANELA_GPU="$B/janela-falsa.sh" F2E_ENVELOPE_SHA_ESPERADO=$SHA_FALSA bash abre-f2e-medicao.sh "$B/coleta-1b" "$B/coleta-3b" > "$B/o8.txt" 2>&1; espera "medicao com coleta nao aprovada" 2 $?
[ ! -e "$B/chamadas.txt" ] && ok "sem chamada" || ruim "chamou"
# o envelope versionado tem o sha gravado na lib
grep -q "^ENVELOPE_SHA_REVISADO=$(sha256sum "$AQUI/janela-gpu-strata-f2e.sh" | cut -d' ' -f1)\$" "$AQUI/abre-f2e-lib.sh" && ok "a lib trava o sha de janela-gpu-strata-f2e.sh" || ruim "sha da lib diferente do envelope versionado"
echo "---"
echo "pasta temporaria deste teste (apagar a mao, caminho literal): $B"
[ "$FALHAS" = 0 ] && echo "TODOS OS TESTES PASSARAM" || echo "$FALHAS FALHA(S)"
exit "$FALHAS"
