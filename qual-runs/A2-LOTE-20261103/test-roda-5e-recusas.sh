#!/usr/bin/env bash
# Teste do roda-5e-recusas.sh com binário FALSO (sem GPU): (1) binário que recusa como o real: tudo OK, código 0; (2) um caso bloqueia: prazo individual, código 1, sem
# processo remanescente; (3) binário que escuta na porta em vez de recusar: FALHOU (porta escutando) e código 1; (4) código errado: código 1. Pasta mktemp -d, não apaga nada.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
D=$(mktemp -d -p "${TMPDIR:-/tmp}" recusas-5e-teste.XXXXXX) || exit 1
falhas=0
confere() { if eval "$2"; then echo "ok: $1"; else echo "FALHOU: $1"; falhas=$((falhas+1)); fi; }
cat > "$D/falso" <<'F'
#!/usr/bin/env bash
# Recusa como o real (código 2, mensagem útil, saída vazia). Modos: FALSO_MODO=bloqueia (modelo inexistente trava, com descendente), escuta (abre a porta e fica),
# rc1 (sai com 1 em vez de 2).
MSG="lastro-servidor: recusado: falta --modelo; falta --porta; --porta fora de 1..65535; --porta duplicado; --modelo duplicado; opção desconhecida: --endereco; argumento inesperado; modelo inexistente; sem tokenizer.json; sem pesos; indisponível em 127.0.0.1; PRIMEIRA_PALAVRA_AUTORIZADA; CUDA_VISIBLE_DEVICES tem de ser GPU-8f42"
case " $* " in *" --modelo "*nao-existe*) [ "${FALSO_MODO:-}" = bloqueia ] && { sleep 77 & wait; exit 0; };; esac
if [ "${FALSO_MODO:-}" = escuta ]; then
  # deixa um ouvinte órfão na porta (5 s) e recusa normalmente: o ss do roteiro tem de pegar
  P=; while [ $# -gt 0 ]; do [ "$1" = --porta ] && P=$2; shift; done
  if [ -n "$P" ] && [ "$P" -ge 1 ] 2>/dev/null && [ "$P" -le 65535 ] 2>/dev/null; then
    python3 -I -c 'import socket, sys, time
s = socket.socket(); s.bind(("127.0.0.1", int(sys.argv[1]))); s.listen(1); time.sleep(5)' "$P" > /dev/null 2>&1 < /dev/null &
    sleep 0.3
  fi
fi
echo "$MSG" >&2
[ "${FALSO_MODO:-}" = rc1 ] && exit 1
exit 2
F
chmod +x "$D/falso"
roda() { FALSO_MODO=$1 RECUSAS_PRAZO=${2:-5} "$AQUI/roda-5e-recusas.sh" --bin "$D/falso" > "$D/saida-$1" 2>&1; RC=$?; }
roda ok; confere "binário que recusa: código 0, RECUSAS OK" "[ $RC = 0 ] && grep -q '^RECUSAS OK' $D/saida-ok && ! grep -q FALHOU $D/saida-ok"
confere "ok: o caso da porta ocupada rodou (3a e 3b)" "grep -q '^ok: 3a' $D/saida-ok && grep -q '^ok: 3b' $D/saida-ok"
confere "ok: o caso 4b (UUID de outra GPU, portão aberto) rodou" "grep -q '^ok: 4b' $D/saida-ok"
t0=$(date +%s); roda bloqueia 1; t1=$(date +%s)
confere "bloqueio: código 1, termina em poucos segundos" "[ $RC = 1 ] && [ $((t1 - t0)) -lt 20 ]"
confere "bloqueio: 'prazo excedido' no caso 2a e os demais continuam" "grep -q 'FALHOU: 2a.*prazo excedido' $D/saida-bloqueia && grep -q '^ok: 2b' $D/saida-bloqueia && grep -q '^ok: 4a' $D/saida-bloqueia"
confere "bloqueio: sem processo remanescente (sleep 77)" "! pgrep -x -f 'sleep 77' > /dev/null"
roda escuta
confere "binário que escuta: código 1 e 'escuta em 127.0.0.1'" "[ $RC = 1 ] && grep -q 'algo escuta em 127.0.0.1' $D/saida-escuta"
roda rc1
confere "código 1 em vez de 2: código 1 e 'código 1;'" "[ $RC = 1 ] && grep -q 'código 1;' $D/saida-rc1"
sleep 6   # deixa o falso do modo 'escuta' (5 s) acabar
echo "pasta de teste: $D"
if [ "$falhas" = 0 ]; then echo "TODOS OS TESTES PASSARAM"; exit 0; else echo "$falhas FALHA(S)"; exit 1; fi
