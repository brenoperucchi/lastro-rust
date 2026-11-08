#!/usr/bin/env bash
# Teste do prazo individual do roda-5d-recusas.sh com binário FALSO (sem GPU): um caso bloqueia (com descendente), os demais respondem como recusa.
# Esperado: termina no prazo, sem processo remanescente, código 1, 'prazo excedido' registrado, os demais casos continuam. Pasta mktemp -d, não apaga nada.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
D=$(mktemp -d -p "${TMPDIR:-/tmp}" recusas-5d-teste.XXXXXX) || exit 1
falhas=0
confere() { if eval "$2"; then echo "ok: $1"; else echo "FALHOU: $1"; falhas=$((falhas+1)); fi; }
cat > "$D/falso" <<'F'
#!/usr/bin/env bash
# bloqueia (com um descendente) quando o modelo é inexistente SEM texto; senão recusa como o real: código 2, mensagem útil, saída vazia.
case " $* " in *" --modelo "*nao-existe*) [ $# -eq 2 ] && { sleep 77 & wait; exit 0; };; esac
echo "lastro: recusado: entrada vazia ou só espaços; modelo inexistente; passam de 2048 posições; sem tokenizer.json; sem pesos; --novos fora de 1..64; opção desconhecida: --x; --modelo duplicado; --novos duplicado; texto duplicado; falta --modelo; Month DD, YYYY" >&2
exit 2
F
chmod +x "$D/falso"
t0=$(date +%s)
RECUSAS_PRAZO=1 "$AQUI/roda-5d-recusas.sh" --bin "$D/falso" > "$D/saida" 2>&1; rc=$?
t1=$(date +%s)
confere "código 1" "[ $rc = 1 ]"
confere "termina em poucos segundos (prazo individual, não espera 77 s)" "[ $((t1 - t0)) -lt 20 ]"
confere "registra 'prazo excedido' no caso 3d" "grep -q 'FALHOU: 3d.*prazo excedido' $D/saida"
confere "continua os demais casos (3a e 4h rodaram e ok)" "grep -q '^ok: 3a' $D/saida && grep -q '^ok: 4h' $D/saida"
confere "sem processo remanescente (sleep 77)" "! pgrep -x -f 'sleep 77' > /dev/null"
confere "1 falha contada" "grep -q '^1 FALHA' $D/saida"
echo "pasta de teste: $D"
if [ "$falhas" = 0 ]; then echo "TODOS OS TESTES PASSARAM"; exit 0; else echo "$falhas FALHA(S)"; cat "$D/saida"; exit 1; fi
