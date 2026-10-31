#!/bin/bash
# Validação da comparação com o llama.cpp, antes da medida (plano, "Validação"; portão do criterio.txt). Grava validacao.txt e
# os arquivos de argmax em validacao/; os logits crus (centenas de MB) ficam em $VAL_TMP, fora do repositório.
#   VAL_TMP=<pasta temporária> bash validacao.sh
set -euo pipefail
cd /home/brenoperucchi/Devs/strata-opencode
R=qual-runs/LLAMACPP-COMPARA-20261030
V=$R/validacao
: "${VAL_TMP:?defina VAL_TMP}"
mkdir -p $V
G=/home/brenoperucchi/models/gguf
F=tools/llama-forcado/forcado
PY="taskset -c 16-27 tools/.venv-ref/bin/python -I -B"
COMUM="-ngl 99 -c 2048 --fit off --no-warmup -t 8"
F32="-ctk f32 -ctv f32 -fa off"
dir() { case $1 in 1b) echo granite-3.1-1b-a400m-instruct;; 3b) echo granite-3.1-3b-a800m-instruct;; esac; }
{
echo "Validação da comparação com o llama.cpp, $(date '+%d/%m %H:%M') (forcado sha $(sha256sum $F | cut -c1-16), llama.cpp 0893ec9)."
echo; echo "1. Tokenizador do GGUF contra os tokens dos traços (forcado --tokeniza):"
for m in 1b 3b; do
  mkdir -p $VAL_TMP/gguf-$m
  taskset -c 0-15 $F --tokens $R/tokens/$m --tokeniza qual-runs/TRACOS-20261029/$(dir $m)/textos --argmax $VAL_TMP/gguf-$m -m $G/$(dir $m)-bf16.gguf -ngl 0 2>/dev/null | sed "s/^/  $m /" || true
  $PY - $R/tokens/$m $VAL_TMP/gguf-$m /home/brenoperucchi/models/$(dir $m) <<'PYEOF'
import difflib, sys
from collections import Counter
from transformers import AutoTokenizer
a_dir, b_dir, hf = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(hf, local_files_only=True)
kinds = Counter()
for c in range(8):
    a = [int(x) for x in open(f"{a_dir}/conversa-{c}.txt").read().split()]
    b = [int(x) for x in open(f"{b_dir}/conversa-{c}.txt").read().split()]
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op != "equal":
            kinds[(" ".join(tok.convert_ids_to_tokens(a[i1:i2])), " ".join(tok.convert_ids_to_tokens(b[j1:j2])))] += 1
print("  diferenças (difflib, HF → GGUF): " + "; ".join(f"{x!r} → {y!r}: {n}" for (x, y), n in kinds.most_common()))
PYEOF
done
echo "  O forcado passa os ids dos traços: os dois motores recebem a mesma entrada."
for m in 1b 3b; do
  d=$(dir $m)
  taskset -c 0-15 $F --tokens $R/tokens/$m-texto-a --warm "" --convs 0 --logits $VAL_TMP/$m-texto-a.f32 -m $G/$d-bf16.gguf $COMUM $F32 > /dev/null 2>&1
  taskset -c 0-15 $F --tokens $R/tokens/$m --argmax $V/argmax-$m-gpu32.txt --logits $VAL_TMP/$m-gpu32.f32 -m $G/$d-bf16.gguf $COMUM $F32 > $V/$m-gpu32.out 2> $VAL_TMP/$m-gpu32.err
  taskset -c 0-15 $F --tokens $R/tokens/$m --argmax $V/argmax-$m-gpu16.txt -m $G/$d-bf16.gguf $COMUM > $V/$m-gpu16.out 2> /dev/null
  taskset -c 0-15 $F --tokens $R/tokens/$m --argmax $V/argmax-$m-cpumoe.txt -m $G/$d-bf16.gguf $COMUM $F32 --cpu-moe > $V/$m-cpumoe.out 2> /dev/null
done
echo; echo "2. Portão: logits do forcado tudo na GPU (KV F32, FA desligado) contra o transformers (confere_logits.py)."
for m in 1b 3b; do
  echo "  $m, texto-a (toda posição com margem > 0,1):"; $PY tools/llama-forcado/confere_logits.py /home/brenoperucchi/models/$(dir $m) $R/tokens/$m-texto-a 0 $VAL_TMP/$m-texto-a.f32 2>/dev/null | command grep -E "^(total|erro L2|divergências|posições|portão)" | sed "s/^/    /"
  echo "  $m, conversas 4 a 7 (≥ 99% das posições com margem > 0,1):"; $PY tools/llama-forcado/confere_logits.py /home/brenoperucchi/models/$(dir $m) $R/tokens/$m 4,5,6,7 $VAL_TMP/$m-gpu32.f32 --min 0.99 2>/dev/null | sed "s/^/    /"
done
echo; echo "3. Registro: argmax contra o próximo token do traço (o portão de antes da rodada 43, que reprovou no 1B):"
for m in 1b 3b; do $PY tools/llama-forcado/confere_argmax.py $R/tokens/$m $V/argmax-$m-gpu32.txt /home/brenoperucchi/models/$(dir $m) 2>/dev/null | command grep -v "^  conversa" | sed "s/^/  /"; done
echo; echo "4. Portão entre configurações nos pilotos (argmax contra a passada tudo na GPU F32, todas as posições, ≥ 99%):"
for m in 1b 3b; do for l in gpu16 cpumoe; do $PY tools/llama-forcado/confere_argmax.py $R/tokens/$m $V/argmax-$m-$l.txt /home/brenoperucchi/models/$(dir $m) --ref $V/argmax-$m-gpu32.txt --todas --min 0.99 2>/dev/null | sed "s/^/  /"; done; done
echo; echo "5. Tempos dos pilotos (ms/token nas conversas 4 a 7, aquecimento na 3):"
for m in 1b 3b; do for l in gpu32 gpu16 cpumoe; do echo "  $m $l: $(command grep '^total' $V/$m-$l.out)"; done; done
echo "  log da passada F32: $(command grep -h 'flash_attn  ' $VAL_TMP/1b-gpu32.err | head -1 | sed 's/^ *//'); $(command grep -h 'K (f32)' $VAL_TMP/1b-gpu32.err | head -1 | sed 's/^.*K (f32)/K (f32)/')"
echo; echo "6. Calibração do --moe-cache-mib (forcado sem decode, --cpu-moe, linha de log do cache):"
for x in 1b:231:579:1155:2307 3b:581:1445:2885:5765; do IFS=: read m a b c e <<< "$x"; for mib in $a $b $c $e; do echo -n "  $m $mib MiB: "; taskset -c 0-15 $F --tokens $R/tokens/$m --warm "" --convs "" -m $G/$(dir $m)-bf16.gguf $COMUM $F32 --cpu-moe --moe-cache-mib $mib 2>&1 | command grep -E "slots \(|MoE cache size" | sed 's/^.*impl: *//' | tr '\n' ' '; echo; done; done
echo; echo "7. Banda H2D de uma cópia síncrona (banda.cu, mediana de 200):"; sed "s/^/  /" $V/banda.txt
} > $R/validacao.txt
