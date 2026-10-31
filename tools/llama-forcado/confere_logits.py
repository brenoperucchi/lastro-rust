"""Confere os logits do forcado contra o transformers (float32, CPU, teacher forcing sobre os mesmos ids), posição a posição,
em todas as posições das conversas (prompt e resposta). Portão de correção da comparação com o llama.cpp (rodada 43): argmax
igual ao do transformers em toda posição em que a margem top-1 − top-2 do transformers passe de 0,1 logit; o erro relativo
L2 por posição é relatado (mediana, máximo e a posição do máximo).

Com --min X, o portão exige argmax igual em pelo menos a fração X dessas posições (padrão 1, toda posição; nas conversas o
critério usa 0,99).

  python -I -B confere_logits.py <dir do modelo HF> <tokens dir> <convs, ex. 4,5,6,7> <logits f32 do forcado> [--min X] [--threads N]
"""
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM

hf, tokens_dir, convs, path = sys.argv[1], sys.argv[2], [int(c) for c in sys.argv[3].split(",")], sys.argv[4]
threads = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 12
MARGEM = 0.1
MIN = float(sys.argv[sys.argv.index("--min") + 1]) if "--min" in sys.argv else 1.0
torch.set_num_threads(threads)
model = AutoModelForCausalLM.from_pretrained(hf, local_files_only=True, dtype=torch.float32)
model.eval()
V = model.config.vocab_size
got_all = np.memmap(path, dtype=np.float32, mode="r")
off = 0
errs, pos_of = [], []
FAIXAS = ((0, 256), (256, 512), (512, 1024), (1024, 1 << 30))  # as faixas de posição do harness e do m2a-cache
por_faixa = [[] for _ in FAIXAS]
n_gate = n_ok = 0
bad = []
for c in convs:
    toks = [int(x) for x in open(f"{tokens_dir}/conversa-{c}.txt").read().split()]
    with torch.inference_mode():
        ref = model(torch.tensor([toks]), use_cache=False).logits[0].float().numpy()
    T = len(toks)
    got = np.asarray(got_all[off:off + T * V]).reshape(T, V)
    off += T * V
    e = np.linalg.norm(got - ref, axis=1) / np.linalg.norm(ref, axis=1)
    errs.extend(e.tolist())
    for k, (a, z) in enumerate(FAIXAS):
        por_faixa[k].extend(e[a:z].tolist())
    pos_of.extend((c, t) for t in range(T))
    s = np.sort(ref, axis=1)
    margin = s[:, -1] - s[:, -2]
    same = got.argmax(1) == ref.argmax(1)
    gate = margin > MARGEM
    n_gate += int(gate.sum())
    n_ok += int((same & gate).sum())
    for t in np.where(~same)[0]:
        bad.append((c, int(t), float(margin[t]), bool(gate[t])))
    print(f"conversa {c}: {T} posições, argmax igual {int(same.sum())}/{T}, erro L2 mediano {np.median(e):.2e}, máx {e.max():.2e} "
          f"(posição {int(e.argmax())})")
assert off == got_all.size, (off, got_all.size)
errs = np.array(errs)
i = int(errs.argmax())
print(f"total: {len(errs)} posições; erro relativo L2 mediano {np.median(errs):.2e}, p99 {np.quantile(errs, 0.99):.2e}, "
      f"máx {errs.max():.2e} (conversa {pos_of[i][0]}, posição {pos_of[i][1]})")
print("erro L2 mediano por faixa de posição: " + "  ".join(
    f"{a}{f'-{z}' if z < 1 << 30 else '+'}: {np.median(v):.2e} ({len(v)})" for (a, z), v in zip(FAIXAS, por_faixa) if v))
print(f"divergências de argmax: {len(bad)}; margem do transformers nelas: " + ", ".join(f"{b[2]:.3f}" for b in bad))
print(f"posições com margem do transformers > {MARGEM}: {n_gate}; argmax igual em {n_ok} ({n_ok / n_gate:.2%})")
print(f"portão (≥ {MIN:.0%}): {'PASSA' if n_ok >= MIN * n_gate else 'NÃO PASSA'}")
