"""Exporta os tokens.npy dos traços (qual-runs/TRACOS-20261029) para texto, um arquivo por conversa, para o forcado.

  tools/.venv-ref/bin/python -I -B tools/llama-forcado/exporta_tokens.py <dir do traço do modelo> <dir de saída>
"""
import os
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for c in range(8):
    t = np.load(os.path.join(src, f"conversa-{c}", "tokens.npy"))
    with open(os.path.join(out, f"conversa-{c}.txt"), "w") as f:
        f.write(" ".join(str(int(x)) for x in t.reshape(-1)) + "\n")
    print(c, t.shape, t.dtype)
