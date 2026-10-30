#!/usr/bin/env python3
"""Referência do M1a (plano docs/PLANO-runtime-rust-moe-20261029.md): entrada e saída reais do bloco MoE de algumas camadas.

Roda um Granite MoE no transformers (CPU, float32 a partir dos pesos BF16) sobre um texto, captura com hooks a entrada e a
saída de `block_sparse_moe` e a decisão do roteador nas camadas pedidas, e grava em binário cru (little-endian), para o Rust
ler sem dependência de formato:
  layer-<L>.x.f32      [T, H]  entrada do bloco (saída do post_attention_layernorm)
  layer-<L>.y.f32      [T, H]  saída do bloco
  layer-<L>.idx.i32    [T, k]  experts escolhidos (ordem do top-k)
  layer-<L>.w.f32      [T, k]  pesos do softmax sobre o top-k
  meta.json            T, H, k, camadas, modelo, versões

  tools/.venv-ref/bin/python -I tools/ref_moe_block.py --model <dir> --text-file <txt> --layers 0,11,23 --out <dir>
"""
import argparse
import json
import os

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--layers", required=True, help="lista separada por vírgula")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32, trust_remote_code=False
    ).eval()
    layers = [int(x) for x in args.layers.split(",")]
    got = {}

    def block_hook(L):
        def f(_mod, inp, out):
            got.setdefault(L, {})["x"] = inp[0].detach()[0].float().numpy()
            got[L]["y"] = out.detach()[0].float().numpy()

        return f

    def router_hook(L):
        def f(_mod, _inp, out):
            _logits, weights, index = out
            got.setdefault(L, {})["w"] = weights.detach().float().numpy()
            got[L]["idx"] = index.detach().numpy().astype(np.int32)

        return f

    for L in layers:
        moe = model.model.layers[L].block_sparse_moe
        moe.register_forward_hook(block_hook(L))
        moe.router.register_forward_hook(router_hook(L))

    text = open(args.text_file, encoding="utf-8").read()
    ids = tok(text, return_tensors="pt").input_ids[:, : args.max_tokens]
    with torch.inference_mode():
        model(ids, use_cache=False)

    os.makedirs(args.out, exist_ok=True)
    for L in layers:
        for name, ext in (("x", "f32"), ("y", "f32"), ("idx", "i32"), ("w", "f32")):
            arr = np.ascontiguousarray(got[L][name])
            arr.astype("<f4" if ext == "f32" else "<i4").tofile(os.path.join(args.out, f"layer-{L}.{name}.{ext}"))
    cfg = model.config
    meta = {
        "model": os.path.realpath(args.model),
        "text_file": os.path.realpath(args.text_file),
        "tokens": int(ids.shape[1]),
        "hidden": int(cfg.hidden_size),
        "intermediate": int(cfg.intermediate_size),
        "experts": int(cfg.num_local_experts),
        "k": int(cfg.num_experts_per_tok),
        "layers": layers,
        "dtype": "float32 (pesos BF16 convertidos)",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: meta[k] for k in ("tokens", "hidden", "intermediate", "experts", "k", "layers")}))


if __name__ == "__main__":
    main()
