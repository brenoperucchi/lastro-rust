#!/usr/bin/env python3
"""Referência de correção do runtime MoE em Rust (plano docs/PLANO-runtime-rust-moe-20261029.md).

Roda um MoE do Hugging Face no transformers, na CPU e em float32 a partir dos pesos BF16, com teacher forcing sobre um texto
fixo, e grava por posição:
  - tokens.npy      int32 [T]
  - logits.npy      float32 [T, V]   (logits da próxima posição)
  - experts.npy     int32 [L, T, k]  (experts escolhidos pelo roteador em cada camada, ordem do top-k)
  - meta.json       modelo, hashes dos arquivos de config, versões, k, forma
Com --experts-only, os logits não são gravados (traço de roteamento para o simulador).

O runtime compara os logits dele com estes posição a posição; os experts servem para depurar o roteamento e como traço para
o simulador de colocação (M0.5). Rodar com o ambiente tools/.venv-ref (torch CPU + transformers):

  tools/.venv-ref/bin/python -I tools/ref_logits.py --model /home/brenoperucchi/models/<modelo> --text-file <txt> --out <dir>
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="diretório local com config.json e *.safetensors")
    ap.add_argument("--text-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--experts-only", action="store_true", help="grava só tokens e experts (traço de roteamento)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = AutoConfig.from_pretrained(args.model, local_files_only=True)
    k = getattr(cfg, "num_experts_per_tok")
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32, trust_remote_code=False
    ).eval()

    text = open(args.text_file, encoding="utf-8").read()
    ids = tok(text, return_tensors="pt").input_ids[:, : args.max_tokens]
    t0 = time.time()
    with torch.inference_mode():
        out = model(ids, output_router_logits=True, use_cache=False)
    secs = time.time() - t0

    logits = out.logits[0].float().numpy()
    # router_logits: uma entrada por camada MoE, [T, n_experts] (batch 1 achatado).
    experts = np.stack([torch.topk(r.float(), k, dim=-1).indices.numpy() for r in out.router_logits]).astype(np.int32)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "tokens.npy"), ids[0].numpy().astype(np.int32))
    if not args.experts_only:
        np.save(os.path.join(args.out, "logits.npy"), logits)
    np.save(os.path.join(args.out, "experts.npy"), experts)
    meta = {
        "model": os.path.realpath(args.model),
        "architectures": cfg.architectures,
        "files_sha256": {
            f: sha256(os.path.join(args.model, f))
            for f in sorted(os.listdir(args.model))
            if f.endswith(".json") and not f.startswith(".")
        },
        "text_file": os.path.realpath(args.text_file),
        "text_sha256": sha256(args.text_file),
        "tokens": int(ids.shape[1]),
        "vocab": int(logits.shape[1]),
        "moe_layers": int(experts.shape[0]),
        "experts_per_tok": int(k),
        "dtype": "float32 (pesos BF16 convertidos)",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "forward_seconds": round(secs, 2),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: meta[k] for k in ("tokens", "vocab", "moe_layers", "experts_per_tok", "forward_seconds")}))


if __name__ == "__main__":
    main()
