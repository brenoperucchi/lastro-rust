"""Linha secundária da comparação com o llama.cpp: o llama-server em uso normal. Para cada conversa 4 a 7, manda o prompt do
traço (os ids até o cabeçalho do assistente) ao /completion em greedy (temperatura 0, seed 1, n_predict 1024, sem cache de
prompt), grava o predicted_per_token_ms (só a resposta) e confere o texto gerado contra a resposta do traço.

  python -I -B servidor.py <tokens dir> <dir do modelo HF> <porta>
"""
import json
import sys
import urllib.request

from transformers import AutoTokenizer

tokens_dir, hf, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
tok = AutoTokenizer.from_pretrained(hf, local_files_only=True)
head = tok.encode("<|start_of_role|>assistant<|end_of_role|>", add_special_tokens=False)
tot_n = tot_ms = 0.0
for c in (4, 5, 6, 7):
    toks = [int(x) for x in open(f"{tokens_dir}/conversa-{c}.txt").read().split()]
    start = next(i + len(head) for i in range(len(toks)) if toks[i:i + len(head)] == head)
    want = tok.decode(toks[start:-2])  # fora o <|end_of_text|> e a quebra de linha do template
    body = {"prompt": toks[:start], "n_predict": 1024, "temperature": 0, "seed": 1, "cache_prompt": False, "return_tokens": True}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        out = json.load(r)
    got, tm = out["content"], out["timings"]
    first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
    same = got == want
    tot_n += tm["predicted_n"]
    tot_ms += tm["predicted_n"] * tm["predicted_per_token_ms"]
    print(f"conversa {c}: prompt {start} tokens, gerados {tm['predicted_n']}, predicted_per_token_ms {tm['predicted_per_token_ms']:.3f}, "
          f"texto {'igual ao traço' if same else f'diverge do traço no caractere {first} de {len(want)}'}")
print(f"total: {int(tot_n)} tokens gerados, ms/tok {tot_ms / tot_n:.3f}, tok/s {1000 * tot_n / tot_ms:.1f}")
