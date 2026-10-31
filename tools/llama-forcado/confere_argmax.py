"""Confere o argmax do forcado contra o próximo token do traço, nas posições da resposta (validação da comparação com o
llama.cpp, plano: "Validação antes da medida").

- A resposta começa depois de `<|start_of_role|>assistant<|end_of_role|>` (ids achados na sequência do traço).
- Ficam de fora as duas últimas posições de cada conversa (o `<|end_of_text|>` e a quebra de linha que o template acrescenta).
- Os traços são o texto do servidor tokenizado pelo HF, que corta `\\n\\n` em dois `Ċ`; o GGUF usa um token só, `ĊĊ`. Onde o
  próximo token do traço é o primeiro `Ċ` de um par, o argmax `ĊĊ` conta como concordância.
- Portão: concordância ≥ 99% e toda divergência com margem top-1 − top-2 < 0,5 logit.
- Cada divergência é classificada: "retokenização" quando o texto do argmax é um prefixo do texto do traço a partir da
  posição seguinte (o servidor gerou o argmax e o HF retokenizou o mesmo texto de outro jeito, como `Ġresult`+`ou` contra
  `Ġres`+`ul`+`to`+`u`); "quase-empate" no resto. Só registro: o portão acima não muda.
- Com --ref <argmax de outra execução>, compara o argmax das duas nas posições da resposta (só registro, sem portão).
- Com --ref e --todas, é o portão de correção entre configurações (rodada 43): em todas as posições cronometradas (prompt e
  resposta), o argmax tem de ser igual ao da referência (a passada tudo na GPU F32 do mesmo harness) em pelo menos --min
  das posições em que a margem da referência passa de 0,1 logit.

  python -I -B tools/llama-forcado/confere_argmax.py <tokens dir> <argmax.txt> <dir do modelo HF> [--ref <argmax.txt> [--todas --min 0.99]]
"""
import sys

from transformers import AutoTokenizer

tokens_dir, path, hf = sys.argv[1], sys.argv[2], sys.argv[3]
ref_path = sys.argv[sys.argv.index("--ref") + 1] if "--ref" in sys.argv else None
tok = AutoTokenizer.from_pretrained(hf, local_files_only=True)
NL = tok.convert_tokens_to_ids("Ċ")
NL2 = tok.convert_tokens_to_ids("ĊĊ")
head = tok.encode("<|start_of_role|>assistant<|end_of_role|>", add_special_tokens=False)


def load(p):
    rows = {}
    for line in open(p):
        if line.startswith("#"):
            continue
        c, t, nxt, am, mg = line.split()
        rows[(int(c), int(t))] = (int(nxt), int(am), float(mg))
    return rows


rows = load(path)
ref = load(ref_path) if ref_path else None
if "--todas" in sys.argv:
    assert ref is not None and rows.keys() == ref.keys(), "o --todas pede --ref com as mesmas posições"
    need = float(sys.argv[sys.argv.index("--min") + 1])
    conf = [k for k in rows if ref[k][2] > 0.1]
    same = sum(rows[k][1] == ref[k][1] for k in conf)
    big = sum(rows[k][1] != ref[k][1] and ref[k][2] >= 0.5 for k in rows)
    rate = same / len(conf)
    print(f"{path}: argmax igual ao da referência em {same}/{len(conf)} posições com margem da referência > 0,1 ({rate:.2%}); "
          f"em todas: {sum(rows[k][1] == ref[k][1] for k in rows)}/{len(rows)}; divergências com margem da referência ≥ 0,5: {big}")
    print(f"portão entre configurações (≥ {need:.1%}): {'PASSA' if rate >= need else 'NÃO PASSA'}")
    sys.exit(0)
convs = sorted({c for c, _ in rows})
n = ok = 0
bad = []
same_ref = n_ref = 0
for c in convs:
    toks = [int(x) for x in open(f"{tokens_dir}/conversa-{c}.txt").read().split()]
    start = next(i + len(head) for i in range(len(toks)) if toks[i:i + len(head)] == head)
    for t in range(start - 1, len(toks) - 3):  # a posição t prevê toks[t + 1]; fora as duas últimas previsões
        nxt, am, mg = rows[(c, t)]
        assert nxt == toks[t + 1]
        n += 1
        hit = am == nxt or (nxt == NL and toks[t + 2] == NL and am == NL2)
        ok += hit
        if not hit:
            follow = tok.decode(toks[t + 1:t + 8])
            kind = "retokenização" if follow.startswith(tok.decode([am])) else "quase-empate"
            bad.append((c, t, tok.convert_ids_to_tokens(nxt), tok.convert_ids_to_tokens(am), mg, kind))
        if ref is not None:
            n_ref += 1
            same_ref += am == ref[(c, t)][1]
rate = ok / n
big = [b for b in bad if b[4] >= 0.5]
retok = [b for b in bad if b[5] == "retokenização"]
print(f"{path}: {ok}/{n} posições da resposta concordam com o traço ({rate:.2%}); divergências {len(bad)}, com margem ≥ 0,5: {len(big)}")
print(f"  divergências por retokenização: {len(retok)} (com margem ≥ 0,5: {sum(b[4] >= 0.5 for b in retok)}); quase-empates: "
      f"{len(bad) - len(retok)} (com margem ≥ 0,5: {sum(b[4] >= 0.5 for b in bad if b[5] != 'retokenização')}); "
      f"concordância contando a retokenização: {(ok + len(retok)) / n:.2%}")
for b in bad:
    print(f"  conversa {b[0]} posição {b[1]}: traço {b[2]!r}, argmax {b[3]!r}, margem {b[4]:.3f}, {b[5]}")
if ref is not None:
    print(f"argmax igual ao de {ref_path}: {same_ref}/{n_ref} ({same_ref / n_ref:.2%})")
else:
    print(f"portão: {'PASSA' if rate >= 0.99 and not big else 'NÃO PASSA'}")
