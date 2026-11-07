#!/usr/bin/env python3
"""Referência do degrau 5a (tokenizador/detokenizador offline em Rust): grava
qual-runs/A2-LOTE-20261103/5a-referencia/casos.json com textos, ids do HF, template renderizado, data usada e decode com e sem
clean_up_tokenization_spaces. CPU, sem rede, sem GPU. Imprime o sha256 de casos.json.

  env CUDA_VISIBLE_DEVICES= LC_ALL=C tools/.venv-ref/bin/python -I tools/gera_ref_5a.py

Critério congelado: qual-runs/A2-LOTE-20261103/criterio-5a-tokenizador.txt. A data do template é FIXA (DATA abaixo): o
`strftime_now` do transformers é substituído por um relógio fixo (o resto do pipeline do HF fica intacto).
"""
import datetime as _dt
import hashlib
import json
import os
import sys

import tokenizers
import transformers
from transformers import AutoTokenizer
from transformers.utils import chat_template_utils

RAIZ = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MODELO_1B = "/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct"
MODELO_3B = "/home/brenoperucchi/models/granite-3.1-3b-a800m-instruct"
DATA = _dt.datetime(2026, 11, 7, 12, 0, 0)  # "November 07, 2026"
SAIDA = os.path.join(RAIZ, "qual-runs", "A2-LOTE-20261103", "5a-referencia", "casos.json")


class Relogio(_dt.datetime):
    _fixo = DATA

    @classmethod
    def now(cls, tz=None):
        return cls._fixo


def com_data(d):
    Relogio._fixo = d
    chat_template_utils.datetime = Relogio


def ler(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def bordas():
    c = {}
    c["vazio"] = ""
    c["so-um-espaco"] = " "
    c["so-espacos"] = "     "
    c["so-quebra"] = "\n"
    c["quebras-repetidas"] = "a\n\n\n\nb\n\n"
    c["quebras-e-espacos"] = "linha 1  \n  \n \n linha 4\n\n\n   "
    c["tabulacao"] = "a\tb\t\tc\t"
    c["tabulacao-e-espacos"] = "\t \t x \t\t"
    c["crlf"] = "uma\r\nduas\r\n\r\ntres\r"
    c["espacos-no-fim"] = "fim com espaços   "
    c["espaco-no-inicio"] = "   começo com espaços"
    c["acentos-ptbr"] = "Ação, coração, pão, não, órgão, açúcar, vovô, avó, à, é, ê, ü, ç. Português: ÁÉÍÓÚ ÀÂÊÔ ÃÕ Ç."
    c["paragrafo-ptbr"] = (
        "Explique em português como funciona um cache LRU, quando ele é uma boa escolha e quando não é.\n\n"
        "O índice de inflação ficou em 4,5% ao ano; R$ 1.234,56 não é pouco. “Aspas tipográficas” e ‘simples’ — travessão – meia-risca…"
    )
    c["emoji"] = "Olá 😀 mundo 👍🏽 família 👨‍👩‍👧‍👦 bandeira 🇧🇷 coração ❤️ fim"
    c["emoji-so"] = "😀"
    c["cjk"] = "日本語のテキストと中文和한국어 입니다。你好，世界！"
    c["cjk-misturado"] = "Hello 世界 123 こんにちは abc"
    c["arabe-hebraico"] = "مرحبا بالعالم שלום עולם"
    c["devanagari-marcas"] = "नमस्ते दुनिया क्षत्रिय हिन्दी कि की"
    c["tailandes"] = "สวัสดีชาวโลก"
    c["numeros-romanos-e-letras-circuladas"] = "Ⅻ ⅷ Ⓐ ⓐ ℵ ℂ ª º"
    c["digitos-outros-scripts"] = "٠١٢٣٤٥٦٧٨٩ ०१२३४५६७८९ ১২৩ ๑๒๓ １２３４ ²³ ½ ¼ ① ⑩ ⁵ ₇"
    c["simbolos"] = "© ® ™ € £ ¥ § ¶ † ‡ • ° ± × ÷ ≠ ≤ ≥ ∞ ∑ √ ∂ → ← ↔ ⇒ ♠ ♥ ★ ☆"
    c["end-of-text-literal"] = "antes<|end_of_text|>depois"
    c["end-of-text-so"] = "<|end_of_text|>"
    c["end-of-text-repetido"] = "<|end_of_text|><|end_of_text|>\n<|end_of_text|>"
    c["start-of-role-literal"] = "x<|start_of_role|>user<|end_of_role|>oi<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
    c["tool-call-e-fim"] = "<|tool_call|>[{\"name\": \"f\"}]"
    c["adicionados-diversos"] = "<fim_prefix>a<fim_middle>b<fim_suffix>c<fim_pad><filename><gh_stars><reponame><jupyter_code> x"
    c["quase-adicionado"] = "<|end_of_tex> <|end_of_text| <|end_of_text|> < |end_of_text|> <|End_Of_Text|> <fim_prefi>"
    c["especial-colado-em-palavra"] = "abc<|end_of_text|>def<|start_of_role|>ghi"
    c["numero-longo"] = "3141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067"
    c["numeros-e-separadores"] = "1,234,567.89 0x1F 1e-9 2026-11-07 12:34:56 +55 (11) 91234-5678 100% 3.14"
    c["urls"] = "Veja https://example.com/a/b?c=d&e=f#frag e http://localhost:8080/api/v1/users/42, também ftp://x.y/z e user@mail.example.org."
    c["codigo-simbolos"] = (
        "fn main() -> Result<(), Box<dyn Error>> {\n    let v: Vec<_> = (0..10).map(|x| x * 2).collect();\n"
        "    println!(\"{:?}\", v); // ok\n    if a != b && c >= d || !e { return Err(\"x\".into()); }\n}\n"
    )
    c["codigo-python-indentado"] = "def f(x):\n    if x:\n        return [i**2 for i in range(x)]\n\n\n    return None\n"
    c["json"] = '{"a": [1, 2.5, null, true], "b": {"c": "d\\n", "e": "\\u00e9"}}'
    c["regex-e-shell"] = "grep -E '^[A-Za-z_]+\\s*=\\s*\\d+$' file | sort -u > out.txt && echo \"$HOME/$(date +%F)\""
    c["contracoes-ingles"] = "I'm sure he's right, don't you think? We'll see; they've said it'd work, you're welcome. It's the dogs' toys. 'Tis the season. I'M HE'S DON'T O'Neil rock'n'roll"
    c["contracoes-borda"] = "'s 't 're 've 'm 'll 'd ' '' 'S 'RE 'x 'sa he's'd"
    c["aspas-tipograficas"] = "“Olá”, disse ela. ‘Não’, respondeu. «francês» „alemão“ 「japonês」"
    c["controle"] = "a\x00b\x01c\x07d\x08e\x0bf\x0cg\x1bh\x1fi\x7fj\x85k\x9fl"
    c["controle-so"] = "\x00\x01\x02\x03"
    c["unicode-espacos"] = "a\u00a0b\u2003c\u2009d\u200be\u3000f\u202fg\u2028h\u2029i\ufeffj"
    c["nfd-acentos"] = "a\u0301e\u0301i\u0301o\u0301u\u0301 c\u0327a\u0303 n\u0303 o\u0308 e\u0302"
    c["nfd-e-nfc"] = "café café São São ação ação"
    c["zalgo"] = "z\u0351\u035b\u0360a\u0301\u0302\u0303l\u0334\u0335g\u0489o"
    c["u+fffd-e-privado"] = "\ufffd \ue000 \U000f0000 \U0010fffd"
    c["plano-astral"] = "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝟏𝟐𝟑 𐍈 𠀀 𝒜"
    c["plano-4-a-7-sem-byte-f1"] = "a\U00040000b \U0005fffdc\U0007ffffd \U00040000"  # byte 0xF1 ausente do vocabulário: o HF descarta o byte
    c["repeticao-longa"] = "a" * 300 + " " + "!" * 120 + " " + "ab" * 150 + " " + " " * 64 + "z"
    c["maiusculas-e-underscores"] = "CamelCaseWord snake_case_word SCREAMING_CASE kebab-case-word __dunder__ $var @user #tag"
    c["pontuacao-e-espacos"] = "Olá , mundo ! Tudo bem ? Sim .  Fim ( ok ) [x] {y} ... ?! -- -"
    c["texto-misto-longo"] = ler(os.path.join(RAIZ, "tools", "ref-texts", "texto-a.txt")) + "\n\n" + "Fim — ação, 日本語 😀\n"
    return c


def varredura():
    """Blocos de 512 valores escalares (sem substitutos): cada ponto c entra como 'a'+c+'1'+c+' ' (letra, número, outro, espaço)."""
    faixas = [(0, 0x32400), (0xE0000, 0xE0200), (0xF0000, 0xF0010), (0xFFFF0, 0xFFFFE), (0x10FFF0, 0x110000)]
    blocos = []
    for lo, hi in faixas:
        for ini in range(lo, hi, 512):
            cps = [c for c in range(ini, min(ini + 512, hi)) if not 0xD800 <= c <= 0xDFFF]
            if cps:
                blocos.append((ini, "".join(f"a{chr(c)}1{chr(c)} " for c in cps)))
    return blocos


def main():
    tok = AutoTokenizer.from_pretrained(MODELO_1B, local_files_only=True)
    assert open(MODELO_1B + "/tokenizer.json", "rb").read() == open(MODELO_3B + "/tokenizer.json", "rb").read()
    com_data(DATA)
    data_txt = DATA.strftime("%B %d, %Y")
    assert data_txt == "November 07, 2026", data_txt

    casos, decs = [], []

    def texto(nome, cat, t):
        ids = tok(t).input_ids
        casos.append({
            "nome": nome, "categoria": cat, "tipo": "texto", "texto": t, "ids": ids,
            "decode_sem_cleanup": tok.decode(ids, clean_up_tokenization_spaces=False),
            "decode_com_cleanup": tok.decode(ids),
            "decode_com_cleanup_forcado": tok.clean_up_tokenization(tok.decode(ids, clean_up_tokenization_spaces=False)),
        })

    def chat(nome, cat, msgs, agp, data=DATA):
        com_data(data)
        r = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=agp)
        ids = tok(r).input_ids
        ids_t = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=agp)
        ids_t = ids_t["input_ids"] if not isinstance(ids_t, list) else ids_t
        casos.append({
            "nome": nome, "categoria": cat, "tipo": "chat", "mensagens": msgs, "add_generation_prompt": agp,
            "data": data.strftime("%B %d, %Y"), "renderizado": r, "ids": ids, "ids_template_igual_ids": list(ids_t) == ids,
            "decode_sem_cleanup": tok.decode(ids, clean_up_tokenization_spaces=False),
            "decode_com_cleanup": tok.decode(ids),
            "decode_com_cleanup_forcado": tok.clean_up_tokenization(tok.decode(ids, clean_up_tokenization_spaces=False)),
        })
        com_data(DATA)
        return r

    # 1. textos de tools/ref-texts
    texto("texto-a", "ref-texts", ler(os.path.join(RAIZ, "tools/ref-texts/texto-a.txt")))
    prompts = json.load(open(os.path.join(RAIZ, "tools/ref-texts/prompts-traco.json"), encoding="utf-8"))
    for i, p in enumerate(prompts):
        texto(f"prompt-{i}-cru", "ref-texts", p)

    # 2. pedidos renderizados pelo template (sistema padrão, data fixa) e variantes
    for i, p in enumerate(prompts):
        chat(f"pedido-{i}", "pedido-template", [{"role": "user", "content": p}], True)
    chat("pedido-sem-gen-prompt", "template-variantes", [{"role": "user", "content": prompts[0]}], False)
    chat("com-sistema", "template-variantes", [{"role": "system", "content": "Você é um assistente conciso."}, {"role": "user", "content": "Oi"}], True)
    chat("com-sistema-vazio", "template-variantes", [{"role": "system", "content": ""}, {"role": "user", "content": "Oi"}], True)
    chat("so-sistema-com-gen-prompt", "template-variantes", [{"role": "system", "content": "S"}], True)
    chat("so-sistema-sem-gen-prompt", "template-variantes", [{"role": "system", "content": "S"}], False)
    chat("multi-turno", "template-variantes", [
        {"role": "user", "content": "Olá"}, {"role": "assistant", "content": "Oi! Como posso ajudar?"},
        {"role": "user", "content": "Qual a capital do Brasil?"}], True)
    chat("multi-turno-com-sistema", "template-variantes", [
        {"role": "system", "content": "Responda em inglês."}, {"role": "user", "content": "Olá"},
        {"role": "assistant", "content": "Hello!"}, {"role": "user", "content": "Obrigado"}, {"role": "assistant", "content": "Welcome."}], False)
    chat("conteudo-vazio", "template-variantes", [{"role": "user", "content": ""}], True)
    chat("conteudo-com-especiais", "template-variantes", [{"role": "user", "content": "x<|end_of_text|>y<|start_of_role|>z"}], True)
    chat("sistema-no-meio", "template-variantes", [{"role": "user", "content": "a"}, {"role": "system", "content": "b"}, {"role": "user", "content": "c"}], True)
    chat("usuario-duplo", "template-variantes", [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}], False)
    chat("assistente-primeiro", "template-variantes", [{"role": "assistant", "content": "a"}], True)
    chat("data-um-digito", "template-variantes", [{"role": "user", "content": "oi"}], True, _dt.datetime(2026, 1, 5))
    chat("data-bissexto", "template-variantes", [{"role": "user", "content": "oi"}], True, _dt.datetime(2024, 2, 29))

    # 3. conversas dos traços (1B e 3B): texto cru e o template com a data do arquivo reproduzindo o arquivo byte a byte
    for sufixo, modelo in (("1b", "granite-3.1-1b-a400m-instruct"), ("3b", "granite-3.1-3b-a800m-instruct")):
        for i in range(8):
            arq = os.path.join(RAIZ, "qual-runs/TRACOS-20261029", modelo, "textos", f"conversa-{i}.txt")
            t = ler(arq)
            texto(f"conversa-{sufixo}-{i}", "conversa-traco", t)
            ini = t.index("<|start_of_role|>assistant<|end_of_role|>") + len("<|start_of_role|>assistant<|end_of_role|>")
            resp = t[ini : t.rindex("<|end_of_text|>\n")]
            r = chat(f"conversa-{sufixo}-{i}-template", "conversa-template", [{"role": "user", "content": prompts[i]}, {"role": "assistant", "content": resp}],
                     False, _dt.datetime(2026, 10, 29))
            if r != t:
                sys.exit(f"o template com a data de 29/10 NÃO reproduz {arq}")

    # 4. bordas
    for nome, t in bordas().items():
        texto("borda-" + nome, "borda", t)

    # 5. varredura Unicode
    for ini, t in varredura():
        texto(f"varredura-U+{ini:06X}", "varredura-unicode", t)

    # 6. decodificação: ids gerados (guloso) das referências e sequências cortadas no meio de um caractere
    for sufixo, d in (("1b", "REF-greedy-granite-1b"), ("3b", "REF-greedy-granite-3b")):
        g = json.load(open(os.path.join(RAIZ, "qual-runs", d, "greedy.json")))
        for i, c in enumerate(g["cases"]):
            ids = c["generated_ids"]
            decs.append({"nome": f"greedy-{sufixo}-{i}", "categoria": "decode-guloso", "ids": ids,
                         "decode_sem_cleanup": tok.decode(ids, clean_up_tokenization_spaces=False),
                         "decode_com_cleanup": tok.decode(ids), "generated_text": c["generated_text"],
                         "decode_com_cleanup_forcado": tok.clean_up_tokenization(tok.decode(ids, clean_up_tokenization_spaces=False))})
    for nome, t in (("emoji", "Olá 😀👍🏽"), ("cjk", "日本語のテキスト"), ("nfd", "e\u0301a\u0303"), ("familia", "👨‍👩‍👧")):
        ids = tok(t).input_ids
        for k in range(1, len(ids)):
            p = ids[:k]
            decs.append({"nome": f"cortado-{nome}-{k}", "categoria": "decode-cortado", "ids": p,
                         "decode_sem_cleanup": tok.decode(p, clean_up_tokenization_spaces=False), "decode_com_cleanup": tok.decode(p),
                         "decode_com_cleanup_forcado": tok.clean_up_tokenization(tok.decode(p, clean_up_tokenization_spaces=False))})

    meta = {
        "data_fixada": data_txt, "tokenizers": tokenizers.__version__, "transformers": transformers.__version__,
        "tokenizer_json_sha256": hashlib.sha256(open(MODELO_1B + "/tokenizer.json", "rb").read()).hexdigest(),
        "tokenizer_config_sha256": hashlib.sha256(open(MODELO_1B + "/tokenizer_config.json", "rb").read()).hexdigest(),
        "n_casos": len(casos), "n_decodificacoes": len(decs),
        "ids_template_diferem_de_ids_dos_textos": [c["nome"] for c in casos if c["tipo"] == "chat" and not c["ids_template_igual_ids"]],
        "n_casos_cleanup_forcado_difere": sum(c["decode_com_cleanup_forcado"] != c["decode_sem_cleanup"] for c in casos),
        "n_decodificacoes_cleanup_forcado_difere": sum(c["decode_com_cleanup_forcado"] != c["decode_sem_cleanup"] for c in decs),
        "nota": "No transformers 5.x o decode padrão IGNORA clean_up para BPE (decode_com_cleanup == decode_sem_cleanup); decode_com_cleanup_forcado aplica clean_up_tokenization() legado explicitamente. decode_sem_cleanup = decode(ids, clean_up_tokenization_spaces=False); decode_com_cleanup = decode(ids) padrão (clean_up True).",
    }
    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "casos": casos, "decodificacoes": decs}, f, ensure_ascii=False, separators=(",", ":"))
    print(SAIDA, meta["n_casos"], "casos", meta["n_decodificacoes"], "decodificações")
    print("sha256", hashlib.sha256(open(SAIDA, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
