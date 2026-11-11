#!/usr/bin/env python3
"""Extrator da F2C, plano de desempenho das cópias (criterio-f2c-20261111.md; decisão do Breno de 11/11 15:23), sem GPU, só biblioteca padrão.

    python3 -I f2c-extrai.py --saida DIR_DA_JANELA [--ids cmp-ids-prompt.json] [--refs 5c-referencia] [--textos 5c-textos/textos.json]
                             [--coleta DIR_DA_COLETA]

NÃO roda nada: lê o que roda-f2c-janela.sh gravou e grava DIR/f2c-resultado.json e DIR/f2c-resultado.md (um modelo por janela).
Reaproveita f2b-extrai-v2.py (o f2b-extrai.py de 14:10 fica congelado e intocado) no contrato v2 de registros do nível B (17 por posição) e, por baixo, f2-extrai.py e instrumento-extrai.py (leitura de processo, 5c, integridade, potência).

Layout lido (escrito por roda-f2c-janela.sh):
  DIR/celulas/ctrl-<modelo>-L100-medio-<NN>-<Dvelho|Dnovo|B2novo>/   9 processos, quadrado latino, `--repete 3` (1 aquecimento + 3 medidas)
  Dvelho = base+gancho sem nada ligado; Dnovo = candidato sem nada ligado; B2novo = candidato com perfil-bloco, perfil-f2 e perfil-f2b (instrumento v2).
Estatística única (criterio-f2c, seção 3): mediana das medianas dos 3 processos de cada braço, texto médio, razão novo sobre velho menos 1; referência única = D_velho da própria janela.
Hipóteses: HC1 (ids), HC1b (hash dos logits e do KV, só se os dois arquivos do gancho forem dados), HC2a e HC2b (H_PRE_LAYER por camada, descritivas), HC3 (prefill +1,5%), HC4 (decode +2%),
HC5 (resíduo do nível B até 3%), HC6 (B2_novo contra D_novo: até 3% comprovada, até 10% com ressalva, acima SEM PROVA).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import statistics
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))


def carrega(nome, arq):
    spec = importlib.util.spec_from_file_location(nome, os.path.join(AQUI, arq))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B = carrega("f2b_extrai_v2", "f2b-extrai-v2.py")
F, I, X = B.F, B.I, B.X
fin, mediana = B.fin, B.mediana
MODELOS, LAYERS_DO_MODELO, EPS = B.MODELOS, B.LAYERS_DO_MODELO, B.EPS
CONTRATO = 2
BRACOS = ("Dvelho", "Dnovo", "B2novo")
ORDEM_CONTROLE = ["Dvelho", "Dnovo", "B2novo", "B2novo", "Dvelho", "Dnovo", "Dnovo", "B2novo", "Dvelho"]
LIM_PREFILL, LIM_DECODE = 0.015, 0.02       # HC3 e HC4 (decisão do Breno 15:23, ponto 5)
CTRL_RE = re.compile(r"^ctrl-(1b|3b)-L100-medio-(\d\d)-(Dvelho|Dnovo|B2novo)$")
ORDEM_RE = r"^(\d\d) (ctrl-%s-L100-medio-(\d\d)-(Dvelho|Dnovo|B2novo)) n=(\d+) braco=(Dvelho|Dnovo|B2novo) inicio=\S+ fim=\S+ rc=(\d+)$"
# degrau do H_PRE_LAYER por camada, fixado pelos dados de 14:10 (plano, seção 5): primeira camada do patamar e transição
DEGRAU = {"1b": {"antes_ate": 10, "transicao": 11, "patamar_de": 12}, "3b": {"antes_ate": 8, "transicao": 9, "patamar_de": 10}}
FIXO_CHAVES = ("criterio-f2c", "plano-copias", "decisao-copias", "binario-base", "binario-cand", "binario-ctx", "script-f2b-extrai-v2", "script-f2c-extrai", "script-f2-extrai",
               "script-instrumento-extrai", "script-perfil-extrai", "script-cmp-extrai", "script-lastro", "script-monitor", "script-janela", "script-limite", "criterio-cmp", "criterio-degrau1",
               "criterio-degrau2", "cmp-ids-prompt", "texto-curto", "texto-medio", "texto-longo", "textos-json", "ref5c-1b", "ref5c-3b")


class Recusa(Exception):
    pass


def quadrado_latino_ok(ordem):
    """Três linhas de três: cada braço exatamente uma vez em cada linha e em cada posição (coluna)."""
    if len(ordem) != 9 or any(x not in BRACOS for x in ordem):
        return False
    linhas = [ordem[0:3], ordem[3:6], ordem[6:9]]
    return all(sorted(l) == sorted(BRACOS) for l in linhas) and all(sorted(l[c] for l in linhas) == sorted(BRACOS) for c in range(3))


# ---------------------------------------------------------------------------------------------------------------------------------- classificação (pura)
def razao(novo, velho):
    return (novo / velho - 1.0) if fin(novo) is not None and fin(velho) is not None and velho > 0 else None


def classifica_janela_c(prefill, passo):
    """`prefill` e `passo`: {braco: mediana das medianas dos 3 processos (ms)}. Devolve as razões e as classes de HC3, HC4 e HC6. Pura.
    HC3: prefill D_novo contra D_velho, regressão se passar de +1,5%. HC4: decode, regressão se passar de +2%. HC6: B2_novo contra D_novo (prefill): |razão| <= 3% comprovada,
    <= 10% com ressalva, acima SEM PROVA. Qualquer HC3 ou HC4 acima do limite, ou HC6 SEM PROVA, é regressão; queda nunca é regressão."""
    r3 = razao(prefill.get("Dnovo"), prefill.get("Dvelho"))
    r4 = razao(passo.get("Dnovo"), passo.get("Dvelho"))
    r6 = razao(prefill.get("B2novo"), prefill.get("Dnovo"))
    r6p = razao(passo.get("B2novo"), passo.get("Dnovo"))
    hc3 = "sem dado" if r3 is None else ("REGRESSÃO" if r3 > LIM_PREFILL + EPS else "dentro do limite")
    hc4 = "sem dado" if r4 is None else ("REGRESSÃO" if r4 > LIM_DECODE + EPS else "dentro do limite")
    hc6 = "sem dado" if r6 is None else F.classe_por_desvio(r6)
    regressao = [n for n, c in (("HC3", hc3), ("HC4", hc4)) if c == "REGRESSÃO"] + (["HC6"] if hc6 == "SEM PROVA" else [])
    return {"razao_prefill_novo_sobre_velho": r3, "razao_decode_novo_sobre_velho": r4, "razao_prefill_b2_sobre_dnovo": r6, "razao_decode_b2_sobre_dnovo": r6p,
            "HC3": hc3, "HC4": hc4, "HC6": hc6, "regressao": regressao}


# ---------------------------------------------------------------------------------------------------------------------------------- gancho (HC1b), espelho de gancho_coleta.rs
def hash_bits(valores):
    import struct
    return hashlib.sha256(b"".join(struct.pack("<f", v) for v in valores)).hexdigest()


def compara_gancho(a, b):
    """`a` e `b`: a lista `pedidos` de gancho-logits-<modelo>.json. Devolve as divergências (vazio = idênticos bit a bit). Olha os HASHES: ids iguais com 1 ULP de diferença num logit falham."""
    d = []
    if not a or not b:
        return ["conjunto sem pedidos: nada a comparar"]
    if len(a) != len(b):
        return ["%d pedidos contra %d" % (len(a), len(b))]
    for x, y in zip(a, b):
        p = x.get("indice")
        if x.get("indice") != y.get("indice") or x.get("aquecimento") != y.get("aquecimento"):
            d.append("pedido %s: índice ou marca de aquecimento diferentes" % p)
            continue
        if x.get("ids") != y.get("ids"):
            d.append("pedido %s: ids do prompt diferentes" % p)
        if x.get("gerados") != y.get("gerados"):
            d.append("pedido %s: ids gerados diferentes" % p)
        lx, ly = x.get("logits") or [], y.get("logits") or []
        if len(lx) != len(ly):
            d.append("pedido %s: %d hashes de logits contra %d" % (p, len(lx), len(ly)))
        for i, (u, v) in enumerate(zip(lx, ly)):
            if u != v:
                d.append("pedido %s: logits do %s diferem (primeira divergência)" % (p, "último token do prefill" if i == 0 else "passo %d do decode" % i))
                break
        kx, ky = x.get("kv") or [], y.get("kv") or []
        if len(kx) != len(ky):
            d.append("pedido %s: KV de %d camadas contra %d" % (p, len(kx), len(ky)))
        for l, (u, v) in enumerate(zip(kx, ky)):
            for lado, s, t in (("K", u[0], v[0]), ("V", u[1], v[1])):
                if s != t:
                    d.append("pedido %s: KV da camada %d (%s) difere no fim do prefill" % (p, l, lado))
        if not lx or not kx:
            d.append("pedido %s: sem hashes de logits ou de KV (gancho desligado?)" % p)
    return d


HEX64 = re.compile(r"[0-9a-f]{64}")
VARIAVEL_GANCHO = "LASTRO_GANCHO_LOGITS"


def valida_gancho(doc, m, layers, ids_esperados, gerados_esperados, n_repete):
    """Envelope e cobertura de UMA coleção: tipo, versão, variável, modelo, n_repete + 1 pedidos (índice 0 = aquecimento), ids do prompt e gerados iguais aos da referência 5c,
    um hash de logits por token gerado, KV de `layers` camadas, todos os hashes com 64 dígitos hexadecimais. Devolve a lista de problemas (vazia = coleção completa e coerente)."""
    pr = []
    if not isinstance(doc, dict) or doc.get("tipo") != "gancho-logits" or doc.get("versao") != 1 or not isinstance(doc.get("pedidos"), list):
        return ["fora do esquema (tipo gancho-logits, versao 1, lista de pedidos)"]
    if doc.get("variavel") != VARIAVEL_GANCHO:
        pr.append("campo variavel diferente de %s" % VARIAVEL_GANCHO)
    if doc.get("modelo") != m:
        pr.append("coleção do modelo %r numa coleta do modelo %s" % (doc.get("modelo"), m))
    peds = doc["pedidos"]
    if len(peds) != n_repete + 1:
        pr.append("%d pedidos (esperados %d: 1 aquecimento e %d medidos)" % (len(peds), n_repete + 1, n_repete))
    for i, x in enumerate(peds):
        if not isinstance(x, dict):
            pr.append("pedido %d malformado" % i)
            continue
        if x.get("indice") != i or x.get("aquecimento") is not (i == 0):
            pr.append("pedido %d com índice ou marca de aquecimento incoerentes" % i)
        if x.get("ids") != ids_esperados:
            pr.append("pedido %d com ids de prompt diferentes dos esperados (%d ids)" % (i, len(ids_esperados)))
        if x.get("gerados") != gerados_esperados:
            pr.append("pedido %d com ids gerados diferentes da referência 5c" % i)
        lg, kv = x.get("logits"), x.get("kv")
        if not isinstance(lg, list) or len(lg) != len(gerados_esperados) or not all(isinstance(h, str) and HEX64.fullmatch(h) for h in lg):
            pr.append("pedido %d: logits sem %d hashes sha256 válidos (1 do prefill e os passos do decode)" % (i, len(gerados_esperados)))
        if not isinstance(kv, list) or len(kv) != layers or not all(isinstance(par, list) and len(par) == 2 and all(isinstance(h, str) and HEX64.fullmatch(h) for h in par) for par in kv):
            pr.append("pedido %d: KV sem %d camadas de pares [K, V] sha256 válidos" % (i, layers))
    return pr[:20]


N_REPETE_COLETA = 3   # contrato congelado: --repete 3 = 1 aquecimento + 3 medidas = exatamente 4 pedidos (outra quantidade só por critério explícito)


def le_coleta(coleta, m, ids_arq, refs_dir, par_esperado=None, entradas_janela=None):
    """`par_esperado` = (sha base, sha candidato) da integridade validada da JANELA de medida (None só no modo --coleta-so, que identifica o par ao qual a prova se aplica); exige igualdade por papel.
    `entradas_janela` = {chave: sha} das entradas/critério da janela, que têm de ser as mesmas da coleta.
    Lê a pasta da execução de coleta (roda-f2c-janela.sh --coleta) e devolve o resultado do HC1b: DIR/gancho-original/gancho-logits-<m>.json, DIR/gancho-candidato/gancho-logits-<m>.json,
    DIR/coleta-manifesto.txt (modelo, n_repete, shas dos binários, dos dois arquivos e das entradas), DIR/integridade-fixos.txt, DIR/ambiente.txt. Sem prova completa: SEM PROVA."""
    def sem(motivos):
        return {"classe": "SEM PROVA", "divergencias": [], "motivo": "; ".join(motivos)[:700], "problemas": motivos}
    if not coleta or not os.path.isdir(coleta):
        return {"classe": "SEM PROVA", "divergencias": [], "motivo": "pasta da coleta não fornecida (--coleta): a coleta é uma execução própria", "problemas": []}
    man = X.le_chave_valor(os.path.join(coleta, "coleta-manifesto.txt"))
    pr = []
    if man.get("modelo") != m:
        pr.append("manifesto da coleta declara o modelo %r, a janela é do %s" % (man.get("modelo"), m))
    try:
        n_rep = int(man.get("n_repete", ""))
    except ValueError:
        n_rep = 0
    if n_rep != N_REPETE_COLETA:
        pr.append("manifesto com n_repete %r: o contrato exige %d (4 pedidos: 1 aquecimento e 3 medidos)" % (man.get("n_repete"), N_REPETE_COLETA))
    for k in ("binario_original", "binario_candidato", "gancho_original_sha", "gancho_candidato_sha", "ids_sha", "refs_sha"):
        if not HEX64.fullmatch(man.get(k, "")):
            pr.append("manifesto sem %s (sha256)" % k)
    pr += ["integridade da coleta: " + x for x in integridade_fixos(coleta, m)]
    pr += ["integridade da coleta (antes x depois): " + x for x in X.integridade(coleta)]
    fx = le_fixos_obs(coleta)
    if fx.get("binario-base") != man.get("binario_original") or fx.get("binario-cand") != man.get("binario_candidato"):
        pr.append("shas dos binários do manifesto diferentes dos itens binario-base/binario-cand da integridade")
    if man.get("binario_original") == man.get("binario_candidato"):
        pr.append("coleta com o mesmo binário nos dois papéis")
    if fx.get("cmp-ids-prompt") != man.get("ids_sha") or fx.get("ref5c-%s" % m) != man.get("refs_sha"):
        pr.append("shas de ids/referência do manifesto diferentes dos da integridade")
    if par_esperado is not None and (man.get("binario_original"), man.get("binario_candidato")) != tuple(par_esperado):
        pr.append("par incompatível: a coleta é do par (original %s, candidato %s) e a janela mediu (base %s, candidato %s); coleta de outro experimento, papéis invertidos ou só o candidato trocado" % (
            (man.get("binario_original") or "?")[:12], (man.get("binario_candidato") or "?")[:12], ((par_esperado[0] or "?")[:12]), ((par_esperado[1] or "?")[:12])))
    if entradas_janela is not None:
        # TODAS as chaves de integridade-antes.txt (entradas, critério, scripts, config, safetensors/índice do modelo, binários): mesmo conjunto e mesmo sha na janela e na coleta
        ant_c = X.le_shas(os.path.join(coleta, "integridade-antes.txt"))
        if not entradas_janela or not ant_c:
            pr.append("integridade-antes.txt ausente ou vazio na janela ou na coleta: entradas não vinculadas")
        else:
            dif = sorted(k for k in set(ant_c) | set(entradas_janela) if ant_c.get(k) != entradas_janela.get(k))
            if dif:
                pr.append("entradas/critério/modelo diferentes entre a janela e a coleta (ou ausentes de um lado): %s" % ", ".join(dif)[:300])
    amb = X.le_chave_valor(os.path.join(coleta, "ambiente.txt"))
    if amb.get("binario_base") != man.get("binario_original") or amb.get("binario_cand") != man.get("binario_candidato"):
        pr.append("ambiente.txt da coleta com shas de binário diferentes do manifesto")
    if X.le_txt(os.path.join(coleta, "potencia.txt")) is None and "potencia_fim" not in amb:
        pr.append("sem prova de potência na coleta")
    elif amb.get("potencia_inicio") != "370.00_W" or amb.get("potencia_fim") != "370.00_W":
        pr.append("potência da coleta fora de 370.00 W (ou ilegível)")
    arq_o = os.path.join(coleta, "gancho-original", "gancho-logits-%s.json" % m)
    arq_c = os.path.join(coleta, "gancho-candidato", "gancho-logits-%s.json" % m)
    if not (os.path.isfile(arq_o) and os.path.isfile(arq_c)):
        return sem(pr + ["arquivo de gancho ausente (gancho-original/ ou gancho-candidato/)"])
    if os.path.realpath(arq_o) == os.path.realpath(arq_c) or os.path.samefile(arq_o, arq_c):
        return sem(pr + ["o mesmo arquivo nos dois papéis (original e candidato)"])
    for papel, arq, chave in (("original", arq_o, "gancho_original_sha"), ("candidato", arq_c, "gancho_candidato_sha")):
        h = hashlib.sha256(open(arq, "rb").read()).hexdigest()
        if h != man.get(chave):
            pr.append("sha do arquivo %s (%s) diferente do manifesto" % (papel, h[:12]))
    ids_doc = X.le_json(ids_arq) or {}
    ids_esp = (((ids_doc.get("modelos") or {}).get(m) or {}).get("medio") or {}).get("ids")
    ref = X.le_json(os.path.join(refs_dir, "ref5c-%s.json" % m)) or {}
    ger_esp = next((c.get("gerados") for c in (ref.get("casos") or []) if c.get("nome") == "medio"), None)
    if not isinstance(ids_esp, list) or not isinstance(ger_esp, list):
        return sem(pr + ["ids do texto médio ou referência 5c indisponíveis para conferir a cobertura"])
    docs = {}
    for papel, arq in (("original", arq_o), ("candidato", arq_c)):
        d = X.le_json(arq)
        docs[papel] = d
        pr += ["coleção %s: %s" % (papel, x) for x in valida_gancho(d, m, LAYERS_DO_MODELO[m], ids_esp, ger_esp, N_REPETE_COLETA)]
    if pr:
        return sem(pr)
    dv = compara_gancho(docs["original"]["pedidos"], docs["candidato"]["pedidos"])
    return {"classe": "dentro" if not dv else "REGRESSÃO", "divergencias": dv, "problemas": [], "arquivos_sha": {"original": man["gancho_original_sha"], "candidato": man["gancho_candidato_sha"]},
            "binarios": {"original": man["binario_original"], "candidato": man["binario_candidato"]}, "par_aplicavel": {"base": man["binario_original"], "candidato": man["binario_candidato"]}}


# ---------------------------------------------------------------------------------------------------------------------------------- leitura da janela
def confere_ordem(saida, m, nomes_pastas):
    txt = X.le_txt(os.path.join(saida, "ordem.txt"))
    if txt is None:
        return "ordem.txt ausente: a ordem efetiva do controle não foi provada"
    rx = re.compile(ORDEM_RE % re.escape(m))
    linhas = []
    for x in txt.splitlines():
        mt = rx.match(x.strip())
        if mt:
            linhas.append({"chave": mt.group(2), "braco_nome": mt.group(4), "n": int(mt.group(5)), "braco": mt.group(6), "rc": int(mt.group(7))})
        elif len(x.split(" ")) > 1 and x.split(" ")[1].startswith("ctrl-%s-L100-" % m):
            return "linha de ordem.txt ilegível para o controle: %r" % x.strip()[:80]
    if len(linhas) != 9:
        return "ordem.txt com %d linhas do controle (esperadas 9): registro truncado ou ausente" % len(linhas)
    esperado = ["ctrl-%s-L100-medio-%02d-%s" % (m, i + 1, b) for i, b in enumerate(ORDEM_CONTROLE)]
    if [x["chave"] for x in linhas] != esperado:
        return "ordem efetiva de ordem.txt diferente do quadrado latino Dvelho Dnovo B2novo, B2novo Dvelho Dnovo, Dnovo B2novo Dvelho"
    if any(x["braco"] != x["braco_nome"] for x in linhas):
        return "braço registrado em ordem.txt diferente do braço do nome da pasta"
    if any(x["rc"] != 0 or x["n"] != 3 for x in linhas):
        return "ordem.txt com processo de controle de rc != 0 ou n != 3"
    if sorted(nomes_pastas) != sorted(esperado):
        return "pastas do controle diferentes das 9 esperadas"
    return None


def le_fixos_obs(saida):
    """integridade-fixos.txt -> {chave: sha observado} só dos itens com estado OK e sha válido."""
    txt = X.le_txt(os.path.join(saida, "integridade-fixos.txt")) or ""
    r = {}
    for x in txt.splitlines():
        c = x.split()
        if len(c) == 4 and c[3] == "OK" and re.fullmatch(r"[0-9a-f]{64}", c[2]) and c[1] == c[2]:
            r[c[0]] = c[2]
    return r


PROVA_RE = re.compile(r"^(ctrl-(1b|3b)-L100-medio-(\d\d)-([A-Za-z0-9]+)) braco=(\S+) bin=(\S+) ")


def confere_binarios(saida, m):
    """Uma prova por processo canônico (nome, índice e braço da ordem do quadrado latino), braço da linha = braço do nome, sha do binário = o do braço (Dvelho: base+gancho; Dnovo e B2novo: candidato).
    Os shas base/candidato de ambiente.txt têm de ser os do manifesto de integridade (itens binario-base e binario-cand já conferidos contra o FIXO); base e candidato diferentes. Duplicata, ausente, estranho ou de outro modelo: recusa."""
    amb = X.le_chave_valor(os.path.join(saida, "ambiente.txt"))
    base, cand = (amb.get("binario_base") or "").strip(), (amb.get("binario_cand") or "").strip()
    mot = []
    if not (re.fullmatch(r"[0-9a-f]{64}", base) and re.fullmatch(r"[0-9a-f]{64}", cand)):
        return ["ambiente.txt sem os shas binario_base e binario_cand"]
    if base == cand:
        mot.append("binário base+gancho e candidato com o mesmo sha: o par não difere pela O1")
    fx = le_fixos_obs(saida)
    if fx.get("binario-base") != base or fx.get("binario-cand") != cand:
        mot.append("shas de binário de ambiente.txt diferentes dos itens binario-base/binario-cand do manifesto de integridade (ou itens ausentes)")
    txt = X.le_txt(os.path.join(saida, "prova-variavel.txt"))
    if txt is None:
        return mot + ["prova-variavel.txt ausente: o binário de cada processo não foi provado"]
    esperado = {"ctrl-%s-L100-medio-%02d-%s" % (m, i + 1, b): b for i, b in enumerate(ORDEM_CONTROLE)}
    vistos = {}
    for linha in txt.splitlines():
        if not linha.strip():
            continue
        mt = PROVA_RE.match(linha)
        if not mt:
            mot.append("linha de prova-variavel.txt ilegível: %r" % linha.strip()[:80])
            continue
        chave, mod, _nn, braco_nome, braco_linha, sha_bin = mt.groups()
        if chave not in esperado:
            mot.append("prova de processo estranho ou de outro modelo: %s" % chave)
            continue
        if chave in vistos:
            mot.append("prova duplicada para %s" % chave)
            continue
        vistos[chave] = True
        if braco_nome != esperado[chave] or braco_linha != esperado[chave]:
            mot.append("processo %s: braço do nome (%s), da linha (%s) e da ordem canônica (%s) não coincidem" % (chave, braco_nome, braco_linha, esperado[chave]))
            continue
        alvo = base if esperado[chave] == "Dvelho" else cand
        if sha_bin != alvo:
            mot.append("processo %s (braço %s) rodou com o binário %s, não o do braço" % (chave, esperado[chave], sha_bin[:12]))
    for chave in esperado:
        if chave not in vistos:
            mot.append("sem prova de binário para %s" % chave)
    return mot


def integridade_fixos(saida, m):
    txt = X.le_txt(os.path.join(saida, "integridade-fixos.txt"))
    if txt is None:
        return ["integridade FIXA: manifesto integridade-fixos.txt ausente ou ilegível"]
    linhas = [x for x in txt.splitlines() if x.strip()]
    if not linhas:
        return ["integridade FIXA: manifesto vazio"]
    mot, vistas = [], {}
    antes = X.le_shas(os.path.join(saida, "integridade-antes.txt"))
    for x in linhas:
        c = x.split()
        if len(c) != 4:
            mot.append("integridade FIXA: linha truncada ou ilegível: %r" % x.strip()[:80])
            continue
        k, esp, obs, est = c
        if k in vistas:
            mot.append("integridade FIXA: chave %s repetida" % k)
        vistas[k] = c
        if k == "potencia":
            continue
        if est != "OK":
            mot.append("integridade FIXA: %s com estado %s (esperado OK)" % (k, est))
        elif not (re.fullmatch(r"[0-9a-f]{64}", esp) and esp == obs):
            mot.append("integridade FIXA: %s com sha esperado/observado diferentes ou inválidos" % k)
        elif antes.get(k) != obs:
            mot.append("integridade FIXA: %s com sha diferente do de integridade-antes.txt ou ausente dele" % k)
    falta = [k for k in FIXO_CHAVES + ("tokenizer-%s" % m,) if k not in vistas]
    if falta:
        mot.append("integridade FIXA: chave(s) faltando no manifesto: %s" % ", ".join(falta))
    return sorted(set(mot))


def processa(saida, ids_arq, refs_dir, textos_arq, coleta=None):
    refs = X.Refs(ids_arq, refs_dir, textos_arq)
    orc = X.le_json(os.path.join(saida, "orcamento.json"))
    orc = orc if isinstance(orc, dict) else {}
    decl = orc.get("modelos")
    if not (isinstance(decl, list) and len(decl) == 1 and decl[0] in MODELOS):
        raise Recusa("orcamento.json sem um modelo único declarado (\"modelos\": [\"1b\"] ou [\"3b\"])")
    m = decl[0]
    layers = LAYERS_DO_MODELO[m]
    rec = I.prova_potencia(saida)
    if rec:
        raise Recusa("; ".join(rec))
    nomes = sorted(os.listdir(os.path.join(saida, "celulas"))) if os.path.isdir(os.path.join(saida, "celulas")) else []
    if not any(CTRL_RE.match(n_) for n_ in nomes):
        raise Recusa("sem pastas ctrl-<modelo>-L100-medio-NN-<braço>: não é saída do roteiro da F2C")
    integ = sorted(set(X.integridade(saida) + integridade_fixos(saida, m) + confere_binarios(saida, m)))
    if X.le_txt(os.path.join(saida, "teto-excedido.txt")) is None:
        integ = sorted(set(integ + ["teto-excedido.txt ausente ou ilegível: o roteiro sempre o grava (vazio se nenhum teto estourou); sem ele não há prova de que nenhum processo passou do teto"]))
    ref_prob = I.confere_referencias(saida, m, ids_arq, refs_dir, textos_arq)
    nao_rodou = [str(x) for x in (orc.get("nao_rodou") or [])]
    estouros = F.teto_excedido(saida)
    ctrl = sorted((int(CTRL_RE.match(n_).group(2)), CTRL_RE.match(n_).group(3), n_) for n_ in nomes if CTRL_RE.match(n_) and CTRL_RE.match(n_).group(1) == m)
    procs = []
    for (nn, b, nome) in ctrl:
        c = B.le_pasta_b(saida, nome, m, "medio", 3, "B2" if b == "B2novo" else "D", refs, layers, CONTRATO)
        if nome in estouros:
            c["valido_controle"] = False
            c["motivos_f2"] = sorted(set(c.get("motivos_f2", []) + ["teto de processo excedido (processo inválido, sem repetição)"]))
        meds = c.get("meds") or []
        procs.append({"nn": nn, "braco": b, "pasta": nome, "c": c, "prefill": mediana([x["prefill_ms"] for x in meds]), "passo": I.mediana_passo(meds),
                      "valido": bool(c.get("valido_controle")) and c.get("status") != "NÃO RODOU"})
    motivos = []
    if any(x.startswith("ctrl-%s-L100-" % m) for x in nao_rodou):
        motivos.append("controle incompleto: " + "; ".join(x for x in nao_rodou if x.startswith("ctrl-%s-L100-" % m))[:300])
    mot_ordem = confere_ordem(saida, m, [p["pasta"] for p in procs]) if procs else "sem processos de controle"
    if mot_ordem:
        motivos.append(mot_ordem)
    if len(procs) == 9 and not quadrado_latino_ok([p["braco"] for p in procs]):
        motivos.append("ordem das pastas não é um quadrado latino 3x3")
    for p in procs:
        if not p["valido"]:
            motivos.append("%s: processo inválido: %s" % (p["pasta"], "; ".join(p["c"].get("motivos_temporais", []) + p["c"].get("motivos_f2", []) + p["c"].get("motivos_pedidos_f2", []))[:300] or "sem motivo gravado"))
        if fin(p["prefill"]) is None or p["prefill"] <= 0 or fin(p["passo"]) is None or p["passo"] <= 0:
            motivos.append("%s: sem prefill ou passo de decode válido" % p["pasta"])
    for p in procs:   # B2_novo: motivos estruturais de pedido (IB1..IB4, identidade) ou menos de 3 pedidos medidos válidos invalidam o processo; IB5 acima de 3% segue ressalva
        if p["braco"] != "B2novo" or p["c"].get("status") == "NÃO RODOU":
            continue
        est = list(p["c"].get("motivos_pedidos_f2") or [])
        if est or len(p["c"].get("pedidos_b") or []) != 3:
            p["valido"] = False
            motivos.append("%s: B2_novo com %d de 3 pedidos medidos válidos%s" % (p["pasta"], len(p["c"].get("pedidos_b") or []), (": " + "; ".join(est))[:300] if est else ""))
    for p in procs:   # o aquecimento de cada B2_novo faz o papel do smoke: sequência de 17 registros (IB1..IB5) e I8 válidas desde o primeiro pedido
        if p["braco"] != "B2novo" or p["c"].get("status") == "NÃO RODOU":
            continue
        aq = p["c"].get("aquecimento_b")
        if not aq:
            motivos.append("%s: aquecimento sem registros do nível B (sequência de 17 não provada)" % p["pasta"])
        else:
            if aq["problemas"]:
                motivos.append("%s: aquecimento do B2_novo inválido: %s" % (p["pasta"], "; ".join(aq["problemas"])[:200]))
            if aq.get("problemas_hi"):
                motivos.append("%s: aquecimento do B2_novo com I8 falha: %s" % (p["pasta"], "; ".join(aq["problemas_hi"])[:200]))
    div = [dict(x, pasta=p["pasta"]) for p in procs for x in p["c"]["seq"]["divergencias"]]
    ref_ausente = [p["pasta"] for p in procs if p["c"]["status"] != "NÃO RODOU" and not p["c"]["seq"]["referencia_disponivel"]]
    gerados = {b: {tuple(x["gerados"]) for p in procs if p["braco"] == b for x in p["c"].get("meds") or []} for b in BRACOS}
    ids_ok = all(len(gerados[b]) == 1 for b in BRACOS) and gerados["Dvelho"] == gerados["Dnovo"] == gerados["B2novo"]
    bloq = []
    if integ:
        bloq.append("integridade quebrada")
    if ref_prob:
        bloq.append("; ".join(ref_prob))
    if ref_ausente:
        bloq.append("referência 5c indisponível em: " + ", ".join(sorted(set(ref_ausente))))
    if motivos:
        bloq.append("controle incompleto ou inválido")
    por_braco = {b: [p for p in procs if p["braco"] == b and p["valido"]] for b in BRACOS}
    completo = all(len(por_braco[b]) == 3 for b in BRACOS) and not motivos
    pref = {b: (statistics.median([p["prefill"] for p in por_braco[b]]) if por_braco[b] else None) for b in BRACOS}
    pas = {b: (statistics.median([p["passo"] for p in por_braco[b]]) if por_braco[b] else None) for b in BRACOS}
    cls = classifica_janela_c(pref, pas) if completo else {"HC3": "SEM PROVA", "HC4": "SEM PROVA", "HC6": "SEM PROVA", "regressao": [], "razao_prefill_novo_sobre_velho": None,
                                                           "razao_decode_novo_sobre_velho": None, "razao_prefill_b2_sobre_dnovo": None, "razao_decode_b2_sobre_dnovo": None}
    if bloq and completo:
        cls = dict(cls, HC3="SEM PROVA", HC4="SEM PROVA", HC6="SEM PROVA", calculado_antes_do_bloqueio=cls)
    # HC1: ids
    hc1 = "SEM PROVA" if (bloq or not completo) else ("dentro" if (ids_ok and not div) else "REGRESSÃO")
    # HC2, HC5: só o B2_novo
    pedidos_b2 = [pe for p in por_braco["B2novo"] for pe in p["c"].get("pedidos_b", [])]
    hc2, hc5, tabela, hc2b = "sem dado", "sem dado", [], {}
    if pedidos_b2 and not bloq:
        pares = [B.pares_elegiveis(pe["f2"], pe["f2b"]) for pe in pedidos_b2]
        med = B.medidas_celula_b(pares, layers)
        tabela = med.get("h_pre_layer_mediana_camada_ns") or []
        dg = DEGRAU[m]
        antes = [x for l, x in enumerate(tabela) if l <= dg["antes_ate"] and x is not None]
        patamar = [x for l, x in enumerate(tabela) if l >= dg["patamar_de"] and x is not None]
        hc2b = {"soma_antes_do_degrau_ns": sum(antes), "soma_patamar_ns": sum(patamar), "camadas_antes": dg["antes_ate"] + 1, "camada_transicao": dg["transicao"], "camadas_patamar_de": dg["patamar_de"],
                "b_lanca_parcela": med.get("hb3"), "b_dtod_parcela": med.get("hb4"), "b_lanca_mais_b_dtod_parcela": (med["hb3"] + med["hb4"]) if fin(med.get("hb3")) is not None and fin(med.get("hb4")) is not None else None}
        hc2 = "descritivo"
        hc5 = "ressalva" if any(pe["ressalva_residuo"] for pe in pedidos_b2) else "comprovada"
    elif bloq:
        hc2 = hc5 = "SEM PROVA"
    # HC1b: execução de coleta própria (roda-f2c-janela.sh --coleta), validada em envelope, cobertura e procedência
    fxj = le_fixos_obs(saida)
    hc1b = le_coleta(coleta, m, ids_arq, refs_dir, par_esperado=(fxj.get("binario-base"), fxj.get("binario-cand")), entradas_janela=X.le_shas(os.path.join(saida, "integridade-antes.txt")))
    regressao = list(cls.get("regressao", []))
    if hc1 == "REGRESSÃO":
        regressao.append("HC1")
    if hc1b["classe"] == "REGRESSÃO":
        regressao.append("HC1b")
    return {"tipo": "f2c-resultado", "contrato_b": CONTRATO, "criterio": "criterio-f2c-20261111.md", "modelo_da_janela": m, "layers": layers, "integridade_problemas": integ, "referencia_problemas": ref_prob,
            "bloqueios": bloq, "motivos_controle": sorted(set(motivos)), "ordem_efetiva": mot_ordem or "confere com ordem.txt (quadrado latino, rc 0, n 3)",
            "processos": [{"nn": p["nn"], "braco": p["braco"], "pasta": p["pasta"], "valido": p["valido"], "prefill_ms_mediana": p["prefill"], "passo_ms_mediana": p["passo"]} for p in procs],
            "prefill_ms_mediana_por_braco": pref, "passo_ms_mediana_por_braco": pas, "comparacao": cls, "HC1": hc1, "HC1b": hc1b, "HC2a": hc2, "HC2b": hc2b, "HC5": hc5,
            "h_pre_layer_mediana_camada_ns_b2_novo": tabela, "ids_iguais_entre_braços": ids_ok, "divergencias_5c": div, "regressao": regressao, "orcamento": orc, "teto_excedido": sorted(estouros)}


# ---------------------------------------------------------------------------------------------------------------------------------- relatório
f, pc = F.f, F.pc


def relatorio(res):
    m = res["modelo_da_janela"]
    o = ["# F2C, plano de cópias, janela do modelo %s (criterio-f2c-20261111.md)" % m.upper(), "",
         "Resultado conforme os limites congelados: prefill +1,5%, decode +2%, B2_novo contra D_novo até 3% comprovada, até 10% com ressalva, acima SEM PROVA. Sem tolerância numérica: qualquer id ou bit diferente é regressão.", ""]
    if res["referencia_problemas"]:
        o += ["**Referência 5c sem prova:**", ""] + ["- %s" % x for x in res["referencia_problemas"]] + [""]
    if res["integridade_problemas"]:
        o += ["**Integridade QUEBRADA: tudo SEM PROVA (medidas brutas preservadas só como descrição).**", ""] + ["- %s" % x for x in res["integridade_problemas"]] + [""]
    c = res["comparacao"]
    o += ["## Braços (mediana das medianas dos 3 processos, texto médio)", "", "| braço | prefill (ms) | passo de decode (ms) |", "|---|---|---|"]
    for b in BRACOS:
        o.append("| %s | %s | %s |" % (b, f(res["prefill_ms_mediana_por_braco"].get(b), 3), f(res["passo_ms_mediana_por_braco"].get(b), 4)))
    o += ["", "- ordem efetiva: %s" % res["ordem_efetiva"]]
    for p in res["processos"]:
        o.append("  - %02d %s %s: válido %s, prefill %s ms, passo %s ms" % (p["nn"], p["braco"], p["pasta"], p["valido"], f(p["prefill_ms_mediana"], 3), f(p["passo_ms_mediana"], 4)))
    for x in res["motivos_controle"]:
        o.append("- %s" % x)
    o += ["", "## Hipóteses", "",
          "- HC1 (ids iguais aos de D_velho e às referências 5c): **%s**" % res["HC1"],
          "- HC1b (hash dos bits dos logits e do KV): **%s**%s" % (res["HC1b"]["classe"], (" (%s)" % res["HC1b"]["motivo"]) if res["HC1b"].get("motivo") else ""),
          "- HC3 (prefill D_novo contra D_velho, razão %s; limite +1,5%%): **%s**" % (pc(c.get("razao_prefill_novo_sobre_velho")), c["HC3"]),
          "- HC4 (decode D_novo contra D_velho, razão %s; limite +2%%): **%s**" % (pc(c.get("razao_decode_novo_sobre_velho")), c["HC4"]),
          "- HC6 (B2_novo contra D_novo, prefill, razão %s; decode %s): **%s**" % (pc(c.get("razao_prefill_b2_sobre_dnovo")), pc(c.get("razao_decode_b2_sobre_dnovo")), c["HC6"]),
          "- HC5 (resíduo do nível B em B2_novo, até 3%%): **%s**" % res["HC5"],
          "- HC2a e HC2b (H_PRE_LAYER por camada de B2_novo, descritivas): **%s**" % res["HC2a"]]
    if res["HC2b"]:
        d = res["HC2b"]
        o.append("  - soma antes do degrau (camadas 0 a %d): %s ns; soma do patamar (camada %d em diante): %s ns; transição na camada %d (fora das duas somas); parcelas B_LANCA %s, B_DTOD %s, B_LANCA + B_DTOD %s" % (
            d["camadas_antes"] - 1, f(d["soma_antes_do_degrau_ns"], 0), d["camadas_patamar_de"], f(d["soma_patamar_ns"], 0), d["camada_transicao"], pc(d["b_lanca_parcela"]), pc(d["b_dtod_parcela"]), pc(d["b_lanca_mais_b_dtod_parcela"])))
    if res["h_pre_layer_mediana_camada_ns_b2_novo"]:
        o += ["", "| camada | H_PRE_LAYER mediana (ns) |", "|---|---|"] + ["| %d | %s |" % (l, f(x, 0)) for l, x in enumerate(res["h_pre_layer_mediana_camada_ns_b2_novo"])]
    o += ["", "## Regressão", "", "**%s**" % (", ".join(res["regressao"]) if res["regressao"] else "nenhuma"),
          "", "Sem ganho em prefill_ms mas com H_PRE_LAYER menor não é regressão. A comparação com o B2 de 14:10 é só contexto (outra janela, outro instrumento). Qualquer mudança de desempenho no motor além desta exige nova decisão do Breno."]
    return "\n".join(o) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="extrator da F2C (plano de cópias)")
    ap.add_argument("--saida")
    ap.add_argument("--ids", default=os.path.join(AQUI, "cmp-ids-prompt.json"))
    ap.add_argument("--refs", default=os.path.join(AQUI, "5c-referencia"))
    ap.add_argument("--textos", default=os.path.join(AQUI, "5c-textos", "textos.json"))
    ap.add_argument("--coleta", default=None, help="pasta da execução de coleta (roda-f2c-janela.sh --coleta) para o HC1b")
    ap.add_argument("--coleta-so", action="store_true", help="--saida é a pasta de uma execução de coleta (roda-f2c-janela.sh --coleta): só o HC1b, gravando f2c-coleta-resultado.json/.md")
    a = ap.parse_args(argv)
    if not a.saida or not os.path.isdir(a.saida):
        print("pasta de saída inexistente: %s" % a.saida, file=sys.stderr)
        return 2
    if a.coleta_so:
        man = X.le_chave_valor(os.path.join(a.saida, "coleta-manifesto.txt"))
        m = man.get("modelo")
        if m not in MODELOS:
            print("f2c-extrai: RECUSADO: coleta-manifesto.txt sem modelo válido", file=sys.stderr)
            return 2
        h = le_coleta(a.saida, m, a.ids, a.refs)
        rc_txt = X.le_txt(os.path.join(a.saida, "teste-gpu.rc"))
        try:
            rc_t = int(rc_txt.strip()) if rc_txt is not None else None
        except ValueError:
            rc_t = None
        proc_ok = bool(re.fullmatch(r"[0-9a-f]{40}", man.get("teste_gpu_commit", ""))) and bool(HEX64.fullmatch(man.get("teste_gpu_exe_sha", "")))
        tg = {"executado": rc_txt is not None, "rc": rc_t, "passou": rc_t == 0 and proc_ok, "procedencia_completa": proc_ok, "commit_worktree": man.get("teste_gpu_commit"), "sha_executavel": man.get("teste_gpu_exe_sha")}
        h["par_aplicavel"] = {"base": man.get("binario_original"), "candidato": man.get("binario_candidato")}
        with open(os.path.join(a.saida, "f2c-coleta-resultado.json"), "w", encoding="utf-8") as fo:
            json.dump({"tipo": "f2c-coleta-resultado", "modelo": m, "HC1b": h, "teste_gpu": tg}, fo, ensure_ascii=False, indent=1)
        with open(os.path.join(a.saida, "f2c-coleta-resultado.md"), "w", encoding="utf-8") as fo:
            fo.write("# F2C, coleta HC1b, modelo %s\n\nHC1b: **%s** (vale só para o par: base %s, candidato %s)\n\nTeste de erro em GPU: %s\n\n%s\n" % (m.upper(), h["classe"], (man.get("binario_original") or "?")[:12], (man.get("binario_candidato") or "?")[:12],
                "não executado" if not tg["executado"] else ("passou" if tg["passou"] else ("rc 0, procedência incompleta (commit do worktree e sha do executável não gravados)" if tg["rc"] == 0 else "FALHOU (rc %s)" % tg["rc"])), "\n".join("- " + x for x in (h.get("problemas") or []) + (h.get("divergencias") or [])) or "sem divergência nem problema de cobertura/procedência"))
        print("f2c-extrai: HC1b %s; gravados f2c-coleta-resultado.json e f2c-coleta-resultado.md" % h["classe"])
        return 0
    try:
        res = processa(a.saida, a.ids, a.refs, a.textos, a.coleta)
    except Recusa as ex:
        print("f2c-extrai: RECUSADO: %s" % ex, file=sys.stderr)
        with open(os.path.join(a.saida, "f2c-resultado.json"), "w", encoding="utf-8") as fo:
            json.dump({"tipo": "f2c-resultado", "recusado": True, "motivo": str(ex)}, fo, ensure_ascii=False, indent=1)
        with open(os.path.join(a.saida, "f2c-resultado.md"), "w", encoding="utf-8") as fo:
            fo.write("# F2C: RECUSADO\n\n%s\n\nNenhuma classificação foi feita; os dados brutos ficam na pasta.\n" % ex)
        return 2
    X.limpa_nao_finitos(res)
    with open(os.path.join(a.saida, "f2c-resultado.json"), "w", encoding="utf-8") as fo:
        json.dump(res, fo, ensure_ascii=False, indent=1, allow_nan=False)
    with open(os.path.join(a.saida, "f2c-resultado.md"), "w", encoding="utf-8") as fo:
        fo.write(relatorio(res))
    print("f2c-extrai: regressão %s; gravados f2c-resultado.json e f2c-resultado.md" % (", ".join(res["regressao"]) or "nenhuma"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
