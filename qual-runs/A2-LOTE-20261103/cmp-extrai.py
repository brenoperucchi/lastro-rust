#!/usr/bin/env python3
"""Extrator da comparação Lastro x llama.cpp (opção d), conforme criterio-comparacao-llamacpp-20261108.md (emendas 1 a 6).

    python3 -I cmp-extrai.py --saida DIR_DA_JANELA [--ids cmp-ids-prompt.json] [--refs 5c-referencia] [--textos 5c-textos/textos.json]

Só biblioteca padrão. NÃO roda nada, NÃO usa GPU: lê o que a janela gravou, confere ids e contagens, calcula mediana/min/max por célula e as razões
Lastro/llama.cpp (S1, S2 e S3 sempre SEPARADAS e rotuladas) e grava DIR/cmp-resultado.json e DIR/cmp-resultado.md.

Layout lido (escrito por roda-cmp-lastro.sh, roda-cmp-llamacpp.sh e roda-cmp-janela.sh):
  DIR/integridade-antes.txt, integridade-depois.txt ("chave sha" por linha), integridade-fixos.txt (opcional, "chave OK|FALHA ..."), ambiente.txt (opcional),
  DIR/orcamento.json (cortes), DIR/ordem.txt (ordem gravada);
  DIR/celulas/<lado>-<modelo>-<serie>-<texto>-<estado>/celula.json (+ monitor.csv, portao.txt)
    lastro:   pK/saida/{medidas-<modelo>.json,gera-<modelo>.json,gera-final.txt}, pK/rc.txt, pK/wall_ms.txt  (frio: p0 descartado, p1..pN; quente: p1)
    llamacpp: cmdline.txt, version.txt, servidor-K.log, carga-K.ms, resp-K.json, resp-K.rc, resp-K.ms  (K=0 é o descartado; 1..N as medidas)
Definições (emenda 2, E3; emenda 4): decode = média por passo (Lastro: soma(passo_ms)/nº de passos, nunca por diferença de totais; llama.cpp:
timings.predicted_per_token_ms, ou predicted_ms/(predicted_n-1) quando o campo não vem). Razão = mediana Lastro / mediana llama.cpp (menor que 1 =
Lastro mais rápido). Nada é estimado: sem prova ou não rodou, não há razão.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

NOVOS = 64
PASSOS = NOVOS - 1
RUIDOSA = 0.15            # emenda 4, item 2 (palpite declarado antes dos dados)
TEMP_MAX = 80.0           # item (4) (palpite declarado antes dos dados)
COBERTURA_TOL = 5.0       # r111: a última amostra do monitor tem de ficar a no máximo 5 s antes do fim da célula
COBERTURA_INI_TOL = 5.0   # r112 (emenda 7): a 1ª amostra do monitor tem de ficar a no máximo 5 s depois de inicio_coleta_epoch (palpite declarado, igual às outras)
LACUNA_MAX = 10.0         # r111: intervalo máximo entre amostras consecutivas do monitor (a coleta é a 1 Hz)
EMPATE = 1e-3             # item (2): gap da referência abaixo disto = empate
DIFERE_RAZOES = 1.5       # E7
FATIA = 0.30              # E7
MEM_NOMINAL = 2400        # item (4), MT/s
MODELOS = ("1b", "3b")
TEXTOS = ("curto", "medio", "longo")
ESTADOS = ("frio", "quente")
SERIES = ("S1", "S2", "S3")
ROTULO_SERIE = {
    "S1": "llama.cpp S1 (tudo na GPU, -ngl 99, -fa on, KV f16 no llama.cpp contra f32 no Lastro)",
    "S2": "llama.cpp S2 (--n-cpu-moe 22 no 1B e 29 no 3B, -ngl 99, -fa on, KV f16 no llama.cpp contra f32 no Lastro; descarregamento de ordem de grandeza parecida, "
          "NÃO o mesmo: o Lastro descarrega ~90% dos experts de cada camada, o llama.cpp camadas inteiras; op offload ligado, então o prefill da S2 não deve ser "
          "lido como 'experts calculados na CPU')",
    "S3": "llama.cpp S3 (KV f32 com -fa off, paridade de armazenamento; configuração prática, sem atribuir a razão só ao motor)",
}


def rotulo_razao(serie):
    return "Lastro 10% experts GPU contra " + ROTULO_SERIE[serie]


# ---------------------------------------------------------------------------------------------------------------------------------- estatística
def media(v):
    return sum(v) / len(v)


def estat(v):
    """mediana/min/max e a marca de ruidosa (emenda 4, item 2: (max-min)/mediana > 15% com 2 ou mais medidas; com 2 a mediana é a média das duas)."""
    if not v:
        return None
    med = statistics.median(v)
    mn, mx = min(v), max(v)
    ruidosa = len(v) >= 2 and med > 0 and (mx - mn) / med > RUIDOSA
    return {"n": len(v), "mediana": med, "min": mn, "max": mx, "ruidosa": bool(ruidosa), "valores": list(v)}


def num(v, nome, positivo=False):
    """Número JSON finito (r111, rev-1 P2): recusa bool, texto, NaN, Infinity e o que estoura o float. positivo=True exige > 0 (passos e prefill); senão >= 0."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError("%s não é número (%r)" % (nome, v))
    try:
        x = float(v)
    except OverflowError:
        raise ValueError("%s fora do intervalo do float" % nome)
    if not math.isfinite(x):
        raise ValueError("%s não finito (%r)" % (nome, v))
    if (positivo and x <= 0) or x < 0:
        raise ValueError("%s fora do domínio (%s, esperado %s)" % (nome, x, "> 0" if positivo else ">= 0"))
    return x


def media_por_passo(passo_ms):
    return sum(passo_ms) / len(passo_ms)


def passo_llama(t):
    """ms por passo de decode do llama.cpp: predicted_per_token_ms; sem o campo, predicted_ms/(predicted_n-1) (63 passos, nunca /64).
    Levanta ValueError se o valor não for finito e positivo."""
    if t.get("predicted_per_token_ms") is not None:
        return num(t["predicted_per_token_ms"], "predicted_per_token_ms", True), "predicted_per_token_ms"
    n = t.get("predicted_n")
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("predicted_n inválido para predicted_ms/(predicted_n-1): %r" % (n,))
    v = num(t.get("predicted_ms"), "predicted_ms", True) / (n - 1)
    if not math.isfinite(v):
        raise ValueError("passo calculado não finito")
    return v, "predicted_ms/(predicted_n-1)"


def _rejeita_constante(c):
    raise ValueError("constante JSON não finita: %s" % c)


def limpa_nao_finitos(o):
    """Troca por None todo float não finito dentro de o (dict/list) e devolve True se achou algum (rede de segurança: o JSON final é estrito)."""
    achou = False
    if isinstance(o, dict):
        for k in list(o):
            if isinstance(o[k], float) and not math.isfinite(o[k]):
                o[k] = None
                achou = True
            elif isinstance(o[k], (dict, list)):
                achou = limpa_nao_finitos(o[k]) or achou
    elif isinstance(o, list):
        for i, x in enumerate(o):
            if isinstance(x, float) and not math.isfinite(x):
                o[i] = None
                achou = True
            elif isinstance(x, (dict, list)):
                achou = limpa_nao_finitos(x) or achou
    return achou


# ---------------------------------------------------------------------------------------------------------------------------------- leitura
def le_json(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f, parse_constant=_rejeita_constante)
    except (OSError, ValueError, RecursionError):
        return None


def le_txt(p):
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def le_num(p):
    t = le_txt(p)
    try:
        x = float(t.strip().split()[0])
    except (AttributeError, ValueError, IndexError):
        return None
    return x if math.isfinite(x) else None


def le_num_dom(p, nome, mot):
    """Como le_num, mas se o arquivo existe e o valor é ilegível, não finito ou negativo, registra o motivo (SEM PROVA da célula)."""
    x = le_num(p)
    if x is None and le_txt(p) is not None:
        mot.append("%s ilegível ou não finito" % nome)
    elif x is not None and x < 0:
        mot.append("%s negativo (%s)" % (nome, x))
        return None
    return x


def le_chave_valor(p):
    r = {}
    for linha in (le_txt(p) or "").splitlines():
        if "=" in linha:
            k, v = linha.split("=", 1)
            r[k.strip()] = v.strip()
    return r


def le_shas(p):
    r = {}
    for linha in (le_txt(p) or "").splitlines():
        partes = linha.split()
        if partes:
            r[partes[0]] = partes[1] if len(partes) > 1 else ""
    return r


class Refs:
    def __init__(self, ids_arq, refs_dir, textos_arq):
        d = le_json(ids_arq) or {}
        self.ids = d.get("modelos", {})
        self.refs = {}
        for m in MODELOS:
            r = le_json(os.path.join(refs_dir, "ref5c-%s.json" % m))
            self.refs[m] = {c["nome"]: c for c in (r or {}).get("casos", [])}
        t = le_json(textos_arq) or {}
        self.contagem = {x["nome"]: x.get("ids_com_template", {}) for x in t.get("textos", [])}

    def ids_prompt(self, m, t):
        return (self.ids.get(m, {}).get(t) or {}).get("ids")

    def n_esperado(self, m, t):
        a = self.ids_prompt(m, t)
        return len(a) if a else None

    def ref(self, m, t):
        return self.refs.get(m, {}).get(t)


# ---------------------------------------------------------------------------------------------------------------------------------- integridade
def integridade(saida):
    motivos = []
    antes, depois = le_shas(os.path.join(saida, "integridade-antes.txt")), le_shas(os.path.join(saida, "integridade-depois.txt"))
    if not antes or not depois:
        motivos.append("integridade não conferida: faltam integridade-antes.txt/integridade-depois.txt")
    else:
        if antes != depois:
            motivos.append("integridade: sha mudou entre antes e depois")
        for k, h in list(antes.items()) + list(depois.items()):
            if not h:
                motivos.append("integridade: %s ilegível" % k)
    fixos = le_txt(os.path.join(saida, "integridade-fixos.txt"))
    for linha in (fixos or "").splitlines():
        if "FALHA" in linha:
            motivos.append("integridade: " + linha.strip())
    return sorted(set(motivos))


# ---------------------------------------------------------------------------------------------------------------------------------- células
def ts_epoch(txt):
    """Carimbo do nvidia-smi ('2026/11/08 19:14:28.123', hora local) em segundos; None se ilegível."""
    base, _, frac = txt.strip().partition(".")
    try:
        t = time.mktime(time.strptime(base, "%Y/%m/%d %H:%M:%S"))
        return t + float("0." + frac) if frac.isdigit() else t
    except (ValueError, OverflowError):
        return None


def le_monitor(p):
    """(temperaturas, carimbos) das linhas legíveis do monitor.csv (timestamp,util,sm,mem,temp,memoria); None se o arquivo não existe. Linha ilegível é ignorada."""
    t = le_txt(p)
    if t is None:
        return None
    temps, stamps = [], []
    for linha in t.splitlines():
        c = [x.strip() for x in linha.split(",")]
        if len(c) < 5:
            continue
        try:
            v = float(c[4])
        except ValueError:
            continue
        if not math.isfinite(v):
            continue
        temps.append(v)
        stamps.append(ts_epoch(c[0]))
    return temps, stamps


def temp_max_csv(p):
    """Maior temperatura legível (5ª coluna do monitor.csv); None se não há nenhuma."""
    r = le_monitor(p)
    return max(r[0]) if r and r[0] else None


def analisa_monitor(d, meta):
    """Falha FECHADA (r111, rev-1 P2): sem temperatura legível, com falha de coleta registrada, com coleta que não cobre a célula ou com > 80 C, a célula é SEM PROVA."""
    mot = []
    if meta.get("monitor_falha"):
        mot.append("falha do monitor registrada pelo roteiro: %s" % meta.get("monitor_falha"))
    r = le_monitor(os.path.join(d, "monitor.csv"))
    if r is None:
        mot.append("monitor.csv ausente: sem prova de temperatura da GPU")
        return mot, None
    temps, stamps = r
    if not temps:
        mot.append("monitor.csv vazio ou sem amostra de temperatura legível: sem prova de temperatura da GPU")
        return mot, None
    tm = max(temps)
    if tm > TEMP_MAX:
        mot.append("temperatura da GPU %.0f C > %.0f C em amostra da célula" % (tm, TEMP_MAX))
    try:
        ini, fim = num(meta.get("inicio_epoch"), "inicio_epoch"), num(meta.get("fim_epoch"), "fim_epoch")
    except ValueError:
        mot.append("celula.json sem inicio_epoch/fim_epoch legíveis: cobertura do monitor não verificável")
        return mot, tm
    try:
        ini_col = num(meta.get("inicio_coleta_epoch"), "inicio_coleta_epoch")
    except ValueError:
        mot.append("celula.json sem inicio_coleta_epoch legível: cobertura inicial do monitor não verificável")
        ini_col = None
    if None in stamps:
        mot.append("carimbos de tempo do monitor.csv ilegíveis: cobertura não verificável")
        return mot, tm
    ts = sorted(stamps)
    if ts[-1] < fim - COBERTURA_TOL:
        mot.append("a coleta do monitor parou %.0f s antes do fim da célula (última amostra antes do fim - %.0f s)" % (fim - ts[-1], COBERTURA_TOL))
    elif ts[0] > fim:
        mot.append("amostras do monitor todas depois do fim da célula")
    if ini_col is not None and ts[0] > ini_col + COBERTURA_INI_TOL:
        mot.append("a coleta do monitor só começou %.0f s depois do início da coleta da célula (limite %.0f s)" % (ts[0] - ini_col, COBERTURA_INI_TOL))
    lac = max((b - a for a, b in zip(ts, ts[1:])), default=0.0)
    if lac > LACUNA_MAX:
        mot.append("lacuna de %.0f s na coleta do monitor (máximo %.0f s)" % (lac, LACUNA_MAX))
    return mot, tm


def confere_portao_pmon(d):
    """Emenda 8: se a célula tem portao.txt, ele precisa trazer `pmon_antes_veredito=OK` (amostra pmon de 3 s antes da célula, falha fechada).
    Sem a linha (portão de formato antigo) ou com outro valor, SEM PROVA. Sem portao.txt na pasta, vale só o portao_ok do celula.json."""
    arq = os.path.join(d, "portao.txt")
    if not os.path.exists(arq):
        return []
    try:
        with open(arq, encoding="utf-8", errors="replace") as fh:
            linhas = fh.read().splitlines()
    except OSError:
        return ["portao.txt ilegível: sem prova do portão da GPU"]
    ver = [l.split("=", 1)[1] for l in linhas if l.startswith("pmon_antes_veredito=")]
    if not ver:
        return ["portao.txt sem a amostra pmon (emenda 8): sem prova do portão da GPU"]
    if ver[-1] != "OK":
        return ["portão pmon antes da célula: %s" % ver[-1]]
    return []


def base_celula(d, meta):
    mot = []
    if meta.get("portao_ok") is False:
        mot.append("portão antes da célula falhou: %s" % meta.get("portao_motivo", "sem motivo gravado"))
    elif meta.get("portao_ok") is None:
        mot.append("portão antes da célula sem registro")
    mot += confere_portao_pmon(d)
    if meta.get("rc") not in (None, 0):
        mot.append("o roteiro da célula terminou com código %s (%s)" % (meta.get("rc"), meta.get("falhas", "")))
    m2, tm = analisa_monitor(d, meta)
    return mot + m2, tm


def le_celula_lastro(d, meta, refs):
    m, t, est = meta["modelo"], meta["texto"], meta["estado"]
    mot, tm = base_celula(d, meta)
    esperado = refs.ids_prompt(m, t)
    nesp = refs.n_esperado(m, t)
    procs = sorted([x for x in os.listdir(d) if x.startswith("p") and x[1:].isdigit()], key=lambda s: int(s[1:]))
    meds, cargas, walls = [], [], []
    for p in procs:
        pd = os.path.join(d, p)
        k = int(p[1:])
        rc = le_num(os.path.join(pd, "rc.txt"))
        wall = le_num_dom(os.path.join(pd, "wall_ms.txt"), "%s/wall_ms.txt" % p, mot)
        if rc is None or int(rc) != 0:
            mot.append("%s: execução com erro ou código != 0 (rc=%s)" % (p, "?" if rc is None else int(rc)))
        sd = os.path.join(pd, "saida")
        mj = le_json(os.path.join(sd, "medidas-%s.json" % m))
        final = le_chave_valor(os.path.join(sd, "gera-final.txt"))
        gj = le_json(os.path.join(sd, "gera-%s.json" % m))
        if mj is None:
            mot.append("%s: sem medidas-%s.json" % (p, m))
            continue
        # emenda 4, item 4: cruza medidas-<nome>.json com gera-final.txt e com o gera-<nome>.json da mesma saída (o envelope da janela é conferido em integridade()).
        if final.get("integro") != "1":
            mot.append("%s: gera-final.txt ausente ou sem integro=1" % p)
        if gj is None:
            mot.append("%s: sem gera-%s.json" % (p, m))
        elif gj.get("falha") or gj.get("integro") is False:
            mot.append("%s: gera-%s.json com falha/integridade quebrada (%s)" % (p, m, gj.get("falha")))
        if mj.get("modelo") != m or mj.get("texto") != t:
            mot.append("%s: medidas-%s.json é de %s/%s, não %s/%s" % (p, m, mj.get("modelo"), mj.get("texto"), m, t))
        entradas = mj.get("medidas", [])
        if not isinstance(entradas, list):
            mot.append("%s: medidas-%s.json sem lista de medidas" % (p, m))
            continue
        ignora = est == "frio" and k == 0   # processo descartado do frio (o "aquecimento" do item (3)): só conferido, nunca medido
        soma_total = 0.0
        t_seq_primeira = None
        soma_valida = True
        for e in entradas:
            if not isinstance(e, dict):
                mot.append("%s: entrada de medida inválida" % p)
                soma_valida = False
                continue
            if e.get("falha"):
                mot.append("%s: pedido '%s' com falha: %s" % (p, e.get("rotulo"), e.get("falha")))
                continue
            rot = "%s/%s" % (p, e.get("rotulo"))
            try:
                tot = num(e.get("total_ms"), "total_ms")
                tseq = num(e.get("T_sequencia_ms"), "T_sequencia_ms")
                soma_total += tot
            except ValueError as ex:
                mot.append("%s: valor numérico inválido: %s" % (rot, ex))
                soma_valida = False
                continue
            if t_seq_primeira is None:
                t_seq_primeira = tseq
            if e.get("descartada") or ignora:
                continue
            g = e.get("gerados") or []
            ids = e.get("ids") or []
            lista = e.get("passo_ms")
            try:
                if not isinstance(lista, list):
                    raise ValueError("passo_ms ausente ou não é lista")
                passo = [num(x, "passo_ms", True) for x in lista]
                pre = num(e.get("prefill_ms"), "prefill_ms", True)
                cont = e.get("cont") or {}
                if not isinstance(cont, dict):
                    raise ValueError("cont não é objeto")
                needed, cpu_, hits_ = (num(cont.get(k) or 0, "cont." + k) for k in ("needed", "cpu", "hits"))
                guardada = None if e.get("media_por_passo_ms") is None else num(e.get("media_por_passo_ms"), "media_por_passo_ms")
                if not passo:
                    mp = None
                else:
                    mp = media_por_passo(passo)
                    if not (math.isfinite(mp) and math.isfinite(sum(passo))):
                        raise ValueError("média ou soma dos passos não finita (estouro do float)")
            except (ValueError, TypeError, OverflowError) as ex:
                mot.append("%s: valor numérico inválido: %s" % (rot, ex))
                continue
            if len(g) != NOVOS:
                mot.append("%s: %d tokens gerados, não %d (contagem divergente)" % (rot, len(g), NOVOS))
            if len(passo) != PASSOS:
                mot.append("%s: %d passos de decode, não %d" % (rot, len(passo), PASSOS))
            if esperado is None:
                mot.append("%s: sem ids de prompt esperados em cmp-ids-prompt.json" % rot)
            elif ids != esperado:
                mot.append("%s: ids do prompt diferentes de cmp-ids-prompt.json (%d contra %d)" % (rot, len(ids), len(esperado)))
            if nesp is not None and len(ids) != nesp:
                mot.append("%s: prompt com %d ids, esperado %d" % (rot, len(ids), nesp))
            if not passo:
                continue
            if guardada is None or abs(guardada - mp) > 1e-6 * max(1.0, mp):
                mot.append("%s: media_por_passo_ms guardada (%s) não bate com soma/len (%.6f)" % (rot, guardada, mp))
            meds.append({
                "processo": p, "rotulo": e.get("rotulo"), "prefill_ms": pre, "decode_ms_passo": mp, "mediana_passos_descritiva": statistics.median(passo),
                "total_ms": tot, "T_sequencia_ms": tseq, "soma_passos_ms": sum(passo), "gerados": g, "ids_n": len(ids), "cont": cont,
                "fracao_cpu_needed": (cpu_ / needed) if needed else None,
                "fracao_hits_needed": (hits_ / needed) if needed else None,
            })
        if wall is not None and soma_valida and entradas and not any(e.get("falha") for e in entradas):
            walls.append((k, wall))
            if (est == "frio" and k >= 1 and len(entradas) == 1) or est == "quente":
                cargas.append(wall - soma_total + (t_seq_primeira or 0.0))
    return {"meds": meds, "cargas_limite_ms": cargas, "esperado_n": nesp, "temp_max": tm}, mot


def le_celula_llama(d, meta, refs):
    m, t, est = meta["modelo"], meta["texto"], meta["estado"]
    mot, tm = base_celula(d, meta)
    nesp = refs.n_esperado(m, t)
    cmd = le_txt(os.path.join(d, "cmdline.txt")) or ""
    if not cmd.strip():
        mot.append("sem cmdline.txt (linha de comando do servidor não gravada)")
    if "--host 127.0.0.1" not in cmd:
        mot.append("linha de comando sem --host 127.0.0.1")
    if est == "frio":
        if "--no-warmup" not in cmd:
            mot.append("frio sem --no-warmup na linha de comando")
    elif "--no-warmup" in cmd:
        mot.append("quente com --no-warmup na linha de comando (o quente usa o aquecimento padrão)")
    if not (le_txt(os.path.join(d, "version.txt")) or "").strip():
        mot.append("sem llama-server --version gravado")
    logs = sorted(x for x in os.listdir(d) if x.startswith("servidor-") and x.endswith(".log"))
    # emenda 6 (2): a prova do frio sem aquecimento é a linha de comando gravada com --no-warmup. A linha 'warming up the model' é LOG_TRC no build 0893ec9
    # e não sai na verbosidade padrão: a checagem do log é só INFORMATIVA (não é prova nem falha).
    obs = []
    if est == "frio":
        for lg in logs:
            if "warming up" in (le_txt(os.path.join(d, lg)) or "").lower():
                obs.append("%s tem a linha 'warming up' (informativo: a prova do frio é --no-warmup na linha de comando; a ausência da linha não prova nada)" % lg)
    ks = sorted(int(x[5:-5]) for x in os.listdir(d) if x.startswith("resp-") and x.endswith(".json") and x[5:-5].isdigit())
    meds, cargas = [], []
    for k in ks:
        rc = le_txt(os.path.join(d, "resp-%d.rc" % k))
        http = (rc or "").split()
        if len(http) < 2 or http[0] != "0" or http[1] != "200":
            mot.append("pedido %d: curl/HTTP %s" % (k, rc.strip() if rc else "sem registro"))
        carga = le_num_dom(os.path.join(d, "carga-%d.ms" % k), "carga-%d.ms" % k, mot)
        if carga is not None and ((est == "frio" and k >= 1) or (est == "quente" and k == 0)):
            cargas.append(carga)
        if k == 0:
            continue
        r = le_json(os.path.join(d, "resp-%d.json" % k))
        rot = "pedido %d" % k
        if not isinstance(r, dict):
            mot.append("%s: resposta ilegível" % rot)
            continue
        tm_ = r.get("timings")
        if not isinstance(tm_, dict) or "prompt_ms" not in tm_ or "predicted_ms" not in tm_:
            mot.append("%s: resposta sem timings" % rot)
            continue
        pn, gn = tm_.get("prompt_n"), tm_.get("predicted_n")
        toks = r.get("tokens") or []
        try:
            pre = num(tm_.get("prompt_ms"), "prompt_ms", True)
            pred = num(tm_.get("predicted_ms"), "predicted_ms")
        except ValueError as ex:
            mot.append("%s: valor numérico inválido: %s" % (rot, ex))
            continue
        if nesp is not None and pn != nesp:
            mot.append("%s: prompt_n %s, esperado %s (contagem de prompt divergente)" % (rot, pn, nesp))
        if gn != NOVOS or len(toks) != NOVOS:
            mot.append("%s: predicted_n %s e %d tokens, esperado %d (parada antes de 64 ou contagem divergente)" % (rot, gn, len(toks), NOVOS))
            continue
        try:
            passo, fonte = passo_llama(tm_)
        except ValueError as ex:
            mot.append("%s: valor numérico inválido: %s" % (rot, ex))
            continue
        meds.append({"pedido": k, "prefill_ms": pre, "decode_ms_passo": passo, "fonte_decode": fonte, "gerados": toks, "prompt_n": pn,
                     "predicted_ms": pred, "content": r.get("content")})
    return {"meds": meds, "cargas_ms": cargas, "esperado_n": nesp, "temp_max": tm, "kv_fa": le_txt(os.path.join(d, "kv-fa.txt")), "observacoes": obs}, mot


def num_ou_none(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def resume_celula(meta, dados, mot):
    meds = dados["meds"]
    esp = meta.get("n_medidas")
    if esp is not None and len(meds) != int(esp):
        mot.append("%d medidas aceitas de %s previstas" % (len(meds), esp))
    if not meds:
        mot.append("nenhuma medida utilizável")
    r = {
        "chave": meta.get("chave"), "lado": meta["lado"], "modelo": meta["modelo"], "texto": meta["texto"], "estado": meta["estado"], "serie": meta["serie"],
        "n_medidas_previstas": esp, "n_medidas": len(meds), "status": "SEM PROVA" if mot else "OK", "motivos": sorted(set(mot)),
        "prefill": estat([x["prefill_ms"] for x in meds]), "decode": estat([x["decode_ms_passo"] for x in meds]),
        "temp_max": dados.get("temp_max"), "observacoes": dados.get("observacoes", []),
        "desktop_g_sm_antes": num_ou_none(meta.get("desktop_g_sm_antes")), "desktop_g_sm_depois": num_ou_none(meta.get("desktop_g_sm_depois")),
    }
    if meta["lado"] == "lastro":
        r["fracao_cpu_needed"] = estat([x["fracao_cpu_needed"] for x in meds if x["fracao_cpu_needed"] is not None])
        r["fracao_hits_needed"] = estat([x["fracao_hits_needed"] for x in meds if x["fracao_hits_needed"] is not None])
        r["cont_por_medida"] = [dict(x["cont"], processo=x["processo"], rotulo=x["rotulo"], fracao_cpu_needed=x["fracao_cpu_needed"]) for x in meds]
        r["fatias_total_ms"] = {
            "prefill_ms": estat([x["prefill_ms"] / x["total_ms"] for x in meds if x["total_ms"] > 0]),
            "soma_passos_ms": estat([x["soma_passos_ms"] / x["total_ms"] for x in meds if x["total_ms"] > 0]),
            "T_sequencia_ms": estat([x["T_sequencia_ms"] / x["total_ms"] for x in meds if x["total_ms"] > 0]),
        }
        r["carga_lastro_limite_superior_ms"] = estat(dados["cargas_limite_ms"])
        r["decode_mediana_dos_passos_descritiva"] = estat([x["mediana_passos_descritiva"] for x in meds])
    else:
        r["carga_ms"] = estat(dados["cargas_ms"])
        r["fonte_decode"] = sorted({x["fonte_decode"] for x in meds})
        r["kv_fa_log"] = (dados.get("kv_fa") or "").strip() or None
    if limpa_nao_finitos(r):
        mot.append("valor derivado não finito (estouro do float) na célula")
        r["status"], r["motivos"] = "SEM PROVA", sorted(set(mot))
    r["_meds"] = meds
    return r


# ---------------------------------------------------------------------------------------------------------------------------------- sequências
def primeira_divergencia(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compara_sequencias(seq_l, seq_c, ref):
    """Compara par a par: Lastro x referência, llama.cpp x referência e Lastro x llama.cpp (item (2)). Divergência é REPORTADA, nunca ajustada."""
    refg = ref["gerados"] if ref else None
    gaps = (ref or {}).get("gaps") or []
    div = []

    def acha(nome, a, b, usa_gap):
        i = primeira_divergencia(a, b)
        if i is None:
            return
        gap = gaps[i] if usa_gap and i < len(gaps) else None
        div.append({"par": nome, "passo": i, "id_a": a[i] if i < len(a) else None, "id_b": b[i] if i < len(b) else None, "gap_referencia": gap,
                    "empate": (gap is not None and gap < EMPATE)})

    for s in sorted({tuple(x) for x in seq_l}):
        if refg is not None:
            acha("Lastro x referência", list(s), refg, True)
    for s in sorted({tuple(x) for x in seq_c}):
        if refg is not None:
            acha("llama.cpp x referência", list(s), refg, True)
    for sl in sorted({tuple(x) for x in seq_l}):
        for sc in sorted({tuple(x) for x in seq_c}):
            acha("Lastro x llama.cpp", list(sl), list(sc), False)
    # Entre as repetições do mesmo lado também (o guloso devia dar a mesma sequência).
    for nome, seqs in (("Lastro entre repetições", seq_l), ("llama.cpp entre repetições", seq_c)):
        u = sorted({tuple(x) for x in seqs})
        for x in u[1:]:
            acha(nome, list(u[0]), list(x), False)
    mesma = not div and refg is not None
    # item (2): a velocidade continua válida se a divergência começa em passo de empate (gap < 1e-3) ou se os dois lados têm o mesmo nº de tokens (sempre 64 aqui,
    # contagens diferentes já são SEM PROVA); a marca "sequências diferentes" vale para qualquer divergência.
    return {"mesma_sequencia": mesma, "sequencias_diferentes": bool(div), "divergencias": div, "referencia_disponivel": refg is not None}


# ---------------------------------------------------------------------------------------------------------------------------------- razões
def razao(l, c):
    if l is None or c is None or not c["mediana"]:
        return None
    try:
        x = l["mediana"] / c["mediana"]
    except (ArithmeticError, TypeError):
        return None
    return x if math.isfinite(x) else None


def fatias_e_gargalo(cl):
    """E7: fatias do Lastro sobre total_ms. Só T_sequencia não tem equivalente de custo no llama.cpp (prompt_ms e predicted_ms cobrem prefill e decode)."""
    equiv = {"prefill_ms": "prompt_ms", "soma_passos_ms": "predicted_ms", "T_sequencia_ms": None}
    out = {}
    for k, e in cl["fatias_total_ms"].items():
        med = e["mediana"] if e else None
        out[k] = {"fatia_mediana": med, "equivalente_llama_cpp": equiv[k],
                  "candidata_a_gargalo": bool(med is not None and med > FATIA and equiv[k] is None)}
    return out


def monta_razao(cl, cc, refs, serie):
    m, t, est = cl["modelo"], cl["texto"], cl["estado"]
    r = {"modelo": m, "texto": t, "estado": est, "serie": serie, "rotulo": rotulo_razao(serie)}
    if cl["status"] == "NÃO RODOU" or cc["status"] == "NÃO RODOU":
        r["status"] = "NÃO RODOU"
        r["motivos"] = ["célula do %s não rodou" % ("Lastro" if cl["status"] == "NÃO RODOU" else "llama.cpp")]
        return r
    if cl["status"] != "OK" or cc["status"] != "OK":
        r["status"] = "SEM PROVA"
        r["motivos"] = (["Lastro: " + x for x in cl["motivos"]] if cl["status"] != "OK" else []) + (["llama.cpp: " + x for x in cc["motivos"]] if cc["status"] != "OK" else [])
        return r
    if cl["n_medidas"] != cc["n_medidas"]:
        r["status"] = "SEM PROVA"
        r["motivos"] = ["repetições diferentes (Lastro %d, llama.cpp %d): razão só com as mesmas repetições (E8)" % (cl["n_medidas"], cc["n_medidas"])]
        return r
    seq = compara_sequencias([x["gerados"] for x in cl["_meds"]], [x["gerados"] for x in cc["_meds"]], refs.ref(m, t))
    # texto: informativo (o medidas-<nome>.json do Lastro não traz o texto detokenizado).
    ref = refs.ref(m, t)
    txt = None
    if ref:
        conts = {x.get("content") for x in cc["_meds"]}
        txt = {"llama_cpp_igual_a_referencia": conts == {ref.get("texto_gerado")}, "lastro": "não disponível no medidas-<nome>.json (informativo)"}
    r["status"] = "OK"
    r["motivos"] = []
    r["sequencia"] = seq
    r["texto_informativo"] = txt
    marca = "sequências diferentes" if seq["sequencias_diferentes"] else ("mesma sequência" if seq["mesma_sequencia"] else "sem referência para conferir")
    r["marca_sequencia"] = marca
    for nome in ("prefill", "decode"):
        rz = razao(cl[nome], cc[nome])
        if rz is None or not rz > 0:
            r["status"] = "SEM PROVA"
            r["motivos"] = ["razão de %s não calculável ou não finita: não publicada" % nome]
            for k in ("sequencia", "texto_informativo", "marca_sequencia", "prefill", "decode"):
                r.pop(k, None)
            return r
        r[nome] = {"razao_lastro_sobre_llama_cpp": rz, "lastro": cl[nome], "llama_cpp": cc[nome],
                   "ruidosa": bool(cl[nome]["ruidosa"] or cc[nome]["ruidosa"]),
                   "marca": ("sequências diferentes" if (nome == "decode" and seq["sequencias_diferentes"]) else None)}
    r["fracao_cpu_needed"] = cl["fracao_cpu_needed"]
    r["fracao_hits_needed"] = cl["fracao_hits_needed"]
    p, d = r["prefill"]["razao_lastro_sobre_llama_cpp"], r["decode"]["razao_lastro_sobre_llama_cpp"]
    if p and d:
        alta, baixa = max(p, d), min(p, d)
        r["razoes_diferem_mais_de_1_5x"] = (alta / baixa) > DIFERE_RAZOES
    else:
        r["razoes_diferem_mais_de_1_5x"] = None
    if r["razoes_diferem_mais_de_1_5x"]:
        r["gargalo"] = {"hipotese_nao_conclusao": True, "fatias_do_lastro": fatias_e_gargalo(cl), "fracao_cpu_needed": cl["fracao_cpu_needed"],
                        "fracao_hits_needed": cl["fracao_hits_needed"], "fase_de_menor_desempenho_relativo": "prefill" if p > d else "decode"}
    return r


def nao_rodou(meta):
    return {"chave": meta["chave"], "lado": meta["lado"], "modelo": meta["modelo"], "texto": meta["texto"], "estado": meta["estado"], "serie": meta["serie"],
            "status": "NÃO RODOU", "motivos": ["a janela não rodou esta célula (ordem de corte E8 ou janela interrompida)"], "n_medidas": 0, "prefill": None, "decode": None}


def chave(lado, m, serie, t, est):
    return "%s-%s-%s-%s-%s" % (lado, m, serie, t, est)


def processa(saida, ids_arq, refs_dir, textos_arq):
    refs = Refs(ids_arq, refs_dir, textos_arq)
    integ = integridade(saida)
    amb = le_chave_valor(os.path.join(saida, "ambiente.txt"))
    mem = amb.get("memoria_mts")
    obs_amb = []
    if mem is None:
        obs_amb.append("memória do host (MT/s) não conferida pelo roteiro: declarar à parte")
    else:
        try:
            if int(float(mem)) != MEM_NOMINAL:
                integ.append("memória do host em %s MT/s, fora de %d MT/s" % (mem, MEM_NOMINAL))
        except ValueError:
            obs_amb.append("memoria_mts ilegível: %s" % mem)
    orc = le_json(os.path.join(saida, "orcamento.json"))
    orc = orc if isinstance(orc, dict) else {}
    # emenda 6: cada janela roda UM modelo; orcamento.json declara "modelos". Sem a declaração (janela antiga ou de teste), os dois.
    decl = orc.get("modelos")
    modelos = tuple(m for m in MODELOS if m in decl) if isinstance(decl, list) and decl else MODELOS
    celulas = {}
    cdir = os.path.join(saida, "celulas")
    achadas = {}
    if os.path.isdir(cdir):
        for nome in sorted(os.listdir(cdir)):
            meta = le_json(os.path.join(cdir, nome, "celula.json"))
            if isinstance(meta, dict) and {"lado", "modelo", "texto", "estado", "serie"} <= set(meta):
                meta["chave"] = chave(meta["lado"], meta["modelo"], meta["serie"], meta["texto"], meta["estado"])
                achadas[meta["chave"]] = (os.path.join(cdir, nome), meta)
    for m in modelos:
        for t in TEXTOS:
            for est in ESTADOS:
                previstas = [("lastro", "L")] + [("llamacpp", s) for s in SERIES]
                for lado, serie in previstas:
                    k = chave(lado, m, serie, t, est)
                    meta0 = {"chave": k, "lado": lado, "modelo": m, "texto": t, "estado": est, "serie": serie}
                    if k not in achadas:
                        celulas[k] = nao_rodou(meta0)
                        continue
                    d, meta = achadas[k]
                    dados, mot = (le_celula_lastro if lado == "lastro" else le_celula_llama)(d, meta, refs)
                    celulas[k] = resume_celula(meta, dados, mot)
    razoes = []
    for m in modelos:
        for t in TEXTOS:
            for est in ESTADOS:
                cl = celulas[chave("lastro", m, "L", t, est)]
                for s in SERIES:
                    razoes.append(monta_razao(cl, celulas[chave("llamacpp", m, s, t, est)], refs, s))
    # integridade quebrada = SEM PROVA em tudo (item (4)).
    if integ:
        for r in razoes:
            if r["status"] == "OK":
                r.update({"status": "SEM PROVA", "motivos": integ[:], "prefill": None, "decode": None})
        for c in celulas.values():
            if c["status"] == "OK":
                c["status"] = "SEM PROVA"
                c["motivos"] = sorted(set(c["motivos"] + integ))
    ordem = (le_txt(os.path.join(saida, "ordem.txt")) or "").splitlines()
    for c in celulas.values():
        c.pop("_meds", None)
    return {
        "tipo": "cmp-resultado", "criterio": "criterio-comparacao-llamacpp-20261108.md (emendas 1 a 6)", "descritivo": True, "modelos_da_janela": list(modelos),
        "aviso_rotulo": "Toda razão é 'Lastro 10% experts GPU contra llama.cpp <série>'. Nenhuma isola o motor; a fração cpu/needed vem ao lado. Sem meta, sem PASSA/REPROVA.",
        "integridade_problemas": integ, "observacoes_ambiente": obs_amb, "orcamento": orc, "ordem": ordem,
        "celulas": celulas, "razoes": razoes,
    }


# ---------------------------------------------------------------------------------------------------------------------------------- relatório
def f(x, n=2):
    return "-" if x is None else ("%.*f" % (n, x))


def pct(x):
    return "-" if x is None else "%.1f%%" % (100 * x)


def faixa(e, n=2):
    return "-" if not e else "%s (%s a %s)%s" % (f(e["mediana"], n), f(e["min"], n), f(e["max"], n), " ruidosa" if e["ruidosa"] else "")


def relatorio(res):
    L = []
    L.append("# Comparação Lastro x llama.cpp (opção d): resultado descritivo")
    L.append("")
    L.append("Critério: %s. Sem meta e sem PASSA/REPROVA de velocidade. **Toda razão é Lastro com 10%% dos experts na GPU contra a série indicada do llama.cpp; "
             "nenhuma razão isola o motor.** Razão = mediana Lastro / mediana llama.cpp (menor que 1 = Lastro mais rápido), por série, nunca misturadas." % res["criterio"])
    L.append("")
    L.append("Declarações: KV f16 no llama.cpp contra f32 no Lastro (S1, S2); a ponte de correção é a igualdade dos ids gerados. Fronteira do prefill: `prompt_ms` do llama.cpp "
             "termina depois de amostrar o 1º token, `prefill_ms` do Lastro antes do argmax (sem ajuste). Decode: média por passo (63 passos nos dois lados). "
             "Quente e frio com o cache de página do GGUF/safetensors quente (o 1º contato com o disco não entra). Carga do Lastro é LIMITE SUPERIOR (inclui a cauda de encerramento).")
    L.append("")
    if res["integridade_problemas"]:
        L.append("## Integridade QUEBRADA: tudo SEM PROVA")
        L += ["- " + x for x in res["integridade_problemas"]]
        L.append("")
    for o in res["observacoes_ambiente"]:
        L.append("Observação: " + o)
    if res.get("modelos_da_janela") and tuple(res["modelos_da_janela"]) != MODELOS:
        L.append("Janela do(s) modelo(s): %s (emenda 6: uma janela por modelo; o outro modelo é outra janela e não aparece aqui)." % ", ".join(res["modelos_da_janela"]))
    orc = res.get("orcamento") or {}
    if orc:
        L.append("")
        L.append("## Orçamento (E8) e cortes")
        L.append("Cortes aplicados: %s" % (", ".join(orc.get("cortes") or []) or "nenhum"))
        L.append("Repetições finais: %s" % json.dumps(orc.get("reps_finais", orc.get("reps", {})), ensure_ascii=False))
    if res.get("ordem"):
        L.append("")
        L.append("## Ordem de execução gravada")
        L += ["    " + x for x in res["ordem"]]
    for s in SERIES:
        L.append("")
        L.append("## %s" % rotulo_razao(s))
        L.append("")
        L.append("| modelo | texto | estado | n | razão prefill | razão decode | marca | cpu/needed (Lastro) | Lastro prefill ms | llama.cpp prefill ms | Lastro decode ms/passo | llama.cpp decode ms/passo |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in [x for x in res["razoes"] if x["serie"] == s]:
            if r["status"] != "OK":
                L.append("| %s | %s | %s | - | %s | %s | %s | - | - | - | - | - |" % (r["modelo"], r["texto"], r["estado"], r["status"], r["status"], "; ".join(r["motivos"])[:200]))
                continue
            p, d = r["prefill"], r["decode"]
            marca = [r["marca_sequencia"]] + (["ruidosa"] if (p["ruidosa"] or d["ruidosa"]) else [])
            L.append("| %s | %s | %s | %d | %s%s | %s%s | %s | %s | %s | %s | %s | %s |" % (
                r["modelo"], r["texto"], r["estado"], p["lastro"]["n"], f(p["razao_lastro_sobre_llama_cpp"], 3), " (ruidosa)" if p["ruidosa"] else "",
                f(d["razao_lastro_sobre_llama_cpp"], 3), " (sequências diferentes)" if d["marca"] else (" (ruidosa)" if d["ruidosa"] else ""), "; ".join(marca),
                faixa(r["fracao_cpu_needed"], 3), faixa(p["lastro"]), faixa(p["llama_cpp"]), faixa(d["lastro"], 3), faixa(d["llama_cpp"], 3)))
        if s == "S2":
            L.append("")
            L.append("S2 é aproximação declarada (camadas inteiras no llama.cpp, ~90% dos experts de cada camada no Lastro); op offload ligado: o prefill da S2 não deve ser lido como 'experts calculados na CPU'.")
    gargalos = [r for r in res["razoes"] if r.get("gargalo")]
    L.append("")
    L.append("## Leitura da diferença (E7, hipótese e nunca conclusão)")
    if not gargalos:
        L.append("Nenhuma célula com razões de prefill e decode diferindo em mais de 1,5x.")
    for r in gargalos:
        g = r["gargalo"]
        fat = "; ".join("%s %s%s" % (k, pct(v["fatia_mediana"]), " CANDIDATA A GARGALO" if v["candidata_a_gargalo"] else "") for k, v in g["fatias_do_lastro"].items())
        L.append("- %s %s %s, %s: fase de menor desempenho relativo = %s; fatias do Lastro sobre total_ms: %s; cpu/needed %s; hits/needed %s" % (
            r["modelo"], r["texto"], r["estado"], r["serie"], g["fase_de_menor_desempenho_relativo"], fat, faixa(g["fracao_cpu_needed"], 3), faixa(g["fracao_hits_needed"], 3)))
    L.append("")
    L.append("## Carga do modelo (descritiva, sem razão) e tokens/s de decode")
    L.append("")
    L.append("| célula | carga ms (mediana, min a max) | decode tokens/s (mediana) |")
    L.append("|---|---|---|")
    for k, c in res["celulas"].items():
        if c["status"] == "NÃO RODOU":
            continue
        carga = c.get("carga_lastro_limite_superior_ms") if c["lado"] == "lastro" else c.get("carga_ms")
        rot = " (LIMITE SUPERIOR)" if c["lado"] == "lastro" else (" (frio sem aquecimento)" if c["estado"] == "frio" else " (inclui aquecimento padrão)")
        tps = (1000.0 / c["decode"]["mediana"]) if c.get("decode") and c["decode"]["mediana"] else None
        L.append("| %s | %s%s | %s |" % (k, faixa(carga, 0), rot, f(tps, 1)))
    sp = [(k, c) for k, c in res["celulas"].items() if c["status"] == "SEM PROVA"]
    nr = [k for k, c in res["celulas"].items() if c["status"] == "NÃO RODOU"]
    L.append("")
    L.append("## Condição do desktop (emenda 8)")
    L.append("O desktop (compositor Hyprland, etc.) divide a 4090 nos DOIS lados em todas as células: é condição constante declarada, fora do SEM PROVA. "
             "Soma de sm% dos PIDs de tipo G, medida por `nvidia-smi pmon -c 3 -s um` antes e depois de cada célula (maior valor das 3 voltas; `-` = sem leitura). "
             "Só decide SEM PROVA: PID fora da árvore medida de tipo C ou C+G com sm > 5%, ou soma G > 60% (palpites declarados antes dos dados).")
    L.append("")
    L.append("| célula | desktop sm% antes | desktop sm% depois |")
    L.append("|---|---|---|")
    for k, c in res["celulas"].items():
        if c.get("status") == "NÃO RODOU":
            continue
        L.append("| %s | %s | %s |" % (k, f(c.get("desktop_g_sm_antes"), 0), f(c.get("desktop_g_sm_depois"), 0)))
    L.append("")
    L.append("## SEM PROVA (sem razão publicada)")
    if not sp:
        L.append("Nenhuma célula.")
    for k, c in sp:
        L.append("- %s: %s" % (k, "; ".join(c["motivos"])))
    obs = [(k, c) for k, c in res["celulas"].items() if c.get("observacoes")]
    if obs:
        L.append("")
        L.append("## Observações informativas (não são prova nem falha)")
        for k, c in obs:
            L.append("- %s: %s" % (k, "; ".join(c["observacoes"])))
    L.append("")
    L.append("## NÃO RODOU (nunca vira razão, PASSA nem estimativa)")
    L.append(", ".join(nr) if nr else "Nenhuma célula.")
    L.append("")
    return "\n".join(L) + "\n"


def main(argv=None):
    aqui = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="extrator da comparação Lastro x llama.cpp")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--ids", default=os.path.join(aqui, "cmp-ids-prompt.json"))
    ap.add_argument("--refs", default=os.path.join(aqui, "5c-referencia"))
    ap.add_argument("--textos", default=os.path.join(aqui, "5c-textos", "textos.json"))
    a = ap.parse_args(argv)
    if not os.path.isdir(a.saida):
        print("pasta de saída inexistente: %s" % a.saida, file=sys.stderr)
        return 2
    res = processa(a.saida, a.ids, a.refs, a.textos)
    limpa_nao_finitos(res)     # rede de segurança: o JSON final é estrito (allow_nan=False)
    with open(os.path.join(a.saida, "cmp-resultado.json"), "w", encoding="utf-8") as fo:
        json.dump(res, fo, ensure_ascii=False, indent=1, allow_nan=False)
    with open(os.path.join(a.saida, "cmp-resultado.md"), "w", encoding="utf-8") as fo:
        fo.write(relatorio(res))
    n = {"OK": 0, "SEM PROVA": 0, "NÃO RODOU": 0}
    for c in res["celulas"].values():
        n[c["status"]] += 1
    print("cmp-extrai: células OK %d, SEM PROVA %d, NÃO RODOU %d; gravados cmp-resultado.json e cmp-resultado.md" % (n["OK"], n["SEM PROVA"], n["NÃO RODOU"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
