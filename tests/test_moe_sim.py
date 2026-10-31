"""Invariantes do simulador de colocação de experts (tools/moe_sim.py, M0.5).

Precisam de numpy: rodam com tools/.venv-ref/bin/python -m pytest; no Python do sistema, sem numpy, são puladas.
"""
import os
import sys

import pytest

np = pytest.importorskip("numpy")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import moe_sim as m  # noqa: E402

P = m.PARAMS["3b"]
L, E, K = 4, 16, 4
POLICIES = ["lru_sync", "belady", "lru_cpu", "lfu_cpu", "custo", "oraculo"]


def trace(T, seed, hot=None):
    rng = np.random.default_rng(seed)
    a = np.empty((L, T, K), dtype=np.int32)
    for l in range(L):
        for t in range(T):
            pool = hot if hot is not None and rng.random() < 0.8 else np.arange(E)
            a[l, t] = rng.choice(pool, K, replace=False)
    return a


def run(pol, cap, train, test, **kw):
    if pol in ("belady", "oraculo"):
        kw["next_use"] = m.next_use_fn(train + test, L)
    s = m.Sim(pol, cap, L, E, K, P, **kw)
    s.run(train, measure=False)
    s.run(test, measure=True)
    return s


@pytest.mark.parametrize("pol", POLICIES)
def test_orcamento_nunca_e_excedido(pol):
    train, test = [trace(200, 1)], [trace(200, 2)]
    cap = 12
    s = m.Sim(pol, cap, L, E, K, P, next_use=m.next_use_fn(train + test, L) if pol in ("belady", "oraculo") else None)
    for a in train + test:
        for t in range(a.shape[1]):
            for l in range(L):
                s.layer(l, a[l, t])
                assert len(s.resident) + len(s.inflight) <= cap


@pytest.mark.parametrize("pol", POLICIES)
def test_com_todos_os_experts_cabendo_nao_ha_espera_depois_de_aquecer(pol):
    train, test = [trace(300, 3)], [trace(100, 4)]
    s = run(pol, L * E, train, test)
    assert s.stats["hits"] == s.stats["needed"]
    assert s.stats["wait_ms"] == pytest.approx(0.0, abs=1e-9)


def test_calibracao_camadas_bate_com_a_formula():
    test = [trace(50, 5)]
    tudo = m.Sim("camadas", 0, L, E, K, P, resident_layers=set(range(L)))
    tudo.run(test, measure=True)
    assert tudo.stats["ms"] == pytest.approx(50 * L * (P["g"] + K * P["h"]))
    nada = m.Sim("camadas", 0, L, E, K, P, resident_layers=set())
    nada.run(test, measure=True)
    # Sem cópia nenhuma, a CPU custa c_cpu_quiet e não há o_layer (o lru_cpu com backlog 0 do m2a).
    assert nada.stats["ms"] == pytest.approx(50 * L * (P["g"] + K * P["c_cpu_quiet"]))


def test_belady_acerta_pelo_menos_tanto_quanto_lru_no_modo_sincrono():
    hot = np.arange(6)
    train, test = [trace(300, 6, hot)], [trace(300, 7, hot)]
    for cap in (8, 16, 32):
        lru, bel = run("lru_sync", cap, train, test), run("belady", cap, train, test)
        assert bel.stats["hits"] >= lru.stats["hits"]


def test_custo_nunca_espera_mais_que_so_cpu_sem_cache():
    # Com orçamento mínimo, a política por custo só usa a cópia quando ela termina antes da CPU.
    train, test = [trace(100, 8)], [trace(100, 9)]
    custo = run("custo", K, train, test)
    nada = m.Sim("camadas", 0, L, E, K, P, resident_layers=set())
    nada.run(test, measure=True)
    assert custo.stats["wait_ms"] <= nada.stats["wait_ms"] + 1e-9



class CheckedLru(m.Sim):
    """Sim que confere a cada despejo a propriedade do LRU: sai o expert de uso mais antigo entre os que podem sair. O empate
    entre experts usados no mesmo passo fica livre (o runtime desempata pelo slot, o simulador pela versão)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.used = {}
        self.evictions = 0

    def layer(self, l, experts):
        for e in experts:
            self.used[(l, int(e))] = self.step
        super().layer(l, experts)

    def victim(self, pinned):
        found = super().victim(pinned)
        if found is not None:
            oldest = min(self.used[k] for k in self.resident if k not in pinned)
            assert self.used[found[1]] == oldest, f"passo {self.step}: despejou {found[1]} usado em {self.used[found[1]]}, havia um de {oldest}"
            self.evictions += 1
        return found


@pytest.mark.parametrize("pol", ["lru_sync", "lru_cpu"])
def test_despejo_do_lru_leva_o_expert_de_uso_mais_antigo(pol):
    # Regressão de 29/10 (revisão 26): o expert copiado só entrava no heap ao assentar, na camada seguinte, com a recência
    # dela; empatava com os acertos dessa camada e o desempate por versão despejava o acerto, que era mais recente. No 1B a
    # 25%, o simulador dava 44,6% de acerto no lru_sync contra 48,1% do runtime (rust/moe-rt/src/cache.rs).
    test = [trace(300, 11, np.arange(10)), trace(200, 12)]
    for cap in (6, 12, 24, 40):
        s = CheckedLru(pol, cap, L, E, K, P)
        s.run(test, measure=True)
        assert s.evictions > 0, f"cap {cap}: nenhum despejo, o teste não exercitou nada"


@pytest.mark.parametrize("protocol, cpu, sync", [("runtime", 4, 0), ("m05", 3, 1)])
def test_expert_carregando_vai_a_cpu_no_protocolo_do_runtime(protocol, cpu, sync):
    # O ExpertCache devolve Lookup::Loading para a chave em voo e o lru_cpu a calcula na CPU; o M0.5 esperava a cópia se ela
    # terminasse antes da CPU. Aqui a cópia termina 1 µs depois do início da camada.
    s = m.Sim("lru_cpu", 16, 1, E, K, P, protocol=protocol)
    s.inflight[(0, 0)] = s.now + P["g"] + 0.001
    s.layer(0, np.arange(K))
    assert s.stats["cpu"] == cpu
    assert s.stats["layers_sync"] == sync


@pytest.mark.parametrize("protocol, admitidas", [("runtime", 0), ("m05", K)])
def test_backlog_do_runtime_conta_copias_em_voo(protocol, admitidas):
    # Como o ExpertCache, o runtime recusa a admissão com BACKLOG cópias não assentadas, mesmo com a fila do PCIe vazia em
    # tempo; o M0.5 olhava só o tempo de fila.
    s = m.Sim("lru_cpu", 64, 1, E, K, P, protocol=protocol)
    for e in range(m.BACKLOG):
        s.inflight[(0, E - 1 - e)] = 1e9
    s.layer(0, np.arange(K))
    assert s.stats["copies_bg"] == admitidas
    assert s.stats["dropped"] == K - admitidas


@pytest.mark.parametrize("backlog", [0, m.BACKLOG])
def test_camada_com_trafego_de_copia_paga_a_cpu_disputada_e_o_layer(backlog):
    # Medido no m2a: sem cópia (backlog 0) a CPU custa c_cpu_quiet por expert e a camada não tem sobra; com cópia em
    # segundo plano, c_cpu e o_layer.
    s = m.Sim("lru_cpu", 64, 1, E, K, P, backlog=backlog)
    s.layer(0, np.arange(K))
    assert s.stats["copies_bg"] == min(K, backlog)
    esperado = P["g"] + K * P["c_cpu_quiet"] if backlog == 0 else P["g"] + K * P["c_cpu"] + P["o_layer"]
    assert s.now == pytest.approx(esperado)


def test_custo_copia_com_a_cpu_ocupada_na_banda_disputada():
    # c_cpu entre c_copy + h e 2·c_copy + h: a primeira falta é copiada com a CPU ociosa, a segunda vai à CPU e a terceira
    # é copiada com a CPU ocupada. Backlog 0 tira a admissão em segundo plano da fila do PCIe.
    p = dict(P, c_cpu=0.35, c_cpu_quiet=0.35)
    c_copy = p["expert_mb"] / m.PCIE_GBS
    c_copy_cpu = p["expert_mb"] / p["pcie_gbs_cpu"]
    assert c_copy + p["h"] < p["c_cpu"] < 2 * c_copy + p["h"] and c_copy + c_copy_cpu + p["h"] < 2 * p["c_cpu"]
    s = m.Sim("custo", 64, 1, E, K, p, backlog=0)
    s.layer(0, np.arange(3))
    assert s.stats["copies_sync"] == 2 and s.stats["cpu"] == 1
    assert s.inflight[(0, 0)] == pytest.approx(p["g"] + c_copy)
    assert s.inflight[(0, 2)] == pytest.approx(p["g"] + c_copy + c_copy_cpu)


def test_copia_com_a_cpu_ocupada_usa_a_banda_disputada():
    c_copy = P["expert_mb"] / m.PCIE_GBS
    c_copy_cpu = P["expert_mb"] / P["pcie_gbs_cpu"]
    assert c_copy_cpu > c_copy
    bg = m.Sim("lru_cpu", 16, 1, E, K, P)
    bg.layer(0, np.arange(1))
    assert bg.inflight[(0, 0)] == pytest.approx(P["g"] + c_copy_cpu)
    sync = m.Sim("lru_sync", 16, 1, E, K, P)
    sync.layer(0, np.arange(1))
    assert sync.stats["copies_sync"] == 1
    assert sync.now == pytest.approx(P["g"] + c_copy + P["h"] + P["o_layer"])


RAIZ = os.path.join(os.path.dirname(__file__), "..")
VALIDACAO = [(mod, os.path.join(RAIZ, "qual-runs", "TRACOS-20261029", f"granite-3.1-{mod}-a{a}m-instruct"),
              os.path.join(RAIZ, "qual-runs", "M2F-FUSOES-20261029", f"m2a-{mod}-final{suf}.txt"))
             for mod, a in (("1b", 400), ("3b", 800)) for suf in ("-1", "-backlog0-1", "-backlog0-2")]


@pytest.mark.parametrize("mod, tracos, m2a", VALIDACAO)
def test_simulador_reproduz_o_m2a(mod, tracos, m2a, capsys):
    # Recalibração de 30/10: o_layer e pcie_gbs_cpu foram ajustados só no 1B (lru_sync e lru_cpu a 10%); o 3B é a validação
    # deles. c_cpu e c_cpu_quiet são medidos em cada modelo (lru_cpu com backlog 4 e 0).
    if not (os.path.isdir(tracos) and os.path.isfile(m2a)):
        pytest.skip("traços ou saída do m2a ausentes")
    test = m.load(tracos, "teste")
    L_, _, k = test[0].shape
    E_ = int(max(a.max() for a in m.load(tracos, "treino") + test)) + 1
    assert m.validate(m2a, test, L_, E_, k, m.PARAMS[mod], "runtime"), capsys.readouterr().out


LINHA_1B = "lru_cpu      25%   192 |    9.54    9.46   104.8 |   0.0% 100.0%   0.00   0.00 |     0.81     7.57 | 82824/82824 |  | x\n"
CAB = "M2a: granite-3.1-1b-a400m-instruct; conversas [4, 5, 6, 7] (3451 tokens); 768 experts de 3.15 MB; CPU com 8 threads; backlog {b}"


@pytest.mark.parametrize("b", [0, 1, 2, 4])
@pytest.mark.parametrize("sufixo", ["", "; admissão evento"])
def test_backlog_sai_do_cabecalho_antigo_e_do_atual(tmp_path, b, sufixo):
    # Regressão da rodada 36 (rev-1): o cabeçalho atual termina em "; admissão evento" e o backlog caía para 4 em silêncio.
    f = tmp_path / "m2a.txt"
    f.write_text(CAB.format(b=b) + sufixo + "\n\n" + LINHA_1B)
    rows, backlog = m.read_m2a(str(f))
    assert backlog == b
    assert list(rows) == [("lru_cpu", 192)]


def test_le_a_linha_do_lfu_cpu(tmp_path):
    f = tmp_path / "m2a.txt"
    f.write_text(CAB.format(b=4) + "; admissão evento\n\n" + LINHA_1B.replace("lru_cpu", "lfu_cpu"))
    assert list(m.read_m2a(str(f))[0]) == [("lfu_cpu", 192)]


@pytest.mark.parametrize("texto", [CAB.format(b=4) + "; admissão barreira\n" + LINHA_1B,
                                   CAB.format(b=4) + "; admissão evento; aquecimento [0, 1, 2, 3]\n" + LINHA_1B,
                                   "M2a: granite; 768 experts de 3.15 MB\n" + LINHA_1B, LINHA_1B])
def test_cabecalho_que_o_simulador_nao_reproduz_e_erro(tmp_path, texto):
    f = tmp_path / "m2a.txt"
    f.write_text(texto)
    with pytest.raises(ValueError):
        m.read_m2a(str(f))


@pytest.mark.parametrize("corpo", ["", "resident     100%   768 |    3.01    3.01   332.2 | 100.0%   0.0%   0.00   0.00 |\n"])
def test_validacao_sem_nenhuma_linha_reprova(tmp_path, corpo):
    # Regressão da rodada 36 (rev-1): sem linha comparada, a validação imprimia sucesso e saía com 0.
    f = tmp_path / "m2a.txt"
    f.write_text(CAB.format(b=4) + "\n" + corpo)
    assert m.validate(str(f), [trace(10, 1)], L, E, K, P, "runtime") is False


# -- busca antecipada (A0 do plano, 31/10) ------------------------------------------------------------------------------

@pytest.mark.parametrize("pol, hits, cpu, copies, dropped, wait", [
    ("lru_cpu", 777, 4023, 1358, 2663, 298.3241999998917),
    ("lfu_cpu", 883, 3917, 275, 3642, 233.11839999998975),
])
def test_sem_busca_antecipada_o_simulador_nao_muda(pol, hits, cpu, copies, dropped, wait):
    # Valores do moe_sim.py de fddc2cc, antes da busca antecipada: sem `prefetch`, nada pode mudar.
    st = run(pol, 12, [trace(200, 22)], [trace(300, 21)]).stats
    assert (st["hits"], st["cpu"], st["copies_bg"], st["dropped"]) == (hits, cpu, copies, dropped)
    assert st["wait_ms"] == wait


@pytest.mark.parametrize("pred", ["oraculo", "anterior", "frequente"])
def test_orcamento_nunca_e_excedido_com_busca_antecipada(pred):
    s = m.Sim("lfu_cpu", 12, L, E, K, P, prefetch={"pred": pred, "d": 2, "admissao": "livre", "espera": True})
    s.run([trace(200, 31), trace(200, 32)], measure=False)
    assert s.stats["copies_pf"] > 0
    orig = s.layer

    def checked(l, experts, pred=None):
        orig(l, experts, pred)
        assert len(s.resident) + len(s.inflight) <= s.cap

    s.layer = checked
    s.run([trace(100, 33)], measure=True)


def test_o_que_o_oraculo_traz_e_usado():
    # O oráculo traz exatamente os experts do alvo, reservados até ele: com a cópia chegando antes do alvo (camada longa) e
    # orçamento folgado, quase tudo o que ele traz é usado. Não vale "acerta mais que sem busca" em geral: com o g real a
    # cópia chega tarde, e com orçamento apertado a admissão livre despeja experts quentes (o tráfego que o A0 mede).
    train, test = [trace(300, 41, np.arange(8))], [trace(300, 42, np.arange(8))]
    s = m.Sim("lfu_cpu", 48, L, E, K, dict(P, g=1.0), prefetch={"pred": "oraculo", "d": 3, "admissao": "livre", "espera": True})
    s.run(train, measure=False)
    s.run(test, measure=True)
    pf = s.stats
    assert pf["copies_pf"] > 0
    assert pf["pf_useful"] > 0.8 * pf["copies_pf"]


def test_preditor_anterior_so_le_o_passado():
    # Alvo d camadas à frente; o preditor "anterior" devolve o roteamento do token anterior na camada do alvo, que já
    # aconteceu: no mesmo token (camada do alvo > l) ou no token atual (o alvo passou para o próximo token).
    a = trace(6, 51)
    s = m.Sim("lfu_cpu", 12, L, E, K, P, prefetch={"pred": "anterior", "d": 1, "admissao": "livre", "espera": False})
    for d in range(1, L):
        s.prefetch["d"] = d
        for t in range(1, 5):
            for l in range(L):
                keys, alvo = s.predict(a, t, l)
                t2, l2 = divmod(t * L + l + d, L)
                assert alvo == s.step + d
                assert keys == [(l2, int(e)) for e in a[l2, t2 - 1]]
                assert (t2 - 1, l2) < (t, l) or (t2 - 1 == t and l2 < l)


def test_expert_reservado_nao_e_despejado_antes_do_uso():
    # Um expert trazido para o alvo daqui a 2 camadas não pode sair na admissão da camada seguinte, mesmo sendo o de
    # menor frequência (nunca foi usado).
    s = m.Sim("lfu_cpu", K + 1, 1, E, K, P, prefetch={"pred": "oraculo", "d": 2, "admissao": "livre", "espera": False})
    s.layer(0, np.arange(K))  # K experts quentes, residentes depois do aquecimento abaixo
    for _ in range(50):
        s.now += 10.0
        s.layer(0, np.arange(K))
    s.now += 10.0
    s.layer(0, np.arange(K), ([(0, E - 1)], s.step + 2))
    assert (0, E - 1) in s.inflight or (0, E - 1) in s.resident
    s.now += 10.0
    s.layer(0, np.arange(K, 2 * K))  # faltas: a admissão do lfu_cpu procura vítima
    assert (0, E - 1) in s.inflight or (0, E - 1) in s.resident


@pytest.mark.parametrize("espera, cpu, sync", [(False, K, 0), (True, K - 1, 1)])
def test_espera_do_evento_na_copia_em_voo(espera, cpu, sync):
    # No protocolo sem parada por camada, a stream de cálculo pode esperar o evento de uma cópia em voo que termina antes da
    # CPU; sem a espera, a chave carregando vai à CPU, como no ExpertCache de hoje.
    s = m.Sim("lfu_cpu", 16, 1, E, K, P, prefetch={"pred": "oraculo", "d": 1, "admissao": "livre", "espera": espera})
    s.inflight[(0, 0)] = s.now + P["g"] + 0.001
    s.layer(0, np.arange(K))
    assert s.stats["cpu"] == cpu
    assert s.stats["layers_sync"] == sync


# -- protocolo token (A0c do plano, 31/10): residência publicada no começo do token ---------------------------------------

def _token_vazio(s, ultima):
    """Um token em que só a última camada pede experts."""
    for l in range(L - 1):
        s.layer(l, [])
    s.layer(L - 1, ultima)


@pytest.mark.parametrize("pol", ["lru_cpu", "lfu_cpu"])
def test_copia_terminada_no_meio_do_token_so_vale_no_seguinte(pol):
    # As K faltas da última camada do token 0 são copiadas durante o token 1 (a tabela dele já foi publicada). No runtime,
    # o settle de cada camada assenta todas antes da última camada do token 1; no protocolo token, elas já terminaram mas
    # só entram na tabela do token 2.
    p = dict(P, g=0.4)
    hits = {}
    for protocol in ("runtime", "token"):
        s = m.Sim(pol, 64, L, E, K, p, protocol=protocol)
        _token_vazio(s, np.arange(K))
        assert s.stats["copies_bg"] == K
        for t in (1, 2):
            for l in range(L - 1):
                s.layer(l, [])
            if protocol == "token" and t == 1:
                assert all(k not in s.table and s.inflight[k] <= s.now + p["g"] for k in [(L - 1, e) for e in range(K)])
            h0 = s.stats["hits"]
            s.layer(L - 1, np.arange(K))
            hits[protocol, t] = s.stats["hits"] - h0
    assert hits == {("runtime", 1): K, ("runtime", 2): K, ("token", 1): 0, ("token", 2): K}


def test_vitima_de_camada_a_frente_segue_lida_no_token_e_a_copia_espera_a_camada():
    # Orçamento de K slots, todos com experts da camada 1. No token 2, as faltas da camada 0 escolhem vítimas da camada 1:
    # no protocolo token, a vítima continua na tabela do token (a GPU a lê na camada 1) e a cópia só entra na fila do PCIe
    # quando o host vê o doorbell da camada 2 (fim da camada 1 mais o g da 2); no runtime, a vítima sai na hora e a camada
    # 1 vai à CPU.
    p = dict(P, g=1.0)
    out = {}
    for protocol in ("runtime", "token"):
        s = m.Sim("lru_cpu", K, L, E, K, p, protocol=protocol)
        for _ in range(2):
            for l in range(L):
                s.layer(l, np.arange(K) if l == 1 else [])
        assert set(s.resident) == {(1, e) for e in range(K)}
        s.layer(0, np.arange(K, 2 * K))
        if protocol == "token":
            assert s.stats["copies_deferred"] == K
            assert all(s.inflight[(0, e)] == float("inf") for e in range(K, 2 * K))
        h0 = s.stats["hits"]
        s.layer(1, np.arange(K))
        out[protocol] = s.stats["hits"] - h0
        if protocol == "token":
            fim_camada_1 = s.now
            assert all(s.inflight[(0, e)] == float("inf") for e in range(K, 2 * K))
            s.layer(2, [])
            assert all(fim_camada_1 + p["g"] < s.inflight[(0, e)] < float("inf") for e in range(K, 2 * K))
    assert out == {"runtime": 0, "token": K}


def test_vitima_da_ultima_camada_sai_no_doorbell_do_fim_do_token():
    p = dict(P, g=1.0)
    s = m.Sim("lru_cpu", K, L, E, K, p, protocol="token")
    for _ in range(2):
        for l in range(L):
            s.layer(l, np.arange(K) if l == L - 1 else [])
    s.layer(0, np.arange(K, 2 * K))
    assert s.stats["copies_deferred"] == K
    for l in range(1, L - 1):
        s.layer(l, [])
    assert all(s.inflight[(0, e)] == float("inf") for e in range(K, 2 * K))
    s.layer(L - 1, np.arange(K))
    assert not s.deferred
    assert all(s.now < s.inflight[(0, e)] < float("inf") for e in range(K, 2 * K))


@pytest.mark.parametrize("protocol, hits", [("runtime", 1), ("token", 0)])
def test_copia_que_termina_dentro_do_primeiro_g_nao_entra_na_tabela(protocol, hits):
    # A tabela do token vai para a GPU antes do g da primeira camada: uma cópia que termina no meio desse g fica para o
    # token seguinte. No runtime, o settle depois do g a aproveita.
    s = m.Sim("lru_cpu", 16, 1, E, K, dict(P, g=0.1), protocol=protocol)
    s.inflight[(0, 0)] = 0.05
    s.layer(0, np.arange(K))
    assert s.stats["hits"] == hits


def test_parada_da_barreira_entra_na_espera():
    s = m.Sim("lru_cpu", 16, 1, E, K, P, protocol="token", token_barreira=True)
    s.inflight[(0, E - 1)] = 5.0
    s.layer(0, np.arange(K))
    assert s.stats["barrier_ms"] == pytest.approx(5.0)
    assert s.stats["wait_ms"] >= 5.0


@pytest.mark.parametrize("pol", ["lru_cpu", "lfu_cpu"])
@pytest.mark.parametrize("barreira", [False, True])
def test_orcamento_e_adiadas_no_protocolo_token(pol, barreira):
    s = m.Sim(pol, 12, L, E, K, P, protocol="token", token_barreira=barreira)
    orig = s.layer

    def checked(l, experts, pred=None):
        orig(l, experts)
        assert len(s.resident) + len(s.inflight) <= s.cap
        # A adiada da camada l começa no doorbell de l + 1; a da última, no fim do token.
        assert all(lv >= l for lv, _ in s.deferred)
        assert l < L - 1 or not s.deferred
        assert set(s.resident) <= s.table

    s.layer = checked
    s.run([trace(200, 61), trace(200, 62)], measure=True)
    assert s.stats["copies_bg"] > 0 and s.stats["copies_deferred"] > 0


def test_barreira_do_token_nao_depende_do_tempo_da_copia():
    # O modo TokenBarrier do A1: o começo do token espera todas as cópias, então acertos, cópias e experts na CPU são os
    # mesmos com qualquer banda; sem a barreira, a banda muda o resultado. lru_cpu e traço sem conjunto quente, para o backlog
    # encher (no lfu_cpu quase todo descarte é recusa da frequência, e o backlog não seria exercitado).
    train, test = [trace(200, 71)], [trace(300, 72)]
    res = {}
    for barreira in (False, True):
        for gbs in (14.1, 2.0):
            s = m.Sim("lru_cpu", 12, L, E, K, dict(P, pcie_gbs_cpu=gbs), protocol="token", token_barreira=barreira)
            if barreira:
                ob = s.begin_token

                def bt(_ob=ob, _s=s):
                    _ob()
                    assert not _s.inflight  # tudo assentado no começo do token

                s.begin_token = bt
            s.run(train, measure=False)
            s.run(test, measure=True)
            st = s.stats
            res[barreira, gbs] = (st["hits"], st["cpu"], st["copies_bg"], st["dropped"])
    assert all(r[3] > 0 for r in res.values())  # o backlog encheu (houve descarte)
    assert res[True, 14.1] == res[True, 2.0]
    assert res[False, 14.1] != res[False, 2.0]


@pytest.mark.parametrize("protocol", ["runtime", "token"])
def test_admissao_do_token_comeca_depois_da_cpu_da_camada(protocol):
    # Rodada 52: no protocolo token, as admissões vêm depois da resposta da CPU (o Served). Com slots livres e K faltas, a
    # primeira cópia termina c_copy depois do fim da CPU da camada; no runtime, c_copy depois do começo da camada.
    s = m.Sim("lru_cpu", 64, 2, E, K, P, protocol=protocol)
    s.layer(0, np.arange(K))
    primeira = min(s.inflight.values())
    inicio = P["g"]
    if protocol == "token":
        assert primeira == pytest.approx(inicio + K * P["c_cpu_quiet"] + s.c_copy_cpu)
    else:
        assert primeira == pytest.approx(inicio + s.c_copy_cpu)


def test_backlog_do_token_e_o_do_fim_da_cpu():
    # Quatro cópias que terminam no meio da CPU da camada: no instante da admissão (fim da CPU) o backlog já está livre, e
    # as K faltas são admitidas.
    s = m.Sim("lru_cpu", 64, 2, E, K, P, protocol="token")
    for e in range(m.BACKLOG):
        s.inflight[(1, E - 1 - e)] = P["g"] + 0.5 * P["c_cpu_quiet"]
    s.layer(0, np.arange(K))
    assert s.stats["copies_bg"] == K
    assert s.stats["dropped"] == 0
