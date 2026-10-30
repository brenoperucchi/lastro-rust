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
