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
    assert nada.stats["ms"] == pytest.approx(50 * L * (P["g"] + K * P["c_cpu"]))


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
