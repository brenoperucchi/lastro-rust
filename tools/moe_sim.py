#!/usr/bin/env python3
"""Simulador de colocação de experts (M0.5 do plano docs/PLANO-runtime-rust-moe-20261029.md).

Reproduz um decode em batch 1 sobre traços de roteamento reais (experts.npy [L, T, k] de tools/ref_logits.py --experts-only)
com um orçamento de VRAM de C slots de expert, e compara políticas pelo tempo gasto além do ideal "tudo residente".

Modelo de custo (por expert, em ms; valores do M0 na RTX 4090, ver PARAMS):
  - acerto: h na GPU;
  - falta servida por cópia: o PCIe é uma fila única (c_copy por expert); a GPU calcula quando a cópia termina;
  - falta servida pela CPU: c_cpu por expert, em fila na CPU, em paralelo com a GPU dentro da camada;
  - cópia em segundo plano (admissão): entra na fila do PCIe e o expert só vale como residente quando ela termina;
    não é enfileirada se a fila já passa de BACKLOG cópias.
  - cada camada gasta g na GPU antes dos experts (atenção, roteador).
Tempo da camada = max(fim da GPU, fim da CPU). "Espera" = tempo da camada além de g + k·h (o ideal com tudo residente).

Políticas:
  camadas    camadas inteiras residentes, as outras na CPU (o -ncmoe do llama.cpp); serve de calibração
  estatica   os C experts mais usados no treino ficam fixos; o resto na CPU (perfil estático)
  lru_sync   toda falta é copiada antes de calcular; despejo LRU (o cache síncrono do M2a)
  belady     como lru_sync, mas despeja o expert cujo próximo uso é o mais distante (oráculo, limite do modo síncrono)
  lru_cpu    falta na CPU; o expert é admitido em segundo plano, despejo LRU
  lfu_cpu    falta na CPU; admite só se a frequência com decaimento vence a do residente mais fraco (estilo contagem)
  custo      por falta, escolhe cópia ou CPU pelo que termina antes dadas as filas atuais; espera uma cópia em voo
             quando é mais barato que a CPU; admissão como lfu_cpu (a hipótese "colocação por custo medido")
  oraculo    falta na CPU; admite e despeja pelo próximo uso conhecido (limite do modo CPU + admissão)

  tools/.venv-ref/bin/python -I tools/moe_sim.py qual-runs/TRACOS-20261029/<modelo> --model 3b
"""
import argparse
import glob
import heapq
import json
import math
import os
from collections import OrderedDict

import numpy as np

# Custos em ms. c_copy = bytes do expert / 20 GB/s (M0, cópia pinned). c_cpu = (1/tok_s com --cpu-moe − 1/tok_s na GPU) /
# (camadas · k), do baseline llama.cpp. h e g dividem o tempo por camada do llama.cpp com tudo na GPU.
PARAMS = {
    "1b": {"expert_mb": 3 * 1024 * 512 * 2 / 1e6, "c_cpu": 0.095, "h": 0.008, "g": 0.009, "tg_gpu": 567, "tg_cpu": 50},
    "3b": {"expert_mb": 3 * 1536 * 512 * 2 / 1e6, "c_cpu": 0.131, "h": 0.012, "g": 0.012, "tg_gpu": 290, "tg_cpu": 27},
}
PCIE_GBS = 20.0
BACKLOG = 4
HALF_LIFE = 128  # tokens; decaimento da frequência de lfu_cpu e custo


def load(dirpath, which):
    convs = sorted(glob.glob(os.path.join(dirpath, "conversa-*", "experts.npy")))
    arrs = [np.load(p) for p in convs]
    return [a for i, a in enumerate(arrs) if (i < len(arrs) // 2) == (which == "treino")]


class Sim:
    def __init__(self, policy, cap, L, E, k, p, static_set=None, resident_layers=None, next_use=None):
        self.policy, self.cap, self.L, self.E, self.k, self.p = policy, cap, L, E, k, p
        self.c_copy = p["expert_mb"] / 1e3 / PCIE_GBS * 1e3
        self.now = 0.0
        self.pcie_free = 0.0
        self.resident = OrderedDict()  # chave -> None (ordem LRU)
        self.inflight = {}  # chave -> instante em que a cópia termina
        self.requested = {}  # chave em voo -> passo do último pedido (a recência do LRU quando a cópia assentar)
        self.freq = {}  # chave -> (log2 do valor, passo da última atualização)
        self.heap = []  # (prioridade de despejo, versão, chave), despejo lazy
        self.version = {}
        self.static_set = static_set or set()
        self.resident_layers = resident_layers or set()
        self.next_use = next_use  # função (chave, passo) -> próximo passo de uso
        self.step = 0
        self.stats = {"tokens": 0, "needed": 0, "hits": 0, "cpu": 0, "copies_sync": 0, "copies_bg": 0, "wait_ms": 0.0}

    # -- frequência com decaimento: a ordem entre chaves não muda com o tempo se usarmos log2(v) + passo/H
    def touch_freq(self, key):
        lv, s0 = self.freq.get(key, (-math.inf, self.step))
        v = 2.0 ** (lv - (self.step - s0) / (HALF_LIFE * self.L)) if lv > -math.inf else 0.0
        self.freq[key] = (math.log2(v + 1.0), self.step)

    def score(self, key):
        lv, s0 = self.freq.get(key, (-math.inf, 0))
        return lv + s0 / (HALF_LIFE * self.L)

    def prio(self, key, used=None):
        """Prioridade de despejo: o menor sai primeiro. `used` é o passo do último uso, se não for o atual."""
        if self.policy in ("lfu_cpu", "custo"):
            return self.score(key)
        if self.policy in ("belady", "oraculo"):
            return -self.next_use(key, self.step)
        return self.step if used is None else used  # LRU: último uso

    def push(self, key, used=None):
        v = self.version.get(key, 0) + 1
        self.version[key] = v
        heapq.heappush(self.heap, (self.prio(key, used), v, key))

    def settle(self):
        for key, f in list(self.inflight.items()):
            if f <= self.now:
                del self.inflight[key]
                self.resident[key] = None
                # A recência é a do pedido, não a da camada em que a cópia assenta: senão o expert copiado empata com os
                # acertos da camada seguinte e o desempate pode despejar um acerto mais recente.
                self.push(key, self.requested.pop(key, None))

    def victim(self, pinned):
        """Residente a despejar (fora das chaves em uso nesta camada), ou None."""
        held = []
        found = None
        while self.heap:
            prio, v, key = heapq.heappop(self.heap)
            if key not in self.resident or self.version.get(key) != v:
                continue
            if key in pinned:
                held.append((prio, v, key))
                continue
            found = (prio, key)
            break
        for item in held:
            heapq.heappush(self.heap, item)
        if found:
            heapq.heappush(self.heap, (found[0], self.version[found[1]], found[1]))
        return found

    def make_room(self, pinned, candidate_prio=None):
        """Libera um slot se preciso. Com candidate_prio, só despeja quem tem prioridade menor que o candidato."""
        if len(self.resident) + len(self.inflight) < self.cap:
            return True
        vic = self.victim(pinned)
        if vic is None or (candidate_prio is not None and vic[0] >= candidate_prio):
            return False
        del self.resident[vic[1]]
        return True

    def copy_sync(self, key, pinned):
        if not self.make_room(pinned):
            return None
        start = max(self.pcie_free, self.now)
        self.pcie_free = start + self.c_copy
        self.inflight[key] = self.pcie_free
        self.requested[key] = self.step
        self.stats["copies_sync"] += 1
        return self.pcie_free

    def admit_bg(self, key, pinned):
        if self.cap == 0 or self.pcie_free - self.now > BACKLOG * self.c_copy:
            return
        if self.policy == "lru_cpu":
            ok = self.make_room(pinned)
        else:
            ok = self.make_room(pinned, candidate_prio=self.prio(key))
        if ok:
            self.pcie_free = max(self.pcie_free, self.now) + self.c_copy
            self.inflight[key] = self.pcie_free
            self.requested[key] = self.step
            self.stats["copies_bg"] += 1

    def layer(self, l, experts):
        p = self.p
        self.now += p["g"]
        self.settle()
        start = self.now
        keys = [(l, int(e)) for e in experts]
        pinned = set(keys)
        for key in keys:
            self.touch_freq(key)
        n_gpu, gpu_ready, cpu_t = 0, start, start
        for key in keys:
            self.stats["needed"] += 1
            if self.policy == "camadas":
                if l in self.resident_layers:
                    n_gpu += 1
                    self.stats["hits"] += 1
                else:
                    cpu_t += p["c_cpu"]
                    self.stats["cpu"] += 1
                continue
            if self.policy == "estatica":
                if key in self.static_set:
                    n_gpu += 1
                    self.stats["hits"] += 1
                else:
                    cpu_t += p["c_cpu"]
                    self.stats["cpu"] += 1
                continue
            if key in self.resident:
                self.resident.move_to_end(key)
                self.push(key)
                n_gpu += 1
                self.stats["hits"] += 1
                continue
            sync = self.policy in ("lru_sync", "belady")
            if key in self.inflight:
                self.requested[key] = self.step
                f = self.inflight[key]
                if self.policy == "custo":
                    wait = max(f, start) + p["h"] <= cpu_t + p["c_cpu"]
                else:
                    wait = sync or f - start <= p["c_cpu"]
                if wait:
                    gpu_ready = max(gpu_ready, f)
                    n_gpu += 1
                else:
                    cpu_t += p["c_cpu"]
                    self.stats["cpu"] += 1
                continue
            if sync:
                f = self.copy_sync(key, pinned)
                assert f is not None, "orçamento menor que os experts de uma camada"
                gpu_ready = max(gpu_ready, f)
                n_gpu += 1
                continue
            if self.policy == "custo":
                fin_copy = max(self.pcie_free, start) + self.c_copy + p["h"]
                if fin_copy < cpu_t + p["c_cpu"]:
                    f = self.copy_sync(key, pinned)
                    if f is not None:
                        gpu_ready = max(gpu_ready, f)
                        n_gpu += 1
                        continue
            cpu_t += p["c_cpu"]
            self.stats["cpu"] += 1
            self.admit_bg(key, pinned)
        gpu_end = max(start + n_gpu * p["h"], gpu_ready + p["h"]) if n_gpu else start
        end = max(gpu_end, cpu_t)
        self.stats["wait_ms"] += (end - start) - len(keys) * p["h"]
        self.now = end
        self.step += 1

    def run(self, convs, measure):
        if measure:
            for k in self.stats:
                self.stats[k] = 0 if k != "wait_ms" else 0.0
            t0 = self.now
        for a in convs:
            L, T, _ = a.shape
            for t in range(T):
                for l in range(L):
                    self.layer(l, a[l, t])
                self.stats["tokens"] += 1
        if measure:
            self.stats["ms"] = self.now - t0


def next_use_fn(convs_all, L):
    """Para o oráculo: lista ordenada dos passos de uso de cada chave no fluxo treino + teste."""
    uses = {}
    step = 0
    for a in convs_all:
        _, T, _ = a.shape
        for t in range(T):
            for l in range(L):
                for e in a[l, t]:
                    uses.setdefault((l, int(e)), []).append(step)
                step += 1
    import bisect

    def f(key, s):
        lst = uses.get(key, [])
        i = bisect.bisect_right(lst, s)
        return lst[i] if i < len(lst) else 10**12

    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces")
    ap.add_argument("--model", choices=PARAMS, required=True)
    ap.add_argument("--caps", default="0.1,0.25,0.5,0.75")
    ap.add_argument("--json-out")
    args = ap.parse_args()
    p = PARAMS[args.model]
    train, test = load(args.traces, "treino"), load(args.traces, "teste")
    L, _, k = train[0].shape
    E = int(max(a.max() for a in train + test)) + 1
    total = L * E
    nu = next_use_fn(train + test, L)
    counts = {}
    for a in train:
        for l in range(L):
            for e in a[l].ravel():
                counts[(l, int(e))] = counts.get((l, int(e)), 0) + 1

    c_copy = p["expert_mb"] / PCIE_GBS
    print(f"modelo {args.model}: {L} camadas × {E} experts (k={k}), {total} experts; treino {sum(a.shape[1] for a in train)} "
          f"tokens, teste {sum(a.shape[1] for a in test)} tokens; c_copy {c_copy:.3f} ms, c_cpu {p['c_cpu']} ms, h {p['h']} ms")

    # Calibração contra o llama.cpp: tudo na GPU, metade das camadas e nenhuma.
    for frac, tg in ((1.0, p["tg_gpu"]), (0.5, None), (0.0, p["tg_cpu"])):
        s = Sim("camadas", 0, L, E, k, p, resident_layers=set(range(int(round(frac * L)))))
        s.run(test, measure=True)
        print(f"  calibração camadas residentes {frac:.0%}: {s.stats['tokens'] / s.stats['ms'] * 1e3:.0f} tok/s simulado"
              + (f" (llama.cpp {tg} tok/s)" if tg else ""))

    results = []
    policies = ["camadas", "estatica", "lru_sync", "belady", "lru_cpu", "lfu_cpu", "custo", "oraculo"]
    for frac in [float(x) for x in args.caps.split(",")]:
        cap = int(frac * total)
        print(f"\norçamento {frac:.0%} = {cap} slots ({cap * p['expert_mb'] / 1e3:.2f} GB)")
        print("  política   tok/s   acerto   CPU   cópias síncronas/token   cópias bg/token   espera ms/token   vs lru_sync")
        rows = {}
        for pol in policies:
            kw = {}
            if pol == "camadas":
                kw["resident_layers"] = set(range(int(cap // E)))
            if pol == "estatica":
                kw["static_set"] = set(sorted(counts, key=counts.get, reverse=True)[:cap])
            if pol in ("belady", "oraculo"):
                kw["next_use"] = nu
            s = Sim(pol, cap, L, E, k, p, **kw)
            s.run(train, measure=False)  # aquecimento (o oráculo enxerga o fluxo inteiro)
            s.run(test, measure=True)
            st = s.stats
            rows[pol] = st
        base = rows["lru_sync"]["wait_ms"]
        for pol in policies:
            st = rows[pol]
            n = st["tokens"]
            print(f"  {pol:9s} {n / st['ms'] * 1e3:6.0f}   {st['hits'] / st['needed']:6.1%}  {st['cpu'] / st['needed']:5.1%}"
                  f"   {st['copies_sync'] / n:21.1f}   {st['copies_bg'] / n:15.1f}   {st['wait_ms'] / n:15.2f}"
                  f"   {(st['wait_ms'] / base - 1) * 100:+6.1f}%")
            results.append({"cap_frac": frac, "cap": cap, "policy": pol, **{k2: v for k2, v in st.items()}})
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"model": args.model, "params": p, "pcie_gbs": PCIE_GBS, "backlog": BACKLOG,
                       "half_life": HALF_LIFE, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
