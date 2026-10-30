#!/usr/bin/env python3
"""Simulador de colocação de experts (M0.5 do plano docs/PLANO-runtime-rust-moe-20261029.md).

Reproduz um decode em batch 1 sobre traços de roteamento reais (experts.npy [L, T, k] de tools/ref_logits.py --experts-only)
com um orçamento de VRAM de C slots de expert, e compara políticas pelo tempo gasto além do ideal "tudo residente".

Modelo de custo (por expert, em ms, na RTX 4090; ver PARAMS):
  - acerto: h na GPU;
  - falta servida por cópia: o PCIe é uma fila única (c_copy por expert); a GPU calcula quando a cópia termina;
  - falta servida pela CPU: em fila na CPU, em paralelo com a GPU dentro da camada; c_cpu por expert numa camada com
    tráfego de cópia, c_cpu_quiet numa camada sem (a CPU e o DMA disputam a memória). Camada com tráfego = começa com
    cópia em voo ou enfileira uma;
  - pelo mesmo motivo, com a CPU calculando experts a cópia vai a pcie_gbs_cpu em vez de PCIE_GBS: toda cópia em segundo
    plano, e a cópia síncrona pedida numa camada em que a CPU já trabalha;
  - cópia em segundo plano (admissão): entra na fila do PCIe e o expert só vale como residente quando ela termina, no
    começo de uma camada; não é enfileirada se já há BACKLOG cópias a caminho (protocolo "runtime", o do ExpertCache) ou se
    a fila do PCIe já passa de BACKLOG cópias (protocolo "m05", o do M0.5).
  - cada camada gasta g na GPU antes dos experts (atenção, roteador).
  - camada com tráfego de cópia e algum expert fora da GPU (na CPU, ou esperando uma cópia) gasta o_layer a mais. É uma
    sobra medida, sem causa isolada: aparece no lru_sync e no lru_cpu e some no lru_cpu com backlog 0 (só CPU, sem cópia).
    O tempo de despachar e juntar as threads da CPU já está no c_cpu.
Tempo da camada = max(fim da GPU, fim da CPU) (+ o_layer). "Espera" = tempo da camada além de g + k·h (o ideal com tudo
residente).

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

Parâmetros: "runtime" (padrão) vem do nosso runtime na velocidade do M2f (cópia e CPU por expert medidas no m2a, h dos dois
kernels de um expert no m2c, g o resto do tempo de GPU por camada do m2b); "llamacpp" é o do M0.5 (custos do llama.cpp).
Protocolo: "runtime" (padrão) faz o lru_cpu como o ExpertCache (chave carregando sempre vai à CPU; backlog conta cópias
pendentes); "m05" é o do M0.5 (espera a cópia em voo se ela termina antes da CPU; backlog pelo tempo de fila do PCIe).

  tools/.venv-ref/bin/python -I tools/moe_sim.py qual-runs/TRACOS-20261029/<modelo> --model 3b
  tools/.venv-ref/bin/python -I tools/moe_sim.py qual-runs/TRACOS-20261029/<modelo> --model 3b --validar <saída do m2a-cache>
"""
import argparse
import glob
import heapq
import json
import math
import os
from collections import OrderedDict

import numpy as np

# Custos em ms. c_copy = bytes do expert / 20 GB/s nos dois conjuntos (M0, cópia pinned; o m2a mediu 0,157 e 0,233 ms por
# cópia, igual à conta).
# M0.5: c_cpu = (1/tok_s com --cpu-moe − 1/tok_s na GPU) / (camadas · k), do baseline llama.cpp; h e g dividem o tempo por
# camada do llama.cpp com tudo na GPU.
PARAMS_LLAMACPP = {
    "1b": {"expert_mb": 3 * 1024 * 512 * 2 / 1e6, "c_cpu": 0.095, "h": 0.008, "g": 0.009, "tg_gpu": 567, "tg_cpu": 50},
    "3b": {"expert_mb": 3 * 1536 * 512 * 2 / 1e6, "c_cpu": 0.131, "h": 0.012, "g": 0.012, "tg_gpu": 290, "tg_cpu": 27},
}
# Runtime (M2f, qual-runs/M2F-FUSOES-20261029; 8 threads de CPU): c_cpu e c_cpu_quiet = CPU ms medido por token contra os
# experts na CPU por token simulados em camadas com e sem tráfego de cópia (stats cpu_traffic e cpu_quiet), por mínimos
# quadrados sem pesos nos 4 lru_cpu e nos 4 lru_cpu com backlog 0 (m2a-*-final-1.txt, m2a-*-final-backlog0-{1,2}.txt),
# iterado até o ponto fixo, porque as contagens dependem do próprio c_cpu (2 iterações; resíduo máximo 0,22 e 0,32
# ms/token). Um custo fixo por camada com CPU no lugar da disputa deixa 0,70 e 0,99 ms. h = gemv_swiglu + gemv_axpby do m2c (fila; 2,74 + 2,22 µs no 1B, 2,82 + 2,37
# no 3B); g = GPU por token do m2b (2,99 e 4,83 ms, modo normal) ÷ camadas − k·h. tg_gpu = 1000 ÷ esse tempo de GPU por token.
# o_layer = 36,7 µs, ajustado só no lru_sync do 1B (m2a-1b-final-1.txt, 4 orçamentos, mínimos quadrados pela origem do
# tempo medido além do simulado contra as camadas com cópia); validado no resto (--validar). No 3B, o ajuste próprio daria
# 25 µs, e o critério de ±10% distingue pouco esse termo (2–8% do tempo).
# pcie_gbs_cpu = 14,1 GB/s: no lru_cpu a 10% do 1B o backlog fica sempre cheio, então a vazão medida é a banda efetiva
# (10,13 ms/token ÷ 45,3 cópias = 0,224 ms por 3,15 MB). O 3B, que não entrou no ajuste, dá 14,2 (17,53 ÷ 52,6 por 4,72 MB).
O_LAYER_RUNTIME = 0.0367
PCIE_GBS_CPU_RUNTIME = 14.1
PARAMS_RUNTIME = {
    "1b": {"expert_mb": 3 * 1024 * 512 * 2 / 1e6, "c_cpu": 0.0464, "c_cpu_quiet": 0.0401, "h": 0.00496, "g": 2.99 / 24 - 8 * 0.00496, "o_layer": O_LAYER_RUNTIME, "pcie_gbs_cpu": PCIE_GBS_CPU_RUNTIME,
           "tg_gpu": 1000 / 2.99, "tg_cpu": None},
    "3b": {"expert_mb": 3 * 1536 * 512 * 2 / 1e6, "c_cpu": 0.0694, "c_cpu_quiet": 0.0610, "h": 0.00519, "g": 4.83 / 32 - 8 * 0.00519, "o_layer": O_LAYER_RUNTIME, "pcie_gbs_cpu": PCIE_GBS_CPU_RUNTIME,
           "tg_gpu": 1000 / 4.83, "tg_cpu": None},
}
PARAMS_SETS = {"runtime": PARAMS_RUNTIME, "llamacpp": PARAMS_LLAMACPP}
PARAMS = PARAMS_RUNTIME
PCIE_GBS = 20.0
BACKLOG = 4
HALF_LIFE = 128  # tokens; decaimento da frequência de lfu_cpu e custo


def load(dirpath, which):
    convs = sorted(glob.glob(os.path.join(dirpath, "conversa-*", "experts.npy")))
    arrs = [np.load(p) for p in convs]
    return [a for i, a in enumerate(arrs) if (i < len(arrs) // 2) == (which == "treino")]


class Sim:
    def __init__(self, policy, cap, L, E, k, p, static_set=None, resident_layers=None, next_use=None, protocol="runtime", backlog=BACKLOG):
        assert protocol in ("runtime", "m05")
        self.policy, self.cap, self.L, self.E, self.k, self.p, self.protocol = policy, cap, L, E, k, p, protocol
        self.backlog = backlog
        self.c_copy = p["expert_mb"] / 1e3 / PCIE_GBS * 1e3
        self.c_copy_cpu = p["expert_mb"] / p.get("pcie_gbs_cpu", PCIE_GBS)  # cópia com a CPU calculando experts
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
        self.stats = {"tokens": 0, "needed": 0, "hits": 0, "cpu": 0, "copies_sync": 0, "copies_bg": 0, "dropped": 0, "wait_ms": 0.0,
                      "layers_cpu": 0, "layers_sync": 0, "layers_traffic": 0, "cpu_traffic": 0, "cpu_quiet": 0}

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

    def copy_sync(self, key, pinned, cpu_busy=False):
        if not self.make_room(pinned):
            return None
        start = max(self.pcie_free, self.now)
        self.pcie_free = start + (self.c_copy_cpu if cpu_busy else self.c_copy)
        self.inflight[key] = self.pcie_free
        self.requested[key] = self.step
        self.stats["copies_sync"] += 1
        return self.pcie_free

    def admit_bg(self, key, pinned):
        if self.protocol == "runtime":
            full = len(self.inflight) >= self.backlog  # como o ExpertCache: cópias ainda não assentadas
        else:
            full = self.pcie_free - self.now > self.backlog * self.c_copy_cpu
        if self.cap == 0 or full:
            self.stats["dropped"] += 1
            return
        if self.policy == "lru_cpu":
            ok = self.make_room(pinned)
        else:
            ok = self.make_room(pinned, candidate_prio=self.prio(key))
        if ok:
            self.pcie_free = max(self.pcie_free, self.now) + self.c_copy_cpu
            self.inflight[key] = self.pcie_free
            self.requested[key] = self.step
            self.stats["copies_bg"] += 1
        else:
            self.stats["dropped"] += 1

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
        busy0 = bool(self.inflight)  # tráfego de cópia desde o começo da camada
        copies0 = self.stats["copies_sync"] + self.stats["copies_bg"]
        c_cpu = p["c_cpu"] if busy0 else p.get("c_cpu_quiet", p["c_cpu"])
        n_cpu = 0
        for key in keys:
            self.stats["needed"] += 1
            if self.policy == "camadas":
                if l in self.resident_layers:
                    n_gpu += 1
                    self.stats["hits"] += 1
                else:
                    cpu_t += c_cpu
                    n_cpu += 1
                    self.stats["cpu"] += 1
                continue
            if self.policy == "estatica":
                if key in self.static_set:
                    n_gpu += 1
                    self.stats["hits"] += 1
                else:
                    cpu_t += c_cpu
                    n_cpu += 1
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
                    wait = max(f, start) + p["h"] <= cpu_t + c_cpu
                elif self.protocol == "runtime" and not sync:
                    wait = False  # ExpertCache: Lookup::Loading vai à CPU
                else:
                    wait = sync or f - start <= c_cpu
                if wait:
                    gpu_ready = max(gpu_ready, f)
                    n_gpu += 1
                else:
                    cpu_t += c_cpu
                    n_cpu += 1
                    self.stats["cpu"] += 1
                continue
            if sync:
                f = self.copy_sync(key, pinned)
                assert f is not None, "orçamento menor que os experts de uma camada"
                gpu_ready = max(gpu_ready, f)
                n_gpu += 1
                continue
            if self.policy == "custo":
                busy = cpu_t > start
                fin_copy = max(self.pcie_free, start) + (self.c_copy_cpu if busy else self.c_copy) + p["h"]
                if fin_copy < cpu_t + c_cpu:
                    f = self.copy_sync(key, pinned, cpu_busy=busy)
                    if f is not None:
                        gpu_ready = max(gpu_ready, f)
                        n_gpu += 1
                        continue
            cpu_t += c_cpu
            n_cpu += 1
            self.stats["cpu"] += 1
            self.admit_bg(key, pinned)
        traffic = busy0 or self.stats["copies_sync"] + self.stats["copies_bg"] > copies0
        if traffic and not busy0:
            cpu_t += n_cpu * (p["c_cpu"] - c_cpu)  # a cópia enfileirada nesta camada corre junto com a CPU
        self.stats["layers_cpu"] += cpu_t > start
        self.stats["layers_sync"] += gpu_ready > start
        self.stats["layers_traffic"] += traffic
        self.stats["cpu_traffic" if traffic else "cpu_quiet"] += n_cpu
        gpu_end = max(start + n_gpu * p["h"], gpu_ready + p["h"]) if n_gpu else start
        end = max(gpu_end, cpu_t)
        if traffic and (cpu_t > start or gpu_ready > start):
            end += p.get("o_layer", 0.0)
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


def read_m2a(path):
    """Linhas lru_sync/lru_cpu da tabela principal do m2a-cache: ({(política, slots): {ms, hit, cpu, sync, bg}}, backlog)."""
    import re

    rows = {}
    backlog = BACKLOG
    for line in open(path):
        b = re.search(r"; backlog (\d+)$", line.rstrip())
        if line.startswith("M2a:") and b:
            backlog = int(b[1])
        m = re.match(r"^(lru_sync|lru_cpu)\s+(\d+)%\s+(\d+)\s+\|\s+([\d.]+)\s+[\d.]+\s+[\d.]+\s+\|\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            rows[(m[1], int(m[3]))] = {"frac": int(m[2]) / 100, "ms": float(m[4]), "hit": float(m[5]) / 100, "cpu": float(m[6]) / 100,
                                       "sync": float(m[7]), "bg": float(m[8])}
    return rows, backlog


def validate(path, test, L, E, k, p, protocol):
    """Simula o que o m2a mediu, nas mesmas condições (cache vazio, conversas de teste em sequência, sem aquecimento), e
    compara. Critério da recalibração: acerto a ±2 pontos e ms/token a ±10%."""
    meas, backlog = read_m2a(path)
    gpu_tok = L * (p["g"] + k * p["h"])
    print(f"validação contra {path} (protocolo {protocol}, backlog {backlog}); residente simulado {gpu_tok:.2f} ms/token")
    print("  política   orçam.  slots | acerto sim  medido   Δpt | ms/tok sim  medido     Δ | cóp.s sim/med | cóp.bg sim/med | ok")
    ok_all = True
    for (pol, cap), mm in sorted(meas.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        s = Sim(pol, cap, L, E, k, p, protocol=protocol, backlog=backlog)
        s.run(test, measure=True)
        st = s.stats
        n = st["tokens"]
        ms = st["ms"] / n
        hit = st["hits"] / st["needed"]
        dpt = (hit - mm["hit"]) * 100
        dms = ms / mm["ms"] - 1
        ok = abs(dpt) <= 2 and abs(dms) <= 0.10
        ok_all &= ok
        print(f"  {pol:9s} {mm['frac']:6.0%} {cap:6d} | {hit:10.1%} {mm['hit']:7.1%} {dpt:+5.1f} | {ms:10.2f} {mm['ms']:7.2f} {dms:+6.1%}"
              f" | {st['copies_sync'] / n:6.1f}/{mm['sync']:<6.1f} | {st['copies_bg'] / n:6.1f}/{mm['bg']:<6.1f} | {'sim' if ok else 'NÃO'}")
    print("resultado: " + ("dentro do critério em todas as linhas" if ok_all else "fora do critério em pelo menos uma linha"))
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces")
    ap.add_argument("--model", choices=PARAMS, required=True)
    ap.add_argument("--caps", default="0.1,0.25,0.5,0.75")
    ap.add_argument("--params", choices=PARAMS_SETS, default="runtime")
    ap.add_argument("--protocol", choices=("runtime", "m05"), default="runtime")
    ap.add_argument("--cpu-scale", type=float, default=1.0, help="multiplica c_cpu e c_cpu_quiet (sensibilidade)")
    ap.add_argument("--validar", nargs="+", help="saídas do m2a-cache para comparar com a simulação nas mesmas condições")
    ap.add_argument("--json-out")
    args = ap.parse_args()
    p = dict(PARAMS_SETS[args.params][args.model])
    for key in ("c_cpu", "c_cpu_quiet"):
        if key in p:
            p[key] *= args.cpu_scale
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
          f"tokens, teste {sum(a.shape[1] for a in test)} tokens; parâmetros {args.params}, protocolo {args.protocol}; "
          f"c_copy {c_copy:.3f} ms ({p['expert_mb'] / p.get('pcie_gbs_cpu', PCIE_GBS):.3f} com a CPU ocupada), c_cpu {p['c_cpu']:.4f} ms ({p.get('c_cpu_quiet', p['c_cpu']):.4f} sem cópia), "
          f"h {p['h']:.4f} ms, g {p['g']:.4f} ms, o_layer {p.get('o_layer', 0.0):.4f} ms")
    if args.validar:
        ok = [validate(path, test, L, E, k, p, args.protocol) for path in args.validar]
        raise SystemExit(0 if all(ok) else 1)

    # Calibração contra o llama.cpp: tudo na GPU, metade das camadas e nenhuma.
    ref = "runtime, GPU do m2b" if args.params == "runtime" else "llama.cpp"
    for frac, tg in ((1.0, p["tg_gpu"]), (0.5, None), (0.0, p["tg_cpu"])):
        s = Sim("camadas", 0, L, E, k, p, resident_layers=set(range(int(round(frac * L)))))
        s.run(test, measure=True)
        print(f"  calibração camadas residentes {frac:.0%}: {s.stats['tokens'] / s.stats['ms'] * 1e3:.0f} tok/s simulado"
              + (f" ({ref} {tg:.0f} tok/s)" if tg else ""))

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
            s = Sim(pol, cap, L, E, k, p, protocol=args.protocol, **kw)
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
            json.dump({"model": args.model, "params": p, "params_set": args.params, "protocol": args.protocol, "pcie_gbs": PCIE_GBS, "backlog": BACKLOG,
                       "half_life": HALF_LIFE, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
