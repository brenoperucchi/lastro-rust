"""Previsão da comparação com o llama.cpp (plano, seção "Comparação com o llama.cpp nos Granite 1B e 3B"). Aquecimento na
conversa 3, medida nas 4 a 7. Rodar a partir da raiz do projeto, antes da primeira execução do rodar.sh.

- Cache do llama.cpp: acerto pela réplica do SLRU de src/llama-moe-cache.cpp (0893ec9), escrita pelo rev-2 na rodada 42 e
  conferida pelo exec contra o código (slot vazio primeiro, promoção do acerto ao protegido, protegido limitado à metade,
  vítima da probation primeiro, sem tirar experts da mesma camada pedidos agora). Tempo = GPU F32 do harness + faltas por
  token × custo de uma falta (três cópias pageable, uma por tensor gate/up/down, medidas pelo banda.cu) + camadas × SYNC.
- Nosso runtime: o simulador (tools/moe_sim.py, PARAMS_RUNTIME, backlog 4) para lru_sync, lru_cpu e lfu_cpu.
- Duração: (tokens do aquecimento + da medida) × ms/token previsto, mais CARGA_S por execução.
"""
import hashlib
import re
import sys
from collections import OrderedDict

sys.path.insert(0, "tools")
import moe_sim as m  # noqa: E402

TR = {"1b": "qual-runs/TRACOS-20261029/granite-3.1-1b-a400m-instruct", "3b": "qual-runs/TRACOS-20261029/granite-3.1-3b-a800m-instruct"}
FRACS = (0.1, 0.25, 0.5)
# Pilotos lidos do validacao.txt (item 5, ms/token médio das conversas 4 a 7): tudo na GPU F32 (-ctk/-ctv f32, -fa off),
# tudo na GPU F16 (FA padrão) e --cpu-moe F32; e o custo de uma cópia pageable síncrona por tensor (validacao/banda.txt,
# 1 MiB no 1B, 1,5 MiB no 3B). O llama-bench de 29/10 (tg128, sem taskset) dava 20 e 37 ms no --cpu-moe.
PILOTO = {(m, l): float(x) for m, l, x in re.findall(r"^  (1b|3b) (gpu32|gpu16|cpumoe): total: \d+ tokens, ms/tok ([\d.]+)",
                                                     open("qual-runs/LLAMACPP-COMPARA-20261030/validacao.txt").read(), re.M)}
GPU_F32 = {m: PILOTO[(m, "gpu32")] for m in ("1b", "3b")}
GPU_F16 = {m: PILOTO[(m, "gpu16")] for m in ("1b", "3b")}
CPU_F32 = {m: PILOTO[(m, "cpumoe")] for m in ("1b", "3b")}
BANDA = {round(float(a), 1): float(b) for a, b in re.findall(r"^([\d.]+) MiB: pageable ([\d.]+) ms",
                                                            open("qual-runs/LLAMACPP-COMPARA-20261030/validacao/banda.txt").read(), re.M)}
COPIA_TENSOR = {"1b": BANDA[1.0], "3b": BANDA[1.5]}
SYNC = 0.020  # ms por camada com cache (leitura dos ids e sincronização), com ou sem falta; estimativa
# Nosso residente na parte aquecida do C (média das 10 execuções aquecidas, analise.txt).
RESIDENTE = {"1b": 2.863, "3b": 4.452}
TOKENS_AQUEC = {"1b": 954, "3b": 969}
TOKENS_MEDIDA = {"1b": 3451, "3b": 3814}
CARGA_S = 8


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class SLRU:
    def __init__(s, n, E):
        s.n, s.E = n, E
        s.prob, s.prot = OrderedDict(), OrderedDict()  # chave -> None, o menos recente primeiro

    def run(s, convs, count):
        E = s.E
        hit = need = 0
        for a in convs:
            L, T, _ = a.shape
            for t in range(T):
                for l in range(L):
                    keys = [l * E + e for e in dict.fromkeys(int(x) for x in a[l, t])]
                    cur = set(keys)
                    h = 0
                    for key in keys:  # promove os acertos antes de escolher vítimas
                        if key in s.prob or key in s.prot:
                            h += 1
                            s.prob.pop(key, None)
                            s.prot.pop(key, None)
                            s.prot[key] = None
                            if len(s.prot) > s.n // 2:
                                kk, _ = s.prot.popitem(last=False)
                                s.prob[kk] = None
                    for key in sorted(keys):
                        if key in s.prob or key in s.prot:
                            continue
                        if len(s.prob) + len(s.prot) < s.n:
                            s.prob[key] = None
                            continue
                        victim = None
                        for seg in (s.prob, s.prot):
                            for kk in seg:
                                if kk // E != l or kk not in cur:
                                    victim = (seg, kk)
                                    break
                            if victim:
                                break
                        del victim[0][victim[1]]
                        s.prob[key] = None
                    if count:
                        hit += h
                        need += len(keys)
        return hit, need


def ours(policy, cap, L, E, k, p, warm, test):
    s = m.Sim(policy, cap, L, E, k, p, backlog=4)
    s.run(warm, False)
    s.run(test, True)
    st = s.stats
    return st["ms"] / st["tokens"], st["hits"] / st["needed"]


print("simulador:", sha("tools/moe_sim.py"), "| validacao.txt:", sha("qual-runs/LLAMACPP-COMPARA-20261030/validacao.txt")[:16])
print("pilotos (ms/token):", {f"{m} {l}": v for (m, l), v in sorted(PILOTO.items())}, "| cópia por tensor (ms):", COPIA_TENSOR)
print(f"SYNC {SYNC} ms por camada; carga {CARGA_S} s por execução")
total_s = 0.0
for mod in ("1b", "3b"):
    train, test = m.load(TR[mod], "treino"), m.load(TR[mod], "teste")
    warm = [train[3]]
    L, _, k = test[0].shape
    E = int(max(a.max() for a in train + test)) + 1
    p = m.PARAMS[mod]
    falta = 3 * COPIA_TENSOR[mod]
    gpu16, cpu = GPU_F16[mod], CPU_F32[mod]
    extra_f32 = GPU_F32[mod] - gpu16
    ntok = TOKENS_AQUEC[mod] + TOKENS_MEDIDA[mod]
    rows = []  # (linha, orçamento, ms/token previsto, acerto, repetições)
    log_hit = {}  # acerto da réplica somado sobre o aquecimento e a medida, para o hit rate do destrutor do llama.cpp
    rows.append(("llama.cpp GPU F32", "100%", GPU_F32[mod], None, 3))
    rows.append(("llama.cpp GPU F16", "100%", gpu16, None, 1))
    rows.append(("llama.cpp --cpu-moe", "0%", cpu, None, 1))
    for frac, ncm in zip(FRACS, {"1b": (22, 18, 12), "3b": (29, 24, 16)}[mod]):
        rows.append((f"llama.cpp -ncmoe {ncm}", f"{(L - ncm) / L:.1%}", GPU_F32[mod] + ncm / L * (cpu - GPU_F32[mod]), None, 1))
    for frac in FRACS:
        cap = int(frac * L * E)
        sl = SLRU(cap, E)
        hw, nw = sl.run(warm, True)
        h, n = sl.run(test, True)
        hit = h / n
        log_hit[frac] = (hw + h) / (nw + n)
        miss_tok = (1 - hit) * L * k
        rows.append((f"llama.cpp cache {cap}", f"{frac:.0%}", GPU_F32[mod] + miss_tok * falta + L * SYNC, hit, 3))
    rows.append((f"llama.cpp cache cheio {L * E}", "100%", GPU_F32[mod] + L * SYNC, 1.0, 1))
    rows.append(("nosso resident", "100%", RESIDENTE[mod], None, 3))
    for pol, reps in (("lru_cpu", 3), ("lfu_cpu", 3), ("lru_sync", 1)):
        for frac in FRACS:
            ms, hit = ours(pol, int(frac * L * E), L, E, k, p, warm, test)
            rows.append((f"nosso {pol}", f"{frac:.0%}", ms, hit, reps))
    rows.append(("nosso cache_full", "100%", RESIDENTE[mod], 1.0, 1))
    print(f"\n{mod}: {L} camadas, {E} experts por camada, k {k}; falta = 3 × {COPIA_TENSOR[mod]} ms; GPU F32 − F16 {extra_f32:+.3f} ms")
    print("linha                         orçam. | ms/tok   tok/s | acerto | repetições | duração (s)")
    for name, b, ms, hit, reps in rows:
        dur = reps * (ntok * ms / 1000 + CARGA_S)
        total_s += dur
        print(f"{name:29s} {b:>6s} | {ms:6.2f} {1000 / ms:7.1f} | {'' if hit is None else f'{hit:6.1%}':6s} | {reps:10d} | {dur:8.0f}")
    print("hit rate previsto no log do llama.cpp (réplica somada sobre aquecimento e medida): "
          + "  ".join(f"{frac:.0%}: {h:.2%}" for frac, h in log_hit.items()))
print(f"\nduração prevista sem a linha do servidor: {total_s / 3600:.2f} h")
