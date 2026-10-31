"""Teste de carga da fonte HX1500i com as duas GPUs (31/10). Etapas contínuas:
  0–120 s   4090 em carga máxima (burn, HGEMM)
  120–240 s + 3080 Ti em carga máxima (limitada a 300 W pelo gpu-power-limit.service)
  240–360 s + CPU inteira (stress-ng, 28 workers, matrixprod)
Cada filho recebe a própria duração e termina sozinho mesmo se este roteiro cair. Amostra a cada 0,5 s o hwmon da fonte
(corsairpsu) e o nvidia-smi das duas GPUs. Aborta (mata os filhos pelo PID exato) se: potência total > 1300 W,
+12 V > 110 A, +12 V < 11,6 V, VRM da fonte > 63 °C ou alguma GPU > 85 °C.
  python3 carga.py > carga.txt   (da pasta deste arquivo)"""
import csv
import glob
import os
import subprocess
import sys
import time

U4090 = "GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd"
U3080 = "GPU-b22d9465-8552-9f97-812e-380b590645f8"
LIM = {"power1": (">", 1300.0), "curr2": (">", 110.0), "in1": ("<", 11.6), "temp1": (">", 63.0)}
GPU_TEMP_MAX = 85.0
ETAPAS = [(0, "4090"), (120, "+3080 Ti"), (240, "+CPU")]
FIM = 360

hw = next(h for h in glob.glob("/sys/class/hwmon/hwmon*") if open(f"{h}/name").read().strip() == "corsairpsu")
DIV = {"power1": 1e6, "power2": 1e6, "curr2": 1e3, "in1": 1e3, "temp1": 1e3}


def psu():
    return {k: int(open(f"{hw}/{k}_input").read()) / d for k, d in DIV.items()}


def gpus():
    out = subprocess.run(["nvidia-smi", "--query-gpu=uuid,power.draw,temperature.gpu,clocks.sm",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
    r = {}
    for line in out.strip().splitlines():
        u, p, t, c = [x.strip() for x in line.split(",")]
        tag = "g4090" if u == U4090 else "g3080"
        r[f"{tag}_w"], r[f"{tag}_c"], r[f"{tag}_mhz"] = float(p), float(t), float(c)
    return r


def burn(uuid, secs):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)
    return subprocess.Popen(["taskset", "-c", "0-15", "./burn", str(secs)], env=env, stdout=subprocess.PIPE, text=True)


filhos = []
campos = ["t", "etapa", "power1", "power2", "curr2", "in1", "temp1", "g4090_w", "g4090_c", "g4090_mhz", "g3080_w", "g3080_c", "g3080_mhz"]
pico = {}
motivo = None
t0 = time.monotonic()
prox = 0
with open("amostras.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=campos)
    w.writeheader()
    print(f"início {time.strftime('%H:%M:%S')}; antes: {psu()} {gpus()}", flush=True)
    while True:
        t = time.monotonic() - t0
        if t >= FIM:
            break
        while prox < len(ETAPAS) and t >= ETAPAS[prox][0]:
            nome = ETAPAS[prox][1]
            resta = int(FIM - ETAPAS[prox][0])
            if nome == "4090":
                filhos.append(burn(U4090, resta))
            elif nome == "+3080 Ti":
                filhos.append(burn(U3080, resta))
            else:
                filhos.append(subprocess.Popen(["stress-ng", "--cpu", "28", "--cpu-method", "matrixprod", "--timeout", f"{resta}s", "--quiet"]))
            print(f"{time.strftime('%H:%M:%S')} etapa {nome} (pid {filhos[-1].pid})", flush=True)
            prox += 1
        etapa = ETAPAS[prox - 1][1]
        s = {"t": round(t, 2), "etapa": etapa, **psu(), **gpus()}
        w.writerow(s)
        f.flush()
        for k, v in s.items():
            if isinstance(v, float):
                pk = (etapa, k)
                pico[pk] = max(pico.get(pk, v), v) if k != "in1" else min(pico.get(pk, v), v)
        for k, (op, lim) in LIM.items():
            if (op == ">" and s[k] > lim) or (op == "<" and s[k] < lim):
                motivo = f"{k} = {s[k]} (limite {op} {lim})"
        for k in ("g4090_c", "g3080_c"):
            if s.get(k, 0) > GPU_TEMP_MAX:
                motivo = f"{k} = {s[k]} °C"
        if motivo:
            break
        time.sleep(max(0.0, 0.5 - (time.monotonic() - t0 - t)))

if motivo:
    print(f"{time.strftime('%H:%M:%S')} ABORTADO: {motivo}", flush=True)
    for p in filhos:
        if p.poll() is None:
            p.kill()
for p in filhos:
    try:
        out = p.communicate(timeout=30)[0]
        if out:
            print(f"pid {p.pid}: {out.strip()}")
    except subprocess.TimeoutExpired:
        p.kill()
print(f"fim {time.strftime('%H:%M:%S')}; depois: {psu()}")
print("\npicos por etapa (in1: mínimo)")
for etapa in [e for _, e in ETAPAS]:
    linha = {k: v for (e, k), v in pico.items() if e == etapa}
    if linha:
        print(f"  {etapa:9s} total {linha['power1']:.0f} W | +12V {linha['power2']:.0f} W {linha['curr2']:.1f} A min {linha['in1']:.3f} V"
              f" | VRM {linha['temp1']:.1f} °C | 4090 {linha.get('g4090_w', 0):.0f} W {linha.get('g4090_c', 0):.0f} °C"
              f" | 3080 Ti {linha.get('g3080_w', 0):.0f} W {linha.get('g3080_c', 0):.0f} °C")
sys.exit(2 if motivo else 0)
