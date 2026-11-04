// A2-LOTE (GPU): tempo de UM expert do Granite para n posições juntas. Critério: criterio.txt (v1.2).
// Kernel de lote dedicado em CUDA C++ (não é o cutile do motor): um warp por linha de peso, a linha vai UMA vez para registradores
// e é usada pelas n posições. Dois kernels por expert (gate_up+SwiGLU, depois down), 64 experts em rodízio dentro de um CUDA Graph.
// Também mede a variante "serial" (n execuções de n=1 seguidas do MESMO expert; da 2ª em diante os pesos vêm da L2, então é informativo e NÃO é o w=n da DRAM). Uso: CUDA_VISIBLE_DEVICES=<UUID da 4090> lote_gpu [--var=0|1|2] [--so-n1] [--rapido]
// Saída: GPU <modelo> n=<n> lote_us=<x> serial_us=<y> err_lote_serial=<e> err_ref=<e>   (saída 3 se algum erro > 1e-3)
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

static const int NEXP = 64;
typedef __nv_bfloat16 bf;

__device__ __forceinline__ float bflo(unsigned u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bfhi(unsigned u) { return __uint_as_float(u & 0xffff0000u); }

__device__ __forceinline__ float dot8(const uint4& w, const float* x) {
  float4 a = __ldg(reinterpret_cast<const float4*>(x));
  float4 b = __ldg(reinterpret_cast<const float4*>(x) + 1);
  return bflo(w.x) * a.x + bfhi(w.x) * a.y + bflo(w.y) * a.z + bfhi(w.y) * a.w +
         bflo(w.z) * b.x + bfhi(w.z) * b.y + bflo(w.w) * b.z + bfhi(w.w) * b.w;
}
__device__ __forceinline__ float wsum(float v) {
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// Cada linha de peso é repartida entre S warps (split-K): cada warp lê NV/S vetores por lane e as somas parciais se juntam em memória
// compartilhada. Mais warps por linha = mais leituras de DRAM em voo no n=1. S=1 é a v1.1. Exige n <= 32 e a linha divisível por R = linhas por bloco.
// gate_up: linhas [0,inter) = gate, [inter,2*inter) = up (como no motor). a[p*inter+r] = silu(g)*u.
template <int HID, int S>
__global__ void k_gu(const bf* __restrict__ gu, const float* __restrict__ x, float* __restrict__ a, int n, int inter) {
  constexpr int NV = HID / 256, NVS = NV / S;
  static_assert(NV % S == 0, "S tem de dividir NV");
  __shared__ float sm[6 * 64];
  int wid = threadIdx.x / 32, lane = threadIdx.x & 31, rl = wid / S, s = wid % S;
  int r = blockIdx.x * (blockDim.x / 32 / S) + rl;
  uint4 g[NVS], u[NVS];
  const uint4* gp = reinterpret_cast<const uint4*>(gu + (size_t)r * HID);
  const uint4* up = reinterpret_cast<const uint4*>(gu + (size_t)(inter + r) * HID);
#pragma unroll
  for (int v = 0; v < NVS; v++) { g[v] = gp[(s * NVS + v) * 32 + lane]; u[v] = up[(s * NVS + v) * 32 + lane]; }
  for (int p = 0; p < n; p++) {
    float sg = 0, su = 0;
#pragma unroll
    for (int v = 0; v < NVS; v++) {
      const float* xp = x + (size_t)p * HID + (s * NVS + v) * 256 + lane * 8;
      sg += dot8(g[v], xp);
      su += dot8(u[v], xp);
    }
    sg = wsum(sg); su = wsum(su);
    if (lane == 0) { sm[wid * 64 + p * 2] = sg; sm[wid * 64 + p * 2 + 1] = su; }
  }
  __syncthreads();
  if (s == 0 && lane < n) {
    float sg = 0, su = 0;
    for (int k = 0; k < S; k++) { sg += sm[(rl * S + k) * 64 + lane * 2]; su += sm[(rl * S + k) * 64 + lane * 2 + 1]; }
    a[(size_t)lane * inter + r] = sg / (1.f + expf(-sg)) * su;
  }
}

template <int COLS, int S>
__global__ void k_dn(const bf* __restrict__ dn, const float* __restrict__ a, float* __restrict__ y, int n, int hid) {
  constexpr int NV = COLS / 256, NVS = NV / S;
  static_assert(NV % S == 0, "S tem de dividir NV");
  __shared__ float sm[6 * 64];
  int wid = threadIdx.x / 32, lane = threadIdx.x & 31, rl = wid / S, s = wid % S;
  int r = blockIdx.x * (blockDim.x / 32 / S) + rl;
  uint4 d[NVS];
  const uint4* dp = reinterpret_cast<const uint4*>(dn + (size_t)r * COLS);
#pragma unroll
  for (int v = 0; v < NVS; v++) d[v] = dp[(s * NVS + v) * 32 + lane];
  for (int p = 0; p < n; p++) {
    float t = 0;
#pragma unroll
    for (int v = 0; v < NVS; v++) t += dot8(d[v], a + (size_t)p * COLS + (s * NVS + v) * 256 + lane * 8);
    t = wsum(t);
    if (lane == 0) sm[wid * 64 + p] = t;
  }
  __syncthreads();
  if (s == 0 && lane < n) {
    float t = 0;
    for (int k = 0; k < S; k++) t += sm[(rl * S + k) * 64 + lane];
    y[(size_t)lane * hid + r] = t;
  }
}

struct Modelo { const char* nome; int hid, inter; };

// Variantes pré-declaradas (criterio.txt v1.2): 0 = v1.1 (S=1/1); 1 = S gate_up 2, down 2; 2 = gate_up com um vetor por warp (S = NV), down 2.
static int g_var = 0;
template <int HID, int S, int R>
static void lanca_gu(const Modelo& m, const bf* gu, const float* x, float* a, int n, cudaStream_t s) {
  k_gu<HID, S><<<m.inter / R, 32 * S * R, 0, s>>>(gu, x, a, n, m.inter);
}
template <int S, int R>
static void lanca_dn(const Modelo& m, const bf* dn, const float* a, float* y, int n, cudaStream_t s) {
  k_dn<512, S><<<m.hid / R, 32 * S * R, 0, s>>>(dn, a, y, n, m.hid);
}
static void lanca(const Modelo& m, const bf* gu, const bf* dn, const float* x, float* a, float* y, int n, cudaStream_t s) {
  if (m.hid == 1024) {
    if (g_var == 0) lanca_gu<1024, 1, 4>(m, gu, x, a, n, s);
    else if (g_var == 1) lanca_gu<1024, 2, 2>(m, gu, x, a, n, s);
    else lanca_gu<1024, 4, 1>(m, gu, x, a, n, s);
  } else {
    if (g_var == 0) lanca_gu<1536, 1, 4>(m, gu, x, a, n, s);
    else if (g_var == 1) lanca_gu<1536, 2, 2>(m, gu, x, a, n, s);
    else lanca_gu<1536, 6, 1>(m, gu, x, a, n, s);
  }
  CK(cudaGetLastError());
  if (g_var == 0) lanca_dn<1, 4>(m, dn, a, y, n, s); else lanca_dn<2, 2>(m, dn, a, y, n, s);
  CK(cudaGetLastError());
}

static double mediana(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
  bool rapido = false, so_n1 = false;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--rapido")) rapido = true;
    else if (!strcmp(argv[i], "--so-n1")) so_n1 = true;
    else if (!strncmp(argv[i], "--var=", 6)) g_var = atoi(argv[i] + 6);
    else { fprintf(stderr, "argumento desconhecido %s\n", argv[i]); return 2; }
  }
  if (g_var < 0 || g_var > 2) { fprintf(stderr, "--var=0|1|2\n"); return 2; }
  int reps = rapido ? 3 : 15, replays = rapido ? 2 : 10;
  cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, 0));
  char uuid[64];
  { const unsigned char* b = (const unsigned char*)pr.uuid.bytes;
    snprintf(uuid, sizeof uuid, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]); }
  printf("# dispositivo: %s uuid=%s\n", pr.name, uuid);
  printf("# modo=%s reps=%d replays=%d\n# variante=%d\n", so_n1 ? "so-n1" : rapido ? "rapido" : "completo", reps, replays, g_var);
  if (strcmp(uuid, "GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd")) { fprintf(stderr, "RECUSADO: não é a 4090 do experimento (uuid %s)\n", uuid); return 4; }
  cudaStream_t st; CK(cudaStreamCreate(&st));
  uint64_t rs = 0x9E3779B97F4A7C15ull;
  auto rnd = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return ((float)(rs >> 40) / 16777216.f - 0.5f) * 0.1f; };
  Modelo ms[2] = {{"1b", 1024, 512}, {"3b", 1536, 512}};
  bool falhou = false;
  for (auto& m : ms) {
    size_t ngu = (size_t)2 * m.inter * m.hid, ndn = (size_t)m.hid * m.inter;
    std::vector<bf> hgu(ngu * NEXP), hdn(ndn * NEXP);
    for (auto& v : hgu) v = __float2bfloat16(rnd());
    for (auto& v : hdn) v = __float2bfloat16(rnd());
    bf *dgu, *ddn; CK(cudaMalloc(&dgu, hgu.size() * 2)); CK(cudaMalloc(&ddn, hdn.size() * 2));
    CK(cudaMemcpy(dgu, hgu.data(), hgu.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddn, hdn.data(), hdn.size() * 2, cudaMemcpyHostToDevice));
    for (int n : {1, 2, 4, 8, 16, 32}) {
      if (so_n1 && n != 1) continue;
      std::vector<float> hx((size_t)n * m.hid);
      for (auto& v : hx) v = rnd() * 10.f;
      float *dx, *da, *dy, *dy2;
      CK(cudaMalloc(&dx, hx.size() * 4)); CK(cudaMalloc(&da, (size_t)n * m.inter * 4));
      CK(cudaMalloc(&dy, (size_t)n * m.hid * 4)); CK(cudaMalloc(&dy2, (size_t)n * m.hid * 4));
      CK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
      // correção: lote, serial e referência em double no expert 0
      lanca(m, dgu, ddn, dx, da, dy, n, st);
      for (int p = 0; p < n; p++) lanca(m, dgu, ddn, dx + (size_t)p * m.hid, da, dy2 + (size_t)p * m.hid, 1, st);
      CK(cudaStreamSynchronize(st));
      std::vector<float> yl((size_t)n * m.hid), ys((size_t)n * m.hid);
      CK(cudaMemcpy(yl.data(), dy, yl.size() * 4, cudaMemcpyDeviceToHost));
      CK(cudaMemcpy(ys.data(), dy2, ys.size() * 4, cudaMemcpyDeviceToHost));
      double mx = 0, e_ls = 0, e_ref = 0;
      bool finito = true;                       // qualquer NaN/inf nas saídas reprova (std::max ignoraria o NaN)
      for (float v : yl) finito = finito && std::isfinite(v);
      for (float v : ys) { finito = finito && std::isfinite(v); mx = std::max(mx, (double)fabsf(v)); }
      if (!finito || !(mx > 0)) mx = NAN;
      for (size_t i = 0; i < yl.size(); i++) e_ls = std::max(e_ls, (double)fabsf(yl[i] - ys[i]) / mx);
      if (!finito || std::isnan(mx)) e_ls = NAN;
      for (int p : {0, n - 1}) {
        std::vector<double> act(m.inter);
        for (int r = 0; r < m.inter; r++) {
          double g = 0, u = 0;
          for (int k = 0; k < m.hid; k++) {
            double xv = hx[(size_t)p * m.hid + k];
            g += (double)__bfloat162float(hgu[(size_t)r * m.hid + k]) * xv;
            u += (double)__bfloat162float(hgu[(size_t)(m.inter + r) * m.hid + k]) * xv;
          }
          act[r] = g / (1.0 + exp(-g)) * u;
        }
        for (int r = 0; r < m.hid; r++) {
          double s = 0;
          for (int k = 0; k < m.inter; k++) s += (double)__bfloat162float(hdn[(size_t)r * m.inter + k]) * act[k];
          e_ref = std::max(e_ref, fabs((double)yl[(size_t)p * m.hid + r] - s) / mx);
          if (!std::isfinite(s)) e_ref = NAN;
        }
      }
      // tempo: grafo com 64 experts em rodízio
      auto grava = [&](bool serial) {
        cudaGraph_t g; cudaGraphExec_t ex;
        CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        for (int e = 0; e < NEXP; e++) {
          const bf* gu = dgu + ngu * e; const bf* dn = ddn + ndn * e;
          if (!serial) lanca(m, gu, dn, dx, da, dy, n, st);
          else for (int p = 0; p < n; p++) lanca(m, gu, dn, dx + (size_t)p * m.hid, da + (size_t)p * m.inter, dy2 + (size_t)p * m.hid, 1, st);
        }
        CK(cudaStreamEndCapture(st, &g));
        CK(cudaGraphInstantiate(&ex, g, 0));
        return ex;
      };
      double tt[2];
      for (int serial = 0; serial < 2; serial++) {
        cudaGraphExec_t ex = grava(serial);
        for (int i = 0; i < 3; i++) CK(cudaGraphLaunch(ex, st));
        CK(cudaStreamSynchronize(st));
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        std::vector<double> v;
        for (int r = 0; r < reps; r++) {
          CK(cudaEventRecord(e0, st));
          for (int i = 0; i < replays; i++) CK(cudaGraphLaunch(ex, st));
          CK(cudaEventRecord(e1, st)); CK(cudaEventSynchronize(e1));
          float ms_; CK(cudaEventElapsedTime(&ms_, e0, e1));
          v.push_back(ms_ * 1000.0 / (replays * NEXP));
        }
        tt[serial] = mediana(v);
        CK(cudaGraphExecDestroy(ex));
      }
      printf("GPU %s n=%d lote_us=%.2f serial_us=%.2f err_lote_serial=%.2e err_ref=%.2e\n", m.nome, n, tt[0], tt[1], e_ls, e_ref);
      if (!finito || std::isnan(mx)) e_ref = NAN;
      if (!(e_ls <= 1e-3) || !(e_ref <= 1e-3)) falhou = true;
      cudaFree(dx); cudaFree(da); cudaFree(dy); cudaFree(dy2);
    }
    cudaFree(dgu); cudaFree(ddn);
  }
  if (falhou) { fprintf(stderr, "CORRECAO FALHOU\n"); return 3; }
  return 0;
}
