// Banda e latência de cópia com as duas GPUs (4090 e 3080 Ti), para a análise x8/x8 contra x16/x4.
// Device 0 e 1 seguem CUDA_VISIBLE_DEVICES (o roteiro fixa a ordem 4090, 3080 Ti pelos UUIDs).
// Mede, com memória pinned e cópias assíncronas numa stream, tempo do host do enqueue ao fim (mediana de 200 depois de 20):
//   - H2D e D2H em cada GPU, sozinha;
//   - H2D nas duas ao mesmo tempo (cada uma na sua thread de host, partida comum): soma de banda no complexo raiz;
//   - GPU0 -> GPU1 com cudaMemcpyPeerAsync sem habilitar P2P (o driver passa pelo host).
//   banda2 <KiB> [<KiB> ...]
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
using clk = std::chrono::steady_clock;
static const int N = 220, AQ = 20;

static double mediana(std::vector<double> t) { std::sort(t.begin(), t.end()); return t[t.size() / 2]; }

struct Buf { void * d; void * h; cudaStream_t s; };
static Buf aloca(int dev, size_t n) {
    Buf b; OK(cudaSetDevice(dev)); OK(cudaMalloc(&b.d, n)); OK(cudaMallocHost(&b.h, n)); OK(cudaStreamCreate(&b.s)); return b;
}

static std::vector<double> mede(int dev, Buf b, size_t n, cudaMemcpyKind k) {
    OK(cudaSetDevice(dev));
    std::vector<double> t;
    for (int i = 0; i < N; i++) {
        auto t0 = clk::now();
        if (k == cudaMemcpyHostToDevice) OK(cudaMemcpyAsync(b.d, b.h, n, k, b.s)); else OK(cudaMemcpyAsync(b.h, b.d, n, k, b.s));
        OK(cudaStreamSynchronize(b.s));
        double dt = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        if (i >= AQ) t.push_back(dt);
    }
    return t;
}

static void linha(const char * nome, size_t n, double ms) {
    printf("  %-34s %9.4f ms  %7.2f GB/s\n", nome, ms, n / ms / 1e6);
}

int main(int argc, char ** argv) {
    int ndev = 0; OK(cudaGetDeviceCount(&ndev));
    if (ndev < 2) { fprintf(stderr, "preciso de 2 GPUs visíveis\n"); return 1; }
    for (int d = 0; d < 2; d++) {
        cudaDeviceProp p; OK(cudaGetDeviceProperties(&p, d));
        printf("device %d: %s, pci %02x:%02x\n", d, p.name, p.pciBusID, p.pciDeviceID);
    }
    int pode01 = 0, pode10 = 0; OK(cudaDeviceCanAccessPeer(&pode01, 0, 1)); OK(cudaDeviceCanAccessPeer(&pode10, 1, 0));
    printf("cudaDeviceCanAccessPeer 0->1 %d, 1->0 %d\n", pode01, pode10);
    for (int a = 1; a < argc; a++) {
        size_t n = (size_t) (atof(argv[a]) * 1024);
        printf("%.0f KiB:\n", atof(argv[a]));
        Buf b0 = aloca(0, n), b1 = aloca(1, n);
        linha("H2D device 0 sozinha", n, mediana(mede(0, b0, n, cudaMemcpyHostToDevice)));
        linha("H2D device 1 sozinha", n, mediana(mede(1, b1, n, cudaMemcpyHostToDevice)));
        linha("D2H device 0 sozinha", n, mediana(mede(0, b0, n, cudaMemcpyDeviceToHost)));
        linha("D2H device 1 sozinha", n, mediana(mede(1, b1, n, cudaMemcpyDeviceToHost)));
        // as duas ao mesmo tempo: as threads andam juntas, volta a volta (barreira por contador)
        std::vector<double> t0v, t1v; std::atomic<int> barreira{0};
        auto corre = [&](int dev, Buf b, std::vector<double> * out) {
            OK(cudaSetDevice(dev));
            for (int i = 0; i < N; i++) {
                barreira.fetch_add(1); while (barreira.load() < 2 * (i + 1)) {}
                auto t0 = clk::now();
                OK(cudaMemcpyAsync(b.d, b.h, n, cudaMemcpyHostToDevice, b.s)); OK(cudaStreamSynchronize(b.s));
                double dt = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
                if (i >= AQ) out->push_back(dt);
            }
        };
        std::thread th0(corre, 0, b0, &t0v), th1(corre, 1, b1, &t1v); th0.join(); th1.join();
        double m0 = mediana(t0v), m1 = mediana(t1v);
        linha("H2D device 0 com a 1 junto", n, m0);
        linha("H2D device 1 com a 0 junto", n, m1);
        printf("  %-34s %7.2f GB/s\n", "soma das duas", n / m0 / 1e6 + n / m1 / 1e6);
        // GPU0 -> GPU1 sem P2P habilitado
        std::vector<double> tp;
        OK(cudaSetDevice(0));
        for (int i = 0; i < N; i++) {
            auto t0 = clk::now();
            OK(cudaMemcpyPeerAsync(b1.d, 1, b0.d, 0, n, b0.s)); OK(cudaStreamSynchronize(b0.s));
            double dt = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            if (i >= AQ) tp.push_back(dt);
        }
        linha("device 0 -> device 1 (peer, via host)", n, mediana(tp));
        for (Buf * b : {&b0, &b1}) { cudaFree(b->d); cudaFreeHost(b->h); cudaStreamDestroy(b->s); }
    }
    return 0;
}
