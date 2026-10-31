// Decode forçado no llama.cpp: passa as conversas dos traços um token por vez em batch 1, como o m2a-cache do runtime
// (rust/moe-rt/src/bin/m2a-cache.rs), para comparar os dois motores com os mesmos tokens. O KV é limpo no começo de cada
// conversa e o cache MoE (--moe-cache-mib) persiste no processo entre as conversas. O tempo de um token é llama_decode +
// llama_get_logits_ith(-1), que sincroniza e entrega a linha inteira do vocabulário no host; o argmax e a margem ficam fora
// do cronômetro.
//
//   forcado --tokens <dir com conversa-N.txt> [--warm 3] [--convs 4,5,6,7] [--argmax <arquivo>] [--tokeniza <dir textos>]
//           [--logits <arquivo f32 cru, n_vocab por posição cronometrada>]
//           <flags do llama.cpp: -m, -ngl, -t, -c, --fit, --no-warmup, -ctk, -ctv, -fa, --cpu-moe, -ncmoe, --moe-cache-mib>
//
// --tokeniza só confere o tokenizador do GGUF contra os tokens dos traços (os textos dos traços), sem cronometrar nada;
// com --argmax <pasta>, grava ali os ids do GGUF de cada texto.
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int> parse_list(const std::string & s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(std::stoi(item));
        }
    }
    return out;
}

static std::vector<llama_token> read_tokens(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "não abriu %s\n", path.c_str());
        exit(1);
    }
    std::vector<llama_token> out;
    long v;
    while (f >> v) {
        out.push_back((llama_token) v);
    }
    return out;
}

static double mean(const std::vector<double> & v) {
    double s = 0;
    for (double x : v) s += x;
    return v.empty() ? NAN : s / v.size();
}

static double median(std::vector<double> v) {
    if (v.empty()) return NAN;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

int main(int argc, char ** argv) {
    std::string tokens_dir, argmax_path, textos_dir, logits_path;
    std::vector<int> warm = {3}, convs = {4, 5, 6, 7};
    std::vector<char *> rest = {argv[0]};
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "falta o valor de %s\n", a.c_str()); exit(1); }
            return argv[++i];
        };
        if (a == "--tokens") tokens_dir = next();
        else if (a == "--warm") warm = parse_list(next());
        else if (a == "--convs") convs = parse_list(next());
        else if (a == "--argmax") argmax_path = next();
        else if (a == "--tokeniza") textos_dir = next();
        else if (a == "--logits") logits_path = next();
        else rest.push_back(argv[i]);
    }
    if (tokens_dir.empty()) {
        fprintf(stderr, "uso: forcado --tokens <dir> [--warm 3] [--convs 4,5,6,7] [--argmax f] [--tokeniza dir] <flags do llama.cpp>\n");
        return 1;
    }

    common_params params;
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    if (!textos_dir.empty()) {
        // só o tokenizador: carrega o modelo sem contexto
        auto init = common_init_from_params(params, /*model_only=*/true);
        llama_model * model = init->model();
        if (!model) { fprintf(stderr, "modelo não carregou\n"); return 1; }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        bool all_ok = true;
        for (int c = 0; c < 8; c++) {
            std::ifstream f(textos_dir + "/conversa-" + std::to_string(c) + ".txt");
            std::stringstream ss;
            ss << f.rdbuf();
            std::string text = ss.str();
            std::vector<llama_token> got(text.size() + 16);
            int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), got.data(), (int) got.size(), false, true);
            if (n < 0) { fprintf(stderr, "tokenize falhou na conversa %d\n", c); return 1; }
            got.resize(n);
            auto ref = read_tokens(tokens_dir + "/conversa-" + std::to_string(c) + ".txt");
            size_t first = 0;
            while (first < got.size() && first < ref.size() && got[first] == ref[first]) first++;
            if (!argmax_path.empty()) {  // com --tokeniza, --argmax é a pasta onde gravar os ids do GGUF
                std::ofstream o(argmax_path + "/conversa-" + std::to_string(c) + ".txt");
                for (size_t k = 0; k < got.size(); k++) o << got[k] << (k + 1 < got.size() ? ' ' : '\n');
            }
            bool ok = got == ref;
            all_ok &= ok;
            printf("tokeniza conversa-%d: gguf %zu, traço %zu, %s", c, got.size(), ref.size(), ok ? "iguais" : "DIFEREM");
            if (!ok) printf(" (primeira diferença na posição %zu)", first);
            printf("\n");
        }
        printf("tokenizador: %s\n", all_ok ? "IGUAL nos 8 textos" : "DIFERENTE");
        llama_backend_free();
        return all_ok ? 0 : 2;
    }

    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) { fprintf(stderr, "modelo ou contexto não carregou\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_memory_t mem = llama_get_memory(ctx);

    FILE * am = nullptr;
    if (!argmax_path.empty()) {
        am = fopen(argmax_path.c_str(), "w");
        if (!am) { fprintf(stderr, "não abriu %s\n", argmax_path.c_str()); return 1; }
        fprintf(am, "# conversa posição token_seguinte argmax margem_top1_top2\n");
    }
    FILE * lf = nullptr;
    if (!logits_path.empty()) {
        lf = fopen(logits_path.c_str(), "wb");
        if (!lf) { fprintf(stderr, "não abriu %s\n", logits_path.c_str()); return 1; }
    }

    // faixas de posição iguais às do m2a-cache
    const int bucket_lo[] = {0, 256, 512, 1024};
    const int bucket_hi[] = {256, 512, 1024, 1 << 30};
    const char * bucket_name[] = {"0-256", "256-512", "512-1024", "1024+"};

    auto run_conv = [&](int c, bool timed, std::vector<double> & ms, std::vector<double> * by_pos) -> bool {
        auto toks = read_tokens(tokens_dir + "/conversa-" + std::to_string(c) + ".txt");
        llama_memory_clear(mem, true);
        for (size_t t = 0; t < toks.size(); t++) {
            llama_token tok = toks[t];
            auto t0 = std::chrono::steady_clock::now();
            int rc = llama_decode(ctx, llama_batch_get_one(&tok, 1));
            const float * lg = rc == 0 ? llama_get_logits_ith(ctx, -1) : nullptr;
            double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (rc != 0 || !lg) {
                fprintf(stderr, "llama_decode falhou (rc %d) na conversa %d, posição %zu\n", rc, c, t);
                return false;
            }
            for (int v = 0; v < n_vocab; v++) {  // fora do cronômetro: logits não finitos param a execução (rodada 43)
                if (!std::isfinite(lg[v])) {
                    fprintf(stderr, "logit não finito na conversa %d, posição %zu, token %d\n", c, t, v);
                    return false;
                }
            }
            if (!timed) continue;
            ms.push_back(dt);
            for (int b = 0; b < 4; b++) {
                if ((int) t >= bucket_lo[b] && (int) t < bucket_hi[b]) by_pos[b].push_back(dt);
            }
            if (lf) fwrite(lg, sizeof(float), n_vocab, lf);
            if (am) {
                int best = 0, second = -1;
                for (int v = 1; v < n_vocab; v++) {
                    if (lg[v] > lg[best]) { second = best; best = v; }
                    else if (second < 0 || lg[v] > lg[second]) second = v;
                }
                long nxt = t + 1 < toks.size() ? (long) toks[t + 1] : -1;
                fprintf(am, "%d %zu %ld %d %.4f\n", c, t, nxt, best, lg[best] - lg[second]);
            }
        }
        return true;
    };

    std::vector<double> scratch;
    std::vector<double> scratch_pos[4];
    for (int c : warm) {
        if (!run_conv(c, false, scratch, scratch_pos)) return 1;
    }
    std::vector<double> all, by_pos[4];
    printf("forcado: %s; conversas", params.model.path.c_str());
    for (int c : convs) printf(" %d", c);
    printf("; aquecimento");
    for (int c : warm) printf(" %d", c);
    printf("; threads %d; n_ctx %u\n", params.cpuparams.n_threads, llama_n_ctx(ctx));
    for (int c : convs) {
        std::vector<double> ms;
        if (!run_conv(c, true, ms, by_pos)) return 1;
        printf("conversa %d: %zu tokens, ms/tok %.3f, mediana %.3f\n", c, ms.size(), mean(ms), median(ms));
        all.insert(all.end(), ms.begin(), ms.end());
    }
    printf("total: %zu tokens, ms/tok %.3f, mediana %.3f, tok/s %.1f\n", all.size(), mean(all), median(all), 1000.0 / mean(all));
    printf("mediana por posição:");
    for (int b = 0; b < 4; b++) printf(" %s %.3f (%zu)", bucket_name[b], median(by_pos[b]), by_pos[b].size());
    printf("\n");
    fflush(stdout);
    if (am) fclose(am);
    if (lf) fclose(lf);
    init.reset();  // o destrutor do cache MoE imprime hits e hit rate no stderr
    llama_backend_free();
    return 0;
}
