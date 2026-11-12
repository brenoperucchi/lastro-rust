//! A2, O3 (escrever direto no cache K/V), teste SÓ DE COMPILAÇÃO (sem GPU, sem lançar nada) dos dois
//! contratos do plano (`plano-o3-cache-direto-20261112.md`, seção 3, O3-A):
//! - `rope_kv`: partição mapeada `[1, head_dim]` do cache K, `iter_indices_within` na linha `cur`, as duas metades unidas com `cat` ao
//!   longo das colunas e UM `store` no índice cunhado;
//! - `gemv_row`: partição mapeada `[1, BM]` do cache V, um `store` de tile `[1, BM]` no índice cunhado, com o índice de coluna vindo do
//!   índice cunhado (e não do id do bloco).
//! Este é o teste dos kernels de PRODUÇÃO (`gpu::kernels::{rope_kv, gemv_row}`), nas formas do 1B e do 3B.
//! O compilador do cutile só gera o IR (alvo fixo `sm_120`); nada aqui abre um contexto CUDA.
use moe_rt::gpu::kernels::__module_ast_self;
use cutile::compile_api::KernelCompiler;
use cutile::cutile_compiler::compiler::utils::CompileOptions;
use cutile::cutile_compiler::hints::DebugInfoLevel;

fn compila(nome: &str, generics: &[&str], strides: &[(&str, &[i32])]) -> Result<String, String> {
    let g: Vec<String> = generics.iter().map(|s| s.to_string()).collect();
    KernelCompiler::new(__module_ast_self, "kernels", nome)
        .target("sm_120")
        .generics(g)
        .strides(strides)
        .options(CompileOptions::default().debug_info(DebugInfoLevel::None))
        .compile()
        .map(|a| a.ir_text())
        .map_err(|e| e.to_string())
}

/// Pilha maior, como nos testes do cutile (o compilador recursa fundo).
fn com_pilha<F: FnOnce() + Send + 'static>(f: F) {
    std::thread::Builder::new().stack_size(8_000_000).spawn(f).unwrap().join().unwrap();
}

#[test]
fn o3_kernel_rope_kv_de_producao_compila() {
    com_pilha(|| {
        // 1B e 3B: head_dim 64 (half 32); o cache é `[max_len, kvd]`, a partição `[1, head_dim]`, map `[1,1]`.
        let ir = compila("rope_kv", &["32", "64", "1", "1"], &[("kc", &[512, 1]), ("x", &[32, 1]), ("cs", &[32, 1])]).expect("rope_kv deve compilar");
        assert!(ir.contains("cat"), "o IR do rope_kv deveria ter o cat:\n{ir}");
    });
}

#[test]
fn o3_kernel_gemv_row_de_producao_compila() {
    com_pilha(|| {
        // gemv de V: BM = 2, BK = tile_k(hidden); hidden 1024 (1B) e 1536 (3B, tile_k = 512).
        compila("gemv_row", &["2", "1024", "1024", "1", "1"], &[("vc", &[512, 1]), ("w", &[1024, 1]), ("x", &[1])]).expect("gemv_row (1B) deve compilar");
        compila("gemv_row", &["2", "512", "1536", "1", "1"], &[("vc", &[512, 1]), ("w", &[1536, 1]), ("x", &[1])]).expect("gemv_row (3B) deve compilar");
    });
}
