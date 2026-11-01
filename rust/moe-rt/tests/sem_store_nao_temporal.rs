//! A1.3b: a condição do plano (A1.0, rodada 54) para o `y` escrito pelo `Pool` sem `sfence` antes da flag: nenhuma escrita usa
//! store não temporal, que o `Release` da flag não ordena. Esta é uma varredura TEXTUAL de `rust/moe-rt/src`, sem exceção de
//! arquivo: pega os intrínsecos e as instruções das famílias `stream`, `movnt`, `maskmov` (a forma não temporal; o
//! `vmaskmovps` do AVX é um store mascarado comum e não entra), `movdir` e `directstore`, em qualquer caixa. Não é prova sobre
//! código gerado nem sobre dependências; se alguém introduzir um desses stores, a flag passa a exigir um `sfence` na thread
//! escritora antes de entregar a conclusão ao `Pool`, e este teste obriga a decidir.
use std::path::Path;

const PADROES: &[&str] = &[
    "_mm_stream", "_mm256_stream", "_mm512_stream", "stream_si", "stream_ps", "stream_pd", "movnt", "_mm_maskmove", "maskmovdqu", "maskmovq",
    "movdir", "_directstore", "non_temporal", "nontemporal",
];

/// Os padrões achados em `texto` (comparação sem distinguir caixa).
fn achados(texto: &str) -> Vec<&'static str> {
    let t = texto.to_ascii_lowercase();
    PADROES.iter().copied().filter(|p| t.contains(p)).collect()
}

fn varre(dir: &Path, saida: &mut Vec<String>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            varre(&p, saida);
        } else if p.extension().is_some_and(|x| x == "rs") {
            for a in achados(&std::fs::read_to_string(&p).unwrap()) {
                saida.push(format!("{}: {a}", p.display()));
            }
        }
    }
}

#[test]
fn nenhum_store_nao_temporal_em_src() {
    let mut saida = Vec::new();
    varre(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut saida);
    assert!(saida.is_empty(), "store não temporal no código: {saida:?}");
}

/// O detector em si, sem plantar nada no projeto: cada forma conhecida é acusada, e o store mascarado comum do AVX não.
#[test]
fn detector_acusa_as_formas_nao_temporais_e_poupa_o_vmaskmov() {
    for ruim in [
        "_mm_stream_ps(p, v)",
        "_mm256_stream_si256(p, v)",
        "core::arch::x86_64::_mm_maskmoveu_si128(a, m, p)",
        "_mm_maskmove_si64(a, m, p)",
        "asm!(\"maskmovdqu xmm0, xmm1\")",
        "asm!(\"MOVNTDQ [rdi], xmm0\")",
        "asm!(\"movntps [rdi], xmm0\")",
        "asm!(\"movdiri [rdi], rax\")",
        "_directstoreu_u64(p, v)",
        "#[nontemporal]",
    ] {
        assert!(!achados(ruim).is_empty(), "o detector não acusou: {ruim}");
    }
    for bom in ["_mm256_maskstore_ps(p, m, v)", "asm!(\"vmaskmovps [rdi], ymm1, ymm0\")", "core::ptr::write_volatile(p, v)", "movaps"] {
        assert!(achados(bom).is_empty(), "o detector acusou à toa: {bom}");
    }
}
