//! B0: o compilador recusa as violações do protocolo do cache (cada caso em `ui/` tem um gêmeo que compila em `ui-pass/`),
//! e o `unsafe` do cache fica só no núcleo.
//!
//! As mensagens em `tests/ui/*.stderr` fazem parte do teste e foram geradas com rustc 1.97.0 (2d8144b78 2026-07-07); outra
//! versão pode mudar o texto e pedir `TRYBUILD=overwrite`, conferindo que cada caso continua falhando pelo mesmo motivo.

#[test]
fn protocolo_do_cache_recusado_pelo_compilador() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
    t.pass("tests/ui-pass/*.rs");
}

/// `forbid` não serve: o `pub mod core` precisa do `allow`. Então `deny` mais este teste, que recusa qualquer outro atributo
/// que mencione `unsafe_code` em `cache.rs` (um `allow(dead_code, unsafe_code)` ou um `expect(unsafe_code)` inclusive).
#[test]
fn cache_proibe_unsafe_fora_do_nucleo() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cache.rs")).unwrap();
    let lines: Vec<&str> = src.lines().map(str::trim).collect();
    let attrs: Vec<&str> = lines.iter().copied().filter(|l| l.starts_with('#') && l.contains("unsafe_code")).collect();
    assert_eq!(attrs, ["#![deny(unsafe_code)]", "#[allow(unsafe_code)]"], "atributos de unsafe_code em cache.rs mudaram");
    let next = lines.iter().skip_while(|l| **l != "#[allow(unsafe_code)]").nth(1).copied();
    assert_eq!(next, Some("pub mod core;"), "o allow(unsafe_code) não está no pub mod core");
}
