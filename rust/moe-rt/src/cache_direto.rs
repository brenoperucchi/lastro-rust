//! O3 (escrever direto no cache K/V): a parte pura, sem GPU, do que os kernels `rope_kv` e `gemv_row` precisam do lado do host.
//!
//! Os dois kernels recebem o cache da camada (`[max_len, kv·D]`) como partição mapeada e a linha `cur` como escalar `i32`; fora de
//! `0..max_len` o `iter_indices_within` leria ou escreveria fora do tensor (os kernels têm `unchecked_accesses`), então o lado do host
//! confere antes do primeiro lançamento e devolve erro em vez de lançar.

/// A linha `cur` do cache como início `i32` do `iter_indices_within`, ou o motivo da recusa. Recusa `cur >= max_len` (linha fora do
/// cache) e qualquer valor que não caiba em `i32`.
pub fn inicio_da_linha(cur: usize, max_len: usize) -> Result<i32, &'static str> {
    if cur >= max_len {
        return Err("posição fora do cache KV (cur >= max_len)");
    }
    i32::try_from(cur).map_err(|_| "posição do cache KV não cabe em i32")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aceita_a_primeira_e_a_ultima_linha() {
        assert_eq!(inicio_da_linha(0, 512), Ok(0));
        assert_eq!(inicio_da_linha(511, 512), Ok(511));
    }

    #[test]
    fn recusa_linha_igual_ou_maior_que_o_cache() {
        assert!(inicio_da_linha(512, 512).is_err());
        assert!(inicio_da_linha(513, 512).is_err());
        assert!(inicio_da_linha(0, 0).is_err());
        assert!(inicio_da_linha(usize::MAX, usize::MAX).is_err());
    }

    #[test]
    fn recusa_o_que_nao_cabe_em_i32() {
        let grande = i32::MAX as usize + 1;
        assert!(inicio_da_linha(grande, grande + 1).is_err());
        assert_eq!(inicio_da_linha(i32::MAX as usize, grande + 1), Ok(i32::MAX));
    }
}
