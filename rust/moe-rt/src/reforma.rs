//! A2, plano de desempenho das cópias (O1, política de erro A): a lógica "tira o tensor do campo, reforma, empresta ao lançador, devolve na
//! forma original" fica aqui, genérica sobre um traço mínimo ([`Reformavel`]), para ser testada em CPU com um tensor falso. O adaptador
//! do `Tensor<f32>` do cutile e o uso estão em `cuda_dense.rs`.
//!
//! Política A (decisão do Breno, 11/11 15:23): em qualquer erro o tensor volta ao campo na forma original antes de o erro propagar. Se a
//! restauração falhar, o dono é mantido e a stream é esperada ANTES de qualquer descarte (nunca liberar primeiro e sincronizar depois);
//! o campo fica vazio e o próximo uso acusa "buffer ausente".

/// O que a lógica precisa de um tensor: a forma e a reforma que, ao falhar, devolve o dono quando ainda o tem.
pub(crate) trait Reformavel: Sized {
    type Erro;
    fn forma(&self) -> Vec<usize>;
    /// Reforma sem mover dado. Em erro devolve o dono (`Some`) com a forma inalterada; `None` só se a implementação já o perdeu
    /// (o adaptador real confere antes de consumir o tensor, então não chega lá).
    /// Contrato de vida útil: `reformar` NUNCA solta o tensor sem antes ele estar seguro contra kernels em voo; devolver `None` só vale se
    /// a implementação já esperou a stream (ou se não há trabalho enfileirado sobre o tensor). O ramo `None` de `com_forma` espera de
    /// novo por segurança, mas não pode consertar um descarte que já aconteceu dentro de `reformar`.
    fn reformar(self, forma: &[usize]) -> Result<Self, (Option<Self>, Self::Erro)>;
}

/// Tira o tensor de `slot`, reforma para `forma`, chama `usa` com ele emprestado e o devolve a `slot` na forma original.
/// `espera` bloqueia até a stream terminar; só é chamada quando a restauração falha.
/// O erro de `usa` tem prioridade sobre o da restauração.
pub(crate) fn com_forma<T, R, E>(slot: &mut Option<T>, forma: &[usize], ausente: impl FnOnce() -> E, espera: impl FnOnce(), usa: impl FnOnce(&mut T) -> Result<R, E>) -> Result<R, E>
where
    T: Reformavel,
    E: From<T::Erro>,
{
    let t = slot.take().ok_or_else(ausente)?;
    let original = t.forma();
    let mut t = match t.reformar(forma) {
        Ok(t) => t,
        Err((dono, e)) => {
            match dono {
                Some(t) => *slot = Some(t),
                // dono perdido dentro de `reformar` (fora do contrato do adaptador real): só resta esperar a stream
                None => espera(),
            }
            return Err(e.into());
        }
    };
    let r = usa(&mut t);
    match t.reformar(&original) {
        Ok(t) => {
            *slot = Some(t);
            r
        }
        Err((dono, e)) => {
            // O dono segue vivo até a espera; só depois dela o tensor cai.
            espera();
            drop(dono);
            match r {
                Err(e0) => Err(e0),
                Ok(_) => Err(e.into()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    type Log = Rc<RefCell<Vec<String>>>;

    /// Tensor falso: forma, um registro de eventos e quais chamadas de `reformar` falham (contadas por tensor, a partir de 1).
    struct Fake {
        forma: Vec<usize>,
        log: Log,
        falha_em: Rc<RefCell<Vec<usize>>>,
        chamadas: usize,
        perde_dono: bool,
    }

    impl Drop for Fake {
        fn drop(&mut self) {
            self.log.borrow_mut().push("drop".into());
        }
    }

    impl Reformavel for Fake {
        type Erro = String;
        fn forma(&self) -> Vec<usize> {
            self.forma.clone()
        }
        fn reformar(mut self, forma: &[usize]) -> Result<Self, (Option<Self>, String)> {
            self.chamadas += 1;
            if self.falha_em.borrow().contains(&self.chamadas) {
                let perde = self.perde_dono;
                if perde {
                    return Err((None, "reforma falhou (dono perdido)".into()));
                }
                return Err((Some(self), "reforma falhou".into()));
            }
            self.forma = forma.to_vec();
            Ok(self)
        }
    }

    fn novo(log: &Log, falha_em: &[usize], perde_dono: bool) -> Option<Fake> {
        Some(Fake { forma: vec![8], log: log.clone(), falha_em: Rc::new(RefCell::new(falha_em.to_vec())), chamadas: 0, perde_dono })
    }

    /// O mesmo contrato de cima sem restaurar quando `usa` falha (mutante 1) ou soltando o tensor antes de esperar (mutante 2).
    fn com_forma_mutante_sem_restaurar<T: Reformavel, R, E: From<T::Erro>>(slot: &mut Option<T>, forma: &[usize], ausente: impl FnOnce() -> E, espera: impl FnOnce(), usa: impl FnOnce(&mut T) -> Result<R, E>) -> Result<R, E> {
        let t = slot.take().ok_or_else(ausente)?;
        let original = t.forma();
        let mut t = match t.reformar(forma) {
            Ok(t) => t,
            Err((dono, e)) => {
                if let Some(t) = dono {
                    *slot = Some(t);
                }
                return Err(e.into());
            }
        };
        let r = usa(&mut t);
        if r.is_err() {
            return r; // BUG: o tensor é solto e o campo fica vazio
        }
        match t.reformar(&original) {
            Ok(t) => {
                *slot = Some(t);
                r
            }
            Err((dono, e)) => {
                espera();
                drop(dono);
                r.and(Err(e.into()))
            }
        }
    }

    fn com_forma_mutante_libera_antes<T: Reformavel, R, E: From<T::Erro>>(slot: &mut Option<T>, forma: &[usize], ausente: impl FnOnce() -> E, espera: impl FnOnce(), usa: impl FnOnce(&mut T) -> Result<R, E>) -> Result<R, E> {
        let t = slot.take().ok_or_else(ausente)?;
        let original = t.forma();
        let mut t = match t.reformar(forma) {
            Ok(t) => t,
            Err((dono, e)) => {
                if let Some(t) = dono {
                    *slot = Some(t);
                }
                return Err(e.into());
            }
        };
        let r = usa(&mut t);
        match t.reformar(&original) {
            Ok(t) => {
                *slot = Some(t);
                r
            }
            Err((dono, e)) => {
                drop(dono); // BUG: libera antes de esperar a stream
                espera();
                r.and(Err(e.into()))
            }
        }
    }

    /// Roda todos os cenários contra a função `$f` e devolve a primeira violação do contrato (ou `Ok`).
    macro_rules! verifica {
        ($f:path) => {{
            let r: Result<(), String> = (|| {
                // sucesso: o lançador vê a forma nova, o campo volta na original, sem espera nem descarte
                let log: Log = Rc::default();
                let mut slot = novo(&log, &[], false);
                let visto = RefCell::new(vec![]);
                let r = $f(&mut slot, &[2, 4], || "ausente".to_string(), || log.borrow_mut().push("espera".into()), |t: &mut Fake| {
                    visto.borrow_mut().push(t.forma.clone());
                    Ok::<u8, String>(7)
                });
                if r != Ok(7) || *visto.borrow() != vec![vec![2, 4]] || slot.as_ref().map(|t| t.forma.clone()) != Some(vec![8]) || !log.borrow().is_empty() {
                    return Err(format!("sucesso: {r:?} {:?} {:?} log {:?}", visto.borrow(), slot.as_ref().map(|t| t.forma.clone()), log.borrow()));
                }
                // o lançador falha: o campo volta na forma original e o erro dele propaga
                let log: Log = Rc::default();
                let mut slot = novo(&log, &[], false);
                let r: Result<u8, String> = $f(&mut slot, &[2, 4], || "ausente".to_string(), || log.borrow_mut().push("espera".into()), |_t: &mut Fake| Err("lançamento falhou".to_string()));
                if r != Err("lançamento falhou".to_string()) || slot.as_ref().map(|t| t.forma.clone()) != Some(vec![8]) || !log.borrow().is_empty() {
                    return Err(format!("erro do lançador: {r:?} campo {:?} log {:?}", slot.as_ref().map(|t| t.forma.clone()), log.borrow()));
                }
                // a reforma de ida falha: o campo segue com o dono na forma original e o lançador não roda
                let log: Log = Rc::default();
                let mut slot = novo(&log, &[1], false);
                let rodou = RefCell::new(false);
                let r: Result<u8, String> = $f(&mut slot, &[2, 4], || "ausente".to_string(), || log.borrow_mut().push("espera".into()), |_t: &mut Fake| {
                    *rodou.borrow_mut() = true;
                    Ok(1)
                });
                if r.is_ok() || *rodou.borrow() || slot.as_ref().map(|t| t.forma.clone()) != Some(vec![8]) || !log.borrow().is_empty() {
                    return Err(format!("reforma de ida: {r:?} rodou {} campo {:?} log {:?}", rodou.borrow(), slot.as_ref().map(|t| t.forma.clone()), log.borrow()));
                }
                // a restauração falha com o dono em mãos: espera ANTES do descarte; o campo fica vazio; o erro é o da restauração
                let log: Log = Rc::default();
                let mut slot = novo(&log, &[2], false);
                let r: Result<u8, String> = $f(&mut slot, &[2, 4], || "ausente".to_string(), || log.borrow_mut().push("espera".into()), |_t: &mut Fake| Ok(1));
                if r != Err("reforma falhou".to_string()) || slot.is_some() || *log.borrow() != vec!["espera".to_string(), "drop".to_string()] {
                    return Err(format!("restauração falhou: {r:?} campo {} log {:?}", slot.is_some(), log.borrow()));
                }
                // restauração falha E o lançador falhou: vale o erro do lançador, mesma ordem espera depois drop
                let log: Log = Rc::default();
                let mut slot = novo(&log, &[2], false);
                let r: Result<u8, String> = $f(&mut slot, &[2, 4], || "ausente".to_string(), || log.borrow_mut().push("espera".into()), |_t: &mut Fake| Err("lançamento falhou".to_string()));
                if r != Err("lançamento falhou".to_string()) || slot.is_some() || *log.borrow() != vec!["espera".to_string(), "drop".to_string()] {
                    return Err(format!("duas falhas: {r:?} log {:?}", log.borrow()));
                }
                // restauração falha com o dono já perdido pelo adaptador: ao menos espera
                let log: Log = Rc::default();
                let mut slot = novo(&log, &[2], true);
                let r: Result<u8, String> = $f(&mut slot, &[2, 4], || "ausente".to_string(), || log.borrow_mut().push("espera".into()), |_t: &mut Fake| Ok(1));
                if r.is_ok() || slot.is_some() || !log.borrow().contains(&"espera".to_string()) {
                    return Err(format!("dono perdido: {r:?} log {:?}", log.borrow()));
                }
                // campo vazio: erro de ausência, o lançador não roda
                let mut slot: Option<Fake> = None;
                let rodou = RefCell::new(false);
                let r: Result<u8, String> = $f(&mut slot, &[2, 4], || "ausente".to_string(), || {}, |_t: &mut Fake| {
                    *rodou.borrow_mut() = true;
                    Ok(1)
                });
                if r != Err("ausente".to_string()) || *rodou.borrow() {
                    return Err(format!("campo vazio: {r:?}"));
                }
                Ok(())
            })();
            r
        }};
    }

    #[test]
    fn com_forma_cumpre_a_politica_a_em_todos_os_cenarios() {
        verifica!(com_forma).unwrap();
    }

    #[test]
    fn mutante_que_nao_restaura_no_erro_do_lancador_falha_o_teste() {
        let r = verifica!(com_forma_mutante_sem_restaurar);
        assert!(r.is_err(), "o teste tinha de reprovar o mutante que não restaura: {r:?}");
        assert!(r.unwrap_err().contains("erro do lançador"));
    }

    #[test]
    fn mutante_que_libera_antes_de_esperar_falha_o_teste() {
        let r = verifica!(com_forma_mutante_libera_antes);
        assert!(r.is_err(), "o teste tinha de reprovar o mutante que libera antes de esperar: {r:?}");
    }
}
