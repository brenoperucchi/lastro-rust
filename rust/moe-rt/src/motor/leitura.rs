//! A2 (textos longos), etapa 2: ler N ids em blocos de até `max_positions` posições, de uma chamada só.
//!
//! A rotina só encadeia o que o motor já tem (`next_published`, `enqueue_step`, `finish_token`) e o preparo do bloco, que quem conhece
//! os ids (o backend de modelo) implementa por [`PreparaPasso`]. Os logits da última posição NÃO saem daqui: ficam disponíveis pela
//! leitura separada do backend (`read_logits`). Valida tudo antes de qualquer efeito (cache, posição, motor); depois de um erro
//! no meio da leitura o motor fica encerrado e não há promessa sobre o cache.
//!
//! O que os testes deste módulo (backend falso) validam é a lógica: divisão em blocos, validação, ordem, posição e erro. Não validam
//! a correção numérica nem a velocidade da GPU.
use super::backend::Backend;
use super::engine::{Engine, Published};
use super::Error;

/// O que a leitura precisa do backend de modelo: os limites, a posição e o preparo de um bloco de ids.
pub trait PreparaPasso {
    type Error;
    /// O tamanho do vocabulário: todo id tem de ser menor que ele.
    fn vocab(&self) -> usize;
    /// O maior bloco que o preparo aceita (o `max_pos` do backend de modelo).
    fn max_positions(&self) -> usize;
    /// O comprimento máximo do cache (`max_len` da construção do modelo).
    fn max_len(&self) -> usize;
    /// A posição do próximo token na sequência do backend (fonte da validação de limite).
    fn posicao(&self) -> u64;
    /// Prepara os ids do bloco (1 a `max_positions` ids) para o próximo passo, com a stream ociosa.
    fn prepara(&self, ids: &[usize]) -> Result<(), Self::Error>;
}

#[derive(Debug)]
pub enum ErroLeitura<EB, EP> {
    /// Nenhum id.
    Vazia,
    /// O id no `indice` não cabe no vocabulário.
    IdForaDoVocabulario { indice: usize, id: usize, vocab: usize },
    /// `posicao + n` passa do `max_len`.
    CacheCheio { posicao: u64, n: usize, max_len: usize },
    /// `posicao + n` estoura a soma numérica.
    Estouro,
    /// O contador de posição do motor e o do backend divergem: nada foi feito.
    ContadoresDivergem { motor: u64, backend: u64 },
    /// O motor aceita blocos maiores do que o preparo do backend: recusado antes de qualquer passo, com o motor aberto.
    BackendComMenosPosicoes { motor: usize, backend: usize },
    /// O motor já estava encerrado: nada foi feito.
    MotorEncerrado,
    /// O preparo de um bloco falhou (o motor foi encerrado).
    Preparo(EP),
    /// O motor falhou ao obter a publicação ou num passo (o motor está encerrado).
    Motor(Error<EB>),
}

/// Os tamanhos dos blocos para `n` ids: blocos cheios de `max_positions` e, no fim, um resto de 1 a `max_positions - 1` quando `n` não é
/// múltiplo. `n = 0` dá lista vazia (entrada inválida para [`le_ids`]); `max_positions = 0` conta como 1.
pub fn plano_blocos(n: usize, max_positions: usize) -> Vec<usize> {
    let mp = max_positions.max(1);
    let mut v = vec![mp; n / mp];
    if n % mp != 0 {
        v.push(n % mp);
    }
    v
}

/// Toda a validação da entrada, antes de qualquer efeito.
pub fn valida<EB, EP>(ids: &[usize], vocab: usize, posicao: u64, max_len: usize) -> Result<(), ErroLeitura<EB, EP>> {
    if ids.is_empty() {
        return Err(ErroLeitura::Vazia);
    }
    if let Some((indice, &id)) = ids.iter().enumerate().find(|(_, &id)| id >= vocab) {
        return Err(ErroLeitura::IdForaDoVocabulario { indice, id, vocab });
    }
    let fim = posicao.checked_add(ids.len() as u64).ok_or(ErroLeitura::Estouro)?;
    if fim > max_len as u64 {
        return Err(ErroLeitura::CacheCheio { posicao, n: ids.len(), max_len });
    }
    Ok(())
}

/// Lê `ids` em blocos de até `eng.max_positions()`, a partir da posição atual da sequência. Devolve `Ok(ids.len())`; a posição do motor e a
/// do backend avançam exatamente `ids.len()`.
pub fn le_ids<B: Backend, P: PreparaPasso>(eng: &mut Engine<B>, prep: &P, ids: &[usize]) -> Result<usize, ErroLeitura<B::Error, P::Error>> {
    le_ids_com_pendente(eng, prep, ids, &mut None)
}

/// Como [`le_ids`], mas o primeiro bloco usa o `Published` já em mãos em `pendente` (guardado do pedido anterior, em motor reutilizado)
/// em vez de esperar um novo. Com `None` é idêntica a `le_ids`. O `pendente` só é consumido quando o primeiro bloco chega a ser publicado.
pub fn le_ids_com_pendente<B: Backend, P: PreparaPasso>(eng: &mut Engine<B>, prep: &P, ids: &[usize], pendente: &mut Option<Published>) -> Result<usize, ErroLeitura<B::Error, P::Error>> {
    valida(ids, prep.vocab(), prep.posicao(), prep.max_len())?;
    if eng.is_closed() {
        return Err(ErroLeitura::MotorEncerrado);
    }
    if eng.max_positions() > prep.max_positions() {
        return Err(ErroLeitura::BackendComMenosPosicoes { motor: eng.max_positions(), backend: prep.max_positions() });
    }
    let (motor, backend) = (eng.position(), prep.posicao());
    if motor != backend {
        return Err(ErroLeitura::ContadoresDivergem { motor, backend });
    }
    let mut ini = 0;
    for m in plano_blocos(ids.len(), eng.max_positions()) {
        // A publicação vem antes do preparo para que um motor que falha aqui não deixe o cache avançado por um bloco que não rodou.
        // `next_published` só consulta o canal e chama `relax`, sem tocar na stream de cálculo; a ordem testada na GPU (preparo e
        // depois `next_published`) e esta são equivalentes.
        let obtido = match pendente.take() {
            Some(p) => Ok(p),
            None => eng.next_published(),
        };
        let p = match obtido {
            Ok(p) => p,
            Err(e) => {
                eng.encerra();
                return Err(ErroLeitura::Motor(e));
            }
        };
        if let Err(e) = prep.prepara(&ids[ini..ini + m]) {
            eng.encerra();
            return Err(ErroLeitura::Preparo(e));
        }
        eng.enqueue_step(p, m).map_err(ErroLeitura::Motor)?;
        eng.finish_token().map_err(ErroLeitura::Motor)?;
        ini += m;
    }
    Ok(ids.len())
}
