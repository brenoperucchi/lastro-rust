//! A2 (textos longos), etapa 3: as regras PURAS da correção da leitura em blocos nos modelos reais (sem GPU, testadas em CPU). Critério:
//! `qual-runs/A2-LOTE-20261103/criterio-leitura-correcao.txt` (tolerâncias aprovadas pelo Breno em 07/11/2026, congeladas antes de qualquer dado real).
//!
//! Regra única de rótulos, em três níveis (NÃO é o rótulo de velocidade do E4b): (1) validade da execução por célula, que dá SEM PROVA
//! e impede ler as comparações como prova; (2) comparação numérica só em célula válida, OK ou FALHA NUMÉRICA (valor não finito em qualquer
//! vetor é FALHA NUMÉRICA); (3) rótulo da célula e do conjunto, com REPROVA vencendo SEM PROVA.
use crate::e4b_regra::{argmax, erro_l2_rel, erro_max_rel};

/// Limite entre modos (blocos de 8 contra um por vez, e continuação contra uma leitura só): norma máxima relativa, logits e KV.
pub const TOL_ENTRE_MODOS: f64 = 1e-5;
/// Limite dos logits contra a passada residente: erro L2 relativo (e argmax igual).
pub const TOL_LOGITS_CONTRA_X: f64 = 1e-4;
/// O `max_len` de construção da etapa 3 (múltiplo de `ATTN_BN` = 64).
pub const MAX_LEN: usize = 2048;
/// Cenário C: ids lidos de uma vez.
pub const N_C: usize = 500;
/// Cenário A: ids lidos de uma vez.
pub const N_A: usize = 2003;
/// Continuação: duas leituras seguidas (soma `N_A`).
pub const N_CONT: [usize; 2] = [1205, 798];
/// Blocos por passo do modo de blocos.
pub const BLOCO: usize = 8;
/// Faixas de posição do perfil de erro do KV (só descrição): `[ini, fim)`.
pub const FAIXAS: [(usize, usize); 4] = [(0, 64), (64, 512), (512, 1024), (1024, 2048)];

/// A fonte dos ids de cada célula: `(conversa do cenário C, conversas concatenadas do cenário A)`, dentro da pasta de traços do modelo.
/// Hipótese a confirmar na execução (e no `--ensaio`): os tamanhos bastam e todo id cabe no vocabulário.
pub fn fontes_de_ids(nome: &str) -> Option<(usize, [usize; 2])> {
    match nome {
        "1b" => Some((4, [0, 1])),
        "3b" => Some((2, [0, 1])),
        _ => None,
    }
}

/// Monta `n` ids a partir das conversas dadas, na ordem (concatenadas). Recusa se faltarem ids ou se algum for negativo ou não couber no vocabulário.
pub fn monta_ids(conversas: &[Vec<i32>], n: usize, vocab: usize) -> Result<Vec<usize>, String> {
    let total: usize = conversas.iter().map(|c| c.len()).sum();
    if total < n {
        return Err(format!("as conversas têm {total} ids, menos que os {n} pedidos"));
    }
    let mut v = Vec::with_capacity(n);
    for &t in conversas.iter().flatten().take(n) {
        if t < 0 || t as usize >= vocab {
            return Err(format!("id {t} fora do vocabulário ({vocab})"));
        }
        v.push(t as usize);
    }
    Ok(v)
}

/// O resto da divisão em blocos de `BLOCO` (0 = sem resto).
pub fn resto(n: usize) -> usize {
    n % BLOCO
}

/// O resultado de uma comparação numérica numa célula VÁLIDA.
#[derive(Clone, Debug, PartialEq)]
pub struct Comparacao {
    pub nome: String,
    pub erro: f64,
    pub limite: f64,
    /// Algum vetor tinha valor não finito (ou tamanho diferente): o erro é infinito e a comparação é FALHA NUMÉRICA.
    pub nao_finito: bool,
    /// Só nas comparações de logits contra a passada residente: o argmax bateu.
    pub argmax_ok: bool,
    pub ok: bool,
}

fn tem_nao_finito(a: &[f32], b: &[f32]) -> bool {
    a.is_empty() || a.len() != b.len() || a.iter().chain(b).any(|v| !v.is_finite())
}

/// Comparação entre modos: norma máxima relativa contra `TOL_ENTRE_MODOS`. Não finito, vazio ou de tamanho diferente é FALHA NUMÉRICA.
pub fn compara_entre_modos(nome: &str, got: &[f32], want: &[f32]) -> Comparacao {
    let nf = tem_nao_finito(got, want);
    let erro = erro_max_rel(got, want);
    Comparacao { nome: nome.into(), erro, limite: TOL_ENTRE_MODOS, nao_finito: nf, argmax_ok: true, ok: !nf && erro <= TOL_ENTRE_MODOS }
}

/// Logits contra a passada residente: erro L2 relativo contra `TOL_LOGITS_CONTRA_X` e argmax igual.
pub fn compara_logits_x(nome: &str, got: &[f32], x: &[f32]) -> Comparacao {
    let nf = tem_nao_finito(got, x);
    let erro = erro_l2_rel(got, x);
    let am = !nf && argmax(got) == argmax(x);
    Comparacao { nome: nome.into(), erro, limite: TOL_LOGITS_CONTRA_X, nao_finito: nf, argmax_ok: am, ok: !nf && erro <= TOL_LOGITS_CONTRA_X && am }
}

/// O KV (K e V) de todas as camadas, comparado entre modos; o erro da comparação é o maior de todas as camadas.
pub fn compara_kv(nome: &str, got: &[(Vec<f32>, Vec<f32>)], want: &[(Vec<f32>, Vec<f32>)]) -> Comparacao {
    if got.len() != want.len() || got.is_empty() {
        return Comparacao { nome: nome.into(), erro: f64::INFINITY, limite: TOL_ENTRE_MODOS, nao_finito: true, argmax_ok: true, ok: false };
    }
    let mut c = compara_entre_modos(nome, &got[0].0, &want[0].0);
    for ((gk, gv), (wk, wv)) in got.iter().zip(want) {
        for (g, w) in [(gk, wk), (gv, wv)] {
            let x = compara_entre_modos(nome, g, w);
            c.nao_finito |= x.nao_finito;
            c.erro = c.erro.max(x.erro);
            c.ok &= x.ok;
        }
    }
    c.ok &= !c.nao_finito && c.erro <= TOL_ENTRE_MODOS;
    c
}

/// Perfil do erro do KV por faixa de posições, SÓ DESCRIÇÃO (não decide nada): para cada faixa, `max|got - want|` das posições da faixa,
/// dividido pela norma máxima de `want` na camada (a mesma normalização de `erro_max_rel`). `kvd` é o tamanho de uma posição.
pub fn perfil_kv(got: &[(Vec<f32>, Vec<f32>)], want: &[(Vec<f32>, Vec<f32>)], kvd: usize) -> Vec<f64> {
    let mut perfil = vec![0f64; FAIXAS.len()];
    for ((gk, gv), (wk, wv)) in got.iter().zip(want) {
        for (g, w) in [(gk, wk), (gv, wv)] {
            if kvd == 0 || g.len() != w.len() {
                return vec![f64::INFINITY; FAIXAS.len()];
            }
            let norma = w.iter().fold(0f64, |m, v| m.max(v.abs() as f64)).max(1e-6);
            for (i, &(ini, fim)) in FAIXAS.iter().enumerate() {
                let (a, b) = (ini * kvd, (fim * kvd).min(w.len()));
                if a >= b {
                    continue;
                }
                let e = g[a..b].iter().zip(&w[a..b]).fold(0f64, |m, (x, y)| if x.is_finite() && y.is_finite() { m.max((*x as f64 - *y as f64).abs()) } else { f64::INFINITY }) / norma;
                perfil[i] = perfil[i].max(e);
            }
        }
    }
    perfil
}

/// Os contadores de um modo (experts pedidos, acertos na GPU, CPU).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Contadores {
    pub needed: u64,
    pub hits: u64,
    pub cpu: u64,
}

/// A validade dos contadores: `hits > 0` e `cpu > 0` nos dois modos e iguais entre eles.
pub fn contadores_validos(m8: Contadores, m1: Contadores) -> bool {
    m8.hits > 0 && m8.cpu > 0 && m1.hits > 0 && m1.cpu > 0 && m8 == m1
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rotulo {
    Passa,
    Reprova,
    SemProva,
}

impl Rotulo {
    pub fn nome(self) -> &'static str {
        match self {
            Rotulo::Passa => "PASSA",
            Rotulo::Reprova => "REPROVA",
            Rotulo::SemProva => "SEM PROVA",
        }
    }
}

/// O rótulo de uma célula: `valida = false` (qualquer condição de validade do nível 1) dá SEM PROVA e as comparações não são lidas;
/// senão REPROVA se houver ao menos uma FALHA NUMÉRICA (comparação não OK), senão PASSA. Sem nenhuma comparação, nada se prova: SEM PROVA.
pub fn rotulo_celula(valida: bool, comparacoes: &[Comparacao]) -> Rotulo {
    if !valida || comparacoes.is_empty() {
        return Rotulo::SemProva;
    }
    if comparacoes.iter().any(|c| !c.ok) {
        Rotulo::Reprova
    } else {
        Rotulo::Passa
    }
}

/// O rótulo do conjunto: REPROVA se alguma célula REPROVA (falha numérica em execução válida vale mesmo que outras fiquem sem prova);
/// senão SEM PROVA se alguma estiver sem prova (ou se não houver célula); senão PASSA. Integridade quebrada: todas as células SEM PROVA.
pub fn rotulo_conjunto(celulas: &[Rotulo], integro: bool) -> Rotulo {
    if !integro {
        return Rotulo::SemProva;
    }
    if celulas.contains(&Rotulo::Reprova) {
        Rotulo::Reprova
    } else if celulas.is_empty() || celulas.contains(&Rotulo::SemProva) {
        Rotulo::SemProva
    } else {
        Rotulo::Passa
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(n: &str) -> Comparacao {
        Comparacao { nome: n.into(), erro: 0.0, limite: 1.0, nao_finito: false, argmax_ok: true, ok: true }
    }
    fn ruim(n: &str) -> Comparacao {
        Comparacao { ok: false, erro: 2.0, ..ok(n) }
    }

    #[test]
    fn constantes_do_cenario_batem_com_o_recorte() {
        assert_eq!(MAX_LEN % 64, 0);
        assert!(N_A <= MAX_LEN && N_CONT.iter().sum::<usize>() == N_A);
        // Os restos "significativos" do recorte: 4, 3, 5 e 6; nenhum é zero.
        assert_eq!([resto(N_C), resto(N_A), resto(N_CONT[0]), resto(N_CONT[1])], [4, 3, 5, 6]);
        assert_eq!(FAIXAS.last().unwrap().1, MAX_LEN);
        assert_eq!(TOL_ENTRE_MODOS, 1e-5);
        assert_eq!(TOL_LOGITS_CONTRA_X, 1e-4);
    }

    #[test]
    fn fontes_so_para_1b_e_3b() {
        assert_eq!(fontes_de_ids("1b"), Some((4, [0, 1])));
        assert_eq!(fontes_de_ids("3b"), Some((2, [0, 1])));
        assert_eq!(fontes_de_ids("7b"), None);
    }

    #[test]
    fn monta_ids_concatena_confere_tamanho_e_vocabulario() {
        let c = vec![vec![1, 2, 3], vec![4, 5]];
        assert_eq!(monta_ids(&c, 4, 10).unwrap(), vec![1, 2, 3, 4]);
        assert!(monta_ids(&c, 6, 10).is_err(), "faltam ids");
        assert!(monta_ids(&c, 5, 5).is_err(), "id 5 não cabe em vocab 5");
        assert!(monta_ids(&[vec![-1]], 1, 10).is_err(), "id negativo");
        assert_eq!(monta_ids(&c, 3, 4).unwrap(), vec![1, 2, 3], "só os ids usados contam para o vocabulário");
    }

    #[test]
    fn comparacao_entre_modos_aceita_o_limite_e_recusa_acima() {
        let w = vec![1.0f32, -2.0, 4.0];
        assert!(compara_entre_modos("a", &w, &w).ok);
        let perto: Vec<f32> = w.iter().map(|v| v + 1e-6).collect(); // 1e-6 / 4 = 2,5e-7
        assert!(compara_entre_modos("a", &perto, &w).ok);
        let longe: Vec<f32> = w.iter().map(|v| v + 1e-4).collect(); // 2,5e-5 > 1e-5
        let c = compara_entre_modos("a", &longe, &w);
        assert!(!c.ok && !c.nao_finito && c.erro > TOL_ENTRE_MODOS);
    }

    #[test]
    fn valor_nao_finito_em_qualquer_vetor_e_falha_numerica_uniforme() {
        let w = vec![1.0f32, 2.0];
        for (g, x) in [(vec![f32::NAN, 2.0], w.clone()), (w.clone(), vec![1.0, f32::INFINITY]), (vec![f32::NEG_INFINITY, 2.0], w.clone())] {
            let a = compara_entre_modos("m", &g, &x);
            let b = compara_logits_x("x", &g, &x);
            assert!(a.nao_finito && !a.ok && a.erro.is_infinite(), "{a:?}");
            assert!(b.nao_finito && !b.ok && b.erro.is_infinite() && !b.argmax_ok, "{b:?}");
        }
        // Tamanho diferente e vetor vazio também são FALHA NUMÉRICA, nunca "sem dado".
        assert!(!compara_entre_modos("m", &[1.0], &[1.0, 2.0]).ok);
        assert!(!compara_entre_modos("m", &[], &[]).ok);
        assert!(!compara_logits_x("x", &[], &[]).ok);
    }

    #[test]
    fn logits_contra_x_exigem_l2_e_argmax() {
        let x = vec![0.1f32, 5.0, 0.2, 0.3];
        assert!(compara_logits_x("x", &x, &x).ok);
        // L2 pequeno, mas o argmax muda: reprova.
        let troca = vec![0.1f32, 4.99, 5.0, 0.3];
        let c = compara_logits_x("x", &troca, &x);
        assert!(!c.ok && !c.argmax_ok);
        // L2 acima do limite com argmax igual: reprova.
        let ruido = vec![0.1f32, 5.0, 0.2, 0.3 + 1.0];
        assert!(!compara_logits_x("x", &ruido, &x).ok);
    }

    #[test]
    fn kv_usa_o_maior_erro_de_todas_as_camadas_e_exige_o_mesmo_numero_de_camadas() {
        let a = vec![(vec![1.0f32; 4], vec![1.0f32; 4]); 3];
        assert!(compara_kv("kv", &a, &a).ok);
        let mut b = a.clone();
        b[2].1[3] = 1.5;
        let c = compara_kv("kv", &b, &a);
        assert!(!c.ok && (c.erro - 0.5).abs() < 1e-9, "{c:?}");
        assert!(!compara_kv("kv", &a[..2], &a).ok, "número de camadas diferente");
        assert!(!compara_kv("kv", &[], &[]).ok);
        let mut n = a.clone();
        n[0].0[0] = f32::NAN;
        let c = compara_kv("kv", &n, &a);
        assert!(!c.ok && c.nao_finito);
    }

    #[test]
    fn perfil_do_kv_separa_as_faixas_de_posicao() {
        let kvd = 2;
        let n = 1100usize;
        let want = vec![(vec![1.0f32; n * kvd], vec![1.0f32; n * kvd])];
        let mut got = want.clone();
        got[0].0[100 * kvd] += 0.01; // posição 100: faixa 1 (64..512)
        got[0].1[1050 * kvd] += 0.1; // posição 1050: faixa 3 (1024..2048)
        let p = perfil_kv(&got, &want, kvd);
        assert_eq!(p[0], 0.0);
        assert!((p[1] - 0.01).abs() < 1e-6, "{p:?}");
        assert_eq!(p[2], 0.0);
        assert!((p[3] - 0.1).abs() < 1e-6, "{p:?}");
        assert!(perfil_kv(&got, &want, 0).iter().all(|e| e.is_infinite()));
    }

    #[test]
    fn contadores_exigem_hits_cpu_positivos_e_iguais() {
        let c = Contadores { needed: 10, hits: 4, cpu: 6 };
        assert!(contadores_validos(c, c));
        assert!(!contadores_validos(c, Contadores { hits: 5, cpu: 5, needed: 10 }), "diferentes");
        let sem_cpu = Contadores { needed: 10, hits: 10, cpu: 0 };
        assert!(!contadores_validos(sem_cpu, sem_cpu), "cpu = 0 não testa o caminho da CPU");
        let sem_hits = Contadores { needed: 10, hits: 0, cpu: 10 };
        assert!(!contadores_validos(sem_hits, sem_hits));
    }

    #[test]
    fn rotulo_da_celula_segue_a_precedencia() {
        let todas_ok = [ok("a"), ok("b")];
        assert_eq!(rotulo_celula(true, &todas_ok), Rotulo::Passa);
        assert_eq!(rotulo_celula(true, &[ok("a"), ruim("b")]), Rotulo::Reprova);
        // Execução inválida: SEM PROVA mesmo com comparações reprovadas (não são lidas como prova) e mesmo com todas OK.
        assert_eq!(rotulo_celula(false, &[ok("a"), ruim("b")]), Rotulo::SemProva);
        assert_eq!(rotulo_celula(false, &todas_ok), Rotulo::SemProva);
        assert_eq!(rotulo_celula(true, &[]), Rotulo::SemProva, "sem comparação nada se prova");
    }

    #[test]
    fn rotulo_do_conjunto_reprova_vence_sem_prova() {
        use Rotulo::*;
        assert_eq!(rotulo_conjunto(&[Passa, Passa], true), Passa);
        assert_eq!(rotulo_conjunto(&[Passa, SemProva], true), SemProva);
        assert_eq!(rotulo_conjunto(&[SemProva, Reprova], true), Reprova, "uma falha numérica válida vence a célula sem prova");
        assert_eq!(rotulo_conjunto(&[Reprova, Passa], true), Reprova);
        assert_eq!(rotulo_conjunto(&[], true), SemProva);
        // Integridade quebrada: tudo SEM PROVA, até o que seria REPROVA ou PASSA.
        assert_eq!(rotulo_conjunto(&[Passa, Passa], false), SemProva);
        assert_eq!(rotulo_conjunto(&[Reprova], false), SemProva);
    }

    #[test]
    fn rotulos_tem_nomes_proprios_sem_os_da_velocidade() {
        for (r, n) in [(Rotulo::Passa, "PASSA"), (Rotulo::Reprova, "REPROVA"), (Rotulo::SemProva, "SEM PROVA")] {
            assert_eq!(r.nome(), n);
        }
    }
}
