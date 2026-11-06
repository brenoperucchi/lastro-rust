//! A2-PASSO E4b: a regra de decisão do critério `qual-runs/A2-LOTE-20261103/criterio-e4b.txt` (seções 2 a 5), em código puro.
//!
//! Nada aqui toca a GPU: recebe tempos e erros já medidos e devolve o rótulo da célula (um modelo) e o resultado do A2. Existe para
//! que a regra fique escrita e testada ANTES da janela de medida, e para que o programa de medida (`src/bin/e4b-medida.rs`) não
//! decida nada por conta própria. Os testes repetem a tabela de conferência da seção 5 do critério.
//!
//! Escolhas de implementação que o critério não diz (registradas para a revisão):
//! - mediana de um número par de ganhos (n = 12): média dos dois valores centrais;
//! - n > 13 não existe pelo desenho (15 pares menos 2 de aquecimento); se aparecer, a célula é INVALIDA, para não decidir sobre um
//!   conjunto que o critério não previu;
//! - "mais de X% pior" é estritamente maior que X (exatamente X% pior ainda não é SEM_CONCLUSAO);
//! - ganho do par: `1 - tp8 / tp1`; com `tp1` não positivo ou não finito o par é perdido (não entra em n).

/// Pares por modelo, aquecimento e pares usados (seção 4 do critério).
pub const PARES: usize = 15;
pub const AQUECIMENTO: usize = 2;
pub const PARES_USADOS: usize = PARES - AQUECIMENTO;
/// Menor n com rótulo de velocidade (seção 4: abaixo disso a célula é INVALIDA).
pub const N_MINIMO: usize = 11;
/// Posições por repetição (seção 2).
pub const BLOCO: usize = 64;
/// Repetições do controle C antes e depois dos pares (seção 2).
pub const C_REPETICOES_LADO: usize = 5;
/// Limite de parada do A2: ganho mínimo por posição (mediana) para PASSA.
pub const GANHO_MINIMO: f64 = 0.15;
/// X da seção 2, em porcentagem (aceito pelo Breno em 06/11/2026).
pub const X_PCT: f64 = 5.0;
/// Tolerâncias da opção 1: norma máxima relativa na passada de portão; L2 relativa nas repetições de tempo.
pub const TOL_PORTAO_MAX: f64 = 1e-5;
pub const TOL_REPETICAO_L2: f64 = 1e-4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rotulo {
    Passa,
    NaoPassa,
    /// Mediana >= 15% sem a contagem mínima de pares positivos. Conta como SEM_CONCLUSAO.
    Ruidoso,
    SemConclusao,
    Invalida,
}

impl Rotulo {
    pub fn nome(self) -> &'static str {
        match self {
            Rotulo::Passa => "PASSA",
            Rotulo::NaoPassa => "NAO_PASSA",
            Rotulo::Ruidoso => "RUIDOSO (SEM_CONCLUSAO)",
            Rotulo::SemConclusao => "SEM_CONCLUSAO",
            Rotulo::Invalida => "INVALIDA",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResultadoA2 {
    /// 1B e 3B PASSAM: abre a proposta de próximo passo ao Breno (não é autorização).
    AbreProposta,
    /// 1B e 3B NAO_PASSAM: fecha o A2.
    FechaA2,
    /// Qualquer outro par, inclusive INVALIDA, SEM_CONCLUSAO ou RUIDOSO: volta ao Breno.
    SemConclusao,
}

/// O controle C de uma célula: o tempo por posição de cada uma das repetições, `None` para a repetição perdida.
#[derive(Clone, Debug, Default)]
pub struct ControleC {
    pub antes: Vec<Option<f64>>,
    pub depois: Vec<Option<f64>>,
}

impl ControleC {
    /// Todas as 10 observações válidas (5 antes e 5 depois), sem refazer perdas.
    pub fn completo(&self) -> bool {
        self.antes.len() == C_REPETICOES_LADO
            && self.depois.len() == C_REPETICOES_LADO
            && self.antes.iter().chain(&self.depois).all(|v| v.is_some_and(|x| x.is_finite() && x > 0.0))
    }

    /// A mediana das 10 observações, ou `None` se o controle está incompleto.
    pub fn mediana(&self) -> Option<f64> {
        if !self.completo() {
            return None;
        }
        let v: Vec<f64> = self.antes.iter().chain(&self.depois).flatten().copied().collect();
        Some(mediana(&v))
    }
}

/// Tudo o que a regra precisa de uma célula.
#[derive(Clone, Debug)]
pub struct Celula {
    /// Os dois portões de correção passaram (a passada de portão e todas as repetições usadas).
    pub portoes_ok: bool,
    /// Os ganhos dos pares USADOS (sem os de aquecimento e sem os perdidos), um por par.
    pub ganhos: Vec<f64>,
    /// Mediana do tempo por posição do M=1 do caminho do passo, nos pares usados (para comparar com C).
    pub tp1_passo_mediana: f64,
    pub c: ControleC,
}

/// Ganho do par a partir de `tp(1)` e `tp(8)` (tempo por posição). `None` se algum dos dois não é utilizável.
pub fn ganho_do_par(tp1: f64, tp8: f64) -> Option<f64> {
    if !(tp1.is_finite() && tp8.is_finite()) || tp1 <= 0.0 || tp8 <= 0.0 {
        return None;
    }
    Some(1.0 - tp8 / tp1)
}

/// Mediana; para tamanho par, a média dos dois centrais. Vetor vazio dá NaN.
pub fn mediana(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

/// Mínimo, quartis e máximo (descritivo; não decide). Quartis por posição mais próxima em `(n-1)·p`, como o `a1-motor`.
pub fn resumo(v: &[f64]) -> [f64; 5] {
    if v.is_empty() {
        return [f64::NAN; 5];
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let q = |p: f64| s[(((s.len() - 1) as f64) * p).round() as usize];
    [s[0], q(0.25), mediana(&s), q(0.75), s[s.len() - 1]]
}

/// O mínimo de pares com ganho positivo para PASSA: `ceil(10·n / 13)` (n = 13: 10; n = 12: 10; n = 11: 9).
pub fn minimo_positivos(n: usize) -> usize {
    (10 * n).div_ceil(13)
}

/// A regra da seção 5, na precedência INVALIDA > SEM_CONCLUSAO (inclui RUIDOSO) > PASSA > NAO_PASSA.
pub fn decide(c: &Celula) -> Rotulo {
    let n = c.ganhos.len();
    if !c.portoes_ok || n < N_MINIMO || n > PARES_USADOS || c.ganhos.iter().any(|g| !g.is_finite()) {
        return Rotulo::Invalida;
    }
    // SEM_CONCLUSAO por controle: C incompleto, ou o M=1 do passo mais de X% pior que a mediana de C.
    let Some(med_c) = c.c.mediana() else { return Rotulo::SemConclusao };
    if !c.tp1_passo_mediana.is_finite() || (c.tp1_passo_mediana - med_c) / med_c * 100.0 > X_PCT {
        return Rotulo::SemConclusao;
    }
    let med = mediana(&c.ganhos);
    let positivos = c.ganhos.iter().filter(|&&g| g > 0.0).count();
    if med >= GANHO_MINIMO {
        if positivos >= minimo_positivos(n) { Rotulo::Passa } else { Rotulo::Ruidoso }
    } else {
        Rotulo::NaoPassa
    }
}

/// O resultado do A2 a partir dos rótulos do 1B e do 3B.
pub fn resultado_a2(r1b: Rotulo, r3b: Rotulo) -> ResultadoA2 {
    match (r1b, r3b) {
        (Rotulo::Passa, Rotulo::Passa) => ResultadoA2::AbreProposta,
        (Rotulo::NaoPassa, Rotulo::NaoPassa) => ResultadoA2::FechaA2,
        _ => ResultadoA2::SemConclusao,
    }
}

/// Erro máximo relativo (norma máxima do vetor): `max|got - want| / max(|want|, 1e-6)`. Infinito se o tamanho difere ou há valor não finito.
pub fn erro_max_rel(got: &[f32], want: &[f32]) -> f64 {
    if got.is_empty() || got.len() != want.len() || got.iter().chain(want).any(|v| !v.is_finite()) {
        return f64::INFINITY;
    }
    let m = want.iter().fold(0f64, |m, v| m.max(v.abs() as f64)).max(1e-6);
    got.iter().zip(want).fold(0f64, |m, (a, b)| m.max((*a as f64 - *b as f64).abs())) / m
}

/// Erro relativo na norma L2, como no `a1-motor`. Infinito se o tamanho difere, há valor não finito ou a referência é toda zero.
pub fn erro_l2_rel(got: &[f32], want: &[f32]) -> f64 {
    if got.is_empty() || got.len() != want.len() || got.iter().chain(want).any(|v| !v.is_finite()) {
        return f64::INFINITY;
    }
    let (mut num, mut den) = (0f64, 0f64);
    for (a, b) in got.iter().zip(want) {
        num += ((*a as f64) - (*b as f64)).powi(2);
        den += (*b as f64).powi(2);
    }
    if den == 0.0 { f64::INFINITY } else { (num / den).sqrt() }
}

pub fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0)
}

/// O portão por repetição de tempo (item (b) da seção 3): L2 relativa <= 1e-4 e argmax igual contra a referência X.
pub fn portao_repeticao(got: &[f32], x: &[f32]) -> bool {
    erro_l2_rel(got, x) <= TOL_REPETICAO_L2 && argmax(got) == argmax(x)
}

/// A ordem dos pares usados: ABBA a partir do primeiro par usado. `true` = o M=1 vai primeiro.
pub fn m1_primeiro(par_usado: usize) -> bool {
    matches!(par_usado % 4, 0 | 3)
}

#[cfg(test)]
mod testes {
    use super::*;

    fn c_ok(med: f64) -> ControleC {
        ControleC { antes: vec![Some(med); 5], depois: vec![Some(med); 5] }
    }

    /// Uma célula com `n` pares, `pos` deles positivos e a mediana pedida (os valores são montados para dar essa mediana).
    fn celula(n: usize, mediana_alvo: f64, pos: usize) -> Celula {
        // Ganhos: os `pos` positivos cobrem a metade de cima (>= mediana quando cabem); o resto é negativo.
        let mut g: Vec<f64> = Vec::new();
        let altos = n.div_ceil(2).min(pos);
        for _ in 0..altos {
            g.push(mediana_alvo.max(0.0).max(1e-3));
        }
        for _ in altos..pos {
            g.push(0.001);
        }
        while g.len() < n {
            g.push(-0.1);
        }
        Celula { portoes_ok: true, ganhos: g, tp1_passo_mediana: 1.0, c: c_ok(1.0) }
    }

    fn com_mediana(mut c: Celula, med: f64) -> Celula {
        // Ajusta os valores centrais para a mediana exata pedida sem mudar quem é positivo.
        let n = c.ganhos.len();
        c.ganhos.sort_by(f64::total_cmp);
        if n % 2 == 1 {
            c.ganhos[n / 2] = med;
        } else {
            c.ganhos[n / 2 - 1] = med;
            c.ganhos[n / 2] = med;
        }
        c
    }

    /// Cada linha da tabela de conferência da seção 5 do critério.
    #[test]
    fn tabela_de_conferencia_do_criterio() {
        // n=13, mediana 20%, 10 positivos -> PASSA
        let c = com_mediana(celula(13, 0.20, 10), 0.20);
        assert_eq!(c.ganhos.iter().filter(|&&g| g > 0.0).count(), 10);
        assert_eq!(decide(&c), Rotulo::Passa);
        // n=13, mediana 20%, 9 positivos -> RUIDOSO (a mediana de 13 com 9 positivos só chega a >= 15% se os positivos estiverem acima dela)
        let mut c = celula(13, 0.20, 9);
        c = com_mediana(c, 0.20);
        assert!(mediana(&c.ganhos) >= 0.15);
        assert_eq!(c.ganhos.iter().filter(|&&g| g > 0.0).count(), 9);
        assert_eq!(decide(&c), Rotulo::Ruidoso);
        // n=13, mediana 20%, 7 positivos -> RUIDOSO
        let c = com_mediana(celula(13, 0.20, 7), 0.20);
        assert_eq!(c.ganhos.iter().filter(|&&g| g > 0.0).count(), 7);
        assert_eq!(decide(&c), Rotulo::Ruidoso);
        // n=13, mediana 15% exato, 10 positivos -> PASSA
        let c = com_mediana(celula(13, 0.15, 10), 0.15);
        assert_eq!(mediana(&c.ganhos), 0.15);
        assert_eq!(decide(&c), Rotulo::Passa);
        // n=13, mediana 14,9%, 13 positivos -> NAO_PASSA
        let c = Celula { portoes_ok: true, ganhos: vec![0.149; 13], tp1_passo_mediana: 1.0, c: c_ok(1.0) };
        assert_eq!(decide(&c), Rotulo::NaoPassa);
        // n=12, mediana 20%, 10 positivos -> PASSA
        let c = com_mediana(celula(12, 0.20, 10), 0.20);
        assert_eq!(c.ganhos.iter().filter(|&&g| g > 0.0).count(), 10);
        assert_eq!(decide(&c), Rotulo::Passa);
        // n=12, mediana 20%, 9 positivos -> RUIDOSO
        let c = com_mediana(celula(12, 0.20, 9), 0.20);
        assert_eq!(c.ganhos.iter().filter(|&&g| g > 0.0).count(), 9);
        assert_eq!(decide(&c), Rotulo::Ruidoso);
        // n=11, mediana 20%, 9 positivos -> PASSA
        let c = com_mediana(celula(11, 0.20, 9), 0.20);
        assert_eq!(c.ganhos.iter().filter(|&&g| g > 0.0).count(), 9);
        assert_eq!(decide(&c), Rotulo::Passa);
        // n=11, mediana 20%, 8 positivos -> RUIDOSO
        let c = com_mediana(celula(11, 0.20, 8), 0.20);
        assert_eq!(c.ganhos.iter().filter(|&&g| g > 0.0).count(), 8);
        assert_eq!(decide(&c), Rotulo::Ruidoso);
        // n=10 (qualquer mediana) -> INVALIDA
        let c = Celula { portoes_ok: true, ganhos: vec![0.5; 10], tp1_passo_mediana: 1.0, c: c_ok(1.0) };
        assert_eq!(decide(&c), Rotulo::Invalida);
        // n=13, mediana 20%, 10 positivos, C com 9 válidas -> SEM_CONCLUSAO
        let mut c = com_mediana(celula(13, 0.20, 10), 0.20);
        c.c.antes[0] = None;
        assert_eq!(decide(&c), Rotulo::SemConclusao);
        // n=13, mediana 5%, tp(1) do passo 8% pior que C (X=5) -> SEM_CONCLUSAO
        let mut c = Celula { portoes_ok: true, ganhos: vec![0.05; 13], tp1_passo_mediana: 1.08, c: c_ok(1.0) };
        assert_eq!(decide(&c), Rotulo::SemConclusao);
        // portão falhou e C incompleto -> INVALIDA
        c.portoes_ok = false;
        c.c.antes[0] = None;
        assert_eq!(decide(&c), Rotulo::Invalida);
    }

    #[test]
    fn minimos_de_positivos_por_n() {
        assert_eq!((minimo_positivos(13), minimo_positivos(12), minimo_positivos(11)), (10, 10, 9));
    }

    #[test]
    fn x_exato_nao_e_mais_de_x() {
        // Exatamente 5% pior que C ainda não é SEM_CONCLUSAO; 5,01% é.
        let mk = |tp1: f64| Celula { portoes_ok: true, ganhos: vec![0.2; 13], tp1_passo_mediana: tp1, c: c_ok(100.0) };
        assert_eq!(decide(&mk(105.0)), Rotulo::Passa);
        assert_eq!(decide(&mk(105.01)), Rotulo::SemConclusao);
    }

    #[test]
    fn precedencia_invalida_depois_sem_conclusao_depois_velocidade() {
        let base = Celula { portoes_ok: true, ganhos: vec![0.2; 13], tp1_passo_mediana: 1.0, c: c_ok(1.0) };
        assert_eq!(decide(&base), Rotulo::Passa);
        // C incompleto vence PASSA e NAO_PASSA.
        let mut c = base.clone();
        c.c.depois.pop();
        assert_eq!(decide(&c), Rotulo::SemConclusao);
        let mut c = Celula { ganhos: vec![0.0; 13], ..base.clone() };
        c.c.depois[4] = None;
        assert_eq!(decide(&c), Rotulo::SemConclusao);
        // Portão falho vence tudo.
        let mut c = base.clone();
        c.portoes_ok = false;
        c.c.depois[4] = None;
        assert_eq!(decide(&c), Rotulo::Invalida);
        // n > 13 não existe pelo desenho: não decide.
        let c = Celula { ganhos: vec![0.2; 14], ..base.clone() };
        assert_eq!(decide(&c), Rotulo::Invalida);
        // Ganho não finito (tempo corrompido) invalida.
        let mut c = base;
        c.ganhos[3] = f64::NAN;
        assert_eq!(decide(&c), Rotulo::Invalida);
    }

    #[test]
    fn controle_c_pede_as_dez_observacoes_validas() {
        assert!(c_ok(1.0).completo());
        let mut c = c_ok(1.0);
        c.depois.push(Some(1.0));
        assert!(!c.completo(), "mais de 5 de um lado não é o desenho");
        let mut c = c_ok(1.0);
        c.antes[2] = None;
        assert!(c.mediana().is_none());
        let mut c = c_ok(1.0);
        c.antes[0] = Some(0.0);
        assert!(!c.completo(), "tempo zero não vale");
        let mut c = c_ok(1.0);
        c.depois[0] = Some(f64::NAN);
        assert!(!c.completo());
    }

    #[test]
    fn resultado_do_a2() {
        use Rotulo::*;
        assert_eq!(resultado_a2(Passa, Passa), ResultadoA2::AbreProposta);
        assert_eq!(resultado_a2(NaoPassa, NaoPassa), ResultadoA2::FechaA2);
        for (a, b) in [(Passa, NaoPassa), (NaoPassa, Passa), (NaoPassa, SemConclusao), (Passa, Ruidoso), (Ruidoso, NaoPassa), (Invalida, Passa), (NaoPassa, Invalida), (Invalida, Invalida)] {
            assert_eq!(resultado_a2(a, b), ResultadoA2::SemConclusao, "{a:?} {b:?}");
        }
    }

    #[test]
    fn ganho_e_mediana() {
        assert_eq!(ganho_do_par(2.0, 1.0), Some(0.5));
        assert_eq!(ganho_do_par(1.0, 2.0), Some(-1.0));
        assert_eq!(ganho_do_par(0.0, 1.0), None);
        assert_eq!(ganho_do_par(f64::NAN, 1.0), None);
        assert_eq!(mediana(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(mediana(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!(mediana(&[]).is_nan());
        assert_eq!(resumo(&[1.0, 2.0, 3.0, 4.0, 5.0]), [1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn abba_comeca_no_primeiro_par_usado() {
        let ordem: Vec<bool> = (0..8).map(m1_primeiro).collect();
        assert_eq!(ordem, [true, false, false, true, true, false, false, true]);
    }

    #[test]
    fn tolerancias_dos_portoes() {
        let x = [1.0f32, 5.0, 2.0];
        assert!(portao_repeticao(&x, &x));
        // L2 relativa acima de 1e-4 reprova; argmax diferente reprova.
        assert!(!portao_repeticao(&[1.0, 5.0 + 1e-3, 2.0], &x));
        assert!(!portao_repeticao(&[1.0, 1.9999, 2.0], &x));
        assert!(!portao_repeticao(&[1.0, f32::NAN, 2.0], &x));
        assert!(!portao_repeticao(&[], &x));
        assert!(!portao_repeticao(&[1.0, 5.0], &x), "tamanho diferente reprova");
        assert!(erro_max_rel(&x, &x) == 0.0);
        assert!(erro_max_rel(&[1.0, 5.0 + 1e-4, 2.0], &x) > TOL_PORTAO_MAX);
        assert!(erro_max_rel(&[1.0, 5.0 + 1e-6, 2.0], &x) <= TOL_PORTAO_MAX);
        assert!(erro_max_rel(&[], &x).is_infinite());
        assert!(erro_l2_rel(&[0.0, 0.0], &[0.0, 0.0]).is_infinite(), "referência toda zero reprova");
    }
}
