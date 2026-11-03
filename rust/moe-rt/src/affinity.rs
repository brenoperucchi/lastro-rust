//! A1-DIAG10: afinidade de threads para o harness N (criterio.txt v6, seção 4). Só Linux, sem dependência nova: chama
//! `sched_setaffinity`/`sched_getaffinity` da libc que o std já liga. `pid = 0` age sobre a thread que chama.
//!
//! Configurações:
//! - A0: nada é fixado aqui (o `taskset -c 0-15` de fora vale para tudo, como no A1.4);
//! - A1w: os 8 participantes do cálculo (quem chama o `Pool` + 7 `moe-cpu-*`) um por núcleo físico (CPU 0 e 2,4,...,14);
//! - A1m: só a thread principal do motor, a que gira na stream, vai para o E-core 17 (feito por `a1-motor`).
use std::fmt;
use std::sync::Mutex;

/// Plano de colocação lido por `Pool::new` (A1w). O binário o define antes de criar o motor ou o cache; `None` deixa tudo como estava.
static COMPUTE_PLAN: Mutex<Option<Vec<usize>>> = Mutex::new(None);

pub fn set_compute_plan(plan: Option<Vec<usize>>) {
    *COMPUTE_PLAN.lock().unwrap_or_else(|e| e.into_inner()) = plan;
}

pub fn compute_plan() -> Option<Vec<usize>> {
    COMPUTE_PLAN.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Conjunto de CPUs de até 1024 (o tamanho do `cpu_set_t` da glibc).
const WORDS: usize = 16;

extern "C" {
    fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
    fn sched_getaffinity(pid: i32, size: usize, mask: *mut u64) -> i32;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Afin {
    A0,
    A1w,
    A1m,
}

impl Afin {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "a0" | "A0" => Ok(Afin::A0),
            "a1w" | "A1w" => Ok(Afin::A1w),
            "a1m" | "A1m" => Ok(Afin::A1m),
            _ => Err(format!("afinidade desconhecida: {s} (a0, a1w ou a1m)")),
        }
    }

    /// CPUs dos participantes do cálculo, `[0]` para quem chama o `Pool` e `[i]` para o trabalhador `moe-cpu-i`; só na A1w.
    pub fn compute_plan(self, threads: usize) -> Option<Vec<usize>> {
        (self == Afin::A1w).then(|| a1w_plan(threads))
    }

    /// CPU da thread principal do motor (a que gira na stream); só na A1m.
    pub fn main_cpu(self) -> Option<usize> {
        (self == Afin::A1m).then_some(MOTOR_MAIN_E_CORE)
    }
}

/// E-core da thread principal do motor na A1m.
pub const MOTOR_MAIN_E_CORE: usize = 17;

/// Um participante por núcleo físico: o das posições pares dos P-cores 0 a 14 (a CPU 2k e a 2k+1 compartilham o núcleo).
pub fn a1w_plan(threads: usize) -> Vec<usize> {
    assert!(threads <= 8, "A1w tem 8 núcleos físicos nos P-cores");
    (0..threads).map(|i| 2 * i).collect()
}

#[derive(Debug)]
pub struct AfinError(pub String);

impl fmt::Display for AfinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AfinError {}

/// Fixa a thread que chama em `cpus`.
pub fn set_current(cpus: &[usize]) -> Result<(), AfinError> {
    let mut m = [0u64; WORDS];
    for &c in cpus {
        if c >= WORDS * 64 {
            return Err(AfinError(format!("CPU {c} fora do conjunto")));
        }
        m[c / 64] |= 1 << (c % 64);
    }
    // SAFETY: `m` tem `WORDS * 8` bytes, o tamanho passado; pid 0 é a thread que chama.
    let r = unsafe { sched_setaffinity(0, WORDS * 8, m.as_ptr()) };
    if r == 0 {
        Ok(())
    } else {
        Err(AfinError(format!("sched_setaffinity {cpus:?}: {}", std::io::Error::last_os_error())))
    }
}

/// A máscara efetiva da thread que chama.
pub fn get_current() -> Result<Vec<usize>, AfinError> {
    let mut m = [0u64; WORDS];
    // SAFETY: `m` tem `WORDS * 8` bytes, o tamanho passado.
    let r = unsafe { sched_getaffinity(0, WORDS * 8, m.as_mut_ptr()) };
    if r != 0 {
        return Err(AfinError(format!("sched_getaffinity: {}", std::io::Error::last_os_error())));
    }
    Ok((0..WORDS * 64).filter(|c| m[c / 64] >> (c % 64) & 1 == 1).collect())
}

/// Lista "0-3,8" como o kernel escreve (`Cpus_allowed_list`, `thread_siblings_list`).
pub fn parse_list(s: &str) -> Vec<usize> {
    let mut v = Vec::new();
    for p in s.trim().split(',').filter(|p| !p.is_empty()) {
        match p.split_once('-') {
            Some((a, b)) => v.extend(a.parse::<usize>().unwrap()..=b.parse::<usize>().unwrap()),
            None => v.push(p.parse().unwrap()),
        }
    }
    v
}

/// Núcleo físico de uma CPU: o menor id entre os irmãos de HT (`thread_siblings_list`).
pub fn physical_core(cpu: usize) -> usize {
    std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"))
        .ok()
        .and_then(|s| parse_list(&s).into_iter().min())
        .unwrap_or(cpu)
}

/// Controle do harness (criterio.txt, seção 4): cada participante do cálculo está fixo numa CPU única e as CPUs dos 8
/// participantes caem em núcleos físicos distintos. `core_of` entra por parâmetro para o teste usar uma topologia fixa.
pub fn one_core_each(masks: &[Vec<usize>], core_of: &dyn Fn(usize) -> usize) -> bool {
    let mut cores = Vec::new();
    for m in masks {
        if m.len() != 1 {
            return false;
        }
        cores.push(core_of(m[0]));
    }
    cores.sort_unstable();
    cores.windows(2).all(|w| w[0] != w[1])
}

/// Todas as threads do processo: `(tid, comm, Cpus_allowed_list)`, para gravar a afinidade efetiva por papel na saída.
pub fn dump_threads() -> Vec<(u32, String, String)> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir("/proc/self/task") {
        for e in rd.flatten() {
            let tid: u32 = e.file_name().to_string_lossy().parse().unwrap_or(0);
            let comm = std::fs::read_to_string(e.path().join("comm")).unwrap_or_default().trim().to_string();
            let allowed = std::fs::read_to_string(e.path().join("status"))
                .unwrap_or_default()
                .lines()
                .find_map(|l| l.strip_prefix("Cpus_allowed_list:").map(|v| v.trim().to_string()))
                .unwrap_or_default();
            out.push((tid, comm, allowed));
        }
    }
    out.sort();
    out
}

/// Linhas `AFINIDADE ...` (afinidade efetiva de cada thread do processo) e `CONTROLE ...` na saída. Os participantes do cálculo são
/// os `moe-cpu-*` e quem chama o `Pool`: a thread `moe-servico` no motor, ou a principal (`caller_is_main`) no m2a-cache.
pub fn relata(pname: &str, budget: &str, afin: Afin, threads: usize, caller_is_main: bool) {
    let th = dump_threads();
    let pid = std::process::id();
    for (tid, comm, mask) in &th {
        println!("AFINIDADE {pname} {budget} {afin:?} tid {tid} {comm} {mask}");
    }
    let masks: Vec<Vec<usize>> = th
        .iter()
        .filter(|(tid, c, _)| c.starts_with("moe-cpu-") || if caller_is_main { *tid == pid } else { c == "moe-servico" })
        .map(|(_, _, m)| parse_list(m))
        .collect();
    match afin {
        Afin::A1w => {
            let ok = masks.len() == threads && one_core_each(&masks, &physical_core);
            println!("CONTROLE A1w {pname} {budget}: {} participantes, {}", masks.len(), if ok { "núcleos físicos distintos: ok" } else { "FALHOU" });
        }
        Afin::A1m => {
            let principal = th.iter().find(|(tid, _, _)| *tid == pid).map(|(_, _, m)| m.clone()).unwrap_or_default();
            println!("CONTROLE A1m {pname} {budget}: thread principal em {principal}");
        }
        Afin::A0 => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Topologia desta máquina: a CPU 2k e a 2k+1 dividem o núcleo k, para as CPUs 0 a 15.
    fn topo(c: usize) -> usize {
        if c < 16 { c / 2 } else { 100 + c }
    }

    #[test]
    fn plano_a1w_tem_oito_nucleos_fisicos() {
        let p = a1w_plan(8);
        assert_eq!(p, vec![0, 2, 4, 6, 8, 10, 12, 14]);
        let masks: Vec<Vec<usize>> = p.iter().map(|&c| vec![c]).collect();
        assert!(one_core_each(&masks, &topo));
    }

    #[test]
    fn so_os_sete_trabalhadores_fixos_reprova_o_controle() {
        // quem chama continua livre em 0-15: máscara de 16 CPUs, não uma CPU única
        let mut masks: Vec<Vec<usize>> = vec![(0..16).collect()];
        masks.extend(a1w_plan(8)[1..].iter().map(|&c| vec![c]));
        assert!(!one_core_each(&masks, &topo));
    }

    #[test]
    fn dois_participantes_no_mesmo_nucleo_reprova_o_controle() {
        let masks = vec![vec![0], vec![1], vec![4], vec![6], vec![8], vec![10], vec![12], vec![14]];
        assert!(!one_core_each(&masks, &topo));
    }

    #[test]
    fn set_e_get_current_concordam() {
        let antes = get_current().unwrap();
        let alvo = antes[0];
        std::thread::spawn(move || {
            set_current(&[alvo]).unwrap();
            assert_eq!(get_current().unwrap(), vec![alvo]);
        })
        .join()
        .unwrap();
        assert_eq!(get_current().unwrap(), antes, "a fixação de outra thread não pode vazar para esta");
    }

    #[test]
    fn parse_list_le_faixas() {
        assert_eq!(parse_list("0-2,5\n"), vec![0, 1, 2, 5]);
    }
}
