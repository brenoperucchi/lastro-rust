//! Telemetria da máquina lida pelo monitor: GPU pela NVML, CPU/RAM/disco por `/proc` e `/sys`. Tudo é só leitura.
use nvml_wrapper::enum_wrappers::device::{PcieUtilCounter, TemperatureSensor};
use nvml_wrapper::Nvml;
use std::collections::HashSet;
use std::fs;
use std::time::Instant;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Gpu {
    pub name: String,
    pub load_pct: f64,
    pub vram_used_gb: f64,
    pub vram_total_gb: f64,
    pub temp_c: f64,
    pub power_w: f64,
    pub power_limit_w: f64,
    pub pcie_gen: u32,
    pub pcie_width: u32,
    /// Tráfego PCIe recente (MB/s), do ponto de vista da GPU: para a GPU (host→device) e saindo dela.
    pub pcie_to_gpu_mbs: f64,
    pub pcie_from_gpu_mbs: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Host {
    pub cpu_pct: f64,
    pub cores: usize,
    pub threads: usize,
    pub ram_used_gb: f64,
    pub ram_total_gb: f64,
    pub disk_read_mbs: f64,
    pub disk_write_mbs: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sample {
    /// `None` quando a NVML não está disponível (a tela mostra "sem GPU").
    pub gpu: Option<Gpu>,
    pub host: Host,
}

pub struct Sampler {
    nvml: Option<Nvml>,
    gpu_index: u32,
    cores: usize,
    threads: usize,
    disks: Vec<String>,
    prev_cpu: Option<(u64, u64)>,
    prev_disk: Option<(Instant, u64, u64)>,
}

impl Sampler {
    pub fn new(gpu_index: u32) -> Self {
        // O driver do Arch instala `libnvidia-ml.so`, mas outras distribuições só trazem o `.so.1`.
        let nvml = Nvml::init().ok().or_else(|| Nvml::builder().lib_path("libnvidia-ml.so.1".as_ref()).init().ok());
        let (cores, threads) = cpu_topology();
        Self { nvml, gpu_index, cores, threads, disks: physical_disks(), prev_cpu: None, prev_disk: None }
    }

    pub fn sample(&mut self) -> Sample {
        let gpu = self.nvml.as_ref().and_then(|n| read_gpu(n, self.gpu_index));
        let mut host = Host { cores: self.cores, threads: self.threads, ..Default::default() };
        if let Some((busy, total)) = cpu_times() {
            if let Some((pb, pt)) = self.prev_cpu {
                let dt = total.saturating_sub(pt);
                if dt > 0 {
                    host.cpu_pct = busy.saturating_sub(pb) as f64 / dt as f64 * 100.0;
                }
            }
            self.prev_cpu = Some((busy, total));
        }
        if let Some((total_kb, avail_kb)) = meminfo() {
            host.ram_total_gb = total_kb as f64 / (1 << 20) as f64;
            host.ram_used_gb = total_kb.saturating_sub(avail_kb) as f64 / (1 << 20) as f64;
        }
        let now = Instant::now();
        if let Some((rd, wr)) = disk_sectors(&self.disks) {
            if let Some((t, prd, pwr)) = self.prev_disk {
                let secs = now.duration_since(t).as_secs_f64();
                if secs > 0.0 {
                    host.disk_read_mbs = rd.saturating_sub(prd) as f64 * 512.0 / 1e6 / secs;
                    host.disk_write_mbs = wr.saturating_sub(pwr) as f64 * 512.0 / 1e6 / secs;
                }
            }
            self.prev_disk = Some((now, rd, wr));
        }
        Sample { gpu, host }
    }
}

fn read_gpu(nvml: &Nvml, index: u32) -> Option<Gpu> {
    let d = nvml.device_by_index(index).ok()?;
    let mem = d.memory_info().ok();
    Some(Gpu {
        name: d.name().unwrap_or_default(),
        load_pct: d.utilization_rates().map(|u| u.gpu as f64).unwrap_or(0.0),
        // GiB, como o "24 GB" da placa e a RAM abaixo.
        vram_used_gb: mem.as_ref().map(|m| m.used as f64 / (1u64 << 30) as f64).unwrap_or(0.0),
        vram_total_gb: mem.as_ref().map(|m| m.total as f64 / (1u64 << 30) as f64).unwrap_or(0.0),
        temp_c: d.temperature(TemperatureSensor::Gpu).map(|t| t as f64).unwrap_or(0.0),
        power_w: d.power_usage().map(|p| p as f64 / 1e3).unwrap_or(0.0),
        power_limit_w: d.enforced_power_limit().map(|p| p as f64 / 1e3).unwrap_or(0.0),
        pcie_gen: d.current_pcie_link_gen().unwrap_or(0),
        pcie_width: d.current_pcie_link_width().unwrap_or(0),
        // A NVML dá KB/s; "Receive" é o que a GPU recebe (host→device).
        pcie_to_gpu_mbs: d.pcie_throughput(PcieUtilCounter::Receive).map(|v| v as f64 / 1e3).unwrap_or(0.0),
        pcie_from_gpu_mbs: d.pcie_throughput(PcieUtilCounter::Send).map(|v| v as f64 / 1e3).unwrap_or(0.0),
    })
}

/// (tempo ocupado, tempo total) somados de todas as CPUs, em jiffies.
pub fn parse_cpu_times(stat: &str) -> Option<(u64, u64)> {
    let line = stat.lines().find(|l| l.starts_with("cpu "))?;
    let v: Vec<u64> = line.split_whitespace().skip(1).filter_map(|x| x.parse().ok()).collect();
    if v.len() < 4 {
        return None;
    }
    let idle = v[3] + v.get(4).copied().unwrap_or(0); // idle + iowait
    let total: u64 = v.iter().take(8).sum(); // sem guest/guest_nice, que já entram em user/nice
    Some((total - idle, total))
}

fn cpu_times() -> Option<(u64, u64)> {
    parse_cpu_times(&fs::read_to_string("/proc/stat").ok()?)
}

/// (MemTotal, MemAvailable) em KiB.
pub fn parse_meminfo(text: &str) -> Option<(u64, u64)> {
    let field = |name: &str| {
        text.lines().find(|l| l.starts_with(name)).and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
    };
    Some((field("MemTotal:")?, field("MemAvailable:")?))
}

fn meminfo() -> Option<(u64, u64)> {
    parse_meminfo(&fs::read_to_string("/proc/meminfo").ok()?)
}

/// Discos físicos: os de `/sys/block` com `device` (exclui dm, zram, loop, que repetiriam o tráfego dos físicos).
fn physical_disks() -> Vec<String> {
    let Ok(entries) = fs::read_dir("/sys/block") else { return vec![] };
    entries
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("device").exists())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect()
}

/// Setores lidos e escritos somados dos discos `names` em `/proc/diskstats`.
pub fn parse_diskstats(text: &str, names: &[String]) -> Option<(u64, u64)> {
    let mut rd = 0;
    let mut wr = 0;
    let mut found = false;
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() > 9 && names.iter().any(|n| n == f[2]) {
            rd += f[5].parse::<u64>().ok()?;
            wr += f[9].parse::<u64>().ok()?;
            found = true;
        }
    }
    found.then_some((rd, wr))
}

fn disk_sectors(names: &[String]) -> Option<(u64, u64)> {
    parse_diskstats(&fs::read_to_string("/proc/diskstats").ok()?, names)
}

fn cpu_topology() -> (usize, usize) {
    let text = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let mut threads = 0;
    let mut cores = HashSet::new();
    let (mut phys, mut core) = (String::new(), String::new());
    for line in text.lines().chain(std::iter::once("")) {
        if line.trim().is_empty() {
            if !core.is_empty() {
                cores.insert((phys.clone(), core.clone()));
            }
            phys.clear();
            core.clear();
            continue;
        }
        let (k, v) = line.split_once(':').unwrap_or((line, ""));
        match k.trim() {
            "processor" => threads += 1,
            "physical id" => phys = v.trim().to_string(),
            "core id" => core = v.trim().to_string(),
            _ => {}
        }
    }
    let threads = threads.max(1);
    (if cores.is_empty() { threads } else { cores.len() }, threads)
}
