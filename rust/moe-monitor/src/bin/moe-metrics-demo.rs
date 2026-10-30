//! Publicador de demonstração: simula um runtime atendendo requisições (prompt, depois resposta) e publica as métricas
//! no mesmo ponto que o runtime usará, para ver o monitor funcionando antes de o runtime existir. Os números são
//! inventados (marcados com o nome de modelo "demo").
//!
//!   moe-metrics-demo [--metrics <arquivo>] [--seconds <n>]
use moe_monitor::snapshot::{
    default_path, now_ms, Context, Event, ExpertCache, Model, Publisher, Request, Snapshot, Speed, State, Window,
};
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn main() {
    let mut path = default_path();
    let mut seconds = 60.0f64;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--metrics" => path = PathBuf::from(args.next().expect("--metrics <arquivo>")),
            "--seconds" => seconds = args.next().and_then(|v| v.parse().ok()).expect("--seconds <n>"),
            other => panic!("opção desconhecida: {other}"),
        }
    }
    let publisher = Publisher::new(&path).expect("diretório das métricas");
    println!("publicando métricas de demonstração em {} por {seconds} s", publisher.path().display());
    let mut rng = 0x2545_f491_4f6c_dd1du64;
    let mut noise = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut snap = Snapshot {
        model: Model { name: "demo (Granite 3B)".into(), layers: 32, experts: 40, experts_per_tok: 8 },
        context: Context { used: 0, max: 131_072 },
        cache: ExpertCache { slots_total: 640, ..Default::default() },
        started_ms: now_ms(),
        settings: [
            ("política de colocação", "custo (demonstração)"),
            ("orçamento de VRAM", "640 slots (50%)"),
            ("threads da CPU", "8"),
            (moe_monitor::ui::EXPERT_MB_KEY, "4,7"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect(),
        ..Default::default()
    };
    let log = |snap: &mut Snapshot, level: &str, message: String| {
        snap.events.push(Event { ts_ms: now_ms(), level: level.into(), message });
        let n = snap.events.len();
        if n > 200 {
            snap.events.drain(..n - 200);
        }
    };
    log(&mut snap, "info", "demonstração iniciada: números inventados".into());
    let mut phase = 0u8;
    let start = Instant::now();
    let (prompt, answer) = (3_000u64, 400u64);
    let mut req_start = now_ms();
    let mut last_decode = 0.0;
    while start.elapsed().as_secs_f64() < seconds {
        // Ciclo de 16 s: 2 s ocioso, 2 s lendo o prompt, 12 s gerando.
        let t = start.elapsed().as_secs_f64() % 16.0;
        let fill = |x: f64| x.clamp(0.0, 1.0);
        let now_phase = if t < 2.0 { 0 } else if t < 4.0 { 1 } else { 2 };
        if now_phase != phase {
            phase = now_phase;
            let msg = match phase {
                0 => "requisição concluída".to_string(),
                1 => format!("nova requisição: prompt de {prompt} tokens"),
                _ => "prompt lido; gerando".to_string(),
            };
            log(&mut snap, "info", msg);
            if phase == 2 && noise() < 0.3 {
                log(&mut snap, "aviso", "fila de cópias cheia; falta servida pela CPU".into());
            }
        }
        snap.state = if t < 2.0 {
            snap.speed = Speed { decode_last: last_decode, ..Default::default() };
            State::Idle
        } else if t < 4.0 {
            if t < 2.3 {
                req_start = now_ms();
            }
            let done = (fill((t - 2.0) / 2.0) * prompt as f64) as u64;
            snap.speed = Speed {
                prefill_now: 1_500.0 + 100.0 * noise(),
                prefill_request: 1_480.0,
                decode_last: last_decode,
                ..Default::default()
            };
            snap.context.used = done;
            State::Reading { done, total: prompt }
        } else {
            let tokens = (fill((t - 4.0) / 12.0) * answer as f64) as u64;
            last_decode = 32.0 + 6.0 * noise();
            snap.speed = Speed { decode_now: last_decode, prefill_request: 1_480.0, ..Default::default() };
            snap.context.used = prompt + tokens;
            State::Generating { tokens, max_tokens: answer }
        };
        let gen = matches!(snap.state, State::Generating { .. });
        let hit = 0.80 + 0.06 * noise();
        let cpu = (1.0 - hit) * (0.7 + 0.1 * noise());
        snap.cache.slots_resident = 600 + (40.0 * noise()) as u32;
        snap.cache.slots_loading = if gen { (6.0 * noise()) as u32 } else { 0 };
        snap.cache.hit_frac = if gen { hit } else { 0.0 };
        snap.cache.cpu_frac = if gen { cpu } else { 0.0 };
        snap.cache.copy_frac = if gen { 1.0 - hit - cpu } else { 0.0 };
        snap.cache.copies_inflight = snap.cache.slots_loading;
        snap.cache.h2d_gbs = if gen { 4.0 + 8.0 * noise() } else { 0.0 };
        if gen {
            snap.cache.total_hits += 200;
            snap.cache.total_cpu += 40;
            snap.cache.total_copies += 16;
        }
        snap.window = if gen {
            Window { exposure_frac: 0.30 + 0.05 * noise(), gpu_wait_frac: 0.55 + 0.05 * noise(), ms_per_token: 1e3 / last_decode }
        } else {
            Window::default()
        };
        // A requisição termina na volta do ciclo para "ocioso" (phase acabou de virar 0 acima).
        if phase == 0 && t < 2.0 && snap.requests.first().map_or(true, |r| r.started_ms != req_start) && start.elapsed().as_secs_f64() > 16.0 {
            snap.requests.insert(
                0,
                Request {
                    started_ms: req_start,
                    prompt_tokens: prompt,
                    reused_tokens: 0,
                    output_tokens: answer,
                    duration_ms: now_ms().saturating_sub(req_start),
                    prefill_tok_s: 1_480.0,
                    decode_tok_s: last_decode,
                    status: "ok".into(),
                },
            );
            snap.requests.truncate(5);
        }
        publisher.publish(&snap).expect("publicar");
        std::thread::sleep(Duration::from_millis(250));
    }
    println!("fim da demonstração; o monitor passa a mostrar o runtime como parado");
}
