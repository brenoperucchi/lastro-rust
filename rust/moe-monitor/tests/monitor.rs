//! Testes do monitor: esquema e publicação, leitura de /proc, formatação, teclas e desenho de cada visão.
use moe_monitor::fmt::{compact, duration, num};
use moe_monitor::snapshot::{
    now_ms, read, Event, ExpertCache, Model, Publisher, Reading, Request, Snapshot, State, SCHEMA, STALE_MS,
};
use moe_monitor::sys::{parse_cpu_times, parse_diskstats, parse_meminfo, Gpu, Host, Sample};
use moe_monitor::ui::{ellipsize_left, Action, App, View};
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::{KeyCode, KeyModifiers};
use ratatui::Terminal;
use std::fs;
use std::path::PathBuf;

/// Arquivo de métricas num diretório só do teste; ao sair de escopo apaga o arquivo e o diretório vazio (sem recursão).
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
        if let Some(dir) = self.0.parent() {
            let _ = fs::remove_dir(dir);
        }
    }
}

fn scratch(name: &str) -> Scratch {
    let dir = std::env::temp_dir().join(format!("moe-monitor-test-{}-{name}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    Scratch(dir.join("metrics.json"))
}

fn demo() -> Snapshot {
    Snapshot {
        model: Model { name: "teste".into(), layers: 32, experts: 40, experts_per_tok: 8 },
        state: State::Generating { tokens: 120, max_tokens: 400 },
        cache: ExpertCache { slots_total: 640, slots_resident: 600, hit_frac: 0.8, ..Default::default() },
        ..Default::default()
    }
}

#[test]
fn publicacao_e_lida_como_viva_e_nao_deixa_temporario() {
    let tmp = scratch("vivo");
    let path = &tmp.0;
    let p = Publisher::new(path).unwrap();
    p.publish(&demo()).unwrap();
    match read(path, now_ms()) {
        Reading::Live(s) => {
            assert_eq!(s.schema, SCHEMA);
            assert_eq!(s.pid, std::process::id());
            assert_eq!(s.state, State::Generating { tokens: 120, max_tokens: 400 });
        }
        other => panic!("esperava Live, veio {other:?}"),
    }
    let left: Vec<_> = fs::read_dir(path.parent().unwrap()).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert_eq!(left, vec![std::ffi::OsString::from("metrics.json")]);
}

#[test]
fn publicacao_velha_vira_runtime_parado() {
    let tmp = scratch("velho");
    let path = &tmp.0;
    Publisher::new(path).unwrap().publish(&demo()).unwrap();
    match read(path, now_ms() + STALE_MS + 1_000) {
        Reading::Stale { age_ms, .. } => assert!(age_ms > STALE_MS),
        other => panic!("esperava Stale, veio {other:?}"),
    }
}

#[test]
fn arquivo_ausente_invalido_ou_de_outro_esquema() {
    let tmp = scratch("invalido");
    let path = &tmp.0;
    let _ = fs::remove_file(path);
    assert_eq!(read(path, now_ms()), Reading::Absent);
    fs::write(path, b"{meio json").unwrap();
    assert!(matches!(read(path, now_ms()), Reading::Invalid(_)));
    let mut s = demo();
    s.schema = SCHEMA + 1;
    s.ts_ms = now_ms();
    fs::write(path, serde_json::to_vec(&s).unwrap()).unwrap();
    assert!(matches!(read(path, now_ms()), Reading::Invalid(m) if m.contains("esquema")));
}

#[test]
fn numeros_em_pt_br() {
    assert_eq!(num(1234.5, 1), "1.234,5");
    assert_eq!(num(1_234_567.0, 0), "1.234.567");
    assert_eq!(num(0.0, 0), "0");
    assert_eq!(num(-3.26, 1), "-3,3");
    assert_eq!(num(-0.01, 0), "0");
    assert_eq!(num(f64::NAN, 1), "—");
}

#[test]
fn leitura_de_proc() {
    let stat = "cpu  100 0 50 800 50 0 0 0 0 0\ncpu0 1 2 3 4\n";
    assert_eq!(parse_cpu_times(stat), Some((150, 1000)));
    let mem = "MemTotal:       98000000 kB\nMemFree: 1 kB\nMemAvailable:   48000000 kB\n";
    assert_eq!(parse_meminfo(mem), Some((98_000_000, 48_000_000)));
    let disk = "259 0 nvme0n1 10 0 2000 0 5 0 400 0 0 0 0\n259 1 nvme0n1p1 9 0 1999 0 5 0 399 0 0 0 0\n\
                254 0 dm-0 1 0 777 0 1 0 777 0 0 0 0\n";
    assert_eq!(parse_diskstats(disk, &["nvme0n1".into()]), Some((2000, 400)));
    assert_eq!(parse_diskstats(disk, &["sda".into()]), None);
}

#[test]
fn contagens_e_duracoes() {
    assert_eq!(compact(5_123), "5K");
    assert_eq!(compact(262_144), "256K");
    assert_eq!(compact(900), "900");
    assert_eq!(duration(5_100), "5,1 s");
    assert_eq!(duration(125_000), "2 min 05 s");
    assert_eq!(ellipsize_left("/run/user/1000/moe-rt/metrics.json", 12), "…etrics.json");
    assert_eq!(ellipsize_left("curto", 12), "curto");
}

fn sample() -> Sample {
    Sample {
        gpu: Some(Gpu {
            name: "NVIDIA GeForce RTX 4090".into(),
            vram_used_gb: 3.0,
            vram_total_gb: 24.0,
            power_limit_w: 450.0,
            pcie_gen: 4,
            pcie_width: 16,
            ..Default::default()
        }),
        host: Host { cores: 20, threads: 28, ram_used_gb: 47.0, ram_total_gb: 94.0, ..Default::default() },
    }
}

fn live_snap() -> Snapshot {
    let mut s = demo();
    s.schema = SCHEMA;
    s.ts_ms = now_ms();
    s.requests = vec![
        Request { started_ms: now_ms(), prompt_tokens: 4_784, output_tokens: 800, decode_tok_s: 32.0, duration_ms: 5_100, status: "ok".into(), ..Default::default() },
        Request { started_ms: now_ms(), prompt_tokens: 257, output_tokens: 1, duration_ms: 400, status: "erro".into(), ..Default::default() },
    ];
    s.events = vec![
        Event { ts_ms: now_ms(), level: "info".into(), message: "requisição concluída".into() },
        Event { ts_ms: now_ms(), level: "aviso".into(), message: "fila de cópias cheia".into() },
    ];
    s
}

fn screen(app: &App, w: u16, h: u16) -> String {
    let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
    term.draw(|f| app.draw(f)).unwrap();
    let buf = term.backend().buffer().clone();
    (0..h).map(|y| (0..w).map(|x| buf[(x, y)].symbol().to_string()).collect::<String>() + "\n").collect()
}

fn key(app: &mut App, c: char) -> Action {
    app.on_key(KeyCode::Char(c), KeyModifiers::NONE)
}

#[test]
fn todas_as_visoes_desenham_em_qualquer_tamanho() {
    for rt in [Reading::Live(Box::new(live_snap())), Reading::Absent, Reading::Invalid("x".repeat(300))] {
        for (w, h) in [(200u16, 60u16), (140, 40), (120, 30), (80, 24), (60, 20), (40, 12), (20, 6), (8, 3)] {
            for v in '1'..='7' {
                let mut app = App::new("/run/user/1000/moe-rt/metrics.json".into());
                app.update(sample(), match &rt {
                    Reading::Live(s) => Reading::Live(s.clone()),
                    Reading::Absent => Reading::Absent,
                    Reading::Invalid(e) => Reading::Invalid(e.clone()),
                    Reading::Stale { snap, age_ms } => Reading::Stale { snap: snap.clone(), age_ms: *age_ms },
                });
                key(&mut app, v);
                let _ = screen(&app, w, h);
                key(&mut app, '?');
                let _ = screen(&app, w, h);
            }
        }
    }
}

#[test]
fn visao_geral_mostra_o_que_o_strata_mostra() {
    let mut app = App::new("/run/user/1000/moe-rt/metrics.json".into());
    app.update(sample(), Reading::Live(Box::new(live_snap())));
    let s = screen(&app, 140, 40);
    for want in [
        "MOE-RT", "[AO VIVO]", "[1 Visão geral]", "[Gerando]", "Velocidade", "Carga da GPU", "VRAM", "Temp. GPU", "Potência",
        "PCIe", "Gen4 x16", "CPU", "Leitura do disco", "Cache de experts", "Cópias RAM→VRAM", "Experts na CPU",
        "Exposição (Fase 0a)", "Contexto", "Experts na VRAM", "RAM do sistema", "Requisições recentes", "4.784", "5,1 s",
        "1-7 visões",
    ] {
        assert!(s.contains(want), "faltou {want:?} na tela:\n{s}");
    }
}

#[test]
fn teclas_trocam_visao_congelam_e_saem() {
    let mut app = App::new("m".into());
    assert_eq!(app.view, View::Overview);
    key(&mut app, '3');
    assert_eq!(app.view, View::Requests);
    key(&mut app, '6');
    assert_eq!(app.view, View::Logs);
    assert_eq!(key(&mut app, 'r'), Action::Refresh);
    key(&mut app, 't');
    assert!(!app.dark);
    key(&mut app, '?');
    assert!(app.help);
    key(&mut app, 'x'); // qualquer tecla fecha a ajuda, sem outro efeito
    assert!(!app.help && !app.quit);
    key(&mut app, 'q');
    assert!(app.quit);
    let mut app = App::new("m".into());
    app.on_key(KeyCode::Char('c'), KeyModifiers::CONTROL);
    assert!(app.quit);
}

#[test]
fn espaco_congela_a_tela() {
    let mut app = App::new("m".into());
    app.update(sample(), Reading::Live(Box::new(live_snap())));
    key(&mut app, ' ');
    assert!(app.frozen);
    app.update(Sample::default(), Reading::Absent);
    assert!(matches!(app.rt, Reading::Live(_)), "congelado não deve trocar a amostra");
    assert!(screen(&app, 140, 40).contains("[CONGELADO]"));
    key(&mut app, ' ');
    app.update(Sample::default(), Reading::Absent);
    assert_eq!(app.rt, Reading::Absent);
}

#[test]
fn filtro_nas_requisicoes() {
    let mut app = App::new("m".into());
    app.update(sample(), Reading::Live(Box::new(live_snap())));
    key(&mut app, '3');
    key(&mut app, '/');
    assert!(app.editing);
    for c in "erro".chars() {
        key(&mut app, c); // durante a edição, 'r' e 'o' são texto, não comandos
    }
    app.on_key(KeyCode::Enter, KeyModifiers::NONE);
    assert!(!app.editing);
    assert_eq!(app.filter, "erro");
    let s = screen(&app, 120, 30);
    assert!(s.contains("257") && !s.contains("4.784"), "filtro não aplicado:\n{s}");
    app.on_key(KeyCode::Esc, KeyModifiers::NONE);
    assert!(app.filter.is_empty());
    assert!(screen(&app, 120, 30).contains("4.784"));
}

#[test]
fn filtro_fora_de_requisicoes_e_logs_so_avisa() {
    let mut app = App::new("m".into());
    key(&mut app, '/');
    assert!(!app.editing);
    assert!(app.status.contains("Requisições"));
}
