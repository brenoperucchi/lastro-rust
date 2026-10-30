//! Monitor de console do runtime MoE, na disposição do monitor de console do Strata.
//!
//!   moe-monitor [--metrics <arquivo>] [--interval <s>] [--gpu <n>] [--once [--view <1-7>] [--size <colunas>x<linhas>]]
//!
//! Lê as métricas que o runtime publica (padrão `$XDG_RUNTIME_DIR/moe-rt/metrics.json`) e a telemetria da máquina;
//! não escreve nada. Teclas: 1-7 visões, / filtro, espaço congela, r atualiza, t tema, b navegador, ? ajuda, q sai.
use moe_monitor::snapshot::{default_path, now_ms, read};
use moe_monitor::sys::Sampler;
use moe_monitor::ui::{Action, App};
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::{self, Event, KeyEventKind};
use ratatui::Terminal;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn main() -> std::io::Result<()> {
    let mut path = default_path();
    let mut interval = 1.0f64;
    let mut gpu = 0u32;
    let mut once = false;
    let mut size: Option<(u16, u16)> = None;
    let mut view = '1';
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--metrics" => path = PathBuf::from(args.next().expect("--metrics <arquivo>")),
            "--interval" => interval = args.next().and_then(|v| v.parse().ok()).expect("--interval <segundos>"),
            "--gpu" => gpu = args.next().and_then(|v| v.parse().ok()).expect("--gpu <índice>"),
            "--once" => once = true,
            "--view" => view = args.next().and_then(|v| v.chars().next()).filter(|c| ('1'..='7').contains(c)).expect("--view <1-7>"),
            "--size" => {
                let v = args.next().expect("--size <colunas>x<linhas>");
                let (w, h) = v.split_once('x').expect("--size <colunas>x<linhas>");
                size = Some((w.parse().expect("colunas"), h.parse().expect("linhas")));
            }
            "-h" | "--help" => {
                println!("moe-monitor [--metrics <arquivo>] [--interval <s>] [--gpu <n>] [--once [--view <1-7>] [--size <c>x<l>]]");
                return Ok(());
            }
            other => panic!("opção desconhecida: {other}"),
        }
    }
    let mut sampler = Sampler::new(gpu);
    let mut app = App::new(path.display().to_string());

    if once {
        // Uma tela em texto puro, sem TUI: duas amostras para a CPU e o disco terem taxa.
        sampler.sample();
        std::thread::sleep(Duration::from_millis(300));
        app.update(sampler.sample(), read(&path, now_ms()));
        app.on_key(ratatui::crossterm::event::KeyCode::Char(view), ratatui::crossterm::event::KeyModifiers::NONE);
        let (w, h) = size.unwrap_or_else(|| ratatui::crossterm::terminal::size().unwrap_or((120, 40)));
        let mut term = Terminal::new(TestBackend::new(w, h))?;
        term.draw(|f| app.draw(f))?;
        let buf = term.backend().buffer();
        for y in 0..h {
            let line: String = (0..w).map(|x| buf[(x, y)].symbol().to_string()).collect();
            println!("{}", line.trim_end());
        }
        return Ok(());
    }

    let mut terminal = ratatui::init();
    sampler.sample();
    app.update(sampler.sample(), read(&path, now_ms()));
    let tick = Duration::from_secs_f64(interval.max(0.1));
    let mut last = Instant::now();
    let result = loop {
        if let Err(e) = terminal.draw(|f| app.draw(f)) {
            break Err(e);
        }
        let timeout = tick.saturating_sub(last.elapsed());
        let mut refresh = false;
        match event::poll(timeout) {
            Ok(true) => match event::read() {
                Ok(Event::Key(k)) if k.kind == KeyEventKind::Press => {
                    refresh = app.on_key(k.code, k.modifiers) == Action::Refresh;
                }
                Ok(_) => {}
                Err(e) => break Err(e),
            },
            Ok(false) => {}
            Err(e) => break Err(e),
        }
        if app.quit {
            break Ok(());
        }
        if refresh || last.elapsed() >= tick {
            app.update(sampler.sample(), read(&path, now_ms()));
            last = Instant::now();
        }
    };
    ratatui::restore();
    result
}
