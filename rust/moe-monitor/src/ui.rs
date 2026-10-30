//! TUI do monitor, na disposição do monitor de console do Strata: cabeçalho com estado e `[AO VIVO]`, linha de visões
//! 1–7, e por visão o conteúdo; teclas `1-7`, `/` filtro, espaço congela, `r` atualiza, `t` tema, `b` navegador,
//! `?` ajuda, `q` sai. O [`App`] guarda o estado e trata as teclas sem tocar no terminal (testável com `TestBackend`).
use crate::fmt::{clock_of, compact, duration, num};
use crate::snapshot::{now_ms, Reading, Snapshot, State};
use crate::sys::Sample;
use ratatui::crossterm::event::{KeyCode, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Clear, Padding, Paragraph, Row, Sparkline, Table, Wrap};
use ratatui::Frame;
use std::collections::{HashMap, VecDeque};

const HISTORY: usize = 300;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    Overview,
    Serving,
    Requests,
    Engine,
    Hardware,
    Logs,
    Connect,
}

pub const VIEWS: [(View, &str); 7] = [
    (View::Overview, "Visão geral"),
    (View::Serving, "Serviço"),
    (View::Requests, "Requisições"),
    (View::Engine, "Motor"),
    (View::Hardware, "Hardware"),
    (View::Logs, "Logs"),
    (View::Connect, "Conectar"),
];

fn view_name(v: View) -> &'static str {
    VIEWS.iter().find(|(x, _)| *x == v).map(|(_, n)| *n).unwrap_or("")
}

struct Theme {
    accent: Color,
    blue: Color,
    amber: Color,
    red: Color,
    dim: Color,
    track: Color,
    text: Color,
}

// Paleta própria (laranja-ferrugem), para não confundir com o verde-água do monitor do Strata.
const DARK: Theme = Theme {
    accent: Color::Rgb(255, 138, 61),
    blue: Color::Rgb(110, 190, 255),
    amber: Color::Rgb(240, 210, 100),
    red: Color::Rgb(255, 99, 99),
    dim: Color::Rgb(135, 135, 140),
    track: Color::Rgb(50, 44, 40),
    text: Color::Rgb(232, 228, 224),
};

const LIGHT: Theme = Theme {
    accent: Color::Rgb(194, 65, 12),
    blue: Color::Rgb(3, 105, 161),
    amber: Color::Rgb(161, 98, 7),
    red: Color::Rgb(185, 28, 28),
    dim: Color::Rgb(105, 100, 98),
    track: Color::Rgb(231, 222, 214),
    text: Color::Rgb(28, 25, 23),
};

/// Séries recentes para os gráficos.
#[derive(Default)]
pub struct History {
    series: HashMap<&'static str, VecDeque<f64>>,
}

impl History {
    pub fn push(&mut self, key: &'static str, v: f64) {
        let s = self.series.entry(key).or_default();
        if s.len() == HISTORY {
            s.pop_front();
        }
        s.push_back(if v.is_finite() { v.max(0.0) } else { 0.0 });
    }

    /// Últimas `n` amostras escaladas para o `Sparkline` (u64), e o máximo na mesma escala.
    fn bars(&self, key: &str, n: usize, max: Option<f64>) -> (Vec<u64>, u64) {
        let Some(s) = self.series.get(key) else { return (vec![], 1) };
        let tail: Vec<f64> = s.iter().skip(s.len().saturating_sub(n)).copied().collect();
        let top = max.unwrap_or_else(|| tail.iter().cloned().fold(0.0, f64::max)).max(1e-9);
        let scale = 1000.0 / top;
        (tail.iter().map(|v| (v * scale).round() as u64).collect(), 1000)
    }

    pub fn record(&mut self, sys: &Sample, rt: Option<&Snapshot>) {
        if let Some(g) = &sys.gpu {
            self.push("gpu", g.load_pct);
            self.push("vram", g.vram_used_gb);
            self.push("temp", g.temp_c);
            self.push("power", g.power_w);
            self.push("pcie", g.pcie_to_gpu_mbs);
        }
        self.push("cpu", sys.host.cpu_pct);
        self.push("ram", sys.host.ram_used_gb);
        self.push("disk", sys.host.disk_read_mbs);
        if let Some(s) = rt {
            let reading = matches!(s.state, State::Reading { .. });
            self.push("speed", if reading { s.speed.prefill_now } else { s.speed.decode_now });
            self.push("hit", s.cache.hit_frac * 100.0);
            self.push("h2d", s.cache.h2d_gbs);
            self.push("cpu_exp", s.cache.cpu_frac * 100.0);
            self.push("exposure", s.window.exposure_frac * 100.0);
        }
    }
}

pub struct App {
    pub view: View,
    pub frozen: bool,
    pub dark: bool,
    pub help: bool,
    pub filter: String,
    pub editing: bool,
    pub quit: bool,
    pub status: String,
    pub history: History,
    pub sys: Sample,
    pub rt: Reading,
    pub metrics_path: String,
}

/// O que o laço principal deve fazer depois de uma tecla.
#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    None,
    Refresh,
}

impl App {
    pub fn new(metrics_path: String) -> Self {
        Self {
            view: View::Overview,
            frozen: false,
            dark: true,
            help: false,
            filter: String::new(),
            editing: false,
            quit: false,
            status: String::new(),
            history: History::default(),
            sys: Sample::default(),
            rt: Reading::Absent,
            metrics_path,
        }
    }

    /// Nova amostra (ignorada enquanto congelado).
    pub fn update(&mut self, sys: Sample, rt: Reading) {
        if self.frozen {
            return;
        }
        self.history.record(&sys, live(&rt));
        self.sys = sys;
        self.rt = rt;
    }

    pub fn on_key(&mut self, code: KeyCode, mods: KeyModifiers) -> Action {
        if code == KeyCode::Char('c') && mods.contains(KeyModifiers::CONTROL) {
            self.quit = true;
            return Action::None;
        }
        if self.editing {
            match code {
                KeyCode::Enter => {
                    self.editing = false;
                    self.status = if self.filter.is_empty() { String::new() } else { format!("filtro: {}", self.filter) };
                }
                KeyCode::Esc => {
                    self.editing = false;
                    self.filter.clear();
                    self.status = "filtro limpo".into();
                }
                KeyCode::Backspace => {
                    self.filter.pop();
                }
                KeyCode::Char(c) => self.filter.push(c),
                _ => {}
            }
            return Action::None;
        }
        if self.help {
            self.help = false;
            return Action::None;
        }
        match code {
            KeyCode::Char('q') => self.quit = true,
            KeyCode::Char(c @ '1'..='7') => {
                self.view = VIEWS[(c as u8 - b'1') as usize].0;
                self.status.clear();
            }
            KeyCode::Char('/') => {
                if matches!(self.view, View::Requests | View::Logs) {
                    self.editing = true;
                    self.filter.clear();
                    self.status = "filtro: digite e Enter (Esc limpa)".into();
                } else {
                    self.status = "o filtro vale nas visões 3 Requisições e 6 Logs".into();
                }
            }
            KeyCode::Esc => {
                self.filter.clear();
                self.status.clear();
            }
            KeyCode::Char(' ') => {
                self.frozen = !self.frozen;
                self.status = if self.frozen { "tela congelada (espaço retoma)".into() } else { String::new() };
            }
            KeyCode::Char('r') => {
                self.status = "atualizado".into();
                return Action::Refresh;
            }
            KeyCode::Char('t') => {
                self.dark = !self.dark;
                self.status = if self.dark { "tema escuro".into() } else { "tema claro".into() };
            }
            KeyCode::Char('b') => {
                self.status = "o runtime ainda não tem interface web; quando tiver, b abre no navegador".into();
            }
            KeyCode::Char('?') => self.help = true,
            _ => {}
        }
        Action::None
    }

    fn theme(&self) -> &'static Theme {
        if self.dark {
            &DARK
        } else {
            &LIGHT
        }
    }

    pub fn draw(&self, f: &mut Frame) {
        let th = self.theme();
        let [head, tabs, sep, body, keys, status] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .areas(f.area());
        self.header(f, head);
        self.tabs(f, tabs);
        f.render_widget(Paragraph::new("─".repeat(sep.width as usize)).style(Style::new().fg(th.dim)), sep);
        let body = Rect { y: body.y + 1, height: body.height.saturating_sub(1), ..body };
        // Terminal baixo demais: só cabeçalho e rodapé.
        if body.height > 0 {
            self.body(f, body);
        }
        let hints = "1-7 visões   / filtro   Espaço congela   r atualiza   t tema   b navegador   ? ajuda   q sai";
        f.render_widget(Paragraph::new(hints).style(Style::new().fg(th.dim)), keys);
        let line = if self.editing { format!("/{}▏", self.filter) } else { self.status.clone() };
        f.render_widget(Paragraph::new(line).style(Style::new().fg(th.text)), status);
        if self.help {
            self.help_popup(f);
        }
    }

    fn body(&self, f: &mut Frame, body: Rect) {
        match self.view {
            View::Overview => self.overview(f, body),
            View::Serving => self.serving(f, body),
            View::Requests => self.requests_view(f, body),
            View::Engine => self.engine(f, body),
            View::Hardware => self.hardware(f, body),
            View::Logs => self.logs(f, body),
            View::Connect => self.connect(f, body),
        }
    }

    fn snap(&self) -> Option<&Snapshot> {
        match &self.rt {
            Reading::Live(s) => Some(s),
            Reading::Stale { snap, .. } => Some(snap),
            _ => None,
        }
    }

    fn header(&self, f: &mut Frame, area: Rect) {
        let th = self.theme();
        let s = self.snap();
        let model = s.map_or("sem modelo".to_string(), |s| {
            format!("{} {}", s.model.name, if s.context.max > 0 { compact(s.context.max) } else { String::new() })
        });
        let fixed = "MOE-RT".len() + 2 + view_name(self.view).chars().count() + 2 + 2 + model.chars().count();
        let (dot, what) = match (&self.rt, s.map(|s| &s.state)) {
            (Reading::Absent, _) => (th.dim, "Sem runtime".to_string()),
            (Reading::Invalid(_), _) => (th.red, "Métricas ilegíveis".to_string()),
            (Reading::Stale { .. }, _) => (th.amber, "Runtime parado".to_string()),
            (_, Some(State::Idle)) => (th.dim, "Aguardando requisição".to_string()),
            (_, Some(State::Reading { done, total })) => {
                (th.blue, format!("Lendo o prompt · {}%", num(*done as f64 / (*total).max(1) as f64 * 100.0, 0)))
            }
            (_, Some(State::Generating { .. })) => (th.accent, format!("Gerando · {} tok/s", num(s.unwrap().speed.decode_now, 1))),
            (_, Some(State::Queued { waiting })) => (th.amber, format!("Na fila · {waiting}")),
            (_, Some(State::Error { .. })) => (th.red, "Erro".to_string()),
            _ => (th.dim, String::new()),
        };
        let tag = if self.frozen {
            ("[CONGELADO]", th.blue)
        } else {
            match &self.rt {
                Reading::Live(_) => ("[AO VIVO]", th.accent),
                Reading::Stale { .. } => ("[PARADO]", th.amber),
                Reading::Absent => ("[SEM RUNTIME]", th.dim),
                Reading::Invalid(_) => ("[ERRO]", th.red),
            }
        };
        let right = Line::from(vec![
            Span::styled("● ", Style::new().fg(dot)),
            Span::styled(what, Style::new().fg(th.text)),
            Span::raw("   "),
            Span::styled(tag.0, Style::new().fg(tag.1).add_modifier(Modifier::BOLD)),
        ]);
        let rw = right.width() as u16;
        // O caminho das métricas ocupa o que sobra entre o fixo da esquerda e o estado à direita, cortado pela esquerda.
        let room = (area.width as usize).saturating_sub(rw as usize + fixed + 2);
        let left = Line::from(vec![
            Span::styled("MOE-RT", Style::new().fg(th.accent).add_modifier(Modifier::BOLD)),
            Span::raw("  "),
            Span::styled(view_name(self.view), Style::new().fg(th.accent)),
            Span::raw("  "),
            Span::styled(ellipsize_left(&self.metrics_path, room), Style::new().fg(th.accent)),
            Span::raw("  "),
            Span::styled(model, Style::new().fg(th.text).add_modifier(Modifier::BOLD)),
        ]);
        let [l, r] = Layout::horizontal([Constraint::Min(0), Constraint::Length(rw.min(area.width))]).areas(area);
        f.render_widget(Paragraph::new(left), l);
        f.render_widget(Paragraph::new(right).alignment(Alignment::Right), r);
    }

    fn tabs(&self, f: &mut Frame, area: Rect) {
        let th = self.theme();
        let mut spans = Vec::new();
        for (i, (v, name)) in VIEWS.iter().enumerate() {
            if *v == self.view {
                spans.push(Span::styled(format!("[{} {name}]", i + 1), Style::new().fg(th.accent).add_modifier(Modifier::BOLD)));
            } else {
                spans.push(Span::styled(format!("{} {name}", i + 1), Style::new().fg(th.text)));
            }
            spans.push(Span::raw("  "));
        }
        f.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    fn state_row(&self, f: &mut Frame, area: Rect) {
        let th = self.theme();
        let s = self.snap();
        let active = match s.map(|s| &s.state) {
            Some(State::Idle) | None => 0,
            Some(State::Reading { .. }) => 1,
            Some(State::Generating { .. }) => 2,
            Some(State::Queued { .. }) => 3,
            Some(State::Error { .. }) => 4,
        };
        let mut spans = vec![Span::styled("Estado do modelo  ", Style::new().fg(th.accent).add_modifier(Modifier::BOLD))];
        for (i, n) in ["Ocioso", "Lendo", "Gerando", "Na fila", "Erro"].iter().enumerate() {
            if s.is_some() && i == active {
                spans.push(Span::styled(format!("[{n}]"), Style::new().fg(th.text).add_modifier(Modifier::BOLD)));
            } else {
                spans.push(Span::styled(n.to_string(), Style::new().fg(th.dim)));
            }
            spans.push(Span::raw("  "));
        }
        let (detail, ratio, color) = match s.map(|s| (&s.state, s)) {
            Some((State::Reading { done, total }, _)) => (
                format!("{} / {} tokens", num(*done as f64, 0), num(*total as f64, 0)),
                *done as f64 / (*total).max(1) as f64,
                th.blue,
            ),
            Some((State::Generating { tokens, max_tokens }, s)) => (
                format!("{} tokens · {} tok/s", num(*tokens as f64, 0), num(s.speed.decode_now, 1)),
                *tokens as f64 / (*max_tokens).max(1) as f64,
                th.accent,
            ),
            Some((State::Error { message }, _)) => (message.clone(), 0.0, th.red),
            _ => (String::new(), 0.0, th.accent),
        };
        let [line, bar] = Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(area);
        let dw = detail.chars().count() as u16;
        let [l, r] = Layout::horizontal([Constraint::Min(0), Constraint::Length(dw.min(line.width))]).areas(line);
        f.render_widget(Paragraph::new(Line::from(spans)), l);
        f.render_widget(Paragraph::new(detail).style(Style::new().fg(th.text)).alignment(Alignment::Right), r);
        gauge(f, bar, ratio, color, th.track);
    }

    fn overview(&self, f: &mut Frame, area: Rect) {
        let th = self.theme();
        let s = self.snap();
        let live = live(&self.rt);
        let g = self.sys.gpu.clone().unwrap_or_default();
        let has_gpu = self.sys.gpu.is_some();
        let hv = &self.sys.host;
        let dash = || "—".to_string();
        let [state, _, rest] =
            Layout::vertical([Constraint::Length(2), Constraint::Length(1), Constraint::Min(0)]).areas(area);
        self.state_row(f, state);

        let speed_val = live.map_or("— tok/s".to_string(), |s| match s.state {
            State::Reading { .. } => format!("{} tok/s", num(s.speed.prefill_now, 0)),
            State::Generating { .. } => format!("{} tok/s", num(s.speed.decode_now, 1)),
            _ => format!("{} tok/s", num(s.speed.decode_last, 1)),
        });
        let speed_cap = live.map_or(String::new(), |s| {
            match s.state {
                State::Reading { .. } => "prefill agora",
                State::Generating { .. } => "decode agora",
                _ => "última requisição",
            }
            .to_string()
        });
        let c = live.map(|s| &s.cache);
        let w = live.map(|s| &s.window);
        let cards: Vec<CardSpec> = vec![
            CardSpec::new("Velocidade", speed_val, speed_cap, "speed", None, th.accent),
            CardSpec::new(
                "Carga da GPU",
                if has_gpu { format!("{} %", num(g.load_pct, 0)) } else { dash() },
                short_gpu(&g.name),
                "gpu",
                Some(100.0),
                th.accent,
            ),
            CardSpec::new(
                "VRAM",
                if has_gpu { format!("{} / {} GB", num(g.vram_used_gb, 1), num(g.vram_total_gb, 0)) } else { dash() },
                c.map_or("sem runtime".into(), |c| format!("{} experts na VRAM", num(c.slots_resident as f64, 0))),
                "vram",
                has_gpu.then_some(g.vram_total_gb),
                th.accent,
            ),
            CardSpec::new(
                "Temp. GPU",
                if has_gpu { format!("{} °C", num(g.temp_c, 0)) } else { dash() },
                String::new(),
                "temp",
                Some(95.0),
                th.amber,
            ),
            CardSpec::new(
                "Potência",
                if has_gpu { format!("{} W", num(g.power_w, 0)) } else { dash() },
                if has_gpu { format!("de {} W de limite", num(g.power_limit_w, 0)) } else { String::new() },
                "power",
                has_gpu.then_some(g.power_limit_w),
                th.accent,
            ),
            CardSpec::new(
                "PCIe",
                if has_gpu { format!("Gen{} x{}", g.pcie_gen, g.pcie_width) } else { dash() },
                if has_gpu { format!("para a GPU {} MB/s", num(g.pcie_to_gpu_mbs, 0)) } else { String::new() },
                "pcie",
                None,
                th.accent,
            ),
            CardSpec::new(
                "CPU",
                format!("{} %", num(hv.cpu_pct, 0)),
                format!("{} núcleos · {} threads", hv.cores, hv.threads),
                "cpu",
                Some(100.0),
                th.accent,
            ),
            CardSpec::new(
                "Leitura do disco",
                format!("{} MB/s", num(hv.disk_read_mbs, 1)),
                format!("escrita {} MB/s", num(hv.disk_write_mbs, 1)),
                "disk",
                None,
                th.accent,
            ),
            CardSpec::new(
                "Cache de experts",
                c.map_or_else(dash, |c| format!("{} % acerto", num(c.hit_frac * 100.0, 0))),
                c.map_or(String::new(), |c| {
                    format!("{}/{} slots · {} chegando", num(c.slots_resident as f64, 0), num(c.slots_total as f64, 0), c.slots_loading)
                }),
                "hit",
                Some(100.0),
                th.accent,
            ),
            CardSpec::new(
                "Cópias RAM→VRAM",
                c.map_or_else(dash, |c| format!("{} GB/s", num(c.h2d_gbs, 1))),
                c.map_or(String::new(), |c| format!("{} em voo · {} % dos pedidos", c.copies_inflight, num(c.copy_frac * 100.0, 0))),
                "h2d",
                None,
                th.blue,
            ),
            CardSpec::new(
                "Experts na CPU",
                c.map_or_else(dash, |c| format!("{} % dos pedidos", num(c.cpu_frac * 100.0, 0))),
                c.map_or(String::new(), |c| format!("total {}", num(c.total_cpu as f64, 0))),
                "cpu_exp",
                Some(100.0),
                th.amber,
            ),
            CardSpec::new(
                "Exposição (Fase 0a)",
                w.map_or_else(dash, |w| format!("{} %", num(w.exposure_frac * 100.0, 1))),
                w.map_or(String::new(), |w| {
                    format!("GPU {} % · {} ms/tok", num(w.gpu_wait_frac * 100.0, 0), num(w.ms_per_token, 1))
                }),
                "exposure",
                None,
                th.red,
            ),
        ];

        // Três fileiras de quatro cartões; colunas e altura se ajustam ao terminal, e o que não cabe fica de fora.
        let cols = if rest.width >= 100 { 4 } else if rest.width >= 56 { 2 } else { 1 };
        let rows = cards.len().div_ceil(cols);
        let bottom_h: u16 = 9;
        let avail = rest.height.saturating_sub(bottom_h);
        let card_h: u16 = if avail >= rows as u16 * 6 { 6 } else { 5 };
        let fit_rows = ((avail / card_h) as usize).min(rows).max(1.min(rows));
        let card_w = (rest.width / cols as u16).min(32);
        for (ri, row) in cards.chunks(cols).take(fit_rows).enumerate() {
            for (ci, card) in row.iter().enumerate() {
                let r = Rect {
                    x: rest.x + ci as u16 * (card_w + 2).min(rest.width / cols as u16),
                    y: rest.y + ri as u16 * card_h,
                    width: card_w.saturating_sub(1).max(1),
                    height: card_h,
                };
                if r.right() <= rest.right() && r.bottom() <= rest.bottom() {
                    self.card(f, r, card);
                }
            }
        }
        let used = fit_rows as u16 * card_h + 1;
        if rest.height <= used {
            return;
        }
        let bottom = Rect { y: rest.y + used, height: rest.height - used, ..rest };
        let [left, gap, right] =
            Layout::horizontal([Constraint::Percentage(42), Constraint::Length(3), Constraint::Min(0)]).areas(bottom);
        let _ = gap;
        self.bars(f, left, s);
        self.recent(f, right, 6);
    }

    fn card(&self, f: &mut Frame, area: Rect, c: &CardSpec) {
        let th = self.theme();
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(th.dim))
            .title(Line::from(format!(" {} ", c.title)).style(Style::new().fg(th.text)))
            .padding(Padding::horizontal(1));
        let inner = block.inner(area);
        f.render_widget(block, area);
        let [v, cap, spark] =
            Layout::vertical([Constraint::Length(1), Constraint::Length(1), Constraint::Min(0)]).areas(inner);
        f.render_widget(Paragraph::new(c.value.clone()).style(Style::new().fg(th.text).add_modifier(Modifier::BOLD)), v);
        f.render_widget(Paragraph::new(c.caption.clone()).style(Style::new().fg(th.dim)), cap);
        if spark.height > 0 {
            let (data, max) = self.history.bars(c.series, spark.width as usize, c.max);
            f.render_widget(Sparkline::default().data(&data).max(max).style(Style::new().fg(c.color)), spark);
        }
    }

    fn bars(&self, f: &mut Frame, area: Rect, s: Option<&Snapshot>) {
        let th = self.theme();
        let hv = &self.sys.host;
        let mut items: Vec<(String, String, f64, Color)> = Vec::new();
        if let Some(s) = s {
            let frac = s.context.used as f64 / s.context.max.max(1) as f64;
            items.push((
                "Contexto".into(),
                format!("{} / {}  {}%", compact(s.context.used), compact(s.context.max), num(frac * 100.0, 0)),
                frac,
                th.accent,
            ));
            let gb = s.cache.slots_resident as f64 * expert_gb(s);
            items.push((
                "Experts na VRAM".into(),
                format!("{} · {} GB", num(s.cache.slots_resident as f64, 0), num(gb, 1)),
                s.cache.slots_resident as f64 / s.cache.slots_total.max(1) as f64,
                th.accent,
            ));
        } else {
            items.push(("Contexto".into(), "—".into(), 0.0, th.accent));
            items.push(("Experts na VRAM".into(), "—".into(), 0.0, th.accent));
        }
        let ram = hv.ram_used_gb / hv.ram_total_gb.max(1e-9);
        items.push((
            "RAM do sistema".into(),
            format!("{} / {} GB", num(hv.ram_used_gb, 1), num(hv.ram_total_gb, 0)),
            ram,
            if ram > 0.85 { th.red } else { th.accent },
        ));
        for (i, (label, val, ratio, color)) in items.iter().enumerate() {
            let y = area.y + i as u16 * 2;
            if y + 1 >= area.bottom() {
                break;
            }
            let line = Line::from(vec![
                Span::styled(format!("{label:<17}"), Style::new().fg(if i == 0 { th.accent } else { th.text }).add_modifier(
                    if i == 0 { Modifier::BOLD } else { Modifier::empty() },
                )),
                Span::styled(val.clone(), Style::new().fg(th.dim)),
            ]);
            f.render_widget(Paragraph::new(line), Rect { y, height: 1, ..area });
            gauge(f, Rect { y: y + 1, height: 1, ..area }, *ratio, *color, th.track);
        }
    }

    fn filtered_requests(&self) -> Vec<&crate::snapshot::Request> {
        let Some(s) = self.snap() else { return vec![] };
        let q = self.filter.to_lowercase();
        s.requests
            .iter()
            .filter(|r| q.is_empty() || r.status.to_lowercase().contains(&q) || clock_of(r.started_ms).contains(&q))
            .collect()
    }

    fn recent(&self, f: &mut Frame, area: Rect, limit: usize) {
        let th = self.theme();
        if area.height < 2 {
            return;
        }
        f.render_widget(
            Paragraph::new("Requisições recentes").style(Style::new().fg(th.accent).add_modifier(Modifier::BOLD)),
            Rect { height: 1, ..area },
        );
        let table_area = Rect { y: area.y + 1, height: area.height - 1, ..area };
        let reqs: Vec<_> = self.filtered_requests().into_iter().take(limit).collect();
        self.request_table(f, table_area, &reqs);
    }

    fn request_table(&self, f: &mut Frame, area: Rect, reqs: &[&crate::snapshot::Request]) {
        let th = self.theme();
        let header = Row::new(["HORA", "STATUS", "PROMPT", "REUSADO", "SAÍDA", "TOK/S", "DURAÇÃO"])
            .style(Style::new().fg(th.text));
        let rows: Vec<Row> = reqs
            .iter()
            .map(|r| {
                let st_color = match r.status.as_str() {
                    "ok" | "concluída" => th.accent,
                    "em curso" => th.blue,
                    "erro" => th.red,
                    _ => th.amber,
                };
                Row::new(vec![
                    Span::raw(clock_of(r.started_ms)),
                    Span::styled(r.status.clone(), Style::new().fg(st_color)),
                    Span::raw(num(r.prompt_tokens as f64, 0)),
                    Span::raw(num(r.reused_tokens as f64, 0)),
                    Span::raw(num(r.output_tokens as f64, 0)),
                    Span::raw(if r.output_tokens > 1 { num(r.decode_tok_s, 1) } else { "--".into() }),
                    Span::raw(duration(r.duration_ms)),
                ])
                .style(Style::new().fg(th.text))
            })
            .collect();
        let widths = [
            Constraint::Length(9),
            Constraint::Length(10),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Length(9),
        ];
        if rows.is_empty() {
            f.render_widget(Table::new(Vec::<Row>::new(), widths).header(header), Rect { height: 1.min(area.height), ..area });
            if area.height > 1 {
                let msg = if self.snap().is_none() { "sem runtime" } else { "nenhuma requisição" };
                f.render_widget(Paragraph::new(msg).style(Style::new().fg(th.dim)), Rect { y: area.y + 1, height: 1, ..area });
            }
            return;
        }
        f.render_widget(Table::new(rows, widths).header(header).column_spacing(1), area);
    }

    fn serving(&self, f: &mut Frame, area: Rect) {
        let th = self.theme();
        let [state, _, rest] =
            Layout::vertical([Constraint::Length(2), Constraint::Length(1), Constraint::Min(0)]).areas(area);
        self.state_row(f, state);
        let Some(s) = self.snap() else {
            return self.no_runtime(f, rest);
        };
        let up = now_ms().saturating_sub(s.started_ms);
        let rows = vec![
            kv(th, "Modelo", s.model.name.clone()),
            kv(th, "Camadas × experts", format!("{} × {} (k = {})", s.model.layers, s.model.experts, s.model.experts_per_tok)),
            kv(th, "Processo", format!("pid {} · no ar há {}", s.pid, if s.started_ms > 0 { duration(up) } else { "—".into() })),
            kv(th, "Decode agora / última", format!("{} / {} tok/s", num(s.speed.decode_now, 1), num(s.speed.decode_last, 1))),
            kv(th, "Prefill agora / requisição", format!("{} / {} tok/s", num(s.speed.prefill_now, 0), num(s.speed.prefill_request, 0))),
            kv(th, "Contexto", format!("{} / {} tokens", num(s.context.used as f64, 0), num(s.context.max as f64, 0))),
            kv(th, "Requisições na lista", s.requests.len().to_string()),
        ];
        let [info, chart] = Layout::vertical([Constraint::Length(rows.len() as u16 + 1), Constraint::Min(0)]).areas(rest);
        f.render_widget(Paragraph::new(rows), info);
        self.chart(f, chart, "Velocidade (tok/s)", "speed", None, th.accent);
    }

    fn requests_view(&self, f: &mut Frame, area: Rect) {
        let th = self.theme();
        let title = if self.filter.is_empty() {
            "Requisições".to_string()
        } else {
            format!("Requisições · filtro \"{}\"", self.filter)
        };
        f.render_widget(
            Paragraph::new(title).style(Style::new().fg(th.accent).add_modifier(Modifier::BOLD)),
            Rect { height: 1, ..area },
        );
        let reqs = self.filtered_requests();
        self.request_table(f, Rect { y: area.y + 1, height: area.height.saturating_sub(1), ..area }, &reqs);
    }

    fn engine(&self, f: &mut Frame, area: Rect) {
        let th = self.theme();
        let Some(s) = self.snap() else { return self.no_runtime(f, area) };
        let c = &s.cache;
        let mut rows = vec![
            kv(th, "Slots de expert", format!("{} residentes · {} chegando · {} no total", c.slots_resident, c.slots_loading, c.slots_total)),
            kv(th, "Pedidos (janela)", format!(
                "VRAM {} % · cópia {} % · CPU {} %",
                num(c.hit_frac * 100.0, 1), num(c.copy_frac * 100.0, 1), num(c.cpu_frac * 100.0, 1)
            )),
            kv(th, "Totais", format!(
                "{} acertos · {} cópias · {} na CPU",
                num(c.total_hits as f64, 0), num(c.total_copies as f64, 0), num(c.total_cpu as f64, 0)
            )),
            kv(th, "Cópias RAM→VRAM", format!("{} GB/s · {} em voo", num(c.h2d_gbs, 2), c.copies_inflight)),
            kv(th, "Régua Fase 0(a)", format!(
                "exposição {} % · espera pela GPU {} % · {} ms/token",
                num(s.window.exposure_frac * 100.0, 1), num(s.window.gpu_wait_frac * 100.0, 0), num(s.window.ms_per_token, 2)
            )),
        ];
        for (k, v) in &s.settings {
            rows.push(kv(th, k, v.clone()));
        }
        let [info, charts] = Layout::vertical([Constraint::Length(rows.len() as u16 + 1), Constraint::Min(0)]).areas(area);
        f.render_widget(Paragraph::new(rows), info);
        let [a, b] = Layout::vertical([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(charts);
        let [a1, a2] = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(a);
        let [b1, b2] = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(b);
        self.chart(f, a1, "Acerto na VRAM (%)", "hit", Some(100.0), th.accent);
        self.chart(f, a2, "Cópias RAM→VRAM (GB/s)", "h2d", None, th.blue);
        self.chart(f, b1, "Experts na CPU (%)", "cpu_exp", Some(100.0), th.amber);
        self.chart(f, b2, "Exposição (%)", "exposure", None, th.red);
    }

    fn hardware(&self, f: &mut Frame, area: Rect) {
        let th = self.theme();
        let hv = &self.sys.host;
        let mut rows = match &self.sys.gpu {
            Some(g) => vec![
                kv(th, "GPU", g.name.clone()),
                kv(th, "Carga · VRAM", format!("{} % · {} / {} GB", num(g.load_pct, 0), num(g.vram_used_gb, 2), num(g.vram_total_gb, 1))),
                kv(th, "Temperatura · potência", format!("{} °C · {} W de {} W", num(g.temp_c, 0), num(g.power_w, 0), num(g.power_limit_w, 0))),
                kv(th, "PCIe", format!(
                    "Gen{} x{} · para a GPU {} MB/s · da GPU {} MB/s",
                    g.pcie_gen, g.pcie_width, num(g.pcie_to_gpu_mbs, 0), num(g.pcie_from_gpu_mbs, 0)
                )),
            ],
            None => vec![kv(th, "GPU", "NVML indisponível".into())],
        };
        rows.push(kv(th, "CPU", format!("{} % · {} núcleos · {} threads", num(hv.cpu_pct, 0), hv.cores, hv.threads)));
        rows.push(kv(th, "RAM", format!("{} / {} GB", num(hv.ram_used_gb, 1), num(hv.ram_total_gb, 0))));
        rows.push(kv(th, "Disco", format!("leitura {} MB/s · escrita {} MB/s", num(hv.disk_read_mbs, 1), num(hv.disk_write_mbs, 1))));
        let [info, charts] = Layout::vertical([Constraint::Length(rows.len() as u16 + 1), Constraint::Min(0)]).areas(area);
        f.render_widget(Paragraph::new(rows), info);
        let g = self.sys.gpu.clone().unwrap_or_default();
        let specs: [(&str, &'static str, Option<f64>, Color); 8] = [
            ("Carga da GPU (%)", "gpu", Some(100.0), th.accent),
            ("VRAM (GB)", "vram", Some(g.vram_total_gb.max(1.0)), th.accent),
            ("Temperatura (°C)", "temp", Some(95.0), th.amber),
            ("Potência (W)", "power", Some(g.power_limit_w.max(1.0)), th.accent),
            ("PCIe para a GPU (MB/s)", "pcie", None, th.blue),
            ("CPU (%)", "cpu", Some(100.0), th.accent),
            ("RAM (GB)", "ram", Some(hv.ram_total_gb.max(1.0)), th.accent),
            ("Leitura do disco (MB/s)", "disk", None, th.blue),
        ];
        let rows4 = Layout::vertical([Constraint::Ratio(1, 4); 4]).split(charts);
        for (i, spec) in specs.iter().enumerate() {
            let halves = Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).split(rows4[i / 2]);
            self.chart(f, halves[i % 2], spec.0, spec.1, spec.2, spec.3);
        }
    }

    fn logs(&self, f: &mut Frame, area: Rect) {
        let th = self.theme();
        let Some(s) = self.snap() else { return self.no_runtime(f, area) };
        let q = self.filter.to_lowercase();
        let lines: Vec<Line> = s
            .events
            .iter()
            .filter(|e| q.is_empty() || e.message.to_lowercase().contains(&q) || e.level.contains(&q))
            .map(|e| {
                let color = match e.level.as_str() {
                    "erro" => th.red,
                    "aviso" => th.amber,
                    _ => th.dim,
                };
                Line::from(vec![
                    Span::styled(format!("{} ", clock_of(e.ts_ms)), Style::new().fg(th.dim)),
                    Span::styled(format!("{:<6}", e.level), Style::new().fg(color)),
                    Span::styled(e.message.clone(), Style::new().fg(th.text)),
                ])
            })
            .collect();
        if lines.is_empty() {
            f.render_widget(Paragraph::new("nenhum evento").style(Style::new().fg(th.dim)), area);
            return;
        }
        // Mais recentes embaixo: mostra o fim da lista.
        let skip = lines.len().saturating_sub(area.height as usize);
        f.render_widget(Paragraph::new(lines.into_iter().skip(skip).collect::<Vec<_>>()), area);
    }

    fn connect(&self, f: &mut Frame, area: Rect) {
        let th = self.theme();
        let text = vec![
            kv(th, "Métricas do runtime", self.metrics_path.clone()),
            kv(th, "Como o runtime publica", "JSON inteiro por temporário + rename, várias vezes por segundo".into()),
            kv(th, "Esquema", format!("versão {} (moe_monitor::snapshot)", crate::snapshot::SCHEMA)),
            kv(th, "API HTTP", "o runtime ainda não tem; quando tiver, aparece aqui".into()),
            Line::raw(""),
            kv(th, "Ver a tela com dados de teste", "moe-metrics-demo --seconds 120".into()),
            kv(th, "Outro arquivo de métricas", "moe-monitor --metrics <arquivo>".into()),
            kv(th, "Uma tela só, sem TUI", "moe-monitor --once".into()),
        ];
        f.render_widget(Paragraph::new(text).wrap(Wrap { trim: false }), area);
    }

    fn chart(&self, f: &mut Frame, area: Rect, title: &str, series: &'static str, max: Option<f64>, color: Color) {
        let th = self.theme();
        if area.height < 3 || area.width < 6 {
            return;
        }
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(th.dim))
            .title(Line::from(format!(" {title} ")).style(Style::new().fg(th.text)));
        let inner = block.inner(area);
        f.render_widget(block, area);
        let (data, m) = self.history.bars(series, inner.width as usize, max);
        f.render_widget(Sparkline::default().data(&data).max(m).style(Style::new().fg(color)), inner);
    }

    fn no_runtime(&self, f: &mut Frame, area: Rect) {
        let th = self.theme();
        let msg = match &self.rt {
            Reading::Invalid(e) => format!("métricas ilegíveis: {e}"),
            _ => format!("sem runtime: nenhuma publicação em {}", self.metrics_path),
        };
        f.render_widget(Paragraph::new(msg).style(Style::new().fg(th.dim)).wrap(Wrap { trim: false }), area);
    }

    fn help_popup(&self, f: &mut Frame) {
        let th = self.theme();
        let area = f.area();
        let w = 52.min(area.width);
        let h = 14.min(area.height);
        let r = Rect { x: area.x + (area.width - w) / 2, y: area.y + (area.height - h) / 2, width: w, height: h };
        let lines: Vec<Line> = [
            ("1-7", "troca de visão"),
            ("/", "filtra (Requisições e Logs); Enter aplica, Esc limpa"),
            ("Espaço", "congela / retoma a tela"),
            ("r", "atualiza agora"),
            ("t", "tema claro / escuro"),
            ("b", "abre no navegador (quando houver interface web)"),
            ("?", "esta ajuda"),
            ("q, Ctrl+C", "sai"),
        ]
        .iter()
        .map(|(k, d)| {
            Line::from(vec![
                Span::styled(format!(" {k:<10}"), Style::new().fg(th.accent).add_modifier(Modifier::BOLD)),
                Span::styled(d.to_string(), Style::new().fg(th.text)),
            ])
        })
        .collect();
        f.render_widget(Clear, r);
        f.render_widget(
            Paragraph::new(lines).wrap(Wrap { trim: false }).block(
                Block::bordered()
                    .border_type(BorderType::Rounded)
                    .border_style(Style::new().fg(th.accent))
                    .title(" Ajuda · qualquer tecla fecha "),
            ),
            r,
        );
    }
}

struct CardSpec {
    title: &'static str,
    value: String,
    caption: String,
    series: &'static str,
    max: Option<f64>,
    color: Color,
}

impl CardSpec {
    fn new(title: &'static str, value: String, caption: String, series: &'static str, max: Option<f64>, color: Color) -> Self {
        Self { title, value, caption, series, max, color }
    }
}

/// Barra horizontal: parte cheia na cor, resto na cor de trilho (o `Gauge` do ratatui reserva o meio para o rótulo).
fn gauge(f: &mut Frame, area: Rect, ratio: f64, color: Color, track: Color) {
    if area.height == 0 || area.width == 0 {
        return;
    }
    let w = area.width as usize;
    let eighths = (ratio.clamp(0.0, 1.0) * w as f64 * 8.0).round() as usize;
    let (full, part) = (eighths / 8, eighths % 8);
    let partial = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"][part];
    let rest = w - full - usize::from(part > 0);
    let line = Line::from(vec![
        Span::styled("█".repeat(full), Style::new().fg(color)),
        Span::styled(partial, Style::new().fg(color).bg(track)),
        Span::styled(" ".repeat(rest), Style::new().bg(track)),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

fn kv(th: &Theme, k: &str, v: String) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{k:<28}"), Style::new().fg(th.accent)),
        Span::styled(v, Style::new().fg(th.text)),
    ])
}

/// Corta pela esquerda com "…" para caber em `w` caracteres (o fim do caminho é o que identifica).
pub fn ellipsize_left(s: &str, w: usize) -> String {
    let n = s.chars().count();
    if n <= w {
        return s.to_string();
    }
    if w < 2 {
        return String::new();
    }
    let tail: String = s.chars().skip(n - (w - 1)).collect();
    format!("…{tail}")
}

fn short_gpu(name: &str) -> String {
    name.trim_start_matches("NVIDIA ").trim_start_matches("GeForce ").to_string()
}

/// Chave de `settings` com o tamanho de um expert em MB (o runtime publica; a visão Motor mostra).
pub const EXPERT_MB_KEY: &str = "tamanho do expert (MB)";

/// GB de um expert, de `settings[EXPERT_MB_KEY]`; 0 se ausente.
fn expert_gb(s: &Snapshot) -> f64 {
    s.settings.get(EXPERT_MB_KEY).and_then(|v| v.replace(',', ".").parse::<f64>().ok()).map_or(0.0, |mb| mb / 1e3)
}

pub fn live(rt: &Reading) -> Option<&Snapshot> {
    match rt {
        Reading::Live(s) => Some(s),
        _ => None,
    }
}
