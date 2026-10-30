//! Formatação para a tela: números em pt-br, contagens compactas e hora local.

/// Número com vírgula decimal e ponto de milhar (pt-br); "—" se não for finito.
pub fn num(v: f64, decimals: usize) -> String {
    if !v.is_finite() {
        return "—".into();
    }
    let s = format!("{:.*}", decimals, v.abs());
    let (int, frac) = s.split_once('.').map_or((s.as_str(), None), |(a, b)| (a, Some(b)));
    let mut grouped = String::new();
    for (i, c) in int.chars().enumerate() {
        if i > 0 && (int.len() - i) % 3 == 0 {
            grouped.push('.');
        }
        grouped.push(c);
    }
    let sign = if v < 0.0 && v.abs() >= 0.5 * 10f64.powi(-(decimals as i32)) { "-" } else { "" };
    match frac {
        Some(f) => format!("{sign}{grouped},{f}"),
        None => format!("{sign}{grouped}"),
    }
}

/// Contagem compacta como no Strata: 5.123 → "5K", 262.144 → "256K", 1.500.000 → "1,5M" (base 1024 para K, como contexto).
pub fn compact(v: u64) -> String {
    if v >= 1 << 20 {
        let m = v as f64 / (1u64 << 20) as f64;
        if m >= 10.0 { format!("{}M", num(m, 0)) } else { format!("{}M", num(m, 1)) }
    } else if v >= 1 << 10 {
        format!("{}K", v >> 10)
    } else {
        v.to_string()
    }
}

/// Duração curta: "0,4 s", "5,1 s", "2 min 05 s".
pub fn duration(ms: u64) -> String {
    if ms < 60_000 {
        format!("{} s", num(ms as f64 / 1e3, 1))
    } else {
        format!("{} min {:02} s", ms / 60_000, (ms / 1000) % 60)
    }
}

/// HH:MM:SS no fuso local (deslocamento lido uma vez de `date +%z`; UTC se falhar).
pub fn clock_of(ms: u64) -> String {
    let secs = (ms / 1000) as i64 + local_offset_secs();
    let d = secs.rem_euclid(86_400);
    format!("{:02}:{:02}:{:02}", d / 3600, (d % 3600) / 60, d % 60)
}

fn local_offset_secs() -> i64 {
    use std::sync::OnceLock;
    static OFF: OnceLock<i64> = OnceLock::new();
    *OFF.get_or_init(|| {
        let out = std::process::Command::new("date").arg("+%z").output().ok();
        let z = out.and_then(|o| String::from_utf8(o.stdout).ok()).unwrap_or_default();
        let z = z.trim();
        if z.len() == 5 {
            let sign = if z.starts_with('-') { -1 } else { 1 };
            let hh: i64 = z[1..3].parse().unwrap_or(0);
            let mm: i64 = z[3..5].parse().unwrap_or(0);
            sign * (hh * 3600 + mm * 60)
        } else {
            0
        }
    })
}
