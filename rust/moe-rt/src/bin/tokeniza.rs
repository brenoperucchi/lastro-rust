//! Uso manual do tokenizador (degrau 5a), só CPU: lê texto da entrada padrão e imprime ids separados por espaço.
//!
//!   tokeniza <tokenizer.json>                      texto cru -> ids
//!   tokeniza <tokenizer.json> --chat <data>        o texto vira a mensagem de usuário, renderizada pelo template com a data
//!                                                  (ex. "November 07, 2026") e add_generation_prompt
//!   tokeniza <tokenizer.json> --decodifica         ids (separados por espaço) -> texto
use moe_rt::tokenizador::{renderiza_chat, Mensagem, Papel, Tokenizador};
use std::io::{Read, Write};
use std::path::Path;
use std::process::ExitCode;

fn executa(args: &[String]) -> Result<Vec<u8>, String> {
    let caminho = args.first().ok_or("uso: tokeniza <tokenizer.json> [--chat <data> | --decodifica]")?;
    let tok = Tokenizador::carrega(Path::new(caminho)).map_err(|e| e.to_string())?;
    let mut entrada = Vec::new();
    std::io::stdin().read_to_end(&mut entrada).map_err(|e| format!("entrada padrão: {e}"))?;
    let texto = String::from_utf8(entrada).map_err(|_| "a entrada não é UTF-8 válido".to_string())?;
    let modo = args.get(1).map(String::as_str);
    if modo == Some("--decodifica") {
        let ids: Vec<u32> = texto.split_whitespace().map(|s| s.parse().map_err(|_| format!("id inválido: {s:?}"))).collect::<Result<_, _>>()?;
        return tok.decodifica_bytes(&ids).map_err(|e| e.to_string());
    }
    let alvo = if modo == Some("--chat") {
        let data = args.get(2).ok_or("--chat exige a data")?;
        let texto = texto.strip_suffix('\n').unwrap_or(&texto).to_string();
        renderiza_chat(&[Mensagem { papel: Papel::Usuario, conteudo: texto }], true, data).map_err(|e| e.to_string())?
    } else if modo.is_some() {
        return Err(format!("opção desconhecida: {}", modo.unwrap_or("")));
    } else {
        texto
    };
    let ids: Vec<String> = tok.codifica(&alvo).iter().map(u32::to_string).collect();
    Ok(format!("{}\n", ids.join(" ")).into_bytes())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match executa(&args) {
        Ok(saida) => {
            let _ = std::io::stdout().write_all(&saida);
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("tokeniza: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod testes {
    use super::*;

    #[test]
    fn erros_sem_panic() {
        assert!(executa(&[]).is_err());
        assert!(executa(&["/caminho/que/nao/existe.json".into()]).is_err());
    }
}
