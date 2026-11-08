//! Etapa 5e: a lógica pura do servidor local `lastro-servidor` (critério criterio-5e-servidor-20261108.md): HTTP/1.1 mínimo e seguro, roteamento,
//! validações e respostas, tudo testável sem rede e sem GPU. A geração é a do 5d (`lastro_cli::executa`, que chama `geracao_regra::gera`): o corpo
//! da resposta do `POST /gerar` é a MESMA saída do comando `lastro` (texto até o primeiro EOS ou N tokens, mais um newline).
//!
//! Um pedido por conexão (`Connection: close`), um cliente por vez. Recusas 4xx saem sem tocar o motor.

use crate::geracao_regra as gr;
use crate::lastro_cli as cli;
use crate::motor::backend::Backend;
use crate::motor::leitura::PreparaPasso;
use crate::primeira_palavra_regra::Sequencia;
use crate::tokenizador::Tokenizador;
use std::fmt::Debug;
use std::io::{self, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Corpo máximo aceito (64 KiB).
pub const LIMITE_CORPO: usize = 64 * 1024;
/// Linha de requisição e conjunto de cabeçalhos pequenos.
pub const LIMITE_LINHA: usize = 2048;
pub const LIMITE_CABECALHOS: usize = 8192;
/// Prazo para completar cabeçalho + corpo.
pub const PRAZO_LEITURA: Duration = Duration::from_secs(10);
pub const USO: &str = "uso: lastro-servidor --modelo <dir> --porta <1..65535> (escuta só em 127.0.0.1)";

// ---------------------------------------------------------------- argumentos

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArgsServidor {
    pub modelo: PathBuf,
    pub porta: u16,
}

/// `--modelo` e `--porta` obrigatórios, nada duplicado, desconhecido nem posicional. Não existe opção de endereço.
pub fn parse_args(v: Vec<String>) -> Result<ArgsServidor, String> {
    let mut modelo: Option<PathBuf> = None;
    let mut porta: Option<u16> = None;
    let mut it = v.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--modelo" => {
                let x = it.next().filter(|x| !x.starts_with("--")).ok_or("--modelo sem valor")?;
                if modelo.replace(PathBuf::from(x)).is_some() {
                    return Err("--modelo duplicado".into());
                }
            }
            "--porta" => {
                let x = it.next().filter(|x| !x.starts_with("--")).ok_or("--porta sem valor")?;
                let n = if !x.is_empty() && x.len() <= 5 && x.bytes().all(|b| b.is_ascii_digit()) { x.parse::<u32>().unwrap_or(0) } else { 0 };
                if !(1..=65535).contains(&n) {
                    return Err(format!("--porta fora de 1..65535: {x}"));
                }
                if porta.replace(n as u16).is_some() {
                    return Err("--porta duplicado".into());
                }
            }
            outro if outro.starts_with("--") => return Err(format!("opção desconhecida: {outro}")),
            outro => return Err(format!("argumento inesperado: {outro}")),
        }
    }
    Ok(ArgsServidor { modelo: modelo.ok_or("falta --modelo")?, porta: porta.ok_or("falta --porta")? })
}

// ---------------------------------------------------------------- respostas

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resposta {
    pub status: u16,
    pub corpo: Vec<u8>,
}

fn razao(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        411 => "Length Required",
        413 => "Payload Too Large",
        422 => "Unprocessable Content",
        500 => "Internal Server Error",
        _ => "Status",
    }
}

impl Resposta {
    pub fn ok(corpo: impl Into<Vec<u8>>) -> Self {
        Self { status: 200, corpo: corpo.into() }
    }
    /// Recusa/erro: corpo de uma linha (sem quebras internas) mais um newline.
    pub fn msg(status: u16, texto: &str) -> Self {
        let uma_linha: String = texto.chars().map(|c| if c == '\n' || c == '\r' { ' ' } else { c }).collect();
        Self { status, corpo: format!("{uma_linha}\n").into_bytes() }
    }
    /// A resposta HTTP/1.1 completa: linha de status, `text/plain; charset=utf-8`, `Content-Length`, `Connection: close`.
    pub fn serializa(&self) -> Vec<u8> {
        let mut v = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.status,
            razao(self.status),
            self.corpo.len()
        )
        .into_bytes();
        v.extend_from_slice(&self.corpo);
        v
    }
}

// ---------------------------------------------------------------- leitura da requisição

/// Uma conexão com leitura de prazo ajustável (o `TcpStream` real; nos testes, um falso).
pub trait Conexao: Read + Write {
    fn limita_leitura(&mut self, restante: Duration) -> io::Result<()>;
}

impl Conexao for TcpStream {
    fn limita_leitura(&mut self, restante: Duration) -> io::Result<()> {
        self.set_read_timeout(Some(restante.max(Duration::from_millis(1))))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rota {
    Saude,
    Gerar,
    Encerrar,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Requisicao {
    pub rota: Rota,
    /// Query crua (depois do `?`, sem decodificar).
    pub query: String,
    pub corpo: Vec<u8>,
    /// Valor do cabeçalho `Host` (None = ausente; se repetido com valores diferentes, fica um marcador que nunca confere).
    pub host: Option<String>,
    /// A requisição trouxe cabeçalho `Origin` (qualquer valor): vem de página de navegador.
    pub origem: bool,
    /// Versão HTTP/1.0 (a única em que a falta de `Host` é aceita).
    pub http10: bool,
}

/// Recusa 403 (sem tocar o motor) de requisição vinda de página de navegador: qualquer `Origin`, ou `Host` diferente de `127.0.0.1:<porta>` / `localhost:<porta>`
/// (porta efetiva do servidor). Não é autenticação: fecha o POST simples de CORS e o DNS rebinding. `Host` ausente: 403 em HTTP/1.1 (a RFC exige Host nele);
/// em HTTP/1.0 o parser já aceita requisição sem Host, então ela passa (um navegador nunca fala 1.0 sem Host).
pub fn confere_acesso(req: &Requisicao, porta: u16) -> Result<(), Resposta> {
    if req.origem {
        return Err(Resposta::msg(403, "requisição com Origin recusada: este servidor não atende páginas de navegador"));
    }
    match &req.host {
        None if req.http10 => Ok(()),
        None => Err(Resposta::msg(403, "Host ausente recusado")),
        Some(h) => {
            let h = h.to_ascii_lowercase();
            if h == format!("127.0.0.1:{porta}") || h == format!("localhost:{porta}") {
                Ok(())
            } else {
                Err(Resposta::msg(403, "Host recusado: use 127.0.0.1 ou localhost com a porta do servidor"))
            }
        }
    }
}

fn roteia(metodo: &str, caminho: &str) -> Result<Rota, Resposta> {
    let (rota, certo) = match caminho {
        "/saude" => (Rota::Saude, "GET"),
        "/gerar" => (Rota::Gerar, "POST"),
        "/encerrar" => (Rota::Encerrar, "POST"),
        _ => return Err(Resposta::msg(404, "caminho desconhecido")),
    };
    if metodo != certo {
        return Err(Resposta::msg(405, &format!("método {metodo} não serve em {caminho}: use {certo}")));
    }
    Ok(rota)
}

fn acha(buf: &[u8], agulha: &[u8]) -> Option<usize> {
    buf.windows(agulha.len()).position(|w| w == agulha)
}

/// Lê UMA requisição, com o prazo total `prazo` para cabeçalho + corpo (estourou = 408). Todas as recusas voltam como `Err(Resposta)`.
/// Nunca lê além do cabeçalho mais `Content-Length` bytes de corpo (e este só até `LIMITE_CORPO`: acima disso é 413 sem ler o corpo).
pub fn le_requisicao<C: Conexao>(c: &mut C, prazo: Duration) -> Result<Requisicao, Resposta> {
    let limite = Instant::now() + prazo;
    let timeout = || Resposta::msg(408, "tempo esgotado: cabeçalho e corpo não chegaram em 10 s");
    let mut buf: Vec<u8> = Vec::with_capacity(1024);
    // Lê mais um pedaço (no máximo `max` bytes) respeitando o prazo.
    let mais = |c: &mut C, buf: &mut Vec<u8>, max: usize| -> Result<usize, Resposta> {
        loop {
            let agora = Instant::now();
            if agora >= limite {
                return Err(timeout());
            }
            c.limita_leitura(limite - agora).map_err(|e| Resposta::msg(400, &format!("conexão: {e}")))?;
            let mut pedaco = [0u8; 1024];
            let n = max.min(pedaco.len());
            match c.read(&mut pedaco[..n]) {
                Ok(0) => return Err(Resposta::msg(400, "requisição incompleta: a conexão fechou antes do fim")),
                Ok(k) => {
                    buf.extend_from_slice(&pedaco[..k]);
                    return Ok(k);
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => return Err(timeout()),
                Err(e) => return Err(Resposta::msg(400, &format!("leitura: {e}"))),
            }
        }
    };
    // Cabeçalho: até CRLF CRLF, com limites. A linha de requisição tem limite próprio.
    let fim_cab = loop {
        if let Some(i) = acha(&buf, b"\r\n\r\n") {
            break i;
        }
        let linha1 = acha(&buf, b"\r\n");
        if linha1.is_none() && buf.len() > LIMITE_LINHA {
            return Err(Resposta::msg(400, "linha de requisição grande demais"));
        }
        if buf.len() > LIMITE_CABECALHOS {
            return Err(Resposta::msg(400, "cabeçalhos grandes demais"));
        }
        // Nunca passa do limite de cabeçalhos (+ 4 do terminador) por mais de um pedaço.
        let folga = LIMITE_CABECALHOS + 4 - buf.len().min(LIMITE_CABECALHOS + 3);
        mais(c, &mut buf, folga)?;
    };
    if fim_cab > LIMITE_CABECALHOS {
        return Err(Resposta::msg(400, "cabeçalhos grandes demais"));
    }
    let cabecalho = std::str::from_utf8(&buf[..fim_cab]).map_err(|_| Resposta::msg(400, "cabeçalho não é UTF-8/ASCII"))?;
    let mut linhas = cabecalho.split("\r\n");
    let primeira = linhas.next().unwrap_or("");
    if primeira.len() > LIMITE_LINHA {
        return Err(Resposta::msg(400, "linha de requisição grande demais"));
    }
    let partes: Vec<&str> = primeira.split(' ').collect();
    if partes.len() != 3 || partes.iter().any(|p| p.is_empty()) {
        return Err(Resposta::msg(400, "linha de requisição malformada"));
    }
    let (metodo, alvo, versao) = (partes[0], partes[1], partes[2]);
    if versao != "HTTP/1.1" && versao != "HTTP/1.0" {
        return Err(Resposta::msg(400, "versão HTTP não suportada"));
    }
    if !metodo.bytes().all(|b| b.is_ascii_uppercase()) || !alvo.starts_with('/') || alvo.bytes().any(|b| b <= b' ' || b == 0x7f) {
        return Err(Resposta::msg(400, "linha de requisição malformada"));
    }
    let (caminho, query) = alvo.split_once('?').unwrap_or((alvo, ""));
    let mut cl: Option<String> = None;
    let mut codificacao = false;
    let mut host: Option<String> = None;
    let mut origem = false;
    for l in linhas {
        let (nome, valor) = l.split_once(':').ok_or_else(|| Resposta::msg(400, "cabeçalho malformado"))?;
        if nome.is_empty() || nome.bytes().any(|b| b <= b' ' || b == 0x7f) {
            return Err(Resposta::msg(400, "cabeçalho malformado"));
        }
        let valor = valor.trim_matches(|ch| ch == ' ' || ch == '\t');
        if nome.eq_ignore_ascii_case("content-length") {
            match &cl {
                Some(antigo) if antigo != valor => return Err(Resposta::msg(400, "Content-Length repetido e diferente")),
                _ => cl = Some(valor.to_string()),
            }
        } else if nome.eq_ignore_ascii_case("transfer-encoding") {
            codificacao = true;
        } else if nome.eq_ignore_ascii_case("host") {
            host = match host {
                Some(antigo) if antigo != valor => Some("?host repetido".to_string()),
                _ => Some(valor.to_string()),
            };
        } else if nome.eq_ignore_ascii_case("origin") {
            origem = true;
        }
    }
    // Método e caminho antes de qualquer leitura de corpo.
    let rota = roteia(metodo, caminho)?;
    let mut corpo = vec![];
    if metodo == "POST" {
        let tam = match &cl {
            None => return Err(Resposta::msg(411, "Content-Length obrigatório no POST")),
            Some(_) if codificacao => return Err(Resposta::msg(400, "Transfer-Encoding não é suportado")),
            Some(s) if s.is_empty() || s.len() > 18 || !s.bytes().all(|b| b.is_ascii_digit()) => return Err(Resposta::msg(400, "Content-Length inválido")),
            Some(s) => s.parse::<usize>().map_err(|_| Resposta::msg(400, "Content-Length inválido"))?,
        };
        if tam > LIMITE_CORPO {
            return Err(Resposta::msg(413, &format!("corpo acima de {LIMITE_CORPO} bytes")));
        }
        // O que já veio junto do cabeçalho (nunca mais que `tam`).
        let ja = &buf[fim_cab + 4..];
        corpo.extend_from_slice(&ja[..ja.len().min(tam)]);
        while corpo.len() < tam {
            let falta = tam - corpo.len();
            let mut resto = vec![];
            mais(c, &mut resto, falta)?;
            corpo.extend_from_slice(&resto);
        }
    }
    Ok(Requisicao { rota, query: query.to_string(), corpo, host, origem, http10: versao == "HTTP/1.0" })
}

// ---------------------------------------------------------------- validações

/// `%XX` e `+` (espaço); o resultado tem de ser UTF-8.
pub fn urldecode(s: &str) -> Result<String, String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'%' => {
                let h = b.get(i + 1..i + 3).and_then(|x| std::str::from_utf8(x).ok()).filter(|x| x.bytes().all(|c| c.is_ascii_hexdigit()));
                let v = h.and_then(|x| u8::from_str_radix(x, 16).ok()).ok_or_else(|| format!("escape % inválido em {s:?}"))?;
                out.push(v);
                i += 3;
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8(out).map_err(|_| "valor da query não é UTF-8".to_string())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PedidoGera {
    pub texto: String,
    pub novos: usize,
    pub data: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Acao {
    Saude,
    Encerrar,
    Gerar(PedidoGera),
}

/// Valida a requisição já lida. `hoje` é a data padrão (`Month DD, YYYY`). Recusas: 400.
pub fn interpreta(req: &Requisicao, hoje: &str) -> Result<Acao, Resposta> {
    match req.rota {
        Rota::Saude => Ok(Acao::Saude),
        Rota::Encerrar => Ok(Acao::Encerrar),
        Rota::Gerar => {
            let mut novos: Option<usize> = None;
            let mut data: Option<String> = None;
            if !req.query.is_empty() {
                for par in req.query.split('&') {
                    let (k, v) = par.split_once('=').ok_or_else(|| Resposta::msg(400, &format!("parâmetro sem valor na query: {par:?}")))?;
                    let v = urldecode(v).map_err(|e| Resposta::msg(400, &e))?;
                    match k {
                        "novos" => {
                            let n = if !v.is_empty() && v.len() <= 6 && v.bytes().all(|b| b.is_ascii_digit()) { v.parse::<usize>().unwrap_or(0) } else { 0 };
                            if !(1..=cli::MAX_NOVOS).contains(&n) {
                                return Err(Resposta::msg(400, &format!("novos fora de 1..{}: {v}", cli::MAX_NOVOS)));
                            }
                            if novos.replace(n).is_some() {
                                return Err(Resposta::msg(400, "novos repetido"));
                            }
                        }
                        "data" => {
                            let d = cli::parse_data(&v).map_err(|e| Resposta::msg(400, &e))?;
                            if data.replace(d).is_some() {
                                return Err(Resposta::msg(400, "data repetida"));
                            }
                        }
                        outro => return Err(Resposta::msg(400, &format!("parâmetro desconhecido na query: {outro}"))),
                    }
                }
            }
            let texto = String::from_utf8(req.corpo.clone()).map_err(|_| Resposta::msg(400, "corpo não é UTF-8"))?;
            if texto.trim().is_empty() {
                return Err(Resposta::msg(400, "entrada vazia ou só espaços"));
            }
            Ok(Acao::Gerar(PedidoGera { texto, novos: novos.unwrap_or(cli::MAX_NOVOS), data: data.unwrap_or_else(|| hoje.to_string()) }))
        }
    }
}

/// prompt (com template) + novos <= 2048, pelo tokenizador. Recusa: 422.
pub fn valida_tamanho(tok: &Tokenizador, p: &PedidoGera) -> Result<(), Resposta> {
    cli::confere_tamanho(tok, &p.texto, &p.data, p.novos).map(|_| ()).map_err(|e| Resposta::msg(422, &e))
}

// ---------------------------------------------------------------- execução

#[derive(Debug, PartialEq, Eq)]
pub enum Fluxo {
    Continua,
    /// `POST /encerrar` atendido: sair com código 0.
    Encerra,
    /// Falha do motor (o 500 já foi respondido): sair com código 1.
    Falha(String),
}

/// Decide a resposta de uma ação. O motor só é tocado em `Gerar` válido; falha do motor = 500 e `Fluxo::Falha`.
pub fn trata<B, P, C>(acao: Acao, tok: &Tokenizador, motor: &mut gr::MotorGeracao<B, P>, cria: C) -> (Resposta, Fluxo)
where
    B: Backend,
    P: PreparaPasso,
    P::Error: Debug,
    C: FnOnce() -> Result<Sequencia<B, P>, String>,
{
    match acao {
        Acao::Saude => (Resposta::ok("ok\n"), Fluxo::Continua),
        Acao::Encerrar => (Resposta::ok("ate logo\n"), Fluxo::Encerra),
        Acao::Gerar(p) => {
            if let Err(r) = valida_tamanho(tok, &p) {
                return (r, Fluxo::Continua);
            }
            let mut saida: Vec<u8> = vec![];
            match cli::executa(&p.texto, &p.data, p.novos, tok, motor, cria, &mut saida) {
                Ok(_) => (Resposta::ok(saida), Fluxo::Continua),
                Err(e) => (Resposta::msg(500, &format!("falha do motor: {e}")), Fluxo::Falha(e)),
            }
        }
    }
}

/// Atende UMA conexão: lê, valida, trata e responde (a escrita que falhar é ignorada: o cliente foi embora). Devolve o que o laço faz depois.
pub fn atende_conexao<C, B, P, F>(c: &mut C, tok: &Tokenizador, motor: &mut gr::MotorGeracao<B, P>, cria: F, prazo: Duration, hoje: &dyn Fn() -> String, porta: u16) -> Fluxo
where
    C: Conexao,
    B: Backend,
    P: PreparaPasso,
    P::Error: Debug,
    F: FnOnce() -> Result<Sequencia<B, P>, String>,
{
    let (resp, fluxo) = match le_requisicao(c, prazo).and_then(|r| confere_acesso(&r, porta).map(|_| r)).and_then(|r| interpreta(&r, &hoje())) {
        Err(r) => (r, Fluxo::Continua),
        Ok(acao) => trata(acao, tok, motor, cria),
    };
    let _ = c.write_all(&resp.serializa()).and_then(|_| c.flush());
    fluxo
}

/// Laço do servidor: aceita um cliente por vez, em ordem, até `Encerra` (código 0) ou falha do motor (código 1). O motor é encerrado ao sair.
/// `cria` (chamado só se o motor não está vivo; o binário entrega aqui a sequência já criada na partida) monta a sequência.
pub fn serve<B, P>(listener: &TcpListener, tok: &Tokenizador, motor: &mut gr::MotorGeracao<B, P>, mut cria: impl FnMut() -> Result<Sequencia<B, P>, String>, prazo: Duration, hoje: &dyn Fn() -> String) -> Fluxo
where
    B: Backend,
    P: PreparaPasso,
    P::Error: Debug,
{
    // A porta efetiva (a do bind) é a única aceita no cabeçalho Host.
    let porta = listener.local_addr().map(|a| a.port()).unwrap_or(0);
    loop {
        let mut s = match listener.accept() {
            Ok((s, _)) => s,
            Err(e) => {
                eprintln!("lastro-servidor: accept: {e}");
                std::thread::sleep(Duration::from_millis(50));
                continue;
            }
        };
        let _ = s.set_write_timeout(Some(prazo));
        let fluxo = atende_conexao(&mut s, tok, motor, &mut cria, prazo, hoje, porta);
        // Fecha limpo: avisa o fim da escrita e descarta (pouco, com tempo curto) o que o cliente ainda mandava, para o fechamento não virar RST
        // e comer a resposta; não lê além disso.
        let _ = s.shutdown(Shutdown::Write);
        let _ = s.set_read_timeout(Some(Duration::from_millis(100)));
        let mut lixo = [0u8; 4096];
        for _ in 0..16 {
            match s.read(&mut lixo) {
                Ok(n) if n > 0 => {}
                _ => break,
            }
        }
        if fluxo != Fluxo::Continua {
            return fluxo;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motor::fake::{FakeCfg, FakeCpu, FakeGpu, SchedCfg};
    use crate::motor::{AdmitMode, Config};
    use crate::primeira_palavra_regra::data_fixa_texto;
    use std::collections::VecDeque;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    const MODELO_1B: &str = "/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct";
    const VOCAB: usize = 49155;
    const HOJE: &str = "November 08, 2026";

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    fn tok() -> Tokenizador {
        Tokenizador::carrega(&Path::new(MODELO_1B).join("tokenizer.json")).expect("tokenizer.json")
    }

    // ---- conexão falsa ----

    struct Falsa {
        pedacos: VecDeque<Vec<u8>>,
        /// Atraso antes de cada leitura (simula cliente lento).
        atraso: Duration,
        /// Depois de acabar os pedaços: `true` = fica sem dados (TimedOut), `false` = EOF.
        trava: bool,
        lidos: usize,
        saida: Vec<u8>,
    }
    impl Falsa {
        fn nova(dados: &[u8]) -> Self {
            Self { pedacos: VecDeque::from([dados.to_vec()]), atraso: Duration::ZERO, trava: false, lidos: 0, saida: vec![] }
        }
    }
    impl Read for Falsa {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            std::thread::sleep(self.atraso);
            match self.pedacos.pop_front() {
                None if self.trava => Err(io::ErrorKind::TimedOut.into()),
                None => Ok(0),
                Some(mut p) => {
                    let n = p.len().min(b.len());
                    b[..n].copy_from_slice(&p[..n]);
                    if n < p.len() {
                        self.pedacos.push_front(p.split_off(n));
                    }
                    self.lidos += n;
                    Ok(n)
                }
            }
        }
    }
    impl Write for Falsa {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            self.saida.extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    impl Conexao for Falsa {
        fn limita_leitura(&mut self, _: Duration) -> io::Result<()> {
            Ok(())
        }
    }

    fn post(alvo: &str, corpo: &[u8]) -> Vec<u8> {
        let mut r = format!("POST {alvo} HTTP/1.1\r\nContent-Length: {}\r\n\r\n", corpo.len()).into_bytes();
        r.extend_from_slice(corpo);
        r
    }

    fn le(dados: &[u8]) -> Result<Requisicao, Resposta> {
        le_requisicao(&mut Falsa::nova(dados), Duration::from_secs(5))
    }

    fn status(r: Result<Requisicao, Resposta>) -> u16 {
        r.unwrap_err().status
    }

    // ---- argumentos ----

    #[test]
    fn args_validos_e_recusados() {
        let a = parse_args(v(&["--porta", "8080", "--modelo", "/m"])).unwrap();
        assert_eq!(a, ArgsServidor { modelo: "/m".into(), porta: 8080 });
        assert_eq!(parse_args(v(&["--modelo", "/m", "--porta", "65535"])).unwrap().porta, 65535);
        assert_eq!(parse_args(v(&["--modelo", "/m", "--porta", "1"])).unwrap().porta, 1);
        let casos: [(&[&str], &str); 14] = [
            (&[], "falta --modelo"),
            (&["--modelo", "/m"], "falta --porta"),
            (&["--porta", "80"], "falta --modelo"),
            (&["--modelo", "/m", "--porta", "0"], "fora de 1..65535"),
            (&["--modelo", "/m", "--porta", "65536"], "fora de 1..65535"),
            (&["--modelo", "/m", "--porta", "-1"], "fora de 1..65535"),
            (&["--modelo", "/m", "--porta", "abc"], "fora de 1..65535"),
            (&["--modelo", "/m", "--porta", "99999999999999999999"], "fora de 1..65535"),
            (&["--modelo", "/m", "--porta", "80", "--porta", "81"], "--porta duplicado"),
            (&["--modelo", "/m", "--modelo", "/n", "--porta", "80"], "--modelo duplicado"),
            (&["--modelo", "/m", "--porta", "80", "--endereco", "0.0.0.0"], "opção desconhecida: --endereco"),
            (&["--modelo", "/m", "--porta", "80", "oi"], "argumento inesperado"),
            (&["--modelo"], "--modelo sem valor"),
            (&["--modelo", "/m", "--porta"], "--porta sem valor"),
        ];
        for (a, msg) in casos {
            let e = parse_args(v(a)).unwrap_err();
            assert!(e.contains(msg), "{a:?}: {e}");
        }
    }

    // ---- parser HTTP ----

    #[test]
    fn requisicoes_validas() {
        let r = le(b"GET /saude HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
        assert_eq!((r.rota, r.corpo.len()), (Rota::Saude, 0));
        let r = le(&post("/gerar?novos=5&data=November%2007%2C%202026", "olá".as_bytes())).unwrap();
        assert_eq!((r.rota, r.query.as_str(), r.corpo.as_slice()), (Rota::Gerar, "novos=5&data=November%2007%2C%202026", "olá".as_bytes()));
        let r = le(&post("/encerrar", b"")).unwrap();
        assert_eq!(r.rota, Rota::Encerrar);
        // Nome de cabeçalho sem diferenciar maiúsculas, HTTP/1.0 aceito.
        assert!(le(b"POST /gerar HTTP/1.0\r\ncontent-length: 2\r\n\r\noi").is_ok());
        // Corpo em vários pedaços e bytes além do Content-Length ignorados.
        let r = le(&[post("/gerar", b"abc"), b"EXTRA".to_vec()].concat()).unwrap();
        assert_eq!(r.corpo, b"abc");
    }

    #[test]
    fn recusas_http_por_status() {
        // 400 malformadas
        for ruim in [
            &b"GARBAGE\r\n\r\n"[..],
            b"GET /saude\r\n\r\n",
            b"GET /saude HTTP/2\r\n\r\n",
            b"GET saude HTTP/1.1\r\n\r\n",
            b"get /saude HTTP/1.1\r\n\r\n",
            b"GET /saude HTTP/1.1\r\nSemDoisPontos\r\n\r\n",
            b"GET /saude HTTP/1.1\r\n : x\r\n\r\n",
            b"GET  /saude HTTP/1.1\r\n\r\n",
            b"\xff\xfe /saude HTTP/1.1\r\n\r\n",
            b"POST /gerar HTTP/1.1\r\nContent-Length: abc\r\n\r\n",
            b"POST /gerar HTTP/1.1\r\nContent-Length: -1\r\n\r\n",
            b"POST /gerar HTTP/1.1\r\nContent-Length: \r\n\r\n",
            b"POST /gerar HTTP/1.1\r\nContent-Length: 2\r\nContent-Length: 3\r\n\r\nabc",
            b"POST /gerar HTTP/1.1\r\nContent-Length: 2\r\nTransfer-Encoding: chunked\r\n\r\nab",
            b"POST /gerar HTTP/1.1\r\nContent-Length: 5\r\n\r\nab", // fecha antes do fim do corpo
            b"GET /saude HTTP/1.1\r\nHost: x", // fecha no meio do cabeçalho
        ] {
            assert_eq!(status(le(ruim)), 400, "{:?}", String::from_utf8_lossy(ruim));
        }
        // 404 / 405
        assert_eq!(status(le(b"GET /nada HTTP/1.1\r\n\r\n")), 404);
        assert_eq!(status(le(b"GET / HTTP/1.1\r\n\r\n")), 404);
        assert_eq!(status(le(&post("/saude", b""))), 405);
        assert_eq!(status(le(b"GET /gerar HTTP/1.1\r\n\r\n")), 405);
        assert_eq!(status(le(b"GET /encerrar HTTP/1.1\r\n\r\n")), 405);
        assert_eq!(status(le(b"DELETE /gerar HTTP/1.1\r\n\r\n")), 405);
        // 411 sem Content-Length (e com chunked sozinho)
        assert_eq!(status(le(b"POST /gerar HTTP/1.1\r\nHost: x\r\n\r\n")), 411);
        assert_eq!(status(le(b"POST /gerar HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n")), 411);
        assert_eq!(status(le(b"POST /encerrar HTTP/1.1\r\n\r\n")), 411);
        // 413 acima de 64 KiB; exatamente 64 KiB passa
        assert_eq!(status(le(b"POST /gerar HTTP/1.1\r\nContent-Length: 65537\r\n\r\n")), 413);
        assert_eq!(status(le(b"POST /gerar HTTP/1.1\r\nContent-Length: 99999999999999999999\r\n\r\n")), 400);
        assert!(le(&post("/gerar", &vec![b'a'; LIMITE_CORPO])).is_ok());
        // Cabeçalho e linha de requisição grandes demais
        let mut longa = b"GET /".to_vec();
        longa.extend(std::iter::repeat(b'a').take(LIMITE_LINHA + 10));
        longa.extend_from_slice(b" HTTP/1.1\r\n\r\n");
        assert_eq!(status(le(&longa)), 400);
        let mut grande = b"GET /saude HTTP/1.1\r\n".to_vec();
        for i in 0..2000 {
            grande.extend_from_slice(format!("X-{i}: valor\r\n").as_bytes());
        }
        grande.extend_from_slice(b"\r\n");
        assert_eq!(status(le(&grande)), 400);
        // Sem fim de cabeçalho e sem fim da linha: não lê além do limite.
        let mut f = Falsa::nova(&vec![b'a'; 100_000]);
        assert_eq!(status(le_requisicao(&mut f, Duration::from_secs(5))), 400);
        assert!(f.lidos <= LIMITE_CABECALHOS + 1024, "leu {}", f.lidos);
    }

    #[test]
    fn corpo_acima_do_limite_nao_e_lido() {
        let mut f = Falsa::nova(&[b"POST /gerar HTTP/1.1\r\nContent-Length: 1000000\r\n\r\n".to_vec(), vec![b'a'; 1_000_000]].concat());
        let e = le_requisicao(&mut f, Duration::from_secs(5)).unwrap_err();
        assert_eq!(e.status, 413);
        // Só o cabeçalho (mais, no máximo, um pedaço).
        assert!(f.lidos <= 1024, "leu {}", f.lidos);
        // Com o corpo certo, não lê nem um byte além do Content-Length.
        let mut f = Falsa::nova(&[post("/gerar", &vec![b'a'; 2000]), vec![b'z'; 50_000]].concat());
        assert_eq!(le_requisicao(&mut f, Duration::from_secs(5)).unwrap().corpo.len(), 2000);
        assert!(f.lidos < 2000 + 200 + 1024, "leu {}", f.lidos);
    }

    #[test]
    fn prazo_de_leitura_408() {
        // Gotejando 1 byte por vez: o prazo total corta, mesmo com bytes chegando.
        let dados = b"GET /saude HTTP/1.1\r\nHost: x\r\n\r\n";
        let mut f = Falsa { pedacos: dados.iter().map(|b| vec![*b]).collect(), atraso: Duration::from_millis(30), trava: false, lidos: 0, saida: vec![] };
        let t0 = Instant::now();
        let e = le_requisicao(&mut f, Duration::from_millis(150)).unwrap_err();
        assert_eq!(e.status, 408);
        assert!(t0.elapsed() < Duration::from_secs(2));
        assert!(f.lidos < dados.len());
        // Cabeçalho pronto, corpo que nunca chega (leitura expira).
        let mut f = Falsa::nova(b"POST /gerar HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc");
        f.trava = true;
        assert_eq!(le_requisicao(&mut f, Duration::from_millis(200)).unwrap_err().status, 408);
        // Nada recebido e a leitura expira.
        let mut f = Falsa::nova(b"");
        f.pedacos.clear();
        f.trava = true;
        assert_eq!(le_requisicao(&mut f, Duration::from_millis(200)).unwrap_err().status, 408);
        // O 408 é uma resposta de uma linha.
        let r = Resposta::msg(408, "x");
        assert!(String::from_utf8(r.serializa()).unwrap().starts_with("HTTP/1.1 408 Request Timeout\r\n"));
    }

    #[test]
    fn serializacao_da_resposta() {
        let s = String::from_utf8(Resposta::ok("ok\n").serializa()).unwrap();
        assert_eq!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: 3\r\nConnection: close\r\n\r\nok\n");
        // Content-Length conta bytes, não caracteres.
        let s = Resposta::ok("é\n".as_bytes()).serializa();
        assert!(String::from_utf8_lossy(&s).contains("Content-Length: 3\r\n"));
        // Mensagens de recusa: uma linha, mesmo que o texto traga quebras.
        let r = Resposta::msg(422, "a\nb\r\nc");
        assert_eq!(r.corpo, b"a b  c\n");
    }

    // ---- validações ----

    fn req_gerar(query: &str, corpo: &[u8]) -> Requisicao {
        Requisicao { rota: Rota::Gerar, query: query.into(), corpo: corpo.to_vec(), host: None, origem: false, http10: false }
    }

    #[test]
    fn urldecode_casos() {
        assert_eq!(urldecode("November%2007%2C%202026").unwrap(), "November 07, 2026");
        assert_eq!(urldecode("November+07%2c+2026").unwrap(), "November 07, 2026");
        assert_eq!(urldecode("%C3%A1").unwrap(), "á");
        for ruim in ["%", "%2", "%zz", "%2g", "%C3", "%FF"] {
            assert!(urldecode(ruim).is_err(), "{ruim}");
        }
    }

    #[test]
    fn interpreta_gerar_valido_e_padroes() {
        let a = interpreta(&req_gerar("", b"oi"), HOJE).unwrap();
        assert_eq!(a, Acao::Gerar(PedidoGera { texto: "oi".into(), novos: 64, data: HOJE.into() }));
        let a = interpreta(&req_gerar("novos=7&data=November%2007%2C%202026", " oi \n".as_bytes()), HOJE).unwrap();
        // Sem corte de bordas do texto.
        assert_eq!(a, Acao::Gerar(PedidoGera { texto: " oi \n".into(), novos: 7, data: "November 07, 2026".into() }));
        for n in ["1", "64"] {
            assert!(interpreta(&req_gerar(&format!("novos={n}"), b"oi"), HOJE).is_ok());
        }
        assert_eq!(interpreta(&Requisicao { rota: Rota::Saude, query: "lixo".into(), corpo: vec![], host: None, origem: false, http10: false }, HOJE).unwrap(), Acao::Saude);
        assert_eq!(interpreta(&Requisicao { rota: Rota::Encerrar, query: String::new(), corpo: vec![], host: None, origem: false, http10: false }, HOJE).unwrap(), Acao::Encerrar);
    }

    #[test]
    fn interpreta_recusas_400() {
        for corpo in [&b""[..], b"   ", b"\n\t \r\n"] {
            let e = interpreta(&req_gerar("", corpo), HOJE).unwrap_err();
            assert_eq!((e.status, e.corpo.as_slice()), (400, &b"entrada vazia ou s\xc3\xb3 espa\xc3\xa7os\n"[..]));
        }
        assert_eq!(interpreta(&req_gerar("", &[0xff, 0xfe]), HOJE).unwrap_err().status, 400);
        for q in [
            "novos=0", "novos=65", "novos=-1", "novos=abc", "novos=", "novos", "novos=1&novos=2", "data=2026-11-07", "data=November%207%2C%202026", "data=February%2029%2C%202027",
            "data=", "data=November+07%2C+2026&data=November+07%2C+2026", "x=1", "novos=%zz", "&", "novos=1&", "data=%FF",
        ] {
            let e = interpreta(&req_gerar(q, b"oi"), HOJE).unwrap_err();
            assert_eq!(e.status, 400, "{q}");
            assert_eq!(e.corpo.iter().filter(|b| **b == b'\n').count(), 1, "{q}: uma linha");
        }
    }

    #[test]
    fn recusa_422_prompt_mais_novos() {
        let t = tok();
        let d = data_fixa_texto();
        let p = |texto: String, novos| PedidoGera { texto, novos, data: d.clone() };
        let e = valida_tamanho(&t, &p("word ".repeat(3000), 1)).unwrap_err();
        assert_eq!(e.status, 422);
        assert!(String::from_utf8(e.corpo).unwrap().contains("passam de 2048 posições"));
        // Fronteira exata, como no 5d.
        let mut k = 1;
        while cli::conta_ids(&t, &"word ".repeat(k), &d).unwrap() < 2048 - 64 {
            k += 1;
        }
        let exato = "word ".repeat(k);
        assert!(valida_tamanho(&t, &p(exato.clone(), 64)).is_ok());
        assert_eq!(valida_tamanho(&t, &p(format!("{exato}another few words"), 64)).unwrap_err().status, 422);
    }

    // ---- motor falso (o mesmo do 5d) ----

    fn rota(p: usize, l: usize) -> Vec<u32> {
        vec![((p + l) % 4) as u32, ((p + l + 1 + (p / 4) % 3) % 4) as u32]
    }

    fn motor_falso() -> (Arc<FakeGpu>, crate::motor::Engine<FakeGpu>) {
        let routing: Vec<Vec<Vec<u32>>> = (0..2100).map(|p| (0..2).map(|l| rota(p, l)).collect()).collect();
        let gpu = FakeGpu::new_sched(
            FakeCfg { layers: 2, experts: 4, topk: 2, width: 3, routing, fail_main_at: None, copy_immediate: false, max_positions: 8 },
            SchedCfg { seed: 0xA0A, weights: [1, 1, 1, 1], budget: 300_000 },
        );
        let cfg = Config { layers: 2, experts: 4, topk: 2, width: 3, slots: 0, backlog: 0, admit: AdmitMode::TokenBarrier, first_token: 1, prefill: vec![], max_positions: 8, mutant: None };
        let parts = gpu.parts(&cfg);
        let eng = crate::motor::Engine::new(gpu.clone(), parts, cfg, FakeCpu { weighted: true, ..FakeCpu::default() }).expect("motor");
        (gpu, eng)
    }

    struct Prep {
        gpu: Arc<FakeGpu>,
        visto: Mutex<usize>,
        pos: Mutex<u64>,
    }
    impl Prep {
        fn sincroniza(&self) {
            let n = self.gpu.resets().len();
            let mut v = self.visto.lock().unwrap();
            if n > *v {
                *v = n;
                *self.pos.lock().unwrap() = 0;
            }
        }
    }
    impl PreparaPasso for Prep {
        type Error = &'static str;
        fn vocab(&self) -> usize {
            VOCAB
        }
        fn max_positions(&self) -> usize {
            8
        }
        fn max_len(&self) -> usize {
            2048
        }
        fn posicao(&self) -> u64 {
            self.sincroniza();
            *self.pos.lock().unwrap()
        }
        fn prepara(&self, ids: &[usize]) -> Result<(), &'static str> {
            self.sincroniza();
            *self.pos.lock().unwrap() += ids.len() as u64;
            Ok(())
        }
    }

    fn pico(passo: usize) -> usize {
        100 + (passo * 37) % 5000
    }

    /// `falha_em`: o passo de logits que devolve erro (motor com defeito).
    fn sequencia(falha_em: Option<usize>) -> Result<Sequencia<FakeGpu, Prep>, String> {
        let (gpu, eng) = motor_falso();
        // O passo recomeça em 0 a cada `nova_sequencia` (reset do motor falso): é o que torna o pedido repetido idêntico.
        let passo = Arc::new(Mutex::new((0usize, 0usize)));
        let gpu_le = gpu.clone();
        let le: Box<dyn FnMut() -> Result<Vec<f32>, String>> = Box::new(move || {
            let mut c = passo.lock().unwrap();
            let resets = gpu_le.resets().len();
            if resets > c.1 {
                *c = (0, resets);
            }
            let i = c.0;
            c.0 += 1;
            if falha_em == Some(i) {
                return Err("defeito injetado".into());
            }
            let mut v = vec![0.0f32; VOCAB];
            v[pico(i)] = 5.0;
            Ok(v)
        });
        Ok(Sequencia { eng, prep: Prep { gpu, visto: Mutex::new(0), pos: Mutex::new(0) }, le_logits: le })
    }

    type Motor = gr::MotorGeracao<FakeGpu, Prep>;

    fn hoje() -> String {
        HOJE.to_string()
    }

    const PORTA_T: u16 = 4321;

    /// Insere `Host: 127.0.0.1:<porta>` logo depois da linha de requisição (se o pedido não trouxer Host).
    fn com_host(req: &[u8], porta: u16) -> Vec<u8> {
        if acha(req, b"\r\nHost:").is_some() {
            return req.to_vec();
        }
        match acha(req, b"\r\n") {
            Some(i) => [&req[..i + 2], format!("Host: 127.0.0.1:{porta}\r\n").as_bytes(), &req[i + 2..]].concat(),
            None => req.to_vec(),
        }
    }

    fn corpo_resposta(saida: &[u8]) -> (String, Vec<u8>) {
        let i = acha(saida, b"\r\n\r\n").expect("fim do cabeçalho") + 4;
        (String::from_utf8_lossy(&saida[..i]).to_string(), saida[i..].to_vec())
    }

    fn pede(m: &mut Motor, req: &[u8], falha_em: Option<usize>) -> (Fluxo, String, Vec<u8>) {
        let t = tok();
        let mut c = Falsa::nova(&com_host(req, PORTA_T));
        let f = atende_conexao(&mut c, &t, m, || sequencia(falha_em), Duration::from_secs(5), &hoje, PORTA_T);
        let (cab, corpo) = corpo_resposta(&c.saida);
        (f, cab, corpo)
    }

    #[test]
    fn corpo_do_gerar_igual_ao_do_5d_e_repeticao_identica() {
        let t = tok();
        let data = "November 07, 2026";
        // A saída do `lastro` (5d) com o mesmo motor falso.
        let mut m5d: Motor = gr::MotorGeracao::novo();
        let mut esperado = vec![];
        cli::executa("Hi there", data, 6, &t, &mut m5d, || sequencia(None), &mut esperado).unwrap();
        m5d.encerra().unwrap();
        assert!(esperado.ends_with(b"\n") && !esperado.ends_with(b"\n\n"));
        let req = post("/gerar?novos=6&data=November%2007%2C%202026", b"Hi there");
        let mut m: Motor = gr::MotorGeracao::novo();
        let (f, cab, corpo) = pede(&mut m, &req, None);
        assert_eq!(f, Fluxo::Continua);
        assert!(cab.starts_with("HTTP/1.1 200 OK\r\n") && cab.contains("text/plain; charset=utf-8") && cab.contains("Connection: close"), "{cab}");
        assert!(cab.contains(&format!("Content-Length: {}\r\n", esperado.len())));
        assert_eq!(corpo, esperado);
        // O mesmo pedido de novo, no mesmo motor: idêntico.
        let (_, _, corpo2) = pede(&mut m, &req, None);
        assert_eq!(corpo2, esperado);
        // Pedido diferente no meio não contamina o seguinte.
        let (_, _, outro) = pede(&mut m, &post("/gerar?novos=3&data=November%2007%2C%202026", b"Something else entirely"), None);
        assert!(outro.ends_with(b"\n"));
        let (_, _, corpo3) = pede(&mut m, &req, None);
        assert_eq!(corpo3, esperado);
        assert_eq!(m.criacoes, 1, "o motor é criado uma vez e reutilizado");
        m.encerra().unwrap();
    }

    #[test]
    fn recusas_nao_tocam_o_motor() {
        let mut m: Motor = gr::MotorGeracao::novo();
        let t = tok();
        let grande = "word ".repeat(3000);
        let casos: Vec<(Vec<u8>, u16)> = vec![
            (post("/gerar", b""), 400),
            (post("/gerar", b"   "), 400),
            (post("/gerar", &[0xff]), 400),
            (post("/gerar?novos=0", b"oi"), 400),
            (post("/gerar?data=ontem", b"oi"), 400),
            (post("/gerar", grande.as_bytes()), 422),
            (post("/gerar?novos=1", grande.as_bytes()), 422),
            (b"POST /gerar HTTP/1.1\r\n\r\n".to_vec(), 411),
            (b"POST /gerar HTTP/1.1\r\nContent-Length: 70000\r\n\r\n".to_vec(), 413),
            (b"GET /gerar HTTP/1.1\r\n\r\n".to_vec(), 405),
            (b"GET /x HTTP/1.1\r\n\r\n".to_vec(), 404),
            (b"lixo\r\n\r\n".to_vec(), 400),
        ];
        for (req, st) in casos {
            let mut c = Falsa::nova(&com_host(&req, PORTA_T));
            let f = atende_conexao(&mut c, &t, &mut m, || -> Result<Sequencia<FakeGpu, Prep>, String> { panic!("o motor não pode ser criado numa recusa") }, Duration::from_secs(5), &hoje, PORTA_T);
            assert_eq!(f, Fluxo::Continua);
            let (cab, corpo) = corpo_resposta(&c.saida);
            assert!(cab.starts_with(&format!("HTTP/1.1 {st} ")), "{st}: {cab}");
            assert_eq!(corpo.iter().filter(|b| **b == b'\n').count(), 1, "{st}: uma linha");
        }
        assert_eq!(m.criacoes, 0);
        // Saúde e encerrar não tocam o motor.
        let (f, cab, corpo) = pede(&mut m, b"GET /saude HTTP/1.1\r\n\r\n", None);
        assert_eq!((f, corpo.as_slice()), (Fluxo::Continua, &b"ok\n"[..]));
        assert!(cab.starts_with("HTTP/1.1 200 OK"));
        let (f, _, corpo) = pede(&mut m, &post("/encerrar", b""), None);
        assert_eq!((f, corpo.as_slice()), (Fluxo::Encerra, &b"ate logo\n"[..]));
        assert_eq!(m.criacoes, 0);
    }

    #[test]
    fn host_e_origin_viram_403_sem_tocar_o_motor() {
        let mut m: Motor = gr::MotorGeracao::novo();
        let t = tok();
        let p = PORTA_T;
        let casos: Vec<(String, u16)> = vec![
            // O pedido da sonda: Host de outro nome e Origin.
            ("POST /encerrar HTTP/1.1\r\nHost: evil.example:80\r\nOrigin: https://evil.example\r\nContent-Type: text/plain\r\nContent-Length: 0\r\n\r\n".into(), 403),
            (format!("POST /encerrar HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\nOrigin: null\r\nContent-Length: 0\r\n\r\n"), 403),
            (format!("GET /saude HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\norigin: http://127.0.0.1:{p}\r\n\r\n"), 403),
            ("GET /saude HTTP/1.1\r\nHost: evil.example:4321\r\n\r\n".into(), 403),
            ("GET /saude HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".into(), 403),
            (format!("GET /saude HTTP/1.1\r\nHost: 127.0.0.1:{}\r\n\r\n", p + 1), 403),
            (format!("GET /saude HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\nHost: evil.example:{p}\r\n\r\n"), 403),
            ("GET /saude HTTP/1.1\r\n\r\n".into(), 403),
            // Boas: 127.0.0.1 e localhost com a porta; HTTP/1.0 sem Host é aceito.
            (format!("GET /saude HTTP/1.1\r\nHost: 127.0.0.1:{p}\r\n\r\n"), 200),
            (format!("GET /saude HTTP/1.1\r\nHost: LocalHost:{p}\r\n\r\n"), 200),
            ("GET /saude HTTP/1.0\r\n\r\n".into(), 200),
        ];
        for (req, st) in casos {
            let mut c = Falsa::nova(req.as_bytes());
            let f = atende_conexao(&mut c, &t, &mut m, || -> Result<Sequencia<FakeGpu, Prep>, String> { panic!("o motor não pode ser criado") }, Duration::from_secs(5), &hoje, p);
            assert_eq!(f, Fluxo::Continua, "{req:?}");
            let (cab, corpo) = corpo_resposta(&c.saida);
            assert!(cab.starts_with(&format!("HTTP/1.1 {st} ")), "{st}: {req:?} -> {cab}");
            assert_eq!(corpo.iter().filter(|b| **b == b'\n').count(), 1, "{st}: uma linha");
        }
        assert_eq!(m.criacoes, 0);
    }

    #[test]
    fn rede_loopback_sonda_com_host_e_origin_recebe_403_e_o_servidor_segue() {
        let (addr, h) = sobe(Duration::from_secs(5), None);
        let sonda = b"POST /encerrar HTTP/1.1\r\nHost: evil.example:80\r\nOrigin: https://evil.example\r\nContent-Type: text/plain\r\nContent-Length: 0\r\n\r\n";
        let (cab, corpo) = http(addr, sonda);
        assert!(cab.starts_with("HTTP/1.1 403 "), "{cab}");
        assert_eq!(corpo.iter().filter(|b| **b == b'\n').count(), 1);
        let (cab, corpo) = http(addr, format!("GET /saude HTTP/1.1\r\nHost: localhost:{}\r\n\r\n", addr.port()).as_bytes());
        assert!(cab.starts_with("HTTP/1.1 200 OK") && corpo == b"ok\n");
        http(addr, &post("/encerrar", b""));
        assert_eq!(h.join().unwrap(), Fluxo::Encerra);
    }

    #[test]
    fn leitura_lenta_vira_408_no_atende_conexao() {
        let mut m: Motor = gr::MotorGeracao::novo();
        let t = tok();
        let mut c = Falsa::nova(&com_host(b"POST /gerar HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc", PORTA_T));
        c.trava = true;
        let f = atende_conexao(&mut c, &t, &mut m, || -> Result<Sequencia<FakeGpu, Prep>, String> { panic!("sem motor") }, Duration::from_millis(100), &hoje, PORTA_T);
        assert_eq!(f, Fluxo::Continua);
        assert!(corpo_resposta(&c.saida).0.starts_with("HTTP/1.1 408 "));
    }

    #[test]
    fn falha_do_motor_vira_500_e_encerra() {
        let mut m: Motor = gr::MotorGeracao::novo();
        let (f, cab, corpo) = pede(&mut m, &post("/gerar?novos=4", b"Hi"), Some(2));
        assert!(matches!(f, Fluxo::Falha(ref e) if e.contains("defeito injetado")), "{f:?}");
        assert!(cab.starts_with("HTTP/1.1 500 "), "{cab}");
        assert!(String::from_utf8(corpo.clone()).unwrap().contains("defeito injetado"));
        assert_eq!(corpo.iter().filter(|b| **b == b'\n').count(), 1);
        assert!(!m.viva(), "motor fechado não é reaproveitado");
        m.encerra().unwrap();
        // Falha ao criar o motor também.
        let t = tok();
        let mut m: Motor = gr::MotorGeracao::novo();
        let mut c = Falsa::nova(&com_host(&post("/gerar?novos=4", b"Hi"), PORTA_T));
        let f = atende_conexao(&mut c, &t, &mut m, || Err("sem GPU".into()), Duration::from_secs(5), &hoje, PORTA_T);
        assert_eq!(f, Fluxo::Falha("sem GPU".into()));
        assert!(corpo_resposta(&c.saida).0.starts_with("HTTP/1.1 500 "));
    }

    // ---- rede real em loopback, com o motor falso ----

    use std::net::SocketAddr;

    /// Sobe o laço `serve` numa porta efêmera de 127.0.0.1 (SÓ no teste) e devolve o endereço e a thread (resultado do laço).
    fn sobe(prazo: Duration, falha_em: Option<usize>) -> (SocketAddr, std::thread::JoinHandle<Fluxo>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        assert!(addr.ip().is_loopback());
        let h = std::thread::spawn(move || {
            let t = tok();
            let mut m: Motor = gr::MotorGeracao::novo();
            let f = serve(&l, &t, &mut m, || sequencia(falha_em), prazo, &hoje);
            m.encerra().ok();
            f
        });
        (addr, h)
    }

    fn http(addr: SocketAddr, req: &[u8]) -> (String, Vec<u8>) {
        let mut s = TcpStream::connect(addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
        s.write_all(&com_host(req, addr.port())).unwrap();
        let mut r = vec![];
        s.read_to_end(&mut r).unwrap();
        corpo_resposta(&r)
    }

    #[test]
    fn rede_loopback_fluxo_completo() {
        let (addr, h) = sobe(Duration::from_secs(5), None);
        let (cab, corpo) = http(addr, b"GET /saude HTTP/1.1\r\n\r\n");
        assert!(cab.starts_with("HTTP/1.1 200 OK"));
        assert_eq!(corpo, b"ok\n");
        let req = post("/gerar?novos=5&data=November+07%2C+2026", b"Hello");
        let (cab1, c1) = http(addr, &req);
        assert!(cab1.starts_with("HTTP/1.1 200 OK") && c1.ends_with(b"\n"));
        // Recusa com o motor ligado, e o pedido seguinte continua igual.
        let (cab, corpo) = http(addr, &post("/gerar", b""));
        assert!(cab.starts_with("HTTP/1.1 400 "), "{cab}");
        assert_eq!(corpo, "entrada vazia ou só espaços\n".as_bytes());
        let (_, c2) = http(addr, &req);
        assert_eq!(c1, c2);
        assert!(http(addr, b"GET /nada HTTP/1.1\r\n\r\n").0.starts_with("HTTP/1.1 404 "));
        // Corpo de 1 MB declarado: 413 chega ao cliente, mesmo sem o corpo (o servidor não o lê).
        let (cab, _) = http(addr, b"POST /gerar HTTP/1.1\r\nContent-Length: 1000000\r\n\r\n");
        assert!(cab.starts_with("HTTP/1.1 413 "));
        let (cab, corpo) = http(addr, &post("/encerrar", b""));
        assert!(cab.starts_with("HTTP/1.1 200 OK"));
        assert_eq!(corpo, b"ate logo\n");
        assert_eq!(h.join().unwrap(), Fluxo::Encerra);
        assert!(TcpStream::connect_timeout(&addr, Duration::from_millis(500)).is_err(), "o listener fechou");
    }

    #[test]
    fn rede_loopback_leitura_lenta_nao_derruba_o_servidor() {
        let (addr, h) = sobe(Duration::from_millis(400), None);
        // Cliente que conecta, manda meio cabeçalho e para.
        let mut lento = TcpStream::connect(addr).unwrap();
        lento.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        lento.write_all(b"POST /gerar HTTP/1.1\r\nHost: x\r\nContent-Le").unwrap();
        let t0 = Instant::now();
        let mut r = vec![];
        lento.read_to_end(&mut r).unwrap();
        let dt = t0.elapsed();
        let (cab, _) = corpo_resposta(&r);
        assert!(cab.starts_with("HTTP/1.1 408 "), "{cab}");
        assert!(dt >= Duration::from_millis(350) && dt < Duration::from_secs(5), "{dt:?}");
        // Cliente que conecta e não manda nada.
        let mut mudo = TcpStream::connect(addr).unwrap();
        mudo.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let mut r = vec![];
        mudo.read_to_end(&mut r).unwrap();
        assert!(corpo_resposta(&r).0.starts_with("HTTP/1.1 408 "));
        // O servidor segue de pé para o próximo.
        let (cab, corpo) = http(addr, b"GET /saude HTTP/1.1\r\n\r\n");
        assert!(cab.starts_with("HTTP/1.1 200 OK") && corpo == b"ok\n");
        http(addr, &post("/encerrar", b""));
        assert_eq!(h.join().unwrap(), Fluxo::Encerra);
    }

    #[test]
    fn rede_loopback_falha_do_motor_500_e_sai() {
        let (addr, h) = sobe(Duration::from_secs(5), Some(1));
        let (cab, corpo) = http(addr, &post("/gerar?novos=4", b"Hi"));
        assert!(cab.starts_with("HTTP/1.1 500 "), "{cab}");
        assert!(String::from_utf8(corpo).unwrap().contains("defeito injetado"));
        assert!(matches!(h.join().unwrap(), Fluxo::Falha(_)));
        assert!(TcpStream::connect_timeout(&addr, Duration::from_millis(500)).is_err());
    }
}
