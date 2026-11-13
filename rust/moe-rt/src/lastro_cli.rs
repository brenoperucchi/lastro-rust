//! Etapa 5d: a lógica pura do comando de linha `lastro` (argumentos, recusas, portões e a geração). O laço é o da 5c (`geracao_regra::gera`);
//! este módulo só decide o que recusar ANTES da GPU e como o texto sai. Recusa = código 2; falha de execução = código 1 (no binário).

use crate::geracao_regra as gr;
use crate::leitura_regra::MAX_LEN;
use crate::motor::backend::Backend;
use crate::motor::leitura::PreparaPasso;
use crate::primeira_palavra_regra::Sequencia;
use crate::tokenizador::{formata_data, renderiza_chat, Mensagem, Papel, Tokenizador};
use std::fmt::Debug;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// O mesmo portão e o mesmo UUID dos outros binários que usam a GPU.
pub const VARIAVEL: &str = "PRIMEIRA_PALAVRA_AUTORIZADA";
pub const UUID_3080TI: &str = "GPU-b22d9465-8552-9f97-812e-380b590645f8";
/// Máximo de tokens novos aceito nesta etapa.
pub const MAX_NOVOS: usize = 64;
const MESES: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

pub const USO: &str = "uso: lastro --modelo <dir> [--novos N (1..64, padrão 64)] [--data \"Month DD, YYYY\"] [TEXTO] (sem TEXTO, lê a entrada padrão)";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Args {
    pub modelo: PathBuf,
    pub novos: usize,
    pub data: Option<String>,
    pub texto: Option<String>,
}

/// Argumentos: `--modelo` obrigatório; `--novos`, `--data` e o texto (um só) opcionais; nada duplicado nem desconhecido. `--` encerra as opções.
pub fn parse_args(v: Vec<String>) -> Result<Args, String> {
    let mut modelo: Option<PathBuf> = None;
    let mut novos: Option<usize> = None;
    let mut data: Option<String> = None;
    let mut texto: Option<String> = None;
    let mut it = v.into_iter();
    let mut opcoes = true;
    while let Some(a) = it.next() {
        if opcoes && a == "--" {
            opcoes = false;
            continue;
        }
        if opcoes && a.starts_with("--") {
            let mut val = |nome: &str| -> Result<String, String> {
                match it.next() {
                    Some(x) if !x.starts_with("--") => Ok(x),
                    _ => Err(format!("{nome} sem valor")),
                }
            };
            match a.as_str() {
                "--modelo" => {
                    let x = val("--modelo")?;
                    if modelo.replace(PathBuf::from(x)).is_some() {
                        return Err("--modelo duplicado".into());
                    }
                }
                "--novos" => {
                    let x = val("--novos")?;
                    let n = if !x.is_empty() && x.bytes().all(|b| b.is_ascii_digit()) { x.parse::<usize>().unwrap_or(usize::MAX) } else { 0 };
                    if !(1..=MAX_NOVOS).contains(&n) {
                        return Err(format!("--novos fora de 1..{MAX_NOVOS}: {x}"));
                    }
                    if novos.replace(n).is_some() {
                        return Err("--novos duplicado".into());
                    }
                }
                "--data" => {
                    let x = val("--data")?;
                    if data.replace(x).is_some() {
                        return Err("--data duplicado".into());
                    }
                }
                _ => return Err(format!("opção desconhecida: {a}")),
            }
        } else if texto.replace(a).is_some() {
            return Err("texto duplicado (mais de um argumento de texto)".into());
        }
    }
    Ok(Args { modelo: modelo.ok_or("falta --modelo")?, novos: novos.unwrap_or(MAX_NOVOS), data, texto })
}

/// `--data` no formato do template (`November 07, 2026`); a data tem de existir e voltar idêntica pelo `formata_data`.
pub fn parse_data(s: &str) -> Result<String, String> {
    let erro = || format!("--data tem de ser \"Month DD, YYYY\" (ex.: \"November 07, 2026\"), não {s:?}");
    let (md, ano) = s.split_once(", ").ok_or_else(erro)?;
    let (mes, dia) = md.split_once(' ').ok_or_else(erro)?;
    let m = MESES.iter().position(|x| *x == mes).ok_or_else(erro)? as u32 + 1;
    let (a, d) = (ano.parse::<i32>().map_err(|_| erro())?, dia.parse::<u32>().map_err(|_| erro())?);
    match formata_data(a, m, d) {
        Ok(f) if f == s => Ok(f),
        _ => Err(erro()),
    }
}

/// O texto: o argumento, ou a entrada padrão inteira (UTF-8) se não houve argumento. Vazio ou só espaços é recusado. Sem corte de bordas.
pub fn resolve_texto(arg: Option<String>, mut entrada: impl Read) -> Result<String, String> {
    let t = match arg {
        Some(t) => t,
        None => {
            let mut b = vec![];
            entrada.read_to_end(&mut b).map_err(|e| format!("entrada padrão: {e}"))?;
            String::from_utf8(b).map_err(|_| "entrada padrão não é UTF-8".to_string())?
        }
    };
    if t.trim().is_empty() {
        return Err("entrada vazia ou só espaços".into());
    }
    Ok(t)
}

/// O modelo existe e tem config, tokenizer.json e ao menos um .safetensors.
pub fn valida_modelo(dir: &Path) -> Result<(), String> {
    if !dir.is_dir() {
        return Err(format!("modelo inexistente: {} não é uma pasta", dir.display()));
    }
    for f in ["config.json", "tokenizer.json"] {
        if !dir.join(f).is_file() {
            return Err(format!("modelo sem {f}: {}", dir.display()));
        }
    }
    let pesos = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?.filter_map(|e| e.ok()).any(|e| e.path().extension().is_some_and(|x| x == "safetensors") && std::fs::metadata(e.path()).is_ok_and(|md| md.is_file()));
    if !pesos {
        return Err(format!("modelo sem pesos (*.safetensors): {}", dir.display()));
    }
    Ok(())
}

/// Ids do pedido já com o template; a conta que decide se cabe (antes da GPU).
pub fn conta_ids(tok: &Tokenizador, texto: &str, data: &str) -> Result<usize, String> {
    let r = renderiza_chat(&[Mensagem { papel: Papel::Usuario, conteudo: texto.to_string() }], true, data).map_err(|e| e.to_string())?;
    Ok(tok.codifica(&r).len())
}

/// Recusa o pedido cujo prompt com template mais os novos passa de `MAX_LEN` posições.
pub fn confere_tamanho(tok: &Tokenizador, texto: &str, data: &str, novos: usize) -> Result<usize, String> {
    let ids = conta_ids(tok, texto, data)?;
    if ids + novos > MAX_LEN {
        return Err(format!("texto grande demais: {ids} ids do prompt mais {novos} novos passam de {MAX_LEN} posições"));
    }
    Ok(ids)
}

pub fn exige_autorizacao(valor: Option<&str>) -> Result<(), String> {
    match valor {
        Some("1") => Ok(()),
        _ => Err(format!("o uso da GPU exige o 'autorizo' explícito do Breno ({VARIAVEL}=1)")),
    }
}

/// Variável com o caminho de um arquivo LOCAL (fora do git) com UUIDs extras aceitos, um por linha (`GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`;
/// linhas vazias e comentários com `#` são ignorados). Sem a variável, só a 3080 Ti passa. Os UUIDs do arquivo nunca entram em commit nem em mensagem.
pub const VAR_UUIDS_EXTRA: &str = "LASTRO_UUIDS_EXTRA_ARQ";

/// Formato `GPU-` + 8-4-4-4-12 hexadecimais minúsculos.
pub fn uuid_valido(s: &str) -> bool {
    let Some(r) = s.strip_prefix("GPU-") else { return false };
    let partes: Vec<&str> = r.split('-').collect();
    partes.len() == 5
        && partes.iter().zip([8usize, 4, 4, 4, 12]).all(|(p, n)| p.len() == n && p.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
}

/// Lê a lista local: erro (nunca silêncio) em qualquer linha inválida; a mensagem diz só o número da linha, nunca o conteúdo.
pub fn parse_uuids_extra(texto: &str) -> Result<Vec<String>, String> {
    let mut v = Vec::new();
    for (i, linha) in texto.lines().enumerate() {
        let l = linha.trim();
        if l.is_empty() || l.starts_with('#') {
            continue;
        }
        if !uuid_valido(l) {
            return Err(format!("lista local de UUIDs ({VAR_UUIDS_EXTRA}): linha {} não é um UUID no formato GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", i + 1));
        }
        v.push(l.to_string());
    }
    Ok(v)
}

/// Carrega a lista local a partir do caminho (None ou vazio = sem lista). Falha de leitura é erro (a trava fica fechada).
pub fn le_uuids_extra(caminho: Option<&str>) -> Result<Vec<String>, String> {
    match caminho {
        None | Some("") => Ok(Vec::new()),
        Some(c) => {
            let t = std::fs::read_to_string(c).map_err(|e| format!("lista local de UUIDs ({VAR_UUIDS_EXTRA}) ilegível: {}", e.kind()))?;
            parse_uuids_extra(&t)
        }
    }
}

/// Função ÚNICA da trava: `CUDA_VISIBLE_DEVICES` tem de ser UM único UUID, igual ao da 3080 Ti ou a um da lista `extras`. Lista "a,b", vazio,
/// `0` ou qualquer UUID fora da lista são recusados. A mensagem nunca imprime UUID da lista local.
pub fn exige_uuid_com(valor: Option<&str>, extras: &[String]) -> Result<(), String> {
    match valor {
        Some(v) if v == UUID_3080TI || (uuid_valido(v) && extras.iter().any(|e| e == v)) => Ok(()),
        outro => Err(format!("CUDA_VISIBLE_DEVICES tem de ser um único UUID da lista combinada (a 3080 Ti {UUID_3080TI} ou uma GPU da lista local {VAR_UUIDS_EXTRA}); veio {outro:?}")),
    }
}

/// A trava com a lista local lida do ambiente. A 3080 Ti passa sem tocar no arquivo; qualquer outro valor lê a lista (arquivo ilegível ou inválido = recusa).
pub fn exige_uuid(valor: Option<&str>) -> Result<(), String> {
    if valor == Some(UUID_3080TI) {
        return Ok(());
    }
    let extras = le_uuids_extra(std::env::var(VAR_UUIDS_EXTRA).ok().as_deref())?;
    exige_uuid_com(valor, &extras)
}

/// Portão e UUID, só chamados depois de todas as recusas e imediatamente antes de abrir o device.
pub fn portoes() -> Result<(), String> {
    exige_autorizacao(std::env::var(VARIAVEL).ok().as_deref())?;
    exige_uuid(std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref())
}

/// Um pedido: o laço guloso da 5c (para no primeiro EOS ou em `novos`). `saida` recebe `decodifica(ids sem o EOS)` e um newline, só se tudo deu certo.
pub fn executa<B, P, C>(texto: &str, data: &str, novos: usize, tok: &Tokenizador, motor: &mut gr::MotorGeracao<B, P>, cria: C, saida: &mut dyn Write) -> Result<gr::Geracao, String>
where
    B: Backend,
    P: PreparaPasso,
    P::Error: Debug,
    C: FnOnce() -> Result<Sequencia<B, P>, String>,
{
    gr::gera(Instant::now(), texto, data, tok, motor, cria, gr::CfgGera { novos, para_no_eos: true }, saida)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motor::fake::{FakeCfg, FakeCpu, FakeGpu, SchedCfg};
    use crate::motor::{AdmitMode, Config};
    use crate::primeira_palavra_regra::data_fixa_texto;
    use std::sync::{Arc, Mutex};

    const MODELO_1B: &str = "/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct";
    const VOCAB: usize = 49155;

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    fn tok() -> Tokenizador {
        Tokenizador::carrega(&Path::new(MODELO_1B).join("tokenizer.json")).expect("tokenizer.json")
    }

    fn pasta(nome: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("lastro-cli-{nome}-{}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn args_validos_e_padroes() {
        let a = parse_args(v(&["--modelo", "/m", "oi"])).unwrap();
        assert_eq!(a, Args { modelo: "/m".into(), novos: 64, data: None, texto: Some("oi".into()) });
        let a = parse_args(v(&["oi", "--novos", "7", "--data", "November 07, 2026", "--modelo", "/m"])).unwrap();
        assert_eq!((a.novos, a.data.as_deref(), a.texto.as_deref()), (7, Some("November 07, 2026"), Some("oi")));
        assert_eq!(parse_args(v(&["--modelo", "/m"])).unwrap().texto, None);
        // `--` deixa o texto começar com `--`.
        assert_eq!(parse_args(v(&["--modelo", "/m", "--", "--oi"])).unwrap().texto.as_deref(), Some("--oi"));
        for ok in ["1", "64"] {
            assert!(parse_args(v(&["--modelo", "/m", "--novos", ok])).is_ok());
        }
    }

    #[test]
    fn args_recusados() {
        let casos: [(&[&str], &str); 12] = [
            (&[], "falta --modelo"),
            (&["oi"], "falta --modelo"),
            (&["--modelo", "/m", "--novos", "0"], "fora de 1..64"),
            (&["--modelo", "/m", "--novos", "65"], "fora de 1..64"),
            (&["--modelo", "/m", "--novos", "-1"], "fora de 1..64"),
            (&["--modelo", "/m", "--novos", "abc"], "fora de 1..64"),
            (&["--modelo", "/m", "--x"], "opção desconhecida: --x"),
            (&["--modelo", "/m", "--modelo", "/n"], "--modelo duplicado"),
            (&["--modelo", "/m", "--novos", "3", "--novos", "4"], "--novos duplicado"),
            (&["--modelo", "/m", "--data", "November 07, 2026", "--data", "November 07, 2026"], "--data duplicado"),
            (&["--modelo", "/m", "um", "dois"], "texto duplicado"),
            (&["--modelo"], "--modelo sem valor"),
        ];
        for (a, msg) in casos {
            let e = parse_args(v(a)).unwrap_err();
            assert!(e.contains(msg), "{a:?}: {e}");
        }
        assert!(parse_args(v(&["--modelo", "/m", "--novos"])).unwrap_err().contains("sem valor"));
    }

    #[test]
    fn data_do_template() {
        assert_eq!(parse_data("November 07, 2026").unwrap(), "November 07, 2026");
        assert_eq!(parse_data("February 29, 2028").unwrap(), "February 29, 2028");
        for ruim in ["", "2026-11-07", "November 7, 2026", "Nov 07, 2026", "February 29, 2027", "November 31, 2026", "November 07 2026", "november 07, 2026"] {
            assert!(parse_data(ruim).unwrap_err().contains("Month DD, YYYY"), "{ruim}");
        }
    }

    #[test]
    fn entrada_padrao_ou_argumento() {
        // Argumento vence e a entrada nem é lida.
        struct Proibida;
        impl Read for Proibida {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                panic!("a entrada padrão não deve ser lida com argumento");
            }
        }
        assert_eq!(resolve_texto(Some("oi".into()), Proibida).unwrap(), "oi");
        // Sem argumento, a entrada inteira, sem cortar bordas.
        assert_eq!(resolve_texto(None, &b"linha 1\nlinha 2\n"[..]).unwrap(), "linha 1\nlinha 2\n");
        assert_eq!(resolve_texto(None, "áé ñ".as_bytes()).unwrap(), "áé ñ");
    }

    #[test]
    fn recusa_de_entrada_vazia() {
        for t in ["", "   ", "\n\t \n"] {
            assert_eq!(resolve_texto(Some(t.into()), &b""[..]).unwrap_err(), "entrada vazia ou só espaços");
            assert_eq!(resolve_texto(None, t.as_bytes()).unwrap_err(), "entrada vazia ou só espaços");
        }
        assert!(resolve_texto(None, &b""[..]).is_err());
        assert!(resolve_texto(None, &[0xff, 0xfe][..]).unwrap_err().contains("UTF-8"));
    }

    #[test]
    fn recusa_de_modelo() {
        let base = pasta("modelo");
        let inexistente = base.join("nao-existe");
        assert!(valida_modelo(&inexistente).unwrap_err().contains("modelo inexistente"));
        let arquivo = base.join("arquivo");
        std::fs::write(&arquivo, b"x").unwrap();
        assert!(valida_modelo(&arquivo).unwrap_err().contains("modelo inexistente"));
        let m = base.join("m");
        std::fs::create_dir_all(&m).unwrap();
        assert!(valida_modelo(&m).unwrap_err().contains("sem config.json"));
        std::fs::write(m.join("config.json"), b"{}").unwrap();
        assert!(valida_modelo(&m).unwrap_err().contains("sem tokenizer.json"));
        std::fs::write(m.join("tokenizer.json"), b"{}").unwrap();
        assert!(valida_modelo(&m).unwrap_err().contains("sem pesos"));
        std::fs::write(m.join("model.safetensors"), b"x").unwrap();
        assert!(valida_modelo(&m).is_ok());
        // Pasta ou FIFO com a extensão não são pesos (recusa sem abrir o objeto).
        std::fs::remove_file(m.join("model.safetensors")).unwrap();
        std::fs::create_dir(m.join("model.safetensors")).unwrap();
        assert!(valida_modelo(&m).unwrap_err().contains("sem pesos"));
        std::fs::remove_dir(m.join("model.safetensors")).unwrap();
        assert!(std::process::Command::new("mkfifo").arg(m.join("model.safetensors")).status().unwrap().success());
        assert!(valida_modelo(&m).unwrap_err().contains("sem pesos"));
        std::fs::remove_file(m.join("model.safetensors")).unwrap();
        std::fs::write(m.join("model.safetensors"), b"x").unwrap();
        assert!(valida_modelo(&m).is_ok());
        assert!(valida_modelo(Path::new(MODELO_1B)).is_ok());
        std::fs::remove_dir_all(&base).unwrap();
    }

    #[test]
    fn recusa_de_texto_que_nao_cabe() {
        let t = tok();
        let data = data_fixa_texto();
        let n = conta_ids(&t, "Hi", &data).unwrap();
        assert!(n > 0 && n < 200);
        assert_eq!(confere_tamanho(&t, "Hi", &data, 64).unwrap(), n);
        // Fronteira exata: n + novos == 2048 cabe, mais um não.
        let grande = "word ".repeat(3000);
        let e = confere_tamanho(&t, &grande, &data, 1).unwrap_err();
        assert!(e.contains("passam de 2048 posições"), "{e}");
        let ids = conta_ids(&t, &grande, &data).unwrap();
        assert!(ids > 2048);
        // Procura um texto cujo prompt tem 2048 - 64 ids exatos.
        let mut k = 1;
        while conta_ids(&t, &"word ".repeat(k), &data).unwrap() < 2048 - 64 {
            k += 1;
        }
        let exato = "word ".repeat(k);
        let m = conta_ids(&t, &exato, &data).unwrap();
        assert_eq!(m, 2048 - 64);
        assert!(confere_tamanho(&t, &exato, &data, 64).is_ok());
        assert!(confere_tamanho(&t, &format!("{exato}another few words"), &data, 64).is_err());
    }

    #[test]
    fn trava_aceita_so_a_lista_explicita_de_uuids() {
        // UUID sintético (nunca o de uma máquina real): vive só neste teste
        let extra = "GPU-12345678-1234-1234-1234-123456789abc".to_string();
        let lista = vec![extra.clone()];
        // sem lista local só a 3080 Ti passa
        assert!(exige_uuid_com(Some(UUID_3080TI), &[]).is_ok());
        assert!(exige_uuid_com(Some(&extra), &[]).is_err(), "sem a lista local o UUID extra não passa");
        // com a lista, a 3080 Ti e o extra passam; fora da lista, vazio, 0, None e lista "a,b" não
        assert!(exige_uuid_com(Some(UUID_3080TI), &lista).is_ok() && exige_uuid_com(Some(&extra), &lista).is_ok());
        for v in [None, Some(""), Some("0"), Some("GPU-00000000-0000-0000-0000-000000000000"), Some("GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd")] {
            assert!(exige_uuid_com(v, &lista).is_err(), "{v:?}");
        }
        assert!(exige_uuid_com(Some(&format!("{UUID_3080TI},{extra}")), &lista).is_err(), "lista a,b recusada");
        assert!(exige_uuid_com(Some(&format!("{extra},{extra}")), &lista).is_err(), "lista a,b recusada");
        // a mensagem de recusa nunca imprime o UUID da lista local
        let msg = exige_uuid_com(Some("GPU-00000000-0000-0000-0000-000000000000"), &lista).unwrap_err();
        assert!(!msg.contains(&extra) && msg.contains("CUDA_VISIBLE_DEVICES"), "{msg}");
        // a lista local é validada: linha inválida recusa, sem eco do conteúdo
        assert_eq!(parse_uuids_extra(&format!("# gpu_5090\n\n{extra}\n")).unwrap(), lista);
        let e = parse_uuids_extra("nao-e-uuid\n").unwrap_err();
        assert!(e.contains("linha 1") && !e.contains("nao-e-uuid"), "{e}");
    }

    #[test]
    fn lista_local_por_arquivo_e_fail_closed() {
        let extra = "GPU-12345678-1234-1234-1234-123456789abc";
        let dir = std::env::temp_dir();
        let ok = dir.join(format!("lastro-uuids-teste-{}-ok.txt", std::process::id()));
        let ruim = dir.join(format!("lastro-uuids-teste-{}-ruim.txt", std::process::id()));
        std::fs::write(&ok, format!("{extra}\n")).unwrap();
        std::fs::write(&ruim, "lixo\n").unwrap();
        assert_eq!(le_uuids_extra(Some(ok.to_str().unwrap())).unwrap(), vec![extra.to_string()]);
        assert!(le_uuids_extra(None).unwrap().is_empty() && le_uuids_extra(Some("")).unwrap().is_empty());
        assert!(le_uuids_extra(Some(ruim.to_str().unwrap())).is_err(), "arquivo com linha inválida");
        assert!(le_uuids_extra(Some("/caminho/que/nao/existe/uuids.txt")).is_err(), "arquivo ausente: trava fechada");
        std::fs::remove_file(&ok).unwrap();
        std::fs::remove_file(&ruim).unwrap();
        assert!(uuid_valido(UUID_3080TI) && !uuid_valido("GPU-ABCDEF12-1234-1234-1234-123456789abc") && !uuid_valido("0") && !uuid_valido(""));
    }

    #[test]
    fn portoes_de_gpu() {
        assert!(exige_autorizacao(Some("1")).is_ok());
        for v in [None, Some(""), Some("0"), Some("true")] {
            assert!(exige_autorizacao(v).unwrap_err().contains(VARIAVEL));
        }
        assert!(exige_uuid(Some(UUID_3080TI)).is_ok());
        for v in [None, Some(""), Some("0"), Some("GPU-00000000-0000-0000-0000-000000000000"), Some("GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd")] {
            assert!(exige_uuid(v).unwrap_err().contains("CUDA_VISIBLE_DEVICES"), "{v:?}");
        }
        assert!(exige_uuid(Some(&format!("{UUID_3080TI},0"))).is_err(), "lista de devices");
        assert_eq!(UUID_3080TI, "GPU-b22d9465-8552-9f97-812e-380b590645f8");
        // composição: o portão real (portoes) só abre com a autorização e o UUID certos
        assert!(exige_autorizacao(Some("1")).is_ok() && exige_uuid(Some(UUID_3080TI)).is_ok());
        assert!(exige_autorizacao(Some("1")).is_ok() && exige_uuid(Some("GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd")).is_err());
    }

    #[test]
    fn portoes_reais_recusam_sem_autorizacao() {
        // o portão real: sem a variável de autorização (o ambiente de teste não a define), recusa antes de olhar o device
        if std::env::var(VARIAVEL).is_err() {
            assert!(portoes().unwrap_err().contains(VARIAVEL));
        }
    }

    // ---- O motor falso ----

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

    /// Passo `i` (0 = logits do prefill): pico num id que muda a cada passo; `eos_em` põe o pico no EOS (id 0) naquele passo.
    fn pico(passo: usize, eos_em: Option<usize>) -> usize {
        if eos_em == Some(passo) { 0 } else { 100 + (passo * 37) % 5000 }
    }

    fn sequencia(eos_em: Option<usize>) -> Result<Sequencia<FakeGpu, Prep>, String> {
        let (gpu, eng) = motor_falso();
        let passo = Arc::new(Mutex::new(0usize));
        let le: Box<dyn FnMut() -> Result<Vec<f32>, String>> = Box::new(move || {
            let mut c = passo.lock().unwrap();
            let i = *c;
            *c += 1;
            let mut v = vec![0.0f32; VOCAB];
            v[pico(i, eos_em)] = 5.0;
            Ok(v)
        });
        Ok(Sequencia { eng, prep: Prep { gpu, visto: Mutex::new(0), pos: Mutex::new(0) }, le_logits: le })
    }

    fn roda(texto: &str, novos: usize, eos_em: Option<usize>) -> (Result<gr::Geracao, String>, Vec<u8>) {
        let t = tok();
        let mut m: gr::MotorGeracao<FakeGpu, Prep> = gr::MotorGeracao::novo();
        let mut saida = vec![];
        let r = executa(texto, &data_fixa_texto(), novos, &t, &mut m, || sequencia(eos_em), &mut saida);
        m.encerra().unwrap();
        (r, saida)
    }

    #[test]
    fn saida_e_decode_dos_ids_mais_newline() {
        let t = tok();
        let (g, saida) = roda("Hi", 5, None);
        let g = g.unwrap();
        assert_eq!(g.gerados, (0..5).map(|i| pico(i, None)).collect::<Vec<_>>());
        let ids: Vec<u32> = g.gerados.iter().map(|x| *x as u32).collect();
        let esperado = format!("{}\n", t.decodifica(&ids).unwrap());
        assert_eq!(String::from_utf8(saida).unwrap(), esperado);
        assert!(esperado.ends_with('\n') && !esperado.ends_with("\n\n"));
    }

    #[test]
    fn eos_no_meio_para_e_nao_sai() {
        let t = tok();
        let (g, saida) = roda("Hi", 20, Some(3));
        let g = g.unwrap();
        // 3 tokens normais e o EOS; nada depois.
        assert_eq!(g.gerados.len(), 4);
        assert_eq!(*g.gerados.last().unwrap(), gr::EOS);
        let ids: Vec<u32> = g.gerados[..3].iter().map(|x| *x as u32).collect();
        assert_eq!(String::from_utf8(saida).unwrap(), format!("{}\n", t.decodifica(&ids).unwrap()));
        // EOS já no primeiro token: só o newline.
        let (g, saida) = roda("Hi", 20, Some(0));
        assert_eq!(g.unwrap().gerados, vec![gr::EOS]);
        assert_eq!(saida, b"\n");
    }

    #[test]
    fn novos_limites_no_laco() {
        let (g, _) = roda("Hi", 1, None);
        assert_eq!(g.unwrap().gerados.len(), 1);
        let (g, saida) = roda("Hi", 64, None);
        assert_eq!(g.unwrap().gerados.len(), 64);
        assert!(saida.ends_with(b"\n"));
    }

    #[test]
    fn falha_na_execucao_nao_imprime() {
        let t = tok();
        let mut m: gr::MotorGeracao<FakeGpu, Prep> = gr::MotorGeracao::novo();
        let mut saida = vec![];
        let r = executa("Hi", &data_fixa_texto(), 5, &t, &mut m, || Err("sem motor".into()), &mut saida);
        assert_eq!(r.unwrap_err(), "sem motor");
        assert!(saida.is_empty());
    }
}
