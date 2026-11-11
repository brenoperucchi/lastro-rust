# Pacote da janela F2 do host por operação em L100 (11/11)

## O que é
Uma janela por modelo (1B e 3B, uma de cada vez), de até 1200 s na RTX 3080 Ti (UUID GPU-b22d9465-8552-9f97-812e-380b590645f8; a 4090 e a 5090 nunca entram), só o Lastro, série L100, para medir onde o tempo do host se divide dentro do prefill. O resultado é descritivo: HA a HI são previsões pré-registradas só conferidas, sem veredito geral, sem PASSA e sem frase de causa. Nenhuma mudança de desempenho entra nesta fase. Regras em `criterio-f2-host-20261111.md` (seção 10 decidida pelo Breno em 11/11 08:20); janelas na 3080 Ti autorizadas em caráter permanente pela decisão de 07:00.

## FIXO
Conferido por `roda-f2-janela.sh --verifica-fixos` (sem GPU, só lê) e de novo, no modo real, ANTES de envelope, monitor, timers ou GPU (a recusa usa o código de saída do verificador, sem pipe que o mascare):
- critério e plano (versão 3);
- binário `primeira-palavra` F2, sha256 e182736f4133b494a14847f2cae5edd3dad91a7eeadf9a850180c8c52b0ba2e3, cópia em `/home/brenoperucchi/Devs/lastro-rust/.herdr/binarios/primeira-palavra-f2-e182736f`;
- cópia do binário sem permissão de escrita; subcomandos de planejamento puros (`--planeja-celula`, `--planeja-controle`, `--planeja-tetos --modelo 1b|3b`), sem GPU;
- `f2-extrai.py` e os extratores reaproveitados (`instrumento-extrai.py`, `perfil-extrai.py`, `cmp-extrai.py`), `roda-cmp-lastro.sh`, `roda-cmp-monitor.sh`, `janela-e4b.sh`, `roda-com-limite.sh`;
- critérios do cmp e dos degraus, ids, textos e referências 5c, tokenizer do modelo.
O sha do próprio `roda-f2-janela.sh` não pode estar nele: `--verifica-fixos` o imprime à parte (linha `informativo`) e a `decisao.txt` de cada janela o registra.

## Como abrir (só com o "autorizo" do Breno que nomeia a janela)
```
cd /home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103
./roda-f2-janela.sh --verifica-fixos
PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=GPU-b22d9465-8552-9f97-812e-380b590645f8 \
  ./roda-f2-janela.sh --modelos 1b --saida <pasta-nova>
```
Depois o mesmo para `--modelos 3b`, em outra pasta nova. `--modelos` é obrigatório e as duas variáveis do instrumento (`LASTRO_PERFIL_BLOCO`, `LASTRO_PERFIL_F2`) não podem vir no ambiente (recusa 2). O voxtype é parado antes e religado depois; passo manual do Breno ao fim.

## O que a janela faz (ordem fixa, sem repetição automática)
1. Preflight: portão, UUID, 370 W, nenhum `cargo test`, `primeira-palavra` ou `llama.*` em execução, nenhum processo C ou C+G na 3080 Ti, sha do binário. O bloco obrigatório (smoke + 9 tetos + folga de 180 s) precisa caber no limite; senão a janela não começa (código 2, `decisao.txt`).
2. Smoke: 1 processo no curto, em B, 1 aquecimento e 1 medida, teto de 60 s. Se falhar (inclusive por I8 do aquecimento do smoke), a janela segue e o extrator marca todas as H do modelo SEM PROVA.
3. Controle de 9 processos no médio, `--repete 3`, ordem D A B, B D A, A B D (D nada ligado, A só o perfil por bloco, B perfil por bloco e F2). Cabe inteiro ou os nove viram NÃO RODOU; nunca é cortado. O primeiro processo com erro encerra o controle.
4. Curto em B e longo em B (1 processo cada, 1 aquecimento mais 5 medidas). Corte único e permanente para 3 medidas, decidido por `primeira-palavra --planeja-celula`; NÃO RODOU só pela célula atual com 3 medidas.
- Tetos de processo (matam o grupo do processo, entram em `teto-excedido.txt`): 60 s (1B) e 90 s (3B) no médio e no curto; 75 s (1B) e 120 s (3B) no longo. O teto de 1200 s é global e separado.
- Ao fim: integridade antes e depois (shas), 370 W de novo, `orcamento.json`, `f2-extrai.py` e `decisao.txt`.

## Saídas
`f2-host-resultado.md` e `f2-host-resultado.json` (por modelo), `decisao.txt`, `ordem.txt`, `prova-variavel.txt`, `plano.txt`, `orcamento.json`, `teto-excedido.txt`, `integridade-*.txt`, `celulas/`. Testes sem GPU: `test-roda-f2-janela.sh` e `test-f2-extrai.py`.
