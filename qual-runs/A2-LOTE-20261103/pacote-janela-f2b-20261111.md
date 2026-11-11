# Pacote da janela F2 nível B (pré-roteador) em L100 (11/11)

## O que é
Uma janela por modelo (1B e 3B, uma de cada vez), de até 1200 s na RTX 3080 Ti (UUID GPU-b22d9465-8552-9f97-812e-380b590645f8; a 4090 e a 5090 nunca entram), só o Lastro, série L100, para abrir por dentro o `H_PRE_LAYER` do prefill. Resultado descritivo: HB1 a HB6 só conferidas, sem veredito geral, sem PASSA e sem frase de causa. `B_LANCA` é relatado como "tempo da chamada de envio de trabalho". Regras em `criterio-f2-nivel-b-20261111.md`; decisões do Breno de 12:03 em `.herdr/DECISAO-BRENO-NIVEL-B-TRES-PONTOS-20261111-1203.txt` (HB3 0,60/0,40; braços D, A e B2 com 9 processos por modelo; rótulo acima).

## FIXO
Conferido por `roda-f2b-janela.sh --verifica-fixos` (sem GPU, só lê) e de novo no modo real, ANTES de envelope, monitor, timers ou GPU (a recusa usa o código de saída do verificador, sem pipe que o mascare):
- critério do nível B e plano (versão 2);
- binário `primeira-palavra` do nível B, sha256 22d0bbe048d8c7407f73817b9879f5bc695604bc1bc519e332c9287c193adacf, cópia sem permissão de escrita em `/home/brenoperucchi/Devs/lastro-rust/.herdr/binarios/primeira-palavra-f2b-22d0bbe0`; planejamento pelos subcomandos puros do binário (`--planeja-celula`, `--planeja-controle`, `--planeja-tetos`);
- `f2b-extrai.py`, `f2-extrai.py` (reaproveitado) e os extratores que ele usa, `roda-cmp-lastro.sh`, `roda-cmp-monitor.sh`, `janela-e4b.sh`, `roda-com-limite.sh`;
- critérios do cmp e dos degraus, ids, textos e referências 5c, tokenizer do modelo.
O sha do próprio `roda-f2b-janela.sh` não pode estar nele: `--verifica-fixos` o imprime à parte e a `decisao.txt` de cada janela o registra.

## Como abrir (só com o "autorizo" do Breno que nomeia a janela; o voxtype é cuidado pelo script de abertura em `.herdr`, não por este roteiro)
```
cd /home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103
./roda-f2b-janela.sh --verifica-fixos
PRIMEIRA_PALAVRA_AUTORIZADA=1 CUDA_VISIBLE_DEVICES=GPU-b22d9465-8552-9f97-812e-380b590645f8 \
  ./roda-f2b-janela.sh --modelos 1b --saida <pasta-nova>
```
Depois o mesmo para `--modelos 3b`, em outra pasta nova. `--modelos` é obrigatório. As três variáveis do instrumento (`LASTRO_PERFIL_BLOCO`, `LASTRO_PERFIL_F2`, `LASTRO_PERFIL_F2B`) não podem vir no ambiente (recusa 2); cada processo as recebe por `env`. Se existir `s3080-f2-<modelo>-real-20261111-1100` ao lado do roteiro, o extrator o usa para a comparação de `H_PRE_LAYER` "entre janelas".

## O que a janela faz (ordem fixa, sem repetição automática)
1. Preflight: portão, UUID, 370 W, nenhum `cargo test`, `primeira-palavra` ou `llama.*` em execução, nenhum processo C ou C+G na 3080 Ti, sha do binário. O bloco obrigatório (smoke + 9 tetos + folga de 180 s) precisa caber no limite; senão a janela não começa (código 2, `decisao.txt`).
2. Smoke: 1 processo no curto, em B2, 1 aquecimento e 1 medida, teto de 60 s. Falha (inclusive IB no aquecimento) = a janela segue e o extrator marca todas as H do modelo SEM PROVA.
3. Controle de 9 processos no médio, `--repete 3`, ordem D A B2, B2 D A, A B2 D (D nada ligado, A só o perfil por bloco, B2 as três variáveis). Cabe inteiro ou os nove viram NÃO RODOU; nunca é cortado; o primeiro processo com erro encerra o controle.
4. Curto e longo em B2 (1 processo cada, 1 aquecimento mais 5 medidas). Corte único e permanente para 3 medidas, decidido por `primeira-palavra --planeja-celula`; NÃO RODOU só pela célula atual com 3 medidas.
- Tetos de processo: 60 s (1B) e 90 s (3B) no médio e no curto; 75 s (1B) e 120 s (3B) no longo. O teto de 1200 s é global e separado.
- Ao fim: integridade antes e depois, 370 W de novo, `orcamento.json`, `f2b-extrai.py` e `decisao.txt`. Controle de B2 contra A: até 3% comprovada, até 10% com ressalva, acima de 10% SEM PROVA.

## Saídas
`f2b-host-resultado.md` e `f2b-host-resultado.json` (por modelo), `decisao.txt`, `ordem.txt`, `prova-variavel.txt`, `plano.txt`, `orcamento.json`, `teto-excedido.txt`, `integridade-*.txt`, `celulas/`. O extrator preserva a I8 de cada pedido: a I8 do aquecimento ou do pedido medido do smoke, e a divergência de ids ou referência ausente do smoke (aquecimento incluído, fora das medianas), reprovam o smoke e põem todas as H do modelo em SEM PROVA; fora do smoke a I8 só é registrada (`nota_i8`). A comparação com 11:00 registra o sha256 dos `perfil-f2.json` lidos. Testes sem GPU: `test-roda-f2b-janela.sh` e `test-f2b-extrai.py`.
