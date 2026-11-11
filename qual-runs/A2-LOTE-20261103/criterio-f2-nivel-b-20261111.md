# Critério: F2 nível B, só o pré-roteador (`H_PRE_LAYER` por dentro), prefill do Lastro em L100, janelas na 3080 Ti (11/11/2026, antes de qualquer dado; a versão em vigor é a do sha registrado no FIXO do roteiro)

Deriva só do plano `plano-f2-nivel-b-pre-roteador-20261111.md` (versão 2, aprovada na rodada 158): seções 3 (o que se mede), 4 (instrumento), 5 (custo e controle), 6 (invariantes), 7 (hipóteses), 10 (ordem) e 12 (alcance). Descritivo: sem meta, sem PASSA, sem veredito geral, sem frase causal. O tempo com instrumento não é o tempo de produção. Onde o plano não traz um número ou uma regra, este texto NÃO inventa; os três itens que estavam abertos foram decididos pelo Breno (seção 0). Base do alcance: `/home/brenoperucchi/Devs/lastro-rust/.herdr/DECISAO-BRENO-ALCANCE-INSTRUMENTO-NIVEL-B-20261111-1139.txt`.

## 0. Decisões do Breno de 11/11 12:03 para os três itens antes pendentes (arquivo `/home/brenoperucchi/Devs/lastro-rust/.herdr/DECISAO-BRENO-NIVEL-B-TRES-PONTOS-20261111-1203.txt`)
- (a) HB3: aceita as faixas do rascunho do plano, dentro se >= 0,60, fora se < 0,40, sem leitura no meio; o número bruto é sempre publicado. No código: `LIMIARES_HB3_PLANO`.
- (b) Controle: três braços D (sem instrumento), A (instrumento por bloco) e B2 (instrumento novo completo), 9 processos por modelo. No código: `CONTROLE_B_PLANO`, o padrão. `CONTROLE_B_A_B2` e `CONTROLE_B_B1_A_B2` ficam no código como alternativas NÃO USADAS, sem decisão que as sustente; a ordem de 6 processos A/B2 é derivação antiga e não vale.
- (c) Relatório: fala em "tempo da chamada de envio de trabalho" (a expressão `async_on`), sem prometer isolar `cuLaunchKernel`. No código: `RotuloLanca::SoAsyncOn`.

## 1. Escopo e ambiente
- Só L100, por modelo (1B e 3B), em duas janelas separadas, uma invocação por modelo, no máximo 1200 s cada (teto global, compartilhado entre smoke, controle e células; corte só em células não iniciadas; o controle nunca é cortado), sem repetição automática depois de SEM PROVA.
- Só a RTX 3080 Ti, UUID GPU-b22d9465-8552-9f97-812e-380b590645f8. 4090 e 5090 nunca entram. Só o Lastro.
- Sem mudança de desempenho no motor: só medição. Fora: `route`, `launch_hits`, `add_y`, `post_layers` por dentro (continuam só no nível A), tempo na GPU, Nsight, eventos CUDA, `cuLaunchKernel` isolado, F0, F1, F3, L10, LC, decode, 4090, 5090, llama.cpp, Strata, push.
- Integridade como no critério do nível A: voxtype parado antes e religado depois (passo manual do Breno), `power.limit` de 370 W conferido no começo e no fim, nenhum `cargo test`/`bench` nem outro uso da GPU durante a janela, portão de sha do binário antes de tocar em timers.

## 2. O que liga e o que sai
- `LASTRO_PERFIL_F2B`: só o valor exato `1` liga; ausente, vazia, `0`, `true`, ` 1` ou não UTF-8 = desligado. Exige `LASTRO_PERFIL_BLOCO=1` e `LASTRO_PERFIL_F2=1`; sem as duas a linha de estado diz `perfil-f2b: desligado (exige perfil-bloco e perfil-f2)` e nada muda. Linhas de estado: `perfil-f2b: ligado` ou `perfil-f2b: desligado`.
- Desligado (padrão): o motor guarda `None` no lugar do acumulador, chama `pre_layer` como antes e as variantes internas recebem `Option` vazio: sem acumulador, sem relógio, sem alocação, nenhuma mudança de comportamento. A promessa é só a do plano: um `Option` vazio por ponto de medida. Nenhum teste prova "nenhum branch".
- Saída separada `perfil-f2b.json` (esquema próprio, versão 1) com os registros brutos (op, camada, posição, início, fim) por bloco. `perfil-bloco.json` versão 2 e `perfil-f2.json` não mudam. Soma e contagem por operação são derivadas pelo extrator.

## 3. O que se mede (plano, seção 3)
Registros por camada e bloco (m posições): `B_LOCK` (1 por camada), `B_SETUP` (2 por posição), `B_GENERICS` (1 por posição), `B_ATTN_PREP` (1 por posição), `B_LANCA` (11 por posição: 9 em `attn_block` e 2 em `attention`), `B_DTOD` (6 por posição). `B_RESIDUO` = `H_PRE_LAYER` menos a soma dos intervalos medidos, calculado depois, nunca ajustado.
- `B_LOCK` mede o custo de pegar uma trava livre (sem disputa, plano seção 2), na resolução do relógio.
- `B_LANCA` é, no relatório, o "tempo da chamada de envio de trabalho" (a expressão `async_on`) (decisão do Breno, seção 0 (c)).
- Não se mede execução na GPU, espera da GPU, fila do driver, tempo por kernel na placa nem CPU fora da thread do host.

## 4. Invariantes (IB1 a IB5), por bloco com a F2 ligada
Tolerância de relógio `TOL_RELOGIO_NS` = 1 µs. Os carimbos nunca são ajustados.
- IB1: registros em ordem e sem sobreposição; o primeiro início e o último fim de cada camada dentro de `[ini, fim]` do `H_PRE_LAYER` da mesma camada no `perfil-f2.json` (1 µs). Falha invalida o pedido.
- IB2: soma dos intervalos mais `B_RESIDUO` = `H_PRE_LAYER`. É identidade por construção; vale como teste de cobertura e excesso: resíduo < -1 µs (soma dos intervalos maior que a operação) invalida o pedido. A conservação independente vem da IB4.
- IB3: sequência esperada na ordem de emissão: por camada (0 a layers-1, sem repetir nem faltar) um `B_LOCK` com posição vazia e, para cada posição 0..m-1 em ordem, 21 registros (`B_SETUP` 2, `B_GENERICS` 1, `B_ATTN_PREP` 1, `B_LANCA` 11, `B_DTOD` 6, na ordem de `ORDEM_POS_B`); totais por camada 1, 2m, m, m, 11m, 6m; nenhum registro perdido por capacidade. Troca de posições que preserve só as contagens falha. Divergência invalida o pedido. Com o `Fake` e F2B ligada a contagem é zero e o pedido sai inválido por esta invariante (esperado, não se relaxa).
- IB4: soma de `H_PRE_LAYER` mais as demais operações mais `R_borda` = t3 - t2 (I2 do nível A, 2 µs), e `gen`, `m`, `pos_ini` do bloco de F2B iguais aos do bloco do `perfil-f2.json` e do `perfil-bloco.json`.
- IB5 (ressalva): `B_RESIDUO` / `H_PRE_LAYER` <= 3%; acima, ressalva de cobertura, nunca invalida.
- I7 do nível A (saída intacta): ids gerados em B2 iguais aos de D e de A (e de B1 quando existir). Falha invalida o processo.
- Blocos tratados à parte, fora das medianas: o primeiro bloco de cada pedido e o bloco final com resto (m < 8).

## 5. Hipóteses e classes (por modelo, sobre blocos cheios elegíveis, mediana por pedido e depois mediana entre os pedidos da célula)
Classes: dentro, fora, sem leitura, sem dado, SEM PROVA, descritivo. Nenhuma frase causal. Parcela = soma, entre as camadas, da mediana por camada (entre os blocos do pedido) da soma do grupo, dividida pela soma, entre as camadas, da mediana por camada de `H_PRE_LAYER`.
- HB1: parcela de `B_LOCK`. Descritiva. Texto obrigatório: "mede o custo de pegar uma trava livre".
- HB2: parcela de `B_SETUP` + `B_GENERICS` + `B_ATTN_PREP`. Descritiva.
- HB3: parcela de `B_LANCA`. Dentro se >= 0,60, fora se < 0,40, sem leitura no meio. (decisão do Breno, seção 0 (a))
- HB4: parcela de `B_DTOD`. Descritiva.
- HB5 uniformidade: p90/mediana de `H_PRE_LAYER` por camada (posto mais próximo, `ceil(0,9 n)`, como no nível A) e dispersão entre camadas (maior sobre menor mediana por camada). Descritiva.
- HB6 escala: `H_PRE_LAYER` contra o índice da camada e contra `m` e `pos_ini`. Descritiva, classe "descritivo" (ou "sem dado"); `H_PRE_LAYER` por camada vem das medianas por camada, e a série contra `pos_ini` é calculada pelo extrator a partir dos registros brutos e do `perfil-f2.json` (com blocos cheios `m` é sempre 8).
- Medida do instrumento sobre o motor (denominador): as parcelas dividem por `H_PRE_LAYER` do B2, já inflado pelo instrumento. O relatório traz a mediana de `H_PRE_LAYER` por camada do B2 ao lado da do B de 11:00 (F2 sem F2B), com a ressalva "entre janelas".
- Controle > 10% (B2 contra A): todas as hipóteses daquele modelo SEM PROVA, dados brutos preservados.

## 6. Controle do custo do instrumento
- Mesma janela e classes do nível A. Braços por modelo, só em L100 texto médio (170 posições, resto 2): D (sem variáveis), A (`BLOCO=1`) e B2 (`BLOCO=1`, `F2=1`, `F2B=1`) (decisão do Breno, seção 0 (b)). Quadrado latino de 9 processos por modelo (D A B2, B2 D A, A B2 D), cada processo com `--repete 3`; mínimo 3 processos válidos por braço.
- Medida principal: |desvio| do `prefill_ms` de B2 contra A: <= 3% comprovada; <= 10% com ressalva; > 10% SEM PROVA daquele modelo (dados brutos preservados, sem repetição automática). Apoio: B2 contra D (ou contra B1, se a alternativa for escolhida), classe só descritiva.
- Reportar pior caso, ms por bloco (divisor 22, "diferença de prefill por bloco executado, incluindo primeiro e resto") e desvio do `passo_ms` do decode.
- A variável é passada por processo (`env VAR=1` nos que ligam, `env -u VAR` nos que desligam), com `prova-variavel.txt`, e recusa se o ambiente de quem chama já trouxer qualquer uma das três. Cada braço confere suas linhas de estado e seus arquivos (D nenhum perfil; A só `perfil-bloco.json`; B2 os três).

## 7. Mínimo de amostras
Como no nível A: 3 pedidos válidos em curto e longo e os 9 do médio (eleg. 30, 180 e 132 depois de excluir o primeiro bloco de cada pedido, com corte de 5 para 3 pedidos). Falta do mínimo = SEM PROVA na classificação afetada. Pedido com problema estrutural ou esquema inválido não conta como válido. Controle incompleto invalida todas as hipóteses do modelo.

## 8. Orçamento de tempo
No máximo 1200 s por janela/modelo, global, sem repetição automática após SEM PROVA. Tetos de processo e regra de corte: os do critério do nível A (60 s e 90 s no médio e no curto, 75 s e 120 s no longo, para 1B e 3B), sem afrouxar nenhum. Orçamento insuficiente para o controle = controle incompleto, e todas as hipóteses do modelo ficam SEM PROVA.

## 9. FIXO do pacote (por lista; sha256 registrado antes da primeira janela)
| Item | Estado |
|---|---|
| Este critério (`criterio-f2-nivel-b-20261111.md`) | sha a registrar no FIXO do roteiro da janela; qualquer edição exige atualizar o FIXO |
| Plano `plano-f2-nivel-b-pre-roteador-20261111.md` (versão 2) | sha a registrar |
| Binário instrumentado `primeira-palavra` (cópia congelada) | sha a registrar depois da revisão |
| Binários `lastro` e `lastro-servidor` (mesma compilação) | sha a registrar |
| Extrator e roteiro da janela do nível B | a escrever depois da revisão (fora desta etapa) |
| `--verifica-fixos` antes da janela | passo 6 do plano |

## 10. Pedido aos revisores (plano, seção 12)
Conferir, no código e nos testes: (a) F2B desligada por padrão e só liga com as três variáveis; (b) o padrão do trait é idêntico ao `pre_layer` atual; (c) as variantes de `attn_block_at`, `attn_block` e `attention` mantêm ordem de kernels, argumentos, stream, caminhos de erro, swaps e restauração, e o chamador de `gpu_model.rs` (caminho de `attention` sem medida) não muda; (d) o desligado não aloca nem lê relógio; (e) nada fora do instrumento foi alterado no motor. Os três itens da seção 0 estão decididos pelo Breno.
