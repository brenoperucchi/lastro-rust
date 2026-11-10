# Critério: instrumento de prefill do Lastro, janela na 3080 Ti (10/11/2026, congelado antes de qualquer dado)

Deriva do plano aprovado `plano-instrumento-prefill-20261110.md` (seção 1.3 limiares H1 a H6, seção 2.1 controle, seção 4 validade) e da nota `instrumento-prefill-implementacao-20261110.md` (esquema de `perfil-bloco.json` versão 2, campo `modelo`, classificadores `ClasseH`). Descritivo: sem meta, sem PASSA, sem veredito geral. Não prova causa. O tempo com instrumento não é o tempo de produção. H7 e o nível GPU ficam para a segunda unidade.

## 1. Janelas e ambiente
- Duas janelas, uma por modelo (1B e 3B), em invocações separadas de `roda-instrumento-janela.sh --modelos 1b|3b`, cada uma com no máximo 20 min (limite 1200 s). Sem repetição automática.
- Só a RTX 3080 Ti, UUID GPU-b22d9465-8552-9f97-812e-380b590645f8. A 4090 e a 5090 nunca entram. Só o Lastro (sem llama.cpp).
- Binário: cópia congelada `.herdr/binarios/primeira-palavra-instr-ae14cbde`, sha256 ae14cbde3965493808818f3aa4a8e7665b6397531e36c2abf020c35fc4f3d969. O portão do binário aceita só a 3080 Ti.
- Potência: `power.limit` de 370.00 W exigido no início (recusa 2 antes de tocar em timers) e lido de novo ao fim. Mudança, leitura ausente ou ilegível = integridade quebrada; o extrator RECUSA (código 2, sem classificar) sem a prova `potencia 370.00_W 370.00_W OK` coerente com `potencia_inicio` e `potencia_fim` de `ambiente.txt`.
- Voxtype parado antes (`systemctl --user stop voxtype`, passo manual) e religado ao fim (`start`). Qualquer PID C ou C+G na 3080 Ti recusa a janela (preflight pmon).
- Recusa de iniciar (código 2, antes de timers e GPU) se houver `cargo test`, `cargo nextest`, `cargo bench`, binário de teste do cargo (`target/*/deps/<nome>-<hash>`), `primeira-palavra` ou `llama-server/bench/cli` em execução (incidente do rev-2 da rodada 137: `cargo test` sem filtro ocupou a 3080 Ti por ~10 min). Os testes deste pacote rodam sempre com `CUDA_VISIBLE_DEVICES=` vazio e com filtro.
- Variáveis só de teste no ambiente do modo real recusam (código 2); inclui `LASTRO_PERFIL_BLOCO`.

## 2. A variável do instrumento (ponto P3 do rev-2 da rodada 137)
`LASTRO_PERFIL_BLOCO=1` liga só nos processos L, nunca nos D. O roteiro: (a) recusa se o ambiente de quem chama a trouxer; (b) passa a variável por processo, na chamada (`env LASTRO_PERFIL_BLOCO=1` nos L; `env -u LASTRO_PERFIL_BLOCO` nos D, nos smokes e nas demais séries sem ela); (c) grava em `prova-variavel.txt`, por processo, o ambiente passado e a linha que o binário imprimiu (`perfil-bloco: ligado|desligado`); D tem de imprimir desligado e não produzir `perfil-bloco.json`, L tem de imprimir ligado e produzi-lo. Incoerência vai para a decisão e o extrator invalida o controle (D com perfil ou "ligado", L sem perfil ou sem "ligado" = controle incompleto). Processo sem a linha invalida a célula; nenhum processo com a linha = o extrator recusa.

## 3. Células e ordem
Ordem fixa: (1) preflight e smoke L100; (2) pares de controle; (3) demais células obrigatórias; (4) opcionais (LC). Textos 5c: curto 89, médio 170, longo 364 posições (restos 1, 2 e 4). Séries: L10 (padrão), L100 (`--fracao-slots 1.0`), LC (`--camadas-cpu` 22 no 1B e 29 no 3B).
- (2) Por série L10 e L100, no texto médio: 6 processos na ordem D L L D D L, cada um `--repete 3` (1 aquecimento descartado + 3 medidas). Mínimo inviolável: 3 processos válidos por braço. Os 3 processos L valem como dados principais de (L10, médio) e (L100, médio): 9 pedidos medidos.
- (3) L10 e L100 em curto e longo: 1 processo, 1 aquecimento + 5 medidas; único corte: 5 para 3, nas células ainda não iniciadas.
- (4) LC curto, médio, longo, com smoke antes, só se couber; sem controle, descritiva.
- Orçamento: o bloco (2) é reservado antes de tudo. Antes de tocar em timers o roteiro conta smoke L100 + 12 processos x (portão + carga + 4 x pedido quente) + folga de 180 s; se não couber no limite, a janela NÃO COMEÇA (código 2, `decisao.txt` "SEM PROVA da unidade inteira"), sem encolher o par. Estimativas (palpites): portão 8 s, carga 5 s (1B) e 7 s (3B), pedido quente 5 s (1B) e 9 s (3B), smoke 20 s e 30 s. Cortes só atingem (3) e (4).
- Processo de controle com erro: encerra o controle daquela série, sem repetição; o resto vira NÃO RODOU (prefixo `ctrl-<modelo>-<série>-` em `orcamento.json`) e o controle fica incompleto.
- Smoke: falha marca SÓ a série (L100 ou LC) como SEM PROVA; código 2 interrompe a janela.

## 4. Controle do custo do instrumento (por modelo e série)
desvio_prefill = (mediana das medianas por processo de `prefill_ms` em L menos a de D) / a de D. Classe pela magnitude |desvio| (3% e 10% inclusivos; EPS 1e-12 contra ponto flutuante): <= 3% comprovada; <= 10% com ressalva (frações relativas ao tempo medido com instrumento, nunca ao do D); > 10% todas as H da (modelo, série) SEM PROVA, dados brutos preservados. Reportados também: pior caso (maior L menos menor D, sobre a menor D), desvio em ms por bloco e desvio do `passo_ms`. Controle ausente, truncado, fora de ordem, com processo abortado ou incoerente na variável: "controle incompleto", H1 a H6 e custo SEM PROVA.

## 5. Hipóteses (limiares do plano, 1.3; `>=`, `<=`, `<`, `>` como escritos)
Medianas por (modelo, série, texto), de pedidos medidos, de blocos cheios (m = 8). Saídas: dentro, fora, sem leitura (faixa intermediária), sem dado (ausente, ou série sem previsão), SEM PROVA, descritivo.
- H1 (L100): dentro se F2/B >= 0,70; fora se F3/B >= 0,50; as duas juntas = sem dado.
- H2 (L100): dentro se F3/B >= 0,50 e F3/passo(D) >= 3,0; fora se F3/B < 0,25 ou F3/passo(D) < 1,5.
- H3: L10 dentro se R3 >= 0,50 e S_compute/B >= 0,20, fora se R3 < 0,30 ou S_compute/B < 0,10; L100 dentro se R3 <= 0,10, fora se R3 > 0,25. R3 = (S_compute + S_livro)/B, soma integral; a versão recortada em [t0, t4] sai à parte.
- H4 (L100): dentro se F1/B >= 0,05; fora se F1/B < 0,02.
- H5: L10 dentro se (S_livro + S_end_trab)/B >= 0,10, fora se < 0,03; L100 dentro se <= 0,02, fora se > 0,05.
- H6: R6 mediano por pedido; dentro se R6 >= 0,5 + r/16; fora se R6 <= (r/8) x 1,25; resto r de 1 a 7; faixas que se tocam (r = 6, 7) = texto descritivo. Restos 1, 2 e 4: faixas não se tocam.
- Validade: shas antes e depois, ids iguais à 5c, potência, GPU livre, `perfil-bloco.json` com uma entrada por pedido (1 aquecimento + medidas), `plano_m` igual a `plano_blocos(n, 8)`, `valido` do binário (conservação da categoria A e do serviço), carimbos coerentes. Célula com falha = SEM PROVA só dela.
- Cobertura (ressalva, não invalida): host (R_ini + R_entre + R_fim)/prefill_ms <= 3%; serviço, mediana de S_fora/I_g <= 10%. Pedido com serviço aberto sai das estatísticas de serviço (contado).

## 6. Interpretações (onde o plano deixa margem; a revisão deve confirmar)
1. **Extrapolação do controle (severidade possível: ALTA).** O plano (2.1) diz que o desvio em ms por bloco medido no médio "tem de bater" com o dos outros textos, com variação de até 10%, só como checagem interna. Leitura LITERAL adotada: o B mediano por bloco cheio de cada texto da série (pedidos L); se (max - min)/min > 10%, curto e longo ficam SEM PROVA nas H dessa série (o médio segue). Se na prática o B por bloco variar com o contexto além de 10%, esta regra apaga curto e longo em quase tudo. Alternativa a decidir: tratar como ressalva.
2. **Ids contra a 5c (severidade possível: ALTA).** Divergência da referência 5c em QUALQUER processo da série (D ou L) invalida o instrumento nessa série (H SEM PROVA), como o plano manda. A 5c pode ter sido gerada em outra GPU, e então a divergência não seria do instrumento; por isso o relatório informa também se os ids de L e D do controle coincidem. Na LC a divergência é só reportada.
3. **Razões das H:** razão de medianas (mediana do numerador sobre mediana de B, no mesmo conjunto de blocos), não mediana de razões por bloco. F0 exclui blocos `f0_pendente` (reportados à parte). O denominador de H2 é o `passo_ms` mediano do braço D do texto médio, usado também em curto e longo.
4. **Cobertura:** a do host é a mediana por célula; a do serviço é a mediana de S_fora/I_g por série. Só marcam ressalva.
5. **LC:** sem controle, todas as H saem "descritivo" e a série tem ressalva fixa.
6. **Séries sem previsão** (L10 em H1, H2, H4; LC) = "sem dado", não "sem leitura".
7. **Smoke LC** só roda se couber depois das obrigatórias; falha de smoke de uma série não para as outras.
8. **Controle com erro de processo:** encerra o controle da série sem repetir (plano: sem repetição automática); o custo e as H dessa série ficam SEM PROVA, as células principais seguem.

## 7. FIXO do pacote
O roteiro congela, com `--verifica-fixos`: este critério, o plano, o binário instrumentado, `instrumento-extrai.py`, `perfil-extrai.py`, `cmp-extrai.py`, `roda-cmp-lastro.sh`, `roda-cmp-monitor.sh`, `janela-e4b.sh`, `roda-com-limite.sh`, os critérios cmp, degrau 1 e degrau 2, ids, textos 5c, referências 5c e tokenizers. Os shas vivem no FIXO do roteiro (o critério não pode conter o do roteiro nem o próprio). O sha do roteiro sai como linha informativa de `--verifica-fixos`.
