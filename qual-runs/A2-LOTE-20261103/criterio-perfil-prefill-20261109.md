# Critério: perfil do prefill do Lastro, só por medição (09/11/2026, lastro-rust-exec; versão 2, após a rodada de revisão 121)

Escopo: descobrir onde vai o tempo do prefill (Lastro 17 a 45x atrás do llama.cpp no degrau 1; 2,8 a 17x no degrau 2). **Nenhuma linha de código muda**: binário `primeira-palavra` e `llama-server` como estão, mesmos modelos BF16. 4090 só, sem 5090, GPU livre (voxtype parado, Chrome sem GPU), janela única ≤ 20 min, sem repetição automática. Descritivo, sem meta nem PASSA.
Mudanças da v2 (r121, dois REQUEST_CHANGES): H1 reescrita (o prefill já é em blocos de 8, `BLOCO=8`); P3 antiga retirada (os dados a refutavam); Parte B passa a usar o caminho de prova existente (`--saida`, `--textos-5c`, `--so-texto`, `--repete`) em vez de `--novos 1`; P4 de GPU vira só descrição; texto sobre o S1 corrigido.

## Parte A: o que os dados já existentes mostram (sem GPU, exploratório)
Declaração honesta: calculado com as 6 janelas já feitas, antes de escrever este critério.
Prefill quente (mediana) em ms por posição do prompt (89, 170, 364 posições, template incluído):
- 1B: L10 8,9 / 9,0 / 8,5; L100 1,58 / 1,58 / 1,58; camadas-CPU 8,8 / 8,8 / 8,7. 3B: L10 17,5 / 17,7 / 17,4; L100 2,52 / 2,54 / 2,55; camadas-CPU 17,5 / 17,4 / 17,2.
- Razão custo por posição ÷ passo de decode da mesma série: L100 0,66 a 0,67 (1B 1,58/2,37; 3B 2,52/3,8); L10 e camadas-CPU 0,41 a 0,43.
- llama.cpp S1 quente: 6,9 a 12,8 ms (1B) e 12,9 a 29,3 ms (3B) para 89 a 364 posições (cresce 1,9 a 2,3x para 4,1x posições). S2: ~225 a 243 ms (1B) e ~550 a 590 ms (3B), quase independentes do tamanho.
Leitura exploratória: no Lastro o prefill é **linear nas posições**, custo por posição constante. Isso é compatível com blocos fixos de 8 posições (o motor lê o prompt em `plano_blocos(n, 8)`, um passo por bloco) e **não distingue** esse mecanismo de processamento posição a posição. H1: o gargalo é o custo de um bloco de 8 posições no motor, que vale hoje ~5,3 passos de decode no L100 do 1B (8×1,58 ≈ 12,6 ms contra 2,37 ms) e ~3,3 no L10; o lote quase não amortiza. A parte B não busca causa interna; descreve escala.

## Parte B: confirmação de escala por medição (1 janela, sem código novo)
Rota existente: `primeira-palavra --gera --celula M:<modelo> --saida <dir> --textos-5c <dir novo> --so-texto X --repete 5` (64 tokens, prefill lido à parte, data e template fixos da prova). Diretório novo de textos com os três nomes exigidos: `texto-curto.txt` = corte do texto longo com N≈120 posições, `texto-medio.txt` = corte com N≈250, `texto-longo.txt` = o longo original (364, controle contra as janelas anteriores). N conta posições do prompt com template; N real de cada corte é impresso por `--gera --ensaio --textos-5c <dir novo>` (sem GPU) e registrado com os sha dos textos antes da janela. 1B e 3B, L100 e camadas-CPU (22 no 1B, 29 no 3B); llama.cpp S1 e S2 nos mesmos ids (`n_predict` 64, `cache_prompt=false`, prefill da resposta). 5 medidas quentes por célula; mediana. A comparação com `ref5c` só vale no controle (364); nos cortes a divergência de ids é esperada e não invalida a célula.
Previsões pré-registradas (conferir; nenhuma é meta):
- P1: no Lastro, ms por posição varia no máximo ±15% entre N≈120, 250 e 364 em cada série. Resultado linear é compatível com blocos de 8 e não prova ausência de lote nem causa.
- P2: no llama.cpp S1, prefill de N≈364 fica abaixo de 3x o de N≈120; no S2, abaixo de 1,5x.
- P3: as razões ms/posição ÷ passo de decode ficam em 0,66 a 0,67 ±15% no L100 e 0,41 a 0,43 ±15% no L10/camadas-CPU (as razões da parte A, agora testadas em comprimentos novos).
P4 (só descrição, sem previsão): `nvidia-smi -i <UUID> --query-gpu=timestamp,utilization.gpu --format=csv,noheader -lms 100` e `%CPU` do processo (`/proc`) durante a célula inteira, que inclui carga e criação do motor; os carimbos de início e fim do prefill não saem do binário, então a leitura é da execução toda e não do prefill; não serve para atribuir causa.
Validade como no critério base (integridade FIXA, preflight, portão de GPU). Limite: não separa atenção, experts, cópia e sincronização dentro do bloco (exige instrumento no motor, outra unidade).
Saída: tabela N × série (prefill_ms, ms/posição), razão contra o llama.cpp, P1 a P3 com o dado ao lado, P4 descritivo, e próximos passos sugeridos.
