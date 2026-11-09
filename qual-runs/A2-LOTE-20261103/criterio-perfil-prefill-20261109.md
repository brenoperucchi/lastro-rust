# Critério: perfil do prefill do Lastro, só por medição (09/11/2026, lastro-rust-exec)

Escopo: descobrir onde vai o tempo do prefill (Lastro 17 a 45x atrás do llama.cpp no degrau 1; 2,8 a 17x no degrau 2). **Nenhuma linha de código muda**: binário `primeira-palavra` e `llama-server` como estão, mesmos modelos BF16. 4090 só, sem 5090, GPU livre (voxtype parado, Chrome sem GPU), janela única ≤ 20 min, sem repetição automática. Resultado descritivo, sem meta nem PASSA.

## Parte A: leitura dos dados que já existem (sem GPU)
Declaração honesta: o quadro abaixo foi calculado com as 6 janelas já feitas antes de escrever este critério. Por isso é **exploratório**; a parte B é a confirmação.
Prefill quente (mediana), ms por posição do prompt (89, 170 e 364 posições):
- 1B Lastro L10 8,9 / 9,0 / 8,5; L100 1,58 / 1,58 / 1,58; camadas-CPU 8,8 / 8,8 / 8,7.
- 3B Lastro L10 17,5 / 17,7 / 17,4; L100 2,52 / 2,54 / 2,55; camadas-CPU 17,5 / 17,4 / 17,2.
- llama.cpp S1: 7 a 13 ms (1B) e 13 a 29 ms (3B) no total (quase constante, nada por posição). S2: ~225 a 243 ms (1B) e ~550 a 590 ms (3B) no total, quase independentes do tamanho do prompt.
Leitura exploratória: o prefill do Lastro é **linear nas posições, com custo por posição constante** e próximo ao de um passo de decode; o do llama.cpp é quase **constante no tamanho do prompt**. Hipótese H1: o Lastro processa o prompt posição por posição (sem lote entre posições) e o llama.cpp em lote.

## Parte B: confirmação por medição (1 janela, comando existente `primeira-palavra --gera`, `--novos 1`)
Prompts de comprimentos novos (não usados antes), feitos cortando o texto longo nas posições N = 32, 128, 256, 512 (tokenizados pelo tokenizador do próprio Lastro; N real é impresso). 1B e 3B, L100 e camadas-CPU (N=22 no 1B, 29 no 3B). Em paralelo, o `llama-server` S1 e S2 nos mesmos prompts (`n_predict=1`, `cache_prompt=false`). 1 aquecimento, 5 medidas quentes por célula; mediana.
Previsões pré-registradas, a conferir contra a medição (nenhuma é meta):
- P1: no Lastro, prefill_ms / N varia no máximo ±15% entre os 4 comprimentos em cada série (H1 sustentada). Se crescer com N, há componente superlinear (atenção).
- P2: no llama.cpp S1, prefill_ms cresce menos de 3x de N=32 para N=512 (lote amortiza). Idem S2, onde o custo fixo da CPU domina.
- P3: o custo por posição do Lastro L100 fica dentro de ±30% do custo de um passo de decode L100 do mesmo modelo (medianas por passo já medidas nas janelas do degrau 1), ou seja, prefill ≈ N × passo.
- P4: no Lastro, com L100 a utilização da GPU (amostrada pelo `nvidia-smi dmon`, 100 ms) fica abaixo de 60% durante o prefill; com camadas-CPU a CPU (soma de `%CPU` das threads do processo, via `/proc`) fica acima de 400%. Valores só reportados; leitura é de dado, não prova de causa.
Validade como no critério base (integridade FIXA, preflight, portão de GPU, ids do prefill conferidos contra a referência 5c apenas nos prompts originais, que não entram aqui).
Limite declarado: não separa atenção, experts, cópia e sincronização dentro do prefill do Lastro (isso exigiria instrumento dentro do motor, outra unidade); P4 só dá a dica entre GPU e CPU.
Saída: tabela N × série (prefill_ms, ms/pos), razão contra o llama.cpp, previsões P1 a P4 com o dado ao lado, e a lista de próximos passos que cada resultado sugere.
