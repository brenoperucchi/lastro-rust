# Critério pré-registrado: degrau 2, Lastro com experts por camada inteira contra llama.cpp S2 (09/11/2026, lastro-rust-exec)

Escrito ANTES de qualquer dado do degrau 2. Herda do critério do degrau 1 (`criterio-degrau1-100pct-20261109.md`, sha f7b5d7eb…, com a Emenda 1) e, por ele, do critério base `criterio-comparacao-llamacpp-20261108.md` (emendas 1 a 9, sha f1bd1e0f…) tudo o que não for dito aqui: textos, 64 tokens, frio e quente, validade, SEM PROVA, portão de GPU/CPU, monitor, integridade, janela ≤30 min por modelo, limite, trava, sem 5090, sem push, e as regras de memória, cobertura e cont.needed/cont.cpu da Emenda 1 do degrau 1 (exceto onde este arquivo diz outra coisa). Onde este arquivo conflita com os anteriores, vale este, só para o degrau 2.

## 1. Pergunta
Com os experts posicionados por CAMADA INTEIRA, como no llama.cpp (as primeiras N camadas com os experts só na CPU, as demais com todos os experts na GPU), qual a razão Lastro/llama.cpp S2 (`--n-cpu-moe N`, `-ngl 99`, `-fa on`)? Compara os dois motores com o mesmo desenho de descarregamento, que o degrau 1 e a janela de 10% não tinham.

## 2. Mudanças de procedimento (únicas)
1. Lastro roda com `--camadas-cpu N` (flag nova do `--gera`, exclusiva com `--fracao-slots`, N de 0 a camadas-1; padrão do binário inalterado): os experts das N primeiras camadas ficam SÓ na CPU e os das camadas N em diante ficam todos residentes na GPU (slots = (camadas - N) x experts; prefill da tabela = as chaves camada*E+expert das camadas N em diante; backlog 0). N é FIXO por modelo, igual ao `--n-cpu-moe` da S2: **22 no 1B (24 camadas, 32 experts, 64 slots de 768) e 29 no 3B (32 camadas, 40 experts, 120 slots de 1280)**.
2. Séries: **LC** (Lastro `--camadas-cpu N`) contra **S2**, ambas medidas NA MESMA JANELA (a S2 é remedida; nenhum número de janela anterior entra em razão nova). S1 e S3 não rodam e não aparecem como NÃO RODOU. Séries L (10%) e L100 das janelas anteriores só entram como referência declarada, nunca em razão nova.
3. Células: os mesmos 3 textos x frio/quente x {LC, S2}, por modelo (1B e 3B), 64 tokens, 3 frios e 5 quentes, ordem intercalada como no base, uma janela por modelo. Orçamento: só os cortes (3) quente 5->3 e (4) frio 3->2; o que não couber é NÃO RODOU.
4. Os roteiros (`roda-cmp-lastro.sh`, `roda-cmp-janela.sh`, `cmp-extrai.py`) ganham só o repasse de N (CMP_CAMADAS_CPU, valor fixo por modelo, recusada se difere de 22/29 ou se vem com CMP_FRACAO_SLOTS), a série LC e o modo `--degrau2`. O modo antigo e o degrau 1 não mudam. CMP_CAMADAS_CPU no ambiente de quem chama o modo real da janela é recusada (variável só de teste); a janela a define por dentro. O FIXO e os shas são recalculados no mesmo passo (inclui este arquivo) e conferidos por `--verifica-fixos` antes da janela.
5. Registrar em cada célula a linha de comando gravada (`--camadas-cpu N`) e a linha de formato fixo que o binário imprime no stderr, uma vez por processo: `camadas-cpu N de L; slots S de T` (ex.: `camadas-cpu 22 de 24; slots 64 de 768`).

## 3. Smoke (antes das medidas, dentro da janela de cada modelo)
Uma geração curta com `--camadas-cpu N`, texto curto, 64 tokens, sem medida. Condições para seguir às medidas do modelo (regras de leitura de memória da Emenda 1 do degrau 1: leituras carimbadas dentro do intervalo do binário, mínimo 3, lacunas e bordas com tolerância de 3 s, limite sobre o máximo dessas leituras, amostra pós-saída só descritiva, sem cobertura = falha fechada):
- O processo termina com código 0, sem erro de alocação de memória da GPU, e a memória usada pela GPU, medida com o motor vivo, não passa de 90% da 4090.
- A linha `camadas-cpu N de L; slots S de T` existe e traz exatamente N = 22 (1B) ou 29 (3B), L = 24 ou 32, S = 64 ou 120 e T = 768 ou 1280; a linha de comando do smoke tem `--camadas-cpu N` e não tem `--fracao-slots`.
- `cont.needed` e `cont.cpu` existem, numéricos, finitos e não negativos, com needed > 0 (ausente ou null nunca vira zero). **NÃO se exige cpu == 0**: o esperado é cpu/needed ALTO (a maior parte dos experts pedidos está nas camadas só na CPU); a fração é reportada.
- ids gerados contra a referência 5c: só reportados; divergência não para a janela e a célula leva a marca "sequências diferentes"; nada é ajustado.
Falha do smoke em qualquer item (exceto ids) = a janela do modelo para, nenhuma razão é publicada, resultado "SMOKE DO DEGRAU 2 FALHOU: <motivo>". Sem repetição automática.

## 4. Saída
Descritiva, sem meta, sem PASSA/REPROVA de velocidade. Por célula: razão de prefill e de decode (mediana LC / mediana S2), com min a max, marcas (ruidosa, sequências diferentes) e cpu/needed do Lastro. Mesmas regras de dispersão (>15% = ruidosa) e de SEM PROVA do base. Todo rótulo, com o N do modelo: "Lastro camadas inteiras (N na CPU) contra llama.cpp S2 (--n-cpu-moe N); KV f32 no Lastro contra f16 no llama.cpp; op offload ligado no llama.cpp, prefill da S2 não é leitura de experts na CPU".

## 5. Leitura (pré-registrada como orientação, não como resultado)
Palpites declarados antes dos dados, que NÃO viram critério de aprovação, para a razão de decode LC/S2:
- Até 1,25 (inclusive): equivale.
- Acima de 1,25 até 2 (inclusive): diferença moderada.
- Acima de 2: diferença grande.
O prefill é reportado separado e não entra nas faixas. A leitura final cita as faixas e diz onde o dado caiu; a faixa não é decisão.

## 6. Limites declarados
KV f32 contra f16; fronteira do prefill diferente (base); o prefill da S2 tem op offload ligado e não é leitura de experts na CPU; o posicionamento é parecido mas não idêntico (o Lastro serve na CPU os experts pedidos das N primeiras camadas pelo caminho próprio dele); desktop gráfico como condição comum; memória do host declarada à parte; um Granite BF16 só; sem Strata; não responde sobre outros modelos ou quantizados.

## 7. Rito
Revisão dos roteiros alterados e deste critério ANTES da janela (nível definido pelo Breno); janela com Chrome fechado, voxtype parado e portão de GPU do base; emendas só com nota datada antes dos dados; commits locais, sem push.

## Emenda 1 (09/11/2026, antes de qualquer dado; r118 rev-2 P2)
Seção 3, item de `cont.needed` e `cont.cpu`: vale onde conflita com o texto acima (a frase "NÃO se exige cpu == 0 ... a fração é reportada" fica substituída por esta). Por medida, no smoke e em cada célula LC, com inteiros e (L, N) do modelo (1B: L = 24, N = 22; 3B: L = 32, N = 29): `cpu × L == needed × N`, isto é, cpu/needed = N/L exato. Diferente disso, o posicionamento por camada não está provado: o smoke falha ("SMOKE DO DEGRAU 2 FALHOU: cpu/needed = x, esperado N/L exato: o posicionamento por camada não é o declarado") e a célula LC é SEM PROVA. A fração continua sendo reportada.

## Emenda (10/11, decisão do Breno): placa de medida 3080 Ti

Onde este critério diz "90% da 4090", vale 90% da memória da placa de medida, a RTX 3080 Ti. A série é nova, só da 3080 Ti, e não se compara com números da 4090 (ver a Emenda 3 de `criterio-comparacao-llamacpp-20261108.md`).
