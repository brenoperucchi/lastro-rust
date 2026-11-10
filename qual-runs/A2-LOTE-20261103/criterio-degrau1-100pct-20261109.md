# Critério pré-registrado: degrau 1, Lastro com 100% dos experts na GPU contra llama.cpp S1 (09/11/2026, lastro-rust-exec)

Escrito ANTES de qualquer dado do degrau 1. Herda do critério base `criterio-comparacao-llamacpp-20261108.md` (emendas 1 a 9, sha f1bd1e0f…) tudo o que não for dito aqui: textos, 64 tokens, frio e quente, séries do llama.cpp, validade, SEM PROVA, portão de GPU/CPU, monitor, integridade, janela ≤30 min por modelo, limite, trava, sem 5090, sem push. Onde este arquivo conflita com o base, vale este, só para o degrau 1.

## 1. Pergunta
Com todos os experts residentes na GPU (`--fracao-slots 1.0`, slots = camadas x experts), qual a razão Lastro/llama.cpp S1 (`-ngl 99`, `-fa on`)? Isola, tanto quanto dá hoje, a diferença de motor, sem o efeito de ~89% dos experts na CPU (que explicou o 12x da janela 1B/3B de 09/11).

## 2. Mudanças de procedimento (únicas)
1. Lastro roda com `--fracao-slots 1.0` (flag do commit c56aa6c, padrão 0,10 inalterado). Registrar em cada célula a linha de comando gravada e `slots` impressos.
2. Séries: **L100** (Lastro 100%) contra **S1**, ambas medidas NA MESMA JANELA (o S1 é remedido; não usa números da janela anterior). S2 e S3 não rodam. A coluna "Lastro 10%" da janela anterior só entra no relatório como referência declarada, nunca em razão nova misturada.
3. Células: 3 textos x frio/quente x {L100, S1}, por modelo (1B e 3B), com 3 frios e 5 quentes; ordem intercalada como no base. Uma janela por modelo.
4. Os roteiros (`roda-cmp-lastro.sh`, `roda-cmp-janela.sh`, extrator) ganham só o repasse da fração e a série L100; o FIXO e os shas são recalculados no mesmo passo e conferidos por `--verifica-fixos` antes da janela.

## 3. Smoke (antes das medidas, dentro da janela de cada modelo)
Uma geração curta com `--fracao-slots 1.0`, texto curto, 64 tokens, sem medida. Condições para seguir às medidas do modelo:
- O processo termina com código 0, sem erro de alocação de memória da GPU, e a memória usada pela GPU não passa de 90% da 4090.
- cpu/needed do motor = 0 (nenhum expert servido pela CPU); se não for 0, o 100% não é 100% e a série não vale como L100.
- ids gerados idênticos à referência 5c (igualdade par a par). Divergência: reportada, a janela do modelo segue, e a célula leva a marca "sequências diferentes"; nada é ajustado.
Falha do smoke em qualquer item (exceto ids) = a janela do modelo para, nenhuma razão é publicada, resultado "L100 NÃO SUSTENTA 100%: <motivo>". Sem repetição automática.

## 4. Saída
Descritiva, sem meta, sem PASSA/REPROVA de velocidade. Por célula: razão de prefill e de decode (mediana L100 / mediana S1), com min a max, marcas (ruidosa, sequências diferentes) e frações cpu/needed. Mesmas regras de dispersão (>15% = ruidosa) e de SEM PROVA do base. Todo rótulo: "Lastro 100% experts GPU contra llama.cpp S1; KV f32 no Lastro contra f16 no llama.cpp".

## 5. Leitura (pré-registrada como orientação, não como resultado)
Palpites declarados, escritos antes dos dados e que NÃO viram critério de aprovação:
- Se o decode L100/S1 ≤ 2: o motor é competitivo com a GPU a 100%, e o próximo passo é o degrau 2 (camadas inteiras) só se ainda se quiser igualar S2.
- Se entre 2 e 6: há overhead de motor moderado (KV f32, sincronização, lançamentos); investigar por perfil.
- Se > 6: o motor tem custo próprio grande além do posicionamento; investigar antes de qualquer outro degrau.
- Prefill: a razão será provavelmente maior que a do decode; o relatório separa as duas e não as junta.
A leitura final cita as faixas e diz onde o dado caiu; a faixa não é decisão.

## 6. Limites declarados
KV f32 contra f16; fronteira do prefill diferente (base); desktop gráfico como condição comum; memória do host declarada à parte; um Granite BF16 só; sem Strata; não responde sobre outros modelos ou quantizados.

## 7. Rito
Revisão nível 2 deste critério e dos roteiros alterados ANTES da janela; janela com Chrome fechado e voxtype parado e portão de GPU do base; emendas só com nota datada antes dos dados; commits locais, sem push.

## Emenda 1 (09/11/2026, antes de qualquer dado; decisões de implementação das revisões r116 e r117)
Só aumenta o rigor; vale onde conflita com o texto acima.
1. O binário imprime no stderr, uma vez por processo, `slots N de M`. A linha é OBRIGATÓRIA no smoke e em cada processo L100, com N = M > 0; ausente ou diferente = smoke falha / célula SEM PROVA.
2. Memória do smoke: só valem leituras com carimbo de tempo DENTRO do intervalo em que o binário rodou; mínimo de 3 leituras, lacunas e início/fim cobertos com tolerância de 3 s; limite de 90% sobre o máximo dessas leituras. A amostra pós-saída é só descritiva e nunca aprova. Sem cobertura = falha fechada.
3. `cont.needed` e `cont.cpu` têm de existir, ser numéricos, finitos e não negativos; ausente ou null NUNCA vira zero (smoke falha, célula SEM PROVA). L100 exige needed > 0 e cpu explicitamente 0.
4. Se o smoke real terminar sem o mínimo de leituras (geração muito curta), vale a falha fechada: a janela do modelo para e o resultado é "L100 NÃO SUSTENTA 100%: leituras insuficientes". Sem repetição automática.

## Emenda (10/11, decisão do Breno): placa de medida 3080 Ti

Onde este critério diz "90% da 4090", vale 90% da memória da placa de medida, a RTX 3080 Ti. A série é nova, só da 3080 Ti, e não se compara com números da 4090 (ver a Emenda 10 de `criterio-comparacao-llamacpp-20261108.md`).
