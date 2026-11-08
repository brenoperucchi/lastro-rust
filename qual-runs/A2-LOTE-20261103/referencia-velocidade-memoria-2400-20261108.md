# Mudança de referência de velocidade: memória a 2400 MT/s (08/11/2026)

Escrito pelo `lastro-rust-exec` ANTES de qualquer medida nova. Decisão do Breno (via coordenador, 08/11 17:56).

## Motivo
- 07/11 22:14: o mesmo binário da etapa 4 (sha256 1fa13846a20b1bc3cccf99f0fcd0a1745654615a32979e00f31c65010f35aa2b), mesmos ids, ficou 20% a 21% mais lento (1B C 4,18 s contra 3,45 s; 1B A 16,27 s contra 13,59 s).
- 08/11 17:55 (`dmidecode`, coordenador): 4 pentes DDR4-3200, Configured Memory Speed 2400 MT/s, XMP desligado.
- Breno: com XMP ligado a máquina ficou muito instável. A base passa a ser a memória a 2400 MT/s, sem data para religar o XMP.

## O que NÃO muda
- As metas e rótulos antigos da etapa 4 (medidos às 02:00 de 07/11, memória em XMP) ficam como estão, válidos para aquele hardware. Nada foi apagado nem reescrito.
- Critérios congelados, binários, ids, regras de integridade: os mesmos. O binário continua rotulando contra as metas antigas; os rótulos novos abaixo são aplicados por mim sobre as medianas, em arquivo à parte.

## Regra das metas novas (fixada antes dos dados)
Meta nova = meta antiga x 1,2, arredondada para cima ao inteiro (a lentidão medida foi 1,20 a 1,21 no mesmo binário). A mesma regra vale para quente e frio e para as metas do 5b.

| Célula | Quente antiga | Quente nova | Fria antiga | Fria nova |
|---|---|---|---|---|
| 1B C | 5 s | 6 s | 20 s | 24 s |
| 1B A | 15 s | 18 s | 30 s | 36 s |
| 3B C | 10 s | 12 s | 30 s | 36 s |
| 3B A | 30 s | 36 s | 45 s | 54 s |

Rótulo novo (ATINGE / NÃO ATINGE / SEM PROVA): mediana medida hoje, na mesma regra do critério (quente: ganho >= 15% e mediana do modo 8 <= meta; frio: mediana das 3 repetições <= meta), só com a meta nova. Integridade quebrada ou célula SEM PROVA derruba o rótulo.

## Medidas desta referência
Duas janelas na 4090 (GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd), autorização permanente, 30 min cada, sem repetição automática: quente (1B e 3B) e fria (12 processos), com o binário da etapa 4. Os números de 07/11 22:14 (1B) servem só de comparação.

## Pendência registrada
"Religar XMP 3200 e remedir", gatilho = Breno avisar que religou. Ver `/home/brenoperucchi/Devs/lastro-rust/PENDENCIAS.md`.
