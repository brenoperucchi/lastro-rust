#!/usr/bin/env bash
# Comparação Lastro x llama.cpp (opção d), UMA janela de até 30 min POR MODELO na 4090 (criterio-comparacao-llamacpp-20261108.md, emendas 1 a 6; a mais nova vale sobre a anterior).
# Emenda 6: duas janelas, 1B depois 3B, mesmo critério e mesmos roteiros; --modelos 1b|3b é OBRIGATÓRIO (sem ele, recusa: nunca roda as duas de uma vez por engano).
# Composição já validada nas janelas 5c/5e: janela-e4b.sh (pausa e restauração conferida dos timers) por fora, roda-com-limite.sh (limite) por dentro, e este mesmo
# script em modo interno.
#
#   roda-cmp-janela.sh --modelos 1b|3b --saida DIR [--estado-janela DIR] [--bin PRIMEIRA_PALAVRA] [--limite S] [--sem-s3]
#   roda-cmp-janela.sh --verifica-fixos     (sem GPU, sem autorização, só lê: confere os shas congelados do FIXO contra os arquivos reais; código 0 se todos OK)
#
# Dentro da janela (do modelo escolhido; o sha de tudo é conferido nas duas): (1) sha256 de tudo que o critério manda conferir (binário do Lastro, binários e bibliotecas do llama.cpp, 2 GGUF, config/tokenizer/safetensors dos
# dois modelos, textos, referências, cmp-ids-prompt.json, critério) ANTES e DEPOIS, contra os valores congelados onde há (critério, ids, textos, referências, tokenizers);
# quebrada = SEM PROVA em tudo; (2) ambiente.txt (memória do host se o sistema deixar ler, sem sudo interativo); (3) as células, em BLOCOS por (modelo, texto), na ordem
# da E8: o modelo da janela (S1, S2) em todos os textos, frio e quente; por último S3 (llama.cpp KV f32, -fa off) só se sobrar; lados ALTERNADOS (blocos pares começam pelo
# Lastro, ímpares pelo llama.cpp; dentro do bloco, frio e depois quente); ordem gravada em ordem.txt; (4) cmp-extrai.py (cmp-resultado.json/.md) e decisao.txt.
# Cada célula é um processo (roda-cmp-lastro.sh ou roda-cmp-llamacpp.sh) com pasta NOVA em DIR/celulas/<lado>-<modelo>-<serie>-<texto>-<estado>; sem repetição
# automática (célula com erro fica SEM PROVA e a janela segue; recusa de ambiente, código 2, interrompe tudo). A célula SEM portão (CPU/GPU) NÃO é adiada.
#
# ORÇAMENTO (E8, estimativa; emenda 6 recalibra o lado Lastro e a carga com a janela 5c e declara que o lado llama.cpp segue palpite): nominal quente = 1 aquecimento + 5 medidas, frio = 1 processo descartado + 3 medidas. ANTES de iniciar cada bloco o relógio projeta:
# decorrido + estimativa do que falta + folga (180 s para encerramento, sha final e restauração dos timers) contra o limite. Se estoura, cortes em ordem, um de cada vez,
# reprojetando: (1) S3 inteira não roda; (2) S2 do 3B; depois S2 do 1B; (3) quente 5 -> 3 medidas; (4) frio 3 -> 2 medidas; (5) o que ainda não couber é NÃO RODOU
# (nunca vira razão, PASSA nem estimativa). Os blocos da S3 usam as mesmas repetições do Lastro correspondente e só rodam se couberem. Cortes só mudam repetições e
# quais células rodam. orcamento.json e decisao.txt declaram o que foi cortado. As estimativas (CMP_EST_*, em segundos) são PALPITES de ordem de grandeza.
#
# Falha FECHADA antes de tocar em timers ou na GPU (emenda 9: também se houver PID C ou C+G na 4090; durante a janela o voxtype e o processo de GPU do Chrome ficam FECHADOS): PRIMEIRA_PALAVRA_AUTORIZADA=1 e CUDA_VISIBLE_DEVICES igual ao UUID da 4090 (a 5090 NUNCA entra, nem no llama-server)
# têm de estar no ambiente de quem chama (o "autorizo" do Breno); --saida e --estado-janela têm de ser pastas novas ou vazias. Rede só em 127.0.0.1. Limite padrão 1800 s
# (máx. 1800); o de fora é o de dentro mais 120 s. Fim da janela: aviso (notify-send se houver, ou CMP_AVISO_CMD) e DIR.aviso.txt. Este roteiro NÃO é rodado na entrega.
# Variáveis só para teste (r111, rev-2): JANELA_SH, LIMITE_SH, JANELA_TIMERS, LASTRO_SH_CMP, LLAMA_SH_CMP, EXTRAI_PY, CMP_RELOGIO_ARQ (relógio virtual em segundos), CMP_FOLGA,
# CMP_EST_*, CMP_FAKE_*, MODELO_1B, MODELO_3B, GGUF_DIR, LLAMA_DIR, TEXTOS_5C, REF_5C, IDS_CMP, CRITERIO_CMP, SEM_SHA_FIXO=1, NVIDIA_SMI, CMP_SEM_MEMORIA=1, CMP_TESTE_AMBOS, PROC_STAT,
# CMP_CPU_JANELA, CMP_CPU_ESPERA, LLAMA_SERVER, CURL, MONITOR_SH, ESPERA_PRONTO, LOADAVG_ARQ, SYSTEMCTL, SYSTEMD_RUN, PENDENTE, JANELA_*, LIMITE_*, CGROUP_RAIZ. No modo real (envelope
# externo) qualquer uma presente no ambiente é RECUSADA com código 2; só CMP_MODO_TESTE=1 (modo de teste explícito) as aceita. CMP_AVISO_CMD continua permitida (só avisa).
# Códigos: 0 se decisao.txt foi gravada, o do comando (124 limite, ...) ou 2 recusa de uso, portão ou ambiente.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
UUID=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd
JANELA=${JANELA_SH:-$AQUI/janela-e4b.sh}
LIMITE_SH=${LIMITE_SH:-$AQUI/roda-com-limite.sh}
LASTRO_SH=${LASTRO_SH_CMP:-$AQUI/roda-cmp-lastro.sh}
LLAMA_SH=${LLAMA_SH_CMP:-$AQUI/roda-cmp-llamacpp.sh}
EXTRAI=${EXTRAI_PY:-$AQUI/cmp-extrai.py}
BIN=$AQUI/../../rust/moe-rt/target/release/primeira-palavra
M1=${MODELO_1B:-/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct}
M3=${MODELO_3B:-/home/brenoperucchi/models/granite-3.1-3b-a800m-instruct}
GGUFS=${GGUF_DIR:-/home/brenoperucchi/models/gguf}
LLAMA_DIR=${LLAMA_DIR:-/home/brenoperucchi/Devs/GitHub/llama.cpp/build/bin}
TEXTOS=${TEXTOS_5C:-$AQUI/5c-textos}
REFS=${REF_5C:-$AQUI/5c-referencia}
IDS=${IDS_CMP:-$AQUI/cmp-ids-prompt.json}
CRIT=${CRITERIO_CMP:-$AQUI/criterio-comparacao-llamacpp-20261108.md}
FOLGA=${CMP_FOLGA:-180}
SAIDA=; ESTADO=; LIMITE=1800; INTERNO=0; COM_S3=1; MODELOS=; VERIFICA=0
usa() { echo "uso: roda-cmp-janela.sh --modelos 1b|3b --saida DIR [--estado-janela DIR] [--bin PRIMEIRA_PALAVRA] [--limite S] [--sem-s3] | --verifica-fixos" >&2; }
recusa() { echo "roda-cmp-janela: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
  case $1 in
    --saida) [ $# -ge 2 ] || { usa; exit 2; }; SAIDA=$2; shift 2;;
    --estado-janela) [ $# -ge 2 ] || { usa; exit 2; }; ESTADO=$2; shift 2;;
    --bin) [ $# -ge 2 ] || { usa; exit 2; }; BIN=$2; shift 2;;
    --limite) [ $# -ge 2 ] || { usa; exit 2; }; LIMITE=$2; shift 2;;
    --sem-s3) COM_S3=0; shift;;
    --modelos) [ $# -ge 2 ] || { usa; exit 2; }; MODELOS=$2; shift 2;;
    --verifica-fixos) VERIFICA=1; shift;;
    --interno) INTERNO=1; shift;;
    *) usa; exit 2;;
  esac
done
sha() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }

# SHAs congelados (valem onde o critério já os fixa; binários, GGUF e safetensors só precisam ser iguais antes e depois).
declare -A FIXO=(
  [criterio-cmp]=f1bd1e0f35378f5efe16ea156546d8b96370b754f986c4dca56e2ecd191a5932
  [cmp-ids-prompt]=2a2613cdbaf9acdb76bd9aa416107474b3d888b6aa37b94b429546a5dca78d8b
  [texto-curto]=5b6fad6f3825d430bad4f924d227105032581906739d68c726ac8e4827df2d4d
  [texto-medio]=bb0a067ae8839616f33b21ba41b362bdf16353a57d70321252a2d47851f1c32a
  [texto-longo]=821e0f15139b58173b5c17f5e9428527bb27b03dc10b35e38cbeed044defbaea
  [textos-json]=5b16e34095e882b517a5b9dabb22e956e7d4cf688c0d2a6c4caf9c61abf99e5e
  [ref5c-1b]=9584fcabe7d8b2a1821207267651504c72d317ce9b566cd967d390a1dbbad259
  [ref5c-3b]=968d9b391324a99fbaf5c772a351018b90105f572b32ee2db3140f79da123124
  [tokenizer-1b]=ed944c0b3d71e2d651a5ba0f8e7b350faff30031c8e3353de70b12b9c1bf5654
  [tokenizer-3b]=ed944c0b3d71e2d651a5ba0f8e7b350faff30031c8e3353de70b12b9c1bf5654
)
CHAVES=(); CAMINHOS=()
monta_lista() {
  CHAVES=(); CAMINHOS=()
  add() { CHAVES+=("$1"); CAMINHOS+=("$2"); }
  add binario-lastro "$BIN"
  add llama-server "$LLAMA_DIR/llama-server"
  local f
  while IFS= read -r f; do [ -n "$f" ] && add "llama-lib-$(basename "$f")" "$f"; done < <(find "$LLAMA_DIR" -maxdepth 1 -type f -name 'lib*.so*' 2>/dev/null | sort)
  add gguf-1b "$GGUFS/granite-3.1-1b-a400m-instruct-bf16.gguf"
  add gguf-3b "$GGUFS/granite-3.1-3b-a800m-instruct-bf16.gguf"
  add config-1b "$M1/config.json"; add tokenizer-1b "$M1/tokenizer.json"; add safetensors-1b "$M1/model.safetensors"
  add config-3b "$M3/config.json"; add tokenizer-3b "$M3/tokenizer.json"
  add safetensors-3b-1 "$M3/model-00001-of-00002.safetensors"; add safetensors-3b-2 "$M3/model-00002-of-00002.safetensors"; add indice-3b "$M3/model.safetensors.index.json"
  add texto-curto "$TEXTOS/texto-curto.txt"; add texto-medio "$TEXTOS/texto-medio.txt"; add texto-longo "$TEXTOS/texto-longo.txt"; add textos-json "$TEXTOS/textos.json"
  add ref5c-1b "$REFS/ref5c-1b.json"; add ref5c-3b "$REFS/ref5c-3b.json"
  add cmp-ids-prompt "$IDS"; add criterio-cmp "$CRIT"
}
# coleta_shas ARQ: uma linha "chave sha" por item (vazio = ilegível); hashes em paralelo (os arquivos grandes dominam o tempo).
coleta_shas() {
  local tmp="$SAIDA/.shas.$$" i
  mkdir -p "$tmp"
  for i in "${!CHAVES[@]}"; do ( sha "${CAMINHOS[$i]}" > "$tmp/$i" ) & done
  wait
  : > "$1"
  for i in "${!CHAVES[@]}"; do echo "${CHAVES[$i]} $(cat "$tmp/$i" 2>/dev/null)" >> "$1"; rm -f "$tmp/$i"; done
  rmdir "$tmp" 2>/dev/null || true
}

# --verifica-fixos: só lê (sem GPU, sem autorização, sem tocar em timers): sha dos arquivos reais contra os congelados do FIXO. AUSENTE = arquivo fora do repositório que não existe aqui.
if [ $VERIFICA = 1 ]; then
  monta_lista
  rc_v=0
  for i in "${!CHAVES[@]}"; do
    k=${CHAVES[$i]}
    [ -n "${FIXO[$k]:-}" ] || continue
    h=$(sha "${CAMINHOS[$i]}")
    if [ -z "$h" ]; then echo "$k ${FIXO[$k]} - AUSENTE ${CAMINHOS[$i]##*/}"; rc_v=1
    elif [ "$h" = "${FIXO[$k]}" ]; then echo "$k ${FIXO[$k]} $h OK"
    else echo "$k ${FIXO[$k]} $h FALHA"; rc_v=1; fi
  done
  exit $rc_v
fi
# --modelos: uma janela por modelo (emenda 6); ausente ou inválido = recusa 2. 'ambos' só no modo interno de teste (CMP_TESTE_AMBOS=1), para os cenários antigos do orçamento.
case $MODELOS in
  1b|3b) ;;
  ambos) { [ $INTERNO = 1 ] && [ "${CMP_TESTE_AMBOS:-}" = 1 ]; } || recusa "--modelos ambos só existe no modo interno de teste (CMP_TESTE_AMBOS=1)";;
  '') usa; recusa "--modelos 1b|3b é obrigatório: uma janela por modelo (emenda 6)";;
  *) recusa "--modelos inválido: '$MODELOS' (use 1b ou 3b)";;
esac
[ -n "$SAIDA" ] || { usa; exit 2; }

# ---------------------------------------------------------------- modo interno (dentro da janela, com o portão e o UUID no ambiente)
if [ $INTERNO = 1 ]; then
  [ "${PRIMEIRA_PALAVRA_AUTORIZADA:-}" = 1 ] && [ "${CUDA_VISIBLE_DEVICES:-}" = "$UUID" ] || recusa "modo interno sem portão ou UUID"
  mkdir -p "$SAIDA/celulas" || exit 2
  agora_s() { if [ -n "${CMP_RELOGIO_ARQ:-}" ]; then cat "$CMP_RELOGIO_ARQ"; else date +%s; fi; }
  INICIO=$(agora_s)
  decorrido() { echo $(( $(agora_s) - INICIO )); }
  monta_lista
  coleta_shas "$SAIDA/integridade-antes.txt"
  # ambiente (descritivo): GPU, memória do host (só sem pedir senha), carga
  {
    echo "inicio=$(date -Is)"
    echo "uuid=$UUID"
    echo "limite_s=$LIMITE folga_s=$FOLGA"
    "${NVIDIA_SMI:-nvidia-smi}" --query-gpu=name,driver_version,clocks.max.memory --format=csv,noheader 2>&1 | head -n 2 | sed 's/^/gpu=/'
    echo "loadavg=$(cut -d' ' -f1-3 < /proc/loadavg 2>/dev/null)"
    if [ "${CMP_SEM_MEMORIA:-0}" != 1 ]; then
      mts=$(sudo -n dmidecode -t memory 2>/dev/null | awk -F': ' '/Configured Memory Speed/ && $2 ~ /[0-9]/ { split($2, a, " "); print a[1]; exit }')
      [ -z "$mts" ] || echo "memoria_mts=$mts"
    fi
  } > "$SAIDA/ambiente.txt" 2>&1

  # ---- plano e orçamento
  est() { local v=CMP_EST_$1; echo "${!v:-$2}"; }   # est NOME PADRAO
  # Emenda 6 (r111): recalibração do lado Lastro com a evidência da janela 5c (08/11, 19:14:28 a 19:15:17, 1B e 3B, 3 pedidos cada): o motor de cada modelo foi criado em no máximo
  # 3,0 s (1B) e 4,2 s (3B) (tempo do bloco menos a soma dos total_ms dos pedidos); totais por pedido de 2,8 a 7,3 s no 1B e de 5,6 a 11,9 s no 3B, já contando o T_sequencia da 5c que a
  # prova --so-texto não roda; prefill de 1,5 a 3,3 s (1B) e de 2,9 a 6,3 s (3B); decode de 21,4 e 40,3 ms por passo (64 passos = 1,4 s e 2,6 s). Valores abaixo = o pior caso visto
  # mais margem. Emenda 7 (r112): a frase "que a prova --so-texto não roda" estava errada (o T_sequencia roda no --so-texto frio); o frio por processo sobe para 8 s (1B) e 12 s (3B), o pior
  # total por pedido visto (7,3 e 11,9 s, que já inclui o T_sequencia). O lado llama.cpp NÃO tem evidência (a 5c é só do Lastro): os palpites antigos ficam, declarados como palpite.
  E_PORTAO=$(est PORTAO 8)
  declare -A E_L_CARGA=([1b]=$(est L_CARGA_1B 5) [3b]=$(est L_CARGA_3B 7))
  declare -A E_L_FRIO=([1b]=$(est L_FRIO_1B 8) [3b]=$(est L_FRIO_3B 12))
  declare -A E_L_QUENTE=([1b]=$(est L_QUENTE_1B 5) [3b]=$(est L_QUENTE_3B 9))
  declare -A E_C_CARGA=([1b]=$(est C_CARGA_1B 8) [3b]=$(est C_CARGA_3B 15))
  declare -A E_C_PED=([1b]=$(est C_PED_1B 2) [3b]=$(est C_PED_3B 3))
  E_C_AQUEC=$(est C_AQUEC 3); E_C_FIXO=$(est C_FIXO 3)
  # custo estimado de uma célula
  est_celula() { # lado serie modelo estado n
    local lado=$1 serie=$2 m=$3 estado=$4 n=$5 mult=1
    if [ "$lado" = lastro ]; then
      if [ "$estado" = frio ]; then echo $(( E_PORTAO + (n + 1) * (E_L_CARGA[$m] + E_L_FRIO[$m]) ))
      else echo $(( E_PORTAO + E_L_CARGA[$m] + (n + 1) * E_L_QUENTE[$m] )); fi
    else
      [ "$serie" = S2 ] && mult=2
      if [ "$estado" = frio ]; then echo $(( E_PORTAO + E_C_FIXO + (n + 1) * (E_C_CARGA[$m] + E_C_PED[$m] * mult) ))
      else echo $(( E_PORTAO + E_C_FIXO + E_C_CARGA[$m] + E_C_AQUEC + (n + 1) * E_C_PED[$m] * mult )); fi
    fi
  }
  REPS_Q=5; REPS_F=3; S3=$COM_S3; S2_1B=1; S2_3B=1; PROX_CORTE=1; CORTES=()
  # blocos: "main:modelo:texto" ou "s3:modelo:texto"
  BLOCOS=()
  case $MODELOS in 1b) LISTA_M=(1b);; 3b) LISTA_M=(3b);; *) LISTA_M=(1b 3b);; esac
  for m in "${LISTA_M[@]}"; do for t in curto medio longo; do BLOCOS+=("main:$m:$t"); done; done
  for m in "${LISTA_M[@]}"; do for t in curto medio longo; do BLOCOS+=("s3:$m:$t"); done; done
  est_bloco() { # tipo modelo texto -> segundos com a configuração ATUAL
    local tipo=$1 m=$2 tot=0 s
    if [ "$tipo" = main ]; then
      tot=$(( $(est_celula lastro L "$m" frio "$REPS_F") + $(est_celula lastro L "$m" quente "$REPS_Q") ))
      for s in S1 S2; do
        [ "$s" = S2 ] && { [ "$m" = 1b ] && [ $S2_1B = 0 ] && continue; [ "$m" = 3b ] && [ $S2_3B = 0 ] && continue; }
        tot=$(( tot + $(est_celula llamacpp "$s" "$m" frio "$REPS_F") + $(est_celula llamacpp "$s" "$m" quente "$REPS_Q") ))
      done
    else
      [ $S3 = 1 ] || { echo 0; return; }
      tot=$(( $(est_celula llamacpp S3 "$m" frio "$REPS_F") + $(est_celula llamacpp S3 "$m" quente "$REPS_Q") ))
    fi
    echo "$tot"
  }
  projeta_restante() { # a partir do índice $1 (inclusive)
    local i tot=0 b tipo m t
    for ((i = $1; i < ${#BLOCOS[@]}; i++)); do IFS=: read -r tipo m t <<< "${BLOCOS[$i]}"; tot=$(( tot + $(est_bloco "$tipo" "$m" "$t") )); done
    echo "$tot"
  }
  aplica_corte() { # aplica o PRÓXIMO corte da ordem E8; retorna 1 quando não há mais
    # numa janela de um modelo só, o corte da S2 do OUTRO modelo não existe: pula sem registrar
    while :; do
      if [ $PROX_CORTE = 2 ] && [ "$MODELOS" = 1b ]; then PROX_CORTE=3; continue; fi
      if [ $PROX_CORTE = 3 ] && [ "$MODELOS" = 3b ]; then PROX_CORTE=4; continue; fi
      break
    done
    case $PROX_CORTE in
      1) S3=0; CORTES+=("(1) S3 inteira não roda");;
      2) S2_3B=0; CORTES+=("(2) S2 do 3B não roda");;
      3) S2_1B=0; CORTES+=("(2b) S2 do 1B não roda (nos blocos ainda não iniciados)");;
      4) REPS_Q=3; CORTES+=("(3) quente cai de 5 para 3 medidas (nos blocos ainda não iniciados)");;
      5) REPS_F=2; CORTES+=("(4) frio cai de 3 para 2 medidas (nos blocos ainda não iniciados)");;
      *) return 1;;
    esac
    PROX_CORTE=$((PROX_CORTE + 1))
    echo "corte em $(decorrido) s: ${CORTES[-1]}" >> "$SAIDA/orcamento-log.txt"
    return 0
  }
  : > "$SAIDA/ordem.txt"; : > "$SAIDA/orcamento-log.txt"; : > "$SAIDA/execucoes.txt"
  SEQ=0; ABORTA=0; NAO_RODOU=(); declare -A LREPS=()
  roda_celula() { # lado serie modelo texto estado n
    local lado=$1 serie=$2 m=$3 t=$4 estado=$5 n=$6 chave rc t0
    [ $ABORTA = 0 ] || return 0   # janela interrompida por recusa de ambiente: nada mais roda
    chave="$lado-$m-$serie-$t-$estado"
    SEQ=$((SEQ + 1)); t0=$(decorrido)
    if [ "$lado" = lastro ]; then
      "$LASTRO_SH" --modelo "$m" --texto "$t" --estado "$estado" --n "$n" --saida "$SAIDA/celulas/$chave" --bin "$BIN" > "$SAIDA/celulas-$chave.log" 2>&1; rc=$?
      LREPS["$m-$t-$estado"]=$n
    else
      "$LLAMA_SH" --modelo "$m" --texto "$t" --estado "$estado" --serie "$serie" --n "$n" --ids "$IDS" --saida "$SAIDA/celulas/$chave" > "$SAIDA/celulas-$chave.log" 2>&1; rc=$?
    fi
    printf '%02d %s n=%s inicio=%ss fim=%ss rc=%s\n' "$SEQ" "$chave" "$n" "$t0" "$(decorrido)" "$rc" >> "$SAIDA/ordem.txt"
    [ "$rc" != 2 ] || { ABORTA=1; echo "célula $chave recusou o ambiente (código 2): janela interrompida" >> "$SAIDA/orcamento-log.txt"; }
  }
  nr() { NAO_RODOU+=("$1"); echo "NÃO RODOU: $1" >> "$SAIDA/orcamento-log.txt"; }
  for ((bi = 0; bi < ${#BLOCOS[@]}; bi++)); do
    [ $ABORTA = 0 ] || break
    IFS=: read -r tipo m t <<< "${BLOCOS[$bi]}"
    if [ "$tipo" = main ]; then
      # corta em ordem enquanto a projeção estourar (antes de iniciar o bloco)
      while [ $(( $(decorrido) + $(projeta_restante "$bi") + FOLGA )) -gt "$LIMITE" ]; do aplica_corte || break; done
      custo=$(est_bloco main "$m" "$t")
      if [ $(( $(decorrido) + custo + FOLGA )) -gt "$LIMITE" ]; then nr "bloco $m/$t (todas as séries, frio e quente): não cabe, sem mais cortes"; continue; fi
      nq=$REPS_Q; nf=$REPS_F
      # lados alternados: bloco de índice par começa pelo Lastro, ímpar pelo llama.cpp; frio antes do quente
      for estado in frio quente; do
        [ $estado = frio ] && n=$nf || n=$nq
        series=(S1); { { [ "$m" = 1b ] && [ $S2_1B = 1 ]; } || { [ "$m" = 3b ] && [ $S2_3B = 1 ]; }; } && series=(S1 S2)
        if [ $((bi % 2)) = 0 ]; then
          roda_celula lastro L "$m" "$t" "$estado" "$n"; for s in "${series[@]}"; do roda_celula llamacpp "$s" "$m" "$t" "$estado" "$n"; done
        else
          for s in "${series[@]}"; do roda_celula llamacpp "$s" "$m" "$t" "$estado" "$n"; done; roda_celula lastro L "$m" "$t" "$estado" "$n"
        fi
        [ $ABORTA = 0 ] || break
      done
      # S2 cortada: registra o que não rodou neste bloco
      for estado in frio quente; do
        if [ "$m" = 1b ] && [ $S2_1B = 0 ]; then nr "llamacpp-$m-S2-$t-$estado"; fi
        if [ "$m" = 3b ] && [ $S2_3B = 0 ]; then nr "llamacpp-$m-S2-$t-$estado"; fi
      done
    else
      # S3: só se sobrar, com as repetições do Lastro correspondente (sem novos cortes)
      nf=${LREPS["$m-$t-frio"]:-}; nq=${LREPS["$m-$t-quente"]:-}
      if [ $S3 = 0 ] || [ -z "$nf" ] || [ -z "$nq" ]; then nr "llamacpp-$m-S3-$t (frio e quente): S3 cortada ou sem célula do Lastro correspondente"; continue; fi
      custo=$(( $(est_celula llamacpp S3 "$m" frio "$nf") + $(est_celula llamacpp S3 "$m" quente "$nq") ))
      if [ $(( $(decorrido) + custo + FOLGA )) -gt "$LIMITE" ]; then nr "llamacpp-$m-S3-$t (frio e quente): não sobrou orçamento"; continue; fi
      roda_celula llamacpp S3 "$m" "$t" frio "$nf"; [ $ABORTA = 0 ] || break
      roda_celula llamacpp S3 "$m" "$t" quente "$nq"
    fi
  done
  [ $ABORTA = 0 ] || nr "janela interrompida por recusa de ambiente (código 2) na célula ${SEQ}"

  # ---- integridade depois
  coleta_shas "$SAIDA/integridade-depois.txt"
  INTEGRO=1; NOTAS=()
  if ! cmp -s "$SAIDA/integridade-antes.txt" "$SAIDA/integridade-depois.txt"; then INTEGRO=0; NOTAS+=("sha mudou entre antes e depois"); fi
  : > "$SAIDA/integridade-fixos.txt"
  while read -r k h; do
    [ -n "$h" ] || { INTEGRO=0; NOTAS+=("$k ilegível"); echo "$k - - FALHA ilegível" >> "$SAIDA/integridade-fixos.txt"; continue; }
    if [ -n "${FIXO[$k]:-}" ]; then
      if [ "${SEM_SHA_FIXO:-0}" = 1 ]; then echo "$k ${FIXO[$k]} $h NAO-CONFERIDO" >> "$SAIDA/integridade-fixos.txt"
      elif [ "$h" = "${FIXO[$k]}" ]; then echo "$k ${FIXO[$k]} $h OK" >> "$SAIDA/integridade-fixos.txt"
      else INTEGRO=0; NOTAS+=("$k com sha $h, não o congelado"); echo "$k ${FIXO[$k]} $h FALHA" >> "$SAIDA/integridade-fixos.txt"; fi
    fi
  done < "$SAIDA/integridade-antes.txt"
  # ---- orçamento declarado
  {
    printf '{"modelos": [%s], "limite_s": %s, "folga_s": %s, "decorrido_s": %s, "cortes": [' "$(printf '"%s", ' "${LISTA_M[@]}" | sed 's/, $//')" "$LIMITE" "$FOLGA" "$(decorrido)"
    sep=; for c in "${CORTES[@]:-}"; do [ -n "$c" ] || continue; printf '%s"%s"' "$sep" "$c"; sep=', '; done
    printf '], "reps": {"quente_nominal": 5, "frio_nominal": 3}, "reps_finais": {"quente": %s, "frio": %s}, "s3_ativa_no_fim": %s, "nao_rodou": [' "$REPS_Q" "$REPS_F" "$([ $S3 = 1 ] && echo true || echo false)"
    sep=; for c in "${NAO_RODOU[@]:-}"; do [ -n "$c" ] || continue; printf '%s"%s"' "$sep" "$(echo "$c" | tr -d '"\\')"; sep=', '; done
    printf '], "estimativas_sao_palpites": true}\n'
  } > "$SAIDA/orcamento.json"
  python3 -I "$EXTRAI" --saida "$SAIDA" --ids "$IDS" --refs "$REFS" --textos "$TEXTOS/textos.json" > "$SAIDA/cmp-extrai.out" 2>&1; RC_EXT=$?
  {
    echo "comparação Lastro x llama.cpp (opção d), janela do modelo ${LISTA_M[*]}, critério sha256 $(sha "$CRIT")"
    echo "resultado DESCRITIVO: sem meta, sem PASSA/REPROVA de velocidade; toda razão é 'Lastro 10% experts GPU contra llama.cpp <série>'"
    if [ $INTEGRO = 1 ]; then echo "integridade: íntegra (binários, GGUF, safetensors, textos, referências, ids e critério iguais antes e depois$([ "${SEM_SHA_FIXO:-0}" = 1 ] && echo '; congelados NÃO conferidos: SEM_SHA_FIXO=1' || echo ' e aos congelados onde há'))"
    else echo "integridade: QUEBRADA: tudo SEM PROVA"; for n in "${NOTAS[@]}"; do echo "  $n"; done; fi
    echo "cortes de orçamento: ${CORTES[*]:-nenhum}"
    echo "repetições finais: quente $REPS_Q, frio $REPS_F; decorrido $(decorrido) s de $LIMITE s"
    echo "não rodou: ${#NAO_RODOU[@]} item(ns); ver orcamento.json"
    echo "extrator: $(cat "$SAIDA/cmp-extrai.out") (código $RC_EXT)"
    echo "células executadas: $SEQ (ordem.txt)"
  } > "$SAIDA/decisao.txt"
  cat "$SAIDA/decisao.txt"
  exit 0
fi

# ---------------------------------------------------------------- modo externo (portões, janela e limite)
case $LIMITE in ''|*[!0-9]*) recusa "limite inválido: $LIMITE";; esac
[ "$LIMITE" -ge 1 ] && [ "$LIMITE" -le 1800 ] || recusa "limite fora de 1..1800 s: $LIMITE"
[ "${PRIMEIRA_PALAVRA_AUTORIZADA:-}" = 1 ] || recusa "portão fechado: sem PRIMEIRA_PALAVRA_AUTORIZADA=1 (o 'autorizo' do Breno que nomeia a janela)"
[ "${CUDA_VISIBLE_DEVICES:-}" = "$UUID" ] || recusa "CUDA_VISIBLE_DEVICES tem de ser $UUID (a 4090 combinada; a 5090 nunca); veio '${CUDA_VISIBLE_DEVICES:-}'"
# variáveis só de teste no ambiente do modo real = recusa 2 (r111, rev-2); só CMP_MODO_TESTE=1 as aceita. `compgen -e` lista só as EXPORTADAS (as variáveis internas do roteiro não contam).
if [ "${CMP_MODO_TESTE:-0}" != 1 ]; then
  for v in $(compgen -e); do
    case $v in
      SEM_SHA_FIXO|LLAMA_SERVER|MONITOR_SH|LASTRO_SH_CMP|LLAMA_SH_CMP|EXTRAI_PY|CMP_RELOGIO_ARQ|NVIDIA_SMI|MODELO_1B|MODELO_3B|GGUF_DIR|LLAMA_DIR|TEXTOS_5C|REF_5C|IDS_CMP|CRITERIO_CMP \
      |PROC_STAT|CMP_CPU_JANELA|CMP_CPU_ESPERA|CMP_FOLGA|CMP_SEM_MEMORIA|CMP_TESTE_AMBOS|CURL|ESPERA_PRONTO|LOADAVG_ARQ|SYSTEMCTL|SYSTEMD_RUN|PENDENTE|CGROUP_RAIZ \
      |CMP_EST_*|CMP_FAKE_*|JANELA_*|LIMITE_*)
        recusa "variável só de teste presente no modo real: $v (só CMP_MODO_TESTE=1, o modo de teste explícito, a aceita); limpe o ambiente de quem chama";;
    esac
  done
fi
[ -x "$BIN" ] || recusa "binário não executável: $BIN"
[ -x "$JANELA" ] && [ -x "$LIMITE_SH" ] && [ -x "$LASTRO_SH" ] && [ -x "$LLAMA_SH" ] && [ -x "$AQUI/roda-cmp-monitor.sh" ] || recusa "faltam janela-e4b.sh, roda-com-limite.sh, roda-cmp-lastro.sh, roda-cmp-llamacpp.sh ou roda-cmp-monitor.sh executáveis"
[ -f "$EXTRAI" ] || recusa "falta cmp-extrai.py"
[ -n "$ESTADO" ] || ESTADO=$SAIDA-estado-janela
for d in "$SAIDA" "$ESTADO"; do
  if [ -e "$d" ]; then
    [ -d "$d" ] && [ -z "$(ls -A "$d")" ] || recusa "$d já existe e não está vazio: uma janela nova pede pasta nova"
  fi
done
SAIDA=$(realpath -m "$SAIDA"); BIN=$(realpath "$BIN")
# emenda 9 (r114): falha FECHADA antes de tocar em timers ou na GPU. Só PIDs de tipo G (desktop gráfico) são isentos; QUALQUER PID C ou C+G na 4090 (pmon ou --query-compute-apps:
# voxtype, processo de GPU do Chrome...) recusa a janela com a lista de PIDs e comandos: feche-os (ou abra o Chrome sem GPU) e peça de novo. Isso faz parte do "autorizo" do Breno.
PREFLIGHT=$("$AQUI/roda-cmp-monitor.sh" preflight 2>&1) || recusa "emenda 9: processo(s) de compute na 4090 que precisam ser FECHADOS antes da janela (voxtype, processo de GPU do Chrome...): $PREFLIGHT"
TIMERS=()
[ -z "${JANELA_TIMERS:-}" ] || TIMERS=(--timers "$JANELA_TIMERS")
coleta() {
  {
    echo "$1 $(date '+%d/%m %H:%M:%S')"
    "${NVIDIA_SMI:-nvidia-smi}" --query-gpu=uuid,memory.used,utilization.gpu --format=csv 2>&1 || echo "nvidia-smi indisponível"
    "${NVIDIA_SMI:-nvidia-smi}" --query-compute-apps=pid,used_memory,name --format=csv,noheader 2>&1 || true
    echo "loadavg $(cat "${LOADAVG_ARQ:-/proc/loadavg}" 2>&1)"
    echo "binário $(sha "$BIN")"
  } > "$2" 2>&1 || true
}
PASSA=("CUDA_VISIBLE_DEVICES=$UUID" PRIMEIRA_PALAVRA_AUTORIZADA=1)
for v in MODELO_1B MODELO_3B GGUF_DIR LLAMA_DIR TEXTOS_5C REF_5C IDS_CMP CRITERIO_CMP SEM_SHA_FIXO NVIDIA_SMI PROC_STAT CMP_CPU_JANELA CMP_CPU_ESPERA LASTRO_SH_CMP LLAMA_SH_CMP EXTRAI_PY \
         CMP_RELOGIO_ARQ CMP_FOLGA CMP_SEM_MEMORIA LLAMA_SERVER CURL MONITOR_SH ESPERA_PRONTO LOADAVG_ARQ $(compgen -v | grep -E '^CMP_(EST|FAKE)_' || true); do
  [ -z "${!v+x}" ] || PASSA+=("$v=${!v}")
done
lista_systemd() {
  "${SYSTEMCTL:-systemctl}" --user list-unit-files --type=service,timer > "$SAIDA.unidades-$1.txt" 2>&1 || true
  "${SYSTEMCTL:-systemctl}" --user list-timers --all > "$SAIDA.timers-$1.txt" 2>&1 || true
}
lista_systemd antes
coleta início "$SAIDA.carga-antes.txt"
EXTRA=(); [ $COM_S3 = 1 ] || EXTRA=(--sem-s3)
"$JANELA" --estado "$ESTADO" "${TIMERS[@]}" --limite $((LIMITE + 120)) --kill-apos 30 -- \
  "$LIMITE_SH" --limite "$LIMITE" -- env "${PASSA[@]}" "$AQUI/roda-cmp-janela.sh" --interno --modelos "$MODELOS" --saida "$SAIDA" --bin "$BIN" --limite "$LIMITE" "${EXTRA[@]}"
RC=$?
coleta fim "$SAIDA.carga-depois.txt"
lista_systemd depois
# aviso no fim da janela (sempre, qualquer que seja o código): arquivo ao lado da pasta e notificação se houver como
MSG="janela da comparação Lastro x llama.cpp ($MODELOS) terminou (código $RC): $SAIDA"
echo "$MSG" > "$SAIDA.aviso.txt"
if [ -n "${CMP_AVISO_CMD:-}" ]; then "$CMP_AVISO_CMD" "$MSG" > /dev/null 2>&1 || true
elif command -v notify-send > /dev/null 2>&1; then notify-send "Lastro x llama.cpp" "$MSG" > /dev/null 2>&1 || true; fi
exit $RC
