#!/usr/bin/env bash
# Perfil do prefill do Lastro (criterio-perfil-prefill-20261109.md, versão 3 e Emenda 1, Parte B): UMA janela POR MODELO (duas janelas no total, uma de cada vez), cada uma de até 20 min (estimativa ~7 min no 1B e ~12 min no 3B) na RTX 3080 Ti (12 GB), SÓ o Lastro (sem llama.cpp), descritiva.
# Derivado de roda-cmp-janela.sh: mesmo envelope (janela-e4b.sh pausa e restaura os timers; roda-com-limite.sh põe o limite; este script em modo interno), mesma falha FECHADA
# antes de tocar em timers ou na GPU, mesmo FIXO com shas e --verifica-fixos.
#
#   roda-perfil-janela.sh --modelos 1b|3b --saida DIR [--estado-janela DIR] [--bin PRIMEIRA_PALAVRA] [--limite S]
#   roda-perfil-janela.sh --verifica-fixos     (sem GPU, sem autorização, só lê: confere os shas congelados do FIXO contra os arquivos reais; código 0 só se todos OK)
#
# --modelos 1b|3b é OBRIGATÓRIO (uma janela por modelo; nunca as duas de uma vez). --limite padrão 1200 s e máximo 1200 s (20 min, o do escopo); o de fora é o de dentro mais 120 s.
#
# Antes da janela: systemctl --user stop voxtype; ao fim: systemctl --user start voxtype (passo manual, não automatizado; o voxtype em compute na 3080 Ti recusa a janela com código 2).
#
# Dentro da janela: (1) sha256 de tudo que o critério manda conferir ANTES e DEPOIS, contra os valores congelados onde há (quebrada = SEM PROVA em tudo); (2) ambiente.txt;
# (3) SMOKE antes das células, uma geração curta de 64 tokens por série que muda o posicionamento: L100 (CMP_FRACAO_SLOTS=1.0, como no degrau 1; julgado por `cmp-extrai.py --smoke`) e
# camadas-CPU (CMP_CAMADAS_CPU 22 no 1B e 29 no 3B, como no degrau 2; julgado por `cmp-extrai.py --smoke --degrau2`). Smoke POR SÉRIE: a falha de um smoke (código 0 ou 1, inclusive por memória) marca SÓ aquela série como SEM PROVA (motivo no veredito do smoke) e a janela segue com as outras;
# código 2 (recusa de ambiente) interrompe a janela inteira; sem repetição automática; (4) as células, em ordem fixa gravada em plano.txt (e, executada, em ordem.txt); (5) perfil-extrai.py (perfil-resultado.json/.md) e decisao.txt.
# Séries (todas só Lastro, quentes): L10 (padrão, sem variável), L100 (CMP_FRACAO_SLOTS=1.0) e LC (CMP_CAMADAS_CPU=N); textos curto, medio, longo. Ordem: por SÉRIE (L10, L100, LC) e,
# dentro dela, curto, medio, longo (assim um corte no fim deixa séries inteiras e a P1 de cada uma completa). Cada célula é UMA chamada ao roda-cmp-lastro.sh existente (--estado quente --n N:
# 1 aquecimento descartado dentro do processo + N medidas quentes; sem células frias), em pasta NOVA DIR/celulas/lastro-<modelo>-<série>-<texto>-quente.
#
# P4 (descritivo, por célula, fora da pasta da célula, que o roda-cmp-lastro.sh exige nova): DIR/p4/<chave>.gpu.csv = `nvidia-smi -i <UUID da 3080 Ti> --query-gpu=timestamp,utilization.gpu
# --format=csv,noheader -lms 100` e DIR/p4/<chave>.cpu.csv = %CPU do processo medido por /proc/<pid>/stat a cada ~100 ms (perfil-amostra-cpu.py). Cobre a célula inteira; não atribui nada ao prefill.
#
# ORÇAMENTO (estimativa ~12 min no 3B e ~7 min no 1B, PALPITES de ordem de grandeza; limite duro 20 min): antes de CADA célula o relógio projeta decorrido + estimativa do que falta + folga
# (180 s para encerramento, sha final e restauração dos timers) contra o limite. O ÚNICO corte é quente 5 -> 3 medidas (nas células ainda não iniciadas); a célula que ainda assim não couber é
# NÃO RODOU (nunca vira número nem estimativa). orcamento.json e decisao.txt declaram o que foi cortado. Estimativas (segundos, CMP_EST_*): portão 8, carga 5 (1B) e 7 (3B), por pedido quente 5 (1B)
# e 9 (3B), smoke 20 (1B) e 30 (3B).
#
# Falha FECHADA antes de tocar em timers ou na GPU: PRIMEIRA_PALAVRA_AUTORIZADA=1 e CUDA_VISIBLE_DEVICES igual ao UUID da 3080 Ti (a 4090 e a 5090 NUNCA entram) têm de estar no ambiente de quem chama
# (o "autorizo" do Breno); --saida e --estado-janela têm de ser pastas novas ou vazias; QUALQUER PID C ou C+G na 3080 Ti (voxtype, processo de GPU do Chrome...) recusa com código 2 (emenda 9 do
# critério base). Variáveis só para teste (JANELA_SH, LIMITE_SH, JANELA_TIMERS, LASTRO_SH_CMP, EXTRAI_PY, PERFIL_EXTRAI_PY, PERFIL_AMOSTRA_PY, CMP_RELOGIO_ARQ, CMP_FOLGA, CMP_EST_*, CMP_FAKE_*,
# MODELO_1B, MODELO_3B, TEXTOS_5C, REF_5C, IDS_CMP, CRITERIO_*, SEM_SHA_FIXO, NVIDIA_SMI, CMP_SEM_MEMORIA, PROC_STAT, CMP_CPU_JANELA, CMP_CPU_ESPERA, MONITOR_SH, LOADAVG_ARQ, SYSTEMCTL,
# SYSTEMD_RUN, PENDENTE, JANELA_*, LIMITE_*, CGROUP_RAIZ, CMP_FRACAO_SLOTS, CMP_CAMADAS_CPU...): no modo real qualquer uma presente no ambiente é RECUSADA com código 2; só CMP_MODO_TESTE=1
# as aceita. CMP_AVISO_CMD continua permitida (só avisa). Fim da janela: aviso (notify-send se houver, ou CMP_AVISO_CMD) e DIR.aviso.txt. Este roteiro NÃO é rodado na entrega.
# Códigos: 0 se decisao.txt foi gravada, o do comando (124 limite, ...) ou 2 recusa de uso, portão ou ambiente.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
UUID=GPU-b22d9465-8552-9f97-812e-380b590645f8
JANELA=${JANELA_SH:-$AQUI/janela-e4b.sh}
LIMITE_SH=${LIMITE_SH:-$AQUI/roda-com-limite.sh}
LASTRO_SH=${LASTRO_SH_CMP:-$AQUI/roda-cmp-lastro.sh}
MONITOR=${MONITOR_SH:-$AQUI/roda-cmp-monitor.sh}
EXTRAI=${EXTRAI_PY:-$AQUI/cmp-extrai.py}
PERFIL_EXTRAI=${PERFIL_EXTRAI_PY:-$AQUI/perfil-extrai.py}
AMOSTRA=${PERFIL_AMOSTRA_PY:-$AQUI/perfil-amostra-cpu.py}
BIN=$AQUI/../../rust/moe-rt/target/release/primeira-palavra
M1=${MODELO_1B:-/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct}
M3=${MODELO_3B:-/home/brenoperucchi/models/granite-3.1-3b-a800m-instruct}
TEXTOS=${TEXTOS_5C:-$AQUI/5c-textos}
REFS=${REF_5C:-$AQUI/5c-referencia}
IDS=${IDS_CMP:-$AQUI/cmp-ids-prompt.json}
CRIT_PERFIL=${CRITERIO_PERFIL:-$AQUI/criterio-perfil-prefill-20261109.md}
CRIT=${CRITERIO_CMP:-$AQUI/criterio-comparacao-llamacpp-20261108.md}
CRIT_D1=${CRITERIO_D1:-$AQUI/criterio-degrau1-100pct-20261109.md}
CRIT_D2=${CRITERIO_D2:-$AQUI/criterio-degrau2-camadas-20261109.md}
FOLGA=${CMP_FOLGA:-180}
LIMITE_MAX=1200
SAIDA=; ESTADO=; LIMITE=$LIMITE_MAX; INTERNO=0; MODELOS=; VERIFICA=0
usa() { echo "uso: roda-perfil-janela.sh --modelos 1b|3b --saida DIR [--estado-janela DIR] [--bin PRIMEIRA_PALAVRA] [--limite S] | --verifica-fixos" >&2; }
recusa() { echo "roda-perfil-janela: $*" >&2; exit 2; }
# Portão e registro da potência (Emenda 2 do perfil, Emenda 10 do cmp): le_pot lê só o power.limit da 3080 Ti (vazio se a leitura falhar); tem de ser 370.00 W no preflight (senão recusa 2) e de novo ao fim (senão a janela vira SEM PROVA via manifesto)
POT_OK="370.00 W"
le_pot() { local o; o=$("${NVIDIA_SMI:-nvidia-smi}" -i "$UUID" --query-gpu=power.limit --format=csv,noheader 2> /dev/null) || return 1; printf '%s\n' "$o" | head -n 1 | cut -d, -f1 | sed 's/^ *//;s/ *$//'; }
pot_tok() { local v=${1:-}; [ -n "$v" ] || v=ilegivel; echo "${v// /_}"; }
while [ $# -gt 0 ]; do
  case $1 in
    --saida) [ $# -ge 2 ] || { usa; exit 2; }; SAIDA=$2; shift 2;;
    --estado-janela) [ $# -ge 2 ] || { usa; exit 2; }; ESTADO=$2; shift 2;;
    --bin) [ $# -ge 2 ] || { usa; exit 2; }; BIN=$2; shift 2;;
    --limite) [ $# -ge 2 ] || { usa; exit 2; }; LIMITE=$2; shift 2;;
    --modelos) [ $# -ge 2 ] || { usa; exit 2; }; MODELOS=$2; shift 2;;
    --verifica-fixos) VERIFICA=1; shift;;
    --interno) INTERNO=1; shift;;
    *) usa; exit 2;;
  esac
done
sha() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }

# SHAs congelados (valem onde o critério já os fixa; GGUF, config e safetensors dos modelos só precisam ser iguais antes e depois). Os scripts e o binário daqui entram no FIXO.
# O próprio roda-perfil-janela.sh NÃO pode conter o seu sha: --verifica-fixos o imprime à parte (linha informativa) para conferência por fora.
declare -A FIXO=(
  [criterio-perfil]=fd9b6190773887c436ba3a3293121d4913113d1c995babf7151cab678e8a4e9e
  [binario-lastro]=65faf6e712276b2c6a6029fdd0ac8315fe1f707920f2b18e0205ef511d38a3e4
  [script-perfil-extrai]=755e1d7dd622e98dce9513ef30584e21667a5bc9d8c08cce4198090b2c34ba68
  [script-amostra-cpu]=de17df09d75015b54bc527954dc54944c465328a9bab1fbce6e20bb5a8adb3f4
  [script-cmp-extrai]=c5e44529c7ca958f6943ce12381b1da1a4eadd64453cb271cbc0d1b7536e7585
  [script-lastro]=8e924a36fff414f337857ffc75fc88c5b23e4cf12c3d6314384f3979cab80757
  [script-monitor]=3252c690ea38c447ec7dda24bbbe4ca801f3daefb93e63009564df3c62eac017
  [script-janela]=a168e6672ba87429b8c5ccc0c80ba7daa36335f6a911b91a3744454cf7e63d60
  [script-limite]=196c05649b43e64361766e98e5bab9873636799dfa5d83f70a468436cd4759cc
  [criterio-cmp]=8d9d6ad3ff1ef23d417e41982228ec55638823c613b4a2cbf00d813dc1f85b35
  [criterio-degrau1]=8c3406a9fd3ba55133dfb2c2d3800076c6ee0a765735edf8be65f6ba1a72c7de
  [criterio-degrau2]=bd8cb0510d5a6c3f55ec320a3177e2bec7ebce7b5776529989f0c45a61599b2a
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
# monta_lista ESCOPO: ESCOPO = 1b | 3b (só os arquivos do modelo da janela) ou todos (--verifica-fixos: os dois modelos)
monta_lista() {
  local escopo=$1
  CHAVES=(); CAMINHOS=()
  add() { CHAVES+=("$1"); CAMINHOS+=("$2"); }
  add criterio-perfil "$CRIT_PERFIL"
  add binario-lastro "$BIN"
  add script-perfil-extrai "$PERFIL_EXTRAI"; add script-amostra-cpu "$AMOSTRA"; add script-cmp-extrai "$EXTRAI"; add script-lastro "$LASTRO_SH"; add script-monitor "$MONITOR"
  add script-janela "$JANELA"; add script-limite "$LIMITE_SH"
  add criterio-cmp "$CRIT"; add criterio-degrau1 "$CRIT_D1"; add criterio-degrau2 "$CRIT_D2"
  add cmp-ids-prompt "$IDS"
  add texto-curto "$TEXTOS/texto-curto.txt"; add texto-medio "$TEXTOS/texto-medio.txt"; add texto-longo "$TEXTOS/texto-longo.txt"; add textos-json "$TEXTOS/textos.json"
  add ref5c-1b "$REFS/ref5c-1b.json"; add ref5c-3b "$REFS/ref5c-3b.json"
  if [ "$escopo" = 1b ] || [ "$escopo" = todos ]; then
    add config-1b "$M1/config.json"; add tokenizer-1b "$M1/tokenizer.json"; add safetensors-1b "$M1/model.safetensors"
  fi
  if [ "$escopo" = 3b ] || [ "$escopo" = todos ]; then
    add config-3b "$M3/config.json"; add tokenizer-3b "$M3/tokenizer.json"
    add safetensors-3b-1 "$M3/model-00001-of-00002.safetensors"; add safetensors-3b-2 "$M3/model-00002-of-00002.safetensors"; add indice-3b "$M3/model.safetensors.index.json"
  fi
}
# coleta_shas ARQ: uma linha "chave sha" por item (vazio = ilegível); hashes em paralelo (os arquivos grandes dominam o tempo).
coleta_shas() {
  local tmp="$SAIDA/.shas.$$" i
  mkdir -p "$tmp"
  for i in "${!CHAVES[@]}"; do ( sha "${CAMINHOS[$i]}" > "$tmp/$i" ) & done
  wait
  : > "$1"
  for i in "${!CHAVES[@]}"; do echo "${CHAVES[$i]} $(cat "$tmp/$i" 2>/dev/null)" >> "$1"; done
  # sem remoção (regra do Breno: nada de rm com caminho montado por variável): os temporários ficam em $SAIDA/.shas.PID
}

# --verifica-fixos: só lê (sem GPU, sem autorização, sem tocar em timers). AUSENTE = arquivo que não existe aqui (ex.: tokenizer fora do repositório) e conta como falha (código 1).
if [ $VERIFICA = 1 ]; then
  monta_lista todos
  rc_v=0
  for i in "${!CHAVES[@]}"; do
    k=${CHAVES[$i]}
    [ -n "${FIXO[$k]:-}" ] || continue
    h=$(sha "${CAMINHOS[$i]}")
    if [ -z "$h" ]; then echo "$k ${FIXO[$k]} - AUSENTE ${CAMINHOS[$i]##*/}"; rc_v=1
    elif [ "$h" = "${FIXO[$k]}" ]; then echo "$k ${FIXO[$k]} $h OK"
    else echo "$k ${FIXO[$k]} $h FALHA"; rc_v=1; fi
  done
  echo "informativo: roda-perfil-janela.sh sha256 $(sha "$AQUI/roda-perfil-janela.sh") (autorreferente, fora do FIXO; conferir por fora)"
  exit $rc_v
fi
# --modelos: uma janela por modelo; ausente ou inválido = recusa 2 (nunca as duas de uma vez, nem 'ambos', nem '1b,3b')
case $MODELOS in
  1b|3b) ;;
  '') usa; recusa "--modelos 1b|3b é obrigatório: uma janela por modelo";;
  *) recusa "--modelos inválido: '$MODELOS' (use 1b ou 3b, nunca os dois)";;
esac
[ -n "$SAIDA" ] || { usa; exit 2; }
case $MODELOS in 1b) NCPU=22;; 3b) NCPU=29;; esac

# ---------------------------------------------------------------- modo interno (dentro da janela, com o portão e o UUID no ambiente)
if [ $INTERNO = 1 ]; then
  [ "${PRIMEIRA_PALAVRA_AUTORIZADA:-}" = 1 ] && [ "${CUDA_VISIBLE_DEVICES:-}" = "$UUID" ] || recusa "modo interno sem portão ou UUID"
  mkdir -p "$SAIDA/celulas" "$SAIDA/p4" || exit 2
  # as variáveis de série só existem por célula (env na chamada); nada herdado do ambiente
  unset CMP_FRACAO_SLOTS CMP_CAMADAS_CPU
  agora_s() { if [ -n "${CMP_RELOGIO_ARQ:-}" ]; then cat "$CMP_RELOGIO_ARQ"; else date +%s; fi; }
  INICIO=$(agora_s)
  decorrido() { echo $(( $(agora_s) - INICIO )); }
  SMI=${NVIDIA_SMI:-nvidia-smi}
  monta_lista "$MODELOS"
  coleta_shas "$SAIDA/integridade-antes.txt"
  {
    echo "inicio=$(date -Is)"
    echo "uuid=$UUID"
    echo "modelo=$MODELOS"
    echo "limite_s=$LIMITE folga_s=$FOLGA"
    "$SMI" --query-gpu=name,driver_version,clocks.max.memory --format=csv,noheader 2>&1 | head -n 2 | sed 's/^/gpu=/'
    # Emenda 10/2: limite de potência e link PCIe da placa de medida, registrados (a série vale no limite vigente, hoje 370 W; se mudar, a série é invalidada)
    "$SMI" -i "$UUID" --query-gpu=power.limit,pcie.link.width.current,pcie.link.gen.max --format=csv,noheader 2>&1 | head -n 2 | sed 's/^/gpu_limites=/'
    POT_INI=$(le_pot) || POT_INI=; echo "potencia_inicio=$(pot_tok "$POT_INI")"
    echo "loadavg=$(cut -d' ' -f1-3 < /proc/loadavg 2>/dev/null)"
    if [ "${CMP_SEM_MEMORIA:-0}" != 1 ]; then
      mts=$(sudo -n dmidecode -t memory 2>/dev/null | awk -F': ' '/Configured Memory Speed/ && $2 ~ /[0-9]/ { split($2, a, " "); print a[1]; exit }')
      [ -z "$mts" ] || echo "memoria_mts=$mts"
    fi
  } > "$SAIDA/ambiente.txt" 2>&1

  # ---- plano e orçamento
  est() { local v=CMP_EST_$1; echo "${!v:-$2}"; }   # est NOME PADRAO
  E_PORTAO=$(est PORTAO 8)
  declare -A E_CARGA=([1b]=$(est CARGA_1B 5) [3b]=$(est CARGA_3B 7))
  declare -A E_QUENTE=([1b]=$(est QUENTE_1B 5) [3b]=$(est QUENTE_3B 9))
  declare -A E_SMOKE=([1b]=$(est SMOKE_1B 20) [3b]=$(est SMOKE_3B 30))
  est_celula() { echo $(( E_PORTAO + E_CARGA[$MODELOS] + ($1 + 1) * E_QUENTE[$MODELOS] )); }   # est_celula N
  SERIES=(L10 L100 LC); TEXTOS_L=(curto medio longo)
  CELULAS=()
  for s in "${SERIES[@]}"; do for t in "${TEXTOS_L[@]}"; do CELULAS+=("$s:$t"); done; done
  REPS_Q=5; CORTES=(); NAO_RODOU=()
  : > "$SAIDA/ordem.txt"; : > "$SAIDA/orcamento-log.txt"
  { echo "plano da janela do modelo $MODELOS (ordem fixa; séries todas só Lastro, quentes: 1 aquecimento descartado + N medidas)"
    echo "smoke antes das células: L100 (CMP_FRACAO_SLOTS=1.0), depois LC (CMP_CAMADAS_CPU=$NCPU)"
    i=0; for c in "${CELULAS[@]}"; do i=$((i + 1)); printf '%02d lastro-%s-%s-%s-quente\n' "$i" "$MODELOS" "${c%%:*}" "${c##*:}"; done; } > "$SAIDA/plano.txt"
  SEQ=0; ABORTA=0
  nr() { NAO_RODOU+=("$1"); echo "NÃO RODOU: $1" >> "$SAIDA/orcamento-log.txt"; }
  # env por série: o roda-cmp-lastro.sh só entende CMP_FRACAO_SLOTS=1.0 ou CMP_CAMADAS_CPU=<N fixo do modelo>
  env_serie() { # série -> ENVX
    case $1 in
      L10) ENVX=(-u CMP_FRACAO_SLOTS -u CMP_CAMADAS_CPU);;
      L100) ENVX=(-u CMP_CAMADAS_CPU CMP_FRACAO_SLOTS=1.0);;
      LC) ENVX=(-u CMP_FRACAO_SLOTS CMP_CAMADAS_CPU=$NCPU);;
    esac
  }
  # ---- P4: amostras por célula (arquivos em DIR/p4, fora da pasta da célula)
  GPU_PID=; CPU_PID=
  inicia_amostras() { # chave pid_da_celula
    local pre=(); command -v stdbuf > /dev/null 2>&1 && pre=(stdbuf -oL)
    "${pre[@]}" "$SMI" -i "$UUID" --query-gpu=timestamp,utilization.gpu --format=csv,noheader -lms 100 > "$SAIDA/p4/$1.gpu.csv" 2> "$SAIDA/p4/$1.gpu.err" < /dev/null &
    GPU_PID=$!
    python3 -I "$AMOSTRA" --raiz "$2" --saida "$SAIDA/p4/$1.cpu.csv" > /dev/null 2> "$SAIDA/p4/$1.cpu.err" < /dev/null &
    CPU_PID=$!
  }
  para_amostras() {
    local p
    for p in "$GPU_PID" "$CPU_PID"; do [ -z "$p" ] || kill "$p" 2> /dev/null; done
    for p in "$GPU_PID" "$CPU_PID"; do [ -z "$p" ] || wait "$p" 2> /dev/null; done
    GPU_PID=; CPU_PID=
  }
  trap para_amostras EXIT
  roda_celula() { # série texto n
    local s=$1 t=$2 n=$3 chave rc t0 cpid
    [ $ABORTA = 0 ] || return 0
    chave="lastro-$MODELOS-$s-$t-quente"
    SEQ=$((SEQ + 1)); t0=$(decorrido)
    env_serie "$s"
    env "${ENVX[@]}" "$LASTRO_SH" --modelo "$MODELOS" --texto "$t" --estado quente --n "$n" --saida "$SAIDA/celulas/$chave" --bin "$BIN" > "$SAIDA/celulas-$chave.log" 2>&1 &
    cpid=$!
    inicia_amostras "$chave" "$cpid"
    wait "$cpid"; rc=$?
    para_amostras
    printf '%02d %s n=%s inicio=%ss fim=%ss rc=%s\n' "$SEQ" "$chave" "$n" "$t0" "$(decorrido)" "$rc" >> "$SAIDA/ordem.txt"
    [ "$rc" != 2 ] || { ABORTA=1; echo "célula $chave recusou o ambiente (código 2): janela interrompida" >> "$SAIDA/orcamento-log.txt"; }
  }
  # ---- smoke ANTES das células (L100 e LC), POR SÉRIE: falha (exceto ids) com código 0 ou 1 marca SÓ aquela série como SEM PROVA (motivo em SERIE_FALHA e no veredito.txt do smoke,
  # que o perfil-extrai.py lê); a janela segue com as outras séries, sem repetição. Código 2 (recusa de ambiente) interrompe a janela inteira (ABORTA). SMOKE_FALHA global só para "sem orçamento".
  SMOKE_FALHA=
  declare -A SERIE_FALHA=()
  roda_smoke() { # série
    local s=$1 dir="$SAIDA/smoke-$1" t0 rc_s rc_ev smk=() msg
    env_serie "$s"
    if [ "$s" = LC ]; then smk=(--degrau2); msg="SMOKE DE CAMADAS-CPU FALHOU"; else msg="L100 NÃO SUSTENTA 100%"; fi
    t0=$(decorrido)
    env "${ENVX[@]}" "$LASTRO_SH" --smoke --modelo "$MODELOS" --saida "$dir" --bin "$BIN" > "$SAIDA/smoke-$s.log" 2>&1; rc_s=$?
    printf 'smoke-%s inicio=%ss fim=%ss rc=%s\n' "$s" "$t0" "$(decorrido)" "$rc_s" >> "$SAIDA/ordem.txt"
    [ "$rc_s" != 2 ] || { ABORTA=1; echo "smoke $s recusou o ambiente (código 2): janela interrompida" >> "$SAIDA/orcamento-log.txt"; }
    mkdir -p "$dir"
    python3 -I "$EXTRAI" --smoke "$dir" --modelo "$MODELOS" "${smk[@]}" --ids "$IDS" --refs "$REFS" --textos "$TEXTOS/textos.json" > "$SAIDA/smoke-$s-veredito.out" 2>&1; rc_ev=$?
    if [ $rc_ev = 2 ]; then printf 'veredito=FALHA\nmotivo=smoke não avaliável (extrator recusou: pasta ou modelo)\n' > "$dir/veredito.txt"; fi
    if ! grep -q '^veredito=OK$' "$dir/veredito.txt" 2> /dev/null; then
      local mot
      mot=$(grep '^motivo=' "$dir/veredito.txt" 2> /dev/null | head -n 1 | cut -d= -f2-)
      [ -n "$mot" ] || mot="smoke sem veredito legível"
      SERIE_FALHA[$s]="$msg ($s): $mot"
      echo "SEM PROVA, série $s: ${SERIE_FALHA[$s]} (a janela segue com as outras séries)" >> "$SAIDA/orcamento-log.txt"
    fi
  }
  if [ $(( $(decorrido) + 2 * E_SMOKE[$MODELOS] + FOLGA )) -gt "$LIMITE" ]; then
    SMOKE_FALHA="sem orçamento nem para os dois smokes"
    nr "janela do modelo $MODELOS inteira (sem orçamento para o smoke: $LIMITE s)"
  else
    roda_smoke L100
    [ $ABORTA != 0 ] || roda_smoke LC
  fi
  projeta_restante() { local i tot=0; for ((i = $1; i < ${#CELULAS[@]}; i++)); do [ -z "${SERIE_FALHA[${CELULAS[$i]%%:*}]:-}" ] || continue; tot=$(( tot + $(est_celula "$REPS_Q") )); done; echo "$tot"; }
  for ((ci = 0; ci < ${#CELULAS[@]}; ci++)); do
    s=${CELULAS[$ci]%%:*}; t=${CELULAS[$ci]##*:}
    if [ $ABORTA != 0 ] || [ -n "$SMOKE_FALHA" ]; then
      [ -n "$SMOKE_FALHA" ] || nr "lastro-$MODELOS-$s-$t-quente: janela interrompida por recusa de ambiente (código 2)"
      continue
    fi
    [ -z "${SERIE_FALHA[$s]:-}" ] || continue   # série reprovada no smoke: SEM PROVA (motivo no veredito do smoke), nenhuma célula, nenhuma nova tentativa
    # único corte: quente 5 -> 3 medidas, antes de iniciar a célula, quando a projeção estoura
    if [ $REPS_Q = 5 ] && [ $(( $(decorrido) + $(projeta_restante "$ci") + FOLGA )) -gt "$LIMITE" ]; then
      REPS_Q=3; CORTES+=("quente cai de 5 para 3 medidas (nas células ainda não iniciadas)")
      echo "corte em $(decorrido) s: ${CORTES[-1]}" >> "$SAIDA/orcamento-log.txt"
    fi
    if [ $(( $(decorrido) + $(est_celula "$REPS_Q") + FOLGA )) -gt "$LIMITE" ]; then nr "lastro-$MODELOS-$s-$t-quente: não cabe, sem mais cortes"; continue; fi
    roda_celula "$s" "$t" "$REPS_Q"
  done
  [ $ABORTA = 0 ] || nr "janela interrompida por recusa de ambiente (código 2) na célula ${SEQ}"

  # ---- integridade depois
  coleta_shas "$SAIDA/integridade-depois.txt"
  POT_FIM=$(le_pot) || POT_FIM=; echo "potencia_fim=$(pot_tok "$POT_FIM")" >> "$SAIDA/ambiente.txt"
  INTEGRO=1; NOTAS=()
  if ! cmp -s "$SAIDA/integridade-antes.txt" "$SAIDA/integridade-depois.txt"; then INTEGRO=0; NOTAS+=("sha mudou entre antes e depois"); fi
  # manifesto gravado em arquivo temporário e movido de uma vez: a interrupção entre integridade-depois.txt e este arquivo deixa o manifesto AUSENTE (perfil-extrai.py trata como SEM PROVA em tudo), nunca truncado
  : > "$SAIDA/.integridade-fixos.tmp"
  while read -r k h; do
    [ -n "$h" ] || { INTEGRO=0; NOTAS+=("$k ilegível"); echo "$k - - FALHA ilegível" >> "$SAIDA/.integridade-fixos.tmp"; continue; }
    if [ -n "${FIXO[$k]:-}" ]; then
      if [ "${SEM_SHA_FIXO:-0}" = 1 ]; then echo "$k ${FIXO[$k]} $h NAO-CONFERIDO" >> "$SAIDA/.integridade-fixos.tmp"
      elif [ "$h" = "${FIXO[$k]}" ]; then echo "$k ${FIXO[$k]} $h OK" >> "$SAIDA/.integridade-fixos.tmp"
      else INTEGRO=0; NOTAS+=("$k com sha $h, não o congelado"); echo "$k ${FIXO[$k]} $h FALHA" >> "$SAIDA/.integridade-fixos.tmp"; fi
    fi
  done < "$SAIDA/integridade-antes.txt"
  # power.limit: o mesmo 370 W do preflight no início e ao fim; mudança, leitura ausente ou ilegível = integridade QUEBRADA (tudo SEM PROVA, a causa vai no manifesto e em NOTAS)
  if [ "$POT_INI" = "$POT_OK" ] && [ "$POT_FIM" = "$POT_OK" ]; then echo "potencia ${POT_OK// /_} $(pot_tok "$POT_FIM") OK" >> "$SAIDA/.integridade-fixos.tmp"
  else INTEGRO=0; NOTAS+=("power.limit fora de $POT_OK ou ilegível (início '${POT_INI:-ilegível}', fim '${POT_FIM:-ilegível}')"); echo "potencia inicio=$(pot_tok "$POT_INI") fim=$(pot_tok "$POT_FIM") FALHA" >> "$SAIDA/.integridade-fixos.tmp"; fi
  mv "$SAIDA/.integridade-fixos.tmp" "$SAIDA/integridade-fixos.txt"
  # ---- orçamento declarado
  {
    printf '{"modelos": ["%s"], "limite_s": %s, "folga_s": %s, "decorrido_s": %s, "cortes": [' "$MODELOS" "$LIMITE" "$FOLGA" "$(decorrido)"
    sep=; for c in "${CORTES[@]:-}"; do [ -n "$c" ] || continue; printf '%s"%s"' "$sep" "$c"; sep=', '; done
    printf '], "reps": {"quente_nominal": 5}, "reps_finais": {"quente": %s}, "nao_rodou": [' "$REPS_Q"
    sep=; for c in "${NAO_RODOU[@]:-}"; do [ -n "$c" ] || continue; printf '%s"%s"' "$sep" "$(echo "$c" | tr -d '"\\')"; sep=', '; done
    printf '], "estimativas_sao_palpites": true}\n'
  } > "$SAIDA/orcamento.json"
  python3 -I "$PERFIL_EXTRAI" --saida "$SAIDA" --ids "$IDS" --refs "$REFS" --textos "$TEXTOS/textos.json" > "$SAIDA/perfil-extrai.out" 2>&1; RC_EXT=$?
  {
    echo "perfil do prefill (Parte B), janela do modelo $MODELOS, critério sha256 $(sha "$CRIT_PERFIL")"
    echo "resultado DESCRITIVO: sem meta, sem PASSA/REPROVA; P1 e P3 são previsões pré-registradas conferidas, P2 é descritivo na 3080 Ti (Emenda 2), P4 descreve a célula inteira sem atribuir nada ao prefill; só Lastro, sem llama.cpp"
    for s in L100 LC; do [ -z "${SERIE_FALHA[$s]:-}" ] || echo "SEM PROVA (série $s): ${SERIE_FALHA[$s]}"; done
    if [ -n "$SMOKE_FALHA" ]; then echo "$SMOKE_FALHA"; else echo "smoke L100: $(grep -E '^(veredito|ids|gpu_pico_pct|cpu_needed)=' "$SAIDA/smoke-L100/veredito.txt" 2> /dev/null | tr '\n' ' ')"; echo "smoke LC: $(grep -E '^(veredito|ids|gpu_pico_pct|cpu_needed|camadas_cpu)=' "$SAIDA/smoke-LC/veredito.txt" 2> /dev/null | tr '\n' ' ')"; fi
    if [ $INTEGRO = 1 ]; then echo "integridade: íntegra (binário, scripts, GGUF/safetensors do modelo, textos, referências, ids e critérios iguais antes e depois$([ "${SEM_SHA_FIXO:-0}" = 1 ] && echo '; congelados NÃO conferidos: SEM_SHA_FIXO=1' || echo ' e aos congelados onde há'))"
    else echo "integridade: QUEBRADA: tudo SEM PROVA"; for n in "${NOTAS[@]}"; do echo "  $n"; done; fi
    echo "cortes de orçamento: ${CORTES[*]:-nenhum}"
    echo "repetições finais: quente $REPS_Q; decorrido $(decorrido) s de $LIMITE s"
    echo "não rodou: ${#NAO_RODOU[@]} item(ns); ver orcamento.json"
    echo "extrator: $(cat "$SAIDA/perfil-extrai.out") (código $RC_EXT)"
    echo "células executadas: $SEQ (ordem.txt; plano em plano.txt)"
  } > "$SAIDA/decisao.txt"
  cat "$SAIDA/decisao.txt"
  exit 0
fi

# ---------------------------------------------------------------- modo externo (portões, janela e limite)
case $LIMITE in ''|*[!0-9]*) recusa "limite inválido: $LIMITE";; esac
[ "$LIMITE" -ge 1 ] && [ "$LIMITE" -le $LIMITE_MAX ] || recusa "limite fora de 1..$LIMITE_MAX s: $LIMITE"
[ "${PRIMEIRA_PALAVRA_AUTORIZADA:-}" = 1 ] || recusa "portão fechado: sem PRIMEIRA_PALAVRA_AUTORIZADA=1 (o 'autorizo' do Breno que nomeia a janela)"
[ "${CUDA_VISIBLE_DEVICES:-}" = "$UUID" ] || recusa "CUDA_VISIBLE_DEVICES tem de ser $UUID (a 3080 Ti; a 4090 e a 5090 nunca); veio '${CUDA_VISIBLE_DEVICES:-}'"
# variáveis só de teste no ambiente do modo real = recusa 2; só CMP_MODO_TESTE=1 as aceita. `compgen -e` lista só as EXPORTADAS (as internas do roteiro não contam).
if [ "${CMP_MODO_TESTE:-0}" != 1 ]; then
  for v in $(compgen -e); do
    case $v in
      CMP_FRACAO_SLOTS|CMP_CAMADAS_CPU|CRITERIO_PERFIL|CRITERIO_D1|CRITERIO_D2|SEM_SHA_FIXO|MONITOR_SH|LASTRO_SH_CMP|EXTRAI_PY|PERFIL_EXTRAI_PY|PERFIL_AMOSTRA_PY|CMP_RELOGIO_ARQ|NVIDIA_SMI|MODELO_1B|MODELO_3B|TEXTOS_5C|REF_5C|IDS_CMP|CRITERIO_CMP \
      |PROC_STAT|CMP_CPU_JANELA|CMP_CPU_ESPERA|CMP_FOLGA|CMP_SEM_MEMORIA|ESPERA_PRONTO|LOADAVG_ARQ|SYSTEMCTL|SYSTEMD_RUN|PENDENTE|CGROUP_RAIZ \
      |CMP_EST_*|CMP_FAKE_*|JANELA_*|LIMITE_*)
        recusa "variável só de teste presente no modo real: $v (só CMP_MODO_TESTE=1, o modo de teste explícito, a aceita); limpe o ambiente de quem chama";;
    esac
  done
fi
[ -x "$BIN" ] || recusa "binário não executável: $BIN"
[ -x "$JANELA" ] && [ -x "$LIMITE_SH" ] && [ -x "$LASTRO_SH" ] && [ -x "$MONITOR" ] || recusa "faltam janela-e4b.sh, roda-com-limite.sh, roda-cmp-lastro.sh ou roda-cmp-monitor.sh executáveis"
[ -f "$EXTRAI" ] && [ -f "$PERFIL_EXTRAI" ] && [ -f "$AMOSTRA" ] || recusa "faltam cmp-extrai.py, perfil-extrai.py ou perfil-amostra-cpu.py"
[ -f "$CRIT_PERFIL" ] || recusa "falta o critério do perfil: $CRIT_PERFIL"
[ -n "$ESTADO" ] || ESTADO=$SAIDA-estado-janela
for d in "$SAIDA" "$ESTADO"; do
  if [ -e "$d" ]; then
    [ -d "$d" ] && [ -z "$(ls -A "$d")" ] || recusa "$d já existe e não está vazio: uma janela nova pede pasta nova"
  fi
done
SAIDA=$(realpath -m "$SAIDA"); BIN=$(realpath "$BIN")
# falha FECHADA antes de tocar em timers ou na GPU: QUALQUER PID C ou C+G na 3080 Ti (voxtype, processo de GPU do Chrome...) recusa a janela; só PIDs de tipo G (desktop gráfico) são isentos.
POT_PRE=$(le_pot) || POT_PRE=
[ "$POT_PRE" = "$POT_OK" ] || recusa "power.limit da 3080 Ti tem de ser $POT_OK (lido: '${POT_PRE:-ilegível}'); a série vale só no limite vigente; nada foi tocado, não mexa no limite por este roteiro"
PREFLIGHT=$("$MONITOR" preflight 2>&1) || recusa "processo(s) de compute na 3080 Ti que precisam ser FECHADOS antes da janela (voxtype, processo de GPU do Chrome...): $PREFLIGHT"
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
for v in MODELO_1B MODELO_3B TEXTOS_5C REF_5C IDS_CMP CRITERIO_PERFIL CRITERIO_CMP CRITERIO_D1 CRITERIO_D2 SEM_SHA_FIXO NVIDIA_SMI PROC_STAT CMP_CPU_JANELA CMP_CPU_ESPERA LASTRO_SH_CMP EXTRAI_PY PERFIL_EXTRAI_PY PERFIL_AMOSTRA_PY \
         CMP_RELOGIO_ARQ CMP_FOLGA CMP_SEM_MEMORIA MONITOR_SH ESPERA_PRONTO LOADAVG_ARQ $(compgen -v | grep -E '^CMP_(EST|FAKE)_' || true); do
  [ -z "${!v+x}" ] || PASSA+=("$v=${!v}")
done
lista_systemd() {
  "${SYSTEMCTL:-systemctl}" --user list-unit-files --type=service,timer > "$SAIDA.unidades-$1.txt" 2>&1 || true
  "${SYSTEMCTL:-systemctl}" --user list-timers --all > "$SAIDA.timers-$1.txt" 2>&1 || true
}
lista_systemd antes
coleta início "$SAIDA.carga-antes.txt"
"$JANELA" --estado "$ESTADO" "${TIMERS[@]}" --limite $((LIMITE + 120)) --kill-apos 30 -- \
  "$LIMITE_SH" --limite "$LIMITE" -- env "${PASSA[@]}" "$AQUI/roda-perfil-janela.sh" --interno --modelos "$MODELOS" --saida "$SAIDA" --bin "$BIN" --limite "$LIMITE"
RC=$?
coleta fim "$SAIDA.carga-depois.txt"
lista_systemd depois
# aviso no fim da janela (sempre, qualquer que seja o código): arquivo ao lado da pasta e notificação se houver como
MSG="janela do perfil do prefill do Lastro ($MODELOS) terminou (código $RC): $SAIDA"
echo "$MSG" > "$SAIDA.aviso.txt"
if [ -n "${CMP_AVISO_CMD:-}" ]; then "$CMP_AVISO_CMD" "$MSG" > /dev/null 2>&1 || true
elif command -v notify-send > /dev/null 2>&1; then notify-send "Perfil do prefill (Lastro)" "$MSG" > /dev/null 2>&1 || true; fi
exit $RC
