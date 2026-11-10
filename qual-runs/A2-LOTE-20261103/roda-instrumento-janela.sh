#!/usr/bin/env bash
# Instrumento de prefill do Lastro (criterio-instrumento-prefill-20261110.md, derivado de plano-instrumento-prefill-20261110.md): UMA janela POR MODELO (duas janelas no total, uma de cada vez),
# cada uma de até 20 min na RTX 3080 Ti (12 GB), SÓ o Lastro, com o binário INSTRUMENTADO (cópia congelada fora do git), descritiva.
# Derivado de roda-perfil-janela.sh: mesmo envelope (janela-e4b.sh pausa e restaura os timers; roda-com-limite.sh põe o limite; este script em modo interno), mesma falha FECHADA antes de tocar em timers
# ou na GPU, mesmo FIXO com shas e --verifica-fixos.
#
#   roda-instrumento-janela.sh --modelos 1b|3b --saida DIR [--estado-janela DIR] [--bin PRIMEIRA_PALAVRA_INSTR] [--limite S]
#   roda-instrumento-janela.sh --verifica-fixos     (sem GPU, sem autorização, só lê: confere os shas congelados do FIXO contra os arquivos reais; código 0 só se todos OK)
#
# --modelos 1b|3b é OBRIGATÓRIO (uma janela por modelo; nunca as duas de uma vez). --limite padrão 1200 s e máximo 1200 s (20 min); o de fora é o de dentro mais 120 s. Sem repetição automática.
# Antes da janela: systemctl --user stop voxtype; ao fim: systemctl --user start voxtype (passo manual, não automatizado; o voxtype em compute na 3080 Ti recusa a janela com código 2).
#
# Ordem da janela (seção 2.1 do plano): (1) preflight, (2) PARES DE CONTROLE, (3) demais células obrigatórias, (4) células opcionais (LC).
#  (2) por série L10 e L100, na célula do texto MÉDIO (170 posições): 6 processos na ordem FIXA D L L D D L, cada um `--gera --so-texto medio --repete 3` (1 aquecimento descartado + 3 medidas, 4 pedidos),
#      pasta celulas/ctrl-<modelo>-<série>-medio-<NN>-<D|L>. Os 3 processos L valem também como dados principais das células (L10, médio) e (L100, médio) (9 pedidos medidos). Todo processo D roda com
#      `env -u LASTRO_PERFIL_BLOCO` (não pode produzir perfil-bloco.json); todo processo L roda com LASTRO_PERFIL_BLOCO=1 NA CHAMADA, nunca herdada. A variável nunca liga em D (ponto P3 do rev-2 da rodada 137):
#      o ambiente de quem chama não pode trazê-la (recusa 2 no modo real), smoke e LC rodam sem ela, e prova-variavel.txt grava, por processo, o ambiente passado e a linha `perfil-bloco: ...` que o
#      binário imprimiu (D = desligado, L = ligado; qualquer incoerência vai para a decisão e o extrator invalida o controle).
#  (3) L10 e L100 nos textos curto e longo, 1 processo de 1 aquecimento + 5 medidas (corte único 5 -> 3 nas células ainda não iniciadas, como no roteiro do perfil), pasta celulas/lastro-<modelo>-<série>-<texto>-quente.
#  (4) LC (camadas-CPU 22 no 1B e 29 no 3B), curto, médio, longo, só se sobrar tempo (smoke antes; sem controle, descritiva); é o primeiro corte e nunca encolhe o bloco (2).
# O smoke do L100 roda antes dos pares (a série L100 depende dele); falha de smoke marca SÓ aquela série como SEM PROVA e a janela segue; código 2 interrompe tudo. Cada processo é UMA chamada ao
# roda-cmp-lastro.sh existente (portão pmon, monitor, celula.json).
#
# ORÇAMENTO: o bloco (2) é reservado antes de tudo. Antes de tocar em timers, o modo externo conta: smoke L100 + 12 processos x (portão + carga + 4 x pedido quente) + folga (180 s) contra o limite; se não couber,
# a janela NÃO COMEÇA (código 2, DIR/decisao.txt "SEM PROVA da unidade inteira", nada tocado, sem repetição automática). Estimativas (segundos, PALPITES): CMP_EST_PORTAO 8, CARGA 5 (1B) e 7 (3B),
# pedido quente 5 (1B) e 9 (3B), smoke 20 (1B) e 30 (3B). Dentro da janela o relógio projeta antes de cada processo; o que não couber é NÃO RODOU (nunca vira número).
#
# Falha FECHADA antes de tocar em timers ou na GPU: PRIMEIRA_PALAVRA_AUTORIZADA=1 e CUDA_VISIBLE_DEVICES igual ao UUID da 3080 Ti (a 4090 e a 5090 NUNCA entram); --saida e --estado-janela pastas novas ou vazias;
# power.limit 370.00 W no início (e ao fim, senão a janela vira SEM PROVA e o extrator RECUSA); QUALQUER PID C ou C+G na 3080 Ti recusa (voxtype, GPU do Chrome...); RECUSA TAMBÉM se houver `cargo test`/`cargo nextest`,
# binário de teste do cargo (target/*/deps/<nome>-<hash>), primeira-palavra ou llama.* em execução (incidente do rev-2: `cargo test` sem filtro abriu a 3080 Ti por ~10 min); a lista vem de `ps`, e os ancestrais
# deste script ficam de fora. Variáveis só para teste (JANELA_SH, LIMITE_SH, JANELA_TIMERS, LASTRO_SH_CMP, EXTRAI_PY, PERFIL_EXTRAI_PY, INSTR_EXTRAI_PY, CMP_RELOGIO_ARQ, CMP_FOLGA, CMP_EST_*, CMP_FAKE_*, MODELO_1B,
# MODELO_3B, TEXTOS_5C, REF_5C, IDS_CMP, CRITERIO_*, PLANO_INSTR, SEM_SHA_FIXO, NVIDIA_SMI, CMP_SEM_MEMORIA, PROC_STAT, CMP_CPU_*, MONITOR_SH, LOADAVG_ARQ, SYSTEMCTL, SYSTEMD_RUN, PENDENTE, JANELA_*, LIMITE_*,
# CGROUP_RAIZ, CMP_FRACAO_SLOTS, CMP_CAMADAS_CPU, PROC_LISTA, LASTRO_PERFIL_BLOCO...): no modo real qualquer uma presente no ambiente é RECUSADA com código 2; só CMP_MODO_TESTE=1 as aceita.
# CMP_AVISO_CMD continua permitida (só avisa). Fim da janela: aviso (notify-send se houver, ou CMP_AVISO_CMD) e DIR.aviso.txt. Este roteiro NÃO é rodado na entrega.
# Códigos: 0 se decisao.txt foi gravada, o do comando (124 limite, ...) ou 2 recusa de uso, portão, ambiente ou bloco do par que não cabe.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
UUID=GPU-b22d9465-8552-9f97-812e-380b590645f8
JANELA=${JANELA_SH:-$AQUI/janela-e4b.sh}
LIMITE_SH=${LIMITE_SH:-$AQUI/roda-com-limite.sh}
LASTRO_SH=${LASTRO_SH_CMP:-$AQUI/roda-cmp-lastro.sh}
MONITOR=${MONITOR_SH:-$AQUI/roda-cmp-monitor.sh}
EXTRAI=${EXTRAI_PY:-$AQUI/cmp-extrai.py}
PERFIL_EXTRAI=${PERFIL_EXTRAI_PY:-$AQUI/perfil-extrai.py}
INSTR_EXTRAI=${INSTR_EXTRAI_PY:-$AQUI/instrumento-extrai.py}
BIN=$AQUI/../../.herdr/binarios/primeira-palavra-instr-ae14cbde
M1=${MODELO_1B:-/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct}
M3=${MODELO_3B:-/home/brenoperucchi/models/granite-3.1-3b-a800m-instruct}
TEXTOS=${TEXTOS_5C:-$AQUI/5c-textos}
REFS=${REF_5C:-$AQUI/5c-referencia}
IDS=${IDS_CMP:-$AQUI/cmp-ids-prompt.json}
CRIT_INSTR=${CRITERIO_INSTR:-$AQUI/criterio-instrumento-prefill-20261110.md}
PLANO=${PLANO_INSTR:-$AQUI/plano-instrumento-prefill-20261110.md}
CRIT=${CRITERIO_CMP:-$AQUI/criterio-comparacao-llamacpp-20261108.md}
CRIT_D1=${CRITERIO_D1:-$AQUI/criterio-degrau1-100pct-20261109.md}
CRIT_D2=${CRITERIO_D2:-$AQUI/criterio-degrau2-camadas-20261109.md}
FOLGA=${CMP_FOLGA:-180}
LIMITE_MAX=1200
ORDEM_CTRL=DLLDDL
SAIDA=; ESTADO=; LIMITE=$LIMITE_MAX; INTERNO=0; MODELOS=; VERIFICA=0
usa() { echo "uso: roda-instrumento-janela.sh --modelos 1b|3b --saida DIR [--estado-janela DIR] [--bin PRIMEIRA_PALAVRA_INSTR] [--limite S] | --verifica-fixos" >&2; }
recusa() { echo "roda-instrumento-janela: $*" >&2; exit 2; }
# Potência (Emenda 2 do perfil, Emenda 10 do cmp): le_pot lê só o power.limit da 3080 Ti (vazio se a leitura falhar); 370.00 W no preflight (senão recusa 2) e de novo ao fim (senão integridade QUEBRADA)
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

# SHAs congelados. O próprio roda-instrumento-janela.sh NÃO pode conter o seu sha (nem o critério o do roteiro: o roteiro guarda o sha do critério): --verifica-fixos imprime o do roteiro à parte (linha informativa).
declare -A FIXO=(
  [criterio-instrumento]=4d7fd8a9a647334625700fbc3f1ba98a63abfd46bc4f9fb238659e72cb13da9e
  [plano-instrumento]=39fbab0335ed40f9c75b19815db5194c09ae6e5aa8566e8b13977a97eaf7f8a4
  [binario-instr]=ae14cbde3965493808818f3aa4a8e7665b6397531e36c2abf020c35fc4f3d969
  [script-instrumento-extrai]=c88bb8345ae34322f264fa578102cb4422378547a83500d13633c0d10611799d
  [script-perfil-extrai]=755e1d7dd622e98dce9513ef30584e21667a5bc9d8c08cce4198090b2c34ba68
  [script-cmp-extrai]=22d32816ca9b298eb1ca63b3079ea9a6c681519224726b285153bec697f58865
  [script-lastro]=8e924a36fff414f337857ffc75fc88c5b23e4cf12c3d6314384f3979cab80757
  [script-monitor]=3252c690ea38c447ec7dda24bbbe4ca801f3daefb93e63009564df3c62eac017
  [script-janela]=a168e6672ba87429b8c5ccc0c80ba7daa36335f6a911b91a3744454cf7e63d60
  [script-limite]=196c05649b43e64361766e98e5bab9873636799dfa5d83f70a468436cd4759cc
  [criterio-cmp]=954ec594ec51d1ccff3239c8dba33f86a01363b82330181b2a9ced918abdcd5f
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
  add criterio-instrumento "$CRIT_INSTR"; add plano-instrumento "$PLANO"
  add binario-instr "$BIN"
  add script-instrumento-extrai "$INSTR_EXTRAI"; add script-perfil-extrai "$PERFIL_EXTRAI"; add script-cmp-extrai "$EXTRAI"; add script-lastro "$LASTRO_SH"; add script-monitor "$MONITOR"
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
  local tmp="$SAIDA/.shas-$(basename "$1")" i
  mkdir -p "$tmp"
  for i in "${!CHAVES[@]}"; do ( sha "${CAMINHOS[$i]}" > "$tmp/$i" ) & done
  wait
  : > "$1"
  for i in "${!CHAVES[@]}"; do echo "${CHAVES[$i]} $(cat "$tmp/$i" 2>/dev/null)" >> "$1"; done   # os temporários ficam em DIR/.shas-* (nada é removido por este roteiro)
}
# Processos que recusam a janela (incidente do rev-2 da rodada 137): `cargo test`/`cargo nextest`/`cargo bench` (o subcomando é a primeira palavra depois de `cargo` que não começa com - nem +), binário de teste do
# cargo (target/<perfil>/deps/<nome>-<hash de 16 hex>), primeira-palavra, llama.*. Lista = `ps -eo pid=,args=` (ou o arquivo PROC_LISTA, só em teste: linhas "PID ARGS"); este script e os ancestrais dele ficam de fora.
# Imprime uma linha por processo encontrado; código 0 = nenhum.
cargo_teste() { # "args": 0 se a linha de comando tem `cargo <opções> test|t|nextest|bench`
  local w base achou=0
  for w in $1; do
    base=${w##*/}; base=${base//[\"\']/}
    if [ $achou = 1 ]; then
      case $w in -*|+*) continue;; esac
      case $w in test|t|nextest|bench) return 0;; esac
      achou=0
    fi
    case $base in cargo) achou=1;; cargo-nextest) return 0;; esac
  done
  return 1
}
# Códigos: 0 = nenhum processo proibido, 1 = achou (uma linha por processo), 2 = falha ao ler a lista de processos (recusa com motivo; nunca vira "lista vazia").
# O executável de teste do cargo é reconhecido por `deps/<nome>-<hash de 16 hex>` em QUALQUER raiz de alvo (padrão, CARGO_TARGET_DIR personalizado, relativo).
proc_proibidos() {
  local lista pid args exe base achou=0 anc=" $$ " p rc_l=0
  p=$$; while [ "$p" -gt 1 ] 2> /dev/null; do p=$(awk '{print $4}' "/proc/$p/stat" 2> /dev/null) || break; [ -n "$p" ] || break; anc+="$p "; done
  if [ -n "${PROC_LISTA:-}" ]; then lista=$(cat "$PROC_LISTA" 2> /dev/null); rc_l=$?; else lista=$(ps -eo pid=,args= 2> /dev/null); rc_l=$?; fi
  if [ $rc_l != 0 ]; then echo "falha ao ler a lista de processos (${PROC_LISTA:-ps}, código $rc_l)"; return 2; fi
  while read -r pid args; do
    [ -n "$pid" ] || continue
    case $anc in *" $pid "*) continue;; esac
    exe=${args%% *}; base=${exe##*/}
    if cargo_teste "$args"; then echo "$pid ($args)"; achou=1; continue; fi
    if [[ $args =~ (^|[/\ ])deps/[A-Za-z0-9_]+-[0-9a-f]{16}( |$) ]]; then echo "$pid ($args)"; achou=1; continue; fi
    case $base in primeira-palavra*|llama-server*|llama-bench*|llama-cli*) echo "$pid ($args)"; achou=1;; esac
  done <<< "$lista"
  [ $achou = 0 ]
}
# portão do binário: o executável escolhido (--bin incluso) tem de ter o sha do FIXO ANTES de qualquer envelope, smoke ou célula (SEM_SHA_FIXO=1, só em teste, dispensa)
confere_binario() { [ "${SEM_SHA_FIXO:-0}" = 1 ] || [ "$(sha "$BIN")" = "${FIXO[binario-instr]}" ]; }
# estimativas (palpites) de tempo: est NOME PADRAO
est() { local v=CMP_EST_$1; echo "${!v:-$2}"; }

# --verifica-fixos: só lê (sem GPU, sem autorização, sem tocar em timers). AUSENTE = arquivo que não existe aqui e conta como falha (código 1).
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
  echo "informativo: roda-instrumento-janela.sh sha256 $(sha "$AQUI/roda-instrumento-janela.sh") (autorreferente, fora do FIXO; conferir por fora)"
  exit $rc_v
fi
# --modelos: uma janela por modelo; ausente ou inválido = recusa 2 (nunca as duas de uma vez)
case $MODELOS in
  1b|3b) ;;
  '') usa; recusa "--modelos 1b|3b é obrigatório: uma janela por modelo";;
  *) recusa "--modelos inválido: '$MODELOS' (use 1b ou 3b, nunca os dois)";;
esac
[ -n "$SAIDA" ] || { usa; exit 2; }
case $MODELOS in 1b) NCPU=22;; 3b) NCPU=29;; esac
E_PORTAO=$(est PORTAO 8)
declare -A E_CARGA=([1b]=$(est CARGA_1B 5) [3b]=$(est CARGA_3B 7))
declare -A E_QUENTE=([1b]=$(est QUENTE_1B 5) [3b]=$(est QUENTE_3B 9))
declare -A E_SMOKE=([1b]=$(est SMOKE_1B 20) [3b]=$(est SMOKE_3B 30))
est_celula() { echo $(( E_PORTAO + E_CARGA[$MODELOS] + ($1 + 1) * E_QUENTE[$MODELOS] )); }   # est_celula N (N medidas; o aquecimento vem a mais)
# bloco (2) do plano: smoke L100 + 12 processos de controle (2 séries x 6) com --repete 3, mais a folga de encerramento
bloco_par_s() { echo $(( E_SMOKE[$MODELOS] + 12 * $(est_celula 3) + FOLGA )); }

# ---------------------------------------------------------------- modo interno (dentro da janela, com o portão e o UUID no ambiente)
if [ $INTERNO = 1 ]; then
  [ "${PRIMEIRA_PALAVRA_AUTORIZADA:-}" = 1 ] && [ "${CUDA_VISIBLE_DEVICES:-}" = "$UUID" ] || recusa "modo interno sem portão ou UUID"
  PROIB_I=$(proc_proibidos) || recusa "cargo test, processo CUDA em execução ou lista de processos ilegível (modo interno): $(echo "$PROIB_I" | head -n 3 | tr '\n' ' ')"
  confere_binario || recusa "binário com sha diferente do FIXO (${FIXO[binario-instr]}): $BIN (modo interno)"
  mkdir -p "$SAIDA/celulas" || exit 2
  # a variável do instrumento NÃO é limpa aqui: cada processo recebe o seu ambiente explícito (D: env -u LASTRO_PERFIL_BLOCO; L: LASTRO_PERFIL_BLOCO=1; smoke: -u); o que o ambiente de quem chama trouxer não vale
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
    "$SMI" -i "$UUID" --query-gpu=power.limit,pcie.link.width.current,pcie.link.gen.max --format=csv,noheader 2>&1 | head -n 2 | sed 's/^/gpu_limites=/'
    POT_INI=$(le_pot) || POT_INI=; echo "potencia_inicio=$(pot_tok "$POT_INI")"
    echo "loadavg=$(cut -d' ' -f1-3 < /proc/loadavg 2>/dev/null)"
    if [ "${CMP_SEM_MEMORIA:-0}" != 1 ]; then
      mts=$(sudo -n dmidecode -t memory 2>/dev/null | awk -F': ' '/Configured Memory Speed/ && $2 ~ /[0-9]/ { split($2, a, " "); print a[1]; exit }')
      [ -z "$mts" ] || echo "memoria_mts=$mts"
    fi
  } > "$SAIDA/ambiente.txt" 2>&1
  POT_INI=${POT_INI:-}
  REPS_Q=5; CORTES=(); NAO_RODOU=()
  : > "$SAIDA/ordem.txt"; : > "$SAIDA/orcamento-log.txt"; : > "$SAIDA/prova-variavel.txt"; : > "$SAIDA/controle-ordem.txt"
  OBRIG=("L10:curto" "L10:longo" "L100:curto" "L100:longo"); OPC=("LC:curto" "LC:medio" "LC:longo")
  {
    echo "plano da janela do instrumento, modelo $MODELOS (ordem fixa; só Lastro, quentes, binário instrumentado; sem repetição automática)"
    echo "0 smoke L100 (CMP_FRACAO_SLOTS=1.0, sem a variável do instrumento)"
    for s in L10 L100; do
      i=0; for b in $(echo "$ORDEM_CTRL" | sed 's/./& /g'); do i=$((i + 1)); printf '%s ctrl-%s-%s-medio-%02d-%s (--repete 3; %s)\n' "controle" "$MODELOS" "$s" "$i" "$b" "$([ "$b" = D ] && echo 'env -u LASTRO_PERFIL_BLOCO' || echo 'LASTRO_PERFIL_BLOCO=1')"; done
    done
    for c in "${OBRIG[@]}"; do echo "obrigatória lastro-$MODELOS-${c%%:*}-${c##*:}-quente (LASTRO_PERFIL_BLOCO=1; 5 medidas, corte único para 3)"; done
    echo "opcional: smoke LC (CMP_CAMADAS_CPU=$NCPU), depois"
    for c in "${OPC[@]}"; do echo "opcional lastro-$MODELOS-${c%%:*}-${c##*:}-quente (LASTRO_PERFIL_BLOCO=1)"; done
  } > "$SAIDA/plano.txt"
  SEQ=0; ABORTA=0
  nr() { NAO_RODOU+=("$1"); echo "NÃO RODOU: $1" >> "$SAIDA/orcamento-log.txt"; }
  # ambiente de UM processo: série (L10|L100|LC) e braço (D = sem a variável; L = ligada; S = smoke, sem a variável)
  # (GNU env só lê -u ANTES das atribuições NOME=valor: os -u vêm primeiro)
  env_proc() {
    local u=() a=()
    case $1 in
      L10) u+=(-u CMP_FRACAO_SLOTS -u CMP_CAMADAS_CPU);;
      L100) u+=(-u CMP_CAMADAS_CPU); a+=(CMP_FRACAO_SLOTS=1.0);;
      LC) u+=(-u CMP_FRACAO_SLOTS); a+=(CMP_CAMADAS_CPU=$NCPU);;
    esac
    case $2 in
      L) a+=(LASTRO_PERFIL_BLOCO=1);;
      *) u+=(-u LASTRO_PERFIL_BLOCO);;
    esac
    ENVX=("${u[@]}" "${a[@]}")
  }
  # roda_processo CHAVE SERIE TEXTO N BRACO: UMA chamada ao roda-cmp-lastro.sh (processo único com --repete N); registra a ordem e a prova da variável. Devolve o rc.
  roda_processo() {
    local chave=$1 s=$2 t=$3 n=$4 b=$5 t0 rc linha dir
    SEQ=$((SEQ + 1)); t0=$(decorrido); dir="$SAIDA/celulas/$chave"
    env_proc "$s" "$b"
    env "${ENVX[@]}" "$LASTRO_SH" --modelo "$MODELOS" --texto "$t" --estado quente --n "$n" --saida "$dir" --bin "$BIN" > "$SAIDA/celulas-$chave.log" 2>&1
    rc=$?
    printf '%02d %s n=%s braco=%s inicio=%ss fim=%ss rc=%s\n' "$SEQ" "$chave" "$n" "$b" "$t0" "$(decorrido)" "$rc" >> "$SAIDA/ordem.txt"
    linha=$(grep -h '^perfil-bloco: ' "$dir"/p*/stdout.txt 2> /dev/null | sort -u | tr '\n' '|' | sed 's/|$//')
    printf '%s braco=%s env="%s" linha_do_binario="%s"%s\n' "$chave" "$b" "${ENVX[*]}" "${linha:-ausente}" \
      "$( { [ "$b" = D ] && [ "$linha" != "perfil-bloco: desligado" ]; } || { [ "$b" = L ] && [ "$linha" != "perfil-bloco: ligado" ]; } && echo ' INCOERENTE' || true)" >> "$SAIDA/prova-variavel.txt"
    [ "$rc" != 2 ] || { ABORTA=1; echo "processo $chave recusou o ambiente (código 2): janela interrompida" >> "$SAIDA/orcamento-log.txt"; }
    return $rc
  }
  # ---- smoke POR SÉRIE (L100 antes dos pares; LC antes das opcionais): falha com código 0 ou 1 marca SÓ aquela série como SEM PROVA; código 2 interrompe a janela
  declare -A SERIE_FALHA=()
  roda_smoke() { # série
    local s=$1 dir="$SAIDA/smoke-$1" t0 rc_s rc_ev smk=() msg mot
    env_proc "$s" S
    if [ "$s" = LC ]; then smk=(--degrau2); msg="SMOKE DE CAMADAS-CPU FALHOU"; else msg="L100 NÃO SUSTENTA 100%"; fi
    t0=$(decorrido)
    env "${ENVX[@]}" "$LASTRO_SH" --smoke --modelo "$MODELOS" --saida "$dir" --bin "$BIN" > "$SAIDA/smoke-$s.log" 2>&1; rc_s=$?
    printf 'smoke-%s inicio=%ss fim=%ss rc=%s\n' "$s" "$t0" "$(decorrido)" "$rc_s" >> "$SAIDA/ordem.txt"
    [ "$rc_s" != 2 ] || { ABORTA=1; echo "smoke $s recusou o ambiente (código 2): janela interrompida" >> "$SAIDA/orcamento-log.txt"; }
    mkdir -p "$dir"
    python3 -I "$EXTRAI" --smoke "$dir" --modelo "$MODELOS" "${smk[@]}" --ids "$IDS" --refs "$REFS" --textos "$TEXTOS/textos.json" > "$SAIDA/smoke-$s-veredito.out" 2>&1; rc_ev=$?
    if [ $rc_ev = 2 ]; then printf 'veredito=FALHA\nmotivo=smoke não avaliável (extrator recusou: pasta ou modelo)\n' > "$dir/veredito.txt"; fi
    if ! grep -q '^veredito=OK$' "$dir/veredito.txt" 2> /dev/null; then
      mot=$(grep '^motivo=' "$dir/veredito.txt" 2> /dev/null | head -n 1 | cut -d= -f2-)
      [ -n "$mot" ] || mot="smoke sem veredito legível"
      SERIE_FALHA[$s]="$msg ($s): $mot"
      echo "SEM PROVA, série $s: ${SERIE_FALHA[$s]} (a janela segue com as outras séries)" >> "$SAIDA/orcamento-log.txt"
    fi
  }
  # ---- (1) smoke L100 (o preflight já foi feito pelo modo externo)
  roda_smoke L100
  # ---- (2) pares de controle: bloco reservado; um processo por vez, na ordem fixa; o 1º com erro encerra o controle DA SÉRIE (sem repetição) e o resto vira NÃO RODOU
  for s in L10 L100; do
    [ $ABORTA = 0 ] || { for nn in 1 2 3 4 5 6; do nr "ctrl-$MODELOS-$s-medio-$(printf '%02d' "$nn")-${ORDEM_CTRL:$((nn - 1)):1}: janela interrompida por recusa de ambiente (código 2)"; done; continue; }
    if [ -n "${SERIE_FALHA[$s]:-}" ]; then
      for nn in 1 2 3 4 5 6; do nr "ctrl-$MODELOS-$s-medio-$(printf '%02d' "$nn")-${ORDEM_CTRL:$((nn - 1)):1}: série $s SEM PROVA (smoke)"; done
      continue
    fi
    PAROU=
    for nn in 1 2 3 4 5 6; do
      b=${ORDEM_CTRL:$((nn - 1)):1}; chave="ctrl-$MODELOS-$s-medio-$(printf '%02d' "$nn")-$b"
      if [ $ABORTA != 0 ]; then nr "$chave: janela interrompida por recusa de ambiente (código 2)"; continue; fi
      if [ -n "$PAROU" ]; then nr "$chave: controle da série $s encerrado ($PAROU), sem repetição"; continue; fi
      if [ $(( $(decorrido) + $(est_celula 3) + FOLGA )) -gt "$LIMITE" ]; then PAROU="sem tempo"; nr "$chave: não coube no limite ($LIMITE s)"; continue; fi
      echo "$(printf '%02d' "$nn") $b $chave" >> "$SAIDA/controle-ordem.txt"
      roda_processo "$chave" "$s" medio 3 "$b"; rc=$?
      [ $rc = 0 ] || { [ $ABORTA != 0 ] || PAROU="processo $chave saiu com código $rc"; }
    done
  done
  # ---- (3) células obrigatórias (L10 e L100, curto e longo), 1 processo de 1 aquecimento + REPS_Q medidas; único corte: 5 -> 3 nas células ainda não iniciadas
  projeta_obrig() { local i tot=0 c; for ((i = $1; i < ${#OBRIG[@]}; i++)); do c=${OBRIG[$i]}; [ -z "${SERIE_FALHA[${c%%:*}]:-}" ] || continue; tot=$(( tot + $(est_celula "$REPS_Q") )); done; echo "$tot"; }
  for ((ci = 0; ci < ${#OBRIG[@]}; ci++)); do
    s=${OBRIG[$ci]%%:*}; t=${OBRIG[$ci]##*:}; chave="lastro-$MODELOS-$s-$t-quente"
    if [ $ABORTA != 0 ]; then nr "$chave: janela interrompida por recusa de ambiente (código 2)"; continue; fi
    [ -z "${SERIE_FALHA[$s]:-}" ] || continue
    if [ $REPS_Q = 5 ] && [ $(( $(decorrido) + $(projeta_obrig "$ci") + FOLGA )) -gt "$LIMITE" ]; then
      REPS_Q=3; CORTES+=("quente cai de 5 para 3 medidas (nas células ainda não iniciadas)")
      echo "corte em $(decorrido) s: ${CORTES[-1]}" >> "$SAIDA/orcamento-log.txt"
    fi
    if [ $(( $(decorrido) + $(est_celula "$REPS_Q") + FOLGA )) -gt "$LIMITE" ]; then nr "$chave: não cabe, sem mais cortes"; continue; fi
    roda_processo "$chave" "$s" "$t" "$REPS_Q" L || true
  done
  # ---- (4) opcionais (LC): smoke e células só se couberem; sem controle, descritivas; nunca encolhem o bloco (2)
  LC_OK=1
  if [ $ABORTA != 0 ]; then LC_OK=0; for c in "${OPC[@]}"; do nr "lastro-$MODELOS-${c%%:*}-${c##*:}-quente: janela interrompida por recusa de ambiente (código 2)"; done
  elif [ $(( $(decorrido) + E_SMOKE[$MODELOS] + $(est_celula "$REPS_Q") + FOLGA )) -gt "$LIMITE" ]; then
    LC_OK=0; for c in "${OPC[@]}"; do nr "lastro-$MODELOS-${c%%:*}-${c##*:}-quente: opcional sem orçamento (nem o smoke LC cabe)"; done
  else
    roda_smoke LC
    [ -z "${SERIE_FALHA[LC]:-}" ] || LC_OK=0
  fi
  if [ $LC_OK = 1 ]; then
    for c in "${OPC[@]}"; do
      s=${c%%:*}; t=${c##*:}; chave="lastro-$MODELOS-$s-$t-quente"
      if [ $ABORTA != 0 ]; then nr "$chave: janela interrompida por recusa de ambiente (código 2)"; continue; fi
      if [ $(( $(decorrido) + $(est_celula "$REPS_Q") + FOLGA )) -gt "$LIMITE" ]; then nr "$chave: opcional não cabe"; continue; fi
      roda_processo "$chave" "$s" "$t" "$REPS_Q" L || true
    done
  fi
  [ $ABORTA = 0 ] || nr "janela interrompida por recusa de ambiente (código 2) no processo ${SEQ}"

  # ---- integridade depois
  coleta_shas "$SAIDA/integridade-depois.txt"
  POT_FIM=$(le_pot) || POT_FIM=; echo "potencia_fim=$(pot_tok "$POT_FIM")" >> "$SAIDA/ambiente.txt"
  INTEGRO=1; NOTAS=()
  if ! cmp -s "$SAIDA/integridade-antes.txt" "$SAIDA/integridade-depois.txt"; then INTEGRO=0; NOTAS+=("sha mudou entre antes e depois"); fi
  : > "$SAIDA/.integridade-fixos.tmp"
  while read -r k h; do
    [ -n "$h" ] || { INTEGRO=0; NOTAS+=("$k ilegível"); echo "$k - - FALHA ilegível" >> "$SAIDA/.integridade-fixos.tmp"; continue; }
    if [ -n "${FIXO[$k]:-}" ]; then
      if [ "${SEM_SHA_FIXO:-0}" = 1 ]; then echo "$k ${FIXO[$k]} $h NAO-CONFERIDO" >> "$SAIDA/.integridade-fixos.tmp"
      elif [ "$h" = "${FIXO[$k]}" ]; then echo "$k ${FIXO[$k]} $h OK" >> "$SAIDA/.integridade-fixos.tmp"
      else INTEGRO=0; NOTAS+=("$k com sha $h, não o congelado"); echo "$k ${FIXO[$k]} $h FALHA" >> "$SAIDA/.integridade-fixos.tmp"; fi
    fi
  done < "$SAIDA/integridade-antes.txt"
  if [ "$POT_INI" = "$POT_OK" ] && [ "$POT_FIM" = "$POT_OK" ]; then echo "potencia ${POT_OK// /_} $(pot_tok "$POT_FIM") OK" >> "$SAIDA/.integridade-fixos.tmp"
  else INTEGRO=0; NOTAS+=("power.limit fora de $POT_OK ou ilegível (início '${POT_INI:-ilegível}', fim '${POT_FIM:-ilegível}')"); echo "potencia inicio=$(pot_tok "$POT_INI") fim=$(pot_tok "$POT_FIM") FALHA" >> "$SAIDA/.integridade-fixos.tmp"; fi
  mv "$SAIDA/.integridade-fixos.tmp" "$SAIDA/integridade-fixos.txt"
  # ---- orçamento declarado (nao_rodou traz a chave da pasta de cada processo que não rodou: o extrator usa o prefixo ctrl-<modelo>-<série>- para marcar o controle incompleto)
  {
    printf '{"modelos": ["%s"], "limite_s": %s, "folga_s": %s, "decorrido_s": %s, "cortes": [' "$MODELOS" "$LIMITE" "$FOLGA" "$(decorrido)"
    sep=; for c in "${CORTES[@]:-}"; do [ -n "$c" ] || continue; printf '%s"%s"' "$sep" "$c"; sep=', '; done
    printf '], "reps": {"quente_nominal": 5, "controle": 3}, "reps_finais": {"quente": %s}, "nao_rodou": [' "$REPS_Q"
    sep=; for c in "${NAO_RODOU[@]:-}"; do [ -n "$c" ] || continue; printf '%s"%s"' "$sep" "$(echo "$c" | tr -d '"\\')"; sep=', '; done
    printf '], "estimativas_sao_palpites": true}\n'
  } > "$SAIDA/orcamento.json"
  python3 -I "$INSTR_EXTRAI" --saida "$SAIDA" --ids "$IDS" --refs "$REFS" --textos "$TEXTOS/textos.json" > "$SAIDA/instrumento-extrai.out" 2>&1; RC_EXT=$?
  NINC=$(grep -c ' INCOERENTE$' "$SAIDA/prova-variavel.txt" 2> /dev/null || true)
  {
    echo "instrumento de prefill, janela do modelo $MODELOS, critério sha256 $(sha "$CRIT_INSTR"), roteiro sha256 $(sha "$AQUI/roda-instrumento-janela.sh")"
    echo "resultado DESCRITIVO: sem meta, sem PASSA/REPROVA, sem veredito geral; H1 a H6 são previsões pré-registradas conferidas; H7 não é avaliada; só Lastro, binário instrumentado sha256 $(sha "$BIN")"
    for s in L100 LC; do [ -z "${SERIE_FALHA[$s]:-}" ] || echo "SEM PROVA (série $s): ${SERIE_FALHA[$s]}"; done
    echo "variável do instrumento: ${NINC:-0} processo(s) INCOERENTE(S) em prova-variavel.txt (D tem de imprimir desligado e L ligado)"
    if [ $INTEGRO = 1 ]; then echo "integridade: íntegra (binário, scripts, GGUF/safetensors do modelo, textos, referências, ids e critérios iguais antes e depois$([ "${SEM_SHA_FIXO:-0}" = 1 ] && echo '; congelados NÃO conferidos: SEM_SHA_FIXO=1' || echo ' e aos congelados onde há'))"
    else echo "integridade: QUEBRADA: tudo SEM PROVA (e o extrator recusa sem a prova de potência)"; for n in "${NOTAS[@]}"; do echo "  $n"; done; fi
    echo "cortes de orçamento: ${CORTES[*]:-nenhum}"
    echo "repetições finais: quente $REPS_Q; decorrido $(decorrido) s de $LIMITE s"
    echo "não rodou: ${#NAO_RODOU[@]} item(ns); ver orcamento.json"
    echo "extrator: $(cat "$SAIDA/instrumento-extrai.out") (código $RC_EXT)"
    echo "processos executados: $SEQ (ordem.txt; plano em plano.txt; prova da variável em prova-variavel.txt)"
  } > "$SAIDA/decisao.txt"
  cat "$SAIDA/decisao.txt"
  exit 0
fi

# ---------------------------------------------------------------- modo externo (portões, janela e limite)
case $LIMITE in ''|*[!0-9]*) recusa "limite inválido: $LIMITE";; esac
[ "$LIMITE" -ge 1 ] && [ "$LIMITE" -le $LIMITE_MAX ] || recusa "limite fora de 1..$LIMITE_MAX s: $LIMITE"
[ "${PRIMEIRA_PALAVRA_AUTORIZADA:-}" = 1 ] || recusa "portão fechado: sem PRIMEIRA_PALAVRA_AUTORIZADA=1 (o 'autorizo' do Breno que nomeia a janela)"
[ "${CUDA_VISIBLE_DEVICES:-}" = "$UUID" ] || recusa "CUDA_VISIBLE_DEVICES tem de ser $UUID (a 3080 Ti; a 4090 e a 5090 nunca); veio '${CUDA_VISIBLE_DEVICES:-}'"
# variáveis só de teste no ambiente do modo real = recusa 2; só CMP_MODO_TESTE=1 as aceita. `compgen -e` lista só as EXPORTADAS. LASTRO_PERFIL_BLOCO está na lista: a variável liga só por processo L, nunca herdada.
if [ "${CMP_MODO_TESTE:-0}" != 1 ]; then
  for v in $(compgen -e); do
    case $v in
      LASTRO_PERFIL_BLOCO|CMP_FRACAO_SLOTS|CMP_CAMADAS_CPU|CRITERIO_INSTR|PLANO_INSTR|CRITERIO_D1|CRITERIO_D2|SEM_SHA_FIXO|MONITOR_SH|LASTRO_SH_CMP|EXTRAI_PY|PERFIL_EXTRAI_PY|INSTR_EXTRAI_PY|CMP_RELOGIO_ARQ|NVIDIA_SMI|MODELO_1B|MODELO_3B|TEXTOS_5C|REF_5C|IDS_CMP|CRITERIO_CMP \
      |PROC_STAT|CMP_CPU_JANELA|CMP_CPU_ESPERA|CMP_FOLGA|CMP_SEM_MEMORIA|ESPERA_PRONTO|LOADAVG_ARQ|SYSTEMCTL|SYSTEMD_RUN|PENDENTE|CGROUP_RAIZ|PROC_LISTA \
      |CMP_EST_*|CMP_FAKE_*|JANELA_*|LIMITE_*)
        recusa "variável só de teste presente no modo real: $v (só CMP_MODO_TESTE=1, o modo de teste explícito, a aceita); limpe o ambiente de quem chama";;
    esac
  done
fi
[ -x "$BIN" ] || recusa "binário não executável: $BIN"
[ -x "$JANELA" ] && [ -x "$LIMITE_SH" ] && [ -x "$LASTRO_SH" ] && [ -x "$MONITOR" ] || recusa "faltam janela-e4b.sh, roda-com-limite.sh, roda-cmp-lastro.sh ou roda-cmp-monitor.sh executáveis"
[ -f "$EXTRAI" ] && [ -f "$PERFIL_EXTRAI" ] && [ -f "$INSTR_EXTRAI" ] || recusa "faltam cmp-extrai.py, perfil-extrai.py ou instrumento-extrai.py"
[ -f "$CRIT_INSTR" ] || recusa "falta o critério do instrumento: $CRIT_INSTR"
[ -n "$ESTADO" ] || ESTADO=$SAIDA-estado-janela
for d in "$SAIDA" "$ESTADO"; do
  if [ -e "$d" ]; then
    [ -d "$d" ] && [ -z "$(ls -A "$d")" ] || recusa "$d já existe e não está vazio: uma janela nova pede pasta nova"
  fi
done
SAIDA=$(realpath -m "$SAIDA"); BIN=$(realpath "$BIN")
# cargo test / processo CUDA em execução: recusa ANTES de tocar em timers ou na GPU (incidente do rev-2 da rodada 137)
confere_binario || recusa "binário com sha diferente do FIXO: $BIN tem sha256 $(sha "$BIN"), o congelado é ${FIXO[binario-instr]}; nada foi tocado (sem envelope, timers, smoke)"
PROIB=$(proc_proibidos) || recusa "processo que pode usar a 3080 Ti em execução (cargo test, binário de teste do cargo, primeira-palavra ou llama.*) ou lista de processos ilegível: feche antes da janela; nada foi tocado: $(echo "$PROIB" | head -n 3 | tr '\n' ' ')"
POT_PRE=$(le_pot) || POT_PRE=
[ "$POT_PRE" = "$POT_OK" ] || recusa "power.limit da 3080 Ti tem de ser $POT_OK (lido: '${POT_PRE:-ilegível}'); a série vale só no limite vigente; nada foi tocado, não mexa no limite por este roteiro"
PREFLIGHT=$("$MONITOR" preflight 2>&1) || recusa "processo(s) de compute na 3080 Ti que precisam ser FECHADOS antes da janela (voxtype, processo de GPU do Chrome...): $PREFLIGHT"
# bloco do par (2.1): não cabendo nos 20 min (ou no limite pedido), a janela NÃO COMEÇA: SEM PROVA da unidade inteira, registrada, sem repetição automática e sem encolher o par
NEC=$(bloco_par_s)
if [ "$NEC" -gt "$LIMITE" ]; then
  mkdir -p "$SAIDA" || exit 2
  { echo "SEM PROVA da unidade inteira: a janela do modelo $MODELOS NÃO COMEÇOU (o bloco obrigatório do par de controle não cabe)"
    echo "necessário (estimativa): $NEC s = smoke L100 ${E_SMOKE[$MODELOS]} + 12 processos x $(est_celula 3) + folga $FOLGA; limite pedido: $LIMITE s"
    echo "nada foi tocado (timers, GPU, voxtype); sem repetição automática; o par não é encolhido"; } > "$SAIDA/decisao.txt"
  cat "$SAIDA/decisao.txt" >&2
  exit 2
fi
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
for v in MODELO_1B MODELO_3B TEXTOS_5C REF_5C IDS_CMP CRITERIO_INSTR PLANO_INSTR CRITERIO_CMP CRITERIO_D1 CRITERIO_D2 SEM_SHA_FIXO NVIDIA_SMI PROC_STAT CMP_CPU_JANELA CMP_CPU_ESPERA LASTRO_SH_CMP EXTRAI_PY PERFIL_EXTRAI_PY INSTR_EXTRAI_PY \
         CMP_RELOGIO_ARQ CMP_FOLGA CMP_SEM_MEMORIA MONITOR_SH ESPERA_PRONTO LOADAVG_ARQ PROC_LISTA LASTRO_PERFIL_BLOCO $(compgen -v | grep -E '^CMP_(EST|FAKE)_' || true); do
  [ -z "${!v+x}" ] || PASSA+=("$v=${!v}")
done
lista_systemd() {
  "${SYSTEMCTL:-systemctl}" --user list-unit-files --type=service,timer > "$SAIDA.unidades-$1.txt" 2>&1 || true
  "${SYSTEMCTL:-systemctl}" --user list-timers --all > "$SAIDA.timers-$1.txt" 2>&1 || true
}
lista_systemd antes
coleta início "$SAIDA.carga-antes.txt"
"$JANELA" --estado "$ESTADO" "${TIMERS[@]}" --limite $((LIMITE + 120)) --kill-apos 30 -- \
  "$LIMITE_SH" --limite "$LIMITE" -- env "${PASSA[@]}" "$AQUI/roda-instrumento-janela.sh" --interno --modelos "$MODELOS" --saida "$SAIDA" --bin "$BIN" --limite "$LIMITE"
RC=$?
coleta fim "$SAIDA.carga-depois.txt"
lista_systemd depois
MSG="janela do instrumento de prefill do Lastro ($MODELOS) terminou (código $RC): $SAIDA"
echo "$MSG" > "$SAIDA.aviso.txt"
if [ -n "${CMP_AVISO_CMD:-}" ]; then "$CMP_AVISO_CMD" "$MSG" > /dev/null 2>&1 || true
elif command -v notify-send > /dev/null 2>&1; then notify-send "Instrumento de prefill (Lastro)" "$MSG" > /dev/null 2>&1 || true; fi
exit $RC
