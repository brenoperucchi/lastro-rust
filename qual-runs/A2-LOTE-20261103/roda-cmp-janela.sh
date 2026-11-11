#!/usr/bin/env bash
# Comparação Lastro x llama.cpp (opção d), UMA janela de até 30 min POR MODELO na 3080 Ti (criterio-comparacao-llamacpp-20261108.md, emendas 1 a 6; a mais nova vale sobre a anterior).
# Emenda 6: duas janelas, 1B depois 3B, mesmo critério e mesmos roteiros; --modelos 1b|3b é OBRIGATÓRIO (sem ele, recusa: nunca roda as duas de uma vez por engano).
# Composição já validada nas janelas 5c/5e: janela-e4b.sh (pausa e restauração conferida dos timers) por fora, roda-com-limite.sh (limite) por dentro, e este mesmo
# script em modo interno.
#
#   roda-cmp-janela.sh --modelos 1b|3b --saida DIR [--estado-janela DIR] [--bin PRIMEIRA_PALAVRA] [--limite S] [--sem-s3]
#   roda-cmp-janela.sh --degrau1 --modelos 1b|3b --saida DIR [...]   (degrau 1, criterio-degrau1-100pct-20261109.md: séries L100 e S1 SÓ, sem S2/S3; smoke antes das medidas)
#   roda-cmp-janela.sh --degrau2 --modelos 1b|3b --saida DIR [...]   (degrau 2, criterio-degrau2-camadas-20261109.md: séries LC e S2 SÓ, sem S1/S3; smoke antes das medidas)
#   roda-cmp-janela.sh --so-s3 --modelos 1b|3b --saida DIR [...]    (Emenda 12 do critério base: SÓ as 6 células da S3 do llama.cpp, janela própria; sem Lastro, sem S1/S2, sem smoke, sem razão)
#   roda-cmp-janela.sh --verifica-fixos     (sem GPU, sem autorização, só lê: confere os shas congelados do FIXO contra os arquivos reais; código 0 se todos OK)
#
# Antes da janela: systemctl --user stop voxtype; ao fim: systemctl --user start voxtype (passo manual, não automatizado; o voxtype em compute na 3080 Ti recusa a janela com código 2).
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
# MODO --degrau1 (seção 2 e 3 do critério do degrau 1): o Lastro roda com `--fracao-slots 1.0` (CMP_FRACAO_SLOTS=1.0 repassada ao roda-cmp-lastro.sh; série L100), contra a S1 do llama.cpp
# remedida NA MESMA JANELA; S2 e S3 não rodam e não aparecem como NÃO RODOU. Antes das células, dentro da janela do modelo, o SMOKE (`roda-cmp-lastro.sh --smoke`: uma geração curta, 64 tokens,
# sem medida) é julgado por `cmp-extrai.py --smoke` (código 0, sem erro de alocação, GPU <= 90% da memória, cpu/needed == 0; ids contra a referência 5c só reportados). Falha do smoke (exceto ids) PARA a
# janela do modelo, nenhuma razão é publicada e decisao.txt traz "L100 NÃO SUSTENTA 100%: <motivo>"; sem repetição automática. Orçamento: só os cortes (3) quente 5->3 e (4) frio 3->2; o que não couber é NÃO RODOU.
#
# MODO --degrau2: o Lastro roda com `--camadas-cpu N` (CMP_CAMADAS_CPU=22 no 1B, 29 no 3B, definida só aqui por dentro; série LC) contra a S2 do llama.cpp remedida NA MESMA JANELA; S1 e S3 não rodam e não
# aparecem como NÃO RODOU. Smoke antes das células, julgado por `cmp-extrai.py --smoke --degrau2`; falha do smoke PARA a janela do modelo ("SMOKE DO DEGRAU 2 FALHOU: <motivo>"), sem repetição.
# Orçamento: só os cortes (3) quente 5->3 e (4) frio 3->2; o que não couber é NÃO RODOU. --degrau1 e --degrau2 são exclusivos.
#
# MODO --so-s3 (Emenda 12): só a S3 do llama.cpp (KV f32, -fa off), 3 textos x frio/quente (6 células), na janela própria; o teto de 1800 s e a folga de 180 s ficam como estão. Sem células do Lastro, sem S1/S2 e sem smoke
# (o smoke é do Lastro). Orçamento: só os cortes (3) quente 5->3 e (4) frio 3->2; bloco que não couber é NÃO RODOU (nunca estimado). Sem Lastro na janela o extrator NÃO calcula razão: só medidas descritivas do llama.cpp.
# --so-s3 é exclusivo com --degrau1, --degrau2 e --sem-s3.
#
# Falha FECHADA antes de tocar em timers ou na GPU (emenda 9: também se houver PID C ou C+G na 3080 Ti; durante a janela o voxtype e o processo de GPU do Chrome ficam FECHADOS): PRIMEIRA_PALAVRA_AUTORIZADA=1 e CUDA_VISIBLE_DEVICES igual ao UUID da 3080 Ti (a 5090 NUNCA entra, nem no llama-server)
# têm de estar no ambiente de quem chama (o "autorizo" do Breno); --saida e --estado-janela têm de ser pastas novas ou vazias. Rede só em 127.0.0.1. Limite padrão 1800 s
# (máx. 1800); o de fora é o de dentro mais 120 s. Fim da janela: aviso (notify-send se houver, ou CMP_AVISO_CMD) e DIR.aviso.txt. Este roteiro NÃO é rodado na entrega.
# Variáveis só para teste (r111, rev-2): JANELA_SH, LIMITE_SH, JANELA_TIMERS, LASTRO_SH_CMP, LLAMA_SH_CMP, EXTRAI_PY, CMP_RELOGIO_ARQ (relógio virtual em segundos), CMP_FOLGA,
# CMP_EST_*, CMP_FAKE_*, MODELO_1B, MODELO_3B, GGUF_DIR, LLAMA_DIR, TEXTOS_5C, REF_5C, IDS_CMP, CRITERIO_CMP, SEM_SHA_FIXO=1, NVIDIA_SMI, CMP_SEM_MEMORIA=1, CMP_TESTE_AMBOS, PROC_STAT,
# CMP_CPU_JANELA, CMP_CPU_ESPERA, LLAMA_SERVER, CURL, MONITOR_SH, ESPERA_PRONTO, LOADAVG_ARQ, SYSTEMCTL, SYSTEMD_RUN, PENDENTE, JANELA_*, LIMITE_*, CGROUP_RAIZ. No modo real (envelope
# externo) qualquer uma presente no ambiente é RECUSADA com código 2; só CMP_MODO_TESTE=1 (modo de teste explícito) as aceita. CMP_AVISO_CMD continua permitida (só avisa).
# Códigos: 0 se decisao.txt foi gravada, o do comando (124 limite, ...) ou 2 recusa de uso, portão ou ambiente.
set -u
AQUI=$(cd "$(dirname "$0")" && pwd -P)
UUID=GPU-b22d9465-8552-9f97-812e-380b590645f8
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
CRIT_D1=${CRITERIO_D1:-$AQUI/criterio-degrau1-100pct-20261109.md}
CRIT_D2=${CRITERIO_D2:-$AQUI/criterio-degrau2-camadas-20261109.md}
FOLGA=${CMP_FOLGA:-180}
SAIDA=; ESTADO=; LIMITE=1800; INTERNO=0; COM_S3=1; MODELOS=; VERIFICA=0; DEGRAU1=0; DEGRAU2=0; SO_S3=0
usa() { echo "uso: roda-cmp-janela.sh [--degrau1|--degrau2|--so-s3] --modelos 1b|3b --saida DIR [--estado-janela DIR] [--bin PRIMEIRA_PALAVRA] [--limite S] [--sem-s3] | --verifica-fixos" >&2; }
recusa() { echo "roda-cmp-janela: $*" >&2; exit 2; }
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
    --sem-s3) COM_S3=0; shift;;
    --modelos) [ $# -ge 2 ] || { usa; exit 2; }; MODELOS=$2; shift 2;;
    --verifica-fixos) VERIFICA=1; shift;;
    --degrau1) DEGRAU1=1; shift;;
    --degrau2) DEGRAU2=1; shift;;
    --so-s3) SO_S3=1; shift;;
    --interno) INTERNO=1; shift;;
    *) usa; exit 2;;
  esac
done
[ $DEGRAU1 = 0 ] || [ $DEGRAU2 = 0 ] || { echo "roda-cmp-janela: --degrau1 e --degrau2 são exclusivos" >&2; exit 2; }
[ $SO_S3 = 0 ] || { [ $DEGRAU1 = 0 ] && [ $DEGRAU2 = 0 ] && [ $COM_S3 = 1 ]; } || { echo "roda-cmp-janela: --so-s3 é exclusivo com --degrau1, --degrau2 e --sem-s3" >&2; exit 2; }
sha() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }

# SHAs congelados (valem onde o critério já os fixa; binários, GGUF e safetensors só precisam ser iguais antes e depois).
declare -A FIXO=(
  [criterio-cmp]=8d9d6ad3ff1ef23d417e41982228ec55638823c613b4a2cbf00d813dc1f85b35
  [cmp-ids-prompt]=2a2613cdbaf9acdb76bd9aa416107474b3d888b6aa37b94b429546a5dca78d8b
  [texto-curto]=5b6fad6f3825d430bad4f924d227105032581906739d68c726ac8e4827df2d4d
  [texto-medio]=bb0a067ae8839616f33b21ba41b362bdf16353a57d70321252a2d47851f1c32a
  [texto-longo]=821e0f15139b58173b5c17f5e9428527bb27b03dc10b35e38cbeed044defbaea
  [textos-json]=5b16e34095e882b517a5b9dabb22e956e7d4cf688c0d2a6c4caf9c61abf99e5e
  [ref5c-1b]=9584fcabe7d8b2a1821207267651504c72d317ce9b566cd967d390a1dbbad259
  [ref5c-3b]=968d9b391324a99fbaf5c772a351018b90105f572b32ee2db3140f79da123124
  [tokenizer-1b]=ed944c0b3d71e2d651a5ba0f8e7b350faff30031c8e3353de70b12b9c1bf5654
  [tokenizer-3b]=ed944c0b3d71e2d651a5ba0f8e7b350faff30031c8e3353de70b12b9c1bf5654
  [criterio-degrau1]=8c3406a9fd3ba55133dfb2c2d3800076c6ee0a765735edf8be65f6ba1a72c7de
  [criterio-degrau2]=bd8cb0510d5a6c3f55ec320a3177e2bec7ebce7b5776529989f0c45a61599b2a
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
  # degrau 1: o critério novo entra na lista de integridade (e no --verifica-fixos, que confere tudo o que o FIXO congela)
  if [ "$DEGRAU1" = 1 ] || [ "$DEGRAU2" = 1 ] || [ "$VERIFICA" = 1 ]; then add criterio-degrau1 "$CRIT_D1"; fi
  if [ "$DEGRAU2" = 1 ] || [ "$VERIFICA" = 1 ]; then add criterio-degrau2 "$CRIT_D2"; fi
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
  # degrau 1: a fração vai por variável ao roda-cmp-lastro.sh (série L100); fora dele, nunca (a janela antiga roda a 10%)
  unset CMP_CAMADAS_CPU
  if [ $DEGRAU1 = 1 ]; then export CMP_FRACAO_SLOTS=1.0; SL=L100; COM_S3=0
  elif [ $SO_S3 = 1 ]; then unset CMP_FRACAO_SLOTS; SL=L; COM_S3=1   # SL não é usado: nenhuma célula do Lastro roda
  elif [ $DEGRAU2 = 1 ]; then
    unset CMP_FRACAO_SLOTS; SL=LC; COM_S3=0
    case $MODELOS in 1b) export CMP_CAMADAS_CPU=22;; 3b) export CMP_CAMADAS_CPU=29;; *) recusa "--degrau2 exige --modelos 1b ou 3b";; esac
  else unset CMP_FRACAO_SLOTS; SL=L; fi
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
    # Emenda 10/2: limite de potência e link PCIe da placa de medida, registrados (a série vale no limite vigente, hoje 370 W; se mudar, a série é invalidada)
    "${NVIDIA_SMI:-nvidia-smi}" -i "$UUID" --query-gpu=power.limit,pcie.link.width.current,pcie.link.gen.max --format=csv,noheader 2>&1 | head -n 2 | sed 's/^/gpu_limites=/'
    POT_INI=$(le_pot) || POT_INI=; echo "potencia_inicio=$(pot_tok "$POT_INI")"
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
  # degrau 1: sem S2 e sem S3 desde o início (não são cortes de orçamento, são fora do desenho); só valem os cortes (3) e (4)
  if [ $DEGRAU1 = 1 ]; then S2_1B=0; S2_3B=0; PROX_CORTE=4; fi
  # degrau 2: a S2 é a única série do llama.cpp (sem S1, sem S3); só valem os cortes (3) e (4)
  if [ $DEGRAU2 = 1 ]; then PROX_CORTE=4; fi
  # --so-s3: sem S1/S2 (fora do desenho) e a S3 é o objetivo, não um corte; só valem os cortes (3) e (4)
  if [ $SO_S3 = 1 ]; then S2_1B=0; S2_3B=0; PROX_CORTE=4; fi
  # blocos: "main:modelo:texto" ou "s3:modelo:texto"
  BLOCOS=()
  case $MODELOS in 1b) LISTA_M=(1b);; 3b) LISTA_M=(3b);; *) LISTA_M=(1b 3b);; esac
  [ $SO_S3 = 1 ] || for m in "${LISTA_M[@]}"; do for t in curto medio longo; do BLOCOS+=("main:$m:$t"); done; done
  [ $SO_S3 = 0 ] || for m in "${LISTA_M[@]}"; do for t in curto medio longo; do BLOCOS+=("s3:$m:$t"); done; done
  [ $SO_S3 = 1 ] || [ $DEGRAU1 = 1 ] || [ $DEGRAU2 = 1 ] || for m in "${LISTA_M[@]}"; do for t in curto medio longo; do BLOCOS+=("s3:$m:$t"); done; done
  est_bloco() { # tipo modelo texto -> segundos com a configuração ATUAL
    local tipo=$1 m=$2 tot=0 s
    if [ "$tipo" = main ]; then
      tot=$(( $(est_celula lastro L "$m" frio "$REPS_F") + $(est_celula lastro L "$m" quente "$REPS_Q") ))
      for s in S1 S2; do
        [ $DEGRAU2 = 1 ] && [ "$s" = S1 ] && continue
        [ "$s" = S2 ] && [ $DEGRAU2 = 0 ] && { [ "$m" = 1b ] && [ $S2_1B = 0 ] && continue; [ "$m" = 3b ] && [ $S2_3B = 0 ] && continue; }
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
  # ---- degrau 1: smoke ANTES das medidas (seção 3 do critério); falha (exceto ids) para a janela do modelo, sem repetição
  SMOKE_FALHA=
  if [ $DEGRAU1 = 1 ] || [ $DEGRAU2 = 1 ]; then
    if [ $DEGRAU2 = 1 ]; then SMK_ARGS=(--degrau2); SMK_MSG="SMOKE DO DEGRAU 2 FALHOU"; else SMK_ARGS=(); SMK_MSG="L100 NÃO SUSTENTA 100%"; fi
    t0_smoke=$(decorrido)
    "$LASTRO_SH" --smoke --modelo "$MODELOS" --saida "$SAIDA/smoke" --bin "$BIN" > "$SAIDA/smoke.log" 2>&1; rc_smoke=$?
    printf 'smoke inicio=%ss fim=%ss rc=%s\n' "$t0_smoke" "$(decorrido)" "$rc_smoke" >> "$SAIDA/ordem.txt"
    [ "$rc_smoke" != 2 ] || { ABORTA=1; echo "smoke recusou o ambiente (código 2): janela interrompida" >> "$SAIDA/orcamento-log.txt"; }
    mkdir -p "$SAIDA/smoke"
    python3 -I "$EXTRAI" --smoke "$SAIDA/smoke" --modelo "$MODELOS" "${SMK_ARGS[@]}" --ids "$IDS" --refs "$REFS" --textos "$TEXTOS/textos.json" > "$SAIDA/smoke-veredito.out" 2>&1; rc_ev=$?
    if [ $rc_ev = 2 ]; then printf 'veredito=FALHA\nmotivo=smoke não avaliável (extrator recusou: pasta ou modelo)\n' > "$SAIDA/smoke/veredito.txt"; fi
    if ! grep -q '^veredito=OK$' "$SAIDA/smoke/veredito.txt" 2> /dev/null; then
      SMOKE_FALHA=$(grep '^motivo=' "$SAIDA/smoke/veredito.txt" 2> /dev/null | head -n 1 | cut -d= -f2-)
      [ -n "$SMOKE_FALHA" ] || SMOKE_FALHA="smoke sem veredito legível"
      echo "$SMK_MSG: $SMOKE_FALHA" >> "$SAIDA/orcamento-log.txt"
      nr "janela do modelo $MODELOS inteira ($SMK_MSG: $SMOKE_FALHA)"
    fi
  fi
  for ((bi = 0; bi < ${#BLOCOS[@]}; bi++)); do
    [ $ABORTA = 0 ] || break
    [ -z "$SMOKE_FALHA" ] || break
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
        series=(S1)
        if [ $DEGRAU2 = 1 ]; then series=(S2)
        elif { [ "$m" = 1b ] && [ $S2_1B = 1 ]; } || { [ "$m" = 3b ] && [ $S2_3B = 1 ]; }; then series=(S1 S2); fi
        if [ $((bi % 2)) = 0 ]; then
          roda_celula lastro "$SL" "$m" "$t" "$estado" "$n"; for s in "${series[@]}"; do roda_celula llamacpp "$s" "$m" "$t" "$estado" "$n"; done
        else
          for s in "${series[@]}"; do roda_celula llamacpp "$s" "$m" "$t" "$estado" "$n"; done; roda_celula lastro "$SL" "$m" "$t" "$estado" "$n"
        fi
        [ $ABORTA = 0 ] || break
      done
      # S2 cortada: registra o que não rodou neste bloco
      for estado in frio quente; do
        [ $DEGRAU1 = 0 ] && [ $DEGRAU2 = 0 ] || break
        if [ "$m" = 1b ] && [ $S2_1B = 0 ]; then nr "llamacpp-$m-S2-$t-$estado"; fi
        if [ "$m" = 3b ] && [ $S2_3B = 0 ]; then nr "llamacpp-$m-S2-$t-$estado"; fi
      done
    elif [ $SO_S3 = 1 ]; then
      # --so-s3: corta em ordem (3) e (4) enquanto a projeção estourar; as repetições são as da configuração atual (não há Lastro a acompanhar)
      while [ $(( $(decorrido) + $(projeta_restante "$bi") + FOLGA )) -gt "$LIMITE" ]; do aplica_corte || break; done
      nf=$REPS_F; nq=$REPS_Q
      custo=$(( $(est_celula llamacpp S3 "$m" frio "$nf") + $(est_celula llamacpp S3 "$m" quente "$nq") ))
      if [ $(( $(decorrido) + custo + FOLGA )) -gt "$LIMITE" ]; then nr "llamacpp-$m-S3-$t (frio e quente): não cabe, sem mais cortes"; continue; fi
      roda_celula llamacpp S3 "$m" "$t" frio "$nf"; [ $ABORTA = 0 ] || break
      roda_celula llamacpp S3 "$m" "$t" quente "$nq"
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
  POT_FIM=$(le_pot) || POT_FIM=; echo "potencia_fim=$(pot_tok "$POT_FIM")" >> "$SAIDA/ambiente.txt"
  INTEGRO=1; NOTAS=()
  if ! cmp -s "$SAIDA/integridade-antes.txt" "$SAIDA/integridade-depois.txt"; then INTEGRO=0; NOTAS+=("sha mudou entre antes e depois"); fi
  # manifesto gravado em arquivo temporário e movido de uma vez: a interrupção antes do mv deixa o manifesto AUSENTE (cmp-extrai.py trata como SEM PROVA em tudo), nunca truncado
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
    printf '{"modelos": [%s], "limite_s": %s, "folga_s": %s, "decorrido_s": %s, "cortes": [' "$(printf '"%s", ' "${LISTA_M[@]}" | sed 's/, $//')" "$LIMITE" "$FOLGA" "$(decorrido)"
    sep=; for c in "${CORTES[@]:-}"; do [ -n "$c" ] || continue; printf '%s"%s"' "$sep" "$c"; sep=', '; done
    printf '], "degrau1": %s, "degrau2": %s, "so_s3": %s, "reps": {"quente_nominal": 5, "frio_nominal": 3}, "reps_finais": {"quente": %s, "frio": %s}, "s3_ativa_no_fim": %s, "nao_rodou": [' "$([ $DEGRAU1 = 1 ] && echo true || echo false)" "$([ $DEGRAU2 = 1 ] && echo true || echo false)" "$([ $SO_S3 = 1 ] && echo true || echo false)" "$REPS_Q" "$REPS_F" "$([ $S3 = 1 ] && echo true || echo false)"
    sep=; for c in "${NAO_RODOU[@]:-}"; do [ -n "$c" ] || continue; printf '%s"%s"' "$sep" "$(echo "$c" | tr -d '"\\')"; sep=', '; done
    printf '], "estimativas_sao_palpites": true}\n'
  } > "$SAIDA/orcamento.json"
  EXT_ARGS=(); [ $DEGRAU1 = 0 ] || EXT_ARGS=(--degrau1); [ $DEGRAU2 = 0 ] || EXT_ARGS=(--degrau2); [ $SO_S3 = 0 ] || EXT_ARGS=(--so-s3)
  python3 -I "$EXTRAI" --saida "$SAIDA" --ids "$IDS" --refs "$REFS" --textos "$TEXTOS/textos.json" "${EXT_ARGS[@]}" > "$SAIDA/cmp-extrai.out" 2>&1; RC_EXT=$?
  {
    if [ $SO_S3 = 1 ]; then
      echo "S3 sozinha do llama.cpp (--so-s3, Emenda 12), janela do modelo ${LISTA_M[*]}, critério base sha256 $(sha "$CRIT")"
      echo "resultado DESCRITIVO: sem meta, sem PASSA/REPROVA; SEM Lastro na janela, logo SEM razão Lastro/llama.cpp (o critério exige o Lastro na MESMA janela): só medidas do llama.cpp S3 (KV f32, -fa off)"
    elif [ $DEGRAU1 = 1 ]; then
      echo "degrau 1, Lastro x llama.cpp S1, janela do modelo ${LISTA_M[*]}, critério do degrau 1 sha256 $(sha "$CRIT_D1"), critério base sha256 $(sha "$CRIT")"
      echo "resultado DESCRITIVO: sem meta, sem PASSA/REPROVA de velocidade; todo rótulo: Lastro 100% experts GPU contra llama.cpp S1; KV f32 no Lastro contra f16 no llama.cpp"
      if [ -n "$SMOKE_FALHA" ]; then echo "L100 NÃO SUSTENTA 100%: $SMOKE_FALHA"; else echo "smoke: $(grep -E '^(veredito|ids|gpu_pico_pct|cpu_needed)=' "$SAIDA/smoke/veredito.txt" 2> /dev/null | tr '\n' ' ')"; fi
    elif [ $DEGRAU2 = 1 ]; then
      echo "degrau 2, Lastro camadas inteiras x llama.cpp S2, janela do modelo ${LISTA_M[*]}, critério do degrau 2 sha256 $(sha "$CRIT_D2"), critério do degrau 1 sha256 $(sha "$CRIT_D1"), critério base sha256 $(sha "$CRIT")"
      echo "resultado DESCRITIVO: sem meta, sem PASSA/REPROVA de velocidade; todo rótulo: Lastro camadas inteiras (${CMP_CAMADAS_CPU} na CPU) contra llama.cpp S2 (--n-cpu-moe ${CMP_CAMADAS_CPU}); KV f32 no Lastro contra f16 no llama.cpp; op offload ligado no llama.cpp, prefill da S2 não é leitura de experts na CPU"
      if [ -n "$SMOKE_FALHA" ]; then echo "SMOKE DO DEGRAU 2 FALHOU: $SMOKE_FALHA"; else echo "smoke: $(grep -E '^(veredito|ids|gpu_pico_pct|cpu_needed|camadas_cpu)=' "$SAIDA/smoke/veredito.txt" 2> /dev/null | tr '\n' ' ')"; fi
    else
      echo "comparação Lastro x llama.cpp (opção d), janela do modelo ${LISTA_M[*]}, critério sha256 $(sha "$CRIT")"
      echo "resultado DESCRITIVO: sem meta, sem PASSA/REPROVA de velocidade; toda razão é 'Lastro 10% experts GPU contra llama.cpp <série>'"
    fi
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
[ "${CUDA_VISIBLE_DEVICES:-}" = "$UUID" ] || recusa "CUDA_VISIBLE_DEVICES tem de ser $UUID (a 3080 Ti; a 4090 e a 5090 nunca); veio '${CUDA_VISIBLE_DEVICES:-}'"
# variáveis só de teste no ambiente do modo real = recusa 2 (r111, rev-2); só CMP_MODO_TESTE=1 as aceita. `compgen -e` lista só as EXPORTADAS (as variáveis internas do roteiro não contam).
if [ "${CMP_MODO_TESTE:-0}" != 1 ]; then
  for v in $(compgen -e); do
    case $v in
      CMP_FRACAO_SLOTS|CMP_CAMADAS_CPU|CRITERIO_D1|CRITERIO_D2|SEM_SHA_FIXO|LLAMA_SERVER|MONITOR_SH|LASTRO_SH_CMP|LLAMA_SH_CMP|EXTRAI_PY|CMP_RELOGIO_ARQ|NVIDIA_SMI|MODELO_1B|MODELO_3B|GGUF_DIR|LLAMA_DIR|TEXTOS_5C|REF_5C|IDS_CMP|CRITERIO_CMP \
      |PROC_STAT|CMP_CPU_JANELA|CMP_CPU_ESPERA|CMP_FOLGA|CMP_SEM_MEMORIA|CMP_TESTE_AMBOS|CURL|ESPERA_PRONTO|LOADAVG_ARQ|SYSTEMCTL|SYSTEMD_RUN|PENDENTE|CGROUP_RAIZ \
      |CMP_EST_*|CMP_FAKE_*|JANELA_*|LIMITE_*)
        recusa "variável só de teste presente no modo real: $v (só CMP_MODO_TESTE=1, o modo de teste explícito, a aceita); limpe o ambiente de quem chama";;
    esac
  done
fi
[ -x "$BIN" ] || recusa "binário não executável: $BIN"
[ -x "$JANELA" ] && [ -x "$LIMITE_SH" ] && [ -x "$LASTRO_SH" ] && [ -x "$LLAMA_SH" ] && [ -x "$AQUI/roda-cmp-monitor.sh" ] || recusa "faltam janela-e4b.sh, roda-com-limite.sh, roda-cmp-lastro.sh, roda-cmp-llamacpp.sh ou roda-cmp-monitor.sh executáveis"
[ -f "$EXTRAI" ] || recusa "falta cmp-extrai.py"
[ $DEGRAU1 = 0 ] || [ -f "$CRIT_D1" ] || recusa "falta o critério do degrau 1: $CRIT_D1"
[ $DEGRAU2 = 0 ] || { [ -f "$CRIT_D2" ] && [ -f "$CRIT_D1" ]; } || recusa "falta o critério do degrau 2 ou do degrau 1: $CRIT_D2 $CRIT_D1"
[ -n "$ESTADO" ] || ESTADO=$SAIDA-estado-janela
for d in "$SAIDA" "$ESTADO"; do
  if [ -e "$d" ]; then
    [ -d "$d" ] && [ -z "$(ls -A "$d")" ] || recusa "$d já existe e não está vazio: uma janela nova pede pasta nova"
  fi
done
SAIDA=$(realpath -m "$SAIDA"); BIN=$(realpath "$BIN")
# emenda 9 (r114): falha FECHADA antes de tocar em timers ou na GPU. Só PIDs de tipo G (desktop gráfico) são isentos; QUALQUER PID C ou C+G na 3080 Ti (pmon ou --query-compute-apps:
# voxtype, processo de GPU do Chrome...) recusa a janela com a lista de PIDs e comandos: feche-os (ou abra o Chrome sem GPU) e peça de novo. Isso faz parte do "autorizo" do Breno.
POT_PRE=$(le_pot) || POT_PRE=
[ "$POT_PRE" = "$POT_OK" ] || recusa "power.limit da 3080 Ti tem de ser $POT_OK (lido: '${POT_PRE:-ilegível}'); a série vale só no limite vigente; nada foi tocado, não mexa no limite por este roteiro"
PREFLIGHT=$("$AQUI/roda-cmp-monitor.sh" preflight 2>&1) || recusa "emenda 9: processo(s) de compute na 3080 Ti que precisam ser FECHADOS antes da janela (voxtype, processo de GPU do Chrome...): $PREFLIGHT"
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
for v in MODELO_1B MODELO_3B GGUF_DIR LLAMA_DIR TEXTOS_5C REF_5C IDS_CMP CRITERIO_CMP CRITERIO_D1 CRITERIO_D2 SEM_SHA_FIXO NVIDIA_SMI PROC_STAT CMP_CPU_JANELA CMP_CPU_ESPERA LASTRO_SH_CMP LLAMA_SH_CMP EXTRAI_PY \
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
[ $DEGRAU1 = 0 ] || EXTRA+=(--degrau1)
[ $DEGRAU2 = 0 ] || EXTRA+=(--degrau2)
[ $SO_S3 = 0 ] || EXTRA+=(--so-s3)
"$JANELA" --estado "$ESTADO" "${TIMERS[@]}" --limite $((LIMITE + 120)) --kill-apos 30 -- \
  "$LIMITE_SH" --limite "$LIMITE" -- env "${PASSA[@]}" "$AQUI/roda-cmp-janela.sh" --interno --modelos "$MODELOS" --saida "$SAIDA" --bin "$BIN" --limite "$LIMITE" "${EXTRA[@]}"
RC=$?
coleta fim "$SAIDA.carga-depois.txt"
lista_systemd depois
# aviso no fim da janela (sempre, qualquer que seja o código): arquivo ao lado da pasta e notificação se houver como
MSG="janela da comparação Lastro x llama.cpp ($MODELOS$([ $DEGRAU1 = 1 ] && echo ', degrau 1, L100 contra S1')$([ $DEGRAU2 = 1 ] && echo ', degrau 2, LC contra S2')$([ $SO_S3 = 1 ] && echo ', só a S3 do llama.cpp')) terminou (código $RC): $SAIDA"
echo "$MSG" > "$SAIDA.aviso.txt"
if [ -n "${CMP_AVISO_CMD:-}" ]; then "$CMP_AVISO_CMD" "$MSG" > /dev/null 2>&1 || true
elif command -v notify-send > /dev/null 2>&1; then notify-send "Lastro x llama.cpp" "$MSG" > /dev/null 2>&1 || true; fi
exit $RC
