#!/usr/bin/env bash
# F2D: janela do plano da O3, escrever direto no cache K/V (plano-o3-cache-direto-20261112.md versão 2, aprovada na rodada 173; decisão do Breno de 12/11 00:37 em .herdr/DECISAO-BRENO-O3-OITO-PONTOS-20261112-0037.txt;
# critério em criterio-f2d-20261112.md). UMA janela POR MODELO (uma de cada vez), até 20 min na RTX 3080 Ti, SÓ o Lastro, descritiva. Derivado de roda-f2c-janela.sh (mesmo envelope, mesma falha FECHADA antes de tocar em timers ou na
# GPU, mesmo FIXO com shas e --verifica-fixos sem pipe mascarando o código de saída, mesmos portões de UUID e potência). O voxtype NÃO é tratado aqui: o script de abertura em .herdr cuida; este roteiro só recusa se ele estiver em compute.
#
#   roda-f2d-janela.sh --modelos 1b|3b --saida DIR [--estado-janela DIR] [--bin-base BIN] [--bin-cand BIN] [--limite S]
#   roda-f2d-janela.sh --modelos 1b|3b --saida DIR --coleta [--teste-gpu --worktree-cand DIR]   (execução própria da prova bit a bit, HC1b: base e candidato, um processo cada, só LASTRO_GANCHO_LOGITS=1; opcional: teste de erro em GPU)
#   roda-f2d-janela.sh --verifica-fixos     (sem GPU, sem autorização, só lê: confere os shas congelados do FIXO contra os arquivos reais; código 0 só se todos OK; PENDENTE conta como falha)
#
# TRÊS BINÁRIOS (dois commits), cada um com sha no FIXO e conferido ANTES de parar timers (nenhum é escrito por este roteiro; cópias congeladas fora do git, sem permissão de escrita):
#   C0 (D_O1): base comum = o HEAD da O1 mais a infraestrutura comum (instrumento v3 com contrato e capacidade selecionáveis, produção ainda no contrato v2 de 17 registros, kernels e as 2 cópias do cache K/V mantidos), gancho de logits, tudo desligado;
#   C1 (D_O3 e B2_O3): C0 mais a O3 (o diff C0 a C1 se restringe à O3 e à emissão dos registros removidos: gpu.rs com os dois kernels novos, cuda_dense.rs, perfil_bloco.rs só na escolha do contrato de produção v3, cuda_backend.rs se preciso e os testes; os dois commits locais e o comando de build estão em criterio-f2d-20261112.md);
#   22d0bbe0 (14:10): SÓ CONTEXTO, confere o sha e nunca roda nesta janela.
# BRAÇOS (todos no texto MÉDIO, série L100, quente, `--repete 3` = 1 aquecimento + 3 medidas por processo): D_O1 = binário de C0, nada ligado; D_O3 = binário de C1, nada ligado; B2_O3 = binário de C1, LASTRO_PERFIL_BLOCO=1,
# LASTRO_PERFIL_F2=1 e LASTRO_PERFIL_F2B=1 (instrumento v3). LASTRO_GANCHO_LOGITS nunca liga na janela de medida (a coleta de logits é o modo --coleta, execução própria com os mesmos portões). As variáveis ligam SÓ por processo, na chamada, nunca herdadas (recusa 2 se vierem no
# ambiente); prova-variavel.txt grava o ambiente passado, o sha do binário e as linhas `perfil-bloco:`, `perfil-f2:` e `perfil-f2b:` que o binário imprimiu.
# ORDEM: 9 processos por modelo, 3 por braço, intercalados em QUADRADO LATINO 3x3 (cada braço uma vez em cada linha e em cada posição): D_O1 D_O3 B2_O3, B2_O3 D_O1 D_O3, D_O3 B2_O3 D_O1.
# Pasta de cada processo: celulas/ctrl-<modelo>-L100-medio-<NN>-<DO1|DO3|B2O3>. Sem smoke, sem curto e sem longo: exatamente os 9 processos do plano.
# --limite padrão 1200 s e máximo 1200 s, TETO GLOBAL por janela/modelo (os 9 processos juntos); o de fora é o de dentro mais 120 s. Sem repetição automática.
# TETOS DE PROCESSO (como no pacote F2): 60 s (1B) e 90 s (3B) no médio. Passar do teto = processo morto, em teto-excedido.txt (inválido, sem repetição).
# PLANEJAMENTO: quem decide é `--planeja-controle` do binário candidato (fonte única, perfil_bloco.rs). O controle cabe inteiro (decorrido + 9 tetos + folga 180 s <= limite) ou os nove viram NÃO RODOU; nunca é cortado. Se os
# nove tetos + folga não couberem no limite, a janela NÃO COMEÇA (código 2, DIR/decisao.txt, nada tocado).
#
# Falha FECHADA antes de tocar em timers ou na GPU: PRIMEIRA_PALAVRA_AUTORIZADA=1 e CUDA_VISIBLE_DEVICES igual ao UUID da 3080 Ti; --saida e --estado-janela pastas novas ou vazias; power.limit 370.00 W no início (e ao fim,
# senão SEM PROVA e o extrator RECUSA); QUALQUER PID C ou C+G na 3080 Ti recusa; RECUSA TAMBÉM se houver `cargo test`/`cargo nextest`, binário de teste do cargo, primeira-palavra ou llama.* em execução; o sha dos dois binários
# que rodam é conferido ANTES de parar timers (--verifica-fixos, que confere também o de contexto). Variáveis só para teste (JANELA_SH, LIMITE_SH, JANELA_TIMERS, LASTRO_SH_CMP, EXTRAI_PY, PERFIL_EXTRAI_PY, INSTR_EXTRAI_PY, F2_EXTRAI_PY,
# F2B_EXTRAI_PY, F2D_EXTRAI_PY, CMP_RELOGIO_ARQ, CMP_FOLGA, F2D_TETO_FORCADO, CMP_FAKE_*, MODELO_1B, MODELO_3B, TEXTOS_5C, REF_5C, IDS_CMP, CRITERIO_*, PLANO_O3, SEM_SHA_FIXO, NVIDIA_SMI, ..., LASTRO_PERFIL_BLOCO,
# LASTRO_PERFIL_F2, LASTRO_PERFIL_F2B, LASTRO_GANCHO_LOGITS): no modo real qualquer uma presente no ambiente é RECUSADA com código 2; só CMP_MODO_TESTE=1 as aceita. CMP_AVISO_CMD continua permitida. Este roteiro NÃO é rodado na entrega.
# Códigos: 0 se decisao.txt foi gravada, o do comando (124 limite, ...) ou 2 recusa de uso, portão, ambiente ou orçamento que não cabe.
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
F2_EXTRAI=${F2_EXTRAI_PY:-$AQUI/f2-extrai.py}
F2B_EXTRAI=${F2B_EXTRAI_PY:-$AQUI/f2b-extrai-v3.py}
F2D_EXTRAI=${F2D_EXTRAI_PY:-$AQUI/f2d-extrai.py}
BIN_BASE=$AQUI/../../.herdr/binarios/primeira-palavra-f2d-base-4a6f26c7
BIN_CAND=$AQUI/../../.herdr/binarios/primeira-palavra-f2d-cand-f295e90a
BIN_CTX=$AQUI/../../.herdr/binarios/primeira-palavra-f2b-22d0bbe0
M1=${MODELO_1B:-/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct}
M3=${MODELO_3B:-/home/brenoperucchi/models/granite-3.1-3b-a800m-instruct}
TEXTOS=${TEXTOS_5C:-$AQUI/5c-textos}
REFS=${REF_5C:-$AQUI/5c-referencia}
IDS=${IDS_CMP:-$AQUI/cmp-ids-prompt.json}
CRIT_F2D=${CRITERIO_F2D:-$AQUI/criterio-f2d-20261112.md}
PLANO=${PLANO_O3:-$AQUI/plano-o3-cache-direto-20261112.md}
DECISAO=${DECISAO_O3:-$AQUI/../../.herdr/DECISAO-BRENO-O3-OITO-PONTOS-20261112-0037.txt}
CRIT=${CRITERIO_CMP:-$AQUI/criterio-comparacao-llamacpp-20261108.md}
CRIT_D1=${CRITERIO_D1:-$AQUI/criterio-degrau1-100pct-20261109.md}
CRIT_D2=${CRITERIO_D2:-$AQUI/criterio-degrau2-camadas-20261109.md}
FOLGA=${CMP_FOLGA:-180}
LIMITE_MAX=1200
# quadrado latino 3x3: DO1 = D_O1, DO3 = D_O3, B2O3 = B2_O3
ORDEM_CTRL=(DO1 DO3 B2O3 B2O3 DO1 DO3 DO3 B2O3 DO1)
SAIDA=; ESTADO=; LIMITE=$LIMITE_MAX; INTERNO=0; MODELOS=; VERIFICA=0; COLETA=0; TESTE_GPU=0; T_TMP_GPU=; WT_CAND=; COLETA_DIR=
usa() { echo "uso: roda-f2d-janela.sh --modelos 1b|3b --saida DIR [--estado-janela DIR] [--coleta-dir DIR (medição: coleta do mesmo modelo, para o HC1b no resultado)] [--bin-base BIN] [--bin-cand BIN] [--limite S] [--coleta [--teste-gpu --worktree-cand DIR]] | --verifica-fixos" >&2; }
recusa() { echo "roda-f2d-janela: $*" >&2; exit 2; }
# Potência: le_pot lê só o power.limit da 3080 Ti (vazio se a leitura falhar); 370.00 W no preflight (senão recusa 2) e de novo ao fim (senão integridade QUEBRADA)
POT_OK="370.00 W"
le_pot() { local o; o=$("${NVIDIA_SMI:-nvidia-smi}" -i "$UUID" --query-gpu=power.limit --format=csv,noheader 2> /dev/null) || return 1; printf '%s\n' "$o" | head -n 1 | cut -d, -f1 | sed 's/^ *//;s/ *$//'; }
pot_tok() { local v=${1:-}; [ -n "$v" ] || v=ilegivel; echo "${v// /_}"; }
while [ $# -gt 0 ]; do
  case $1 in
    --saida) [ $# -ge 2 ] || { usa; exit 2; }; SAIDA=$2; shift 2;;
    --estado-janela) [ $# -ge 2 ] || { usa; exit 2; }; ESTADO=$2; shift 2;;
    --bin-base) [ $# -ge 2 ] || { usa; exit 2; }; BIN_BASE=$2; shift 2;;
    --bin-cand) [ $# -ge 2 ] || { usa; exit 2; }; BIN_CAND=$2; shift 2;;
    --limite) [ $# -ge 2 ] || { usa; exit 2; }; LIMITE=$2; shift 2;;
    --modelos) [ $# -ge 2 ] || { usa; exit 2; }; MODELOS=$2; shift 2;;
    --verifica-fixos) VERIFICA=1; shift;;
    --interno) INTERNO=1; shift;;
    --coleta) COLETA=1; shift;;
    --teste-gpu) TESTE_GPU=1; shift;;
    --coleta-dir) [ $# -ge 2 ] || { usa; exit 2; }; COLETA_DIR=$2; shift 2;;
    --worktree-cand) [ $# -ge 2 ] || { usa; exit 2; }; WT_CAND=$2; shift 2;;
    *) usa; exit 2;;
  esac
done
# variáveis só de teste no ambiente do modo real OU interno = recusa 2 (validado logo após os argumentos, antes de consumir qualquer override); só CMP_MODO_TESTE=1 as aceita. `compgen -e` lista só as EXPORTADAS.
if [ "${VERIFICA:-0}" != 1 ] && [ "${CMP_MODO_TESTE:-0}" != 1 ]; then
  for v in $(compgen -e); do
    case $v in
      LASTRO_PERFIL_BLOCO|LASTRO_PERFIL_F2|LASTRO_PERFIL_F2B|LASTRO_GANCHO_LOGITS|CMP_FRACAO_SLOTS|CMP_CAMADAS_CPU|CRITERIO_F2D|PLANO_O3|DECISAO_O3|F2B_EXTRAI_PY|F2D_EXTRAI_PY|CRITERIO_D1|CRITERIO_D2|SEM_SHA_FIXO|MONITOR_SH|LASTRO_SH_CMP|EXTRAI_PY|PERFIL_EXTRAI_PY|INSTR_EXTRAI_PY|F2_EXTRAI_PY|CMP_RELOGIO_ARQ|NVIDIA_SMI|MODELO_1B|MODELO_3B|TEXTOS_5C|REF_5C|IDS_CMP|CRITERIO_CMP \
      |PROC_STAT|CMP_CPU_JANELA|CMP_CPU_ESPERA|CMP_FOLGA|CMP_SEM_MEMORIA|ESPERA_PRONTO|LOADAVG_ARQ|SYSTEMCTL|SYSTEMD_RUN|PENDENTE|CGROUP_RAIZ|PROC_LISTA|F2D_TETO_FORCADO|F2D_CARGO_TESTE \
      |CMP_EST_*|CMP_FAKE_*|JANELA_*|LIMITE_*)
        recusa "variável só de teste presente no modo real: $v (só CMP_MODO_TESTE=1, o modo de teste explícito, a aceita); limpe o ambiente de quem chama";;
    esac
  done
fi
sha() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }

# SHAs congelados. O próprio roda-f2d-janela.sh NÃO pode conter o seu sha (nem o critério o do roteiro nem o do extrator): --verifica-fixos imprime o do roteiro e o dos extratores à parte (linhas informativas).
# PREENCHER_NO_CONGELAMENTO = (nenhum mais; vale para o que ainda não for congelado, como nos testes) ainda não congelado (binários e arquivos que mudam até a aprovação): --verifica-fixos imprime PENDENTE e sai 1, e o modo real recusa; nenhum sha de binário inexistente foi inventado aqui.
declare -A FIXO=(
  [criterio-f2d]=9f595163622092c3e049ff8e406234106c82d07f056ed6ee57060c1308ac1804
  [plano-o3]=e4edd1d013c4ac56ee0b145527f60f8490a0931e3f1d995ef5172f649795b032
  [decisao-o3]=1208279673f3058d33ba3667ed956c2259bbfef48adb475fcaec6e102844048a
  [binario-base]=4a6f26c7eaf9898cc625ce16e617cc0c68983670abcfec9502f09fd6aab9579a
  [binario-cand]=f295e90a6a2d3b90332ba5bf11f2a0e337aa646806d2a5e308c6f5c2f00c9cea
  [binario-ctx]=22d0bbe048d8c7407f73817b9879f5bc695604bc1bc519e332c9287c193adacf
  [script-f2b-extrai-v3]=2bcd8761cba3eef9bb254998ff83e6094028777008e296581b6c14ae85062912
  [script-f2d-extrai]=098963c9f04c71e6a837ef3eb04397dd455fe13421e15255a5dd85d06d41df15
  [script-f2-extrai]=f68129e832a5f617cc682b966293a502c4c095439c6e8673828e80e4363391fc
  [script-instrumento-extrai]=6f7169d4bcfa5946b15801543b783b702ead6e16e39b3014293aaa59c76f23d7
  [script-perfil-extrai]=755e1d7dd622e98dce9513ef30584e21667a5bc9d8c08cce4198090b2c34ba68
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
  add criterio-f2d "$CRIT_F2D"; add plano-o3 "$PLANO"; add decisao-o3 "$DECISAO"
  add binario-base "$BIN_BASE"; add binario-cand "$BIN_CAND"; add binario-ctx "$BIN_CTX"
  add script-f2b-extrai-v3 "$F2B_EXTRAI"; add script-f2d-extrai "$F2D_EXTRAI"; add script-f2-extrai "$F2_EXTRAI"; add script-instrumento-extrai "$INSTR_EXTRAI"; add script-perfil-extrai "$PERFIL_EXTRAI"; add script-cmp-extrai "$EXTRAI"; add script-lastro "$LASTRO_SH"; add script-monitor "$MONITOR"
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
# Processos que recusam a janela: `cargo test`/`cargo nextest`/`cargo bench`, binário de teste do cargo (target/<perfil>/deps/<nome>-<hash de 16 hex>), primeira-palavra, llama.*.
# Lista = `ps -eo pid=,args=` (ou o arquivo PROC_LISTA, só em teste: linhas "PID ARGS"); este script e os ancestrais dele ficam de fora. Imprime uma linha por processo encontrado.
cargo_teste() { # "args": 0 só se o EXECUTÁVEL (primeira palavra, basename) é `cargo` com `test|t|nextest|bench` como subcomando, ou `cargo-nextest`; descrição de dispatcher, grep, bash -c etc. não contam
  local w base first=1 achou=0
  for w in $1; do
    base=${w##*/}; base=${base//[\"\']/}
    if [ $first = 1 ]; then
      first=0
      case $base in cargo) achou=1; continue;; cargo-nextest) return 0;; *) return 1;; esac
    fi
    case $w in -*|+*) continue;; esac
    case $w in test|t|nextest|bench) return 0;; esac
    return 1
  done
  return 1
}
# Códigos: 0 = nenhum processo proibido, 1 = achou, 2 = falha ao ler a lista de processos (recusa com motivo; nunca vira "lista vazia").
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
# --worktree-cand (teste de erro em GPU): só vale se o HEAD é o commit candidato do critério (linha "- hash do commit candidato: <40 hex>") e a árvore está limpa (o alvo de compilação target-f2d fica de fora); PREENCHER recusa.
confere_wt() {
  local cc h st rc_st
  cc=$(sed -n 's/^- hash do commit candidato: *//p' "$CRIT_F2D" 2> /dev/null | head -n 1 | tr -d ' ')
  [[ $cc =~ ^[0-9a-f]{40}$ ]] || { echo "o hash do commit candidato ainda não foi preenchido no critério (PREENCHER NO CONGELAMENTO)"; return 1; }
  h=$(git -C "$WT_CAND" rev-parse HEAD 2> /dev/null) || { echo "$WT_CAND não é um repositório git"; return 1; }
  [ "$h" = "$cc" ] || { echo "HEAD de $WT_CAND ($h) diferente do commit candidato do critério ($cc)"; return 1; }
  # falha FECHADA: o código de saída do git status é capturado ANTES de testar se a saída está vazia (índice ilegível sai 128 com stdout vazio); stdout, stderr e código têm de estar limpos
  st=$(git -C "$WT_CAND" status --porcelain -- . ':(exclude)target-f2d' 2>&1); rc_st=$?
  [ $rc_st = 0 ] || { echo "não foi possível verificar o estado do worktree $WT_CAND (git status saiu com código $rc_st): $(echo "$st" | head -n 2 | tr '\n' ' ')"; return 1; }
  [ -z "$st" ] || { echo "árvore de $WT_CAND com modificações ou com saída inesperada do git status: $(echo "$st" | head -n 2 | tr '\n' ' ')"; return 1; }
  return 0
}
# portão dos binários que RODAM: base+gancho e candidato têm de ter o sha do FIXO ANTES de qualquer envelope ou célula (SEM_SHA_FIXO=1, só em teste, dispensa). PENDENTE (sha ainda não congelado) nunca confere.
confere_um() { [ "$(sha "$1")" = "$2" ] && [ "$2" != PREENCHER_NO_CONGELAMENTO ]; }
confere_binarios() { [ "${SEM_SHA_FIXO:-0}" = 1 ] || { confere_um "$BIN_BASE" "${FIXO[binario-base]}" && confere_um "$BIN_CAND" "${FIXO[binario-cand]}"; }; }

# --verifica-fixos: só lê (sem GPU, sem autorização, sem tocar em timers). AUSENTE = arquivo que não existe aqui e conta como falha (código 1); PENDENTE = sha ainda não congelado, também falha.
if [ $VERIFICA = 1 ]; then
  monta_lista todos
  rc_v=0
  for i in "${!CHAVES[@]}"; do
    k=${CHAVES[$i]}
    [ -n "${FIXO[$k]:-}" ] || continue
    h=$(sha "${CAMINHOS[$i]}")
    if [ "${FIXO[$k]}" = PREENCHER_NO_CONGELAMENTO ]; then echo "$k ${FIXO[$k]} ${h:--} PENDENTE ${CAMINHOS[$i]##*/}"; rc_v=1
    elif [ -z "$h" ]; then echo "$k ${FIXO[$k]} - AUSENTE ${CAMINHOS[$i]##*/}"; rc_v=1
    elif [ "$h" = "${FIXO[$k]}" ]; then echo "$k ${FIXO[$k]} $h OK"
    else echo "$k ${FIXO[$k]} $h FALHA"; rc_v=1; fi
  done
  echo "informativo: roda-f2d-janela.sh sha256 $(sha "$AQUI/roda-f2d-janela.sh") (autorreferente, fora do FIXO; conferir por fora)"
  exit $rc_v
fi
# --modelos: uma janela por modelo; ausente ou inválido = recusa 2 (nunca as duas de uma vez)
case $MODELOS in
  1b|3b) ;;
  '') usa; recusa "--modelos 1b|3b é obrigatório: uma janela por modelo";;
  *) recusa "--modelos inválido: '$MODELOS' (use 1b ou 3b, nunca os dois)";;
esac
[ -n "$SAIDA" ] || { usa; exit 2; }
[ $TESTE_GPU = 0 ] || [ $COLETA = 1 ] || recusa "--teste-gpu só existe junto de --coleta"
[ -z "$COLETA_DIR" ] || [ $COLETA = 0 ] || recusa "--coleta-dir só existe na janela de medida (sem --coleta)"
[ -z "$COLETA_DIR" ] || [ -d "$COLETA_DIR" ] || recusa "--coleta-dir '$COLETA_DIR' não é uma pasta"
[ $TESTE_GPU = 0 ] || [ -n "$WT_CAND" ] || recusa "--teste-gpu exige --worktree-cand DIR (worktree do commit candidato, já compilado com CARGO_TARGET_DIR=DIR/target-f2d)"
# --worktree-cand canonicalizado com realpath (inode real, sem symlink nem `..`) e recusado já aqui, antes de qualquer mktemp, timer ou GPU, se ficar sob /tmp (testes fora do /tmp).
if [ $TESTE_GPU = 1 ]; then
  WT_REAL=$(realpath -e -- "$WT_CAND" 2> /dev/null) || recusa "--worktree-cand '$WT_CAND' não existe ou não é resolvível (realpath)"
  case "$WT_REAL/" in /tmp/*) recusa "--worktree-cand ($WT_REAL) está sob /tmp: o worktree do candidato e o TMPDIR do teste de GPU ficam fora do /tmp";; esac
  WT_CAND=$WT_REAL
fi
# Tetos de processo em segundos (tabela do ponto 8; a mesma de perfil_bloco.rs::teto_processo_s, conferida por teste). F2D_TETO_FORCADO só em teste.
teto_proc() { # TEXTO (medio)
  if [ -n "${F2D_TETO_FORCADO:-}" ]; then echo "$F2D_TETO_FORCADO"; return; fi
  [ "$MODELOS" = 1b ] && echo 60 || echo 90
}
# Coleta (--coleta): dois processos (original e candidato), cada um com o gancho ligado (lê o KV por camada: mais lento), teto de processo = 2 x o do médio; o teste de GPU opcional tem teto próprio.
TETO_TESTE_GPU=300
teto_coleta() { if [ -n "${F2D_TETO_FORCADO:-}" ]; then echo "$F2D_TETO_FORCADO"; else echo $(( 2 * $(teto_proc medio) )); fi; }
TETOS9=$(for _ in 1 2 3 4 5 6 7 8 9; do printf '%s,' "$(teto_proc medio)"; done); TETOS9=${TETOS9%,}
# bloco obrigatório da janela: nove tetos + folga (o controle nunca é cortado)
bloco_obrig_s() {
  if [ $COLETA = 1 ]; then echo $(( 2 * $(teto_coleta) + FOLGA + (TESTE_GPU * TETO_TESTE_GPU) )); else echo $(( 9 * $(teto_proc medio) + FOLGA )); fi
}

# ---------------------------------------------------------------- modo interno (dentro da janela, com o portão e o UUID no ambiente)
if [ $INTERNO = 1 ]; then
  [ "${PRIMEIRA_PALAVRA_AUTORIZADA:-}" = 1 ] && [ "${CUDA_VISIBLE_DEVICES:-}" = "$UUID" ] || recusa "modo interno sem portão ou UUID"
  confere_binarios || recusa "binário com sha diferente do FIXO (base ${FIXO[binario-base]}, candidato ${FIXO[binario-cand]}): $BIN_BASE ou $BIN_CAND (modo interno)"
  PROIB_I=$(proc_proibidos) || recusa "cargo test, processo CUDA em execução ou lista de processos ilegível (modo interno): $(echo "$PROIB_I" | head -n 3 | tr '\n' ' ')"
  if [ $TESTE_GPU = 1 ]; then WT_MSG=$(confere_wt) || recusa "--worktree-cand recusado (modo interno): $WT_MSG"; fi
  mkdir -p "$SAIDA/celulas" || exit 2
  # as variáveis do instrumento NÃO são limpas aqui: cada processo recebe o seu ambiente explícito (D_O1 e D_O3: env -u nas três e no gancho; B2_O3: as três); o que o ambiente de quem chama trouxer não vale
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
    echo "binario_base=$(sha "$BIN_BASE")"
    echo "binario_cand=$(sha "$BIN_CAND")"
    echo "binario_ctx=$(sha "$BIN_CTX") (só contexto, não roda)"
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
  NAO_RODOU=()
  : > "$SAIDA/ordem.txt"; : > "$SAIDA/orcamento-log.txt"; : > "$SAIDA/prova-variavel.txt"; : > "$SAIDA/teto-excedido.txt"
  {
    if [ $COLETA = 1 ]; then
      echo "plano da COLETA da F2D (HC1b, prova bit a bit), modelo $MODELOS: dois processos no texto médio, --repete 3, só LASTRO_GANCHO_LOGITS=1 (original = C0, depois candidato = C1); teto de processo $(teto_coleta) s; teto global $LIMITE s; folga $FOLGA s$([ $TESTE_GPU = 1 ] && echo "; mais o teste de erro em GPU (motor_reforma_gpu, motor_passo_gpu e o3_kernels_gpu --ignored, E4B_GPU_AUTORIZADO=1, teto $TETO_TESTE_GPU s)")"
    fi
    echo "plano da janela da F2D (plano da O3), modelo $MODELOS (ordem fixa; só Lastro, L100, quentes, texto médio; sem repetição automática; sem smoke, curto ou longo)"
    echo "tetos de processo: médio $(teto_proc medio) s; teto global $LIMITE s; folga $FOLGA s"
    echo "binários: D_O1 = C0 (base comum); D_O3 e B2_O3 = C1 (C0 mais a O3); 22d0bbe0 só contexto (não roda)"
    i=0; for b in "${ORDEM_CTRL[@]}"; do i=$((i + 1)); printf 'controle ctrl-%s-L100-medio-%02d-%s (--repete 3; braço %s)\n' "$MODELOS" "$i" "$b" "$b"; done
  } > "$SAIDA/plano.txt"
  SEQ=0; ABORTA=0
  nr() { NAO_RODOU+=("$1"); echo "NÃO RODOU: $1" >> "$SAIDA/orcamento-log.txt"; }
  # binário e ambiente de UM processo, por braço (D_O1: C0 sem variável; D_O3: C1 sem variável; B2_O3: C1 com as três). GNU env só lê -u ANTES das atribuições NOME=valor: os -u vêm primeiro.
  # LASTRO_GANCHO_LOGITS é sempre removida: o gancho de coleta não liga numa janela de medida.
  bin_do_braco() { case $1 in DO1) echo "$BIN_BASE";; *) echo "$BIN_CAND";; esac; }
  env_proc() {
    local u=(-u CMP_CAMADAS_CPU -u LASTRO_GANCHO_LOGITS) a=(CMP_FRACAO_SLOTS=1.0)
    case $1 in
      DO1|DO3) u+=(-u LASTRO_PERFIL_BLOCO -u LASTRO_PERFIL_F2 -u LASTRO_PERFIL_F2B);;
      B2O3) a+=(LASTRO_PERFIL_BLOCO=1 LASTRO_PERFIL_F2=1 LASTRO_PERFIL_F2B=1);;
    esac
    ENVX=("${u[@]}" "${a[@]}")
  }
  # roda_processo CHAVE N BRACO TETO: UMA chamada ao roda-cmp-lastro.sh (processo único com --repete N) sob `timeout` (o grupo inteiro morre no teto); registra a ordem, a prova das variáveis, o binário e o teto excedido.
  roda_processo() {
    local chave=$1 n=$2 b=$3 teto=$4 t0 rc lb lf lx dir bin inc=
    SEQ=$((SEQ + 1)); t0=$(decorrido); dir="$SAIDA/celulas/$chave"
    env_proc "$b"; bin=$(bin_do_braco "$b")
    timeout -k 5 "$teto" env "${ENVX[@]}" "$LASTRO_SH" --modelo "$MODELOS" --texto medio --estado quente --n "$n" --saida "$dir" --bin "$bin" > "$SAIDA/celulas-$chave.log" 2>&1
    rc=$?
    printf '%02d %s n=%s braco=%s inicio=%ss fim=%ss rc=%s\n' "$SEQ" "$chave" "$n" "$b" "$t0" "$(decorrido)" "$rc" >> "$SAIDA/ordem.txt"
    [ "$rc" != 124 ] && [ "$rc" != 137 ] || { echo "$chave teto=${teto}s rc=$rc" >> "$SAIDA/teto-excedido.txt"; echo "teto de processo excedido em $chave ($teto s): processo inválido, sem repetição" >> "$SAIDA/orcamento-log.txt"; }
    lb=$(grep -h '^perfil-bloco: ' "$dir"/p*/stdout.txt 2> /dev/null | sort -u | tr '\n' '|' | sed 's/|$//')
    lf=$(grep -h '^perfil-f2: ' "$dir"/p*/stdout.txt 2> /dev/null | sort -u | tr '\n' '|' | sed 's/|$//')
    lx=$(grep -h '^perfil-f2b: ' "$dir"/p*/stdout.txt 2> /dev/null | sort -u | tr '\n' '|' | sed 's/|$//')
    case $b in
      DO1|DO3) [ "$lb" = "perfil-bloco: desligado" ] && [ "$lf" = "perfil-f2: desligado" ] && [ "$lx" = "perfil-f2b: desligado" ] || inc=" INCOERENTE";;
      B2O3) [ "$lb" = "perfil-bloco: ligado" ] && [ "$lf" = "perfil-f2: ligado" ] && [ "$lx" = "perfil-f2b: ligado" ] || inc=" INCOERENTE";;
    esac
    printf '%s braco=%s bin=%s env="%s" linha_bloco="%s" linha_f2="%s" linha_f2b="%s"%s\n' "$chave" "$b" "$(sha "$bin")" "${ENVX[*]}" "${lb:-ausente}" "${lf:-ausente}" "${lx:-ausente}" "$inc" >> "$SAIDA/prova-variavel.txt"
    [ "$rc" != 2 ] || { ABORTA=1; echo "processo $chave recusou o ambiente (código 2): janela interrompida" >> "$SAIDA/orcamento-log.txt"; }
    return $rc
  }
  # consulta o planejamento ao binário candidato (fonte única); saída vazia ou código != 0 = falha FECHADA (devolve "erro")
  planeja() { # --planeja-controle  args...
    local o; o=$("$BIN_CAND" "$@" 2> /dev/null) || { echo erro; return 0; }
    case $o in cabe|nao-cabe) echo "$o";; *) echo erro;; esac
  }
  # ---- controle de 9 processos no MÉDIO, quadrado latino: cabe inteiro (decorrido + 9 tetos + folga) ou os nove viram NÃO RODOU; o 1º com erro encerra o controle (sem repetição) e o resto vira NÃO RODOU; nunca é cortado
  if [ $COLETA = 1 ]; then
    TETO_C=$(teto_coleta)
    for papel in original candidato; do
      chave="coleta-$MODELOS-medio-$papel"
      if [ $ABORTA != 0 ]; then nr "$chave: coleta interrompida por recusa de ambiente (código 2)"; continue; fi
      if [ $papel = original ]; then bin=$BIN_BASE; else bin=$BIN_CAND; fi
      SEQ=$((SEQ + 1)); t0=$(decorrido); dir="$SAIDA/celulas/$chave"
      timeout -k 5 "$TETO_C" env -u CMP_CAMADAS_CPU -u LASTRO_PERFIL_BLOCO -u LASTRO_PERFIL_F2 -u LASTRO_PERFIL_F2B CMP_FRACAO_SLOTS=1.0 LASTRO_GANCHO_LOGITS=1 "$LASTRO_SH" --modelo "$MODELOS" --texto medio --estado quente --n 3 --saida "$dir" --bin "$bin" > "$SAIDA/celulas-$chave.log" 2>&1
      rc=$?
      printf '%02d %s n=3 papel=%s inicio=%ss fim=%ss rc=%s\n' "$SEQ" "$chave" "$papel" "$t0" "$(decorrido)" "$rc" >> "$SAIDA/ordem.txt"
      [ "$rc" != 124 ] && [ "$rc" != 137 ] || { echo "$chave teto=${TETO_C}s rc=$rc" >> "$SAIDA/teto-excedido.txt"; echo "teto de processo excedido em $chave: coleta inválida, sem repetição" >> "$SAIDA/orcamento-log.txt"; }
      lb=$(grep -h '^perfil-bloco: ' "$dir"/p*/stdout.txt 2> /dev/null | sort -u | tr '\n' '|' | sed 's/|$//'); lf=$(grep -h '^perfil-f2: ' "$dir"/p*/stdout.txt 2> /dev/null | sort -u | tr '\n' '|' | sed 's/|$//'); lx=$(grep -h '^perfil-f2b: ' "$dir"/p*/stdout.txt 2> /dev/null | sort -u | tr '\n' '|' | sed 's/|$//')
      inc=; [ "$lb" = "perfil-bloco: desligado" ] && [ "$lf" = "perfil-f2: desligado" ] && [ "$lx" = "perfil-f2b: desligado" ] || inc=" INCOERENTE"
      printf '%s papel=%s bin=%s env="LASTRO_GANCHO_LOGITS=1 (só ele; instrumento desligado)"%s\n' "$chave" "$papel" "$(sha "$bin")" "$inc" >> "$SAIDA/prova-variavel.txt"
      arq=$(ls "$dir"/p*/saida/gancho-logits-"$MODELOS".json 2> /dev/null | head -n 1)
      if [ "$rc" = 0 ] && [ -n "$arq" ]; then mkdir -p "$SAIDA/gancho-$papel" && cp "$arq" "$SAIDA/gancho-$papel/gancho-logits-$MODELOS.json"; else nr "$chave: sem arquivo de gancho (rc $rc)"; fi
      [ "$rc" != 2 ] || { ABORTA=1; echo "processo $chave recusou o ambiente (código 2): coleta interrompida" >> "$SAIDA/orcamento-log.txt"; }
    done
    if [ $TESTE_GPU = 1 ] && [ $ABORTA = 0 ]; then
      SEQ=$((SEQ + 1))
      # TMPDIR do teste de GPU fora do /tmp (regra do Breno: testes fora do /tmp): pasta própria dentro do alvo `target-f2d` do worktree candidato (o git NÃO o ignora: a
      # conferência de árvore limpa o exclui por `:(exclude)target-f2d`), criada por mktemp e removida ao fim só pelo caminho literal que o mktemp imprimiu (nunca reatribuído nem montado).
      mkdir -p "$WT_CAND/target-f2d" 2> /dev/null
      T_TMP_GPU=$(mktemp -d "$WT_CAND/target-f2d/tmp-teste-gpu.XXXXXX" 2> /dev/null)
      case "$T_TMP_GPU" in
        /tmp/*|"") T_TMP_GPU_OK=0;;
        *) T_TMP_GPU_OK=1;;
      esac
      if [ $T_TMP_GPU_OK = 0 ]; then
        echo "não foi possível criar o TMPDIR do teste de GPU fora do /tmp em $WT_CAND/target-f2d" > "$SAIDA/teste-gpu.txt"; rc_t=2
      elif [ -n "${F2D_CARGO_TESTE:-}" ]; then
        TMPDIR=$T_TMP_GPU timeout -k 5 "$TETO_TESTE_GPU" env E4B_GPU_AUTORIZADO=1 "$F2D_CARGO_TESTE" > "$SAIDA/teste-gpu.txt" 2>&1; rc_t=$?
      else
        ( cd "$WT_CAND/rust/moe-rt" && TMPDIR=$T_TMP_GPU CARGO_TARGET_DIR="$WT_CAND/target-f2d" timeout -k 5 "$TETO_TESTE_GPU" env E4B_GPU_AUTORIZADO=1 cargo test --release --test motor_reforma_gpu --test motor_passo_gpu --test o3_kernels_gpu -- --ignored --nocapture ) > "$SAIDA/teste-gpu.txt" 2>&1; rc_t=$?
      fi
      [ $T_TMP_GPU_OK = 0 ] || rm -rf -- "$T_TMP_GPU"
      echo "$rc_t" > "$SAIDA/teste-gpu.rc"
      printf '%02d teste-gpu papel=erro-em-gpu rc=%s\n' "$SEQ" "$rc_t" >> "$SAIDA/ordem.txt"
      [ "$rc_t" = 0 ] || nr "teste de erro em GPU falhou (rc $rc_t): ver teste-gpu.txt"
    fi
    {
      echo "modelo=$MODELOS"; echo "n_repete=3"
      echo "binario_original=$(sha "$BIN_BASE")"; echo "binario_candidato=$(sha "$BIN_CAND")"
      echo "gancho_original_sha=$(sha "$SAIDA/gancho-original/gancho-logits-$MODELOS.json")"; echo "gancho_candidato_sha=$(sha "$SAIDA/gancho-candidato/gancho-logits-$MODELOS.json")"
      if [ $TESTE_GPU = 1 ]; then
        echo "teste_gpu_commit=$(git -C "$WT_CAND" rev-parse HEAD 2> /dev/null)"
        echo "teste_gpu_tmpdir=${T_TMP_GPU:-}"
        echo "teste_gpu_exe_sha=$(sha "$(ls -t "$WT_CAND"/target-f2d/release/deps/motor_passo_gpu-* 2> /dev/null | grep -v '\.d$' | head -n 1)")"
        echo "teste_gpu_exe_sha_reforma=$(sha "$(ls -t "$WT_CAND"/target-f2d/release/deps/motor_reforma_gpu-* 2> /dev/null | grep -v '\.d$' | head -n 1)")"
        echo "teste_gpu_exe_sha_o3=$(sha "$(ls -t "$WT_CAND"/target-f2d/release/deps/o3_kernels_gpu-* 2> /dev/null | grep -v '\.d$' | head -n 1)")"
      fi
      echo "ids_sha=$(awk '$1=="cmp-ids-prompt"{print $2}' "$SAIDA/integridade-antes.txt")"; echo "refs_sha=$(awk -v k="ref5c-$MODELOS" '$1==k{print $2}' "$SAIDA/integridade-antes.txt")"
    } > "$SAIDA/coleta-manifesto.txt"
  else
  CTRL_PAROU=
  CTRL_OK=$(planeja --planeja-controle --decorrido "$(decorrido)" --tetos "$TETOS9" --folga "$FOLGA" --teto "$LIMITE")
  for nn in 1 2 3 4 5 6 7 8 9; do
    b=${ORDEM_CTRL[$((nn - 1))]}; chave="ctrl-$MODELOS-L100-medio-$(printf '%02d' "$nn")-$b"
    if [ $ABORTA != 0 ]; then nr "$chave: janela interrompida por recusa de ambiente (código 2)"; continue; fi
    if [ "$CTRL_OK" != cabe ]; then nr "$chave: o controle de 9 processos não cabe no limite ($LIMITE s; planejador: $CTRL_OK)"; continue; fi
    if [ -n "$CTRL_PAROU" ]; then nr "$chave: controle encerrado ($CTRL_PAROU), sem repetição"; continue; fi
    roda_processo "$chave" 3 "$b" "$(teto_proc medio)"; rc=$?
    [ $rc = 0 ] || { [ $ABORTA != 0 ] || CTRL_PAROU="processo $chave saiu com código $rc"; }
  done
  [ $ABORTA = 0 ] || nr "janela interrompida por recusa de ambiente (código 2) no processo ${SEQ}"
  fi

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
  # ---- orçamento declarado (nao_rodou traz a chave da pasta de cada processo que não rodou: o extrator usa o prefixo ctrl-<modelo>-L100- para marcar o controle incompleto)
  {
    printf '{"modelos": ["%s"], "limite_s": %s, "folga_s": %s, "decorrido_s": %s, "cortes": [], "reps": {"controle": 3}, "cortado": false, "nao_rodou": [' "$MODELOS" "$LIMITE" "$FOLGA" "$(decorrido)"
    sep=; for c in "${NAO_RODOU[@]:-}"; do [ -n "$c" ] || continue; printf '%s"%s"' "$sep" "$(echo "$c" | tr -d '"\\')"; sep=', '; done
    printf '], "estimativas_sao_tetos_de_processo": true}\n'
  } > "$SAIDA/orcamento.json"
  if [ $COLETA = 1 ]; then
    python3 -I "$F2D_EXTRAI" --coleta-so --saida "$SAIDA" --ids "$IDS" --refs "$REFS" > "$SAIDA/f2d-extrai.out" 2>&1; RC_EXT=$?
  else
    python3 -I "$F2D_EXTRAI" --saida "$SAIDA" --ids "$IDS" --refs "$REFS" --textos "$TEXTOS/textos.json" ${COLETA_DIR:+--coleta "$COLETA_DIR"} > "$SAIDA/f2d-extrai.out" 2>&1; RC_EXT=$?
  fi
  if NINC=$(grep -c ' INCOERENTE$' "$SAIDA/prova-variavel.txt" 2> /dev/null); then :; else
    # grep -c sai 1 com 0 ocorrências (contagem válida); sai 2 se o arquivo não pôde ser lido (então o número é desconhecido)
    [ "${NINC:-}" = 0 ] || NINC="ILEGIVEL (prova-variavel.txt não pôde ser lido)"
  fi
  {
    [ $COLETA = 0 ] || echo "EXECUÇÃO DE COLETA (HC1b): execução própria, fora da janela de medida; não mede desempenho"
    echo "F2D (plano da O3) em L100, janela do modelo $MODELOS, critério sha256 $(sha "$CRIT_F2D"), roteiro sha256 $(sha "$AQUI/roda-f2d-janela.sh")"
    echo "resultado conforme criterio-f2d-20261112.md; só Lastro na 3080 Ti; binários: C0 (base) $(sha "$BIN_BASE"), C1 (candidato) $(sha "$BIN_CAND"), contexto 22d0bbe0 não rodou"
    echo "variáveis do instrumento: ${NINC:-0} processo(s) INCOERENTE(S) em prova-variavel.txt (D_O1 e D_O3 desligados nas três, B2_O3 as três ligadas)"
    echo "tetos de processo excedidos: $(wc -l < "$SAIDA/teto-excedido.txt" 2> /dev/null || echo 0) (teto-excedido.txt)"
    if [ $INTEGRO = 1 ]; then echo "integridade: íntegra (binários, scripts, modelo, textos, referências, ids e critérios iguais antes e depois$([ "${SEM_SHA_FIXO:-0}" = 1 ] && echo '; congelados NÃO conferidos: SEM_SHA_FIXO=1' || echo ' e aos congelados onde há'))"
    else echo "integridade: QUEBRADA: todas as H SEM PROVA, com as medidas brutas preservadas (e o extrator recusa sem a prova de potência)"; for n in "${NOTAS[@]}"; do echo "  $n"; done; fi
    echo "decorrido $(decorrido) s de $LIMITE s"
    echo "não rodou: ${#NAO_RODOU[@]} item(ns); ver orcamento.json"
    echo "extrator: $(cat "$SAIDA/f2d-extrai.out") (código $RC_EXT)"
    echo "processos executados: $SEQ (ordem.txt; plano em plano.txt; prova das variáveis em prova-variavel.txt)"
  } > "$SAIDA/decisao.txt"
  cat "$SAIDA/decisao.txt"
  exit 0
fi

# ---------------------------------------------------------------- modo externo (portões, janela e limite)
case $LIMITE in ''|*[!0-9]*) recusa "limite inválido: $LIMITE";; esac
[ "$LIMITE" -ge 1 ] && [ "$LIMITE" -le $LIMITE_MAX ] || recusa "limite fora de 1..$LIMITE_MAX s: $LIMITE"
[ "${PRIMEIRA_PALAVRA_AUTORIZADA:-}" = 1 ] || recusa "portão fechado: sem PRIMEIRA_PALAVRA_AUTORIZADA=1 (o 'autorizo' do Breno que nomeia a janela)"
[ "${CUDA_VISIBLE_DEVICES:-}" = "$UUID" ] || recusa "CUDA_VISIBLE_DEVICES tem de ser $UUID (a 3080 Ti; a 4090 e a 5090 nunca); veio '${CUDA_VISIBLE_DEVICES:-}'"
[ -x "$BIN_BASE" ] && [ -x "$BIN_CAND" ] || recusa "binário não executável: $BIN_BASE ou $BIN_CAND"
[ -x "$JANELA" ] && [ -x "$LIMITE_SH" ] && [ -x "$LASTRO_SH" ] && [ -x "$MONITOR" ] || recusa "faltam janela-e4b.sh, roda-com-limite.sh, roda-cmp-lastro.sh ou roda-cmp-monitor.sh executáveis"
[ -f "$EXTRAI" ] && [ -f "$PERFIL_EXTRAI" ] && [ -f "$INSTR_EXTRAI" ] && [ -f "$F2_EXTRAI" ] && [ -f "$F2B_EXTRAI" ] && [ -f "$F2D_EXTRAI" ] || recusa "faltam cmp-extrai.py, perfil-extrai.py, instrumento-extrai.py, f2-extrai.py, f2b-extrai-v3.py ou f2d-extrai.py"
[ -f "$CRIT_F2D" ] || recusa "falta o critério da F2D: $CRIT_F2D"
[ -n "$ESTADO" ] || ESTADO=$SAIDA-estado-janela
for d in "$SAIDA" "$ESTADO"; do
  if [ -e "$d" ]; then
    [ -d "$d" ] && LS_D=$(ls -A "$d" 2>&1) && [ -z "$LS_D" ] || recusa "$d já existe e não está vazio (ou não pôde ser lido): uma janela nova pede pasta nova"
  fi
done
SAIDA=$(realpath -m "$SAIDA") && BIN_BASE=$(realpath "$BIN_BASE") && BIN_CAND=$(realpath "$BIN_CAND") && [ -n "$SAIDA" ] && [ -n "$BIN_BASE" ] && [ -n "$BIN_CAND" ] || recusa "realpath falhou para --saida ou para os binários; nada foi tocado"
# sha dos binários (e --verifica-fixos do pacote todo) ANTES de parar timers/voxtype: nada é tocado se algo não bater
confere_binarios || recusa "binário com sha diferente do FIXO (ou ainda PENDENTE de congelamento): base $(sha "$BIN_BASE") contra ${FIXO[binario-base]}, candidato $(sha "$BIN_CAND") contra ${FIXO[binario-cand]}; nada foi tocado (sem envelope, timers, processos)"
if [ "${SEM_SHA_FIXO:-0}" != 1 ]; then
  # captura a saída E o status do verificador antes de qualquer filtro (um pipe com grep mascararia o código de saída dele)
  VF=$("$AQUI/roda-f2d-janela.sh" --verifica-fixos 2>&1); RC_VF=$?
  [ $RC_VF = 0 ] || recusa "--verifica-fixos falhou (código $RC_VF) antes de tocar em envelope, monitor, timers ou GPU (algum arquivo congelado mudou, sumiu ou está PENDENTE); nada foi tocado: $(echo "$VF" | grep -E 'FALHA|AUSENTE|PENDENTE' | head -n 3 | tr '\n' ' ')"
fi
if [ $TESTE_GPU = 1 ]; then
  WT_MSG=$(confere_wt) || recusa "--worktree-cand recusado antes de tocar em timers ou na GPU: $WT_MSG"
fi
# cargo test / processo CUDA em execução: recusa ANTES de tocar em timers ou na GPU
PROIB=$(proc_proibidos) || recusa "processo que pode usar a 3080 Ti em execução (cargo test, binário de teste do cargo, primeira-palavra ou llama.*) ou lista de processos ilegível: feche antes da janela; nada foi tocado: $(echo "$PROIB" | head -n 3 | tr '\n' ' ')"
POT_PRE=$(le_pot) || POT_PRE=
[ "$POT_PRE" = "$POT_OK" ] || recusa "power.limit da 3080 Ti tem de ser $POT_OK (lido: '${POT_PRE:-ilegível}'); a série vale só no limite vigente; nada foi tocado, não mexa no limite por este roteiro"
PREFLIGHT=$("$MONITOR" preflight 2>&1) || recusa "processo(s) de compute na 3080 Ti que precisam ser FECHADOS antes da janela (voxtype, processo de GPU do Chrome...): $PREFLIGHT"
# bloco obrigatório (nove tetos + folga): não cabendo no limite pedido, a janela NÃO COMEÇA: SEM PROVA da unidade inteira, registrada, sem repetição automática e sem encolher o controle
NEC=$(bloco_obrig_s)
if [ "$NEC" -gt "$LIMITE" ]; then
  mkdir -p "$SAIDA" || exit 2
  { echo "SEM PROVA da unidade inteira: a janela do modelo $MODELOS NÃO COMEÇOU (o bloco obrigatório do controle não cabe)"
    echo "necessário (tetos): $NEC s = 9 processos x $(teto_proc medio) + folga $FOLGA; limite pedido: $LIMITE s"
    echo "nada foi tocado (timers, GPU, voxtype); sem repetição automática; o controle não é encolhido"; } > "$SAIDA/decisao.txt"
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
    echo "binário base $(sha "$BIN_BASE") candidato $(sha "$BIN_CAND")"
  } > "$2" 2>&1 || true
}
EXTRA_INT=(); [ $COLETA = 0 ] || EXTRA_INT+=(--coleta); [ $TESTE_GPU = 0 ] || EXTRA_INT+=(--teste-gpu --worktree-cand "$WT_CAND"); [ -z "$COLETA_DIR" ] || EXTRA_INT+=(--coleta-dir "$COLETA_DIR")
PASSA=("CUDA_VISIBLE_DEVICES=$UUID" PRIMEIRA_PALAVRA_AUTORIZADA=1)
for v in MODELO_1B MODELO_3B TEXTOS_5C REF_5C IDS_CMP CRITERIO_F2D PLANO_O3 DECISAO_O3 F2B_EXTRAI_PY F2D_EXTRAI_PY CRITERIO_CMP CRITERIO_D1 CRITERIO_D2 SEM_SHA_FIXO CMP_MODO_TESTE NVIDIA_SMI PROC_STAT CMP_CPU_JANELA CMP_CPU_ESPERA LASTRO_SH_CMP EXTRAI_PY PERFIL_EXTRAI_PY INSTR_EXTRAI_PY F2_EXTRAI_PY \
         CMP_RELOGIO_ARQ CMP_FOLGA CMP_SEM_MEMORIA MONITOR_SH ESPERA_PRONTO LOADAVG_ARQ PROC_LISTA F2D_TETO_FORCADO F2D_CARGO_TESTE LASTRO_PERFIL_BLOCO LASTRO_PERFIL_F2 LASTRO_PERFIL_F2B LASTRO_GANCHO_LOGITS $(compgen -v | grep -E '^CMP_(EST|FAKE)_' || true); do
  [ -z "${!v+x}" ] || PASSA+=("$v=${!v}")
done
lista_systemd() {
  "${SYSTEMCTL:-systemctl}" --user list-unit-files --type=service,timer > "$SAIDA.unidades-$1.txt" 2>&1 || true
  "${SYSTEMCTL:-systemctl}" --user list-timers --all > "$SAIDA.timers-$1.txt" 2>&1 || true
}
lista_systemd antes
coleta início "$SAIDA.carga-antes.txt"
"$JANELA" --estado "$ESTADO" "${TIMERS[@]}" --limite $((LIMITE + 120)) --kill-apos 30 -- \
  "$LIMITE_SH" --limite "$LIMITE" -- env "${PASSA[@]}" "$AQUI/roda-f2d-janela.sh" --interno --modelos "$MODELOS" --saida "$SAIDA" --bin-base "$BIN_BASE" --bin-cand "$BIN_CAND" --limite "$LIMITE" "${EXTRA_INT[@]}"
RC=$?
coleta fim "$SAIDA.carga-depois.txt"
lista_systemd depois
MSG="$([ $COLETA = 1 ] && echo coleta || echo janela) da F2D (plano da O3) do Lastro ($MODELOS) terminou (código $RC): $SAIDA"
echo "$MSG" > "$SAIDA.aviso.txt"
if [ -n "${CMP_AVISO_CMD:-}" ]; then "$CMP_AVISO_CMD" "$MSG" > /dev/null 2>&1 || true
elif command -v notify-send > /dev/null 2>&1; then notify-send "F2D (Lastro)" "$MSG" > /dev/null 2>&1 || true; fi
exit $RC
