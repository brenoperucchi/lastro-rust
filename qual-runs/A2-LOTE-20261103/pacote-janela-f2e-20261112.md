# Pacote das execuções da F2E (plano da O3 na RTX 5090): coletas e janelas, 12/11/2026


> **CONGELAMENTO PENDENTE, NÃO ABRIR (13/11).** Os binários C0/C1 abaixo foram feitos antes da trava de GPU por lista de UUIDs e recusam a 5090. Depois da aprovação da trava, C0 e C1 serão refeitos (mesmo patch nos dois, diff C0 a C1 restrito à O3) e os commits, binários, shas, FIXO e comandos abaixo serão atualizados; só então o critério passa a `Estado do congelamento: COMPLETO`. Enquanto estiver PENDENTE, `abre-f2e-coleta.sh`, `abre-f2e-medicao.sh` e o roteiro recusam com código 2 sem tocar em nada. Os comandos de abertura desta página valem só depois do congelamento.
>
> **Pendência de push (decisão do Breno):** o UUID real da 5090 ainda consta de commits locais anteriores (`83ecfe3`, `73bdbac` e seguintes, até o commit da trava de UUID, que o remove do estado atual). O HEAD não o contém (`git grep` vazio). Antes de qualquer push é preciso decidir sobre reescrever esses commits; nada foi reescrito.

## O que é
A medição da O3 (o gemv de V e o RoPE de K gravam direto na linha `cur` do cache K/V, sem as duas cópias device-para-device) na RTX 5090 de outro computador, chamado aqui de "maquina 5090". Série NOVA só da 5090, sem comparar com a 3080 Ti; a prova de bits e o teste de erro em GPU da 3080 Ti não valem aqui e se refazem na 5090 antes da medição (decisão do Breno de 12/11, relatada em `decisao-f2e-5090-20261112.txt`). Mesma estrutura da F2D (`criterio-f2d-20261112.md`, `pacote-janela-f2d-20261112.md`): braços D_O1 (C0), D_O3 (C1) e B2_O3 (C1), quadrado latino 3x3, HC1 a HC6, limites +1,5% no prefill e +2% no decode, efeito distinguível só abaixo de -1%. Critério: `criterio-f2e-20261112.md`; plano v2: `plano-o3-cache-direto-20261112.md`.

Tudo roda NA maquina 5090, por ssh, a partir do clone do repositório em `~/lastro-rust` (a fonte; o clone local só edita e commita). Nada de nome ou endereço da máquina em arquivo versionado ou resultado publicado.

## Máquina e portões
- GPU: RTX 5090, rótulo `gpu_5090` (o UUID real NÃO entra em commit nem em resultado): fica num arquivo LOCAL fora do git, `qual-runs/A2-LOTE-20261103/gpu-local.uuids` no clone da máquina (um UUID por linha, formato `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`; caminho alternativo em `F2E_GPU_ARQ`), listado em `.gitignore` e em `.git/info/exclude` do clone. Os scripts leem o UUID dali (arquivo ausente ou sem UUID válido = recusa 2 antes de tocar em qualquer coisa) e passam o mesmo arquivo ao motor em `LASTRO_UUIDS_EXTRA_ARQ`: a trava do motor aceita só a 3080 Ti (constante) ou um UUID desse arquivo cujo sha256 conste da lista versionada no código (`SHA_UUID_GPU_5090`, rótulo `gpu_5090`; sha256 `88269ef39d73981809645a172b26d68c98ae0470e0a38c8fb1fed4772d17e1f2`; o UUID tem 128 bits aleatórios, então o hash não permite recuperar o valor na prática), um único UUID em `CUDA_VISIBLE_DEVICES`; sem o arquivo só a 3080 Ti passa, e a 4090 ou qualquer outro UUID no arquivo é recusado. Os scripts de abertura e o roteiro conferem `sha256sum` do UUID contra essa mesma constante antes de tocar no Strata (recusa 2). As mensagens de recusa não trazem o valor de `CUDA_VISIBLE_DEVICES`, só a categoria do erro e o comprimento; as consultas de carga gravam o índice da GPU, não o UUID. Os resultados registram `uuid=gpu_5090`; `CUDA_VISIBLE_DEVICES` tem de ser esse UUID. Limite de potência 600.00 W lido do `nvidia-smi` no preflight e ao fim (valor vigente em 12/11; outro valor recusa). 28 núcleos.
- Sem voxtype. A GPU é ocupada pelo `strata.service` (unidade de usuário, cerca de 30 GB): o Strata é parado e religado SÓ pelo `~/janela-gpu-strata.sh --teto SEG -- <comando>` (log em `~/lastro-janelas`). Os dois scripts de abertura entregam a execução a ele; o roteiro não mexe em serviço nenhum.
- Envelope (versão da rodada 181, cópia versionada em `janela-gpu-strata-f2e.sh`, instalada também como `~/janela-gpu-strata.sh`), sha256 `3e85f052327e20b78c0897f3b9bad46d358c87d8d788dd186a8502eb6e5a96d4`: a obrigação de religar é armada ANTES do stop (TERM, HUP ou INT durante o stop não deixam a produção parada); no EXIT encerra o comando e qualquer unidade `lastro-limite-*` (cgroup vazio confirmado) e SÓ ENTÃO religa; a confirmação de término das unidades FALHA FECHADA (erro de `list-units`/`is-active`, resposta ilegível ou cgroup existente mas ilegível = incerto, nunca "nada vivo"; só vale lista vazia vinda de consulta bem-sucedida, repetida depois de parar as unidades); processo ou unidade remanescente, término não confirmado, NÃO religa e sai 5, com aviso explícito no log, no stdout/stderr e num arquivo `~/lastro-janelas/ATENCAO-STRATA-PARADO-<carimbo>.txt` (a produção fica parada à espera de decisão humana; o exec confere o código do envelope ao fim de cada execução). Com as unidades do Lastro CONFIRMADAS encerradas, GPU ocupada ou `nvidia-smi` ilegível NÃO impede o start: aviso bem visível (log, stdout, ATENCAO) e o envelope segue para o start (produção parada com certeza é pior); se o start falhar, código 6. GPU ocupada ou `nvidia-smi` falho já no portão de início sai 3 sem rodar o comando e religa o Strata; start que falha ou Strata que não volta a `active` em até 60 s (3 tentativas) sai 6 (um código do comando diferente de 0 é preservado e a falha de religação vai para o log); espera a GPU esvaziar antes do start; `timeout -k 30`. O roteiro usa `roda-com-limite-f2e.sh` (cópia com cancelamento da unidade por TERM/HUP/INT; o `roda-com-limite.sh` das outras fases fica intacto), sha256 `7ff702fac093ae342c19795e74036c17829209241e0ceed34dc793c783d07aff`.
- Limpeza: os testes não apagam nada; cada um imprime ao fim `pasta temporaria deste teste (apagar a mao, caminho literal): <caminho>` (pastas `tmp.*` em `~/.cache/lastro-testes`). Apagar depois, uma a uma, por esse caminho literal impresso (nunca por caminho montado com variável).
- Procedência travada antes de tocar no Strata: os scripts de abertura recusam (código 2) se o `~/janela-gpu-strata.sh` não tiver o sha acima, se o clone tiver arquivo rastreado modificado, ou se `F2E_COMMIT` (OBRIGATÓRIO, o commit aprovado) faltar ou não for o HEAD do clone, ou se `git status` falhar (falha de leitura não vale como árvore limpa; no `ambiente.txt` fica `clone_arvore_limpa=erro`). O roteiro recusa `F2E_MODO_TESTE`, `F2E_ENVELOPE_SHA_ESPERADO` e `F2E_JANELA_GPU` no modo real. Commit do clone, árvore limpa, sha do roteiro e sha do envelope vão para `ambiente.txt` e, de lá, para `procedencia_clone` em `f2e-resultado.json` e `f2e-coleta-resultado.json` e para uma linha no `.md`.
- Portão de CPU abaixo de 10% (janela de 3 s em `/proc/stat`, espera de até 20 s) medido na própria máquina antes de cada célula; célula que não passa é SEM PROVA, sem repetição.
- Cutile/JIT: a placa é consultada no driver em tempo de execução (sm_120); os binários não têm alvo fixado na compilação. O script de abertura exporta `CUDA_TOOLKIT_PATH=/usr/local/cuda` e o `bin` dele no `PATH` para o `tileiras`, e `~/.cargo/env` para o `cargo` do teste de GPU.
- Teto de 1200 s por janela e modelo; teto de processo 60 s (1B) e 90 s (3B). O controle (9 processos) cabe inteiro ou os nove viram NÃO RODOU; sem repetição automática.

## Fonte e binários congelados
- Commit base C0: `7c25e1d126c0e36219e5f325131eb38040a645c3`. Commit candidato C1: `1fa8abb77fef9359391fc9bc323602512984094e` (código da O3 em `6f7d049`; mesmo candidato da F2D).
- Binário base (D_O1): sha256 `54c0979a1b91f3afa4a0119bc403f3bff3880d4236f7a8d2ddc6604c79c44ae4`, em `~/lastro-f2e/binarios/primeira-palavra-f2e-base-54c0979a`.
- Binário candidato (D_O3 e B2_O3): sha256 `d38a951d81c4144114d5c306c30093ec938551479d841f91ef65c2beed07f21f`, em `~/lastro-f2e/binarios/primeira-palavra-f2e-cand-d38a951d`.
- Construídos na maquina 5090, em worktrees fora do repositório e fora do /tmp (`~/lastro-f2e/wt-c0` e `wt-c1`), alvo de compilação próprio (`target-f2e` de cada worktree), `--release --bin primeira-palavra`, `CUDA_VISIBLE_DEVICES` vazio, sem rodar nada na GPU; cópias modo 555.
- Executáveis dos testes de GPU, `--no-run` em `~/lastro-f2e/wt-c1/target-f2e/release/deps/`: `motor_reforma_gpu-0273f9e3452a9799` (sha `0200af2e...`), `motor_passo_gpu-9aca5be16f40dc68` (sha `a105095e...`), `o3_kernels_gpu-b2483507a5044105` (sha `18ab08b3...`).
- Scripts novos (cópias derivadas, com UUID e potência da 5090): cadeia de extratores `f2e-cmp-extrai.py`, `f2e-perfil-extrai.py`, `f2e-instrumento-extrai.py`, `f2e-f2-extrai.py`, `f2e-f2b-extrai.py`, `f2e-extrai.py`; `roda-f2e-lastro.sh`, `roda-f2e-monitor.sh`, `roda-com-limite-f2e.sh`, `roda-f2e-janela.sh`. Os da F2C e da F2D ficam intocados. Todos no FIXO de `roda-f2e-janela.sh` (shas conferidos), exceto o próprio roteiro (autorreferente; o verificador o imprime à parte). Sha256 do roteiro nesta versão: `57808d3678bec120327190178f2bdb9271ee4d95e9e672eee7fc62f0d16ef9c7`.

## FIXO
`bash ~/lastro-rust/qual-runs/A2-LOTE-20261103/roda-f2e-janela.sh --verifica-fixos` (rodado na maquina 5090, sem GPU, só lê): código 0, todos OK, também para os modelos em `~/models` (mesmos shas dos usados antes). Conferido de novo no modo real, antes de qualquer processo ou GPU.

## Ordem das execuções
1. Coleta de bits (HC1b) e teste de erro em GPU, 1B e depois 3B (`abre-f2e-coleta.sh`), uma chamada do `janela-gpu-strata.sh`.
2. Medição 1B e depois 3B (`abre-f2e-medicao.sh`), outra chamada.
A medição recusa abrir (código 2, sem tocar no Strata) se a coleta de algum modelo não tiver HC1b `dentro` e o teste de GPU passado com procedência completa.

### Comando da coleta (1B+3B, com `motor_reforma_gpu`, `motor_passo_gpu` e `o3_kernels_gpu`)
Pré-requisito já cumprido: worktree `~/lastro-f2e/wt-c1` no commit candidato, limpo, com os três testes compilados. Rodar desprendido da sessão ssh, porque dura até uns 45 min contando o Strata parar e voltar:
```
ssh <maquina 5090> 'cd ~/lastro-rust/qual-runs/A2-LOTE-20261103 && F2E_COMMIT=<commit aprovado> nohup bash abre-f2e-coleta.sh ~/lastro-f2e/wt-c1 > ~/lastro-f2e/coleta.out 2>&1 &'
```
O script confere FIXOS, logs livres e os três executáveis ANTES de tocar na GPU, e chama `~/janela-gpu-strata.sh --teto 2700 -- bash abre-f2e-coleta.sh --interno <wt> <carimbo>`. Saídas: `qual-runs/A2-LOTE-20261103/s5090-f2e-coleta-{1b,3b}-real-<carimbo>/` (com `f2e-coleta-resultado.json` e `.md`) e o `.log` ao lado; o carimbo aparece em `~/lastro-f2e/coleta.out`.

### Comando da medição (1B e 3B)
Com as pastas das coletas aprovadas (as duas do carimbo acima):
```
ssh <maquina 5090> 'cd ~/lastro-rust/qual-runs/A2-LOTE-20261103 && F2E_COMMIT=<commit aprovado> nohup bash abre-f2e-medicao.sh $PWD/s5090-f2e-coleta-1b-real-<carimbo> $PWD/s5090-f2e-coleta-3b-real-<carimbo> > ~/lastro-f2e/medicao.out 2>&1 &'
```
Chama `~/janela-gpu-strata.sh --teto 2700 -- bash abre-f2e-medicao.sh --interno ...`; roda 1B e 3B com `roda-f2e-janela.sh --modelos <m> --saida s5090-f2e-<m>-real-<carimbo> --coleta-dir <coleta do mesmo modelo>`, de modo que o `f2e-resultado` de cada modelo já sai com o HC1b. Erro no 1B impede o 3B e conserva o código.

## Levar os resultados para o repositório local
Só os resultados, sem os arquivos da máquina (`ambiente.txt`, `carga-*.txt`, logs, células):
```
scp <maquina 5090>:lastro-rust/qual-runs/A2-LOTE-20261103/s5090-f2e-<m>-real-<carimbo>/f2e-resultado.{md,json} qual-runs/A2-LOTE-20261103/s5090-f2e-<m>-real-<carimbo>/
scp <maquina 5090>:lastro-rust/qual-runs/A2-LOTE-20261103/s5090-f2e-coleta-<m>-real-<carimbo>/f2e-coleta-resultado.{md,json} qual-runs/A2-LOTE-20261103/s5090-f2e-coleta-<m>-real-<carimbo>/
```
Antes de commitar, conferir com `grep -il "GPU-\|maquina\|brenoperucchi\|ssh" ` nos quatro arquivos de cada modelo que nenhum traz UUID, nome ou endereço.

## O que cada resultado significa
- HC1b dentro e teste de GPU passou nos dois modelos: a O3 é bit a bit igual à base na 5090 (logits, KV, ids), o caminho de erro deixa o motor utilizável e igual bit a bit, e a medição pode abrir.
- HC1b REGRESSÃO (qualquer bit diferente) ou teste de GPU falhou: a O3 não passa na 5090; o passo que causou se desiste (decisão 5 do Breno), sem afrouxar o critério.
- HC1b SEM PROVA: nada se conclui; a coleta se repete antes de medir.
- Medição, por métrica (prefill e passo de decode), referência D_O1 da mesma janela: "efeito distinguível" (razão abaixo de -1% e as 3 medianas de D_O3 todas abaixo das 3 de D_O1), "sem efeito distinguível" (qualquer outro caso fora da regressão; nulo não é falha do plano), "REGRESSÃO" (acima de +1,5% no prefill ou +2% no decode, HC6 SEM PROVA, erro de processo), "sem dado" ou SEM PROVA (controle incompleto, portão de CPU, integridade quebrada). HC2a e HC2b são descritivas.
- A série é da 5090 e não se compara com os números da 3080 Ti.
- Ids gerados são comparados com as referências 5c (HC1); se a 5090 divergir delas em qualquer id, é regressão do critério, não ajuste de referência. Se a própria base C0 já divergir da 5c na 5090, o resultado se relata como "base diverge da 5c na 5090 (plataforma)", sem atribuir à O3, e volta ao Breno; nenhum limite nem referência muda.

## Testes sem GPU (rodados na maquina 5090, `TMPDIR` em `~/.cache/lastro-testes`)
`test-f2e-extrai.py`, `test-f2e-f2b-extrai.py`, `test-f2e-extrai-e2e.py`, `test-roda-f2e-janela.sh`, `test-abre-f2e.sh` (travas de procedência), `test-janela-gpu-strata.sh` (envelope com `systemctl` e `nvidia-smi` simulados: TERM/HUP/INT durante o stop e entre o retorno e a confirmação, estado inicial inactive sem start, start falho, unidade que não encerra, teto externo), `test-roda-com-limite-f2e.sh` (cancelamento por sinal), `cargo test --release --lib` filtrado (`perfil_bloco`, `cache_direto`, `leitura_tests`, `gancho_coleta`, `reforma`, `geracao_regra`, alvo de compilação `~/lastro-f2e/target-lib`).
