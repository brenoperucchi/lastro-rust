# Registro da janela de comparação 1B de 10/11/2026 16:26 (invalidada)

Registro posterior, fora do critério congelado (o sha do critério da comparação, 954ec594ec51d1ccff3239c8dba33f86a01363b82330181b2a9ced918abdcd5f, é o que as janelas válidas citam; por isso a Emenda 11 não foi reeditada).

A janela s3080-cmp-1b-real-20261110-1626 teve duas causas independentes e não produziu dado válido:

1. As células do llama.cpp abortaram com `CUDA error: no kernel image is available for execution on the device`: o build usado tinha kernels só para a arquitetura 89 (4090). Tratado na Emenda 11 (recompilação no mesmo commit 0893ec9 com as arquiteturas 86 e 89).
2. A janela foi encerrada por TERM às 16:28:38, depois de 2 min 20 s, e a restauração dos timers falhou (omabackup-sync, omabackup-push e inventory-collect ficaram inativos; recuperados depois com recupera-timers-e4b.sh, resultado RECUPERAÇÃO OK). Causa apurada pelo coordenador: o Breno desligou a máquina de forma limpa às 16:28:42, com o systemd parando os serviços. Confere em `journalctl --list-boots`: o boot de 12:13 termina às 16:28:42 e o seguinte começa às 16:44:13; o fim do journal do boot anterior mostra as desmontagens normais (Unmounted /home), sem Xid.

As janelas válidas da série são as de 17:05 (s3080-cmp-1b/3b e s3080-perfil-1b/3b, sufixo 20261110-1705). A janela de 16:13 (recusa no preflight do pmon, nada medido) também não conta.
