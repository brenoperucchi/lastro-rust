#!/bin/bash
# B0: cada mutante muda uma linha de rust/moe-rt/src/cache/core.rs, roda os testes do núcleo e tem de ser reprovado pelo teste
# esperado. O core.rs original é guardado antes e restaurado no fim (o sha256 tem de bater com o de antes).
set -u
cd /home/brenoperucchi/Devs/strata-opencode/rust/moe-rt
export CUDA_TOOLKIT_PATH=/opt/cuda PATH=/opt/cuda/bin:$PATH MOE_RT_REQUIRE_GPU=1
F=src/cache/core.rs
GUARDA=$(mktemp)
cp $F $GUARDA
H0=$(sha256sum < $F)
mutante() { # nome, teste esperado, trecho original, trecho mutante
  cp $GUARDA $F
  python3 - "$3" "$4" <<'PY'
import sys
p = 'src/cache/core.rs'; s = open(p).read(); a, b = sys.argv[1], sys.argv[2]
assert s.count(a) == 1, a
open(p, 'w').write(s.replace(a, b))
PY
  echo "== $1"
  echo "-  $3"
  echo "+  $4"
  out=$(/usr/bin/cargo test --release --lib cache -- --test-threads=1 2>&1)
  echo "$out" | command grep -E '^test .*(ok|FAILED)$|^test result'
  if echo "$out" | command grep -qE "^test cache::core::tests::$2 \.\.\. FAILED$"; then echo "resultado: reprovado por $2 (esperado)"; else echo "resultado: NÃO reprovado por $2"; fi
  echo
}
mutante "pin_layer sem os slots carregando" slot_carregando_fica_fora_do_mapa_e_ganha_o_uso_da_camada "if keys.contains(key) {" "if false && keys.contains(key) {"
mutante "settle sem consultar o evento" slot_carregando_fica_fora_do_mapa_e_ganha_o_uso_da_camada "if !self.slots[s].ready.query()? {" "if false && !self.slots[s].ready.query()? {"
mutante "resident sem a geração" conferencias_em_tempo_de_execucao "State::Resident { .. } if s.gen == id.gen =>" "State::Resident { .. } =>"
mutante "victim sem o relógio do Pins" conferencias_em_tempo_de_execucao "if pins.clock != self.clock {" "if false {"
mutante "copy_in_background sem o evento de ordem" copia_em_segundo_plano_espera_a_leitura_do_slot "        self.order.record(&self.compute)?;
        self.copy.wait_event(&self.order)?;" ""
mutante "victim sem conferir a instância" conferencias_em_tempo_de_execucao "if pins.owner != self.owner || idle.e.owner != self.owner {" "if false {"
mutante "cópia sem conferir a vítima" conferencias_em_tempo_de_execucao "if s.state != State::Reserved || s.gen != v.gen {" "if false {"
cp $GUARDA $F
rm $GUARDA
H1=$(sha256sum < $F)
[ "$H0" = "$H1" ] && echo "core.rs restaurado (sha256 igual ao de antes)" || echo "ERRO: core.rs diferente do original"
/usr/bin/cargo test --release --lib cache 2>&1 | command grep '^test result'
