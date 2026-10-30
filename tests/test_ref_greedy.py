"""Referência gulosa do M1b (tools/ref_greedy.py).

Regressão de 29/10: o `generate` era chamado sem `attention_mask` e, com `pad_token_id = 0`, o transformers mascarava todo
token 0 do prompt, que no Granite também é o `<|end_of_text|>` que o chat template põe entre os turnos. A referência saía de
um prompt diferente do que o runtime recebe. O primeiro passo do `generate` tem de bater com um forward direto no mesmo
prompt.

Precisa de torch e transformers: roda com tools/.venv-ref/bin/python -m pytest; no Python do sistema é pulado.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import ref_greedy  # noqa: E402


def tiny_granite():
    cfg = transformers.GraniteMoeConfig(
        vocab_size=97, hidden_size=64, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2, max_position_embeddings=128,
        pad_token_id=0, bos_token_id=0, eos_token_id=0,
    )
    torch.manual_seed(1)
    return transformers.GraniteMoeForCausalLM(cfg).eval()


def test_primeiro_passo_bate_com_o_forward_mesmo_com_token_0_no_prompt():
    model = tiny_granite()
    for seed in range(8):
        g = torch.Generator().manual_seed(seed)
        ids = torch.randint(1, 97, (1, 24), generator=g)
        ids[0, [3, 9, 15, 20]] = 0  # como os <|end_of_text|> entre turnos
        with torch.inference_mode():
            logits = model(ids, use_cache=False).logits[0, -1]
        top = torch.topk(logits, 2).values
        gen, gaps = ref_greedy.greedy(model, ids, 4, 0)
        # O gap depende dos logits inteiros: com a máscara errada ele muda mesmo quando o argmax por acaso não muda.
        assert gaps[0] == pytest.approx(float(top[0] - top[1]), abs=1e-5), f"semente {seed}"
        assert gen[0] == int(logits.argmax()), f"semente {seed}"
        assert len(gen) == len(gaps) == 4


def test_gera_todos_os_tokens_pedidos_mesmo_quando_o_eos_e_o_mais_provavel():
    """A referência gulosa tem de ter tamanho fixo: o runtime gera N tokens sem parar no EOS, e uma referência mais curta
    deixaria a comparação passar sem cobrir os passos que faltam."""
    model = tiny_granite()
    # O generation_config herda o EOS do config; sem `eos_token_id=None` explícito, o generate pararia no primeiro token
    # (conferido no transformers 5.22.0). Este teste segura essa garantia numa atualização do transformers.
    assert model.generation_config.eos_token_id == 0
    # Empurra o token 0 (EOS no config) para o topo em todo passo.
    hook = model.lm_head.register_forward_hook(lambda _m, _i, out: out + torch.nn.functional.one_hot(torch.tensor(0), out.shape[-1]) * 100.0)
    try:
        ids = torch.randint(1, 97, (1, 12), generator=torch.Generator().manual_seed(3))
        gen, gaps = ref_greedy.greedy(model, ids, 6, 0)
    finally:
        hook.remove()
    assert gen == [0] * 6
    assert len(gaps) == 6
