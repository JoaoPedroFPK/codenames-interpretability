"""Base decoders (T5) and the third instruct decoder Llama-3.1-8B (T6):
registry, prompt template, and the maps every offline stage reads."""

import pytest
from transformers import AutoTokenizer

from codenames.cli import MODEL_REGISTRY, _resolve_loader, build_parser
from codenames.prompts import build_instruction_body, build_prompt

TINY_LLAMA = "trl-internal-testing/tiny-LlamaForCausalLM-3.1"


def test_registry_has_the_new_decoders():
    for key in ("mistral_base", "qwen_base", "llama"):
        assert key in MODEL_REGISTRY, key
        assert callable(_resolve_loader(key))


def test_loader_constants_state_the_checkpoints():
    from codenames.models import llama, mistral_base, qwen_base
    assert mistral_base.MODEL_NAME == "mistral-community/Mistral-7B-v0.2"
    assert mistral_base.PREFIX == "mistral_base"
    assert qwen_base.MODEL_NAME == "Qwen/Qwen2.5-7B"
    assert qwen_base.PREFIX == "qwen_base"
    assert llama.MODEL_NAME == "meta-llama/Llama-3.1-8B-Instruct"
    assert llama.PREFIX == "llama"
    # base models are read through the geometry only: raw instruction body
    assert mistral_base.CHAT_TEMPLATE_STRATEGY == qwen_base.CHAT_TEMPLATE_STRATEGY == "raw"
    assert llama.CHAT_TEMPLATE_STRATEGY == "llama3"


@pytest.fixture(scope="module")
def llama_tok():
    return AutoTokenizer.from_pretrained(TINY_LLAMA)


def test_llama3_template_contains_the_body_verbatim_once(llama_tok):
    hint, cands = "water", ["MOON", "SEA", "SHIP"]
    body, _ = build_instruction_body(hint=hint, candidates=cands, giver_features={},
                                     use_social_context=False)
    prompt, _ = build_prompt(hint=hint, candidates=cands, giver_features={},
                             use_social_context=False, tokenizer=llama_tok,
                             chat_template_strategy="llama3")
    assert prompt.count(body) == 1
    assert "<|start_header_id|>assistant<|end_header_id|>" in prompt


def test_llama3_prompt_tokenises_with_exactly_one_bos(llama_tok):
    """The rendered template starts with <|begin_of_text|> and the tokenizer
    adds one too; the strategy strips the literal so a forward pass sees a
    single BOS (unlike the thesis's Mistral prompts, which carry two)."""
    prompt, _ = build_prompt(hint="water", candidates=["MOON", "SEA"], giver_features={},
                             use_social_context=False, tokenizer=llama_tok,
                             chat_template_strategy="llama3")
    assert not prompt.startswith("<|begin_of_text|>")
    ids = llama_tok(prompt)["input_ids"]
    assert ids[0] == llama_tok.bos_token_id and ids[1] != llama_tok.bos_token_id


def test_llama3_prompt_is_date_stable(llama_tok):
    """Llama-3 templates stamp a date into the system header; it must be a
    frozen string, or the prompt (and every hidden state) changes daily."""
    prompt, _ = build_prompt(hint="water", candidates=["MOON", "SEA"], giver_features={},
                             use_social_context=False, tokenizer=llama_tok,
                             chat_template_strategy="llama3")
    assert "Today Date: 26 Jul 2024" in prompt


def test_flash_attn_and_lens_scope_include_the_new_decoders():
    import codenames.cli as cli
    for key in ("mistral_base", "qwen_base", "llama"):
        assert key in cli._FLASH_ATTN_MODELS
        assert key in cli._LENS_MODELS and key in cli._LENS_TOKENIZERS
        assert key in cli._LENS_PREFIXES and key in cli._LENS_CHAT_TEMPLATES
    args = build_parser().parse_args(["run", "--model", "llama", "--dataset", "/tmp/d.csv",
                                      "--output-dir", "/tmp/o"])
    assert args.model == "llama"


def test_causal_tier_and_analysis_maps_know_llama():
    from codenames.analysis.tables import GENERATION_PREFIXES, MODEL_FAMILY
    from codenames.causal import runner
    import codenames.cli as cli
    assert "llama" in cli._CAUSAL_MODELS and runner._prefix("llama") == "llama"
    assert MODEL_FAMILY["llama"] == "llama"
    assert MODEL_FAMILY["mistral_base"] == "mistral" and MODEL_FAMILY["qwen_base"] == "qwen"
    assert "llama" in GENERATION_PREFIXES
    assert "mistral_base" not in GENERATION_PREFIXES   # geometry only
