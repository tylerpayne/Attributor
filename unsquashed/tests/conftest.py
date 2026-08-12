"""Offline fixtures: a tiny random Llama and a from-scratch fast tokenizer with
a chat template, so the full pipeline is testable without downloads."""

import pytest

CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Paris is the capital of France. Berlin is the capital of Germany.",
    "Attention rollout composes attention matrices across layers.",
    "Documents: the answer to the question is found in the supporting sentences.",
    "Napoleon Bonaparte was a French military officer and statesman.",
    "The mitochondria is the powerhouse of the cell.",
    "0123456789 abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    'Punctuation: , . ; : ! ? " \' ( ) [ ] { } - _ / \\ n/a',
]

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|{{ message['role'] }}|>\n{{ message['content'] }}<|end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


@pytest.fixture(scope="session")
def tiny_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=600,
        special_tokens=["<unk>", "<s>", "</s>"],
        show_progress=False,
    )
    tok.train_from_iterator(CORPUS * 20, trainer)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|user|>", "<|assistant|>", "<|end|>"]}
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


@pytest.fixture(scope="session")
def tiny_model(tiny_tokenizer):
    import torch
    from transformers import AutoModelForCausalLM, LlamaConfig

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(tiny_tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,  # exercise GQA
        max_position_embeddings=1024,
    )
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    model.eval()
    return model
