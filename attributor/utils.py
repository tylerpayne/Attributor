import torch


def find(tokens: torch.Tensor, subtokens: torch.Tensor, tolerance=5):
    assert len(tokens.shape) == 1, "Only tokens tensors with rank 1 are supported"
    tokens_count = tokens.shape[0]

    if not isinstance(subtokens, list):
        assert (
            len(subtokens.shape) == 1
        ), "Only subtokens tensors with rank 1 are supported"
        subtokens_count = subtokens.shape[0]
    else:
        subtokens_count = len(subtokens)

    assert subtokens_count <= tokens_count, "Subtokens must be smaller than tokens"

    for i in range(tokens_count - subtokens_count):
        score = 0
        for j in range(subtokens_count):
            if tokens[i + j] == subtokens[j]:
                score += 1
            elif j > tolerance and j - score > tolerance:
                break
        if subtokens_count - score <= tolerance:
            return i, j
    raise IndexError("Subtokens not found in tokens")


def tokenize(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )


def generate(model, tokenizer, generation_config, messages):
    prompt_tokens = tokenize(tokenizer, messages)
    prompt_tokens = prompt_tokens.to(model.device)
    return model.generate(
        prompt_tokens, tokenizer=tokenizer, generation_config=generation_config
    )
