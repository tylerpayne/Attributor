from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class AttributionSpan:
    effect_indices: list[int] = field(kw_only=True)
    treatment_indices: list[int] = field(kw_only=True)
    attribution: float = field(kw_only=True)

    def effect_text(self, tokenizer, tokens: torch.Tensor, return_range=False) -> str:
        tokens = tokens.squeeze(0)
        assert len(tokens.shape) == 1, "Only tokens tensors with rank 1 are supported"
        text = tokenizer.decode(tokens[self.effect_indices])
        if return_range:
            return text, (self.effect_indices[0], self.effect_indices[-1])
        else:
            return text

    def treatment_text(
        self, tokenizer, tokens: torch.Tensor, return_range=False
    ) -> str:
        tokens = tokens.squeeze(0)
        assert len(tokens.shape) == 1, "Only tokens tensors with rank 1 are supported"
        return f"{tokenizer.decode(tokens[self.treatment_indices])}({self.treatment_indices[0]}-{self.treatment_indices[-1]})"

    def pretty_print(self, tokenizer, tokens) -> str:
        return f"'{self.effect_text(tokenizer, tokens)}' attributed to '{self.treatment_text(tokenizer, tokens)}' with strength {self.attribution}"
