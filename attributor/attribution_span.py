from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class AttributionSpan:
    output_indices: list[int] = field(kw_only=True)
    input_indices: list[int] = field(kw_only=True)
    attribution: float = field(kw_only=True)

    def output_text(self, tokenizer, tokens: torch.Tensor, return_range=False) -> str:
        tokens = tokens.squeeze(0)
        assert len(tokens.shape) == 1, "Only tokens tensors with rank 1 are supported"
        text = tokenizer.decode(tokens[self.output_indices])
        if return_range:
            return text, (self.output_indices[0], self.output_indices[-1])
        else:
            return text

    def input_text(
        self, tokenizer, tokens: torch.Tensor, return_range=False
    ) -> str:
        tokens = tokens.squeeze(0)
        assert len(tokens.shape) == 1, "Only tokens tensors with rank 1 are supported"
        return f"{tokenizer.decode(tokens[self.input_indices])}({self.input_indices[0]}-{self.input_indices[-1]})"

    def pretty_print(self, tokenizer, tokens) -> str:
        return f"'{self.output_text(tokenizer, tokens)}' attributed to '{self.input_text(tokenizer, tokens)}' with strength {self.attribution}"
