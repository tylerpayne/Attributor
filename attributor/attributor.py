import torch

from attributor.attribution import Attribution


class Attributor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def attend(self, tokens: torch.Tensor):
        assert len(tokens.shape) == 1, "Only tokens tensors with rank 1 are supported"

        with torch.no_grad():
            tokens = tokens.unsqueeze(0)
            outputs = self.model(tokens, output_attentions=True)
            return outputs.attentions

    def forward(self, inputs: torch.Tensor, attention: torch.Tensor):
        """
        Simple multi-head self attention with residual connection
        """

        # Remove batch dimension (which must be one), now A is [heads, len, len]
        A = attention.squeeze(0).type(torch.float64)
        # Sum over attention heads (total attention from token i to j). now A is [len, len]
        A = A.sum(axis=0)

        # attention over inputs
        Y = torch.matmul(A, inputs)

        # Residual connections (1 before MLP, 1 after)
        Y += 2 * inputs

        # Normalize columns to 1
        Y /= Y.sum(axis=-1, keepdim=True)

        return Y

    # @torch.compile
    def attribute(self, attentions):
        n = attentions[0].shape[2]
        # Values will get close to 0, use lots of precision
        Y = torch.eye(n, n, dtype=torch.float64).to(attentions[0].device)
        for A in attentions:
            Y = self.forward(Y, A)

        # We now have Y[i, j] = amount that token j (treatement) was attended to when generating token i+1 (effect), so we need to roll the effect axis forward 1
        Y = torch.roll(Y, 1, 0)
        # effect 0 has no treatment, it is just given.
        Y[0, :] = 0
        return Y

    def __call__(self, tokens):
        tokens = tokens.squeeze(0).to(self.model.device)
        attentions = self.attend(tokens)
        attributions = self.attribute(attentions)
        return Attribution(self.model, self.tokenizer, tokens, attributions)
