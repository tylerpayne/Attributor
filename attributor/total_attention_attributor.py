import torch
from transformers import LlamaForCausalLM

from attributor.attribution import Attribution


class Attributor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._attention_head_weights = None

    def _get_attention_head_weights(self):
        if self._attention_head_weights is None:
            if isinstance(self.model, LlamaForCausalLM):
                output_projections = [layer.self_attn.o_proj.weight for layer in self.model.model.layers]
                per_head_output_projections = [o_proj.reshape(self.model.config.num_attention_heads, -1, o_proj.shape[-1]) for o_proj in output_projections]
                attention_head_weights = [torch.linalg.matrix_norm(o_proj.type(torch.float32), dim=[1,2]).unsqueeze(-1).unsqueeze(-1) for o_proj in per_head_output_projections]
                self._attention_head_weights = [a / a.sum() for a in attention_head_weights]
                
            else:
                raise NotImplementedError(type(self.model).__name__)
        
        return self._attention_head_weights
        

    def attend(self, tokens: torch.Tensor):
        assert len(tokens.shape) == 1, "Only tokens tensors with rank 1 are supported"

        with torch.no_grad():
            tokens = tokens.unsqueeze(0)
            outputs = self.model(tokens, output_attentions=True)
            return outputs.attentions

    # @torch.compile(fullgraph=True, dynamic=False)
    def forward(self, inputs: torch.Tensor, attention: torch.Tensor, attention_head_weights: torch.Tensor):
        """
        Simple multi-head self attention with residual connection
        """

        # Remove batch dimension (which must be one), now A is [heads, len, len]
        A = attention.squeeze(0).type(torch.float32)
        A = torch.multiply(A, attention_head_weights)
        A = A.sum(axis=0)
        A /= A.sum(axis=-1, keepdim=True)
        
        # attention over inputs, Y is [heads, len, len]
        Y = torch.matmul(A, inputs)
        
        # Post-attention residual
        # Y += inputs

        # Post-mlp residual has no effect because it just Y += Y, but we will normalize rows
        # Normalize rows to 1
        Y /= Y.sum(axis=-1, keepdim=True)

        return Y

    # @torch.compile
    def attribute(self, attentions):
        n = attentions[0].shape[2]
        # Values will get close to 0, use lots of precision
        # Y = torch.eye(n, n, dtype=torch.float32).to(attentions[0].device)
        # for A, o_proj in zip(attentions, self._get_attention_head_weights()):
        #     Y = self.forward(Y, A, o_proj)

        Y = torch.stack(attentions, 0).sum(axis=(0,1,2))
        Y /= Y.sum(axis=-1)

        # We now have Y[i, j] = amount that token j (input) was attended to when generating token i+1 (output), so we need to roll the output axis forward 1
        Y = torch.roll(Y, 1, 0)
        # output 0 has no input, it is just given.
        Y[0, :] = 0
        return Y
    
    def __call__(self, tokens):
        with torch.no_grad():
            tokens = tokens.squeeze(0).to(self.model.device)
            attentions = self.attend(tokens)
            attributions = self.attribute(attentions)
            return Attribution(self.model, self.tokenizer, tokens, attributions)
