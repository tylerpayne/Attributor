import torch
from transformers import LlamaForCausalLM

from server.models import AttentionHead, Layer, Model


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

    # @torch.compile
    def attribute(self, attentions):
        n = attentions[0].shape[2]

        X = torch.eye(n, n, dtype=torch.float32).to(attentions[0].device)
        model_input = X.tolist()
        model_flow = torch.eye(n, n, dtype=torch.float32).to(X.device)

        layers = []

        for A, o_proj in zip(attentions, self._get_attention_head_weights()):
            layer_input = X.tolist()

            A = A.squeeze(0).type(torch.float32)
            layer_attention_head_attentions = A.tolist()
            layer_attention_head_weights = o_proj.squeeze().tolist()
            # logger.debug(json.dumps(layer_attention_head_weights))
            Xp1 = torch.matmul(A, X)
            layer_attention_head_outputs = Xp1.tolist()

            attention_heads = [
                AttentionHead(
                    inputs = layer_input,
                    flow=layer_attention_head_attentions[i],
                    outputs=layer_attention_head_outputs[i]
                )
                for i in range(len(layer_attention_head_weights))
            ]

            layer_flow = torch.multiply(A, o_proj)
            layer_flow = layer_flow.sum(axis=0)
            layer_flow /= layer_flow.sum(axis=-1, keepdim=True)

            model_flow = torch.matmul(layer_flow, model_flow)

            layer_flow = layer_flow.tolist()

            Xp1 = torch.multiply(Xp1, o_proj)
            Xp1 = Xp1.sum(axis=0)
            Xp1 /= Xp1.sum(axis=-1, keepdim=True)

            layer_preresidual = Xp1.tolist()
            Xp1 += X

            layer_outputs = Xp1.tolist()

            layer = Layer(
                inputs = layer_input,
                flow = layer_flow,
                outputs = layer_outputs,
                attention_heads=attention_heads,
                attention_head_weights=layer_attention_head_weights,
                pre_residual=layer_preresidual
            )

            layers.append(layer)

            X = Xp1
        
        model_output = X.tolist()

        model = Model(
            inputs=model_input,
            flow=model_flow,
            outputs=model_output,
            layers=layers
        )

        return model
    
    def __call__(self, tokens):
        with torch.no_grad():
            tokens = tokens.squeeze(0).to(self.model.device)
            attentions = self.attend(tokens)
            attributions = self.attribute(attentions)
            return attributions
