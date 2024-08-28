import torch
from fastapi import FastAPI
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer#, GenerationConfig

from attributor.utils import tokenize
from server.attributor import Attributor
from server.models import ConfigParams, Message, Model, ModelParms


class AttributionAPI(FastAPI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attribution: Model = None
        self._attributor: Attributor = None

    @property
    def configured(self):
        return self._attributor is not None

    def configure(self, config: ConfigParams):
        self._attributor = None
        torch_dtype = getattr(torch, config.dtype)
        logger.info(
            f"Loading {config.model} with device_map {config.device_map} and dtype {config.dtype}"
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model,
            device_map=config.device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        )

        logger.info(f"Loading tokenizer for {config.model}.")

        tokenizer = AutoTokenizer.from_pretrained(
            config.model, trust_remote_code=True
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # logger.info(f"Loading generation config for {config.model}.")
        # generation_config = GenerationConfig.from_pretrained(
        #     config.model,
        #     trust_remote_code=True,
        #     max_new_tokens=64,
        #     do_sample=False,
        # )

        self._attributor = Attributor(model, tokenizer)

    def attribute(self, messages: list[Message]) -> Model:
        tokens = tokenize(
            self._attributor.tokenizer,
            [m.model_dump(mode="json") for m in messages],
            add_generation_prompt=False,
        )
        self.attribution = self._attributor(tokens)

        params = ModelParms(
            tokens=[
                self._attributor.tokenizer.decode(token_id) for token_id in tokens[0]
            ],
            nr_layers=len(self.attribution.layers),
            nr_attention_heads=len(self.attribution.layers[0].attention_heads),
        )

        return params
