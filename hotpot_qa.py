import logging
import os
from argparse import ArgumentParser
from typing import Callable, Generic, Sequence, TypeVar

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from attributor import get_logger, set_log_level
from attributor.attributor import Attributor
from attributor.evaluation.evaluation_case import EvaluationCase
from attributor.evaluation.evaluator import Evaluator
from attributor.evaluation.openai_verifier import openai_verifier

logger = get_logger()


class HotpotQAEvaluationCase(EvaluationCase):
    context: str
    question: str


TOriginal = TypeVar("TOriginal")
TTransformed = TypeVar("TTransformed")


class TransformedSequence(Sequence, Generic[TOriginal, TTransformed]):
    def __init__(
        self,
        sequence: Sequence[TOriginal],
        transform: Callable[[TOriginal], TTransformed],
    ):
        self._sequence = sequence
        self.transform = transform

    def __getitem__(self, index):
        if index < 0:
            index += len(self._sequence)
        if index < 0 or index >= len(self._sequence):
            raise IndexError("Index out of range")
        return self.transform(self._sequence[index])

    def __len__(self):
        return len(self._sequence)

    def __contains__(self, item):
        raise NotImplementedError


def load_hotpot_qa(trust_remote_code: bool = False):
    ds_path = "hotpotqa/hotpot_qa"
    ds_name = "fullwiki"
    hotpot_qa = load_dataset(
        ds_path, ds_name, split="train", trust_remote_code=trust_remote_code
    )
    return hotpot_qa


def format_hotpot_qa_row(row: dict) -> HotpotQAEvaluationCase:
    documents = dict(zip(row["context"]["title"], row["context"]["sentences"]))
    context = 'Documents:\n"""\n'
    document_sentences = []
    for title, sentences in documents.items():
        context += title + ":"
        context += "\n"
        for sentence in sentences:
            # Append with any whitespace
            context += sentence
            # Strip whitespace before indexing
            sentence = sentence.strip()
            document_sentences.append(sentence)
        context += "\n\n"

    context = context.strip()
    context += '\n"""\n\n'

    context += row["question"].strip()

    context += "\nDo not explain your reasoning. Simply provide the answer or say 'n/a' if the question cannot be answered."

    supporting_sentences = []
    for i, title in enumerate(row["supporting_facts"]["title"]):
        sentence_id = row["supporting_facts"]["sent_id"][i]
        sentence = documents[title][sentence_id]
        sentence = sentence.strip()
        index = document_sentences.index(sentence)
        if index >= 0:
            supporting_sentences.append(index)
        else:
            logger.warning(f"Couldn't find sentence '{sentence}' in context.")

    return HotpotQAEvaluationCase(
        documents=document_sentences,
        expected_output=row["answer"],
        supporting_documents=supporting_sentences,
        context=context,
        question=row["question"],
    )


def main(args):
    torch_dtype = getattr(torch, args.dtype)

    logger.info(
        f"Loading {args.model} with device_map {args.device_map} and dtype {args.dtype}"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
    )

    logger.info(f"Loading tokenizer for {args.model}.")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading generation config for {args.model}.")
    generation_config = GenerationConfig.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        max_new_tokens=64,
        do_sample=False,
    )

    attributor = Attributor(model, tokenizer)

    logger.info("Loading HotPotQA.")
    hotpot_qa = load_hotpot_qa(trust_remote_code=args.trust_remote_code)
    evaluation_cases = TransformedSequence(hotpot_qa, format_hotpot_qa_row)

    progress_dirpath = os.path.join("evaluation_results", args.model, "hotpot_qa")
    if args.overwrite:
        os.rmdir(progress_dirpath)

    def format(case: HotpotQAEvaluationCase):
        return case.context

    def verifier(*verifier_args, **verifier_kwargs):
        return openai_verifier(
            *verifier_args, openai_client=args.openai_api_key, **verifier_kwargs
        )

    evaluator = Evaluator(
        attributor=attributor,
        formatter=format,
        progress_dirpath=progress_dirpath,
        # verifier=verifier,
    )

    evaluator.evaluate(
        cases=evaluation_cases,
        # generation_config=generation_config,
        # max_context_tokens=args.max_context_tokens,
    )


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--max_context_tokens", type=int, default=1000)
    parser.add_argument("--trust_remote_code", default=False, action="store_true")
    parser.add_argument("--overwrite", default=False, action="store_true")
    parser.add_argument("--openai_api_key", default=None)

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--debug", action="store_const", const=logging.DEBUG, dest="log_level"
    )
    group.add_argument(
        "--info", action="store_const", const=logging.INFO, dest="log_level"
    )
    group.add_argument(
        "--warning", action="store_const", const=logging.WARNING, dest="log_level"
    )
    group.add_argument(
        "--error", action="store_const", const=logging.ERROR, dest="log_level"
    )

    parser.set_defaults(log_level=logging.INFO)

    args = parser.parse_args()

    set_log_level(args.log_level)

    return args


if __name__ == "__main__":
    main(get_args())
