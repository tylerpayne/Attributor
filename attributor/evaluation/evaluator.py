import json
import os
from os import PathLike
from typing import Callable, Sequence

import torch
from pydantic import BaseModel
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import GenerationConfig

from attributor import get_logger
from attributor.attributor import Attributor
from attributor.evaluation.evaluation_case import EvaluationCase, EvaluationResult
from attributor.evaluation.metrics import incremental_mean, precision, recall
from attributor.span import Span
from attributor.utils import tokenize

logger = get_logger()


class EvaluationProgress(BaseModel):
    iteration: int
    support: int
    mean_precision: dict[int, float] = {}
    mean_recall: dict[int, float] = {}


class Evaluator:
    def __init__(
        self,
        *,
        attributor: Attributor,
        progress_dirpath: PathLike,
        formatter: Callable[[EvaluationCase], str],
        verifier: Callable[[str, str], bool] | None = None,
    ):
        assert isinstance(attributor, Attributor)
        assert isinstance(progress_dirpath, (str, os.PathLike))

        self.attributor = attributor
        self.verifier = verifier
        self.formatter = formatter
        self.progress_dirpath = progress_dirpath
        os.makedirs(self.progress_dirpath, exist_ok=True)

    @property
    def progress_filepath(self):
        return os.path.join(self.progress_dirpath, "progress.json")

    def save_progress(
        self,
        iterator_start: int,
        progress: EvaluationProgress,
        evaluation_results: list[EvaluationResult],
    ):
        if progress.iteration > iterator_start:
            logger.info("Saving progress.")
            with open(self.progress_filepath, "w+") as fd:
                json.dump(
                    progress.model_dump(mode="json"),
                    fd,
                )

            incremental_results_filepath = os.path.join(
                self.progress_dirpath, f"{iterator_start}-{progress.iteration}.json"
            )

            with open(incremental_results_filepath, "w+") as fd:
                json.dump([e.model_dump(mode="json") for e in evaluation_results], fd)

    def _evaluate_case(
        self,
        *,
        case: EvaluationCase,
        generation_config: GenerationConfig,
        max_context_tokens: int | None = 1000,
    ):
        context = self.formatter(case)
        messages = [{"role": "user", "content": context}]

        prompt_tokens = tokenize(self.attributor.tokenizer, messages)

        if max_context_tokens is not None and (
            prompt_tokens.shape[1] >= max_context_tokens
        ):
            return None

        prompt_tokens = prompt_tokens.to(self.attributor.model.device)

        generated_tokens = self.attributor.model.generate(
            prompt_tokens,
            generation_config=generation_config,
            tokenizer=self.attributor.tokenizer,
            attention_mask=torch.ones_like(prompt_tokens),
        )

        attribution = self.attributor(generated_tokens)

        total_token_count = generated_tokens.shape[1]
        prompt_token_count = prompt_tokens.shape[1]
        output_token_count = total_token_count - prompt_token_count

        output_span = Span(
            start=prompt_token_count,
            end=total_token_count - 1,
            step=1,
            window_size=output_token_count - 1,
        )

        output_text = self.attributor.tokenizer.decode(
            generated_tokens[0, prompt_token_count:-1]
        )

        if self.verifier is not None:
            verification = self.verifier(
                case.expected_output,
                output_text,
            )
        else:
            verification = None

        attributed_documents = attribution.sort(
            output_span,
            case.documents,
        )

        attributed_document_ids = []
        attributed_document_scores = []

        for i, score in attributed_documents:
            attributed_document_ids.append(i)
            attributed_document_scores.append(score)

        return EvaluationResult(
            case=case,
            generated_output=output_text,
            attributed_documents=attributed_document_ids,
            attributed_document_scores=attributed_document_scores,
            verification=verification,
        )

    def _update_metrics(self, result: EvaluationResult, progress: EvaluationProgress):
        for k in [1, 3, 5, 10]:
            result_recall = recall(result, k = k)
            result_precision = precision(result, k = k)

            mean_recall = progress.mean_recall.get(k, None)
            mean_precision = progress.mean_precision.get(k, None)

            progress.mean_recall[k] = (
                result_recall
                if mean_recall is None
                else incremental_mean(
                    mean_recall,
                    result_recall,
                    progress.support,
                )
            )
            progress.mean_precision[k] = (
                result_precision
                if mean_precision is None
                else incremental_mean(
                    mean_precision,
                    result_precision,
                    progress.support,
                )
            )

            logger.info(f"Mean Recall @ {k}: {progress.mean_recall[k]:.1%}")
            logger.info(f"Mean Precision @ {k}: {progress.mean_precision[k]:.1%}")


    def evaluate(
        self,
        *,
        cases: Sequence[EvaluationCase],
        generation_config: GenerationConfig,
        max_context_tokens: int | None = 1000,
    ):
        os.makedirs(self.progress_dirpath, exist_ok=True)

        progress_filepath = os.path.join(self.progress_dirpath, "progress.json")

        if os.path.exists(progress_filepath):
            with open(progress_filepath) as fd:
                progress = json.load(fd)
                progress = EvaluationProgress(**progress)
        else:
            progress = EvaluationProgress(iteration=0, support=0)

        logger.info("Beginning evaluation.")
        iterator_start = progress.iteration
        evaluation_results = []
        iterator = range(iterator_start, len(cases))
        try:
            with logging_redirect_tqdm():
                for i in tqdm(iterator, initial=iterator_start):
                    try:
                        case = cases[i]

                        result = self._evaluate_case(
                            case=case,
                            generation_config=generation_config,
                            max_context_tokens=max_context_tokens,
                        )

                        if result is not None:
                            evaluation_results.append(result)
                            progress.support += 1
                            self._update_metrics(result, progress)

                    except Exception as ex:
                        logger.error(
                            f"Caught exception evaluating case {i}.",
                            ex,
                            exc_info=True,
                            stack_info=True,
                        )

                    progress.iteration += 1
                    torch.cuda.empty_cache()

        except (Exception, KeyboardInterrupt) as ex:
            raise ex
        finally:
            self.save_progress(iterator_start, progress, evaluation_results)
