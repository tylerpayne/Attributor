import torch

from attributor.attribution_span import AttributionSpan
from attributor.span import Span
from attributor.utils import find


class Attribution:
    def __init__(self, model, tokenizer, tokens, attributions):
        self.model = model
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.attributions = attributions

    def imshow(
        self,
        *,
        output_span: Span = None,
        input_span: Span = None,
        ignore_inputs: torch.Tensor = None,
    ):
        from matplotlib import pyplot as plt

        attributions = self._slice_attributions(
            output_span=output_span,
            input_span=input_span,
            ignore_inputs=ignore_inputs,
        )
        attributions /= attributions.max()
        attributions = attributions.cpu()
        plt.imshow(attributions)

    def _rolling_mean(
        self, *, attributions: torch.Tensor, output_span: Span, input_span: Span
    ):
        output_length = attributions.shape[0]
        input_length = attributions.shape[1]

        rolling = [
            torch.stack(
                [
                    attributions[
                        i : i + output_span.window_size,
                        j : j + input_span.window_size,
                    ]
                    for j in range(
                        0,
                        input_length - input_span.window_size + 1,
                        input_span.step,
                    )
                ],
                axis=0,
            )
            for i in range(
                0, output_length - output_span.window_size + 1, output_span.step
            )
        ]
        rolling = torch.stack(rolling, axis=0)
        # total attribution in each window
        rolling = rolling.sum(axis=(-1, -2))
        # normalize each output's input distrubtion
        # rolling /= rolling.sum(axis=-1, keepdims=True)
        return rolling

    def _slice_attributions(
        self,
        *,
        output_span: Span = None,
        input_span: Span = None,
        ignore_inputs: torch.tensor = None,
    ):
        input_span = input_span or Span()
        output_span = output_span or Span()

        attributions = torch.clone(self.attributions)

        if ignore_inputs is not None:
            attributions[:, ignore_inputs] = 0

        attributions = attributions[
            slice(output_span.start, output_span.end),
            slice(input_span.start, input_span.end),
        ]

        if (
            output_span.window_size
            * output_span.step
            * input_span.window_size
            * input_span.step
            != 1
        ):
            attributions = self._rolling_mean(
                attributions=attributions,
                output_span=output_span,
                input_span=input_span,
            )

        return attributions

    def _return_attribution_spans(
        self,
        *,
        attribution_values,
        attribution_argsort,
        output_span: Span = None,
        input_span: Span = None,
    ):
        input_span = input_span or Span()
        output_span = output_span or Span()

        output_start = output_span.start or 0
        input_start = input_span.start or 0

        spans = []
        for i, row in enumerate(attribution_argsort):
            output_indices = torch.arange(
                output_start + i * output_span.step,
                output_start + i * output_span.step + output_span.window_size,
            )
            spans.append([])
            for j, col in enumerate(row):
                input_indices = torch.arange(
                    input_start + col * input_span.step,
                    input_start
                    + col * input_span.step
                    + input_span.window_size,
                )
                input_attribution = attribution_values[i, col]

                attribution_span = AttributionSpan(
                    output_indices=output_indices,
                    input_indices=input_indices,
                    attribution=input_attribution,
                )

                spans[-1].append(attribution_span)

        return spans

    def top_k(
        self,
        *,
        top_k: int = 3,
        top_k_offest: int = 0,
        output_span: Span = None,
        input_span: Span = None,
        ignore_inputs: torch.tensor = None,
    ):
        attributions = self._slice_attributions(
            output_span=output_span,
            input_span=input_span,
            ignore_inputs=ignore_inputs,
        )

        argsort = attributions.argsort(axis=1, descending=True)
        argsort = argsort[:, top_k_offest : top_k + top_k_offest]

        return self._return_attribution_spans(
            attribution_values=attributions,
            attribution_argsort=argsort,
            output_span=output_span,
            input_span=input_span,
        )

    def outliers(
        self,
        *,
        std_threshold: float = 2,
        output_span: slice = None,
        input_span: slice = None,
        ignore_inputs: torch.tensor = None,
    ):
        attributions = self._slice_attributions(
            output_span=output_span,
            input_span=input_span,
            ignore_inputs=ignore_inputs,
        )

        mean = attributions.mean(axis=0, keepdims=True)
        std = attributions.std(axis=0, keepdims=True)
        outliers = attributions > (mean + std_threshold * std)
        outliers = [torch.argwhere(row).squeeze(-1) for row in outliers]

        return self._return_attribution_spans(
            attribution_values=attributions,
            attribution_argsort=outliers,
            output_span=output_span,
            input_span=input_span,
        )

    def get(
        self,
        *,
        output_span: Span = None,
        input_spans: Span | list[Span] | None = None,
        ignore_inputs: torch.tensor = None,
    ):
        output_span = output_span or Span()

        if isinstance(input_spans, (list, tuple)):
            results: list[AttributionSpan] = []
            for span in input_spans:
                results.append(self.get(output_span=output_span, input_spans=span))
            return results
        else:
            input_spans = input_spans or Span()

            attributions = self._slice_attributions(
                output_span=output_span,
                input_span=input_spans,
                ignore_inputs=ignore_inputs,
            )

            output_start = output_span.start or 0
            input_start = input_spans.start or 0

            output_end = output_span.end or self.attributions.shape[0]
            input_end = input_spans.end or self.attributions.shape[1]

            return AttributionSpan(
                output_indices=list(range(output_start, output_end)),
                input_indices=list(range(input_start, input_end)),
                attribution=float(attributions.sum().cpu()),
            )

    def sort(self, output_span: Span, candidate_documents: list[str]):
        candidate_spans = []
        tokens_cpu = self.tokens.cpu()
        for sentence in candidate_documents:
            sentence_tokens = self.tokenizer.encode(sentence)
            start, score = find(tokens_cpu, sentence_tokens)
            if start >= 0:
                end = start + len(sentence_tokens)
                candidate_spans.append(
                    Span(start=start, end=end, step=1, window_size=len(sentence_tokens))
                )
            else:
                print(f"WARNING! Couldn't find document {sentence} in tokens")
        attribution_spans = self.get(
            output_span=output_span, input_spans=candidate_spans
        )
        sorted_spans = sorted(
            enumerate(attribution_spans),
            key=lambda item: item[1].attribution,
            reverse=True,
        )

        result: list[tuple[int, float]] = []
        for i, span in sorted_spans:
            result.append((i, span.attribution))
        return result
