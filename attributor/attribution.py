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
        effect_span: Span = None,
        treatment_span: Span = None,
        ignore_treatments: torch.Tensor = None,
    ):
        from matplotlib import pyplot as plt

        attributions = self._slice_attributions(
            effect_span=effect_span,
            treatment_span=treatment_span,
            ignore_treatments=ignore_treatments,
        )
        attributions /= attributions.max()
        attributions = attributions.cpu()
        plt.imshow(attributions)

    def _rolling_mean(
        self, *, attributions: torch.Tensor, effect_span: Span, treatment_span: Span
    ):
        effect_length = attributions.shape[0]
        treatment_length = attributions.shape[1]

        rolling = [
            torch.stack(
                [
                    attributions[
                        i : i + effect_span.window_size,
                        j : j + treatment_span.window_size,
                    ]
                    for j in range(
                        0,
                        treatment_length - treatment_span.window_size + 1,
                        treatment_span.step,
                    )
                ],
                axis=0,
            )
            for i in range(
                0, effect_length - effect_span.window_size + 1, effect_span.step
            )
        ]
        rolling = torch.stack(rolling, axis=0)
        # total attribution in each window
        rolling = rolling.sum(axis=(-1, -2))
        # normalize each effect's treatment distrubtion
        # rolling /= rolling.sum(axis=-1, keepdims=True)
        return rolling

    def _slice_attributions(
        self,
        *,
        effect_span: Span = None,
        treatment_span: Span = None,
        ignore_treatments: torch.tensor = None,
    ):
        treatment_span = treatment_span or Span()
        effect_span = effect_span or Span()

        attributions = torch.clone(self.attributions)

        if ignore_treatments is not None:
            attributions[:, ignore_treatments] = 0

        attributions = attributions[
            slice(effect_span.start, effect_span.end),
            slice(treatment_span.start, treatment_span.end),
        ]

        if (
            effect_span.window_size
            * effect_span.step
            * treatment_span.window_size
            * treatment_span.step
            != 1
        ):
            attributions = self._rolling_mean(
                attributions=attributions,
                effect_span=effect_span,
                treatment_span=treatment_span,
            )

        return attributions

    def _return_attribution_spans(
        self,
        *,
        attribution_values,
        attribution_argsort,
        effect_span: Span = None,
        treatment_span: Span = None,
    ):
        treatment_span = treatment_span or Span()
        effect_span = effect_span or Span()

        effect_start = effect_span.start or 0
        treatment_start = treatment_span.start or 0

        spans = []
        for i, row in enumerate(attribution_argsort):
            effect_indices = torch.arange(
                effect_start + i * effect_span.step,
                effect_start + i * effect_span.step + effect_span.window_size,
            )
            spans.append([])
            for j, col in enumerate(row):
                treatment_indices = torch.arange(
                    treatment_start + col * treatment_span.step,
                    treatment_start
                    + col * treatment_span.step
                    + treatment_span.window_size,
                )
                treatment_attribution = attribution_values[i, col]

                attribution_span = AttributionSpan(
                    effect_indices=effect_indices,
                    treatment_indices=treatment_indices,
                    attribution=treatment_attribution,
                )

                spans[-1].append(attribution_span)

        return spans

    def top_k(
        self,
        *,
        top_k: int = 3,
        top_k_offest: int = 0,
        effect_span: Span = None,
        treatment_span: Span = None,
        ignore_treatments: torch.tensor = None,
    ):
        attributions = self._slice_attributions(
            effect_span=effect_span,
            treatment_span=treatment_span,
            ignore_treatments=ignore_treatments,
        )

        argsort = attributions.argsort(axis=1, descending=True)
        argsort = argsort[:, top_k_offest : top_k + top_k_offest]

        return self._return_attribution_spans(
            attribution_values=attributions,
            attribution_argsort=argsort,
            effect_span=effect_span,
            treatment_span=treatment_span,
        )

    def outliers(
        self,
        *,
        std_threshold: float = 2,
        effect_span: slice = None,
        treatment_span: slice = None,
        ignore_treatments: torch.tensor = None,
    ):
        attributions = self._slice_attributions(
            effect_span=effect_span,
            treatment_span=treatment_span,
            ignore_treatments=ignore_treatments,
        )

        mean = attributions.mean(axis=0, keepdims=True)
        std = attributions.std(axis=0, keepdims=True)
        outliers = attributions > (mean + std_threshold * std)
        outliers = [torch.argwhere(row).squeeze(-1) for row in outliers]

        return self._return_attribution_spans(
            attribution_values=attributions,
            attribution_argsort=outliers,
            effect_span=effect_span,
            treatment_span=treatment_span,
        )

    def get(
        self,
        *,
        effect_span: Span = None,
        treatment_spans: Span | list[Span] | None = None,
        ignore_treatments: torch.tensor = None,
    ):
        effect_span = effect_span or Span()

        if isinstance(treatment_spans, (list, tuple)):
            results: list[AttributionSpan] = []
            for span in treatment_spans:
                results.append(self.get(effect_span=effect_span, treatment_spans=span))
            return results
        else:
            treatment_spans = treatment_spans or Span()

            attributions = self._slice_attributions(
                effect_span=effect_span,
                treatment_span=treatment_spans,
                ignore_treatments=ignore_treatments,
            )

            effect_start = effect_span.start or 0
            treatment_start = treatment_spans.start or 0

            effect_end = effect_span.end or self.attributions.shape[0]
            treatment_end = treatment_spans.end or self.attributions.shape[1]

            return AttributionSpan(
                effect_indices=list(range(effect_start, effect_end)),
                treatment_indices=list(range(treatment_start, treatment_end)),
                attribution=float(attributions.sum().cpu()),
            )

    def sort(self, effect_span: Span, candidate_documents: list[str]):
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
            effect_span=effect_span, treatment_spans=candidate_spans
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
