import torch

from unsquash.eval.hotpot import format_row
from unsquash.spans import char_span_to_token_span, prepare_case, score_spans


def test_char_span_to_token_span():
    offsets = [(0, 0), (0, 5), (5, 9), (9, 15), (15, 15)]
    span = char_span_to_token_span(offsets, 0, 5)
    assert (span.start, span.end) == (1, 2)
    span = char_span_to_token_span(offsets, 3, 10)  # overlaps tokens 1-3
    assert (span.start, span.end) == (1, 4)
    assert char_span_to_token_span(offsets, 20, 25) is None


FAKE_ROW = {
    "question": "What is the capital of France?",
    "answer": "Paris",
    "context": {
        "title": ["France", "Germany"],
        "sentences": [
            ["Paris is the capital of France.", "France is in Europe."],
            ["Berlin is the capital of Germany.", "Germany is in Europe."],
        ],
    },
    "supporting_facts": {"title": ["France"], "sent_id": [0]},
}


def test_format_row_records_exact_spans():
    case = format_row(FAKE_ROW)
    assert case is not None
    assert case.supporting == {0}
    assert len(case.sentences) == 4
    for text, (a, b) in zip(case.sentences, case.sentence_char_spans):
        assert case.context[a:b] == text


def test_format_row_skips_bad_supporting_fact():
    row = dict(FAKE_ROW)
    row["supporting_facts"] = {"title": ["France", "Atlantis"], "sent_id": [1, 5]}
    case = format_row(row)
    assert case.supporting == {1}


def test_format_row_unusable_case():
    row = dict(FAKE_ROW)
    row["supporting_facts"] = {"title": ["Atlantis"], "sent_id": [0]}
    assert format_row(row) is None


def test_prepare_case_maps_every_sentence(tiny_tokenizer):
    case = format_row(FAKE_ROW)
    prepared = prepare_case(
        tiny_tokenizer,
        context=case.context,
        answer=case.answer,
        sentence_char_spans=case.sentence_char_spans,
    )
    n = prepared.input_ids.shape[0]
    assert n > 0
    # Answer tokens come after every sentence span (assistant turn is last).
    assert all(s is not None for s in prepared.sentence_spans)
    for span in prepared.sentence_spans:
        assert 0 < len(span)
        assert span.end <= prepared.answer_span.start
    assert prepared.answer_span.end <= n

    # Decoding a mapped span recovers the sentence text (module whitespace).
    decoded = tiny_tokenizer.decode(
        prepared.input_ids[prepared.sentence_spans[0].start:
                           prepared.sentence_spans[0].end]
    )
    assert "Paris is the capital of France." in decoded or \
        decoded.strip() in case.sentences[0]


def test_score_spans_reductions():
    Y = torch.zeros(10, 10, dtype=torch.float64)
    Y[8, 2:4] = 0.5   # answer token at 8 attends to tokens 2,3
    answer = char_span_to_token_span([(i, i + 1) for i in range(10)], 8, 9)
    spans = [
        char_span_to_token_span([(i, i + 1) for i in range(10)], 2, 4),  # hit, len 2
        char_span_to_token_span([(i, i + 1) for i in range(10)], 2, 8),  # hit, len 6
        char_span_to_token_span([(i, i + 1) for i in range(10)], 5, 7),  # miss
        None,  # unmapped
    ]
    mean_scores = score_spans(Y, answer, spans, reduction="mean")
    sum_scores = score_spans(Y, answer, spans, reduction="sum")
    # Same total mass: sum ties the long and short span, mean prefers short.
    assert sum_scores[0] == sum_scores[1] == 1.0
    assert mean_scores[0] > mean_scores[1]
    assert mean_scores[2] == 0.0
    assert mean_scores[3] == float("-inf")


def test_prepare_case_without_chat_template(tiny_tokenizer):
    """Base-LM tokenizers (no chat template) fall back to plain text."""
    import copy

    from unsquash.spans import prepare_case

    tok = copy.deepcopy(tiny_tokenizer)
    tok.chat_template = None
    context = "The sky is blue. Grass is green."
    prepared = prepare_case(
        tok,
        context=context,
        answer="blue",
        sentence_char_spans=[(0, 16), (17, 32)],
    )
    assert prepared.answer_span is not None
    assert len(prepared.sentence_spans) == 2
    assert all(s is not None for s in prepared.sentence_spans)
    # the answer span must come from the appended answer, not "blue" in context
    assert prepared.answer_span.start > prepared.sentence_spans[1].start
