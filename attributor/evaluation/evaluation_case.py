from pydantic import BaseModel, SerializeAsAny


class EvaluationCase(BaseModel):
    documents: list[str]
    expected_output: str
    supporting_documents: list[int]


class EvaluationResult(BaseModel):
    case: SerializeAsAny[EvaluationCase]
    generated_output: str
    attributed_documents: list[int]
    attributed_document_scores: list[float]
    verification: bool | None = None
