from attributor.evaluation.evaluation_case import EvaluationResult


def support(results: list[EvaluationResult]):
    return len(results)


def precision(result: EvaluationResult, k=None):
    supporting = result.case.supporting_documents
    attributed = result.attributed_documents[:k]
    k = k or len(supporting)
    attributed_supporting = len(set(attributed).intersection(supporting))
    return attributed_supporting / k


def mean_precision(results: list[EvaluationResult], k=None):
    return sum([precision(r, k=k) for r in results]) / len(results)


def recall(result: EvaluationResult, k=None):
    supporting = result.case.supporting_documents
    attributed = result.attributed_documents[:k]
    k = k or len(supporting)
    attributed_supporting = len(set(attributed).intersection(supporting))
    return attributed_supporting / min(k, len(supporting))


def mean_recall(results: list[EvaluationResult], k=None):
    return [recall(r, k=k) for r in results] / len(results)


def incremental_mean(mean, value, n):
    return mean + ((value - mean) / n)
