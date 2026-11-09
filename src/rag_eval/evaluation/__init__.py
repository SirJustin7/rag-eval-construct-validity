from .base import BaseEvaluator
from .ragas_adapter import RagasEvaluator
from .trulens_adapter import TruLensEvaluator
from .deepeval_adapter import DeepEvalEvaluator

EVALUATORS: dict[str, type[BaseEvaluator]] = {
    "ragas": RagasEvaluator,
    "trulens": TruLensEvaluator,
    "deepeval": DeepEvalEvaluator,
}
