"""DeepEval-Adapter:

Verwendet werden die definierten RAG-Metriken Faithfulness, ContextualRelevancy und AnswerRelevancy

Metriken und Konstrukte:
    faithfulness          -> Faithfulness
    contextual relevancy  -> Context Relevance
    answer relevancy      -> Answer Relevance

ContextualRelevancy statt ContextualPrecision. ContextualPrecision bewertet die Rangfolge der Chunks und ist schon 1, wenn der relevante Chunk vorne steht, egal wie viele irrelevante folgen.
Außerdem braucht sie eine Referenzantwort.
ContextualRelevancy misst den Anteil relevanter Aussagen im Kontext und passt damit zu context_relevance in TruLens und Ragas.
"""

from __future__ import annotations

import logging
import os
import time

# Telemetrie von DeepEval abschalten.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval.models.base_model import DeepEvalBaseLLM
from tqdm import tqdm

from ..data.schema import Construct, GeneratedSample, MetricResult
from .base import BaseEvaluator

logger = logging.getLogger(__name__)


class _ScadsJudge(DeepEvalBaseLLM):
    """Modell-Wrapper für DeepEval, der den Judge-Endpunkt aufruft.

    Ist ein schema (Pydantic) gesetzt, muss strukturiertes JSON zurückkommen.
    Dafür sorgt instructor im JSON-Modus, auch bei offenen Modellen.
    """

    def __init__(
        self, model_name: str, temperature: float = 0.0, timeout: float = 120.0
    ):
        import instructor
        from dotenv import load_dotenv
        from openai import OpenAI

        from .usage import track

        load_dotenv()
        self.temperature = temperature
        # Ohne Timeout wartet der Client 600 s auf eine hängende Antwort.
        client_kwargs = {"timeout": timeout, "max_retries": 2}
        # super().__init__ kürzt den Modellnamen um den Anbieter (meta-llama/Llama-3.3-70B-Instruct -> Llama-3.3-70B-Instruct).
        # Der Endpunkt braucht den vollen Namen, deshalb self._model_id statt self.model_name.
        self._model_id = model_name
        # Erst zählen, dann an instructor geben, sonst fehlen die strukturierten Aufrufe in der Tokenzählung
        self._client = track(OpenAI(**client_kwargs))
        self._structured = instructor.from_openai(
            self._client, mode=instructor.Mode.JSON
        )
        super().__init__(model_name)  # setzt self.model_name und self.model

    # Schnittstelle von DeepEvalBaseLLM
    def load_model(self):
        return self._client

    def get_model_name(self) -> str:
        return self._model_id

    def generate(self, prompt: str, schema=None):
        # Eigene Modelle geben nur das Ergebnis zurück ohne Kosten.
        if schema is not None:
            return self._structured.chat.completions.create(
                model=self._model_id,
                messages=[{"role": "user", "content": prompt}],
                response_model=schema,
                temperature=self.temperature,
            )
        resp = self._client.chat.completions.create(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return resp.choices[0].message.content

    async def a_generate(self, prompt: str, schema=None):
        # Wird nicht gebraucht, DeepEval läuft mit async_mode=False.
        return self.generate(prompt, schema)


# Versuche je Bewertung, zusätzlich zu den Wiederholungen des OpenAI-Clients.
# DeepEval wiederholt selbst nicht, anders als TruLens.
# Ohne diese Versuche gehen bei Timeouts einzelne Scores verloren.
_MEASURE_ATTEMPTS = 3


def _measure_with_retry(
    metric, test_case, construct: Construct, sample_id: str
) -> None:
    """Führt metric.measure aus und wiederholt bei Fehlern mit Backoff.

    Die Metriken speichern Zustand, measure setzt die Felder aber bei jedem Aufruf neu. Ein neuer Versuch auf derselben Instanz ist deshalb möglich.
    """
    last_error: Exception | None = None
    for attempt in range(_MEASURE_ATTEMPTS):
        try:
            metric.measure(test_case)
            if attempt:
                logger.info(
                    "[deepeval] %s/%s im %d. Versuch erfolgreich",
                    sample_id,
                    construct.value,
                    attempt + 1,
                )
            return
        except Exception as e:  # Aussetzer des Endpunkts, Timeouts, JSON-Fehler
            last_error = e
            if attempt < _MEASURE_ATTEMPTS - 1:
                delay = 2**attempt  # 1 s, 2 s
                logger.warning(
                    "[deepeval] %s/%s Versuch %d/%d fehlgeschlagen (%s) — neuer Versuch in %ds",
                    sample_id,
                    construct.value,
                    attempt + 1,
                    _MEASURE_ATTEMPTS,
                    e,
                    delay,
                )
                time.sleep(delay)
    raise RuntimeError(
        f"nach {_MEASURE_ATTEMPTS} Versuchen fehlgeschlagen: {last_error}"
    ) from last_error


class DeepEvalEvaluator(BaseEvaluator):
    name = "deepeval"
    metric_mapping = {
        Construct.FAITHFULNESS: "faithfulness",
        Construct.CONTEXT_RELEVANCE: "contextual_relevancy",
        Construct.ANSWER_RELEVANCE: "answer_relevancy",
    }

    def evaluate(self, samples: list[GeneratedSample]) -> list[MetricResult]:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualRelevancyMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase

        judge = _ScadsJudge(
            self.cfg["judge"]["model"],
            self.cfg["judge"].get("temperature", 0.0),
            timeout=float(self.cfg["judge"].get("request_timeout", 120.0)),
        )
        common_kwargs = dict(model=judge, async_mode=False, include_reason=True)

        # Je Sample neue Metrik-Instanzen, weil die Metriken Zustand speichern.
        def make_metrics():
            return {
                Construct.FAITHFULNESS: FaithfulnessMetric(**common_kwargs),
                Construct.CONTEXT_RELEVANCE: ContextualRelevancyMetric(**common_kwargs),
                Construct.ANSWER_RELEVANCE: AnswerRelevancyMetric(**common_kwargs),
            }

        meta = {"judge_model": self.cfg["judge"]["model"], "framework": self.name}

        results: list[MetricResult] = []
        for s in tqdm(samples, desc="deepeval"):
            test_case = LLMTestCase(
                input=s.query,
                actual_output=s.generated_answer,
                retrieval_context=s.retrieved_contexts,
                # Keine der drei Metriken braucht die Referenz, sie darf None sein.
                expected_output=s.reference_answer,
            )
            for construct, metric in make_metrics().items():
                try:
                    _measure_with_retry(metric, test_case, construct, s.sample_id)
                    results.append(
                        MetricResult(
                            sample_id=s.sample_id,
                            dataset=s.dataset,
                            framework=self.name,
                            construct=construct,
                            native_metric_name=self.metric_mapping[construct],
                            score=float(metric.score),
                            reasoning=getattr(metric, "reason", None),
                            meta=meta,
                        )
                    )
                except Exception as e:  # einzelne Bewertung überspringen
                    logger.warning(
                        "[deepeval] %s/%s fehlgeschlagen: %s",
                        s.sample_id,
                        construct.value,
                        e,
                    )
        return results
