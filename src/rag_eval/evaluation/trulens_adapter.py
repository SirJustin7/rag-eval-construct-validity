"""TruLens-Adapter:

TruLens liefert zu jedem Score eine Begründung, sie wird in MetricResult.reasoning gespeichert.

Metriken und Konstrukte:
    groundedness       -> Faithfulness
    context relevance  -> Context Relevance
    answer relevance   -> Answer Relevance

Eigener Provider statt trulens-providers-openai. Dessen Modul trulens.providers.openai.endpoint importiert langchain.schema, das es in langchain 1.x nicht mehr gibt.
ScadsProvider erbt deshalb direkt von LLMProvider und ruft den Endpunkt über den OpenAI-Client auf.
Die Feedback-Funktionen sind unverändert die von TruLens und werden direkt aufgerufen, ohne TruSession. Sie liefern Score und Begründung als Tupel zurück.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import PrivateAttr
from tqdm import tqdm

from ..data.schema import Construct, GeneratedSample, MetricResult
from .base import BaseEvaluator

logger = logging.getLogger(__name__)

# Der Judge vergibt ganze Punkte, TruLens rechnet das selbst auf [0, 1] um.
# TruLens kennt nur drei Skalen: 0-1, 0-3 (Standard) und 0-10.
# Verwendet wird der Standard 0-3. Einstellbar über frameworks.trulens.score_range.
_DEFAULT_SCORE_RANGE = (0, 3)
_VALID_SCORE_RANGES = {(0, 1), (0, 3), (0, 10)}


def _build_provider(model_name: str, rpm: float, timeout: float = 120.0):
    """Baut den Provider für den Judge. Die Importe stehen hier, damit import rag_eval schnell bleibt."""
    import instructor
    from dotenv import load_dotenv
    from openai import OpenAI
    from trulens.core.feedback.endpoint import Endpoint
    from trulens.feedback.llm_provider import LLMProvider

    from .usage import track

    load_dotenv()

    class ScadsProvider(LLMProvider):
        """LLMProvider für den Judge-Endpunkt.

        TruLens ruft _create_chat_completion mit response_format=<Pydantic-Modell> auf und erwartet eine Instanz davon zurück.
        Wie bei DeepEval sorgt instructor im JSON-Modus dafür, dass auch offene Modelle gültiges JSON liefern.
        """

        model_engine: str = model_name

        _client: Any = PrivateAttr()
        _structured: Any = PrivateAttr()

        def __init__(self, **kwargs):
            # Der Endpoint übernimmt Ratenbegrenzung und Wiederholungen
            kwargs.setdefault("endpoint", Endpoint(name="scads", rpm=rpm))
            super().__init__(**kwargs)
            # Ohne Timeout wartet der Client 600 s auf eine hängende Antwort.
            # max_retries=0, weil schon der Endpoint wiederholt.
            client_kwargs = {"timeout": timeout, "max_retries": 0}
            # Erst zählen, dann an instructor geben, sonst fehlen die strukturierten Aufrufe in der Tokenzählung
            self._client = track(OpenAI(**client_kwargs))
            self._structured = instructor.from_openai(
                self._client, mode=instructor.Mode.JSON
            )

        def _create_chat_completion(
            self, prompt=None, messages=None, response_format=None, **kwargs
        ):
            if messages is None:
                messages = [{"role": "user", "content": prompt}]
            if response_format is not None:
                return self._structured.chat.completions.create(
                    model=self.model_engine,
                    messages=list(messages),
                    response_model=response_format,
                    **kwargs,
                )
            resp = self._client.chat.completions.create(
                model=self.model_engine, messages=list(messages), **kwargs
            )
            return resp.choices[0].message.content

    return ScadsProvider()


def _reason_text(reasons: dict | None) -> str | None:
    """groundedness liefert {"reasons": ...}, die Relevanz-Metriken {"reason": ...}."""
    if not isinstance(reasons, dict):
        return None
    return reasons.get("reasons") or reasons.get("reason")


class TruLensEvaluator(BaseEvaluator):
    name = "trulens"
    metric_mapping = {
        Construct.FAITHFULNESS: "groundedness",
        Construct.CONTEXT_RELEVANCE: "context_relevance",
        Construct.ANSWER_RELEVANCE: "answer_relevance",
    }

    def evaluate(self, samples: list[GeneratedSample]) -> list[MetricResult]:
        judge_model = self.cfg["judge"]["model"]
        temperature = self.cfg["judge"].get("temperature", 0.0)
        fw_cfg = self.cfg.get("frameworks", {}).get("trulens", {}) or {}
        provider = _build_provider(
            judge_model,
            rpm=fw_cfg.get("rpm", 60),
            timeout=float(self.cfg["judge"].get("request_timeout", 120.0)),
        )

        min_score, max_score = tuple(fw_cfg.get("score_range") or _DEFAULT_SCORE_RANGE)
        if (min_score, max_score) not in _VALID_SCORE_RANGES:
            raise ValueError(
                f"frameworks.trulens.score_range={[min_score, max_score]} ist ungültig. "
                f"TruLens kennt nur {sorted(_VALID_SCORE_RANGES)}."
            )

        meta = {
            "judge_model": judge_model,
            "framework": self.name,
            # Die Skala bestimmt, wie viele verschiedene Scores möglich sind.
            "score_scale_native": f"{min_score}-{max_score} (von TruLens auf [0,1] normalisiert)",
        }
        common = {
            "min_score_val": min_score,
            "max_score_val": max_score,
            "temperature": temperature,
        }

        results: list[MetricResult] = []
        for s in tqdm(samples, desc="trulens"):
            # Groundedness prüft die Antwort gegen den ganzen abgerufenen Kontext, wie .on(context.collect()) in TruLens.
            source = "\n\n".join(s.retrieved_contexts)

            def add(construct: Construct, fn, sample=s):
                try:
                    score, reasons = fn()
                    results.append(
                        MetricResult(
                            sample_id=sample.sample_id,
                            dataset=sample.dataset,
                            framework=self.name,
                            construct=construct,
                            native_metric_name=self.metric_mapping[construct],
                            score=float(score),
                            reasoning=_reason_text(reasons),
                            meta=meta,
                        )
                    )
                except Exception as e:  # einzelne Bewertung überspringen
                    logger.warning(
                        "[trulens] %s/%s fehlgeschlagen: %s",
                        sample.sample_id,
                        construct.value,
                        e,
                    )

            add(
                Construct.FAITHFULNESS,
                lambda sample=s, src=source: (
                    provider.groundedness_measure_with_cot_reasons(
                        source=src, statement=sample.generated_answer, **common
                    )
                ),
            )
            add(
                Construct.ANSWER_RELEVANCE,
                lambda sample=s: provider.relevance_with_cot_reasons(
                    prompt=sample.query, response=sample.generated_answer, **common
                ),
            )
            add(
                Construct.CONTEXT_RELEVANCE,
                lambda sample=s: self._context_relevance(provider, sample, common),
            )

        return results

    @staticmethod
    def _context_relevance(provider, s: GeneratedSample, common: dict):
        """Bewertet jeden Chunk einzeln und mittelt.

        So macht es TruLens in der RAG-Triade, np.mean ist die Voreinstellung von Feedback.aggregate. Anders als max zählen beim Mittelwert auch irrelevante Chunks.
        """
        scores, reasons = [], []
        for i, ctx in enumerate(s.retrieved_contexts):
            score, reason = provider.context_relevance_with_cot_reasons(
                question=s.query, context=ctx, **common
            )
            scores.append(float(score))
            reasons.append(f"CHUNK {i} (score={score:.3f}):\n{_reason_text(reason)}")
        if not scores:
            raise ValueError("keine retrieved_contexts vorhanden")
        return float(np.mean(scores)), {"reason": "\n\n".join(reasons)}
