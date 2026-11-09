"""Ragas-Adapter:

Metriken und Konstrukte:
    Faithfulness        -> Faithfulness
    ContextRelevance    -> Context Relevance
    AnswerRelevancy     -> Answer Relevance

ContextRelevance statt ContextPrecision, wie bei DeepEval.
Ragas bewertet den Kontext als Ganzes mit zwei Judge-Aufrufen auf der Skala 0/1/2 und mittelt beide.

Ragas liefert keine Begründungen, reasoning bleibt hier None. Die Metriken in ragas.metrics.collections (0.4.3) füllen reason nicht.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
import types
from typing import Any

from tqdm import tqdm

from ..data.schema import Construct, GeneratedSample, MetricResult
from .base import BaseEvaluator

logger = logging.getLogger(__name__)

_SHIMMED_MODULE = "langchain_community.chat_models.vertexai"


def _install_langchain_vertexai_shim() -> None:
    """Macht import ragas unter langchain 1.x möglich, ohne Downgrade.

    ragas 0.4.3 importiert beim Laden ChatVertexAI aus langchain_community.chat_models.vertexai.
    Das Modul gibt es in langchain-community 0.4 nicht mehr, und schon import ragas scheitert.

    ragas braucht ChatVertexAI nur für eine isinstance-Prüfung (MULTIPLE_COMPLETION_SUPPORTED in ragas/llms/base.py).
    Der Judge hier ist ein OpenAI-kompatibler Client, die Prüfung ergibt also False, und genau das liefert der Platzhalter.
    Mit einem echten VertexAI-Modell geht das nicht, dafür bräuchte es langchain-community < 0.4.
    Ein Downgrade würde aber deepeval und trulens-core brechen.
    """
    if _SHIMMED_MODULE in sys.modules:
        return
    try:
        importlib.import_module(_SHIMMED_MODULE)
        return  # ältere langchain-Version, Modul existiert
    except ModuleNotFoundError:
        pass

    module = types.ModuleType(_SHIMMED_MODULE)

    class ChatVertexAI:
        """Platzhalter, nur für isinstance, wird nie instanziiert."""

    module.ChatVertexAI = ChatVertexAI
    sys.modules[_SHIMMED_MODULE] = module
    # Auch am Elternpaket setzen, damit Attributzugriffe funktionieren.
    parent = importlib.import_module("langchain_community.chat_models")
    parent.vertexai = module
    logger.debug("langchain-Shim für %s registriert", _SHIMMED_MODULE)


def _build_components(cfg: dict[str, Any]):
    """Erzeugt Judge-LLM und Embeddings für ragas."""
    from dotenv import load_dotenv
    from openai import AsyncOpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms.base import llm_factory

    from .usage import track_async

    load_dotenv()

    # Die Metriken rufen ascore auf und brauchen einen asynchronen Client.
    # track_async zählt die Tokens jedes Judge-Aufrufs
    client = track_async(
        AsyncOpenAI(
            timeout=float(cfg["judge"].get("request_timeout", 120.0)),
            max_retries=2,
        )
    )

    # Die Einstellungen werden ausdrücklich gesetzt, sonst nimmt ragas eigene Voreinstellungen (temperature=0.01, top_p=0.1, max_tokens=1024) und der Judge liefe anders als bei DeepEval und TruLens (temperature=0).
    # top_p=1.0 entspricht dem API-Standard, den die anderen Adapter nutzen.
    llm = llm_factory(
        cfg["judge"]["model"],
        client=client,
        temperature=float(cfg["judge"].get("temperature", 0.0)),
        top_p=1.0,
        # Faithfulness zerlegt die Antwort zuerst in Einzelaussagen. Bei langen Aufzählungen reichen 1024 Tokens dafür nicht, und der Score fehlt.
        max_tokens=int(cfg["judge"].get("max_tokens", 4096)),
    )

    # AnswerRelevancy erzeugt aus der Antwort Fragen und vergleicht sie über Embeddings mit der gestellten Frage.
    # Dafür wird dasselbe lokale Modell wie im RAG-System verwendet.
    embeddings = embedding_factory(
        provider="huggingface", model=cfg["rag"]["embedding"]["model"]
    )
    return llm, embeddings


class RagasEvaluator(BaseEvaluator):
    name = "ragas"
    metric_mapping = {
        Construct.FAITHFULNESS: "faithfulness",
        Construct.CONTEXT_RELEVANCE: "context_relevance",
        Construct.ANSWER_RELEVANCE: "answer_relevancy",
    }

    def evaluate(self, samples: list[GeneratedSample]) -> list[MetricResult]:
        _install_langchain_vertexai_shim()

        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextRelevance,
            Faithfulness,
        )

        llm, embeddings = _build_components(self.cfg)
        metrics = {
            Construct.FAITHFULNESS: Faithfulness(llm=llm),
            Construct.CONTEXT_RELEVANCE: ContextRelevance(llm=llm),
            Construct.ANSWER_RELEVANCE: AnswerRelevancy(llm=llm, embeddings=embeddings),
        }
        meta = {
            "judge_model": self.cfg["judge"]["model"],
            "framework": self.name,
            "answer_relevancy_embeddings": self.cfg["rag"]["embedding"]["model"],
        }

        async def score_one(s: GeneratedSample, construct: Construct):
            """Ruft ascore mit den Argumenten auf, die die Metrik erwartet.

            Rückgabe ist das MetricResult von ragas, der Score steht in value.
            """
            metric = metrics[construct]
            if construct is Construct.FAITHFULNESS:
                r = await metric.ascore(
                    user_input=s.query,
                    response=s.generated_answer,
                    retrieved_contexts=s.retrieved_contexts,
                )
            elif construct is Construct.CONTEXT_RELEVANCE:
                r = await metric.ascore(
                    user_input=s.query, retrieved_contexts=s.retrieved_contexts
                )
            else:
                r = await metric.ascore(user_input=s.query, response=s.generated_answer)
            return r

        async def run_all() -> list[MetricResult]:
            out: list[MetricResult] = []
            for s in tqdm(samples, desc="ragas"):
                for construct in metrics:
                    try:
                        r = await score_one(s, construct)
                        score = float(r.value)
                        # Bei internen Fehlern gibt ragas NaN zurück, statt eine Ausnahme zu werfen.
                        if score != score:  # NaN
                            raise ValueError("ragas lieferte NaN (interner Fehler)")
                        out.append(
                            MetricResult(
                                sample_id=s.sample_id,
                                dataset=s.dataset,
                                framework=self.name,
                                construct=construct,
                                native_metric_name=self.metric_mapping[construct],
                                score=score,
                                reasoning=getattr(r, "reason", None),
                                meta=meta,
                            )
                        )
                    except Exception as e:  # einzelne Bewertung überspringen
                        logger.warning(
                            "[ragas] %s/%s fehlgeschlagen: %s",
                            s.sample_id,
                            construct.value,
                            e,
                        )
            return out

        return asyncio.run(run_all())
