"""Datenstrukturen der Pipeline:

Alle Datensätze werden auf RAGSample abgebildet (Frage, Kontexte, Referenzantwort).
Das RAG-System erzeugt daraus ein GeneratedSample, das alle Frameworks mit denselben Eingaben bewerten.
Jeder Score wird als MetricResult gespeichert.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

# Gemittelte Ratings landen durch Gleitkommarauschen auf verschiedenen Bits (0.19999999999999998 und 0.2), zählen dann als zwei Werte und brechen Bindungen in den Rängen.
# Neun Stellen liegen weit unter der feinsten echten Stufe (1/30).
SCORE_STELLEN = 9


def runde_score(x):
    """Score auf SCORE_STELLEN runden, None bleibt None."""
    return None if x is None else round(float(x), SCORE_STELLEN)


class Construct(str, Enum):
    """Die drei Konstrukte der RAG-Triade."""

    # Ragas Faithfulness, TruLens Groundedness, DeepEval Faithfulness, TRACe Adherence
    FAITHFULNESS = "faithfulness"
    CONTEXT_RELEVANCE = "context_relevance"
    ANSWER_RELEVANCE = "answer_relevance"


class RAGSample(BaseModel):
    """Ein Beispiel aus einem Datensatz: Frage, Dokumente, Referenzantwort."""

    sample_id: str
    dataset: str  # "crag" | "ragbench" | "depvalidation"
    query: str
    gold_contexts: list[str] = Field(
        default_factory=list
    )  # Dokumente zur Frage, falls vorhanden
    reference_answer: str | None = None
    meta: dict = Field(default_factory=dict)  # z.B. Domäne und Fragetyp bei CRAG


class GeneratedSample(BaseModel):
    """Ausgabe des RAG-Systems für ein RAGSample."""

    sample_id: str
    dataset: str
    query: str
    retrieved_contexts: list[str]  # top_k Chunks aus dem Retrieval
    generated_answer: str
    reference_answer: str | None = None
    meta: dict = Field(default_factory=dict)


class MetricResult(BaseModel):
    """Ein Score eines Frameworks für ein Konstrukt, in [0, 1]."""

    sample_id: str
    dataset: str
    framework: str  # "ragas" | "trulens" | "deepeval" | "trace"
    construct: Construct
    native_metric_name: str  # Name der Metrik im Framework, z.B. "groundedness"
    score: float
    reasoning: str | None = (
        None  # Begründung des Judge, falls das Framework sie liefert
    )
    meta: dict = Field(default_factory=dict)  # z.B. Judge-Modell

    @field_validator("score")
    @classmethod
    def _rauschen_weg(cls, v: float) -> float:
        return runde_score(v)
