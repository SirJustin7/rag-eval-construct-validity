"""Gemeinsame Schnittstelle der Framework-Adapter.

Jedes Framework bewertet dieselben Tripel (query, retrieved_contexts, generated_answer) und ordnet seine Metriken den drei Konstrukten der RAG-Triade zu."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..data.schema import Construct, GeneratedSample, MetricResult


class BaseEvaluator(ABC):
    """Basisklasse: ein Framework und gemeinsames Ergebnisformat."""

    # Name des Frameworks ("ragas", "trulens", ...)
    name: str

    # Konstrukt -> Name der Metrik im Framework
    metric_mapping: dict[Construct, str]

    # Bei False bricht scripts/03 vor dem ersten Judge-Aufruf ab
    implemented: bool = True

    def __init__(self, cfg: dict[str, Any]):
        """cfg ist die ganze Konfiguration, der Judge steht unter cfg["judge"]."""
        self.cfg = cfg

    @abstractmethod
    def evaluate(self, samples: list[GeneratedSample]) -> list[MetricResult]:
        """Bewertet alle Samples und liefert MetricResults für alle Konstrukte.

        Jeder Adapter soll
        - Scores auf [0, 1] bringen
        - native_metric_name aus metric_mapping setzen
        - Judge-Modell und Framework in meta eintragen
        - fehlgeschlagene Bewertungen loggen und überspringen"""
