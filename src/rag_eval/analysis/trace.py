"""TRACe als vierte Methode auf den Tripeln aus RAGBench:

Die TRACe-Labels stehen nicht in einer Score-Datei, sondern in meta der Antwortdatei ragbench_own_generated.jsonl, weil sie mit RAGBench geliefert werden.
Dieses Modul macht daraus MetricResults, damit RQ2 und RQ3 sie genauso auswerten wie die Frameworks.

Zuordnung zu den Konstrukten:
    adherence_score  -> faithfulness       (binär, True/False)
    relevance_score  -> context_relevance  (abgestuft, Anteil relevanter Kontext)
Für Answer Relevance gibt es kein Gegenstück.
utilization_score und completeness_score messen etwas anderes und gehen nicht ein.

Weil adherence binär ist, wird es mit abgestuften Scores rang-biserial korreliert statt mit Spearman und in der MTMM-Matrix getrennt von den drei Blöcken berichtet.

Zur Plausibilitätsprüfung liefert RAGBench zu denselben Tripeln auch eigene Ragas- und TruLens-Werte, mit anderem Judge und anderen Versionen.
Sie sind nicht direkt vergleichbar, eignen sich aber als grobe Prüfung der Adapter.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

from ..data.schema import Construct, GeneratedSample, MetricResult, runde_score

TRACE = "trace"

# Feld in meta je Konstrukt.
TRACE_FELDER = {
    Construct.FAITHFULNESS: "adherence_score",
    Construct.CONTEXT_RELEVANCE: "relevance_score",
}

# Metriken mit nur zwei Werten, als feste Liste und nicht aus den Daten abgeleitet.
# Eine abgestufte Metrik, die zufällig nur zwei Werte hat, bleibt abgestuft und fällt unter die Ausschlussregel.
BINAERE_METRIKEN = frozenset({f"{TRACE}/{Construct.FAITHFULNESS.value}"})

# Eigene Metrik -> Spalte mit dem Wert, den RAGBench mitliefert.
ORIGINAL_SPALTEN = {
    "ragas/faithfulness": "ragas_faithfulness",
    "ragas/context_relevance": "ragas_context_relevance",
    "trulens/faithfulness": "trulens_groundedness",
    "trulens/context_relevance": "trulens_context_relevance",
}


def trace_ergebnisse(samples: list[GeneratedSample]) -> list[MetricResult]:
    """TRACe-Labels als MetricResults, erste Antwort je Frage.

    meta["extern"] = True kennzeichnet sie als mitgeliefert.
    Sie stammen aus der Annotation von RAGBench und nicht vom eigenen Judge, die Judge-Prüfung in scripts/06 überspringt sie deshalb.
    """
    ergebnisse: list[MetricResult] = []
    gesehen: set[str] = set()
    for s in samples:
        if s.sample_id in gesehen:
            continue
        gesehen.add(s.sample_id)
        for construct, feld in TRACE_FELDER.items():
            wert = s.meta.get(feld)
            if wert is None:
                continue
            ergebnisse.append(
                MetricResult(
                    sample_id=s.sample_id,
                    dataset=s.dataset,
                    framework=TRACE,
                    construct=construct,
                    native_metric_name=feld,
                    score=float(wert),
                    meta={"extern": True, "judge_model": "RAGBench-Annotation"},
                )
            )
    return ergebnisse


def originalwerte(
    samples: list[GeneratedSample], zeilen_je_frage: dict
) -> pd.DataFrame:
    """Die von RAGBench mitgelieferten Ragas- und TruLens-Werte je sample_id.

    zeilen_je_frage: {(subset, RAGBench-id): [HF-Zeilen]}.
    Zu einer Frage gibt es je Antwortmodell eine Zeile.
    Gewählt wird die, deren response genau die bewertete Antwort ist, sonst könnten Werte einer anderen Antwort auf dieselbe Frage genommen werden.

    Raises:
        ValueError: wenn es für eine Antwort keine passende Zeile gibt.
    """
    zeilen = []
    fehlend = []
    for s in samples:
        subset = s.meta.get("subset")
        rid = s.sample_id.split(":", 2)[2]
        treffer = [
            z
            for z in zeilen_je_frage.get((subset, rid), [])
            if z["response"] == s.generated_answer
        ]
        if not treffer:
            fehlend.append(s.sample_id)
            continue
        z = treffer[0]
        zeilen.append(
            {
                "sample_id": s.sample_id,
                "subset": subset,
                **{
                    spalte: runde_score(z.get(spalte))
                    for spalte in ORIGINAL_SPALTEN.values()
                },
            }
        )
    if fehlend:
        raise ValueError(
            f"{len(fehlend)} Antwort(en) ohne passende RAGBench-Zeile, "
            f"z.B. {fehlend[:3]}."
        )
    return pd.DataFrame(zeilen).set_index("sample_id")


def plausibilitaet(eigene: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    """Eigene Scores gegen die von RAGBench, über alle und je Subset.

    eigene:     Wide-Format sample_id x Metrik ("ragas/faithfulness", ...).
    original:   Ausgabe von originalwerte.
    """
    daten = original.join(eigene, how="inner")
    gruppen = [("alle", daten)] + [(str(g), t) for g, t in daten.groupby("subset")]
    zeilen = []
    for metrik, spalte in ORIGINAL_SPALTEN.items():
        if metrik not in daten:
            continue
        for name, teil in gruppen:
            paar = teil[[metrik, spalte]].apply(pd.to_numeric, errors="coerce").dropna()
            n = len(paar)
            if n >= 3 and paar[metrik].nunique() > 1 and paar[spalte].nunique() > 1:
                rho, p_rho = stats.spearmanr(paar[metrik], paar[spalte])
                tau, p_tau = stats.kendalltau(paar[metrik], paar[spalte])
            else:
                rho = p_rho = tau = p_tau = float("nan")
            zeilen.append(
                {
                    "metrik": metrik,
                    "original": spalte,
                    "gruppe": name,
                    "n": n,
                    "spearman_rho": rho,
                    "spearman_p": p_rho,
                    "kendall_tau": tau,
                    "kendall_p": p_tau,
                    "median_eigen": paar[metrik].median() if n else float("nan"),
                    "median_original": paar[spalte].median() if n else float("nan"),
                }
            )
    return pd.DataFrame(zeilen)
