"""RAGBench (Friel et al. 2024):

Etwa 100.000 Beispiele aus fünf Fachdomänen, mit TRACe-Labels (relevance, utilization, adherence, completeness) zu jeder Antwort.
"""

from __future__ import annotations

import random

from datasets import load_dataset

from ..schema import RAGSample

# HuggingFace-Adresse, wie datasets.ragbench.hf_id in config.yaml.
HF_ID = "rungalileo/ragbench"

# Allgemeinwissen, Finanzen, Technik. Dieselben Subsets stehen in config.yaml.
DEFAULT_SUBSETS = ["hotpotqa", "finqa", "techqa"]

# Die vier TRACe-Labels aus RAGBench (mit GPT-4 annotiert). Sie kommen nach RAGSample.meta.
_TRACE_KEYS = [
    "relevance_score",
    "utilization_score",
    "adherence_score",
    "completeness_score",
]


def load_ragbench(
    subsets: list[str] | None = None,
    sample_size: int | None = None,
    seed: int = 42,
) -> list[RAGSample]:
    """Lädt RAGBench von HuggingFace (rungalileo/ragbench) und mappt auf RAGSample.

    Args:
        subsets:        Domänen, z.B. ["hotpotqa", "finqa"]. None = DEFAULT_SUBSETS.
        sample_size:    Gesamtzahl über alle Domänen. None = alles laden.
        seed:           Legt die Zufallsauswahl fest.

    Returns:
        RAGSamples, gleich viele je Domäne.
    """
    subsets = subsets or DEFAULT_SUBSETS

    # Gleich viele je Domäne, z.B. 333 bei drei Domänen = 111 je Domäne.
    per_subset = None if sample_size is None else max(1, sample_size // len(subsets))

    samples: list[RAGSample] = []
    for subset in subsets:
        ds = load_dataset(HF_ID, subset, split="test")
        ds = ds.shuffle(seed=seed)

        # In RAGBench bezeichnet id die Frage, nicht die Zeile. Zu einer Frage gibt es mehrere Zeilen, je eine pro Antwortmodell.
        # Die ersten n Zeilen enthalten deshalb dieselbe Frage mehrfach. Genommen wird je id nur die erste Zeile nach dem Mischen.
        gesehene_fragen: set = set()

        for row in ds:
            if per_subset is not None and len(gesehene_fragen) >= per_subset:
                break
            if row["id"] in gesehene_fragen:
                continue
            gesehene_fragen.add(row["id"])
            samples.append(
                RAGSample(
                    # Mit Subset im Namen, damit die IDs über Domänen eindeutig sind.
                    sample_id=f"ragbench:{subset}:{row['id']}",
                    dataset="ragbench",
                    query=row["question"],
                    gold_contexts=list(row["documents"]),
                    reference_answer=row["response"],
                    meta={
                        "subset": subset,
                        **{k: row.get(k) for k in _TRACE_KEYS},
                    },
                )
            )

    # Geht die Aufteilung nicht genau auf, zufällig auf sample_size kürzen.
    if sample_size is not None and len(samples) > sample_size:
        rng = random.Random(seed)
        samples = rng.sample(samples, sample_size)

    return samples
