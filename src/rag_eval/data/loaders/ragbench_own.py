"""RAGBench mit den Antworten aus RAGBench selbst, für den Vergleich mit TRACe:

Die TRACe-Labels bewerten response, also die Antwort, die in RAGBench mitgeliefert wird.
Der Hauptlauf bewertet dagegen die Antwort des eigenen Generators.
TRACe und die drei Frameworks würden so verschiedene Texte bewerten.
Dieser Datensatz verwendet deshalb die Tripel aus RAGBench:

    query              = question   (wie im Hauptlauf)
    retrieved_contexts = documents  (statt der eigenen top_k Chunks)
    generated_answer   = response   (statt der Antwort des Generators)

Es gibt kein Retrieval und keine eigene Generierung. Der Datensatz läuft wie depvalidation nicht durch 01 und 02, sondern über scripts/02c_import_ragbench_own.py.
Deshalb steht source: "local" in der config.
"""

from __future__ import annotations

from ..schema import GeneratedSample
from .ragbench import load_ragbench


def baue_eigene_tripel(
    subsets: list[str] | None = None,
    sample_size: int | None = None,
    seed: int = 42,
) -> list[GeneratedSample]:
    """Lädt RAGBench und setzt die Antwort aus RAGBench als bewertete Antwort ein.

    Dieselben Argumente wie load_ragbench. Mit den Werten aus der config ergibt sich dieselbe Fragenauswahl wie im Hauptlauf.
    """
    tripel: list[GeneratedSample] = []
    for s in load_ragbench(subsets=subsets, sample_size=sample_size, seed=seed):
        # Eigene sample_id, sonst hätten Bewertungen verschiedener Antworten dieselbe ID. Die ursprüngliche ID steht in meta.
        neue_id = s.sample_id.replace("ragbench:", "ragbench_own:", 1)
        tripel.append(
            GeneratedSample(
                sample_id=neue_id,
                dataset="ragbench_own",
                query=s.query,
                retrieved_contexts=list(s.gold_contexts),
                generated_answer=s.reference_answer or "",
                # Referenz und bewertete Antwort sind hier derselbe Text.
                # DeepEval bekommt sie als expected_output.
                reference_answer=s.reference_answer,
                meta={
                    **s.meta,
                    "ragbench_sample_id": s.sample_id,
                    # Wie meta["rag_system"] im Hauptlauf, hier mit RAGBench als Herkunft.
                    "rag_system": {
                        "generator_model": "RAGBench (response)",
                        "retrieval_scope": "RAGBench documents",
                        "top_k": None,
                    },
                },
            )
        )
    return tripel


def nicht_ueber_pipeline(**_kwargs) -> None:
    """Platzhalter in LOADERS (siehe loaders/__init__.py).

    Eigene Antworten passen nicht zu den TRACe-Labels, die zu response gehören.
    """
    raise NotImplementedError(
        "ragbench_own bringt RAGBenchs eigene (Frage, Kontext, Antwort)-Tripel "
        "mit und wird nicht indexiert oder neu generiert.\n"
        "Stattdessen:  python scripts/02c_import_ragbench_own.py\n"
        "Danach wie gewohnt: python scripts/03_run_frameworks.py "
        "--dataset ragbench_own"
    )
