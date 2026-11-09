"""Das RAG-System aus Indexing, Retrieval und Generation:

Alle Antworten stammen aus demselben, fest konfigurierten System, damit Unterschiede in den Scores auf die Frameworks zurückgehen.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tqdm import tqdm

from .. import naming
from ..data.schema import GeneratedSample, RAGSample
from . import generation, indexing, retrieval


class RAGSystem:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    def index_dataset(self, samples: list[RAGSample], collection_name: str) -> None:
        """Indexiert die Dokumente eines Datensatzes."""
        documents = [ctx for s in samples for ctx in s.gold_contexts]
        indexing.build_index(documents, collection_name, self.cfg["rag"])

    def answer(self, sample: RAGSample, collection_name: str) -> GeneratedSample:
        """Beantwortet eine Frage (Retrieval und Generation)."""
        contexts = retrieval.retrieve(sample.query, collection_name, self.cfg["rag"])
        answer = generation.generate_answer(sample.query, contexts, self.cfg["rag"])
        return GeneratedSample(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            query=sample.query,
            retrieved_contexts=contexts,
            generated_answer=answer,
            reference_answer=sample.reference_answer,
            # Einstellungen mitschreiben, mit denen die Antwort entstanden ist, da top_k zwischen den Läufen variiert.
            # Eigener Schlüssel, damit nichts mit den Feldern der Datensätze kollidiert.
            meta={**sample.meta, "rag_system": self._herkunft()},
        )

    def _herkunft(self) -> dict[str, Any]:
        """Die Einstellungen, mit denen die Antwort erzeugt wurde."""
        rag = self.cfg["rag"]
        return {
            "generator_model": rag["generator"]["model"],
            "temperature": rag["generator"]["temperature"],
            "max_tokens": rag["generator"]["max_tokens"],
            "top_k": rag["retrieval"]["top_k"],
            "retrieval_scope": naming.scope_aus_config(rag),
            "embedding_model": rag["embedding"]["model"],
            "chunk_size": rag["chunking"]["chunk_size"],
            "chunk_overlap": rag["chunking"]["chunk_overlap"],
        }

    def run(
        self,
        samples: list[RAGSample],
        collection_name: str | None = None,
        collection_for: Callable[[RAGSample], str] | None = None,
    ) -> list[GeneratedSample]:
        """Beantwortet mehrere Samples.

        Entweder collection_name (eine Collection für alle, Zuschnitt pooled)
        oder collection_for (Collection je Sample, Zuschnitt per_domain).
        """
        if (collection_name is None) == (collection_for is None):
            raise ValueError("Genau eines von collection_name/collection_for setzen.")
        waehle = collection_for or (lambda _s: collection_name)
        return [self.answer(s, waehle(s)) for s in tqdm(samples, desc="RAG generation")]
