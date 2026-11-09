"""Retrieval: die top_k ähnlichsten Chunks zu einer Frage:"""

from __future__ import annotations

from typing import Any

import chromadb

# Derselbe Embedder wie beim Indexing, damit Frage und Chunks im selben Vektorraum liegen.
from .indexing import _get_embedder


def retrieve(query: str, collection_name: str, cfg: dict[str, Any]) -> list[str]:
    """Gibt die top_k ähnlichsten Chunks zurück, bester Treffer zuerst.

    cfg ist der rag-Block der Konfiguration.
    """
    # normalize_embeddings passt zur Cosinus-Distanz der Collection
    embedder = _get_embedder(
        cfg["embedding"]["model"], cfg["embedding"].get("device", "cpu")
    )
    query_embedding = embedder.encode([query], normalize_embeddings=True)

    client = chromadb.PersistentClient(path=cfg["vector_store"]["persist_dir"])
    collection = client.get_collection(collection_name)

    result = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=cfg["retrieval"]["top_k"],
    )

    # Je Query eine Liste von Treffern, hier gibt es nur eine Query.
    # Bei leerer Collection kann documents None sein.
    documents = result["documents"]
    return documents[0] if documents else []
