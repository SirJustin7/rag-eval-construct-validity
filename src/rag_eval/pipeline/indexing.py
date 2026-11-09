"""Indexing: Dokumente in Chunks zerlegen, einbetten und in ChromaDB speichern:"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Muss zu rag.embedding.model in der Konfiguration passen.
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Modelle und Tokenizer werden nur einmal geladen.
_model_cache: dict[str, SentenceTransformer] = {}
_tokenizer_cache: dict[str, Any] = {}


def _get_embedder(model_name: str, device: str = "cpu") -> SentenceTransformer:
    """Lädt das Embedding-Modell einmal und hält es im Cache.

    device ist fest auf "cpu" vorbelegt statt auf die Automatik von sentence-transformers.
    Auf dem Intel-Mac meldet torch 2.2.2 MPS als verfügbar, der Lauf hängt dann aber unregelmäßig beim Kopieren zurück auf die CPU.
    """
    schluessel = f"{model_name}@{device}"
    if schluessel not in _model_cache:
        _model_cache[schluessel] = SentenceTransformer(model_name, device=device)
    return _model_cache[schluessel]


def _get_tokenizer(model_name: str):
    """Lädt den Tokenizer des Embedding-Modells (für das Chunking in Tokens)."""
    if model_name not in _tokenizer_cache:
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer_cache[model_name]


def chunk_documents(
    documents: list[str],
    chunk_size: int,
    chunk_overlap: int,
    model_name: str = DEFAULT_EMBED_MODEL,
) -> list[str]:
    """Zerlegt Dokumente in überlappende Chunks, gemessen in Tokens.

    Gezählt wird mit dem Tokenizer des Embedding-Modells, weil dessen Token-Limit die Chunkgröße begrenzt.
    """
    tok = _get_tokenizer(model_name)
    # Schrittweite des Fensters, bei overlap > 0 überlappen die Chunks.
    step = max(1, chunk_size - chunk_overlap)

    chunks: list[str] = []
    for doc in documents:
        # In Tokens zählen, aber über offset_mapping aus dem Originaltext schneiden, damit Schreibweise und Zeichensetzung erhalten bleiben.
        offsets = tok(doc, add_special_tokens=False, return_offsets_mapping=True)[
            "offset_mapping"
        ]
        if not offsets:
            continue  # leeres Dokument
        for start in range(0, len(offsets), step):
            window = offsets[start : start + chunk_size]
            if not window:
                break
            char_start = window[0][0]
            char_end = window[-1][1]
            chunks.append(doc[char_start:char_end])
            # Reicht das Fenster bis zum Ende, entsteht kein Rest-Chunk aus reinem Overlap.
            if start + chunk_size >= len(offsets):
                break
    return chunks


def build_index(
    documents: list[str],
    collection_name: str,
    cfg: dict[str, Any],
) -> None:
    """Baut den Vektorindex für eine Collection auf.

    1. Dokumente in Chunks zerlegen
    2. Embeddings berechnen
    3. in ChromaDB speichern (data/vector_stores/)
    """
    model_name = cfg["embedding"]["model"]
    chunk_cfg = cfg["chunking"]

    # 1) Chunking
    chunks = chunk_documents(
        documents, chunk_cfg["chunk_size"], chunk_cfg["chunk_overlap"], model_name
    )
    if not chunks:
        raise ValueError(
            f"[{collection_name}] Keine Chunks erzeugt — sind Dokumente leer?"
        )

    # 2) Embeddings, normalisiert passend zur Cosinus-Distanz
    embedder = _get_embedder(model_name, cfg["embedding"].get("device", "cpu"))
    embeddings = embedder.encode(
        chunks, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )

    # 3) ChromaDB
    persist_dir = cfg["vector_store"]["persist_dir"]
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    # Vorhandene Collection löschen, damit erneutes Indexieren keine Dubletten erzeugt.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": cfg["retrieval"]["similarity"]},
    )

    # In Blöcken schreiben, weil chromadb add()-Aufrufe über einer internen Obergrenze ablehnt (CRAG hat 28.455 Chunks).
    # Die Grenze wird beim Client abgefragt.
    try:
        max_batch = client.get_max_batch_size()
    except Exception:
        max_batch = 5000
    ids = [f"{collection_name}-{i}" for i in range(len(chunks))]
    vektoren = embeddings.tolist()
    bloecke = (len(chunks) + max_batch - 1) // max_batch
    for nr, start in enumerate(range(0, len(chunks), max_batch), 1):
        ende = start + max_batch
        collection.add(
            ids=ids[start:ende],
            embeddings=vektoren[start:ende],
            documents=chunks[start:ende],
        )
        # Fortschritt nur ausgeben, chromadb verträgt keinen zweiten Prozess auf demselben Verzeichnis.
        print(
            f"[{collection_name}] Block {nr}/{bloecke} geschrieben "
            f"({min(ende, len(chunks))}/{len(chunks)} Chunks)",
            flush=True,
        )
    print(f"[{collection_name}] {len(chunks)} Chunks indexiert → {persist_dir}")
