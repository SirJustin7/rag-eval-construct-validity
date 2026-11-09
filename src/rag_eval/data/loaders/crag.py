"""CRAG, Comprehensive RAG Benchmark (Yang et al. 2024):

Verwendet wird die originale Release-Datei crag_task_1_dev_v4_release.jsonl.bz2 (Task 1, 2.706 Fragen), gespiegelt auf HuggingFace unter Rickyoung0221/crag.
Andere Spiegel wie Quivr/CRAG enthalten nur die Suchergebnisse, ohne Frage, Antwort und Fragetyp.

Die fünf HTML-Seiten je Frage werden zu Text umgewandelt und als gold_contexts in den Index gegeben.
Sie sind also das Korpus, aus dem der eigene Retriever wählt, und werden nicht direkt als Kontext durchgereicht.
Sonst hätte top_k bei CRAG keine Wirkung und die Datensätze wären nicht leich behandelt.
CRAG-Seiten sind lang und verrauscht, eine Frage ergibt eshalb deutlich mehr Chunks als bei RAGBench.

domain, question_type, static_or_dynamic und popularity kommen nach meta.
Bei question_type == "false_premise" ist eine Zurückweisung die richtige Antwort.
"""

from __future__ import annotations

import bz2
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..schema import RAGSample

# HF-Repo mit der unveränderten Release-Datei.
HF_REPO = "Rickyoung0221/crag"
HF_FILE = "crag_task_1_dev_v4_release.jsonl.bz2"

# Elemente ohne Inhalt, die sonst Navigation und Skripte in den Index bringen.
_DROP_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "form",
    "svg",
    "iframe",
    "aside",
)

# Kürzere Texte sind meist Menüpunkte oder Cookie-Hinweise.
_MIN_TEXT_LEN = 200

_WS = re.compile(r"\s+")


def _html_to_text(html: str) -> str:
    """Wandelt eine HTML-Seite in Fließtext um.

    lxml statt BeautifulSoup, weil die Seiten sehr groß sind und bs4 dabei deutlich langsamer ist.
    """
    if not html:
        return ""
    try:
        from lxml import html as lxml_html

        baum = lxml_html.fromstring(html)
    except Exception:
        # Eine unlesbare Seite fällt weg, die übrigen Seiten der Frage bleiben.
        return ""
    for tag in _DROP_TAGS:
        for el in baum.iter(tag):
            el.drop_tree()
    return _WS.sub(" ", baum.text_content()).strip()


def _zeilen(pfad: str) -> Iterator[str]:
    """Liest die bz2-Datei zeilenweise, ohne sie ganz in den Speicher zu laden."""
    with bz2.open(pfad, "rt", encoding="utf-8") as f:
        for zeile in f:
            zeile = zeile.strip()
            if zeile:
                yield zeile


def _release_datei(datei: str | None = None) -> str:
    """Pfad zur Release-Datei, beim ersten Aufruf in den HF-Cache geladen.

    Mit datei wird stattdessen eine lokale Kopie verwendet, falls der Mirror nicht mehr erreichbar ist.
    """
    if datei:
        return datei
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset")


def _kontexte(row: dict[str, Any]) -> list[str]:
    """Macht aus den HTML-Seiten einer Frage Texte, ohne Dubletten.

    Manche Fragen enthalten dieselbe Seite mehrfach. Doppelt im Index würde sie das Retrieval verzerren.
    """
    texte: list[str] = []
    gesehen: set[int] = set()
    for seite in row.get("search_results") or []:
        text = _html_to_text(seite.get("page_result") or "")
        if len(text) < _MIN_TEXT_LEN:
            # Seite leer oder unlesbar, dann wenigstens den Snippet nehmen.
            text = (seite.get("page_snippet") or "").strip()
            if len(text) < _MIN_TEXT_LEN:
                continue
        h = hash(text)
        if h in gesehen:
            continue
        gesehen.add(h)
        texte.append(text)
    return texte


def load_crag(
    sample_size: int | None = None,
    seed: int = 42,
    domains: list[str] | None = None,
    datei: str | None = None,
) -> list[RAGSample]:
    """Lädt CRAG (Task 1) und mappt auf RAGSample.

    Args:
        sample_size:    Gesamtzahl der Samples, gleichmäßig auf die Domänen verteilt wie bei den RAGBench-Subsets.
        seed:           Legt Mischung und Auswahl fest.
        domains:        Nur diese CRAG-Domänen. None = alle fünf (finance, sports, music, movie, open).
        datei:          Lokale Kopie der Release-Datei statt Download.

    Die Datei wird zweimal gelesen. Zuerst nur die Domäne jeder Zeile für die Auswahl, danach nur die gewählten Zeilen mit der HTML-Umwandlung.
    """
    pfad = _release_datei(datei)

    # Durchgang 1: Domäne je Zeile. Das Entpacken dauert einige Minuten, das Ergebnis wird deshalb neben der Datei zwischengespeichert.
    cache = Path(pfad).with_suffix(".domainindex.json")
    if cache.exists():
        roh = json.loads(cache.read_text(encoding="utf-8"))
    else:
        roh = defaultdict(list)
        for i, zeile in enumerate(_zeilen(pfad)):
            roh[json.loads(zeile).get("domain") or "unknown"].append(i)
        roh = dict(roh)
        try:
            cache.write_text(json.dumps(roh), encoding="utf-8")
        except OSError:
            pass  # ohne Cache geht es auch, nur langsamer

    nach_domain: dict[str, list[int]] = {
        dom: idx for dom, idx in roh.items() if not domains or dom in domains
    }

    rng = random.Random(seed)
    gewaehlt: set[int] = set()
    if sample_size is None:
        for idx in nach_domain.values():
            gewaehlt.update(idx)
    else:
        # Gleich viele je Domäne, der Rest wird zufällig aufgefüllt.
        je_domain = max(1, sample_size // max(1, len(nach_domain)))
        for dom in sorted(nach_domain):
            idx = sorted(nach_domain[dom])
            rng.shuffle(idx)
            gewaehlt.update(idx[:je_domain])
        if len(gewaehlt) < sample_size:
            rest = sorted({i for idx in nach_domain.values() for i in idx} - gewaehlt)
            rng.shuffle(rest)
            gewaehlt.update(rest[: sample_size - len(gewaehlt)])

    # Durchgang 2: nur die gewählten Zeilen, mit HTML-Umwandlung.
    samples: list[RAGSample] = []
    ohne_kontext = 0
    for i, zeile in enumerate(_zeilen(pfad)):
        if i not in gewaehlt:
            continue
        row = json.loads(zeile)
        kontexte = _kontexte(row)
        if not kontexte:
            # Ohne Kontext kann das RAG-System die Frage nicht bearbeiten.
            ohne_kontext += 1
            continue
        samples.append(
            RAGSample(
                sample_id=f"crag:{row['interaction_id']}",
                dataset="crag",
                query=row["query"],
                gold_contexts=kontexte,
                reference_answer=row.get("answer"),
                meta={
                    "domain": row.get("domain"),
                    "question_type": row.get("question_type"),
                    "static_or_dynamic": row.get("static_or_dynamic"),
                    "popularity": row.get("popularity"),
                    "split": row.get("split"),
                    "query_time": row.get("query_time"),
                    "alt_ans": row.get("alt_ans"),
                    "n_seiten": len(kontexte),
                },
            )
        )

    # Melden, sonst wäre die Stichprobe unbemerkt kleiner als sample_size.
    if ohne_kontext:
        print(
            f"[crag] {ohne_kontext} von {len(gewaehlt)} Samples ohne "
            f"verwertbaren Kontext verworfen (HTML leer/zu kurz)."
        )

    # Geht die Aufteilung nicht genau auf, zufällig auf sample_size kürzen.
    if sample_size is not None and len(samples) > sample_size:
        samples = random.Random(seed).sample(samples, sample_size)
    return samples
