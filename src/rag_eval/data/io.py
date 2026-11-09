"""Lesen und Schreiben von JSONL-Dateien:"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def write_jsonl(items: Iterable[BaseModel], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(item.model_dump_json() + "\n" for item in items)


def read_jsonl(path: Path | str, model: type[T]) -> list[T]:
    with open(path, encoding="utf-8") as f:
        return [model.model_validate(json.loads(line)) for line in f if line.strip()]


class DoppelteSampleIds(ValueError):
    """Mehrere Einträge teilen sich dieselbe sample_id."""


def _doppelte(items) -> dict[str, int]:
    zaehler: dict[str, int] = {}
    for it in items:
        sid = getattr(it, "sample_id", None)
        if sid is not None:
            zaehler[sid] = zaehler.get(sid, 0) + 1
    return {sid: n for sid, n in zaehler.items() if n > 1}


def pruefe_eindeutige_ids(items, kontext: str) -> None:
    """Bricht ab, wenn eine sample_id mehrfach vorkommt.

    In RAGBench bezeichnet id die Frage, nicht die Zeile.
    Zu einer Frage gibt es mehrere Zeilen, je eine pro Antwortmodell, sodass ohne diese Prüfung Fragen doppelt gezogen werden.
    """
    doppelt = _doppelte(items)
    if doppelt:
        beispiele = ", ".join(list(doppelt)[:5])
        raise DoppelteSampleIds(
            f"{kontext}: {len(doppelt)} sample_id(s) kommen mehrfach vor "
            f"(z.B. {beispiele}). Eindeutige IDs sind Voraussetzung — sonst "
            f"zählen Samples doppelt und die Auswertung bricht später ab."
        )


def entdoppele(items, kontext: str) -> list:
    """Behält je sample_id den ersten Eintrag und meldet, wie viele wegfallen.

    Alle Frameworks lesen dieselbe Datei in derselben Reihenfolge, die Auswahl ist also überall dieselbe.
    """
    doppelt = _doppelte(items)
    if not doppelt:
        return list(items)
    gesehen: set[str] = set()
    sauber = []
    for it in items:
        sid = getattr(it, "sample_id", None)
        if sid in gesehen:
            continue
        if sid is not None:
            gesehen.add(sid)
        sauber.append(it)
    print(
        f"⚠️  {kontext}: {len(items) - len(sauber)} Dublette(n) entfernt "
        f"({len(doppelt)} betroffene sample_id(s)); es bleiben {len(sauber)}."
    )
    return sauber
