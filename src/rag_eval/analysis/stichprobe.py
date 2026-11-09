"""Welche Antworten in RQ2 und RQ3 eingehen:

Zwei Regeln, beide prüfen nur den Antworttext und nie die Scores:

* Leere Antworten entstehen, wenn die Generierung fehlschlägt, und fallen immer heraus. Ihre Zahl wird berichtet.
* Abstinenzen (jeder Satz benennt eine Lücke im Kontext, siehe analysis.abstinenz) bekommen bei mehreren Frameworks denselben Extremwert.
    Das erhöht die Übereinstimmung, ohne dass die Frameworks sich inhaltlich einig sind.
    Jeder Koeffizient wird deshalb mit und ohne sie berichtet.

Je Frage zählt die erste Antwort der Datei, wie in correlations.entdoppele_ergebnisse und scripts/03.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..data.io import read_jsonl
from ..data.schema import GeneratedSample, MetricResult
from .abstinenz import Antworttyp, klassifiziere

MIT = "mit_abst"
OHNE = "ohne_abst"
VARIANTEN = {"mit": [MIT], "ohne": [OHNE], "beide": [MIT, OHNE]}


def antworttypen(samples: list[GeneratedSample]) -> dict[str, Antworttyp]:
    """Antworttyp je sample_id, erste Antwort je Frage."""
    typen: dict[str, Antworttyp] = {}
    for s in samples:
        typen.setdefault(s.sample_id, klassifiziere(s.generated_answer))
    return typen


def antworttypen_aus_datei(pfad: str | Path) -> dict[str, Antworttyp]:
    return antworttypen(read_jsonl(pfad, GeneratedSample))


@dataclass(frozen=True)
class Filterbericht:
    variante: str
    n_antworten: int
    n_leer: int
    n_abstinenz: int
    n_abstinenz_entfernt: int
    n_behalten: int

    def zeile(self) -> str:
        return (
            f"{self.variante}: {self.n_behalten} von {self.n_antworten} Antworten "
            f"(leer entfernt: {self.n_leer}, Abstinenzen: {self.n_abstinenz}, "
            f"davon entfernt: {self.n_abstinenz_entfernt})"
        )


def filtere(
    results: list[MetricResult], typen: dict[str, Antworttyp], variante: str
) -> tuple[list[MetricResult], Filterbericht]:
    """Entfernt leere Antworten immer, Abstinenzen nur in der Variante OHNE.

    Raises:
        ValueError: wenn ein Score zu keiner Antwort der Antwortdatei gehört.
            Dann wurde die falsche Antwortdatei gelesen.
    """
    if variante not in (MIT, OHNE):
        raise ValueError(f"Unbekannte Variante {variante!r}.")
    fremd = {r.sample_id for r in results} - typen.keys()
    if fremd:
        beispiel = ", ".join(sorted(fremd)[:3])
        raise ValueError(
            f"{len(fremd)} Score-Zeile(n) ohne Antwort in der Antwortdatei (z.B. {beispiel}). "
            "Score- und Antwortdatei passen nicht zusammen."
        )

    raus = {Antworttyp.LEER} | ({Antworttyp.ABSTINENZ} if variante == OHNE else set())
    behalten = [r for r in results if typen[r.sample_id] not in raus]

    ids = {r.sample_id for r in results}
    zaehle = lambda t: sum(1 for i in ids if typen[i] is t)
    n_abst = zaehle(Antworttyp.ABSTINENZ)
    bericht = Filterbericht(
        variante=variante,
        n_antworten=len(ids),
        n_leer=zaehle(Antworttyp.LEER),
        n_abstinenz=n_abst,
        n_abstinenz_entfernt=n_abst if variante == OHNE else 0,
        n_behalten=len({r.sample_id for r in behalten}),
    )
    return behalten, bericht


@dataclass
class Datensatzlauf:
    """Alles, was RQ2 und RQ3 für einen Datensatz und eine Tiefe brauchen."""

    dataset: str
    top_k: int | None  # None bei importierten Datensätzen
    results: list[MetricResult]  # ohne Dubletten, bei ragbench_own mit TRACe
    typen: dict[str, Antworttyp]
    binaere_metriken: frozenset[str]

    @property
    def name(self) -> str:
        return self.dataset + (f"_k{self.top_k}" if self.top_k is not None else "")


def lade_datensatzlaeufe(
    pfade: list[Path], datasets: list[str], processed_dir: str, cfg: dict | None = None
) -> list[Datensatzlauf]:
    """Fasst die Score-Dateien je Datensatz zusammen und liest die Antwortdatei.

    Bei ragbench_own kommen die TRACe-Labels aus der Antwortdatei als vierte Methode dazu.
    """
    from ..naming import generated_zu_scores, score_datei_teile
    from .correlations import entdoppele_ergebnisse
    from .trace import BINAERE_METRIKEN, trace_ergebnisse

    unbekannt = [str(p) for p in pfade if score_datei_teile(p) is None]
    if unbekannt:
        raise ValueError(
            "Dateinamen folgen nicht dem Score-Schema, die Antwortdatei ist "
            f"nicht bestimmbar: {unbekannt}"
        )

    laeufe = []
    for ds in datasets:
        eigene = [p for p in pfade if score_datei_teile(p)["dataset"] == ds]
        if not eigene:
            continue
        tiefen = {score_datei_teile(p)["top_k"] for p in eigene}
        gen_pfade = {generated_zu_scores(p, processed_dir, cfg) for p in eigene}
        if len(tiefen) > 1 or len(gen_pfade) > 1:
            raise ValueError(
                f"[{ds}] Score-Dateien mehrerer Tiefen/Antwortdateien: {sorted(gen_pfade)}"
            )
        results: list[MetricResult] = []
        for p in eigene:
            results.extend(read_jsonl(p, MetricResult))
        results = entdoppele_ergebnisse(results)
        samples = read_jsonl(next(iter(gen_pfade)), GeneratedSample)
        binaer: frozenset[str] = frozenset()
        if ds == "ragbench_own":
            results += trace_ergebnisse(samples)
            binaer = BINAERE_METRIKEN
        laeufe.append(
            Datensatzlauf(ds, tiefen.pop(), results, antworttypen(samples), binaer)
        )
    return laeufe
