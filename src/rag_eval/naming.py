"""Namen für Collections und Ergebnisdateien:

Das Retrieval kann über zwei Zuschnitte laufen:

pooled
    Eine Chroma-Collection je Datensatz. Alle Domänen liegen zusammen, eine Finanzfrage kann also einen Technik-Chunk abrufen.

per_domain (verwendet)
    Eine Collection je Domäne, bei CRAG fünf (finance, sports, music, movie, open), bei RAGBench drei (hotpotqa, finqa, techqa). CRAGs Webseiten wurden für die jeweilige Frage gesucht.
    Ein gemeinsamer Index aller Fragen bildet kein reales System ab.
    Je Domäne bleibt es trotzdem eine echte Retrieval-Aufgabe, weil jede Collection die Dokumente vieler Fragen enthält.

Der Zuschnitt steht im Namen der Collections und der Ergebnisdateien, so überschreibt ein Lauf den anderen nicht.
"""

from __future__ import annotations

import re
from typing import Any

POOLED = "pooled"
PER_DOMAIN = "per_domain"
ERLAUBTE_SCOPES = (POOLED, PER_DOMAIN)

# meta-Feld mit der Domäne, bei CRAG "domain", bei RAGBench "subset".
STANDARD_DOMAIN_FELDER = {"crag": "domain", "ragbench": "subset"}


def scope_aus_config(cfg: dict[str, Any]) -> str:
    """Liest den Zuschnitt aus dem "rag"-Block und prüft ihn."""
    scope = cfg.get("retrieval", {}).get("scope", POOLED)
    if scope not in ERLAUBTE_SCOPES:
        raise ValueError(
            f"rag.retrieval.scope = {scope!r} ist unbekannt. "
            f"Erlaubt: {', '.join(ERLAUBTE_SCOPES)}."
        )
    return scope


def domain_feld(dataset: str, cfg: dict[str, Any] | None = None) -> str:
    """meta-Feld, das die Domäne dieses Datensatzes trägt."""
    if cfg:
        gesetzt = cfg.get("datasets", {}).get(dataset, {}).get("domain_field")
        if gesetzt:
            return gesetzt
    return STANDARD_DOMAIN_FELDER.get(dataset, "domain")


def domain_von(meta: dict[str, Any], dataset: str, cfg: dict | None = None) -> str:
    """Domäne eines Samples, "unknown" wenn das Feld fehlt.

    Samples ohne Domäne landen zusammen in einer Collection "unknown", der Lauf bricht nicht ab.
    """
    return str(meta.get(domain_feld(dataset, cfg)) or "unknown")


def collection_name(dataset: str, scope: str, domain: str | None = None) -> str:
    """Name der Chroma-Collection.

    pooled     -> "crag"
    per_domain -> "crag__finance"
    """
    if scope == POOLED:
        return dataset
    if domain is None:
        raise ValueError("per_domain braucht eine Domäne.")
    # Chroma erlaubt außer - und _ keine Sonderzeichen.
    sauber = "".join(c if c.isalnum() or c in "-_" else "_" for c in domain)
    return f"{dataset}__{sauber}"


def ist_importiert(dataset: str, cfg: dict[str, Any] | None = None) -> bool:
    """True für Datensätze, die fertige Tripel mitbringen.

    Das sind depvalidation und ragbench_own. Sie laufen nicht durch Indexierung und Generierung. Weil es kein Retrieval gab, stehen Zuschnitt und top_k nicht in ihren Dateinamen.
    """
    if not cfg:
        return dataset == "depvalidation"
    return cfg.get("datasets", {}).get(dataset, {}).get("source") == "local"


def generated_datei(
    dataset: str,
    scope: str,
    top_k: int,
    processed_dir: str,
    cfg: dict[str, Any] | None = None,
) -> str:
    """Pfad der Antwortdatei, mit Zuschnitt und top_k im Namen.

        crag_generated_per_domain_k10.jsonl
    Importierte Datensätze (ragbench_own, depvalidation) haben weder Zuschnitt noch Tiefe, ihre Dateien heißen nur {dataset}_generated.jsonl.
    """
    if ist_importiert(dataset, cfg):
        # Kein Retrieval, also weder Zuschnitt noch top_k.
        return f"{processed_dir}/{dataset}_generated.jsonl"
    return f"{processed_dir}/{dataset}_generated_{scope}_k{top_k}.jsonl"


def top_k_aus_config(cfg: dict[str, Any]) -> int:
    """top_k aus dem "rag"-Block."""
    return int(cfg["retrieval"]["top_k"])


def _sauber(text: str) -> str:
    """Ersetzt "/" und "." in Modellnamen, damit sie in Dateinamen passen."""
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text)


def scores_datei(
    dataset: str,
    framework: str,
    scope: str,
    top_k: int,
    judge: str,
    results_dir: str,
    cfg: dict[str, Any] | None = None,
) -> str:
    """Pfad einer Score-Datei, mit Zuschnitt, top_k und Judge im Namen.

    results/scores/crag_deepeval_per_domain_k10_gpt-4o-mini.jsonl
    """
    if ist_importiert(dataset, cfg):
        return f"{results_dir}/scores/{dataset}_{framework}_{_sauber(judge)}.jsonl"
    return (
        f"{results_dir}/scores/{dataset}_{framework}_{scope}"
        f"_k{top_k}_{_sauber(judge)}.jsonl"
    )


# Die Frameworks einzeln aufgezählt, weil auch Datensatznamen einen Unterstrich haben können (ragbench_own).
_FRAMEWORKS = "ragas|trulens|deepeval"

# Mit Retrieval: {ds}_{fw}_{scope}_k{k}_{judge}
# Importiert:    {ds}_{fw}_{judge}
_SCORE_MUSTER = re.compile(
    rf"^(?P<ds>[a-z0-9_]+?)_(?P<fw>{_FRAMEWORKS})_"
    rf"(?:(?P<scope>{'|'.join(ERLAUBTE_SCOPES)})_k(?P<k>\d+)_)?(?P<judge>.+)$"
)


def score_datei_teile(pfad) -> dict[str, Any] | None:
    """Zerlegt den Namen einer Score-Datei in Datensatz, Framework, Zuschnitt, Tiefe.

    Bei importierten Datensätzen (ragbench_own, depvalidation) sind scope und top_k None. Folgt der Name nicht dem Schema, kommt None zurück.
    """
    from pathlib import Path

    m = _SCORE_MUSTER.match(Path(pfad).stem)
    if not m:
        return None
    return {
        "dataset": m["ds"],
        "framework": m["fw"],
        "scope": m["scope"],
        "top_k": int(m["k"]) if m["k"] else None,
        "judge": m["judge"],
    }


def generated_zu_scores(
    pfad, processed_dir: str, cfg: dict[str, Any] | None = None
) -> str:
    """Antwortdatei, aus der eine Score-Datei entstanden ist.

    Die Ausschlussregel für leere Antworten und Abstinenzen prüft den Antworttext, nicht die Scores, und braucht deshalb diese Datei.
    """
    teile = score_datei_teile(pfad)
    if teile is None:
        raise ValueError(f"{pfad}: Dateiname folgt nicht dem Score-Schema.")
    return generated_datei(
        teile["dataset"], teile["scope"] or "", teile["top_k"] or 0, processed_dir, cfg
    )


def waehle_score_dateien(
    scores_dir,
    datasets: list[str],
    top_k: int | None = None,
    scope: str | None = None,
    judge: str | None = None,
) -> list:
    """Wählt die Score-Dateien für RQ2 und RQ3, getrennt nach Zuschnitt und Tiefe.

    Jeder Koeffizient wird je Datensatz und Tiefe berechnet, k5 und k10 dürfen sich nicht mischen.
    Ohne top_k bzw. scope wird nur gewählt, wenn es genau einen Wert gibt, sonst bricht die Funktion mit einer Liste ab.
    Der Wert aus config.yaml wird nicht genommen, weil er zum letzten Lauf gehört und nicht unbedingt zu dem, der ausgewertet werden soll.
    """
    from pathlib import Path

    gefunden = []
    for p in sorted(Path(scores_dir).glob("*.jsonl")):
        t = score_datei_teile(p)
        if t and judge is not None and t["judge"] != _sauber(judge):
            continue
        if t and t["dataset"] in datasets:
            gefunden.append((p, t["dataset"], t["scope"], t["top_k"]))

    auswahl = []
    for ds in datasets:
        eigene = [(p, s, k) for p, d, s, k in gefunden if d == ds]
        if eigene and all(k is None for _, _, k in eigene):
            auswahl += [p for p, _, _ in eigene]
            continue
        eigene = [e for e in eigene if e[2] is not None]
        for name, wert, index in (("Zuschnitt", scope, 1), ("Tiefe", top_k, 2)):
            werte = sorted({e[index] for e in eigene})
            if wert is None and len(werte) > 1:
                schalter = "--scope" if index == 1 else "--top-k"
                raise SystemExit(
                    f"[{ds}] Score-Dateien fuer mehrere Werte von {name} gefunden: {werte}. "
                    f"Bitte mit {schalter} festlegen — Tiefen und Zuschnitte werden nie gemischt."
                )
            if wert is not None:
                eigene = [e for e in eigene if e[index] == wert]
        auswahl += [p for p, _, _ in eigene]
    return auswahl
