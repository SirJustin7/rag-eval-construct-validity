"""Datensatz zur Validierung von Konfigurationsabhängigkeiten (Simon et al. 2024):

Replikationspaket zu "A Methodology for Evaluating RAG Systems: A Case Study On Configuration Dependency Validation".
500 Abhängigkeiten aus zehn GitHub-Projekten, von den Autoren von Hand als echt oder falsch gelabelt. Die Labels stammen also nicht von einem LLM.

Das Repository enthält die fertigen Tripel (Frage, abgerufener Kontext, Antwort) aus den Läufen des Papers.
Der Loader liefert deshalb direkt GeneratedSample, Indexierung und Generierung entfallen.

Aufbau des Repositories:
    data/results/all_dependencies.csv                               Labels: 500 Zeilen, Spalte final_rating (TRUE/FALSE) und explanation (Begründung)
    data/results/<variante>/all_dependencies_*_<modell>.json        je 500 Einträge mit task_str, context, reasoning, correct

config1 bis config4 und without sind die RAG-Varianten des Papers. Das Holdout-Set (50 Einträge) wird nicht geladen, weil reasoning und correct dort fehlen.

Die Antwort ist ein Urteil mit Begründung, keine offene Antwort auf eine Frage.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from ..schema import GeneratedSample

REPO_URL = "https://github.com/simisimon/dependency-validation-rag"

# Relativ zum Wurzelverzeichnis des Repositories.
GROUND_TRUTH_CSV = "data/results/all_dependencies.csv"
RESULTS_DIR = "data/results"

# Zwei RAG-Varianten mal zwei Generatoren, vor dem ersten Lauf festgelegt.
# Mit gpt-4o allein wären fast alle Urteile richtig und die Scores hätten kaum Streuung.
# llama3:8b bringt auch falsche Urteile in die Stichprobe.
# config1 und config4 sind Ausgangsvarianten aus Simon et al. (Tab. 1) mit gleichem Reranker, aber anderem Embedding-Modell und 5 bzw. 3 Chunks.
DEFAULT_VARIANTS = ["config1", "config4"]
DEFAULT_MODELS = ["gpt-4o-2024-05-13", "llama3:8b"]

# Das Urteil steht in response_dict["isDependency"], die Begründung in reasoning.
# Die Frameworks bewerten beides zusammen als einen Text, sonst fehlt der Begründung die Aussage, die sie begründet.
_VERDICT = {
    True: "Yes, the two configuration options depend on each other.",
    False: "No, the two configuration options do not depend on each other.",
}


def _lade_ground_truth(repo: Path) -> dict[int, dict]:
    """Liest all_dependencies.csv, Schlüssel ist die Spalte index.

    Rückgabe: index -> {label, explanation, category, sub_category, project}.
    """
    pfad = repo / GROUND_TRUTH_CSV
    if not pfad.exists():
        raise FileNotFoundError(
            f"Ground-Truth-Datei nicht gefunden: {pfad}\n"
            f"Erwartet wird eine Kopie von {REPO_URL} (Ordner data/ vollständig)."
        )

    gt: dict[int, dict] = {}
    with open(pfad, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # final_rating ist das Label nach dem Review durch mehrere Autoren, rating der erste Durchgang. Verwendet wird final_rating.
            gt[int(row["index"])] = {
                "label": row["final_rating"].strip().upper() == "TRUE",
                "explanation": row.get("explanation") or None,
                "category": row.get("final_category") or None,
                "sub_category": row.get("sub_category") or None,
                "project": row.get("project") or None,
            }
    return gt


def _ergebnisdatei(repo: Path, variante: str, modell: str) -> Path | None:
    """Sucht die Ergebnisdatei zu Variante und Modell.

    Zwischen Präfix und Modell steht ein Index-Name, der je Variante anders ist (z.B. all_dependencies_all_gpt-4o-2024-05-13.json), deshalb glob.
    Das Ende _<modell>.json schließt die _advanced-Läufe in config2 aus, die andere Prompts verwenden.
    """
    treffer = sorted(
        (repo / RESULTS_DIR / variante).glob(f"all_dependencies_*_{modell}.json")
    )
    return treffer[0] if treffer else None


def _zu_sample(
    eintrag: dict, gt: dict, variante: str, modell: str
) -> GeneratedSample | None:
    """Macht aus einem Ergebniseintrag ein GeneratedSample, None wenn unbrauchbar.

    Ausschlussregel, vor dem Lauf festgelegt: Einträge, deren response_dict ein String ist, fallen weg.
    Dort war die JSON-Ausgabe des Generators fehlerhaft, und auch die Pipeline des Papers hat reasoning und correct leer gelassen.
    Betroffen ist fast nur llama3:8b.
    Repariert wird nichts, sonst entstünde eine Auswahl, die nicht mehr der des Papers entspricht.
    """
    response_dict = eintrag.get("response_dict")
    if not isinstance(response_dict, dict):
        return None

    rationale = eintrag.get("reasoning")
    urteil = response_dict.get("isDependency")
    kontexte = [
        c["content"] for c in (eintrag.get("context") or []) if c.get("content")
    ]

    # Ohne Antwort oder Kontext ist das Tripel unvollständig.
    if not rationale or urteil is None or not kontexte:
        return None

    antwort = f"{_VERDICT[bool(urteil)]} {rationale}".strip()

    return GeneratedSample(
        sample_id=f"depvalidation:{variante}:{modell}:{eintrag['index']}",
        dataset="depvalidation",
        query=eintrag["task_str"],
        retrieved_contexts=kontexte,
        generated_answer=antwort,
        # Die Begründung der Autoren zum Label. Keine Musterantwort, aber der einzige von Menschen geschriebene Referenztext im Datensatz.
        reference_answer=gt.get("explanation"),
        meta={
            "variant": variante,
            "generator": modell,
            "ground_truth": gt.get("label"),
            "predicted": bool(urteil),
            # Urteil richtig oder falsch. Danach werden die Scores in RQ4 getrennt.
            "answer_correct": gt.get("label") == bool(urteil),
            # TP/TN/FP/FN aus dem Repository, zur Kontrolle von answer_correct.
            "correct_class": eintrag.get("correct"),
            "uncertainty": eintrag.get("uncertainty"),
            "category": gt.get("category"),
            "sub_category": gt.get("sub_category"),
            "project": gt.get("project"),
            "human_explanation": gt.get("explanation"),
        },
    )


def load_depvalidation(
    repo_path: str,
    variants: list[str] | None = None,
    models: list[str] | None = None,
    sample_size: int | None = None,
    seed: int = 42,
    balance_correct: bool = False,
    stats: dict | None = None,
) -> list[GeneratedSample]:
    """Lädt die fertigen Tripel aus dem Replikationspaket.

    Args:
        repo_path:       Wurzelverzeichnis der Repository-Kopie.
        variants:        RAG-Varianten (config1 bis config4, without). sNone = DEFAULT_VARIANTS.
        models:          Generatoren. None = DEFAULT_MODELS.
        sample_size:     Gesamtzahl über alle Kombinationen. None = alles.
        seed:            Legt die Auswahl fest.
        balance_correct: Bei True je Kombination gleich viele richtige und falsche Urteile. Die Fehlerquote entspricht dann nicht mehr dem Original.
        stats:           Dict, in das je Variante und Modell geschrieben wird, wie viele Einträge gelesen, verworfen und brauchbar sind.

    Returns:
        GeneratedSamples, gleich viele je Variante und Modell.
    """
    repo = Path(repo_path)
    variants = variants or DEFAULT_VARIANTS
    models = models or DEFAULT_MODELS
    gt = _lade_ground_truth(repo)
    rng = random.Random(seed)

    # Gleich viele je Kombination, damit keine Variante und kein Generator überwiegt.
    kombinationen = [(v, m) for v in variants for m in models]
    pro_kombi = (
        None if sample_size is None else max(1, sample_size // len(kombinationen))
    )

    samples: list[GeneratedSample] = []
    for variante, modell in kombinationen:
        datei = _ergebnisdatei(repo, variante, modell)
        if datei is None:
            raise FileNotFoundError(
                f"Keine Ergebnisdatei für Variante '{variante}' und Modell "
                f"'{modell}' unter {repo / RESULTS_DIR / variante}. "
                f"Verfügbare Varianten: {sorted(p.name for p in (repo / RESULTS_DIR).iterdir() if p.is_dir())}"
            )

        with open(datei, encoding="utf-8") as f:
            eintraege = json.load(f)

        gezogen = [
            s
            for s in (
                _zu_sample(e, gt.get(e["index"], {}), variante, modell)
                for e in eintraege
            )
            if s is not None
        ]

        if stats is not None:
            stats[f"{variante}/{modell}"] = {
                "gelesen": len(eintraege),
                "verworfen": len(eintraege) - len(gezogen),
                "brauchbar": len(gezogen),
            }

        if pro_kombi is not None and len(gezogen) > pro_kombi:
            if balance_correct:
                gezogen = _geschichtet(gezogen, pro_kombi, rng)
            else:
                gezogen = rng.sample(gezogen, pro_kombi)

        samples.extend(gezogen)

    # Geht die Aufteilung nicht genau auf, zufällig auf sample_size kürzen.
    if sample_size is not None and len(samples) > sample_size:
        samples = rng.sample(samples, sample_size)

    # Nach ID sortiert, damit die Reihenfolge nicht von der Ziehung abhängt.
    samples.sort(key=lambda s: s.sample_id)
    return samples


def _geschichtet(
    samples: list[GeneratedSample], n: int, rng: random.Random
) -> list[GeneratedSample]:
    """Zieht je zur Hälfte richtige und falsche Urteile, soweit vorhanden."""
    gruppen: dict[bool, list[GeneratedSample]] = defaultdict(list)
    for s in samples:
        gruppen[bool(s.meta.get("answer_correct"))].append(s)

    haelfte = n // 2
    gezogen: list[GeneratedSample] = []
    for schluessel in (True, False):
        pool = gruppen[schluessel]
        gezogen.extend(rng.sample(pool, min(haelfte, len(pool))))

    # Ist eine Gruppe zu klein, aus dem Rest auffüllen.
    if len(gezogen) < n:
        rest = [s for s in samples if s not in gezogen]
        gezogen.extend(rng.sample(rest, min(n - len(gezogen), len(rest))))
    return gezogen


def nicht_ueber_pipeline(**_kwargs) -> None:
    """Platzhalter in LOADERS (siehe loaders/__init__.py).

    Eigene Antworten würden die des Papers ersetzen, und die Labels der Autoren passten nicht mehr dazu.
    """
    raise NotImplementedError(
        "depvalidation liefert bereits fertige (Frage, Kontext, Antwort)-Tripel "
        "und wird nicht indexiert oder neu generiert.\n"
        "Stattdessen:  python scripts/02b_import_depvalidation.py\n"
        "Danach wie gewohnt: python scripts/03_run_frameworks.py --dataset depvalidation"
    )
