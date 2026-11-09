"""Schritt 2b: Datensatz von Simon et al. (2024) importieren:

Ersetzt für diesen Datensatz die Schritte 01 und 02.
Das Replikationspaket enthält die Tripel (Frage, Kontext, Antwort) schon, mit von Hand vergebenen Labels.
Eigene Antworten würden die des Papers ersetzen, und die Labels passten nicht mehr dazu.

Aufruf:
    python scripts/02b_import_depvalidation.py [--sample-size N] [--balance-correct]

Danach:
    python scripts/03_run_frameworks.py --dataset depvalidation

Das Skript ruft kein LLM auf, es liest nur Dateien.
"""

import argparse
from collections import Counter

from rag_eval.config import load_config
from rag_eval.data.io import write_jsonl
from rag_eval.data.loaders.depvalidation import (
    DEFAULT_MODELS,
    DEFAULT_VARIANTS,
    load_depvalidation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-path",
        default=None,
        help="Wurzelverzeichnis der Repository-Kopie. Standard: config.yaml → "
        "datasets.depvalidation.repo_path",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Gesamtzahl der zu importierenden Tripel (gleichmäßig über "
        "Variante × Generator). Ohne Angabe: alles.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help=f"RAG-Varianten. Standard: {' '.join(DEFAULT_VARIANTS)}",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Generatoren. Standard: {' '.join(DEFAULT_MODELS)}",
    )
    parser.add_argument(
        "--balance-correct",
        action="store_true",
        help="Je zur Hälfte korrekte und falsche Antworten ziehen. Erhöht die "
        "Streuung, verändert aber den Anteil falscher Urteile gegenüber dem Original.",
    )
    args = parser.parse_args()

    cfg = load_config()
    ds_cfg = cfg["datasets"].get("depvalidation", {})
    repo_path = args.repo_path or ds_cfg.get("repo_path")
    if not repo_path:
        parser.error(
            "Kein Pfad zur Repository-Kopie. Entweder --repo-path setzen oder "
            "in config.yaml unter datasets.depvalidation.repo_path eintragen."
        )

    stats: dict = {}
    samples = load_depvalidation(
        repo_path=repo_path,
        stats=stats,
        variants=args.variants or ds_cfg.get("variants"),
        models=args.models or ds_cfg.get("models"),
        sample_size=args.sample_size
        if args.sample_size is not None
        else ds_cfg.get("sample_size"),
        seed=cfg["seed"],
        balance_correct=args.balance_correct or ds_cfg.get("balance_correct", False),
    )

    ziel = f"{cfg['paths']['processed']}/depvalidation_generated.jsonl"
    write_jsonl(samples, ziel)

    # Zusammensetzung ausgeben, vor allem den Anteil falscher Urteile.
    falsch = sum(1 for s in samples if s.meta.get("answer_correct") is False)
    print(f"{len(samples)} Tripel → {ziel}")
    print(
        f"  davon mit falschem Urteil: {falsch} ({falsch / max(len(samples), 1):.0%})"
    )
    print(
        f"  Ground Truth 'ist Abhängigkeit': "
        f"{sum(1 for s in samples if s.meta.get('ground_truth'))}"
    )

    je_kombi = Counter((s.meta["variant"], s.meta["generator"]) for s in samples)
    for (variante, modell), n in sorted(je_kombi.items()):
        print(f"  {variante:10s} {modell:24s} {n:5d}")

    kontexte = [len(s.retrieved_contexts) for s in samples]
    if kontexte:
        print(f"  Kontext-Chunks je Sample: min {min(kontexte)}, max {max(kontexte)}")

    # Ausschlüsse je Kombination. Verworfen wird, wo der Generator kein gültiges JSON geliefert hat (Ausschlussregel im Loader).
    print("\nAusschlüsse je Kombination (unparsebare Generator-Ausgabe):")
    for kombi, z in sorted(stats.items()):
        anteil = z["verworfen"] / max(z["gelesen"], 1)
        print(
            f"  {kombi:36s} {z['verworfen']:3d}/{z['gelesen']:3d} verworfen ({anteil:.0%})"
        )


if __name__ == "__main__":
    main()
