"""Schritt 1: Indexierung:

Lädt die Datensätze aus der Konfiguration, bildet sie auf RAGSample ab und baut die Vektor-Collections auf.

Aufruf: python scripts/01_build_index.py [--dataset crag|ragbench]
"""

import argparse
from pathlib import Path

from rag_eval import naming
from rag_eval.config import load_config
from rag_eval.data.io import pruefe_eindeutige_ids, read_jsonl, write_jsonl
from rag_eval.data.loaders import LOADERS
from rag_eval.data.schema import RAGSample
from rag_eval.pipeline import RAGSystem


def _loader_kwargs(name: str, cfg: dict, sample_size: int | None) -> dict:
    """Stellt die Parameter für den jeweiligen Loader aus der config zusammen.

    Die Loader nehmen verschiedene Argumente (ragbench subsets, crag domains und datei). sample_size von der Kommandozeile überschreibt die config.
    """
    ds = cfg["datasets"].get(name, {})
    kwargs: dict = {"seed": cfg["seed"]}
    size = sample_size if sample_size is not None else ds.get("sample_size")
    if size is not None:
        kwargs["sample_size"] = size
    if name == "ragbench" and ds.get("subsets"):
        kwargs["subsets"] = ds["subsets"]
    if name == "crag":
        if ds.get("domains"):
            kwargs["domains"] = ds["domains"]
        if ds.get("local_file"):
            kwargs["datei"] = ds["local_file"]
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=list(LOADERS),
        default=None,
        help="Nur einen Datensatz indexieren (Standard: alle aktivierten)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Überschreibt die Stichprobengröße aus der config (z.B. 20 zum Testen)",
    )
    parser.add_argument(
        "--reuse-samples",
        action="store_true",
        help="Bereits geladene <dataset>_samples.jsonl wiederverwenden statt neu "
        "zu laden. Fuer CRAG relevant: das Laden entpackt 2,7 GB und wandelt "
        "HTML um (~28 min). Schlaegt nur die Indexierung fehl, waere ein "
        "erneutes Laden reine Zeitverschwendung.",
    )
    args = parser.parse_args()

    cfg = load_config()
    rag = RAGSystem(cfg)

    # Ohne --dataset die Datensätze aus der config, nicht alle aus LOADERS. Dort stehen auch die importierten Datensätze, die nicht indexiert werden.
    datasets = [args.dataset] if args.dataset else list(cfg["datasets"])
    for name in datasets:
        pfad = f"{cfg['paths']['processed']}/{name}_samples.jsonl"
        if args.reuse_samples and Path(pfad).exists():
            samples = read_jsonl(pfad, RAGSample)
            print(
                f"[{name}] {len(samples)} Samples aus {pfad} wiederverwendet "
                f"(nicht neu geladen)."
            )
        else:
            samples = LOADERS[name](**_loader_kwargs(name, cfg, args.sample_size))
            # Doppelte IDs schon hier abfangen, bevor bezahlte Läufe darauf aufbauen.
            pruefe_eindeutige_ids(samples, f"{name}_samples.jsonl")
            write_jsonl(samples, pfad)
        scope = naming.scope_aus_config(cfg["rag"])
        if scope == naming.POOLED:
            rag.index_dataset(samples, collection_name=name)
            print(f"[{name}] {len(samples)} Samples indexiert (Zuschnitt: pooled).")
        else:
            # Eine Collection je Domäne. Samples ohne Domäne kommen in "unknown".
            gruppen: dict[str, list] = {}
            for smp in samples:
                dom = naming.domain_von(smp.meta, name, cfg)
                gruppen.setdefault(dom, []).append(smp)
            for dom in sorted(gruppen):
                coll = naming.collection_name(name, scope, dom)
                rag.index_dataset(gruppen[dom], collection_name=coll)
                print(f"[{name}/{dom}] {len(gruppen[dom])} Samples → {coll}")
            print(
                f"[{name}] {len(samples)} Samples in {len(gruppen)} Collections "
                f"indexiert (Zuschnitt: per_domain)."
            )


if __name__ == "__main__":
    main()
