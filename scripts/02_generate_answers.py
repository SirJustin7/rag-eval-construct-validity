"""Schritt 2: Antworten mit dem RAG-System erzeugen:

Erzeugt für jeden Datensatz die Tripel (query, retrieved_contexts, generated_answer), die danach alle Frameworks bewerten.

Aufruf: python scripts/02_generate_answers.py [--dataset ...] [--sample-size N]

Vor dem ersten Aufruf wird geprüft, ob der Endpunkt das Generator-Modell anbietet.
Sonst scheitert erst das erste Sample, mit einer unklaren Meldung.

Checkpointing: Der Lauf schreibt nach je --chunk-size Samples in die Zieldatei und überspringt beim Neustart, was dort schon steht.
Mit --restart beginnt er von vorn.

Jede Zeile enthält in meta["rag_system"] Generator-Modell, temperature, max_tokens, top_k, Embedding-Modell und Chunk-Parameter, damit sich jede Antwort ihrem Lauf zuordnen lässt.
"""

import argparse
from pathlib import Path

from rag_eval import naming
from rag_eval.config import load_config
from rag_eval.data.io import read_jsonl
from rag_eval.data.loaders import LOADERS
from rag_eval.data.schema import GeneratedSample, RAGSample
from rag_eval.pipeline import RAGSystem, generation

# Nach wie vielen Samples zwischengespeichert wird.
DEFAULT_CHUNK = 25


def _bereits_erzeugt(pfad: Path) -> set[str]:
    """sample_ids, für die schon eine Antwort in pfad steht."""
    if not pfad.exists():
        return set()
    return {g.sample_id for g in read_jsonl(str(pfad), GeneratedSample)}


def _anhaengen(neue: list[GeneratedSample], pfad: Path) -> None:
    """Schreibt an eine bestehende JSONL an (write_jsonl überschreibt)."""
    pfad.parent.mkdir(parents=True, exist_ok=True)
    with open(pfad, "a", encoding="utf-8") as f:
        f.writelines(g.model_dump_json() + "\n" for g in neue)


def _verarbeite(name: str, cfg: dict, rag: RAGSystem, args) -> None:
    """Erzeugt die Antworten für einen Datensatz, mit Checkpointing."""
    scope = naming.scope_aus_config(cfg["rag"])
    top_k = naming.top_k_aus_config(cfg["rag"])
    quelle = Path(f"{cfg['paths']['processed']}/{name}_samples.jsonl")
    ziel = Path(
        naming.generated_datei(name, scope, top_k, cfg["paths"]["processed"], cfg)
    )
    if not quelle.exists():
        print(
            f"[{name}] {quelle} fehlt — erst scripts/01_build_index.py laufen "
            f"lassen. Übersprungen."
        )
        return

    alle = read_jsonl(str(quelle), RAGSample)
    if args.sample_size is not None:
        alle = alle[: args.sample_size]
        print(f"[{name}] Probelauf: nur {len(alle)} Samples.")
    print(f"[{name}] Zuschnitt: {scope}, top_k={top_k} → {ziel.name}")

    if args.restart and ziel.exists():
        ziel.unlink()
        print(f"[{name}] --restart: {ziel.name} gelöscht.")

    fertig = _bereits_erzeugt(ziel)
    offen = [s for s in alle if s.sample_id not in fertig]
    if fertig:
        print(f"[{name}] {len(fertig)} Antworten liegen vor, {len(offen)} offen.")
    if not offen:
        print(f"[{name}] nichts zu tun.")
        return

    geschrieben = len(fertig)
    for i in range(0, len(offen), args.chunk_size):
        block = offen[i : i + args.chunk_size]
        if scope == naming.POOLED:
            erzeugt = rag.run(block, collection_name=name)
        else:
            erzeugt = rag.run(
                block,
                collection_for=lambda smp: naming.collection_name(
                    name, scope, naming.domain_von(smp.meta, name, cfg)
                ),
            )
        _anhaengen(erzeugt, ziel)
        geschrieben += len(erzeugt)
        print(
            f"[{name}] {min(i + len(block), len(offen))}/{len(offen)} "
            f"· insgesamt {geschrieben} Antworten → {ziel.name}"
        )

    print(f"[{name}] fertig: {geschrieben} Antworten in {ziel}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(LOADERS), default=None)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Nur die ersten N Samples beantworten (z.B. 3 für einen Schnelltest).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK,
        help=f"Nach je N Samples zwischenspeichern (Standard {DEFAULT_CHUNK}).",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Vorhandene Antworten ignorieren und von vorn beginnen.",
    )
    parser.add_argument(
        "--skip-precheck",
        action="store_true",
        help="Vorabprüfung des Generator-Modells überspringen.",
    )
    args = parser.parse_args()

    cfg = load_config()

    if not args.skip_precheck:
        generation.pruefe_modell_verfuegbar(cfg["rag"])

    rag = RAGSystem(cfg)

    # Ohne --dataset die Datensätze aus der config, nicht alle aus LOADERS.
    datasets = [args.dataset] if args.dataset else list(cfg["datasets"])

    for name in datasets:
        _verarbeite(name, cfg, rag, args)


if __name__ == "__main__":
    main()
