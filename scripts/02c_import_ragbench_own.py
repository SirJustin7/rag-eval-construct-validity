"""Schritt 2c: Tripel aus RAGBench importieren (TRACe als viertes Framework):

Ersetzt für diesen Datensatz die Schritte 01 und 02.
Bewertet wird die Antwort aus RAGBench (response) mit den Dokumenten aus RAGBench, denn darauf beziehen sich die TRACe-Labels.
Der Hauptlauf bewertet dagegen die Antwort des eigenen Generators.

Aufruf:
    python scripts/02c_import_ragbench_own.py [--sample-size N]

Danach:
    python scripts/03_run_frameworks.py --dataset ragbench_own

Das Skript ruft kein LLM auf, es liest nur Dateien.
"""

import argparse
from collections import Counter
from pathlib import Path

from rag_eval import naming
from rag_eval.config import load_config
from rag_eval.data.io import read_jsonl, write_jsonl
from rag_eval.data.loaders.ragbench_own import baue_eigene_tripel
from rag_eval.data.schema import GeneratedSample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Gesamtzahl der Tripel. Ohne Angabe: der Wert aus config.yaml → "
        "datasets.ragbench.sample_size, damit dieselbe Fragenauswahl "
        "entsteht wie im Hauptlauf.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Zieldatei. Standard: data/processed/ragbench_own_generated.jsonl",
    )
    parser.add_argument(
        "--ohne-abgleich",
        action="store_true",
        help="Nicht auf die Fragen des Hauptlaufs beschraenken. Standard ist der "
        "Abgleich: Die Antwortdatei des Hauptlaufs entstand VOR der "
        "Dublettenkorrektur vom 25.08.2026 und enthaelt daher 317 statt 333 "
        "Fragen. Ohne Beschraenkung wuerden 16 Fragen bewertet, zu denen es "
        "keine Vergleichswerte gibt — bezahlt, aber unpaarbar.",
    )
    args = parser.parse_args()

    cfg = load_config()
    ds_cfg = cfg["datasets"].get("ragbench", {})
    # Aus dem ragbench-Block, damit Domänen, Stichprobengröße und Seed zum Hauptlauf passen.
    subsets = ds_cfg.get("subsets")
    sample_size = args.sample_size or ds_cfg.get("sample_size")
    seed = cfg.get("seed", 42)

    print(
        f"RAGBench-Tripel laden: subsets={subsets}, sample_size={sample_size}, seed={seed}"
    )
    tripel = baue_eigene_tripel(subsets=subsets, sample_size=sample_size, seed=seed)

    # Auf die Fragen des Hauptlaufs beschränken, weil später Frage für Frage verglichen wird.
    haupt = Path(
        naming.generated_datei(
            "ragbench",
            naming.scope_aus_config(cfg["rag"]),
            naming.top_k_aus_config(cfg["rag"]),
            cfg["paths"]["processed"],
            cfg,
        )
    )
    haupt_ids: set[str] = set()
    if haupt.exists():
        haupt_ids = {
            g.sample_id.replace("ragbench:", "ragbench_own:", 1)
            for g in read_jsonl(str(haupt), GeneratedSample)
        }
    if haupt_ids and not args.ohne_abgleich:
        vorher = len(tripel)
        tripel = [t for t in tripel if t.sample_id in haupt_ids]
        fehlend = haupt_ids - {t.sample_id for t in tripel}
        print(
            f"Abgleich mit {haupt.name}: {vorher} → {len(tripel)} Tripel "
            f"(nur Fragen, die auch der Hauptlauf bewertet hat)."
        )
        if fehlend:
            print(
                f"⚠️  {len(fehlend)} Frage(n) des Hauptlaufs fehlen hier: "
                f"{sorted(fehlend)[:5]}"
            )

    leer = [t.sample_id for t in tripel if not t.generated_answer.strip()]
    ohne_kontext = [t.sample_id for t in tripel if not t.retrieved_contexts]
    verteilung = Counter(t.meta.get("subset") for t in tripel)

    ziel = args.out or naming.generated_datei(
        "ragbench_own",
        naming.scope_aus_config(cfg["rag"]),
        naming.top_k_aus_config(cfg["rag"]),
        cfg["paths"]["processed"],
        cfg,
    )
    write_jsonl(tripel, ziel)

    print(f"{len(tripel)} Tripel geschrieben → {ziel}")
    print(f"Domaenen: {dict(verteilung)}")
    if leer:
        print(f"⚠️  {len(leer)} Antwort(en) leer: {leer[:5]}")
    if ohne_kontext:
        print(f"⚠️  {len(ohne_kontext)} Tripel ohne Kontext: {ohne_kontext[:5]}")

    # Gegenprobe: dieselben Fragen wie im Hauptlauf?
    if haupt_ids:
        eigene_ids = {t.sample_id for t in tripel}
        nur_haupt = haupt_ids - eigene_ids
        nur_hier = eigene_ids - haupt_ids
        if nur_haupt or nur_hier:
            print(
                f"⚠️  Fragenmengen weichen ab: {len(nur_haupt)} nur im Hauptlauf, "
                f"{len(nur_hier)} nur hier."
            )
        else:
            print(
                f"Gegenprobe ok: dieselben {len(eigene_ids)} Fragen wie {haupt.name}."
            )
    else:
        print(f"Hinweis: {haupt.name} fehlt, Gegenprobe uebersprungen.")


if __name__ == "__main__":
    main()
