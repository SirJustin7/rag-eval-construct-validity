"""Schritt 7: Manipulierte Antworten erzeugen (RQ3):

Erzeugt die manipulierten Antwortdateien, bewertet aber nichts.
Bewertet wird danach mit scripts/03 und --dataset <name>, ohne --out-dir.
Die Scores landen dann als results/scores/<ds>_<framework>_<judge>.jsonl, wo scripts/12_manipulation_auswerten.py sie sucht.
scripts/04 und 06 lesen sie nicht mit.

Alle Bedingungen verändern nur den Text der vorhandenen Antwort (inject_1 bis inject_3, replace_full, swap_fact).
Der Kontext bleibt gleich, es wird nichts neu generiert.

Beispiel:

    # alle Bedingungen für CRAG, je 100 Samples
    python scripts/07_manipulate.py --dataset crag --sample-size 100

    # danach bewerten (kostet Geld), ohne --out-dir:
    python scripts/03_run_frameworks.py --dataset crag_inject_1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_eval import manipulation as mp
from rag_eval import naming
from rag_eval.config import load_config
from rag_eval.data.io import read_jsonl, write_jsonl
from rag_eval.data.schema import GeneratedSample

GENERIERUNG = ["inject_1", "inject_2", "inject_3", "replace_full", "swap_fact"]


def _lade_generiert(ds: str, cfg: dict) -> list[GeneratedSample]:
    scope = naming.scope_aus_config(cfg["rag"])
    top_k = naming.top_k_aus_config(cfg["rag"])
    pfad = naming.generated_datei(ds, scope, top_k, cfg["paths"]["processed"], cfg)
    return read_jsonl(pfad, GeneratedSample)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["crag", "ragbench"])
    p.add_argument("--bedingungen", nargs="+", default=GENERIERUNG, choices=GENERIERUNG)
    p.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Wie viele Samples je Bedingung (Standard 100).",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = load_config()
    ds = args.dataset
    proc = Path(cfg["paths"]["processed"])

    from rag_eval.data.io import entdoppele

    basis = entdoppele(_lade_generiert(ds, cfg), f"[{ds}] generierte Samples")
    basis = basis[: args.sample_size]
    print(f"[{ds}] Basis: {len(basis)} Samples\n")

    bauer = {
        "inject_1": lambda: mp.inject_claims(basis, 1, args.seed),
        "inject_2": lambda: mp.inject_claims(basis, 2, args.seed),
        "inject_3": lambda: mp.inject_claims(basis, 3, args.seed),
        "replace_full": lambda: mp.replace_answer(basis, args.seed),
        "swap_fact": lambda: mp.swap_fact(basis, args.seed),
    }

    for bed in args.bedingungen:
        manipuliert = bauer[bed]()
        if not manipuliert:
            print(f"[{bed}] nichts erzeugt — uebersprungen.")
            continue
        ziel = proc / f"{ds}_{bed}_generated.jsonl"
        write_jsonl(manipuliert, ziel)
        print(f"[{bed}] {len(manipuliert)} Samples → {ziel}")
        for konstrukt, erw in mp.ERWARTUNG[bed].items():
            print(f"      erwartet · {konstrukt:18s} {erw}")
        print(
            f"      bewerten: python scripts/03_run_frameworks.py --dataset {ds}_{bed}"
        )
        print()


if __name__ == "__main__":
    main()
