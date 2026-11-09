"""RQ2: Konvergenz zwischen den Frameworks:

Korreliert je Konstrukt die Scores der Frameworks paarweise (Spearman ρ, Kendall τ).
Eine niedrige Korrelation heißt, dass gleich benannte Metriken nicht dasselbe messen.

Aufruf: python scripts/04_rq2_convergence.py [--datasets crag ragbench ...]
"""

import argparse
from pathlib import Path

import pandas as pd

from rag_eval.analysis.correlations import (
    convergence_report,
    saturation_report,
    write_convergence_report,
)
from rag_eval.analysis.saturation import ausgeschlossene, formatiere
from rag_eval.analysis.stichprobe import VARIANTEN, filtere, lade_datensatzlaeufe
from rag_eval.config import load_config
from rag_eval.naming import ist_importiert, waehle_score_dateien


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["crag", "ragbench", "ragbench_own"],
        help="RQ2 läuft auf CRAG und RAGBench. "
        "ragbench_own = RAGBenchs eigene Tripel mit TRACe als "
        "vierter Methode; hat keine Tiefe und wird bei jeder "
        "Tiefe identisch neu geschrieben.",
    )
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="Zusaetzlich eine gemeinsame Auswertung ueber die Retrieval-Datensaetze "
        "(nie mit ragbench_own, das andere Antworten bewertet). "
        "Standard ist getrennt: Die Score-Verteilungen unterscheiden sich "
        "je Datensatz (z.B. ragas-trulens/context_relevance 0.68 auf CRAG "
        "gegen 0.26 auf RAGBench), und ein gepoolter Wert beschreibt dann "
        "keinen von beiden.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Retrieval-Tiefe der auszuwertenden Laeufe (5 oder 10). "
        "Pflicht, sobald mehrere Tiefen bewertet sind.",
    )
    parser.add_argument(
        "--scope",
        default=None,
        help="Index-Zuschnitt (per_domain). Pflicht bei mehreren.",
    )
    parser.add_argument(
        "--abstinenz",
        choices=sorted(VARIANTEN),
        default="beide",
        help="Kapitel 4.2.5: jeder Koeffizient mit UND ohne Abstinenzen. "
        "Leere Antworten fallen in jeder Variante heraus.",
    )
    args = parser.parse_args()

    cfg = load_config()
    scores_dir = Path(cfg["paths"]["results"]) / "scores"
    out_dir = Path(cfg["paths"]["results"]) / "rq2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Nur Dateien einer Tiefe, eines Zuschnitts und des Judges aus config.yaml. Die Judge-Wiederholungen liegen im selben Ordner.
    pfade = waehle_score_dateien(
        scores_dir, args.datasets, args.top_k, args.scope, judge=cfg["judge"]["model"]
    )
    for path in pfade:
        print(f"  · {path.name}")

    # Je Datensatz: doppelte Scores entfernen, Antworttypen aus der Antwortdatei, bei ragbench_own die TRACe-Labels dazu.
    datensaetze = lade_datensatzlaeufe(
        pfade, args.datasets, cfg["paths"]["processed"], cfg
    )

    for variante in VARIANTEN[args.abstinenz]:
        # Getrennt je Datensatz. Gepoolt könnten Korrelationen entstehen oder verschwinden, die es in keinem einzelnen Datensatz gibt.
        laeufe = []
        for d in datensaetze:
            teil, bericht = filtere(d.results, d.typen, variante)
            laeufe.append((d.name, teil, bericht, d.binaere_metriken))
        if args.pooled:
            retrieval = [
                lauf
                for lauf, d in zip(laeufe, datensaetze)
                if not ist_importiert(d.dataset, cfg)
            ]
            if retrieval:
                name = "gepoolt" + (f"_k{args.top_k}" if args.top_k else "")
                laeufe.append(
                    (
                        name,
                        [r for lauf in retrieval for r in lauf[1]],
                        None,
                        frozenset(),
                    )
                )

        for name, teilmenge, bericht, binaer in laeufe:
            titel = f"{name} / {variante}"
            if not teilmenge:
                print(f"\n[{titel}] keine Scores gefunden — uebersprungen.")
                continue
            print(
                f"\n{'=' * 62}\n=== {titel.upper()} — {len(teilmenge)} Scores\n{'=' * 62}"
            )
            if bericht:
                print(f"Stichprobe {bericht.zeile()}")

            # Sättigung vor den Korrelationen. Was hier ausgeschlossen wird, zählt unten nicht.
            # Berechnet auf derselben Stichprobe wie die Korrelationen.
            print("\n--- Saettigung je Konstrukt (vorregistrierte Regel) ---")
            for construct, tabelle in saturation_report(teilmenge).items():
                print(f"\n{construct}:")
                print(formatiere(tabelle))
                raus = ausgeschlossene(tabelle)
                if raus:
                    print(
                        f"  → ausgeschlossen von Validitaetsaussagen: {', '.join(raus)}"
                    )

            report = convergence_report(teilmenge, binaer)
            if bericht:
                for tabelle in report.values():
                    tabelle["variante"] = variante
                    tabelle["n_leer_entfernt"] = bericht.n_leer
                    tabelle["n_abstinenz_entfernt"] = bericht.n_abstinenz_entfernt
            csv_path, xlsx_path = write_convergence_report(
                report, out_dir, stem=f"convergence_{name}_{variante}"
            )

            for construct, table in report.items():
                print(f"\n--- {construct} ---")
                spalten = [
                    "framework_a",
                    "framework_b",
                    "n",
                    "koeffizient",
                    "spearman_rho",
                    "kendall_tau",
                    "r_rb",
                    "r_rb_p",
                    "paar_stufe",
                    "belastbar",
                ]
                with pd.option_context("display.width", 200):
                    print(
                        table[spalten].to_string(
                            index=False, float_format=lambda v: f"{v:.3f}"
                        )
                    )
                unbelastbar = table[~table["belastbar"]]
                if not unbelastbar.empty:
                    print(
                        f"  ⚠  {len(unbelastbar)} von {len(table)} Paaren beruhen auf einer "
                        f"gesaettigten Metrik und tragen keine Konvergenzaussage."
                    )
            print(f"\nGeschrieben: {csv_path.name} · {xlsx_path.name}")


if __name__ == "__main__":
    main()
