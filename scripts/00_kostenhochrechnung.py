"""Kostenhochrechnung vor einem bezahlten Lauf:

Rechnet aus den gemessenen Läufen in results/usage.jsonl hoch, was ein geplanter Lauf kosten würde, und stellt es dem Restbudget gegenüber.
Grundlage sind die gemessenen Tokens.

Aufrufe:
    # 333 Samples, 2 Datensätze, zwei top_k-Werte
    python scripts/00_kostenhochrechnung.py --samples 333 --datasets 2 --top-k 5 10

    # gpt-4o auf einer kleinen Teilmenge
    python scripts/00_kostenhochrechnung.py --model gpt-4o --samples 50 --datasets 1 --top-k 10

    # Mehrere Stichprobengrößen vergleichen
    python scripts/00_kostenhochrechnung.py --samples 201 333 501 --datasets 2 --top-k 5 10
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from rag_eval.config import load_config
from rag_eval.evaluation.usage import (
    bisher_ausgegeben,
    hat_preis,
    hochrechnen,
    messungen_je_framework,
)

# Frameworks, die über die API bewertet werden.
ERWARTETE_FRAMEWORKS = ["ragas", "trulens", "deepeval"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--samples",
        nargs="+",
        type=int,
        default=[333],
        help="Stichprobengröße(n) JE DATENSATZ",
    )
    parser.add_argument(
        "--datasets",
        type=int,
        default=2,
        help="Anzahl Datensätze (crag + ragbench = 2)",
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=None,
        help="Geplante top_k-Werte (Default: der Wert aus config.yaml)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Judge-Modell (Default: judge.model aus config.yaml)",
    )
    parser.add_argument(
        "--kontextanteil",
        type=float,
        default=0.5,
        help="Anteil der Token, der linear mit top_k wächst "
        "(Näherung, siehe usage.top_k_faktor)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=25.0,
        help="Gesamtbudget in EUR (Default: 25, Mitte von 20-30)",
    )
    parser.add_argument(
        "--sicherheit",
        type=float,
        default=0.30,
        help="Aufschlag für Messunsicherheit (Default: 0.30)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Zieldatei für den Bericht (Default: <results>/kosten/hochrechnung.md)",
    )
    parser.add_argument(
        "--kein-bericht",
        action="store_true",
        help="Nur auf dem Terminal ausgeben, nichts schreiben",
    )
    args = parser.parse_args()

    # Alles, was ausgegeben wird, kommt auch in den Bericht.
    zeilen: list[str] = []

    def sag(text: str = "") -> None:
        print(text)
        zeilen.append(text)

    cfg = load_config()
    model = args.model or cfg["judge"]["model"]
    top_k_basis = int(cfg["rag"]["retrieval"]["top_k"])
    top_k_werte = args.top_k or [top_k_basis]
    usage_pfad = Path(cfg["paths"]["results"]) / "usage.jsonl"

    # Ohne Preis wäre jede Summe 0 €, das sähe aus wie "kostet nichts".
    if not hat_preis(model):
        raise SystemExit(
            f"Für '{model}' ist kein Preis hinterlegt — vermutlich steht in "
            "config.yaml gerade das kostenlose SCADS-Modell.\n"
            "Hochrechnen lässt sich nur ein bezahltes Modell, also explizit:\n"
            "  python scripts/00_kostenhochrechnung.py --model gpt-4o-mini "
            f"--samples {args.samples[0]} --datasets {args.datasets}"
            + (f" --top-k {' '.join(map(str, args.top_k))}" if args.top_k else "")
        )

    messungen = messungen_je_framework(usage_pfad)
    if not messungen:
        raise SystemExit(
            f"Keine Messungen in {usage_pfad}. Erst einen kleinen Lauf machen:\n"
            "  python scripts/03_run_frameworks.py --sample-size 3 "
            "--out-dir results/scores/vorlaeufe/token-messung"
        )

    sag(f"Judge-Modell:   {model}")
    sag(f"Datensätze:     {args.datasets}")
    sag(f"top_k geplant:  {top_k_werte}  (gemessen bei top_k={top_k_basis})")
    sag()

    sag("Gemessener Verbrauch je Sample:")
    sag(
        f"  {'Framework':<12}{'Aufrufe':>9}{'Input':>10}{'Output':>9}{'Basis':>8}  gemessen mit"
    )
    for fw, pro in sorted(messungen.items()):
        sag(
            f"  {fw:<12}{pro.calls:>9.1f}{pro.prompt_tokens:>10.0f}"
            f"{pro.completion_tokens:>9.0f}{pro.basis_samples:>8}  {pro.gemessen_mit}"
        )

    fehlend = [f for f in ERWARTETE_FRAMEWORKS if f not in messungen]
    if fehlend:
        sag(f"\n⚠  Ohne Messung und daher NICHT eingerechnet: {', '.join(fehlend)}.")
        sag(
            "   Die Summen unten sind damit eine Untergrenze. Nachmessen kostet nichts,"
        )
        sag("   solange der Judge noch auf SCADS zeigt.")

    fremd = [m for p in messungen.values() for m in [p.gemessen_mit] if m != model]
    if fremd:
        sag(
            f"\n⚠  Die Messung stammt von einem anderen Judge ({', '.join(sorted(set(fremd)))})."
        )
        sag(
            f"   Tokenzahlen unter {model} weichen ab — Größenordnung stimmt, Details nicht."
        )

    if len(top_k_werte) > 1 or top_k_werte[0] != top_k_basis:
        sag(
            f"\n⚠  top_k-Skalierung ist eine Näherung (kontextanteil={args.kontextanteil}),"
        )
        sag("   keine Messung. Nach dem ersten Lauf bei abweichendem top_k ersetzen.")

    sag(
        f"\n{'Samples/Datensatz':>18}{'erwartet':>12}{f'+{args.sicherheit:.0%}':>12}"
        f"{'% vom Budget':>14}"
    )
    for n in args.samples:
        er = hochrechnen(
            messungen,
            model,
            n,
            args.datasets,
            top_k_werte,
            top_k_basis,
            args.kontextanteil,
        )
        gesamt = er["gesamt"]
        oben = gesamt * (1 + args.sicherheit)
        sag(f"{n:>18}{gesamt:>11.2f}€{oben:>11.2f}€{oben / args.budget:>13.0%}")

    # Aufschlüsselung für die erste Stichprobengröße
    n0 = args.samples[0]
    er = hochrechnen(
        messungen,
        model,
        n0,
        args.datasets,
        top_k_werte,
        top_k_basis,
        args.kontextanteil,
    )
    sag(f"\nAufschlüsselung für {n0} Samples/Datensatz:")
    for fw, eur in sorted(er["je_framework"].items(), key=lambda x: -x[1]):
        sag(f"  {fw:<12}{eur:>8.2f}€")
    if len(top_k_werte) > 1:
        for k, eur in sorted(er["je_top_k"].items()):
            sag(f"  top_k={k:<6}{eur:>8.2f}€  (Faktor {er['faktoren'][k]:.2f})")
    if er["ohne_preis"]:
        sag(f"  ⚠ kein Preis hinterlegt für: {', '.join(er['ohne_preis'])}")

    # Budget
    ausgegeben = bisher_ausgegeben(usage_pfad)
    rest = args.budget - ausgegeben
    sag(f"\nBudget:         {args.budget:.2f}€")
    sag(f"bereits genutzt:{ausgegeben:>7.2f}€")
    sag(f"verfügbar:      {rest:.2f}€")
    oben0 = er["gesamt"] * (1 + args.sicherheit)
    if oben0 > rest:
        sag(f"\n✗ Der Lauf passt NICHT ins Restbudget ({oben0:.2f}€ > {rest:.2f}€).")
    else:
        sag(f"\n✓ Passt: nach dem Lauf blieben ~{rest - oben0:.2f}€ übrig.")
        sag("  Weitere Läufe, z.B. mit einem anderen Judge, kosten zusätzlich.")

    if args.kein_bericht:
        return

    ziel = (
        Path(args.out)
        if args.out
        else Path(cfg["paths"]["results"]) / "kosten" / "hochrechnung.md"
    )
    ziel.parent.mkdir(parents=True, exist_ok=True)
    kopf = (
        f"# Kostenhochrechnung — {datetime.now():%d.%m.%Y %H:%M}\n\n"
        f"Erzeugt von `scripts/00_kostenhochrechnung.py`, Grundlage `{usage_pfad}`.\n"
        f"Aufruf: `{' '.join(sys.argv)}`\n\n"
        "```\n"
    )
    ziel.write_text(kopf + "\n".join(zeilen) + "\n```\n", encoding="utf-8")
    print(f"\nBericht geschrieben: {ziel}")


if __name__ == "__main__":
    main()
