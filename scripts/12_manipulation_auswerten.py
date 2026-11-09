"""Manipulationsbedingungen gegen den unveränderten Lauf (RQ3):

Gepaarter Vergleich je Metrik: der Score nach der Manipulation gegen den Score desselben Samples im Hauptlauf.
Wilcoxon-Vorzeichen-Rang-Test mit gepaarter rang-biserialer Effektstärke, zusätzlich getrennt nach dem Antworttyp im Hauptlauf (Abstinenz, Teilantwort, Antwort).

Voraussetzung:
    python scripts/07_manipulate.py --dataset crag --bedingungen inject_1 …
    python scripts/03_run_frameworks.py --dataset crag_inject_1     # bezahlt

Aufruf:
    python scripts/12_manipulation_auswerten.py
    python scripts/12_manipulation_auswerten.py --bedingungen inject_1 replace_full

Kostet nichts, rechnet lokal.

Zwei Spalten der Ausgabe:
  * rolle: Context Relevance liest nur Frage und Kontext und wird von der Manipulation nicht erreicht. Dort steht "retest", der Wert zeigt nur die Streuung des Judge.
  * woerter_*: Das Einfügen macht die Antwort deutlich länger. Judges können längere Antworten anders bewerten, deshalb steht die Länge dabei.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_eval.analysis.abstinenz import klassifiziere
from rag_eval.analysis.manipulation_vergleich import (
    erwartungsabgleich,
    vergleichstabelle,
)
from rag_eval.config import load_config
from rag_eval.data.io import entdoppele, read_jsonl
from rag_eval.data.schema import GeneratedSample, MetricResult
from rag_eval.manipulation import ERWARTUNG
from rag_eval.naming import generated_datei, scope_aus_config, top_k_aus_config

GENERIERUNG = ["inject_1", "inject_2", "inject_3", "replace_full", "swap_fact"]


def _scores(
    scores_dir: Path, dataset: str, judge: str, suffix: str = ""
) -> pd.DataFrame:
    """Alle Framework-Scores eines Datensatzes als (sample_id, metric, score)."""
    pfade = sorted(scores_dir.glob(f"{dataset}_*{suffix}_{judge}.jsonl"))
    # crag_* passt sonst auch auf crag_inject_1_*.
    pfade = [
        p
        for p in pfade
        if p.stem.split("_")[0 : len(dataset.split("_"))] == dataset.split("_")
    ]
    zeilen = []
    for p in pfade:
        teil = p.stem[len(dataset) + 1 :]
        if teil.split("_")[0] not in ("ragas", "trulens", "deepeval"):
            continue
        for r in read_jsonl(str(p), MetricResult):
            k = getattr(r.construct, "value", r.construct)
            zeilen.append(
                {
                    "sample_id": r.sample_id,
                    "metric": f"{r.framework}.{k}",
                    "score": r.score,
                }
            )
    return pd.DataFrame(zeilen)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--basis", default="crag", help="Datensatz des Hauptlaufs")
    p.add_argument("--bedingungen", nargs="+", default=GENERIERUNG)
    p.add_argument("--judge", default=None)
    p.add_argument("--scores-dir", default=None)
    p.add_argument("--out-dir", default=None, help="Default: <results>/rq3")
    args = p.parse_args()

    cfg = load_config()
    judge = (args.judge or cfg["judge"]["model"]).replace("/", "-")
    root = Path(cfg["paths"]["results"])
    scores_dir = Path(args.scores_dir) if args.scores_dir else root / "scores"
    out_dir = Path(args.out_dir) if args.out_dir else root / "rq3"
    out_dir.mkdir(parents=True, exist_ok=True)

    scope, top_k = scope_aus_config(cfg["rag"]), top_k_aus_config(cfg["rag"])
    basis = _scores(scores_dir, args.basis, judge, f"_{scope}_k{top_k}")
    if basis.empty:
        sys.exit(
            f"Keine Scores des Hauptlaufs {args.basis} ({scope}, k{top_k}) gefunden."
        )

    # Gruppiert wird nach dem Antworttyp im Hauptlauf. Nach dem Einfügen ist jede Abstinenz eine Teilantwort, die Gruppe wäre dann leer.
    haupt = entdoppele(
        read_jsonl(
            generated_datei(args.basis, scope, top_k, cfg["paths"]["processed"], cfg),
            GeneratedSample,
        ),
        args.basis,
    )
    typ = pd.Series(
        {s.sample_id: klassifiziere(s.generated_answer).name for s in haupt}
    )
    woerter_basis = pd.Series(
        {s.sample_id: len(s.generated_answer.split()) for s in haupt}
    )

    alle, fehlend = [], []
    for bed in args.bedingungen:
        ds = f"{args.basis}_{bed}"
        nach = _scores(scores_dir, ds, judge)
        if nach.empty:
            fehlend.append(ds)
            continue
        pfad = generated_datei(ds, scope, top_k, cfg["paths"]["processed"], cfg)
        laenge = float("nan")
        if Path(pfad).exists():
            manip = read_jsonl(pfad, GeneratedSample)
            laenge = sum(len(s.generated_answer.split()) for s in manip) / max(
                1, len(manip)
            )
        seite = "generierung"
        t = vergleichstabelle(basis, nach, bed, seite=seite, gruppen=typ)
        t["seite"] = seite
        t["woerter_vorher"] = woerter_basis.reindex(nach["sample_id"].unique()).mean()
        t["woerter_nachher"] = laenge
        alle.append(t)
        print(
            f"[{bed}] {nach['sample_id'].nunique()} Samples, "
            f"{nach['metric'].nunique()} Metriken, Antwortlänge "
            f"{t['woerter_vorher'].iloc[0]:.0f} → {laenge:.0f} Wörter"
        )

    if fehlend:
        print("\nNoch nicht bewertet, übersprungen: " + ", ".join(fehlend))
    if not alle:
        sys.exit("Keine bewertete Manipulationsbedingung gefunden.")

    tabelle = erwartungsabgleich(pd.concat(alle, ignore_index=True), ERWARTUNG)
    spalten = [
        "bedingung",
        "metrik",
        "gruppe",
        "n",
        "median_vorher",
        "median_nachher",
        "median_differenz",
        "r_paarweise",
        "p",
        "n_gefallen",
        "n_gestiegen",
        "n_unveraendert",
        "rolle",
        "erwartet",
        "beobachtet",
        "woerter_vorher",
        "woerter_nachher",
    ]
    tabelle = tabelle[spalten]

    for bed, teil in tabelle.groupby("bedingung", sort=False):
        print(f"\n{'=' * 70}\n=== {bed}\n{'=' * 70}")
        zeig = teil[teil["gruppe"] == "alle"]
        print(
            zeig[
                [
                    "metrik",
                    "n",
                    "median_vorher",
                    "median_nachher",
                    "r_paarweise",
                    "p",
                    "n_gefallen",
                    "rolle",
                    "beobachtet",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )

    stem = f"manipulation_{args.basis}_{judge}"
    tabelle.to_csv(out_dir / f"{stem}.csv", index=False)
    with pd.ExcelWriter(out_dir / f"{stem}.xlsx", engine="openpyxl") as w:
        tabelle.to_excel(w, sheet_name="vergleich", index=False)
        tabelle[tabelle["gruppe"] == "alle"].to_excel(
            w, sheet_name="gesamt", index=False
        )
    print(f"\nGeschrieben: {stem}.csv · {stem}.xlsx")


if __name__ == "__main__":
    main()
