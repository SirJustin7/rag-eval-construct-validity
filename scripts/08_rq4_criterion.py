"""RQ4: Kriteriumsvalidität gegen die Korrektheit der Antwort:

Für jede Metrik (Framework x Konstrukt) die rang-biseriale Korrelation nach Cureton (1968) zwischen Score und Korrektheit des Urteils, mit Mann-Whitney-p, über alle Tripel und getrennt je Generator.
Die Generatoren liegen bei der Korrektheit weit auseinander, ein gemeinsamer Koeffizient könnte allein diesen Unterschied zeigen.

Voraussetzung:
    python scripts/02b_import_depvalidation.py                      # 200 Tripel, kostenlos
    python scripts/03_run_frameworks.py --dataset depvalidation     # bezahlt

Aufruf:
    python scripts/08_rq4_criterion.py [--judge gpt-4o-mini]

Ausgabe: results/rq4/kriterium_<judge>.csv und .md

"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from rag_eval.analysis.criterion import kriteriumstabelle
from rag_eval.analysis.plots import criterion_distribution
from rag_eval.analysis.saturation import formatiere, saettigungsreport
from rag_eval.config import load_config
from rag_eval.data.io import read_jsonl
from rag_eval.data.schema import GeneratedSample, MetricResult
from rag_eval.naming import scores_datei

FRAMEWORKS = ["ragas", "trulens", "deepeval"]
DATENSATZ = "depvalidation"


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=cfg["judge"]["model"])
    args = ap.parse_args()

    tripel = f"{cfg['paths']['processed']}/{DATENSATZ}_generated.jsonl"
    if not Path(tripel).exists():
        sys.exit(
            f"{tripel} fehlt — erst scripts/02b_import_depvalidation.py laufen lassen."
        )
    meta = {s.sample_id: s.meta for s in read_jsonl(tripel, GeneratedSample)}

    zeilen = []
    for fw in FRAMEWORKS:
        pfad = Path(
            scores_datei(DATENSATZ, fw, "", 0, args.judge, cfg["paths"]["results"], cfg)
        )
        if not pfad.exists():
            print(f"[{fw}] keine Scores ({pfad}) — uebersprungen.")
            continue
        for r in read_jsonl(pfad, MetricResult):
            m = meta.get(r.sample_id)
            if m is None:
                continue
            zeilen.append(
                {
                    "sample_id": r.sample_id,
                    "metric": f"{fw}.{r.construct.value}",
                    "score": r.score,
                    "richtig": bool(m["answer_correct"]),
                    "generator": m["generator"],
                }
            )
    if not zeilen:
        sys.exit(
            "Keine Scores gefunden. Erst scripts/03_run_frameworks.py --dataset depvalidation."
        )

    df = pd.DataFrame(zeilen).drop_duplicates(["sample_id", "metric"], keep="first")
    tabelle = kriteriumstabelle(df, "generator").sort_values(["metric", "gruppe"])

    # Jeder Koeffizient bekommt die Sättigungsstufe seiner Metrik, wie in RQ2 und RQ3.
    matrix = df.pivot_table(index="sample_id", columns="metric", values="score")
    saettigung = saettigungsreport(matrix).set_index("metrik")
    tabelle["u"] = tabelle["metric"].map(saettigung["ungebundener_paaranteil"])
    tabelle["stufe"] = tabelle["metric"].map(saettigung["stufe"])
    tabelle["belastbar"] = tabelle["stufe"] != "ausgeschlossen"

    out = Path(cfg["paths"]["results"]) / "rq4"
    out.mkdir(parents=True, exist_ok=True)
    stamm = out / f"kriterium_{args.judge.replace('/', '-')}"
    tabelle.to_csv(f"{stamm}.csv", index=False)
    with open(f"{stamm}.md", "w") as f:
        f.write(
            f"# RQ4 — rang-biseriale Korrelation (Cureton 1968), Judge {args.judge}\n\n"
        )
        f.write(tabelle.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")
    print("\n" + formatiere(saettigungsreport(matrix)) + "\n")
    # Abbildung: Trennen die Scores richtige von falschen Urteilen?
    bild = criterion_distribution(
        df, out / f"kriterium_{args.judge.replace('/', '-')}.png", tabelle
    )
    print(f"Abbildung: {bild}")

    with pd.option_context("display.width", 200, "display.max_columns", 25):
        print(tabelle.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n-> {stamm}.csv / .md")


if __name__ == "__main__":
    main()
