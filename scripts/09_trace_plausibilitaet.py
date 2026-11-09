"""Plausibilitätsprüfung auf den Tripeln aus RAGBench:

RAGBench liefert zu jeder Antwort eigene Ragas- und TruLens-Werte mit, mit anderem Judge und anderen Versionen.
Sie sind nicht direkt vergleichbar, aber weil die Antworten dieselben sind, zeigen sie grob, ob die Adapter richtig bewerten.
Berichtet werden Spearman ρ und Kendall τ je Metrik, über alle und je Subset, mit beiden Medianen.

Kostet nichts.
Liest die Score-Dateien aus scripts/03 --dataset ragbench_own und RAGBench aus dem Hugging-Face-Cache.

Aufruf:
    python scripts/09_trace_plausibilitaet.py [--judge gpt-4o-mini]

Ausgabe: results/rq2/plausibilitaet_ragbench_own_<judge>.csv und .md
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from rag_eval.analysis.correlations import entdoppele_ergebnisse, results_to_frame
from rag_eval.analysis.trace import originalwerte, plausibilitaet
from rag_eval.config import load_config
from rag_eval.data.io import read_jsonl
from rag_eval.data.loaders.ragbench import HF_ID
from rag_eval.data.schema import GeneratedSample, MetricResult
from rag_eval.naming import generated_datei, scores_datei

DATENSATZ = "ragbench_own"


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=cfg["judge"]["model"])
    args = ap.parse_args()

    samples = read_jsonl(
        generated_datei(DATENSATZ, "", 0, cfg["paths"]["processed"], cfg),
        GeneratedSample,
    )
    results: list[MetricResult] = []
    for fw in ("ragas", "trulens"):
        pfad = Path(
            scores_datei(DATENSATZ, fw, "", 0, args.judge, cfg["paths"]["results"], cfg)
        )
        if not pfad.exists():
            sys.exit(
                f"{pfad} fehlt — erst scripts/03_run_frameworks.py --dataset {DATENSATZ}."
            )
        results.extend(read_jsonl(pfad, MetricResult))
    df = results_to_frame(entdoppele_ergebnisse(results))
    df["metrik"] = df["framework"] + "/" + df["construct"]
    eigene = df.pivot_table(index="sample_id", columns="metrik", values="score")

    zeilen_je_frage: dict = {}
    for subset in sorted({s.meta["subset"] for s in samples}):
        for row in load_dataset(HF_ID, subset, split="test"):
            zeilen_je_frage.setdefault((subset, row["id"]), []).append(row)
    original = originalwerte(samples, zeilen_je_frage)
    print(
        f"{len(original)} von {len(samples)} Antworten einer RAGBench-Zeile zugeordnet."
    )

    tabelle = plausibilitaet(eigene, original)
    out = Path(cfg["paths"]["results"]) / "rq2"
    out.mkdir(parents=True, exist_ok=True)
    stamm = out / f"plausibilitaet_{DATENSATZ}_{args.judge.replace('/', '-')}"
    tabelle.to_csv(f"{stamm}.csv", index=False)
    with open(f"{stamm}.md", "w") as f:
        f.write(
            f"# Plausibilitaet: eigene Scores ({args.judge}) gegen RAGBenchs mitgelieferte\n\n"
        )
        f.write(tabelle.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(tabelle.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n-> {stamm}.csv / .md")


if __name__ == "__main__":
    main()
