"""Zwei Gegenproben zur Konvergenz von Faithfulness:

    PYTHONPATH=src python scripts/13_faithfulness_robustheit.py

Frage: Ist die schwache Konvergenz von Faithfulness nur ein Deckeneffekt, weil ein großer Teil der beantworteten Fragen genau 1,0 bekommt?

Test 1  Jede Metrik wird auf "hat etwas Ungedecktes gefunden" (Score < 1,0) reduziert, dann Spearman zwischen den drei Paaren. Das entfernt die Bindungen der abgestuften Werte.
Test 2  Nur die Antworten, die beide Metriken eines Paars unter 1,0 bewerten.

Je CRAG und RAGBench bei Tiefe 10 und 5, ohne Abstinenzen, Haupt-Judge.
"""

from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy import stats

from rag_eval.analysis.correlations import results_to_frame
from rag_eval.analysis.stichprobe import OHNE, filtere, lade_datensatzlaeufe
from rag_eval.config import load_config
from rag_eval.naming import waehle_score_dateien

cfg = load_config()
scores = Path(cfg["paths"]["results"]) / "scores"
zeilen = []
for ds in ["crag", "ragbench"]:
    for k in [10, 5]:
        pfade = waehle_score_dateien(
            scores, [ds], k, "per_domain", judge=cfg["judge"]["model"]
        )
        d = lade_datensatzlaeufe(pfade, [ds], cfg["paths"]["processed"], cfg)[0]
        teil, _ = filtere(d.results, d.typen, OHNE)
        df = results_to_frame(teil)
        m = df[df.construct == "faithfulness"].pivot_table(
            index="sample_id", columns="framework", values="score", aggfunc="first"
        )
        for a, b in combinations(["ragas", "trulens", "deepeval"], 2):
            t = m[[a, b]].dropna()
            r1 = stats.spearmanr(t[a] < 1.0, t[b] < 1.0)
            u = t[(t[a] < 1.0) & (t[b] < 1.0)]
            r2 = stats.spearmanr(u[a], u[b]) if len(u) > 2 else None
            zeilen.append(
                dict(
                    lauf=f"{ds}_k{k}",
                    paar=f"{a}-{b}",
                    n=len(t),
                    binaer_rho=r1.statistic,
                    binaer_p=r1.pvalue,
                    n_beide_unter_1=len(u),
                    unter1_rho=r2.statistic if r2 else None,
                )
            )
erg = pd.DataFrame(zeilen)
pd.set_option("display.width", 160)
print(erg.round(3).to_string(index=False))
print(
    "\nTest 1: rho",
    round(erg.binaer_rho.min(), 3),
    "bis",
    round(erg.binaer_rho.max(), 3),
    "| p<0.05:",
    int((erg.binaer_p < 0.05).sum()),
    "von",
    len(erg),
    "| n",
    erg.n.min(),
    "bis",
    erg.n.max(),
)
for lauf, g in erg.groupby("lauf"):
    print(
        f"Test 2 {lauf}: rho {g.unter1_rho.min():.3f} bis {g.unter1_rho.max():.3f}, n {g.n_beide_unter_1.min()}-{g.n_beide_unter_1.max()}"
    )
ziel = Path(cfg["paths"]["results"]) / "rq2" / "faithfulness_robustheit.csv"
erg.to_csv(ziel, index=False)
print("geschrieben:", ziel)
