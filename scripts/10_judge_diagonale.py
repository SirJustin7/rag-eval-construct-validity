"""Judge-Diagonale der MTMM-Matrix:

Die Diagonale zeigt, wie gut jede Metrik mit sich selbst übereinstimmt, wenn nur der Judge wechselt:
Spearman ρ und Kendall τ zwischen den Scores des Haupt-Judges (config judge.model) und jedes weiteren Judges auf denselben Antworten.
Wie in RQ2 und RQ3 mit und ohne Abstinenzen, leere Antworten immer ohne.

Die weiteren Judges werden über die Dateinamen gefunden, also jede Score-Datei mit gleichem Datensatz, Zuschnitt und gleicher Tiefe, aber anderem Judge.

Kostet nichts.
Aufruf:
    python scripts/10_judge_diagonale.py [--dataset ragbench] [--top-k 10]

Ausgabe: results/rq3/judge_diagonale_<ds>_k<k>.csv und .md
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats

from rag_eval.analysis.correlations import entdoppele_ergebnisse
from rag_eval.analysis.saturation import einstufen, ungebundener_paaranteil
from rag_eval.analysis.stichprobe import MIT, OHNE, antworttypen_aus_datei, filtere
from rag_eval.config import load_config
from rag_eval.data.schema import MetricResult
from rag_eval.naming import _sauber, generated_datei, score_datei_teile


def lies_tolerant(pfad: Path) -> list[MetricResult]:
    """Wie read_jsonl, überspringt aber unvollständige Zeilen (laufender Lauf)."""
    ergebnisse = []
    for nr, zeile in enumerate(pfad.read_text().splitlines(), 1):
        if not zeile.strip():
            continue
        try:
            ergebnisse.append(MetricResult.model_validate(json.loads(zeile)))
        except (json.JSONDecodeError, ValueError):
            print(f"  ⚠ {pfad.name}, Zeile {nr}: unvollstaendig, uebersprungen")
    return ergebnisse


def als_tabelle(results: list[MetricResult]) -> pd.Series:
    return pd.Series(
        {(r.sample_id, f"{r.framework}/{r.construct.value}"): r.score for r in results}
    )


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ragbench")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--scope", default="per_domain")
    args = ap.parse_args()

    haupt = _sauber(cfg["judge"]["model"])
    scores_dir = Path(cfg["paths"]["results"]) / "scores"
    judges: dict[str, list[Path]] = {}
    for p in sorted(scores_dir.glob("*.jsonl")):
        t = score_datei_teile(p)
        if t and (t["dataset"], t["scope"], t["top_k"]) == (
            args.dataset,
            args.scope,
            args.top_k,
        ):
            judges.setdefault(t["judge"], []).append(p)
    if haupt not in judges:
        raise SystemExit(
            f"Keine Dateien des Haupt-Judges {haupt} fuer {args.dataset} k{args.top_k}."
        )
    weitere = sorted(j for j in judges if j != haupt)
    if not weitere:
        raise SystemExit(
            "Keine weiteren Judges gefunden — die Judge-Wiederholung fehlt noch."
        )

    typen = antworttypen_aus_datei(
        generated_datei(
            args.dataset, args.scope, args.top_k, cfg["paths"]["processed"], cfg
        )
    )
    n_fragen = len(typen)

    def laden(judge):
        res = []
        for p in judges[judge]:
            res += lies_tolerant(p)
        return entdoppele_ergebnisse(res)

    haupt_res = laden(haupt)
    zeilen = []
    for judge in weitere:
        print(f"[{judge}] {', '.join(p.name for p in judges[judge])}")
        weitere_res = laden(judge)
        for variante in (MIT, OHNE):
            a = als_tabelle(filtere(haupt_res, typen, variante)[0]).unstack()
            b = als_tabelle(filtere(weitere_res, typen, variante)[0]).unstack()
            for metrik in sorted(set(a.columns) & set(b.columns)):
                paar = pd.concat(
                    [a[metrik], b[metrik]], axis=1, keys=["haupt", "weiterer"]
                ).dropna()
                n = len(paar)
                konstant = (
                    n < 3
                    or paar["haupt"].nunique() < 2
                    or paar["weiterer"].nunique() < 2
                )
                rho, p_rho = (
                    (float("nan"),) * 2
                    if konstant
                    else stats.spearmanr(paar["haupt"], paar["weiterer"])
                )
                tau, p_tau = (
                    (float("nan"),) * 2
                    if konstant
                    else stats.kendalltau(paar["haupt"], paar["weiterer"])
                )
                u = ungebundener_paaranteil(paar["weiterer"]) if n else float("nan")
                zeilen.append(
                    {
                        "judge": judge,
                        "variante": variante,
                        "metrik": metrik,
                        "n": n,
                        "anteil_bewertet": b[metrik].notna().sum() / n_fragen,
                        "spearman_rho": rho,
                        "spearman_p": p_rho,
                        "kendall_tau": tau,
                        "kendall_p": p_tau,
                        "median_haupt": paar["haupt"].median(),
                        "median_weiterer": paar["weiterer"].median(),
                        "u_weiterer": u,
                        "stufe_weiterer": einstufen(
                            u, int(paar["weiterer"].nunique())
                        ).value
                        if n
                        else "",
                    }
                )

    tabelle = pd.DataFrame(zeilen)
    out = Path(cfg["paths"]["results"]) / "rq3"
    out.mkdir(parents=True, exist_ok=True)
    stamm = out / f"judge_diagonale_{args.dataset}_k{args.top_k}"
    tabelle.to_csv(f"{stamm}.csv", index=False)
    with open(f"{stamm}.md", "w") as f:
        f.write(
            f"# Judge-Diagonale {args.dataset} k{args.top_k}: {haupt} gegen weitere Judges\n\n"
        )
        f.write(tabelle.to_markdown(index=False, floatfmt=".3f"))
        f.write("\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tabelle.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\n-> {stamm}.csv / .md")


if __name__ == "__main__":
    main()
