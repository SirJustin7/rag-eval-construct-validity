"""Rangkorrelationen für RQ2 und RQ3:

Je Konstrukt werden die Scores der Frameworks paarweise korreliert (Spearman ρ, Kendall τ), also z.B. Faithfulness von Ragas mit Groundedness von TruLens.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

from ..data.schema import MetricResult
from .criterion import rang_biserial
from .saturation import (
    Stufe,
    einstufen,
    saettigungsreport,
    ungebundener_paaranteil,
)


def results_to_frame(results: list[MetricResult]) -> pd.DataFrame:
    """DataFrame im Long-Format, eine Zeile je Sample, Framework und Konstrukt."""
    return pd.DataFrame(
        {
            "sample_id": r.sample_id,
            "dataset": r.dataset,
            "framework": r.framework,
            "construct": r.construct.value,
            "score": r.score,
        }
        for r in results
    )


def entdoppele_ergebnisse(results: list[MetricResult]) -> list[MetricResult]:
    """Behält je sample_id, Framework und Konstrukt den ersten Score.

    In den RAGBench-Läufen kamen 16 der 333 Fragen doppelt vor, weil id in RAGBench die Frage und nicht die Zeile bezeichnet.
    Sie wurden zweimal generiert und bewertet, es bleiben 317 Fragen.

    Der erste Eintrag statt eines Mittelwerts, weil der Generator trotz temperature 0 nicht immer dieselbe Antwort erzeugt.
    Ein Mittelwert würde Bewertungen zweier verschiedener Antworten mischen.
    Alle Frameworks lesen dieselbe Datei in derselben Reihenfolge und behalten dieselbe Antwort.

    score_matrix bricht trotzdem weiter ab, wenn zwei Ergebnisdateien desselben Frameworks eingelesen werden.
    """
    gesehen: set[tuple] = set()
    sauber: list[MetricResult] = []
    for r in results:
        schluessel = (r.sample_id, r.framework, r.construct)
        if schluessel in gesehen:
            continue
        gesehen.add(schluessel)
        sauber.append(r)
    entfernt = len(results) - len(sauber)
    if entfernt:
        print(
            f"⚠️  {entfernt} doppelte Score-Einträge entfernt "
            f"(je sample_id/framework/construct der erste behalten)."
        )
    return sauber


def score_matrix(df: pd.DataFrame, construct: str) -> pd.DataFrame:
    """Wide-Format: Zeilen = Samples, Spalten = Frameworks, Werte = Scores.

    Es bleiben nur Samples, die alle Frameworks bewertet haben.

    Raises:
        ValueError: wenn ein Sample bei einem Framework mehrfach vorkommt.
    """
    subset = df[df["construct"] == construct]

    # Je Sample und Framework darf es nur einen Score geben. Mehrfache Einträge entstehen, wenn zwei Ergebnisdateien desselben Frameworks eingelesen werden.
    # pivot_table würde sie unbemerkt mitteln und die Korrelation verfälschen, deshalb Abbruch.
    dupes = subset.duplicated(subset=["sample_id", "framework"], keep=False)
    if dupes.any():
        details = (
            subset[dupes]
            .groupby(["framework", "sample_id"])
            .size()
            .rename("anzahl")
            .reset_index()
            .head(5)
            .to_string(index=False)
        )
        raise ValueError(
            f"Mehrfache Scores je (sample_id, framework) für Konstrukt "
            f"'{construct}' — vermutlich wurden zwei Ergebnisdateien desselben "
            f"Frameworks eingelesen. Die Korrelation wäre still verfälscht.\n"
            f"Betroffen (erste 5):\n{details}"
        )

    return subset.pivot_table(
        index="sample_id", columns="framework", values="score"
    ).dropna()


def pairwise_correlations(
    matrix: pd.DataFrame, binaer: frozenset[str] | set[str] = frozenset()
) -> pd.DataFrame:
    """Paarweise Spearman ρ und Kendall τ mit p-Werten.

    binaer nennt die Spalten mit nur zwei Werten (TRACe adherence).
    Trifft eine davon auf einen abgestuften Score, wird die rang-biseriale Korrelation nach Cureton mit Mann-Whitney-p berechnet, ρ und τ bleiben leer.
    Die Spalte koeffizient sagt, welcher Wert gilt.

    Jede Zeile enthält auch die Sättigungsstufe beider Metriken (analysis.saturation).
    Ein hohes ρ mit einer Metrik am Deckeneffekt ist keine Konvergenz.
    paar_stufe ist die strengere der beiden Stufen, belastbar ist wahr, wenn keine der beiden ausgeschlossen ist.

    Rückgabe: eine Zeile je Framework-Paar.
    """
    stufen: dict[str, Stufe] = {}
    us: dict[str, float] = {}
    for metrik in matrix.columns:
        spalte = matrix[metrik].dropna()
        us[metrik] = ungebundener_paaranteil(spalte)
        stufen[metrik] = einstufen(us[metrik], int(spalte.nunique()))

    rang = {
        Stufe.AUSWERTBAR: 0,
        Stufe.EINGESCHRAENKT: 1,
        Stufe.AUSGESCHLOSSEN: 2,
    }

    rows = []
    frameworks = list(matrix.columns)
    for i, fw_a in enumerate(frameworks):
        for fw_b in frameworks[i + 1 :]:
            a, b = matrix[fw_a], matrix[fw_b]
            koeff, r_rb, p_rb = paar_koeffizient(a, b, fw_a in binaer, fw_b in binaer)
            if koeff == "spearman":
                rho, p_rho = stats.spearmanr(a, b)
                tau, p_tau = stats.kendalltau(a, b)
            else:
                rho = p_rho = tau = p_tau = float("nan")
            schlimmer = max(stufen[fw_a], stufen[fw_b], key=lambda s: rang[s])
            rows.append(
                {
                    "framework_a": fw_a,
                    "framework_b": fw_b,
                    "n": len(matrix),
                    "koeffizient": koeff,
                    "spearman_rho": rho,
                    "spearman_p": p_rho,
                    "kendall_tau": tau,
                    "kendall_p": p_tau,
                    "r_rb": r_rb,
                    "r_rb_p": p_rb,
                    "u_a": us[fw_a],
                    "u_b": us[fw_b],
                    "stufe_a": stufen[fw_a].value,
                    "stufe_b": stufen[fw_b].value,
                    "paar_stufe": schlimmer.value,
                    "belastbar": schlimmer is not Stufe.AUSGESCHLOSSEN,
                }
            )
    return pd.DataFrame(rows)


def paar_koeffizient(
    a: pd.Series, b: pd.Series, a_binaer: bool, b_binaer: bool
) -> tuple[str, float, float]:
    """Welcher Koeffizient für ein Paar gilt, und bei binären Metriken der Wert.

    Rückgabe (koeffizient, r_rb, p).
    Bei zwei abgestuften Metriken ("spearman", nan, nan), ρ und τ rechnet der Aufrufer.
    Bei zwei binären ("keiner", nan, nan), dafür ist kein Koeffizient vorgesehen.
    """
    if a_binaer and b_binaer:
        return "keiner", float("nan"), float("nan")
    if not (a_binaer or b_binaer):
        return "spearman", float("nan"), float("nan")
    abgestuft, dicho = (b, a) if a_binaer else (a, b)
    e = rang_biserial(abgestuft.to_numpy(), dicho.to_numpy() > 0.5)
    return "rang_biserial", e.r, e.p


def saturation_report(results: list[MetricResult]) -> dict[str, pd.DataFrame]:
    """Sättigungstabelle je Konstrukt, mit derselben Regel wie in RQ4."""
    df = results_to_frame(results)
    return {
        construct: saettigungsreport(score_matrix(df, construct))
        for construct in df["construct"].unique()
    }


def convergence_report(
    results: list[MetricResult], binaere_metriken: frozenset[str] = frozenset()
) -> dict[str, pd.DataFrame]:
    """RQ2: Korrelationstabelle je Konstrukt.

    binaere_metriken: Namen "framework/konstrukt" (z.B. "trace/faithfulness"), die rang-biserial statt per Spearman korreliert werden.
    """
    df = results_to_frame(results)
    report = {}
    for construct in df["construct"].unique():
        matrix = score_matrix(df, construct)
        binaer = {
            fw for fw in matrix.columns if f"{fw}/{construct}" in binaere_metriken
        }
        report[construct] = pairwise_correlations(matrix, binaer)
    return report


def write_convergence_report(
    report: dict[str, pd.DataFrame], out_dir: Path, stem: str = "convergence"
) -> tuple[Path, Path]:
    """Schreibt den Report als eine CSV und eine Excel-Datei.

    In der CSV steht das Konstrukt in der ersten Spalte.
    Die Excel-Datei hat ein Blatt je Konstrukt und ein Gesamtblatt.

    Rückgabe: (Pfad der CSV, Pfad der XLSX).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(
        [table.assign(construct=construct) for construct, table in report.items()],
        ignore_index=True,
    )
    combined = combined[
        ["construct", *[c for c in combined.columns if c != "construct"]]
    ]

    csv_path = out_dir / f"{stem}.csv"
    xlsx_path = out_dir / f"{stem}.xlsx"
    combined.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="alle Konstrukte", index=False)
        for construct, table in report.items():
            # Excel erlaubt höchstens 31 Zeichen je Blattname.
            table.to_excel(writer, sheet_name=construct[:31], index=False)

    return csv_path, xlsx_path
