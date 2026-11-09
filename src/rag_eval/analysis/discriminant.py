"""Diskriminante Validität (RQ3): Multitrait-Multimethod-Matrix und PCA:

Anders als in RQ2 werden hier alle Metriken miteinander korreliert, nicht nur die desselben Konstrukts.
Die Matrix zerfällt nach Campbell und Fiske (1959) in drei Blöcke:

    monotrait-heteromethod    gleiches Konstrukt, anderes Framework  -> soll hoch sein
    heterotrait-monomethod    anderes Konstrukt, gleiches Framework  -> soll niedrig sein
    heterotrait-heteromethod  anderes Konstrukt, anderes Framework   -> soll niedrig sein

Diskriminante Validität liegt vor, wenn der erste Block über den beiden anderen liegt.
Das Kriterium ist eine Rangfolge, kein Schwellenwert, weil Deckeneffekte die absoluten Werte stauchen, die Rangfolge aber nicht.

Das Modul erzeugt keine neuen Scores.
Es liest dieselben Dateien wie RQ2 und ordnet die Zahlen anders an.
Das Judge-Modell kommt aus MetricResult.meta["judge_model"] und steht in jedem Report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..data.schema import MetricResult
from .correlations import paar_koeffizient
from .saturation import Stufe, saettigungsreport

MONO_HETERO = "monotrait-heteromethod"  # konvergent
HETERO_MONO = "heterotrait-monomethod"
HETERO_HETERO = "heterotrait-heteromethod"

BLOCK_ORDER = [MONO_HETERO, HETERO_MONO, HETERO_HETERO]

# Paare mit einer binären Metrik (TRACe adherence). Ihr Koeffizient ist rang-biserial und nicht mit ρ zwischen abgestuften Metriken vergleichbar.
# Sie werden getrennt berichtet und gehen in keinen der drei Blöcke ein.
BINAER_BLOCK = "binaer (gesondert)"


def metric_key(framework: str, construct: str) -> str:
    """Name einer Metrik aus Framework und Konstrukt, z.B. 'ragas/faithfulness'."""
    return f"{framework}/{construct}"


def split_key(key: str) -> tuple[str, str]:
    framework, construct = key.split("/", 1)
    return framework, construct


# Reihenfolge der Matrixachsen: nach Framework gruppiert, innerhalb des Frameworks CR, F, AR, wie in der MTMM-Abbildung der Arbeit.
# Alphabetisch sortiert wäre das Blockmuster nicht zu erkennen.
FRAMEWORK_ORDER = ["ragas", "trulens", "deepeval", "trace"]
CONSTRUCT_ORDER = ["context_relevance", "faithfulness", "answer_relevance"]


def order_metrics(metrics: list[str]) -> list[str]:
    """Bringt die Metriknamen in die MTMM-Reihenfolge.

    Unbekannte Frameworks und Konstrukte kommen alphabetisch ans Ende.
    """

    def sort_key(metric: str) -> tuple[int, str, int, str]:
        framework, construct = split_key(metric)
        fw_idx = (
            FRAMEWORK_ORDER.index(framework)
            if framework in FRAMEWORK_ORDER
            else len(FRAMEWORK_ORDER)
        )
        con_idx = (
            CONSTRUCT_ORDER.index(construct)
            if construct in CONSTRUCT_ORDER
            else len(CONSTRUCT_ORDER)
        )
        return (fw_idx, framework, con_idx, construct)

    return sorted(metrics, key=sort_key)


def block_of(key_a: str, key_b: str) -> str:
    """Block eines Metrikpaares (die Diagonale kommt hier nicht vor)."""
    fw_a, con_a = split_key(key_a)
    fw_b, con_b = split_key(key_b)
    if con_a == con_b:
        return MONO_HETERO
    if fw_a == fw_b:
        return HETERO_MONO
    return HETERO_HETERO


# Daten


@dataclass
class MatrixInfo:
    """Angaben zur Score-Matrix, die in jedem Report stehen."""

    n_samples: int
    n_metrics: int
    dropped_samples: int
    judge_models: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                ("n_samples", self.n_samples),
                ("n_metrics", self.n_metrics),
                ("dropped_samples", self.dropped_samples),
                ("judge_model", ", ".join(self.judge_models) or "unbekannt"),
                ("datasets", ", ".join(self.datasets)),
            ],
            columns=["feld", "wert"],
        )


class JudgeMismatch(ValueError):
    """Die Score-Dateien stammen nicht (nur) vom erwarteten Judge-Modell."""


def verify_judge(info: MatrixInfo, expected: str) -> None:
    """Prüft, dass genau ein Judge-Modell vorkommt, und zwar das erwartete.

    Jeder Score hängt vom Judge ab. Mischt die Matrix Dateien verschiedener Judges, sind die Korrelationen wertlos, sehen aber weiter plausibel aus.
    Deshalb Abbruch statt Warnung.

    Das erwartete Modell kommt aus config.yaml (judge.model).
    Für die Judge-Wiederholungen wird es mit --expect-judge angegeben.

    Raises:
        JudgeMismatch: bei mehreren Judge-Modellen, bei einem abweichenden Modell oder wenn das Modell in den Score-Dateien fehlt.
    """
    gefunden = info.judge_models

    if "unbekannt" in gefunden:
        raise JudgeMismatch(
            "Mindestens eine Score-Zeile hat kein 'judge_model' in meta. Ohne diese "
            "Angabe lässt sich nicht prüfen, womit bewertet wurde — die Datei stammt "
            "vermutlich aus einem Lauf vor der Judge-Umstellung. Entweder die Datei "
            "aussortieren oder den Lauf wiederholen."
        )

    if len(gefunden) > 1:
        raise JudgeMismatch(
            "Die Score-Dateien stammen von MEHREREN Judge-Modellen:\n  "
            + "\n  ".join(gefunden)
            + "\nScores verschiedener Judges dürfen nicht in dieselbe Matrix. "
            "Läufe getrennt auswerten (--files und --stem je Lauf)."
        )

    if gefunden[0] != expected:
        raise JudgeMismatch(
            f"Judge-Modell passt nicht: erwartet '{expected}' (config judge.model), "
            f"gefunden '{gefunden[0]}'.\n"
            "Entweder sind es die falschen Score-Dateien, oder die Auswertung ist "
            f"bewusst gewollt — dann explizit: --expect-judge '{gefunden[0]}'."
        )


def metric_matrix(
    results: list[MetricResult], complete_cases: bool = True
) -> tuple[pd.DataFrame, MatrixInfo]:
    """Wide-Format: Zeilen = Samples, Spalten = Metriken (Framework × Konstrukt).

    Args:
        complete_cases: Nur Samples behalten, die alle Metriken bewertet haben (listenweiser Ausschluss).
            So beruhen alle Zellen auf derselben Stichprobe und die Blockmediane sind vergleichbar.
            Die PCA braucht das ohnehin.
            Eine fehlende Bewertung kostet so ein ganzes Sample, wie viele, steht in MatrixInfo.dropped_samples.

    Raises:
        ValueError: bei mehrfachen Scores je sample_id und Metrik, wie in correlations.score_matrix.
    """
    df = pd.DataFrame(
        {
            "sample_id": r.sample_id,
            "dataset": r.dataset,
            "metric": metric_key(r.framework, r.construct.value),
            "score": r.score,
            "judge_model": r.meta.get("judge_model", "unbekannt"),
            "extern": bool(r.meta.get("extern", False)),
        }
        for r in results
    )
    if df.empty:
        raise ValueError(
            "Keine Scores übergeben — wurden die richtigen Dateien gelesen?"
        )

    dupes = df.duplicated(subset=["sample_id", "metric"], keep=False)
    if dupes.any():
        details = (
            df[dupes]
            .groupby(["metric", "sample_id"])
            .size()
            .rename("anzahl")
            .reset_index()
            .head(5)
            .to_string(index=False)
        )
        raise ValueError(
            "Mehrfache Scores je (sample_id, Metrik) — vermutlich wurden zwei "
            "Ergebnisdateien desselben Frameworks eingelesen (z.B. ein archivierter "
            "Zweitlauf). Die Korrelationen wären still verfälscht.\n"
            f"Betroffen (erste 5):\n{details}"
        )

    wide = df.pivot_table(index="sample_id", columns="metric", values="score")
    wide = wide[order_metrics(list(wide.columns))]
    before = len(wide)
    if complete_cases:
        wide = wide.dropna()

    info = MatrixInfo(
        n_samples=len(wide),
        n_metrics=wide.shape[1],
        dropped_samples=before - len(wide),
        # Mitgelieferte Labels (TRACe) stammen nicht vom eigenen Judge und zählen bei der Judge-Prüfung nicht mit.
        judge_models=sorted(df.loc[~df["extern"], "judge_model"].unique()),
        datasets=sorted(df["dataset"].unique()),
    )
    return wide, info


# MTMM-Matrix


def correlation_table(
    matrix: pd.DataFrame, binaer: frozenset[str] | set[str] = frozenset()
) -> pd.DataFrame:
    """Alle Metrikpaare mit ρ, τ und Block (Long-Format).

    Jedes Paar kommt einmal vor, die Diagonale nicht.
    Spearman und Kendall werden wie in correlations.pairwise_correlations beide berechnet, die Blöcke werden mit ρ ausgewertet.

    Paare mit einer Metrik aus binaer bekommen den rang-biserialen Koeffizienten (r_rb, r_rb_p) und den Block BINAER_BLOCK. ρ, τ und |ρ| bleiben leer, damit sie in keinen Blockmedian eingehen.
    """
    rows = []
    metrics = list(matrix.columns)
    for i, key_a in enumerate(metrics):
        for key_b in metrics[i + 1 :]:
            # Fehlende Werte je Paar entfernen. Nötig bei --pairwise, sonst gibt scipy NaN zurück. n steht deshalb je Paar.
            paar = matrix[[key_a, key_b]].dropna()
            a, b = paar[key_a], paar[key_b]
            koeff, r_rb, p_rb = paar_koeffizient(a, b, key_a in binaer, key_b in binaer)
            if len(paar) < 3 or koeff != "spearman":
                rho = p_rho = tau = p_tau = float("nan")
            else:
                rho, p_rho = stats.spearmanr(a, b)
                tau, p_tau = stats.kendalltau(a, b)
            fw_a, con_a = split_key(key_a)
            fw_b, con_b = split_key(key_b)
            rows.append(
                {
                    "metric_a": key_a,
                    "metric_b": key_b,
                    "framework_a": fw_a,
                    "framework_b": fw_b,
                    "construct_a": con_a,
                    "construct_b": con_b,
                    "block": BINAER_BLOCK
                    if koeff != "spearman"
                    else block_of(key_a, key_b),
                    "n": len(paar),
                    "koeffizient": koeff,
                    "spearman_rho": rho,
                    "spearman_p": p_rho,
                    "kendall_tau": tau,
                    "kendall_p": p_tau,
                    "r_rb": r_rb,
                    "r_rb_p": p_rb,
                    "abs_rho": abs(rho) if pd.notna(rho) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def correlation_square(table: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Macht aus der Long-Tabelle eine quadratische ρ-Matrix (für Heatmap und Export).

    Bei Paaren mit einer binären Metrik steht r_rb in der Zelle.
    """
    square = pd.DataFrame(index=metrics, columns=metrics, dtype=float)
    for m in metrics:
        square.loc[m, m] = 1.0
    for row in table.itertuples():
        wert = row.spearman_rho
        if pd.isna(wert) and "r_rb" in table.columns:
            wert = row.r_rb
        square.loc[row.metric_a, row.metric_b] = wert
        square.loc[row.metric_b, row.metric_a] = wert
    return square


def bewertbare_paare(
    table: pd.DataFrame, ausgeschlossen: frozenset[str] | set[str]
) -> pd.Series:
    """Paare, die in den Blockvergleich und die Verstöße eingehen.

    Nur die drei Blöcke (nicht BINAER_BLOCK) und kein Paar mit einer ausgeschlossenen Metrik.
    Ausgeschlossene Metriken bleiben in der Matrix sichtbar, gehen aber in keinen Blockmedian ein.
    """
    return (
        table["block"].isin(BLOCK_ORDER)
        & ~table["metric_a"].isin(ausgeschlossen)
        & ~table["metric_b"].isin(ausgeschlossen)
    )


def block_summary(
    table: pd.DataFrame, ausgeschlossen: frozenset[str] | set[str] = frozenset()
) -> pd.DataFrame:
    """Kennzahlen je Block, die Hauptaussage von RQ3.

    ausgeschlossen: Metriken, die die Ausschlussregel ausschließt. Ihre Paare fehlen in den Kennzahlen, wie viele je Block, steht in n_ausgelassen.

    Berichtet wird |ρ|, weil in den Heterotrait-Blöcken jeder starke Zusammenhang ein Problem ist, egal mit welchem Vorzeichen.

    Im konvergenten Block reicht der Betrag nicht.
    Dort ist ein negatives ρ das Gegenteil von Konvergenz, der Betrag würde ρ = -0.9 aber als starke Konvergenz zählen.
    Deshalb steht median_signed daneben, und negative_convergent() listet negative konvergente Werte einzeln.
    """
    alle = table[table["block"].isin(BLOCK_ORDER)]
    table = table[bewertbare_paare(table, ausgeschlossen)]
    grouped = (
        table.groupby("block")["abs_rho"]
        .agg(n_paare="size", median="median", mittel="mean", min="min", max="max")
        .reindex(BLOCK_ORDER)
        .reset_index()
    )
    signed = table.groupby("block")["spearman_rho"].median().reindex(BLOCK_ORDER)
    grouped["median_signed"] = signed.to_numpy()
    grouped["n_ausgelassen"] = (
        alle.groupby("block").size().reindex(BLOCK_ORDER).fillna(0).to_numpy()
        - grouped["n_paare"].fillna(0).to_numpy()
    ).astype(int)
    ranks = grouped["median"].rank(ascending=False, method="min")
    grouped["rang_nach_median"] = ranks.astype("Int64")
    grouped["erwartung"] = [
        "hoch (konvergent)",
        "niedrig",
        "niedrig",
    ]
    return grouped


def negative_convergent(table: pd.DataFrame) -> pd.DataFrame:
    """Konvergente Paare mit negativem ρ.

    Zwei Metriken desselben Konstrukts sollen die Samples gleich ordnen.
    Ein negatives ρ heißt, sie widersprechen sich.
    Im Blockmedian über |ρ| sähe das wie Konvergenz aus.
    """
    conv = table[(table["block"] == MONO_HETERO) & (table["spearman_rho"] < 0)]
    return (
        conv[["metric_a", "metric_b", "spearman_rho", "spearman_p", "n"]]
        .sort_values("spearman_rho")
        .reset_index(drop=True)
    )


def ordering_holds(summary: pd.DataFrame) -> bool:
    """Liegt der konvergente Block über beiden anderen?"""
    med = summary.set_index("block")["median"]
    if med.isna().any():
        return False
    return bool(
        med[MONO_HETERO] > med[HETERO_MONO] and med[MONO_HETERO] > med[HETERO_HETERO]
    )


def violations(
    table: pd.DataFrame, ausgeschlossen: frozenset[str] | set[str] = frozenset()
) -> pd.DataFrame:
    """Verstöße nach dem Zeilen- und Spaltenkriterium von Campbell und Fiske.

    Jeder konvergente Wert (Validitätsdiagonale) soll größer sein als jede Heterotrait-Korrelation, an der eine seiner beiden Metriken beteiligt ist.
    Jeder Verstoß wird einzeln ausgegeben, weil er das Metrikpaar nennt, das auf etwas anderes reagiert als auf sein Konstrukt.
    """
    # Binäre Paare haben kein vergleichbares |ρ|, ausgeschlossene Metriken zählen nicht.
    table = table[bewertbare_paare(table, ausgeschlossen)]
    convergent = table[table["block"] == MONO_HETERO]
    hetero = table[table["block"] != MONO_HETERO]

    rows = []
    for conv in convergent.itertuples():
        involved = {conv.metric_a, conv.metric_b}
        for het in hetero.itertuples():
            if not involved & {het.metric_a, het.metric_b}:
                continue
            if het.abs_rho > conv.abs_rho:
                rows.append(
                    {
                        "validitaets_paar": f"{conv.metric_a} ↔ {conv.metric_b}",
                        "validitaets_rho": conv.spearman_rho,
                        "verletzendes_paar": f"{het.metric_a} ↔ {het.metric_b}",
                        "verletzendes_rho": het.spearman_rho,
                        "block": het.block,
                        "differenz_abs": het.abs_rho - conv.abs_rho,
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("differenz_abs", ascending=False).reset_index(
            drop=True
        )
    return result


# Streuung und Deckeneffekt


def variance_check(matrix: pd.DataFrame) -> pd.DataFrame:
    """Streuung je Metrik, als Prüfung vor dem Lesen der Matrix.

    Eine Metrik ohne Streuung kann in keinem Block eine belastbare Korrelation liefern.
    Das ist ein Deckeneffekt und keine Aussage über diskriminante Validität.

    Die Einstufung kommt aus analysis.saturation und ist dieselbe wie in RQ2, damit beide Auswertungen eine Metrik gleich behandeln.
    """
    report = saettigungsreport(matrix).rename(columns={"metrik": "metric"})
    report["gesaettigt"] = report["stufe"] == Stufe.AUSGESCHLOSSEN.value
    return report.sort_values("distinkte_werte").reset_index(drop=True)


# PCA


def pca_report(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Hauptkomponentenanalyse über alle Metriken.

    Auf der Spearman-Korrelationsmatrix, also rangbasiert wie die übrigen Analysen. Die Zahl der Komponenten bestimmt die Parallelanalyse.

    Rückgabe: (Varianztabelle, Ladungen, Warnungen).
    """
    warnings: list[str] = []

    # Die PCA verträgt keine fehlenden Werte. Bei --pairwise enthält die Matrix noch NaN, dann wird hier listenweise ausgeschlossen und das gemeldet.
    # Die MTMM-Matrix rechnet weiter paarweise.
    if matrix.isna().any().any():
        vorher = len(matrix)
        matrix = matrix.dropna()
        warnings.append(
            f"PCA nur auf vollständigen Fällen: {vorher - len(matrix)} von {vorher} "
            "Sample(s) ausgeschlossen. Die MTMM-Matrix oben beruht dagegen auf "
            "paarweisem Ausschluss — die beiden Analysen haben damit ein "
            "unterschiedliches n."
        )
    n_samples, n_metrics = matrix.shape

    if n_samples <= n_metrics:
        warnings.append(
            f"n={n_samples} Samples bei {n_metrics} Variablen — die PCA ist damit "
            "nicht identifizierbar interpretierbar; Ergebnis rein explorativ."
        )
    ratio = n_samples / n_metrics if n_metrics else 0
    if ratio < 5:
        warnings.append(
            f"Verhältnis Fälle:Variablen = {ratio:.1f}:1. Gängige Faustregeln "
            "verlangen 5:1 bis 10:1 (Costello & Osborne 2005) — die Ladungen sind "
            "als qualitativer Hinweis zu lesen, nicht als Ergebnis."
        )

    konstant = [c for c in matrix.columns if matrix[c].nunique() <= 1]
    if konstant:
        warnings.append(
            "Konstante Metriken von der PCA ausgeschlossen (keine Varianz): "
            + ", ".join(konstant)
        )
    usable = matrix.drop(columns=konstant)

    eigen, vektoren = _spearman_eigen(usable.to_numpy(dtype=float))
    schwelle = parallelanalyse(usable)
    behalten = 0
    for ew, sw in zip(eigen, schwelle):  # bis zum ersten Eigenwert unter der Schwelle
        if ew > sw:
            behalten += 1
        else:
            break

    p = len(eigen)
    comps = [f"PC{i + 1}" for i in range(p)]
    variance = pd.DataFrame(
        {
            "komponente": comps,
            "eigenwert": eigen,
            "schwelle_perm_95": schwelle,
            "behalten": [i < behalten for i in range(p)],
            "anteil_varianz": eigen / p,
            "kumuliert": np.cumsum(eigen) / p,
        }
    )
    # Ladung = Eigenvektor · sqrt(Eigenwert), die Korrelation von Metrik und Komponente.
    loadings = (
        pd.DataFrame(
            vektoren * np.sqrt(np.clip(eigen, 0, None)),
            index=usable.columns,
            columns=comps,
        )
        .rename_axis("metric")
        .reset_index()
    )

    return variance, loadings, warnings


def _spearman_eigen(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenwerte (absteigend) und Eigenvektoren der Spearman-Korrelationsmatrix."""
    raenge = np.apply_along_axis(stats.rankdata, 0, x)  # mittlere Ränge bei Bindungen
    r = np.corrcoef(raenge, rowvar=False)
    werte, vektoren = np.linalg.eigh(r)
    ordnung = np.argsort(werte)[::-1]
    return werte[ordnung], vektoren[:, ordnung]


def parallelanalyse(
    matrix: pd.DataFrame, n_perm: int = 999, quantil: float = 0.95, seed: int = 42
) -> np.ndarray:
    """Schwellen der Parallelanalyse (Horn 1965) in Permutationsform (Buja & Eyuboglu 1992).

    Jede Metrik wird für sich permutiert.
    Das zerstört jeden Zusammenhang zwischen den Metriken, die Verteilung jeder einzelnen mit ihren Bindungen bleibt erhalten.
    Rückgabe: je Komponente das quantil der Eigenwerte über n_perm Permutationen (999, mit den Originaldaten zusammen 1000).
    """
    x = np.apply_along_axis(stats.rankdata, 0, matrix.to_numpy(dtype=float))
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, x.shape[1]))
    for i in range(n_perm):
        perm = np.column_stack([rng.permutation(spalte) for spalte in x.T])
        null[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(perm, rowvar=False)))[::-1]
    return np.quantile(null, quantil, axis=0)


def dominant_construct(loadings: pd.DataFrame, component: str) -> str:
    """Welches Konstrukt überwiegt in einer Komponente?"""
    sub = loadings[["metric", component]].copy()
    sub["konstrukt"] = sub["metric"].map(lambda k: split_key(k)[1])
    sub["betrag"] = sub[component].abs()
    per = sub.groupby("konstrukt")["betrag"].mean().sort_values(ascending=False)
    if len(per) < 2 or per.iloc[0] < 1.5 * per.iloc[1]:
        return "gemischt"
    return str(per.index[0])


# Export


def write_discriminant_report(
    out_dir: Path,
    info: MatrixInfo,
    square: pd.DataFrame,
    table: pd.DataFrame,
    summary: pd.DataFrame,
    viol: pd.DataFrame,
    neg_conv: pd.DataFrame,
    varianz: pd.DataFrame,
    pca_variance: pd.DataFrame,
    pca_loadings: pd.DataFrame,
    stem: str = "discriminant",
    stichprobe: pd.DataFrame | None = None,
) -> tuple[Path, Path]:
    """Schreibt alle Tabellen in eine Excel-Datei und die Paare als CSV.

    Wie bei write_convergence_report entstehen alle Tabellen aus einem Lauf.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}_pairs.csv"
    xlsx_path = out_dir / f"{stem}.xlsx"

    table.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        lauf = info.as_frame()
        if stichprobe is not None:
            lauf = pd.concat([lauf, stichprobe], ignore_index=True)
        lauf.to_excel(writer, sheet_name="Lauf", index=False)
        summary.to_excel(writer, sheet_name="Bloecke", index=False)
        square.rename_axis("metric").reset_index().to_excel(
            writer, sheet_name="MTMM-Matrix", index=False
        )
        table.to_excel(writer, sheet_name="alle Paare", index=False)
        binaer = table[table["block"] == BINAER_BLOCK]
        if not binaer.empty:
            binaer.to_excel(writer, sheet_name="binaer gesondert", index=False)
        (
            viol if not viol.empty else pd.DataFrame([{"hinweis": "keine Verstoesse"}])
        ).to_excel(writer, sheet_name="Verstoesse", index=False)
        (
            neg_conv
            if not neg_conv.empty
            else pd.DataFrame([{"hinweis": "alle konvergenten rho >= 0"}])
        ).to_excel(writer, sheet_name="negative Konvergenz", index=False)
        varianz.to_excel(writer, sheet_name="Varianz je Metrik", index=False)
        pca_variance.to_excel(writer, sheet_name="PCA Varianz", index=False)
        pca_loadings.to_excel(writer, sheet_name="PCA Ladungen", index=False)

    return csv_path, xlsx_path
