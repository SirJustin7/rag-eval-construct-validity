"""Sättigung: Welche Metrik hat genug Streuung für eine Rangkorrelation?:

Liegen fast alle Werte einer Metrik auf demselben Punkt, kann sie in keiner Korrelation etwas zeigen, weder Konvergenz noch Diskriminanz.
Liegen z.B. 20 von 21 Werten auf 1,0, hängt jedes ρ an dem einen Sample, das woanders liegt, obwohl n = 21 angegeben wird.

Maß ist der Anteil der Paare, die sich auf der Metrik unterscheiden:
    u = 1 - Σ pᵢ²
über die relativen Häufigkeiten pᵢ der verschiedenen Werte.
Das ist der Anteil ungebundener Paare, also die Größe, die bei Kendalls τ-b im Nenner steht.

Anders als die Standardabweichung hängt u nicht von der Skala ab.
TruLens kennt nur wenige Stufen, DeepEval gibt Anteile mit vielen Nachkommastellen zurück.
Eine Schwelle für die Standardabweichung würde die Frameworks deshalb unterschiedlich streng behandeln.

Die Schwellen wurden vor den Läufen festgelegt.
"""

from __future__ import annotations

from enum import Enum

import pandas as pd

# Unter diesem Wert gilt eine Metrik als gesättigt. Bei u < 0,20 liegen ungefähr 89 % oder mehr der Werte auf dem häufigsten Wert.
U_AUSGESCHLOSSEN = 0.20

# Bis zu diesem Wert wird berichtet, aber gekennzeichnet.
U_EINGESCHRAENKT = 0.40

# Zwei Werte sind keine Rangordnung, sondern zwei Gruppen. Spearman ρ ist dann durch die Verteilung auf die Gruppen nach oben begrenzt, egal wie gut die Frameworks übereinstimmen.
MIN_DISTINKTE_WERTE = 3


class Stufe(str, Enum):
    """Auswertbarkeit einer Metrik, von gut nach schlecht."""

    AUSWERTBAR = "auswertbar"
    EINGESCHRAENKT = "eingeschraenkt"
    AUSGESCHLOSSEN = "ausgeschlossen"


def ungebundener_paaranteil(werte: pd.Series) -> float:
    """u = 1 - Σ pᵢ², der Anteil der Paare, die diese Metrik unterscheidet.

    Entspricht dem Gini-Simpson-Index. Für n < 2 nicht definiert (NaN).
    """
    werte = pd.Series(werte).dropna()
    n = len(werte)
    if n < 2:
        return float("nan")
    p = werte.value_counts() / n
    return float(1.0 - (p**2).sum())


def einstufen(u: float, distinkte_werte: int) -> Stufe:
    """Ordnet eine Metrik nach den festgelegten Schwellen ein.

    Die Regel fragt nur nach der Streuung der Metrik, nicht nach den Korrelationen.
    """
    if pd.isna(u):
        return Stufe.AUSGESCHLOSSEN
    if u < U_AUSGESCHLOSSEN or distinkte_werte < MIN_DISTINKTE_WERTE:
        return Stufe.AUSGESCHLOSSEN
    if u < U_EINGESCHRAENKT:
        return Stufe.EINGESCHRAENKT
    return Stufe.AUSWERTBAR


def saettigungsreport(matrix: pd.DataFrame) -> pd.DataFrame:
    """Eine Zeile je Metrik (Spalte der Score-Matrix), sortiert nach u.

    n_ausserhalb_modus zählt die Werte, die nicht auf dem häufigsten Wert liegen. Auf ihnen beruht ein ρ tatsächlich.
    """
    zeilen = []
    for metrik in matrix.columns:
        spalte = matrix[metrik].dropna()
        counts = spalte.value_counts()
        n = len(spalte)
        modus = counts.index[0] if len(counts) else float("nan")
        anteil_modus = counts.iloc[0] / n if n else float("nan")
        u = ungebundener_paaranteil(spalte)
        distinkt = int(spalte.nunique())
        zeilen.append(
            {
                "metrik": metrik,
                "n": n,
                "distinkte_werte": distinkt,
                "modus": modus,
                "anteil_am_modus": anteil_modus,
                "n_ausserhalb_modus": int(n - counts.iloc[0]) if n else 0,
                "ungebundener_paaranteil": u,
                "std": float(spalte.std()) if n > 1 else float("nan"),
                "stufe": einstufen(u, distinkt).value,
            }
        )
    if not zeilen:
        return pd.DataFrame(
            columns=[
                "metrik",
                "n",
                "distinkte_werte",
                "modus",
                "anteil_am_modus",
                "n_ausserhalb_modus",
                "ungebundener_paaranteil",
                "std",
                "stufe",
            ]
        )
    return (
        pd.DataFrame(zeilen)
        .sort_values("ungebundener_paaranteil")
        .reset_index(drop=True)
    )


def ausgeschlossene(report: pd.DataFrame) -> list[str]:
    """Metriken, auf die sich keine Aussage zur Validität stützen kann."""
    return report.loc[report["stufe"] == Stufe.AUSGESCHLOSSEN.value, "metrik"].tolist()


def eingeschraenkte(report: pd.DataFrame) -> list[str]:
    """Metriken, die nur mit Kennzeichnung berichtet werden."""
    return report.loc[report["stufe"] == Stufe.EINGESCHRAENKT.value, "metrik"].tolist()


def formatiere(report: pd.DataFrame) -> str:
    """Konsolenausgabe für die Auswertungsskripte."""
    if report.empty:
        return "  (keine Metriken)"
    zeilen = []
    symbol = {
        Stufe.AUSGESCHLOSSEN.value: "AUS ",
        Stufe.EINGESCHRAENKT.value: "EING",
        Stufe.AUSWERTBAR.value: "ok  ",
    }
    for r in report.itertuples():
        zeilen.append(
            f"  {symbol[r.stufe]} {r.metrik:34s} u={r.ungebundener_paaranteil:.3f}  "
            f"{r.distinkte_werte:3d} Werte  {r.anteil_am_modus:5.1%} auf {r.modus:g}  "
            f"n außerhalb Modus={r.n_ausserhalb_modus}"
        )
    return "\n".join(zeilen)
