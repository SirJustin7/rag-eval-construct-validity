"""Kriteriumsvalidität für RQ4 und binäre Maße in RQ2 und RQ3:

Zusammenhang zwischen einem abgestuften Score und zwei Gruppen (richtig oder falsch, TRACe adherence ja oder nein) als rang-biseriale Korrelation nach Cureton, in der Variante, die Bindungen an der Gruppengrenze berücksichtigt.

Definition (Cureton 1956, 1968)
Für jedes Paar (a aus der hohen Gruppe, b aus der niedrigen) zählt:
    P  Übereinstimmung, wenn score(a) > score(b)
    Q  Umkehrung,       wenn score(a) < score(b)
    Gleichstand zählt weder als P noch als Q.

    r = (P - Q) / P_max

P_max ist die größte Zahl an Übereinstimmungen, die mit diesen Scores möglich ist.
Dazu bekommt die hohe Gruppe die n_hoch höchsten Scores. Ohne Bindungen ist P_max = n_hoch · n_tief, und r entspricht der Formel von Glass (1966).
Liegt ein Gleichstand genau auf der Grenze zwischen den n_hoch höchsten und den übrigen Werten ("bracket tie"), ist P_max kleiner.
Bei Metriken mit vielen Werten auf 1,0 kommt das oft vor. Die einfache Formel würde den Zusammenhang dort unterschätzen und könnte ±1 nicht erreichen.

p-Wert: Mann-Whitney-U-Test mit Bindungskorrektur, zweiseitig, asymptotisch (von Cureton 1956 als Test für diesen Koeffizienten genannt).
Der exakte Test gilt bei Bindungen nicht.
P - Q = U_hoch - U_tief, der Test prüft also genau den Zähler.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class RangBiserial:
    r: float
    p: float
    n_hoch: int
    n_tief: int
    uebereinstimmungen: int  # P
    umkehrungen: int  # Q
    p_max: int  # größtes mögliches P bei diesen Bindungen

    @property
    def r_glass(self) -> float:
        """Vergleichswert ohne Bindungskorrektur: (P - Q) / (n_hoch · n_tief)."""
        n = self.n_hoch * self.n_tief
        return (self.uebereinstimmungen - self.umkehrungen) / n if n else float("nan")


def rang_biserial(scores, hoch) -> RangBiserial:
    """Rang-biseriale Korrelation nach Cureton (1968) mit Mann-Whitney-p.

    scores: abgestufte Werte. hoch: bool je Wert, True = hohe Gruppe (bei RQ4 Urteil richtig).
    Fehlende Scores werden ausgelassen.
    Positives r heißt, die hohe Gruppe bekommt die höheren Scores.
    """
    s = np.asarray(scores, dtype=float)
    h = np.asarray(hoch, dtype=bool)
    gueltig = ~np.isnan(s)
    s, h = s[gueltig], h[gueltig]
    a, b = s[h], s[~h]
    n_hoch, n_tief = len(a), len(b)
    if n_hoch == 0 or n_tief == 0:
        return RangBiserial(float("nan"), float("nan"), n_hoch, n_tief, 0, 0, 0)

    P = int((a[:, None] > b[None, :]).sum())
    Q = int((a[:, None] < b[None, :]).sum())

    absteigend = np.sort(s)[::-1]
    oben, unten = absteigend[:n_hoch], absteigend[n_hoch:]
    p_max = int((oben[:, None] > unten[None, :]).sum())

    r = (P - Q) / p_max if p_max else float("nan")
    if np.unique(s).size > 1:
        p = float(
            stats.mannwhitneyu(
                a, b, alternative="two-sided", method="asymptotic", use_continuity=True
            ).pvalue
        )
    else:
        p = float("nan")
    return RangBiserial(float(r), p, n_hoch, n_tief, P, Q, p_max)


def kriteriumstabelle(
    scores: pd.DataFrame, gruppen_spalte: str | None = "generator"
) -> pd.DataFrame:
    """Ein Koeffizient je Metrik, über alle Samples und je Gruppe.

    scores: eine Zeile je (sample_id, metric) mit Spalten
        sample_id, metric, score, richtig (bool) und optional gruppen_spalte.
    Rückgabe: je (metric, gruppe) r, r_glass, p, n, Anteil richtig.
    """
    zeilen = []
    gruppen = [("alle", scores)]
    if gruppen_spalte and gruppen_spalte in scores:
        gruppen += [(str(g), teil) for g, teil in scores.groupby(gruppen_spalte)]
    for name, teil in gruppen:
        for metric, m in teil.groupby("metric"):
            e = rang_biserial(m["score"], m["richtig"])
            n = e.n_hoch + e.n_tief
            zeilen.append(
                {
                    "metric": metric,
                    "gruppe": name,
                    "n": n,
                    "anteil_richtig": e.n_hoch / n if n else float("nan"),
                    "r_rb": e.r,
                    "r_glass": e.r_glass,
                    "p": e.p,
                    "P": e.uebereinstimmungen,
                    "Q": e.umkehrungen,
                    "P_max": e.p_max,
                    "distinkte_werte": int(m["score"].nunique()),
                }
            )
    return pd.DataFrame(zeilen)
