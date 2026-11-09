"""Vergleich der Manipulationsbedingungen mit dem unveränderten Lauf:

Jede Bedingung bewertet dieselben Antworten wie der Hauptlauf, nur verändert.
Verglichen wird deshalb paarweise, je Sample und Metrik der Wert nach der Manipulation gegen den Wert davor.
Jedes Sample ist so seine eigene Kontrolle.

Kennzahlen je Metrik und Bedingung:
* Median vorher und nachher und Median der Differenzen.
* Wilcoxon-Vorzeichen-Rang-Test über die Paare, zweiseitig. Er setzt keine Normalverteilung voraus, die Scores sind hier schief verteilt.
* Gepaarte rang-biseriale Effektstärke r = (S+ - S-) / S, mit S+ und S- als Summe der positiven bzw. negativen Ränge.
    r liegt in [-1, 1] und zeigt, wie einheitlich sich die Werte bewegen.
    Der p-Wert allein zeigt das nicht, bei n = 100 wird auch eine kleine Verschiebung signifikant.
* Zahl der gefallenen, gestiegenen und unveränderten Samples.

Getrennt nach Antworttyp: Viele Antworten des Hauptlaufs sind Abstinenzen.
Eine Abstinenz behauptet nichts und ist deshalb immer treu. Kommt dort eine ungedeckte Aussage dazu, sollte Faithfulness stärker fallen als bei einer Antwort, die schon vorher etwas behauptet hat.
vergleichstabelle weist die Typen deshalb auch getrennt aus.

Context Relevance wird in allen drei Frameworks nur aus Frage und Kontext berechnet (Ragas user_input/retrieved_contexts, TruLens question/context, DeepEval ContextualRelevancyMetric ohne actual_output).
Die Manipulation ändert nur die Antwort.
Dass Context Relevance sich nicht bewegt, folgt also aus der Konstruktion und ist kein Befund über die Metrik.
Der Unterschied zeigt nur die Streuung des Judge bei gleicher Eingabe.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

# Konstrukte, die eine Änderung der Antwort nicht erreichen.
UNERREICHBAR_GENERIERUNG = {"context_relevance"}


def _effektstaerke(vorher: pd.Series, nachher: pd.Series) -> tuple[float, float, int]:
    """Gepaarte rang-biseriale Effektstärke, p-Wert und Zahl der Paare mit Differenz.

    Rückgabe (r, p, n_effektiv). r ist positiv, wenn die Werte steigen.
    """
    # Vor dem Zählen und Rangieren auf 12 Stellen runden.
    # Die Skalen sind grob gestuft (TruLens 0 / 0,33 / 0,67 / 1), gleich große Differenzen kommen oft vor, im Gleitkomma aber z.B. als 0.05000000000000004 und -0.049999999999999996.
    # Ohne Rundung bekämen sie verschiedene Ränge.
    d = (nachher - vorher).dropna().round(12)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), 0
    rang = stats.rankdata(d.abs())
    s_plus = rang[d > 0].sum()
    s_minus = rang[d < 0].sum()
    r = (s_plus - s_minus) / rang.sum()
    if n < 6:
        # Zu wenige Paare für den Test, die Effektstärke bleibt.
        return r, float("nan"), n
    p = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue
    return r, float(p), n


def vergleichstabelle(
    vorher: pd.DataFrame,
    nachher: pd.DataFrame,
    bedingung: str,
    seite: str = "generierung",
    gruppen: pd.Series | None = None,
) -> pd.DataFrame:
    """Eine Zeile je Metrik (und Gruppe) mit dem gepaarten Vergleich.

    vorher, nachher:    eine Zeile je sample_id und metric mit Spalte score.
    gruppen:            optional eine Series sample_id -> Gruppe (z.B. Antworttyp im Hauptlauf), dann zusätzlich je Gruppe.
    """
    v = vorher.set_index(["sample_id", "metric"])["score"]
    n_ = nachher.set_index(["sample_id", "metric"])["score"]
    gemeinsam = v.index.intersection(n_.index)
    v, n_ = v.loc[gemeinsam], n_.loc[gemeinsam]

    paare = pd.DataFrame({"vorher": v, "nachher": n_}).reset_index()
    if gruppen is not None:
        paare["gruppe"] = paare["sample_id"].map(gruppen)

    zeilen = []
    aufteilungen = [("alle", paare)]
    if gruppen is not None:
        aufteilungen += [(str(g), teil) for g, teil in paare.groupby("gruppe")]

    for gname, teil in aufteilungen:
        for metrik, m in teil.groupby("metric"):
            konstrukt = metrik.split(".", 1)[1] if "." in metrik else metrik
            r, p, n_eff = _effektstaerke(m["vorher"], m["nachher"])
            # Wie in _effektstaerke auf 12 Stellen runden, sonst zählt ein Gleitkomma-Rest (0.30000000000000004 gegen 0.3) als Bewegung.
            d = (m["nachher"] - m["vorher"]).round(12)
            zeilen.append(
                {
                    "bedingung": bedingung,
                    "metrik": metrik,
                    "gruppe": gname,
                    "n": len(m),
                    "median_vorher": m["vorher"].median(),
                    "median_nachher": m["nachher"].median(),
                    "median_differenz": d.median(),
                    "r_paarweise": r,
                    "p": p,
                    "n_gefallen": int((d < 0).sum()),
                    "n_gestiegen": int((d > 0).sum()),
                    "n_unveraendert": int((d == 0).sum()),
                    # Context Relevance wird von einer Änderung der Antwort nicht erreicht, hier zeigt der Vergleich nur die Streuung des Judge.
                    "rolle": (
                        "retest"
                        if seite == "generierung"
                        and konstrukt in UNERREICHBAR_GENERIERUNG
                        else "befund"
                    ),
                }
            )
    return pd.DataFrame(zeilen)


def erwartungsabgleich(
    tabelle: pd.DataFrame, erwartung: dict[str, dict[str, str]], schwelle: float = 0.10
) -> pd.DataFrame:
    """Stellt jeder Zeile die vorab festgelegte Erwartung gegenüber.

    schwelle ist der Betrag der Effektstärke, ab dem eine Bewegung eine Richtung hat. Darunter heißt das Ergebnis "keine Bewegung".
    """
    aus = []
    for _, z in tabelle.iterrows():
        konstrukt = z["metrik"].split(".", 1)[1] if "." in z["metrik"] else z["metrik"]
        erw = (erwartung.get(z["bedingung"]) or {}).get(konstrukt, "")
        if pd.isna(z["r_paarweise"]) or abs(z["r_paarweise"]) < schwelle:
            richtung = "keine Bewegung"
        else:
            richtung = "niedriger" if z["r_paarweise"] < 0 else "höher"
        aus.append({**z, "erwartet": erw, "beobachtet": richtung})
    return pd.DataFrame(aus)
