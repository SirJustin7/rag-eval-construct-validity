"""Alle Ergebnisse in einer Excel-Datei (results/Auswertung_gesamt.xlsx):

Liest nur, was scripts/04, 06, 08, 09 und 10 geschrieben haben, und rechnet nichts neu.
Nur die Übersicht der Stichproben wird direkt aus den Antwortdateien gezählt.
Ein Blatt je Analyse, Datensatz, Tiefe und Variante, vorne eine Übersicht mit Links und dem Status jeder Quelldatei.
Fehlende Quellen stehen dort als "fehlt".

Aufruf:
    python scripts/11_gesamt_excel.py
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.drawing.image import Image
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from rag_eval.analysis.abstinenz import Antworttyp
from rag_eval.analysis.stichprobe import antworttypen_aus_datei
from rag_eval.config import load_config
from rag_eval.naming import _sauber, generated_datei

# (Teil des Dateinamens, Anzeigename, Datensatz, top_k)
LAEUFE = [
    ("crag_k10", "CRAG k10", "crag", 10),
    ("crag_k5", "CRAG k5", "crag", 5),
    ("ragbench_k10", "RAGBench k10", "ragbench", 10),
    ("ragbench_k5", "RAGBench k5", "ragbench", 5),
    ("ragbench_own", "TRACe-Tripel", "ragbench_own", None),
]
VARIANTEN = [("mit_abst", "mit Abst."), ("ohne_abst", "ohne Abst.")]

# Reihenfolge und Überschriften der Tabellen auf einem RQ3-Blatt.
RQ3_TEILE = [
    ("Lauf", "Lauf und Stichprobe"),
    ("Varianz je Metrik", "Sättigung je Metrik (vorregistrierte Ausschlussregel)"),
    ("Bloecke", "Blöcke (|ρ|, ohne ausgeschlossene Metriken und ohne binäre Paare)"),
    ("MTMM-Matrix", "MTMM-Matrix (Spearman ρ; bei TRACe-adherence rang-biserial r)"),
    (
        "binaer gesondert",
        "Binäre Metrik gesondert (rang-biserial nach Cureton, Mann-Whitney-p)",
    ),
    ("negative Konvergenz", "Konvergente Paare mit negativem ρ"),
    ("Verstoesse", "Verstöße gegen die erwartete Ordnung"),
    ("PCA Varianz", "PCA: Eigenwerte und Parallelanalyse"),
    ("PCA Ladungen", "PCA: Ladungen"),
    ("alle Paare", "Alle Paare"),
]

TITEL = Font(bold=True, size=12)
KOPF = Font(bold=True)
KOPF_FUELLUNG = PatternFill("solid", fgColor="DDE4EE")


def schreibe_tabelle(ws, df: pd.DataFrame, zeile: int, titel: str | None = None) -> int:
    """Schreibt df ab zeile, optional mit Überschrift, und gibt die nächste freie Zeile zurück."""
    if titel:
        ws.cell(zeile, 1, titel).font = TITEL
        zeile += 1
    for j, spalte in enumerate(df.columns, 1):
        c = ws.cell(zeile, j, str(spalte))
        c.font, c.fill = KOPF, KOPF_FUELLUNG
    for i, reihe in enumerate(df.itertuples(index=False), zeile + 1):
        for j, wert in enumerate(reihe, 1):
            if isinstance(wert, float) and pd.isna(wert):
                continue
            c = ws.cell(i, j, wert.item() if hasattr(wert, "item") else wert)
            if isinstance(c.value, float):
                name = str(df.columns[j - 1])
                c.number_format = (
                    "0.00E+00" if name.endswith("_p") or name == "p" else "0.000"
                )
    return zeile + len(df) + 2


def spaltenbreiten(ws, max_breite: int = 45) -> None:
    breiten: dict[int, int] = {}
    for reihe in ws.iter_rows():
        for c in reihe:
            if c.value is not None and not (c.font and c.font.size == 12):
                breiten[c.column] = max(breiten.get(c.column, 0), len(str(c.value)))
    for spalte, b in breiten.items():
        ws.column_dimensions[get_column_letter(spalte)].width = min(
            max(b + 2, 8), max_breite
        )


def stichproben(cfg) -> pd.DataFrame:
    zeilen = []
    for _, anzeige, ds, k in LAEUFE:
        pfad = generated_datei(ds, "per_domain", k or 0, cfg["paths"]["processed"], cfg)
        if not Path(pfad).exists():
            continue
        typen = antworttypen_aus_datei(pfad)
        z = {t: sum(1 for v in typen.values() if v is t) for t in Antworttyp}
        zeilen.append(
            {
                "lauf": anzeige,
                "antwortdatei": Path(pfad).name,
                "fragen": len(typen),
                "leer": z[Antworttyp.LEER],
                "abstinenz": z[Antworttyp.ABSTINENZ],
                "teilantwort": z[Antworttyp.TEILANTWORT],
                "antwort": z[Antworttyp.ANTWORT],
                "anteil_abstinenz": z[Antworttyp.ABSTINENZ] / len(typen),
                "n_mit_abst": len(typen) - z[Antworttyp.LEER],
                "n_ohne_abst": len(typen)
                - z[Antworttyp.LEER]
                - z[Antworttyp.ABSTINENZ],
            }
        )
    return pd.DataFrame(zeilen)


def main() -> None:
    cfg = load_config()
    res = Path(cfg["paths"]["results"])
    rq2, rq3, rq4 = res / "rq2", res / "rq3", res / "rq4"
    judge = _sauber(cfg["judge"]["model"])

    wb = Workbook()
    uebersicht = wb.active
    uebersicht.title = "Übersicht"
    eintraege: list[dict] = []  # Blatt, Inhalt, Quelle, Status

    def eintrag(blatt, inhalt, quelle: Path | None, vorhanden: bool):
        eintraege.append(
            {
                "blatt": blatt if vorhanden else "",
                "inhalt": inhalt,
                "quelle": str(quelle.relative_to(res.parent)) if quelle else "",
                "status": "ok" if vorhanden else "fehlt",
            }
        )

    # Stichproben
    ws = wb.create_sheet("Stichproben")
    schreibe_tabelle(
        ws,
        stichproben(cfg),
        1,
        "Antworttypen je Lauf (Regel nur auf dem Antworttext; leere Antworten fallen immer heraus)",
    )
    spaltenbreiten(ws)
    eintrag("Stichproben", "Leere Antworten, Abstinenzen und n je Lauf", None, True)

    # RQ2
    for datei, anzeige, _, _ in LAEUFE:
        for var, var_anzeige in VARIANTEN:
            blatt = f"RQ2 {anzeige} {var_anzeige}"
            quelle = rq2 / f"convergence_{datei}_{var}.csv"
            if not quelle.exists():
                eintrag(blatt, "RQ2 Konvergenz", quelle, False)
                continue
            ws = wb.create_sheet(blatt)
            df = pd.read_csv(quelle)
            schreibe_tabelle(ws, df, 1, f"RQ2 Konvergenz · {anzeige} · {var_anzeige}")
            spaltenbreiten(ws)
            ws.freeze_panes = "A3"
            eintrag(
                blatt, "RQ2 Konvergenz je Konstrukt und Framework-Paar", quelle, True
            )

    # RQ3
    for datei, anzeige, _, _ in LAEUFE:
        for var, var_anzeige in VARIANTEN:
            blatt = f"RQ3 {anzeige} {var_anzeige}"
            quelle = rq3 / f"discriminant_{datei}_{var}.xlsx"
            if not quelle.exists():
                eintrag(blatt, "RQ3 MTMM und PCA", quelle, False)
                continue
            ws = wb.create_sheet(blatt)
            ws.cell(
                1, 1, f"RQ3 Diskriminante Validität · {anzeige} · {var_anzeige}"
            ).font = Font(bold=True, size=14)
            teile = pd.read_excel(quelle, sheet_name=None)
            zeile = 3
            for name, titel in RQ3_TEILE:
                if name in teile:
                    zeile = schreibe_tabelle(ws, teile[name], zeile, titel)
            spaltenbreiten(ws)
            bild = rq3 / f"discriminant_{datei}_{var}_mtmm_heatmap.png"
            if bild.exists():
                img = Image(str(bild))
                faktor = 560 / img.width
                img.width, img.height = (
                    int(img.width * faktor),
                    int(img.height * faktor),
                )
                ws.add_image(img, f"{get_column_letter(ws.max_column + 2)}3")
            eintrag(
                blatt, "RQ3 MTMM-Matrix, Blöcke, Verstöße, PCA, Heatmap", quelle, True
            )

    # Einzelne Blätter
    einzel = [
        (
            "TRACe Plausibilität",
            rq2 / f"plausibilitaet_ragbench_own_{judge}.csv",
            "Eigene Ragas/TruLens-Scores gegen RAGBenchs mitgelieferte, gepoolt und je Subset",
        ),
        (
            "RQ4 Kriterium",
            rq4 / f"kriterium_{judge}.csv",
            "Rang-biserial Score ↔ Korrektheit (Simon et al.), gepoolt und je Generator",
        ),
    ]
    diagonalen = sorted(rq3.glob("judge_diagonale_*.csv"))
    for d in diagonalen:
        name = "Judge-Diagonale " + d.stem.removeprefix("judge_diagonale_").replace(
            "ragbench", "RAGBench"
        ).replace("crag", "CRAG").replace("_", " ")
        einzel.append(
            (
                name[:31],
                d,
                "Haupt-Judge gegen weitere Judges je Metrik (Spalte anteil_bewertet = Fortschritt)",
            )
        )
    if not diagonalen:
        einzel.append(
            (
                "Judge-Diagonale",
                rq4 / "judge_diagonale_ragbench_k10.csv",
                "Haupt-Judge gegen weitere Judges je Metrik",
            )
        )
    for blatt, quelle, inhalt in einzel:
        if not quelle.exists():
            eintrag(blatt, inhalt, quelle, False)
            continue
        ws = wb.create_sheet(blatt)
        schreibe_tabelle(ws, pd.read_csv(quelle), 1, blatt)
        spaltenbreiten(ws)
        ws.freeze_panes = "A3"
        eintrag(blatt, inhalt, quelle, True)

    # Übersicht
    uebersicht.cell(1, 1, "Auswertung gesamt").font = Font(bold=True, size=14)
    uebersicht.cell(
        2,
        1,
        f"Erstellt {datetime.now():%d.%m.%Y %H:%M} · Haupt-Judge {cfg['judge']['model']} "
        f"· erzeugt von scripts/11_gesamt_excel.py",
    )
    df = pd.DataFrame(eintraege)
    schreibe_tabelle(uebersicht, df, 4)
    for i, e in enumerate(eintraege, 5):
        if e["blatt"]:
            c = uebersicht.cell(i, 1)
            c.hyperlink = f"#'{e['blatt']}'!A1"
            c.font = Font(color="1F4E9A", underline="single")
        if e["status"] == "fehlt":
            uebersicht.cell(i, 4).font = Font(bold=True, color="B00020")
    spaltenbreiten(uebersicht, max_breite=80)
    for zeile in uebersicht.iter_rows():
        for c in zeile:
            c.alignment = Alignment(vertical="top")

    ziel = res / "Auswertung_gesamt.xlsx"
    wb.save(ziel)
    fehlend = [e for e in eintraege if e["status"] == "fehlt"]
    print(f"{len(wb.sheetnames)} Blätter -> {ziel}")
    for e in fehlend:
        print(f"  fehlt: {e['inhalt']} ({e['quelle']})")


if __name__ == "__main__":
    main()
