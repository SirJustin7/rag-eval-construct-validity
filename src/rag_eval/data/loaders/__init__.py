"""Loader für die Datensätze:

Jeder Loader bildet seinen Datensatz auf RAGSample ab, der Rest der Pipeline muss den Datensatz dann nicht mehr kennen.
"""

from .crag import load_crag
from .depvalidation import nicht_ueber_pipeline as depvalidation_nicht_ueber_pipeline
from .ragbench import load_ragbench
from .ragbench_own import nicht_ueber_pipeline as ragbench_own_nicht_ueber_pipeline

LOADERS = {
    "crag": load_crag,
    "ragbench": load_ragbench,
    # depvalidation (Simon et al. 2024) bringt fertige Tripel mit und wird von scripts/02b_import_depvalidation.py direkt nach data/processed/ geschrieben.
    # Der Eintrag ist nur da, damit scripts/03 den Datensatz annimmt. 01 und 02 brechen mit einer Meldung ab.
    "depvalidation": depvalidation_nicht_ueber_pipeline,
    # RAGBench noch einmal, aber mit der Antwort aus RAGBench statt der des eigenen Generators.
    # So bewerten die Frameworks und die TRACe-Labels denselben Text. Import über scripts/02c_import_ragbench_own.py.
    "ragbench_own": ragbench_own_nicht_ueber_pipeline,
}

# Die Manipulationsbedingungen. Sie entstehen mit scripts/07_manipulate.py aus einem fertigen Lauf, ohne Indexierung und Generierung.
# Der Eintrag ist nur da, damit scripts/03 z.B. --dataset crag_inject_1 annimmt.
MANIPULATIONEN = ["inject_1", "inject_2", "inject_3", "replace_full", "swap_fact"]


def _nicht_ueber_pipeline(name: str):
    def platzhalter(**_kwargs) -> None:
        raise NotImplementedError(
            f"{name} ist eine Manipulationsbedingung und wird nicht indexiert "
            f"oder neu generiert.\n"
            f"Stattdessen:  python scripts/07_manipulate.py --dataset crag "
            f"--bedingungen {name.split('_', 1)[1]}\n"
            f"Danach:       python scripts/03_run_frameworks.py --dataset {name}"
        )

    return platzhalter


for _basis in ("crag", "ragbench"):
    for _bed in MANIPULATIONEN:
        LOADERS[f"{_basis}_{_bed}"] = _nicht_ueber_pipeline(f"{_basis}_{_bed}")
