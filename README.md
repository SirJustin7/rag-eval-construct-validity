# Construct validity of automated evaluation frameworks for RAG systems

This repository contains the code and data for the bachelor's thesis of the same name (Leipzig University, 2026). We compare how four evaluation frameworks, Ragas, TruLens, DeepEval and TRACe, compute the three constructs of the RAG triad, faithfulness, context relevance and answer relevance. All frameworks receive identical triples of question, retrieved context and generated answer, and the three libraries share one judge. We assess convergent and discriminant validity in a multitrait-multimethod design, repeat the scoring with two further judges, test the metrics against deliberately inserted errors and relate them to the documented correctness of answers from a separate dataset.

| | Research question | Data |
|---|---|---|
| RQ1: Operationalization | How do established RAG evaluation frameworks operationalize the constructs of the RAG triad, and in which respects do their operationalizations differ? | publications and documentation, source code, release notes |
| RQ2: Convergent Validity | To what extent do the metrics of different frameworks converge when they claim to measure the same construct? | CRAG, RAGBench, RAGBench's own triples |
| RQ3: Discriminant Validity | To what extent do metrics that claim to measure different constructs remain empirically distinguishable? | the scores of RQ2 in multitrait-multimethod matrices, manipulated versions of 100 CRAG answers |
| RQ4: Criterion Validity | To what extent can automated evaluation metrics serve as a proxy for whether a RAG system solves its task? | replication package of Simon et al. (2024) |

## Repository structure

```
config/config.yaml   all parameters of the experiments
src/rag_eval/        package: RAG system, framework adapters, data loaders, analysis
scripts/             numbered steps of the experiments
data/processed/      the answers that were scored
results/scores/      framework scores, one line per sample, framework and construct
results/rq2|rq3|rq4/ analysis outputs
results/Auswertung_gesamt.xlsx   all analysis outputs in one workbook
```

## Scripts and thesis sections

| Step | Script | Thesis |
|---|---|---|
| Build the vector index | `01_build_index.py` | 4.2.1 |
| Generate answers | `02_generate_answers.py` | 4.2.1 |
| Import the triples of Simon et al. | `02b_import_depvalidation.py` | 4.5 |
| Import RAGBench's own triples | `02c_import_ragbench_own.py` | 4.2.3 |
| Score with the three libraries | `03_run_frameworks.py` | 4.2.2 |
| Convergent validity (RQ2) | `04_rq2_convergence.py` | 4.3, 5.3 |
| Further tests of faithfulness | `13_faithfulness_robustheit.py` | 5.3.1 |
| Discriminant validity (RQ3) | `06_rq3_discriminant.py` | 4.4, 5.4 |
| Agreement between the main judge and two further judges | `10_judge_diagonale.py` | 4.4, 5.4.1, Table A.3 |
| Plausibility check against RAGBench's own scores | `09_trace_plausibilitaet.py` | 4.2.3, 5.4.1, Table A.4 |
| Create the manipulated versions | `07_manipulate.py` | 4.4 |
| Compare the manipulated versions with the original answers | `12_manipulation_auswerten.py` | 4.4, 5.4.4 |
| Criterion validity (RQ4) | `08_rq4_criterion.py` | 4.5, 5.5 |
| Answer relevance on the rationale alone | `14_deepeval_ar_ohne_urteilssatz.py`, `15_ragas_trulens_ar_ohne_urteilssatz.py` | 4.5, 5.5, Table A.7 |
| Collect all outputs in one workbook | `11_gesamt_excel.py` | |
| Cost projection before paid runs | `00_kostenhochrechnung.py` | |

RQ1 is answered by an analysis of the frameworks themselves (Sections 4.1 and 5.1). The adapters in `src/rag_eval/evaluation/` show which native metric stands for which construct.

The analysis rules are implemented in `src/rag_eval/analysis/`: abstentions and partial answers in `abstinenz.py` and `stichprobe.py` (4.2.4), the exclusion rule with the untied pair proportion in `saturation.py` (4.2.5).

## Setup

All experiments ran under Python 3.12 on macOS (Intel), with Ragas 0.4.3, DeepEval 2.9.3 and TruLens 1.5.3 (trulens-core 1.4.5). The versions of all dependencies are recorded in `requirements.lock.txt`. They are pinned to torch 2.2.2, the last release with wheels for Intel Macs.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
cp .env.example .env
```

API keys are only needed for steps that call an LLM (see below).

## Reproducing the analyses

All analyses run on the included scores and do not call any LLM.

```bash
# RQ2 and RQ3
python scripts/04_rq2_convergence.py --datasets crag ragbench --top-k 10
python scripts/04_rq2_convergence.py --datasets crag ragbench --top-k 5
python scripts/04_rq2_convergence.py --datasets ragbench_own
python scripts/06_rq3_discriminant.py --datasets crag ragbench --top-k 10
python scripts/06_rq3_discriminant.py --datasets crag ragbench --top-k 5
python scripts/06_rq3_discriminant.py --datasets ragbench_own
python scripts/09_trace_plausibilitaet.py      # downloads RAGBench from HuggingFace
python scripts/10_judge_diagonale.py
python scripts/12_manipulation_auswerten.py
python scripts/13_faithfulness_robustheit.py

# RQ4: the triples are not included, they are rebuilt from the replication package
git clone https://github.com/simisimon/dependency-validation-rag.git data/raw/dependency-validation-rag
python scripts/02b_import_depvalidation.py
python scripts/08_rq4_criterion.py

python scripts/11_gesamt_excel.py
```

The outputs overwrite the files in `results/rq2`, `results/rq3` and `results/rq4`. They reproduce the included files exactly. The Excel files differ only in their creation timestamp.

## Running the full pipeline

Steps 01 and 02 require the raw datasets (downloaded via HuggingFace, see `config/config.yaml`) and a generator endpoint. The generator is `google/gemma-4-26B-A4B-it`, accessed through SCADS.AI. Step 03 and scripts 14 and 15 call the judge (`gpt-4o-mini`) and incur API costs. Paid runs are logged locally in `results/usage.jsonl`, which is not included. The pilot run described in Section 4.2.2 is not part of this repository.

```bash
python scripts/01_build_index.py
python scripts/02_generate_answers.py
python scripts/03_run_frameworks.py
```

The retrieval depth of a run is set by `rag.retrieval.top_k` in `config/config.yaml` (5 or 10). The other runs of the study are scored separately:

```bash
python scripts/02c_import_ragbench_own.py
python scripts/03_run_frameworks.py --dataset ragbench_own

python scripts/07_manipulate.py --dataset crag --sample-size 100
python scripts/03_run_frameworks.py --dataset crag_inject_1      # likewise for the other versions

python scripts/03_run_frameworks.py --dataset ragbench --judge-model <model> --judge-endpoint generator

python scripts/03_run_frameworks.py --dataset depvalidation
```

## File names

Every parameter that varies between runs is part of the file name, so that runs cannot overwrite each other:

```
data/processed/{dataset}_generated_{scope}_k{top_k}.jsonl
results/scores/{dataset}_{framework}_{scope}_k{top_k}_{judge}.jsonl
```

`scope` is the index layout and `top_k` the retrieval depth. The main runs use the per-domain layout. RAGBench's own triples, the manipulated versions and the triples of Simon et al. do not go through retrieval and carry neither scope nor top_k (e.g. `depvalidation_ragas_gpt-4o-mini.jsonl`). The naming is implemented in `src/rag_eval/naming.py`.

The names in the analysis outputs correspond to the terms of the thesis as follows:

| File name | Thesis |
|---|---|
| `crag_k10`, `crag_k5`, `ragbench_k10`, `ragbench_k5` | CRAG 10, CRAG 5, RAGBench 10, RAGBench 5 |
| `ragbench_own` | RAGBench's own triples |
| `mit_abst` | full sample |
| `ohne_abst` | answered questions |
| `inject_1`, `inject_2`, `inject_3` | one, two or three added claims |
| `replace_full` | replaced answer |
| `swap_fact` | changed number |
| `depvalidation` | criterion data |

## Data

- CRAG (Yang et al., 2024), CC BY-NC 4.0, https://github.com/facebookresearch/CRAG
- RAGBench (Friel et al., 2024), CC BY 4.0, https://huggingface.co/datasets/rungalileo/ragbench
- Replication package of Simon et al. (2024), https://github.com/simisimon/dependency-validation-rag
