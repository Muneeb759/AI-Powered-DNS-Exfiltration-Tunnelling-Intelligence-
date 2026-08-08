# AI-Powered DNS Exfiltration & Tunnelling Intelligence

Detection of simulated DNS data exfiltration in **CIC-Bell-DNS-EXF-2021**, built for the Sofstica
AI Hackathon (August 2026).

**Declared primary track: Track 1 — Detection & Alert Prioritization.**
Track 2 (tiered cascade) and Track 3 (analyst case view) are included as *supporting* components
of that submission, not as separate claims — see [`report/TECHNICAL_REPORT.md`](report/TECHNICAL_REPORT.md) §1.

Team: Muneeb (data, splits, leakage audit, latency harness, reproducibility) ·
Serena (baselines, modelling, calibration, cascade, evaluation, SHAP, UI, report).

> **Advisory only.** This is an offline, defensive prototype. It does not block DNS, quarantine
> hosts, or modify firewall rules. Every alert requires a human disposition step.

---

## Headline result

v1 stateless row-level detector, 14 leakage-gated features, threshold selected on `val_thr`:

| | PR-AUC | Recall | Precision | FPR | False alerts / 10k benign |
|---|---|---|---|---|---|
| Combined | **0.6269** | 10.78% | 62.6% | 4.12% | 411.7 |
| Light | 0.1283 | 10.64% | — | 4.12% | 411.7 |
| Heavy | 0.6052 | 10.80% | — | 4.12% | 411.7 |

A higher-recall operating point exists at 47.06% recall / 17.95% FPR; there is nothing in between
(score quantization leaves a 15-point FPR band unreachable). Both are reported — §7.

**Read [§9 first](report/TECHNICAL_REPORT.md) if you read only one section.** It documents three
experiments that produced strong numbers for the same structural reason and were rejected or
demoted as a result — including a PR-AUC 0.99999 model we withdrew.

## Setup

**Python 3.11 or 3.12. Built and verified on 3.12.5.**
The floor is set by `numpy==2.3.2`, which requires `>=3.11` — Python 3.10 or earlier will fail to
resolve. Python 3.13 is untested here: the pins may not have wheels for it, in which case pip
would attempt a source build. If you are on 3.13 and hit a build error, use 3.12.

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

That is the **core** set — everything needed to reproduce the reported score and run the tests
(~400 MB installed, dominated by scipy/pyarrow/pandas wheels).

The Streamlit analyst view and SHAP explanations are **not** required to verify results and are
kept in a separate file, because they roughly double the dependency tree (`shap` alone pulls
numba/llvmlite, ~110 MB). Install only if you want the demo UI:

```bash
pip install -r requirements-demo.txt
```

### Obtain the dataset

**This repository does not redistribute the dataset.** Download CIC-Bell-DNS-EXF-2021 from the
[official UNB page](https://www.unb.ca/cic/datasets/dns-exf-2021.html) and extract it into
`data/raw/` so the tree looks exactly like this (7 directories, 36 CSVs):

```
data/raw/
├── Benign/                              # 12 files: stateless_features-benign_*.pcap.csv
│                                        #           stateful_features-benign_*.pcap.csv
├── Attack_heavy_Benign/
│   ├── Attacks/                         # stateless_features-heavy_*.pcap.csv  (+ stateful_)
│   └── Benign/                          # stateless_features-benign_heavy_*.pcap.csv (+ stateful_)
└── Attack_Light_Benign/
    ├── Attacks/                         # stateless_features-light_*.pcap.csv  (+ stateful_)
    └── Benign/                          # stateless_features-benign_light_*.pcap.csv (+ stateful_)
```

Check it resolved correctly — this must print `36`:

```bash
find data/raw -name "*.csv" | wc -l
```

36 CSV files, 103,384,358 bytes total. Per-file SHA-256 checksums are recorded in
[`results/metrics/provenance.json`](results/metrics/provenance.json); provenance, licence, and
citation requirements are in
[`deliverables/DATA_AND_MODEL_STATEMENT.md`](deliverables/DATA_AND_MODEL_STATEMENT.md).

## Reproduce the reported score

One command. It verifies the committed model against the numbers published in the technical
report and **exits non-zero if they do not match**:

```bash
python scripts/reproduce.py
```

Expected output ends with `PASS: reproduced values match report/TECHNICAL_REPORT.md section 7.`

**What this does:** loads the *committed* model artifact `models/stateless_lgbm_v1.pkl` (tracked in
git — no retraining, so the result does not depend on your thread count or hardware) and scores
the test partition at the published threshold. On a fresh clone it first builds
`data/splits/*.parquet` from `data/raw/`, which takes about a minute and is deterministic
(seed `20260808`, sorted grouping). Splits are git-ignored, which is why the dataset is required
even though the model is not retrained.

To rebuild splits and retrain from raw data first (slower, fully from scratch):

```bash
python scripts/reproduce.py --retrain
```

| Task | Command | `make` equivalent |
|---|---|---|
| Reproduce headline score | `python scripts/reproduce.py` | `make reproduce` |
| Reproduce from scratch | `python scripts/reproduce.py --retrain` | `make reproduce-full` |
| Run tests (8) | `python -m pytest` | `make test` |
| Phase 1 data pipeline | see `Makefile` | `make phase1` |
| Streamlit analyst view | `streamlit run app/streamlit_app.py` | `make app` |
| Regenerate results bundle | `python -m src.deliverables.results_bundle` | `make bundle` |

`make` is a convenience only — it is **not** required, and is often absent on Windows. Every
target has the plain `python` equivalent above.

## Deliverables

| Brief deliverable | Location |
|---|---|
| 1. Working prototype | `app/streamlit_app.py` (`make app`) + the pipeline below |
| 2. Source & reproducibility package | this repo · `requirements.txt` · `make reproduce` |
| 3. Technical report | [`report/TECHNICAL_REPORT.md`](report/TECHNICAL_REPORT.md) |
| 4. Results bundle | [`results/bundle/`](results/bundle/) — predictions keyed on `unit_id`, scoring output, threshold record, manifest |
| 5. Data & model statement | [`deliverables/DATA_AND_MODEL_STATEMENT.md`](deliverables/DATA_AND_MODEL_STATEMENT.md) |
| 6. Pitch & demo | see `report/TECHNICAL_REPORT.md` §9 for the core narrative |

## Repository layout

```
src/data/        loading, split construction, provenance & leakage audit
src/features/    engineered stateless features, pooled stateful context
src/models/      rule baseline, stateless detector, Track 2 cascade
src/eval/        metrics, LODO, window alerting, operating points, ranking quality, failure analysis
src/explain/     SHAP explanations
src/latency/     Stage 1 / Stage 2 latency harnesses
src/deliverables/results bundle builder
app/             Streamlit analyst case view
results/metrics/ every number cited in the report
results/bundle/  the submitted results bundle
tests/           8 tests incl. leakage gate and unit_id source-traceability
```

## Evaluation protocol

Capture-file-grouped split (never random row-level), leakage-gated features with an allowlist,
threshold selection on `val_thr` only, isotonic calibration on `val_cal` only, `test` scored once.
Leave-one-day-out is run as a separate robustness loop. Full detail: report §4–§5, §7b.

## Citation

> Samaneh Mahdavifar, Amgad Hanafy Salem, Princy Victor, Miguel Garzon, Amir H. Razavi,
> Natasha Hellberg, and Arash Habibi Lashkari, "Lightweight Hybrid Detection of Data Exfiltration
> using DNS based on Machine Learning," *11th IEEE ICCNS*, December 3–5, 2021, Weihai, China.
