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

## The problem

DNS is allowed out of almost every network, so it is a natural covert channel: an attacker encodes
stolen data into query names and exfiltrates it past controls that never inspect DNS. The hard part
is not spotting a noisy tunnel — it is that **low-and-slow exfiltration looks like ordinary DNS
traffic**, while detection aggressive enough to catch it buries analysts in false alerts on a
protocol that generates millions of legitimate queries a day.

Our target user is a defender on a small or resource-constrained network: someone who cannot run
heavyweight stateful DNS analytics on every query, and who needs enough evidence attached to an
alert to decide for themselves whether it is real.

## What we built

An offline, defensive detection prototype with three layers, all evaluated on the same
leakage-controlled split:

1. **A row-level detector** (primary) — LightGBM over 14 leakage-gated per-query features, giving
   each DNS query a calibrated risk score and an explicit operating threshold.
2. **A tiered cascade** (supporting) — cheap stateless scoring on every query, escalating only
   elevated sessions to pooled stateful context, which then confirms or declines to confirm.
3. **An analyst case view** (supporting) — a Streamlit triage queue with per-alert SHAP reason
   codes, an explicit uncertainty state, and a mandatory human disposition step.

Plus a mandatory non-ML rule baseline to measure what the ML actually buys.

## How it works

```
        DNS query row
              │
              ▼
   ┌────────────────────────┐   14 gated stateless features (no domain text,
   │  Stage 1: LightGBM     │   no timestamps, no capture/session identity)
   │  + isotonic calibration│   → calibrated risk score
   └────────────────────────┘
              │
              │  score ≥ 0.4 ("suspicious" band)?
              ▼
     session r_sus = fraction of that capture's rows in the band
              │
              │  r_sus > δ (0.5, tuned on val_thr)?
              ▼
   ┌────────────────────────┐   pooled per-capture stateful context;
   │  Stage 2: majority vote│   transparent vote over the 3 features with the
   │  over 3 stateful feats │   largest train-set benign/attack gap
   └────────────────────────┘
              │
              ▼
   confirmed → alert session  │  unconfirmed → HUMAN REVIEW  │  context missing → Stage 1 verdict
```

**Data flow:** raw CSVs → leakage gate (allowlist of `sl_*` features) → capture-file-grouped split
(`train` / `val_cal` / `val_thr` / `test`) → model fit on `train` → calibration on `val_cal` →
threshold on `val_thr` → `test` scored exactly once. Leave-one-day-out runs as a separate
robustness loop. Every number in the report traces to a JSON file under `results/metrics/`.

**Key design decision:** the decision unit is the individual query, not the session or window. We
tested session- and window-level units and found both inherit their label from the capture file
(every capture in this dataset is 100% benign or 100% attack), which inflates scores without
improving real detection — see "What we found" below.

## What we found — read this before the numbers

We built a feature that scored **PR-AUC 0.99999** and then withdrew it.

It counted how often a query's exact feature vector repeated within its own session. That looked
like a genuine behavioural signal, and it survived a leave-one-day-out check. It was still wrong:
every capture file in this dataset is single-composition — 100% benign or 100% attack, never mixed
— so *any* statistic computed by grouping on session is partly a proxy for the session's label.
The LODO check could not have caught it, because every fold shares that same structure. **A test a
broken model passes is not a test.**

The experiment that settled it: we merged one benign and one attack session into a single synthetic
session and re-scored. **PR-AUC collapsed from 0.99999 to 0.082 — worse than chance.** We withdrew
the model.

The same root cause then explained two more results: the cascade's 100% session recall, and a
window-level rule that scored 45.7% recall but **0% on realistically diluted attack traffic**. One
cause, three experiments. Full detail in
[`report/TECHNICAL_REPORT.md`](report/TECHNICAL_REPORT.md) §9 — **if you read one section, read
that one.**

The headline number below is the honest one that survived.

## Headline result

v1 stateless row-level detector, 14 leakage-gated features, threshold selected on `val_thr`:

| | PR-AUC | Recall | Precision | FPR | False alerts / 10k benign |
|---|---|---|---|---|---|
| Combined | **0.6269** | 10.78% | 62.6% | 4.12% | 411.7 |
| Light | 0.1283 | 10.64% | — | 4.12% | 411.7 |
| Heavy | 0.6052 | 10.80% | — | 4.12% | 411.7 |

A higher-recall operating point exists at 47.06% recall / 17.95% FPR; there is nothing in between
(score quantization leaves a 15-point FPR band unreachable). Both are reported — §7.

For comparison, the mandatory non-ML rule baseline (`z(sl_len) + z(sl_entropy)`, signs fit on
train) reaches PR-AUC 0.523 / ROC-AUC 0.700. The ML lift is real but modest, and at the tight
operating point the rule is competitive on precision — reported honestly in §6b rather than framed
around a more flattering comparison.

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

> ⚠️ **`--retrain` overwrites the tracked model artifact** `models/stateless_lgbm_v1.pkl` (and
> `models/stateless_threshold_v1.json`). A retrained model can differ slightly from the committed
> one — LightGBM is not guaranteed bit-identical across thread counts, CPU architectures, or
> library builds — so after running it you may no longer reproduce the published numbers exactly.
> **Use the plain `python scripts/reproduce.py` to verify our reported score.** If you have already
> run `--retrain` and want the original artifacts back:
>
> ```bash
> git checkout -- models/
> ```

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

## Configuration

No environment variables, secrets, or API keys are required — the pipeline is fully offline. All
behaviour is driven by three version-controlled config files, which a reviewer can inspect to
verify our claims rather than take them on trust:

| File | Controls |
|---|---|
| [`configs/schema.lock.yaml`](configs/schema.lock.yaml) | The 14 gated feature names, canonical column names, label encoding, decision unit, and the excluded day (`2020-11-25`) |
| [`configs/leakage_exclusions.yaml`](configs/leakage_exclusions.yaml) | The allowlist gate — every excluded shortcut field with the reason it is excluded (enforced by `LeakageGate` and `tests/test_leakage.py`) |
| [`configs/feature_lists.yaml`](configs/feature_lists.yaml) | Feature lists consumed by the models |

Fixed seed `20260808` throughout; LightGBM runs with `deterministic=True, force_row_wise=True`.

## Evaluation protocol

Capture-file-grouped split (never random row-level), leakage-gated features with an allowlist,
threshold selection on `val_thr` only, isotonic calibration on `val_cal` only, `test` scored once.
Leave-one-day-out is run as a separate robustness loop. Full detail: report §4–§5, §7b.

## What we would improve or add with more time

Ordered by what we think would actually move the result, not by ease.

**1. Get a dataset that can evaluate above the query level.** This is the binding constraint, not a
modelling choice. Because every capture here is single-composition, no session- or window-level
decision unit can be honestly validated — we demonstrated three separate ways this misleads (§9).
We would either obtain captures with genuinely mixed sessions, or build a principled synthetic
mixing harness (our interleaving stress test is a first sketch) and re-evaluate the cascade and
window rule against it. Until then, session-level claims from this dataset are not trustworthy from
*anyone*, including us.

**2. Attack the light-attack ceiling directly.** Light exfiltration is where the system is weakest
(PR-AUC 0.128 vs 0.605 heavy; the light-only LODO fold is the worst at 0.345). Per-query features
alone appear insufficient. The promising direction is genuinely *streaming* temporal features —
trailing-window entropy, inter-query timing, per-source query rate — computed over a bounded
lookback rather than a whole session, so they are deployable and cannot absorb session identity.
We scoped this but deliberately cut it rather than ship an unvalidated feature after the 0.99999
episode.

**3. Break the score-quantization ceiling.** 71,012 benign rows collapse to only 4,289 distinct
scores, which leaves a 15-point FPR band unreachable at any threshold, caps ranking quality beyond
the top ~140 alerts, and leaves isotonic calibration almost nothing to fit (Brier 0.1498 → 0.1486).
Richer continuous features, or a model class producing finer score granularity, would likely help
more than tuning the current one.

**4. Strengthen the statistics we can't currently support.** Session-clustered bootstrap CIs on
window-level results; more than 3 test sessions before making any session-level claim; a proper
sensitivity analysis on the cascade's escalation δ, which currently sits only 0.12 above the benign
session's `r_sus` — a margin we report but cannot estimate the risk of at n=1.

**5. Production hardening we did not attempt.** Streaming ingestion instead of batch CSVs; model
monitoring for score drift; an analyst feedback loop so dispositions become training signal; and
per-tenant threshold tuning, since the right FPR depends entirely on analyst capacity.

**Explicitly out of scope, and we would not claim otherwise:** DGA detection, botnet C2, malware
family attribution, and any claim of generalization to real-world exfiltration tooling. The dataset
cannot support those, and no amount of additional time on *this* data would change that.

## Citation

> Samaneh Mahdavifar, Amgad Hanafy Salem, Princy Victor, Miguel Garzon, Amir H. Razavi,
> Natasha Hellberg, and Arash Habibi Lashkari, "Lightweight Hybrid Detection of Data Exfiltration
> using DNS based on Machine Learning," *11th IEEE ICCNS*, December 3–5, 2021, Weihai, China.
