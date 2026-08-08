# Data and Model Statement

Challenge brief deliverable #5: *"organizer snapshot version and SHA-256, external
datasets/models/APIs, licences, training sources where known, and any data sent to third-party
services."*

Team: Muneeb + Serena · Declared primary track: **Track 1 — Detection & Alert Prioritization**

---

## 1. Dataset snapshot and checksums

| | |
|---|---|
| Dataset | CIC-Bell-DNS-EXF-2021 |
| Source | Canadian Institute for Cybersecurity, University of New Brunswick, with Bell Canada CTI |
| Release used | **Public UNB release** (`release_type: public_unb_release`) |
| Files | 36 (18 stateless feature CSVs + 18 stateful feature CSVs) |
| Total size | 103,384,358 bytes |
| PCAPs included | **No** — this release contains derived feature tables only, no raw packet captures |
| Per-file SHA-256 | All 36 recorded in [`results/metrics/provenance.json`](../results/metrics/provenance.json) |
| Manifest digest (SHA-256 over every `path:sha256` pair, sorted by path) | `4f69e72a0965893d091ebd9974c2d503b759aa683e73d8df2d6711c7d9e1523f` |

### Why there is no single "organizer snapshot version"

The brief anticipated an organizer-provided frozen snapshot carrying a version string, a file
manifest, published checksums, and a split manifest. **None of these were distributed.** Sofstica
confirmed on the hackathon Discord (2026-08-08) that no separate organizer snapshot exists beyond
the official public dataset link in the brief's resources section, and that teams should proceed
with the public release.

Consequently:

- There is **no organizer-published checksum to verify against**, so
  `provenance.json:expected_archive_sha256` is `null` and `has_published_checksums` is `false`.
  The `checksum_matched: true` field means *internal* consistency (every file re-hashed to the
  value recorded at ingest), **not** agreement with an organizer-published value. We state this
  explicitly rather than let a `true` flag imply external verification that did not happen.
  What the checksums therefore *do* establish is **integrity across the pipeline** — that the
  bytes used for training, evaluation, and every reported number are the same bytes ingested at
  the start, and that a judge re-hashing their own copy can confirm they hold identical inputs.
  What they cannot establish is agreement with an authoritative organizer reference, because no
  such reference was published.
- The manifest digest above is **computed by us**, not issued by the organizers. It is provided so
  a judge can confirm they are looking at byte-identical inputs; it is not an authority.
- `has_split_manifest: false` — no organizer split was distributed, so the split of record is this
  project's own capture-file-grouped split ([`docs/SESSION_CONSTRUCTION.md`](../docs/SESSION_CONSTRUCTION.md)).
- `has_data_dictionary: false` — feature semantics were taken from the source paper and the
  official UNB dataset page, not from a shipped dictionary.

Acquisition route: the public release was obtained as a Kaggle mirror of the UNB dataset and
verified against the official UNB page's published Table 2 category statistics by exact row-count
reconciliation (757,211 stateless rows / 262,105 stateful rows across 18 captures) before use.

## 2. External datasets, models, and APIs

**None.** Specifically:

| Category | Used | Detail |
|---|---|---|
| External datasets | No | Only CIC-Bell-DNS-EXF-2021 as described above |
| Pretrained models | No | Every model is trained from scratch on the training partition |
| Embeddings / foundation models | No | — |
| External APIs at train or inference time | No | The pipeline runs fully offline |
| Threat-intel feeds, allow/deny lists, WHOIS, passive DNS | No | No enrichment from any external source |
| Network access required to reproduce | No | Beyond `pip install` of the pinned dependencies |

The prototype has **no runtime network dependency**. The Streamlit analyst view reads only local
files produced by the evaluation pipeline.

## 3. Model training sources

All models are trained solely on the `train` partition of the split described above. No transfer
learning, no external initialization, no distillation from another model.

| Component | Type | Trained on | Version string |
|---|---|---|---|
| Primary detector | LightGBM binary classifier, 14 gated stateless features | `train` | `stateless_lgbm_v1` |
| Probability calibrator | Isotonic regression | `val_cal` only | bundled with above |
| Operating threshold | Selected, not trained | `val_thr` only | see `threshold_selection_record.json` |
| Mandatory baseline | Signed z-score rule (`sl_len`, `sl_entropy`) — non-ML | `train` (normalization + sign only) | `rule_baseline_v1` |
| Cascade Stage 2 | Transparent majority vote over 3 stateful features | `train` captures only | `cascade_over_stateless_lgbm_v1` |
| Window alerting | Rule with grid-searched parameters | `val_thr` only | `window_N200_t0.714257_kn0.1` |

Determinism: fixed seed `20260808`, LightGBM `deterministic=True, force_row_wise=True`. `test` is
scored exactly once per system, at an already-chosen operating point, and is never used for
fitting, calibration, threshold selection, or feature selection.

**Rejected and not submitted:** `stateless_lgbm_v2` (session-repeat features, PR-AUC 0.99999) was
withdrawn after failing a mixed-session stress test (PR-AUC collapsed to 0.082). Its artifacts are
retained under `results/metrics/*_v2_rejected.*` as evidence for TECHNICAL_REPORT.md §9 and are
**not** part of the submitted results.

## 4. Software dependencies

Python 3.12.5 (plan specifies 3.11; verified end-to-end on 3.12.5). Pinned in
[`requirements.txt`](../requirements.txt):

`pandas==2.3.2` · `numpy==2.3.2` · `pyarrow==22.0.0` · `scikit-learn==1.7.2` ·
`lightgbm==4.7.0` · `shap==0.52.0` · `streamlit==1.52.0` · `PyYAML==6.0.2` · `pytest==9.1.1`

All are open-source packages installed from PyPI. None phones home at runtime.

## 5. Data sent to third-party services

**No dataset files, PCAPs, or bulk records were uploaded to any external service.** The dataset was
processed entirely on local hardware, and the prototype makes no outbound network calls.

**Disclosed transfer — LLM-assisted development.** Development of this project used an LLM coding
assistant (Anthropic Claude, via Claude Code). In the course of that assistance, the following were
transmitted to the provider's API:

- Project source code and configuration.
- Aggregate evaluation results (metrics, confusion matrices, row counts, score distributions).
- **Derived per-row feature values for small diagnostic samples** — e.g. the worst-5 false
  negative / false positive tables in TECHNICAL_REPORT.md §12, and similar debugging output,
  each on the order of tens of rows.
- **A small number of raw `sld` and `longest_word` field values** (roughly a dozen rows), printed
  once during Phase 1 schema inspection while establishing that the `sld` column contains IP
  octets and NetBIOS names rather than registrable domains — the finding that led to `sld` being
  excluded as a leakage shortcut.
- `unit_id` / `session_id` / capture-file path strings.

Not transmitted: any PCAP (none exist in this release), any bulk export of the feature tables, the
dataset archive, or any credential.

The brief's safety rules permit sending derived records to an external service where "event rules
and the data licence explicitly allow it **and the transfer is disclosed**." Both limbs are
addressed here. The dataset licence permits redistribution and republication of
CIC-Bell-DNS-EXF-2021 in any form subject to the required citation (§6), which we read as covering
the diagnostic-scale transfers described above. No PCAPs exist in this release; no bulk export,
archive, or credential material was transmitted. We disclose the transfer here per the brief's
requirement rather than asserting a null.

## 6. Licence and required citation

CIC-Bell-DNS-EXF-2021 is distributed by the Canadian Institute for Cybersecurity under the terms
published on the official dataset page. Use and redistribution must follow those terms; the
organizer-provided manifest governs the exact files used for this event.

**Required citation:**

> Samaneh Mahdavifar, Amgad Hanafy Salem, Princy Victor, Miguel Garzon, Amir H. Razavi,
> Natasha Hellberg, and Arash Habibi Lashkari, "Lightweight Hybrid Detection of Data Exfiltration
> using DNS based on Machine Learning," *11th IEEE International Conference on Communication and
> Network Security (ICCNS)*, December 3–5, 2021, Beijing Jiaotong University, Weihai, China.

Official dataset page: University of New Brunswick, Canadian Institute for Cybersecurity,
CIC-Bell-DNS-EXF-2021.

**This repository does not redistribute the dataset.** `data/raw/` is git-ignored; a user must
obtain the data themselves from the official source.

## 7. Scope of claims

Consistent with the brief's "what the data can and cannot support" section, and with
TECHNICAL_REPORT.md §9 and §14: this system **detected represented testbed exfiltration traffic**
under the evaluation protocol described. It is not evidence of detecting unseen real-world
exfiltration tooling, DGA traffic, botnet C2, or novel malicious infrastructure, and no such claim
is made anywhere in this submission.

Detection outputs are **advisory**. The prototype does not block DNS, quarantine hosts, or modify
firewall rules; the analyst interface requires a human disposition step for every alert.
