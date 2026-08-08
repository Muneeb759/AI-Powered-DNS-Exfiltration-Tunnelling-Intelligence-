# Results Bundle

Challenge brief deliverable #4: *"machine-readable predictions for the organizer test IDs,
scoring output, threshold selection record, and model/configuration version."*

Regenerate everything here with:

```bash
python -m src.deliverables.results_bundle
```

## Contents

| File | What it is |
|---|---|
| `MANIFEST.json` | Model/config versions, environment, per-file SHA-256, test-ID definition |
| `scoring_output.json` | All reported metrics: primary detector, mandatory baseline, cascade, window experiment |
| `threshold_selection_record.json` | Every threshold, which partition it was selected on, and the selection rule |
| `predictions_primary_row_level.csv` | **Primary system.** One row per test DNS query |
| `predictions_baseline_rule.csv` | Mandatory non-ML rule baseline, same rows |
| `predictions_cascade_row_level.csv` | Track 2 cascade verdicts, same rows |
| `predictions_window_level.csv` | Window-level experiment (505 windows) — reported, **not** the alerting unit |

## Test IDs

All row-level files are keyed on **`unit_id`** (the decision-unit primary key from
`configs/schema.lock.yaml`) — 100,841 unique IDs covering the test partition exactly.
`session_id` is retained as a grouping column but is *not* an ID: it has only 3 distinct
values across the whole test partition.

Sofstica confirmed on the hackathon Discord (2026-08-08) that no separate organizer snapshot
or split manifest exists beyond the public dataset, so the IDs of record are this project's own
`unit_id`s over the capture-file-grouped test partition (`docs/SESSION_CONSTRUCTION.md`).

## Two operating points, not one

The primary predictions file carries **two** prediction columns, because score quantization
leaves a 15-point FPR band unreachable at any threshold (see `TECHNICAL_REPORT.md` §7):

| Column | Threshold | Test FPR | Recall | Precision |
|---|---|---|---|---|
| `prediction_at_headline_threshold` | 0.713607 | 4.12% | 10.78% | 62.6% |
| `prediction_at_post_cliff_threshold` | 0.712416 | 17.95% | 47.06% | 62.6% |

There is no operating point between them. Which is correct depends on analyst capacity;
both are reported rather than choosing the more flattering one.

## Not included, deliberately

`stateless_lgbm_v2` (the session-repeat feature model, PR-AUC 0.99999) is **not** in this
bundle. It was rejected after a mixed-session stress test collapsed it to PR-AUC 0.082.
Its artifacts are retained under `results/metrics/*_v2_rejected.json` as evidence for
`TECHNICAL_REPORT.md` §9, and are not submitted results.
