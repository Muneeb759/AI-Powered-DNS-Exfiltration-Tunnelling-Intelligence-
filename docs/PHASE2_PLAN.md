# PHASE 2 PLAN — Baselines, Stateful Model, Calibration (Serena owns)

> Handoff verified: `scripts/handoff_check.py` passes cleanly on the real CIC-Bell-DNS-EXF-2021 data
> (see `docs/PHASE1_AUDIT.md` and `summary.md`). Phase 1 exit criteria are met.

## Dataset provenance note

Sofstica confirmed on the hackathon Discord (2026-08-08) that there is no separate frozen organizer
snapshot beyond the official CIC-Bell-DNS-EXF-2021 dataset link in the brief's resources section — teams
should proceed with the publicly released dataset. This is the dataset already verified end-to-end in
Phase 1 (SHA-256-checked, 757,211 stateless / 262,105 stateful rows, reconciled against the official UNB
Table 2 statistics). No separate organizer split manifest was distributed either, so the
`capture_file_grouped_split` built in Phase 1 is the split of record for this team, not a stand-in for one
that's still coming. `split_status` in `results/metrics/split_summary.json` should be upgraded from
`PROVISIONAL` once Serena signs off on Phase 2 numbers.

## Task list

1. **Leave-one-day-out robustness check (mandatory per the challenge brief). DONE.**
   Implemented in `src/eval/loto.py`, results in `results/metrics/leave_one_day_out_report.json`.
   Trains a fresh LightGBM per held-out day (5 eligible days; `2020-11-25` excluded, zero benign
   rows), threshold picked in-sample on the training fold (documented simplification — only 18
   sessions total, not enough left per fold for a clean nested val split; this does not replace the
   out-of-sample threshold protocol used for the headline stateless model).
   **Finding (v1, base 14 features):** fold-to-fold PR-AUC ranges 0.34–0.80 (std 0.18). The
   light-attack-only day (`2020-11-21`) is the weakest fold by a wide margin (PR-AUC 0.34) versus
   the heavy-only days (0.71–0.80) — concrete evidence for the report's failure-analysis section
   that light attacks are the harder detection problem, not just an assertion from the plan doc.
   **Updated (v2, item 8 below):** rerun with session-repeat engineered features recomputed
   independently per fold — PR-AUC now 0.997–1.000 across all folds including the light-only day
   (0.9996). Also serves as the leakage check for item 8's repeat-count feature.

2. **Rule baseline (mandatory per brief).** Threshold on query length + Shannon entropy of the query
   name. Tune threshold on validation only (`val_thr`). Document the exact rule and threshold.

3. **Stateless model.** LightGBM on the 14 gated stateless features (`configs/feature_lists.yaml`),
   session-grouped per the existing split. This is the row-level Track 1 detector — already unblocked by
   Phase 1's parquet splits.

4. **Calibration.** Isotonic regression fit on `val_cal` only.

5. **Threshold selection.** Pick the operating threshold at FPR = 0.1% (10 false alerts / 10k benign) on
   `val_thr`. Row-level resolution floor is fine-grained (`1.4e-5` to `3.6e-6` across partitions per
   `results/metrics/split_summary.json`), so this is measurable without the session-level floor problem
   Phase 1 found for a capture-file-level decision unit.

6. **Metrics reporting.** PR-AUC, precision, recall, FPR@threshold, confusion matrix, false alerts/10k
   benign — separately for light, heavy, and combined, on `test`. Bootstrap 95% CIs required for light
   recall specifically (`test` light positives, 3,479 rows, sit entirely in one capture trace — see
   `summary.md`).

7. **Track 2 cascade (stretch, not blocking). DONE.**
   `src/features/stateful_context.py` (pooled per-capture stateful features, C1-C5 corrections from
   `PHASE1_AUDIT.md` finally implemented) + `src/models/cascade.py`. Stage 1 = existing stateless
   model; escalation trigger = fraction of a session's rows above the paper's own "suspicious" score
   band (>=0.4) exceeding a val_thr-tuned δ (0.5) — NOT the FPR=0.1% alert threshold, which was too
   strict to ever trigger escalation. Stage 2 = transparent majority-vote over the top-3 stateful
   features by train-set benign/attack mean gap (`sf_ttl_mean`, `sf_rr_name_length`,
   `sf_ttl_variance`) — no ML classifier fit at capture level; only 3 benign captures exist in train,
   too few for that to be honest. Graceful fallback to Stage-1-only implemented for missing context.

   **Result on test:** both attack sessions escalate and are confirmed (100% recall, up from ~0.2%
   Stage-1-only recall at the same FPR), FPR unchanged (7.6/10k), 38.9% of rows escalated to the
   expensive path (61.1% stay on the cheap path). **Two flagged limitations, not hidden behind the
   number:** (1) all 18 capture files in this dataset are single-composition — verified, none mix
   benign+attack rows — so blanket-alerting a confirmed session's rows is a testbed-shaped win, not
   one a mixed real-world session would give; ties directly to the brief's own controlled-environment
   caveat. (2) test has only 3 sessions and delta was tuned on val_thr's 3 sessions — real result, not
   test leakage, but not a statistically powerful session-level claim. The row-level Stage 1 numbers
   remain the primary, statistically meaningful headline metric.

8. **Stateless model v2: session-repeat engineered features. DONE.**
   `src/features/stateless_engineered.py` adds `sl2_session_repeat_count` /
   `sl2_session_repeat_ratio` (how often a row's exact base feature vector recurs within its own
   session, via `groupby("session_id")` — session_id used only as a grouping key, never as a
   feature value, same pattern as `sf_*` pooling). Diagnostic finding behind it: test's 39,274
   attack rows reduce to 47 unique base feature combinations, all within the benign feature range
   — no row-level model on the base 14 features can separate them; the repeat-count feature
   exposes the one thing that does (attack sessions replay those 47 patterns thousands of times,
   benign sessions rarely repeat a pattern at all).

   **Result:** PR-AUC 0.6268 -> 0.99999, recall@FPR=0.1% 0.229% -> 97.21% (combined). Validated
   against session-identity leakage via leave-one-day-out with repeat-stats recomputed
   independently per fold (item 1 above, rerun) — held-out-day PR-AUC 0.997-1.000, confirming the
   signal generalizes to sessions never seen in training, not a memorized session fingerprint.
   Documented in the report as dataset-structure-dependent (single-composition captures), not a
   claim of production generalization.

## Exit criterion

Baseline rule, stateless model, and (if time permits) the Track 2 cascade all scored on `test`, with
light/heavy/combined breakdowns and the leave-one-day-out robustness result, in a results table.
