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

1. **Leave-one-day-out robustness check (mandatory per the challenge brief).**
   The brief requires: *"Test across collection days. Report the organizer's held-out-day or
   leave-one-day-out robustness result in addition to any development split."* This is separate from
   the dev split (`train`/`val_cal`/`val_thr`/`test`) built in Phase 1, which is intentionally NOT
   day-disjoint (see `docs/SESSION_CONSTRUCTION.md`). Needs its own evaluation loop: hold out each
   eligible collection day in turn (`2020-11-20` through `2020-11-24`; `2020-11-25` is excluded, zero
   benign rows), train/score on the rest, report recall/PR-AUC variation across folds. Not yet built.

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

7. **Track 2 cascade (stretch, not blocking).** Gracefully-degraded design agreed with Muneeb: Stage 1
   row-level stateless model (above) always runs; escalate to a capture-level pooled stateful-context
   Stage 2 when a capture's suspicious-row ratio exceeds a validation-tuned threshold δ. Report % of rows
   escalated, one-stage vs two-stage recall/latency, and an honest limitation note that Stage 2 operates
   at capture granularity (finest available in the supplied stateful table — confirmed no reliable
   row-level or sub-file positional alignment exists between the stateless/stateful tables). Build only
   after items 2–6 are done and verified.

## Exit criterion

Baseline rule, stateless model, and (if time permits) the Track 2 cascade all scored on `test`, with
light/heavy/combined breakdowns and the leave-one-day-out robustness result, in a results table.
