# Technical Report — AI-Powered DNS Exfiltration & Tunnelling Intelligence

> **Status: DRAFT, post-revert.** This report was rewritten after an internal adversarial review
> and a synthetic mixed-session stress test found that the previous
> headline result (PR-AUC 0.9999) was a dataset-structure artifact, not a real detector. That
> finding, and two related ones discovered while chasing it down, are now the strongest section of
> this report (§9). The headline model is v1 (14 base gated features, PR-AUC 0.627). Nothing here
> is a placeholder figure — every number traces to a file in `results/metrics/`. Team: Muneeb
> (data, splits, leakage audit, latency harness, reproducibility) + Serena (baselines, modelling,
> calibration, cascade, window aggregation, evaluation, SHAP, Streamlit UI, this report).

---

## 1. Problem, target user, and declared track

Network defenders and administrators of small or resource-constrained networks need to detect
DNS-based data exfiltration and tunnelling without the compute budget for a heavyweight,
always-on stateful analysis pipeline, and need enough evidence to make the final call themselves
rather than trust an opaque score.

**Declared primary track: Track 1 — Detection & Alert Prioritization.** Track 1's five strong-
submission criteria are met directly: a stated decision unit (row-level, with a window-level
aggregation reported as a secondary experiment, §8), calibrated confidence with an explicit
operating threshold (§6), separate light/heavy analysis at every reported number, alert ranking
via the continuous score, and reason codes tied to observed features (SHAP, §12).

**Track 2 (Lightweight & Tiered Detection) is retained as supporting work, reframed.** Track 2's
central claim is that measured compute savings justify a recall trade-off. Our own measurement
does not support that claim: Stage 2 costs ~165 μs per escalating session plus a one-time offline
context build, both negligible next to Stage 1's per-query cost. There is no meaningful cost to
trade against recall, so the cascade (§10) is presented as an **interpretable confirmation layer**
that turns a weak row-level signal into a session-level verdict with a transparent rationale — not
as a cost optimization. We are not claiming Track 2 as primary because our own numbers do not
support its central premise.

**Track 3 (Analyst Investigation & Explanation) is retained as supporting work**, unchanged in
scope: the Streamlit case view (§13) that surfaces the evidence this report already produces.

We are declaring one primary track, per the brief's requirement, and are explicit that the other
two are supporting components of that one submission, not parallel claims for separate credit.

## 2. System architecture

```
DNS query row
      |
      v
+----------------------------+
| Stage 1: stateless LightGBM|  <- always runs, cheap (14 gated base features, v1)
| calibrated (isotonic,      |
| val_cal-only)               |
+----------------------------+
      |
      | raw_score >= 0.4 ("suspicious" band, per the original
      | paper's own bin design)?
      v
  session r_sus = fraction of this capture's rows in that band
      |
      | r_sus > delta (0.5, tuned on val_thr)?
      v
+----------------------------+
| Stage 2: pooled stateful   |  <- only for escalated sessions (2 of 3 test sessions)
| context, majority vote     |  <- interpretable CONFIRMATION layer, not a cost optimization (§1)
| over top-3 discriminative  |
| features (train-set only)  |
+----------------------------+
      |
      v
confirmed -> alert whole session   |   unconfirmed -> human review   |   context unavailable -> Stage 1 verdict stands

Separately reported (secondary experiment, not part of the alerting pipeline above):
row-level Stage 1 scores can be pooled into fixed-size windows per capture (§8) --
raises apparent recall on dense attack captures, fails on diluted/interleaved attack
traffic. Not adopted as the alerting unit; reported for what it teaches (§9).
```

Rule baseline, Stage 1, Stage 2, and the window-aggregation experiment all sit on top of the same
Phase 1 data contract (`data/splits/{train,val_cal,val_thr,test}.parquet`), built and independently
re-verified end to end (`scripts/handoff_check.py`, `pytest` 6/6 green) against the real
CIC-Bell-DNS-EXF-2021 dataset.

## 3. Data used

- **Dataset:** CIC-Bell-DNS-EXF-2021 (Mahdavifar et al., 2021, ICCNS), public UNB release, 36
  files, 103,384,358 bytes, SHA-256-verified per file (`results/metrics/provenance.json`).
- **Provenance clarification:** Sofstica confirmed on the hackathon Discord (2026-08-08) that
  there is no separate frozen organizer snapshot beyond the official dataset link in the brief's
  resources section — this is the dataset of record for judging, not a stand-in.
- **Scale:** 757,211 stateless rows / 262,105 stateful rows across 18 capture files, reconciled
  exactly against the official UNB Table 2 category statistics.
- **No row-level or sub-file join exists** between the stateless and stateful tables (verified
  directly: zero shared column names, and stateless row order is not consistently chronological
  across files — see §10).
- **Every one of the 18 capture files is single-composition** — 100% benign or 100% attack rows,
  verified programmatically, none mixed. This single fact drives §9, the most important finding in
  this report.

## 4. Split protocol

`capture_file_grouped_split` (`src/data/splits.py`), not the originally planned leave-one-day-out
day-grouped split — day-grouping left validation partitions with zero light-attack rows.

| Partition | Rows | Benign | Positive | Light | Heavy | Min FPR floor |
|---|---|---|---|---|---|---|
| train | 405,851 | 281,164 | 124,687 | 11,345 | 113,342 | 3.56e-6 |
| val_cal | 95,102 | 49,115 | 45,987 | 10,241 | 35,746 | 2.04e-5 |
| val_thr | 125,016 | 71,012 | 54,004 | 17,618 | 36,386 | 1.41e-5 |
| test | 100,841 | 61,567 | 39,274 | 3,479 | 35,795 | 1.62e-5 |

**Guarantees actually enforced:** `session_id` (capture-file) disjointness, dual-category
composition (every partition has benign+light+heavy), `unit_id` uniqueness.

**Guarantee this split does *not* make:** `collection_day` is not disjoint across partitions.
Measured and reported explicitly (`results/metrics/split_summary.json:day_overlap`). Since
`collection_day`/`timestamp` are excluded features, this does not open a leakage path; it is why
§7's leave-one-day-out check is a separate evaluation loop, not folded into the dev split.

Day `2020-11-25` is excluded entirely (0 benign rows, single-class slice) — drops 30,401
heavy-attack rows from all partitions, a documented trade-off, not silent data loss.

## 5. Excluded leakage fields

Allowlist-based gate (`configs/leakage_exclusions.yaml`, enforced by `LeakageGate` and
`tests/test_leakage.py`):

| Field | Reason |
|---|---|
| `timestamp` | Leaks collection day / execution timing |
| `sld` | Raw domain/IP-octet/NetBIOS identity shortcut (verified in `docs/PHASE1_AUDIT.md` B3) |
| `longest_word`, `subdomain` | Raw string tokens carrying query-identity text |
| `capture_file`, `session_id`, `collection_day`, `unit_id` | Grouping/identifier keys, not features |
| `distinct_ip`, `reverse_dns`, `distinct_domains`, `unique_asn`, `unique_country`, `unique_ttl`, `rr_type` | Stateful-table raw identity/list fields |
| 8 constant stateful columns | `nunique == 1` across the entire dataset |

**Note on §9's finding:** `session_id` was excluded here as a raw feature value from the start.
§9 documents a *different, subtler* violation of this same principle — an aggregate *statistic*
computed by grouping on `session_id` (never the raw value itself) that still functioned as a
label proxy, because the brief's own listed shortcut ("session identity... may reveal the label")
applies to derived statistics, not just raw values. That distinction is exactly what the original
exclusion rule didn't anticipate, and exactly what broke.

**Sanity check (shortcut ablation, mandatory per brief):** a model trained with the excluded
identity shortcuts included reaches PR-AUC **0.6946** vs the gated model's **0.6349** — a real but
modest inflation (`results/metrics/leakage_ablation.json`).

## 6. Baseline

Mandatory non-ML rule baseline: signed `z(sl_len) + z(sl_entropy)`, normalization fit on `train`
only, threshold picked on `val_thr` at FPR=0.1%.

**A bug was found and fixed here, not just tuned:** the original unsigned version summed
`z(sl_len) + z(sl_entropy)` directly and scored at ROC-AUC 0.499 — exactly chance. Diagnosis:
`sl_len` alone has real train-set signal (ROC-AUC 0.63, attack queries are longer), but
`sl_entropy` alone points the *opposite* direction on this dataset (ROC-AUC 0.449 — attack entropy
is slightly *lower* than benign here, contrary to the textbook "exfil payloads are high-entropy"
assumption). Summing the two raw z-scores let the backwards entropy term cancel the real length
signal. Fix: each feature's sign is now determined from train data only (whichever direction
correlates with the attack class there) before summing — a standard baseline convention, not new
modelling.

**Why the entropy sign is reversed here is itself a finding, not just a bug to patch.** The
exfiltration tool in this testbed replays a small set of *structured* query shapes (§9: 39,274
attack rows reduce to 47 unique feature vectors), which are individually low-entropy. Ordinary
benign DNS, by contrast, is full of genuinely high-entropy subdomains — CDN hostnames, hashed
cache keys, randomized service labels. So on this data, high entropy is weak evidence of *benign*
traffic, not attack traffic. This corroborates §9's single-composition/limited-tooling finding from
an entirely independent direction: the attack traffic here is narrower and more repetitive than
real exfiltration would be, and that narrowness shows up even in a two-variable heuristic.

| | PR-AUC | ROC-AUC (supp.) | Recall @ FPR=0.1% | Precision | False alerts / 10k benign |
|---|---|---|---|---|---|
| Combined | 0.523 | 0.700 | 0.229% | 73.2% | 5.36 |
| Light | 0.091 | 0.699 | 0.345% | 26.7% | 5.36 |
| Heavy | 0.500 | 0.700 | 0.218% | 70.3% | 5.36 |

### 6b. Honest comparison: what the ML model actually buys over the rule

The brief requires showing *"measurable value beyond a conventional rule or threshold."* Compared
first at the **tight FPR=0.1% point** — note this is the model's *alternate* operating point, not
its §7 headline; it is used here because it is where the rule baseline's own threshold was selected,
so the two are compared on equal terms:

| | PR-AUC | Recall @ tight point (FPR=0.1%) | Precision |
|---|---|---|---|
| Rule baseline | 0.523 | 0.229% | **73.2%** |
| v1 ML model | **0.627** | 0.229% | 64.3% |

**At that tight point the rule baseline has higher precision than the model at identical recall.**
The model's advantage is concentrated in *ranking quality* (PR-AUC 0.523 → 0.627 combined;
0.091 → 0.128 on light attacks), not at that operating point. We report this rather than selecting
a comparison that flatters the model.

At the model's actual **headline** operating point (threshold 0.713607, 4.12% test FPR) it reaches
10.78% recall at 62.6% precision — a regime the rule baseline cannot match at comparable FPR (see
the next paragraph).

**Where the model is genuinely ahead, measured rather than asserted:** ranking quality across the
whole score range (ROC-AUC 0.810 vs 0.700; PR-AUC 0.627 vs 0.523), and recall at a comparable
operating point — at ~18% test FPR the model reaches 47.1% recall / 62.6% precision versus the
rule's 41.1% / 59.9%. That is a real 6-point recall advantage at equal FPR, not a decisive one.

**Alert ranking — the claim, tested** (`src/eval/ranking_quality.py`,
`results/metrics/ranking_quality_v1.json`). §7's flat ~62.6% precision across every threshold from
4% to 28% FPR implies a constant likelihood ratio, which would mean the score enriches by a fixed
factor but does not *rank* within the flagged pool. We tested this with tie-respecting
precision-at-depth (naive precision@k is invalid here — 22,763 test rows share one score, so a
fixed *k* slices a tie block arbitrarily; naive precision@1000 computes to 0.949 purely as a
row-order artifact and is not reported):

| Depth (tie-respecting) | Cumulative precision | Lift over base rate (0.3895) |
|---|---|---|
| top 48 | 91.7% | 2.35× |
| top 100 | 74.0% | 1.90× |
| top 141 | 63.8% | 1.64× |
| top 1,592 | 62.7% | 1.61× |
| top 6,770 | 62.6% | 1.61× |
| top 46,056 | 62.7% | 1.61× |

**The ranking claim survives only in a narrow form, and we state it that way.** The score does
prioritize the extreme head of the queue — an analyst working the top 50 alerts sees ~92%
precision, the top 100 ~74% — but by depth ~141 (of 100,841 rows) precision has already collapsed
to the flat ~62.6% asymptote and provides no further discrimination. So the model meaningfully
reduces analyst workload for roughly the first 100–140 alerts and not beyond. That is a real but
strictly bounded advantage over the rule's single hard cut, and it is the honest version of Track
1's *"alert ranking that reduces analyst workload"* criterion. See §9 (Experiment 4) for why the
ranking dies where it does.

The model's remaining practical advantage is per-alert reason codes (§12) tied to 14 features
rather than 2. These are workflow advantages plus a modest detection-quality gap, and we are not
claiming more.

## 7. Stateless model (Stage 1 headline, v1)

LightGBM (`model_version: stateless_lgbm_v1`) on the 14 gated stateless features only,
`scale_pos_weight` class balancing, isotonic calibration fit on `val_cal` only (too coarse for
FPR=0.1% resolution — the raw score is used for the operating decision; see code comment in
`src/models/stateless_model.py`), operating threshold selected on `val_thr`.

**Achievable operating points** (`src/eval/operating_points.py`,
`results/metrics/operating_points_v1.json`). Thresholds are enumerated from every *distinct*
benign score on `val_thr` and evaluated on `test`, rather than from a fixed grid of target FPRs:

| Threshold | val_thr FPR | test FPR | Recall (comb.) | Recall (light) | Recall (heavy) | Precision | Alerts/10k benign |
|---|---|---|---|---|---|---|---|
| 0.7354 | 0.07% | 0.08% | 0.23% | 0.23% | 0.23% | 65.9% | 7.5 |
| 0.7181 | 1.04% | 0.97% | 2.54% | 2.53% | 2.54% | 62.7% | 96.5 |
| 0.7143 | 4.22% | 3.94% | 10.35% | 10.23% | 10.36% | 62.7% | 393.7 |
| **0.7136** | **4.40%** | **4.12%** | **10.78%** | **10.64%** | **10.80%** | **62.6%** | **411.7** |
| 0.7124 | 19.42% | 17.95% | 47.06% | 46.22% | 47.14% | 62.6% | 1,794.8 |
| 0.70909 | 21.93% | 20.26% | 53.21% | 52.60% | 53.27% | 62.6% | 2,025.8 |
| 0.70906 | 30.26% | 27.93% | 73.49% | 73.30% | 73.51% | 62.7% | 2,792.6 |

**Why earlier versions of this table had duplicate rows — a real property, not a coarse grid.**
This model's benign score distribution is heavily quantized: 71,012 `val_thr` benign rows produce
only **4,289 distinct scores**, with very large ties. Entire FPR bands are therefore unreachable at
*any* threshold. The two largest gaps (`operating_points_v1.json:unreachable_fpr_bands`):

- **1.04% → 4.22%** FPR: nothing in between (3.2-point gap).
- **4.40% → 19.42%** FPR: nothing in between (15.0-point gap) — 10,662 benign rows share the single
  score 0.712416, so crossing it jumps FPR by 15 points at once.

There is no "10% FPR" operating point for this model. Reporting a table of fixed target FPRs would
imply otherwise; this table does not.

**Chosen headline operating point: threshold 0.7136 — 4.12% test FPR, 10.78% recall, 62.6%
precision.** It is the most recall available below the 15-point cliff. **We also report the
post-cliff point (17.95% FPR → 47.06% recall) rather than hiding it**: a defender willing to accept
~1,795 false alerts per 10,000 benign queries gets 4.4x the recall. That is a real operational
choice, and which side of the cliff is correct depends on analyst capacity, not on which number
looks better in a report.

| | PR-AUC | Recall @ 4.12% FPR | Precision | False alerts / 10k benign |
|---|---|---|---|---|
| Combined | **0.6269** | 10.78% | 62.6% | 411.7 |
| Light | 0.1283 | 10.64% | 12.7% | 411.7 |
| Heavy | 0.6052 | 10.80% | 60.4% | 411.7 |

### Confusion matrix at the headline operating point

Brief requirement: *"a confusion matrix and false alerts per 10,000 benign decisions."* Generated
by `src/eval/operating_points.py` into
`results/metrics/operating_points_v1.json:confusion_matrix_at_headline_threshold` — not typed in.

**Combined** (threshold 0.713607; 39,274 attack rows, 61,567 benign rows):

| | predicted attack | predicted benign | total |
|---|---|---|---|
| **actual attack** | 4,235 (TP) | 35,039 (FN) | 39,274 |
| **actual benign** | 2,535 (FP) | 59,032 (TN) | 61,567 |

**411.7 false alerts per 10,000 benign decisions.**

Per attack slice — each pairs that category against the *full* benign population, so FPR stays
meaningful (a category slice has no benign rows of its own):

| Slice | TP | FN | FP | TN | Recall | Precision | False alerts / 10k benign |
|---|---|---|---|---|---|---|---|
| Light | 370 | 3,109 | 2,535 | 59,032 | 10.64% | 12.7% | 411.7 |
| Heavy | 3,865 | 31,930 | 2,535 | 59,032 | 10.80% | 60.4% | 411.7 |

The light slice's 12.7% precision is the sharpest single statement of the light-attack problem in
this report: at the same threshold and the same false-alert budget, roughly seven of every eight
light-attack alerts are wrong, because only 3,479 light rows exist against 61,567 benign ones.

Light and heavy recall track each other closely but are not identical (10.64% vs 10.80% here;
46.22% vs 47.14% post-cliff) — heavy is consistently marginally higher at every operating point,
as expected, and the two are close because both categories' rows sit in the same quantized score
bands.

**Calibration** (Track 1 criterion: *"calibrated confidence and an explicit operating threshold"*).
Isotonic regression fit on `val_cal` only. Brier score on `test`: **raw 0.14976 → isotonic
0.14855** — a real but very small improvement, and the likely reason is the quantization described
immediately above rather than the raw score already being well calibrated: with only 4,289 distinct
score values and enormous tie blocks, isotonic regression has very few distinct points to fit
against and little room to redistribute probability mass. Full 10-bin reliability tables for both are in
`results/metrics/operating_points_v1.json:calibration`. **Calibrated probabilities are what the
analyst UI displays and what alert ranking uses; the raw score drives the operating threshold**,
because isotonic's step-function output collapses benign scores into a handful of plateaus far too
coarse to resolve a tight FPR budget. Both are reported so the trade-off is visible rather than
asserted.

**Light-attack recall bootstrap CI:** zero-width by construction, not by precision — all 3,479
test-set light-attack rows sit in a single capture trace, so session-grouped bootstrap resampling
only has one cluster to draw from. This is exactly why §7's leave-one-day-out check (below) and
§9's statistical-power argument matter more than a single test-set number.

## 7b. Leave-one-day-out robustness (mandatory per brief, separate from the dev split)

`src/eval/loto.py`, v1 (base 14 features): fresh model per held-out day, threshold picked
in-sample per fold (documented small-N simplification — only 18 sessions total).

| Held-out day | Rows | Positive rows | Categories present | PR-AUC | Recall @ in-sample threshold |
|---|---|---|---|---|---|
| 2020-11-20 | 142,138 | 0 | benign only | n/a (single class) | n/a |
| 2020-11-21 | 167,720 | 28,694 | benign, **light** | **0.345** | 0.26% |
| 2020-11-22 | 131,098 | 69,531 | benign, light, heavy | 0.748 | 0.21% |
| 2020-11-23 | 99,060 | 49,945 | benign, heavy | 0.715 | 0.32% |
| 2020-11-24 | 186,794 | 115,782 | benign, heavy | 0.800 | 0.25% |

**This is what satisfies the brief's *"confidence intervals **or** fold-to-fold variation where the
split permits"* requirement — the second limb.** We report fold-to-fold variation across the four
testable LOTO folds rather than confidence intervals, because the CI route is degenerate on this
data: all 3,479 light-attack test rows sit in a single capture, so a session-clustered bootstrap
has exactly one cluster to resample and returns a zero-width interval that would overstate
precision rather than convey uncertainty (§7). The spread below is the honest uncertainty estimate.

Fold-to-fold PR-AUC: mean 0.652, std 0.180, range 0.345–0.800. **The light-attack-only day is the
weakest fold by a wide margin** — concrete, measured evidence that light-attack detection is the
harder problem for this row-level detector, consistent with the recall-vs-FPR curve above.

## 8. Window-level alerting — a reported experiment, not the alerting unit

The brief permits changing the alerting unit: *"For window- or session-level systems, report the
above at the actual alerting unit and explain how packet/query predictions are aggregated."* We
tested this directly rather than assuming it would help.

**Method** (`src/eval/window_alerting.py`): non-overlapping windows of N consecutive rows within
one `session_id` (never crosses a capture-file or collection-day boundary, by construction). A
window alerts if the fraction of its rows scoring above row-threshold `t` reaches `k/N`. All three
(N, t, k/N) grid-searched on `val_thr` only; `test` scored once.

| Setting | val_thr recall / FPR | test recall / FPR (dense, pure captures) | test recall (diluted mixed-traffic stress test) |
|---|---|---|---|
| Row-level v1 (§7) | — | 10.78% / 4.12% (61,567 benign rows) | not applicable — row-level, no aggregation |
| Window N=200, t=0.7143, k/N=0.10 | 51.5% / 4.2% | **45.69%** / 4.22% (308 benign windows) | **0%** |
| Window N=200, t=0.7143, k/N=0.02 | — | 41.12% / 15.58% (308 benign windows) | **18.2%** |

**Why window-level numbers look strong and why we don't trust them as the headline:** every one of
this dataset's captures is single-composition (§3, §9). A non-overlapping window's label ("does it
contain any attack row") is therefore almost always identical to its capture's label — window-level
alerting is close to reclassifying *captures*, at a different name. The 308 benign windows above
all come from **one** benign capture file. Row-level has 61,567 independent benign rows; window-
level has 308 correlated ones drawn from a single source. That is not a statistical-power argument
we can wave away.

**The mixed-traffic stress test confirms it in practice, not just in theory.** We built a synthetic
capture interleaving 1 attack row per 20 benign rows (3,078 attack rows in 64,638 total,
order-preserved within each source — the closest available approximation of the brief's own
"real session mixing normal queries with a slow, low-volume exfil channel"). At the tuned
operating point (k/N=0.10), **recall drops to 0%** — the diluted stream never pushes any window's
alert fraction above 0.085, short of the 0.10 threshold. A looser setting (k/N=0.02) partially
recovers 18.2% recall, at the cost of a much higher false-alert rate on pure captures (15.6% FPR).

**We are reporting both regimes rather than picking a favorite:** k/N=0.10 is tuned for dense,
pure-composition attack traffic and fails completely once attack queries are diluted into normal
traffic; k/N=0.02 is more robust to dilution but alerts far more often on ordinary sessions. Neither
is adopted as the primary alerting unit. **Row-level remains the headline** (§7) because it is the
only unit in this dataset large enough and independent enough to support a real statistical claim.

**Added cost of window aggregation:** negligible — ~9.5 μs to aggregate a 200-row window (≈0.05
μs/query amortized), versus Stage 1's ~2,228 μs/query inference cost.

## 9. What this dataset can and cannot evaluate

Three separate experiments this project ran — independently, for different reasons — produced the
same failure, and it has one cause.

> **Every capture file in CIC-Bell-DNS-EXF-2021 is single-composition: 100% benign or 100% attack,
> never mixed.** Therefore any decision unit above the individual query — a session-level
> aggregate feature, a session-level cascade verdict, a window pooling many rows — inherits its
> label from the capture it was drawn from, not purely from the behavior of the rows inside it.
> All three of the following produced strong-looking numbers for that same structural reason, and
> all three degrade or invert once traffic looks like a real mixed session. Row-level decisions are
> the only unit this dataset can evaluate with real statistical power: 61,567 independent benign
> test rows, versus 308 benign windows drawn from one capture, versus a cascade decision backed by
> one benign test session.

**Experiment 1 — session-repeat engineered features (fully rejected).** A diagnostic found that
test's 39,274 attack rows reduce to only 47 unique base feature combinations — the exfiltration
tooling replays a small set of query shapes at high volume, while benign traffic rarely repeats a
shape. We built `sl2_session_repeat_count`/`ratio` — counting how often a row's exact feature
vector recurs within its own `session_id` — and it drove PR-AUC from 0.627 to **0.99999**.

*Why we did not trust it, before any stress test:* `session_id` is a prohibited shortcut per the
brief ("session identity... may reveal the label. These identifiers are not valid predictive
evidence"), and since every session is pure-label, any statistic computed by grouping on it is a
function of the group's label, not just the row's own behavior.

*Why the leave-one-day-out check was insufficient, and why that matters:* we recomputed the
repeat-count statistic independently inside each LOTO fold and got near-perfect held-out PR-AUC
(0.997–1.000) on days never seen in training, and initially read that as ruling out session-
identity leakage. It does not. The leak is not a memorized session ID — it is the rule *"high
repeat count → attack,"* and that rule holds in every fold because every fold shares the same
pure-composition structure. **A test that a broken model passes is not a test.**

*The decisive experiment:* we merged one pure-benign and one pure-attack test session under a
single synthetic `session_id`, recomputed the feature on the merged group, and rescored with the
trained model. **PR-AUC collapsed from 0.9999 to 0.082, ROC-AUC to 0.51 (chance), recall to 0%.**
Digging into why: merging caused the benign rows' own internal repetition to coincidentally match
some of the same coarse numeric signatures (length/entropy/label-count, not domain text) that the
attack tool happens to produce, and the model — trained only on pure-composition sessions — had no
mechanism to handle that and collapsed toward scoring everything near zero.

*The precise claim, stated carefully:* a model trained on pure-composition sessions does not
transfer to mixed sessions, because the feature's meaning depends on session composition. We are
**not** claiming the stronger version — "the feature carries no behavioral signal at all" — that
was not shown; the collapse mixes two distinct effects (loss of the feature's intended meaning,
and plain distribution shift from a genuinely different input distribution), and we have not
isolated which dominates. A single-feature check (`sl2_session_repeat_ratio` alone, on pure
sessions) reached PR-AUC 0.88 — but that number was measured on pure sessions, so it is
contaminated evidence *about* the contamination, not a mitigating fact, and we report it only as a
diagnostic, never as a partial defense of the feature.

**Experiment 2 — Track 2 cascade's session-level confirmation (retained, reframed, §10).** Stage
2's "confirmed → alert whole session" rule gives 100% recall by construction whenever an attack
session escalates and is confirmed, because every row in a pure-attack session is, definitionally,
an attack row. That 100% is not evidence the confirmation logic works on partially-attack traffic
— it is evidence the dataset has no partially-attack sessions to test it against. We keep the
cascade (it is genuinely useful as an interpretable escalation-and-confirmation step, §10) but do
not present its 100% recall — or the 93.9% precision that comes with it — as proof of anything
beyond "it worked on the exact sessions available, n=3." Both figures are products of the same
construction, and §10's table now carries that caveat inline rather than adjacent to it.

**Experiment 3 — window-level alerting (§8, reported, not adopted).** Same mechanism: a window's
label is "contains any attack row," which on single-composition captures is nearly identical to
the capture's own label. The mixed-traffic stress test is the direct confirmation: recall collapses
to 0% (tuned setting) or partially survives at 18.2% (looser setting) once attack traffic is
diluted into a realistic stream, exactly mirroring Experiment 1's collapse.

**Experiment 4 — the feature space itself supports coarse separation, not fine ranking.** This one
is not about session composition; it is the other half of why the numbers look the way they do,
and it unifies three otherwise-unrelated observations. Below the top ~141 rows, the model's
precision is *constant* at ~62.6% across every operating point from 4% to 28% FPR (§6b) — a fixed
likelihood ratio of ~1.61, meaning the flagged pool has the same attack/benign mixture no matter
where the threshold sits. The same limitation shows up as: (a) heavy score quantization — 71,012
benign rows collapse into 4,289 distinct scores, leaving a 15-point FPR band unreachable at any
threshold (§7); (b) light and heavy recall nearly identical at every operating point (10.64% vs
10.80%, 46.22% vs 47.14%), because both categories land in the same coarse score bands; and (c)
PR-AUC stuck at 0.627 despite the model having clearly learned *something* real (ROC-AUC 0.810,
and genuine 2.35× enrichment at the head of the queue). One cause underneath all three: **14
coarse, largely integer-valued per-query features can tell "somewhat attack-like" from "not," but
cannot finely order rows within either group.** This is a property of the supplied stateless
feature set, not of LightGBM — and it bounds what any row-level model on these features can
achieve, which is the honest frame for §7's modest headline number.

## 10. Track 2 cascade (retained, reframed as an interpretability layer)

Stage 1 = §7's v1 model. Escalation trigger: fraction of a session's rows scoring above the
original paper's own "suspicious" band (raw score >= 0.4) exceeding delta=0.5 (tuned on val_thr's
3 sessions). Stage 2 = transparent majority vote over the 3 stateful features with the largest
train-set benign/attack mean gap (`sf_ttl_mean`, `sf_rr_name_length`, `sf_ttl_variance`) —
deliberately not a fitted ML classifier, since only 3 benign captures exist in train.

> ### ⚠️ Read the right-hand column with §9 in hand
> **The two-stage numbers below are not a detection result. They are arithmetic.** Every capture
> file in this dataset is single-composition (§3, §9), so once a session is confirmed, blanket-
> alerting its rows is *guaranteed* to catch 100% of its attack rows — there are no benign rows in
> an attack session to get wrong. 100% recall and 93.9% precision are what that construction
> produces on this data; they are **not** evidence the confirmation logic works on mixed traffic,
> because this dataset contains no mixed sessions to test it against. This is **Experiment 2 of
> the three pure-composition artifacts catalogued in §9** — the same root cause that retired a
> PR-AUC 0.99999 feature and a 45.7%-recall window rule.

| | One-stage (Stage 1 v1 only) | Two-stage (cascade) | How to read the two-stage column |
|---|---|---|---|
| Recall (combined) | 10.78% | 100% | **100% by construction** — pure-composition captures, n=2 attack sessions |
| FPR | 4.12% | 4.12% (unchanged) | Unchanged only because the **single** benign test session didn't escalate (n=1) |
| Precision | 62.6% | 93.9% | Same construction; would fall on any session mixing benign + attack rows |
| Light recall | 10.64% | 100% | 100% by construction (one light session, pure) |
| Heavy recall | 10.80% | 100% | 100% by construction (one heavy session, pure) |
| **Sessions escalated** | — | **2 of 3 test sessions** | Session-level n, not row-level |
| **Test sessions total** | — | **3 (2 attack, 1 benign)** | Every session-level claim here rests on n=3 |

Both columns are at the §7 headline threshold (0.713607). **The statistically meaningful column is
the left one** — 61,567 independent benign rows behind it, versus three correlated sessions behind
the right one. Stage 2's genuine contribution is not the recall figure: it is that a session flagged
only weakly at row level (10.78%) receives an *explainable* verdict from three named, human-checkable
stateful features, with an explicit "unconfirmed → human review" state when they disagree.

**Escalation margins — how close the rule is to misfiring** (`cascade_report.json:escalation_margins`).
The escalated-row percentage (38.9%) is unchanged from the rejected v2 model, which initially
looked like a stale artifact. It is not: the cascade genuinely re-ran on v1
(`stage1_model_version: stateless_lgbm_v1` recorded in the artifact). The percentage is unchanged
because the *same two* sessions escalate under both models — but the margins behind that identical
number are very different:

| Session | label | `r_sus` (v1) | margin to δ=0.5 | `r_sus` (rejected v2) |
|---|---|---|---|---|
| benign_heavy_1 | benign | 0.3796 | **−0.120** | 0.0007 |
| heavy_audio | attack | 0.9997 | +0.500 | 0.9980 |
| light_text | attack | 0.9997 | +0.500 | 0.9997 |

Under v1, the benign session sits **0.12 below the escalation cut**, versus 0.50 under v2. The
escalation rule is far more fragile than the unchanged 38.9% suggests: a modest distribution shift
in benign traffic would push it over, escalating benign traffic to Stage 2. With n=1 benign test
session there is no way to estimate how likely that is. Reported because a headline percentage that
does not move is exactly the kind of number that hides a change underneath it (§9).

**We are not claiming this as a compute-saving cascade (§1).** Stage 2 costs ~165 μs per escalated
session — negligible next to Stage 1's per-query cost — so there is no meaningful trade-off to
report between recall and cost. What the cascade genuinely provides: a session flagged only
weakly by Stage 1 (10.78% row-level recall) can still receive a *confirmed, explainable*
session-level verdict from three named, human-checkable stateful features, rather than staying an
unconfirmed weak signal. That is the value — interpretability and confirmation, not efficiency.

**Limitations, in the table above, not buried in prose:** n=3 test sessions, 2 escalated. The
100% recall is real (not test leakage — delta and discriminative features never touch test) but is
read correctly as "this design worked cleanly on the specific sessions available," not a
statistically powerful session-level generalization claim — see §9 for why that number should not
be over-read.

## 11. Latency and throughput

**Environment declared:** OS Windows-11-10.0.26200-SP0; AMD64 Family 23 Model 104 Stepping 1,
AuthenticAMD, 12 logical CPUs; Python 3.12.5; LightGBM 4.7.0, scikit-learn 1.7.2, numpy 2.3.2;
concurrency 1 (single-threaded, sequential calls).

`src/latency/harness.py`, LightGBM stateless classifier, **14 base features — matches the v1
model actually reported as the headline** (no mismatch between the model measured and the model
claimed, unlike the pre-revert version of this report). Median and p95 computed from per-call
latency distributions, not a single averaged figure:

| Mode | Mean | Median | p95 | Throughput |
|---|---|---|---|---|
| Single-query (batch=1) | 2,228.4 μs/query | 2,221.7 μs/query | 2,751.3 μs/query | 448.8 QPS |
| Batched (batch=128) | 22.3 μs/query | 21.9 μs/query | 28.3 μs/query | 44,855.7 QPS |

Batching gives roughly two orders of magnitude throughput improvement, almost entirely from
amortizing per-call inference overhead.

**Stage 2 added cost** (`src/latency/stage2_harness.py`):

| Cost component | When paid | Mean | Median | p95 |
|---|---|---|---|---|
| One-time stateful context build (18 files) | Once per enrichment refresh, offline | 27.8s (n=3: 27.2–28.4s) | n/a (n too small) | n/a |
| Marginal per-escalated-session decision | Once per escalating session | 164.8 μs | 145.5 μs | 272.1 μs |

**Window aggregation added cost** (§8): ~9.5 μs per 200-row window (≈0.05 μs/query amortized) —
negligible next to Stage 1's inference cost.

### Timings including and excluding feature extraction

Brief requirement: *"timings both including and excluding feature extraction or enrichment."*

**This submission performs no per-query feature extraction, and we state that plainly rather than
report a number that does not exist.** CIC-Bell-DNS-EXF-2021 ships *pre-computed* stateless feature
tables; the 14 gated features are read from those CSVs, with only two trivial derivations
(`sl_longest_word_len`, a string length; `sl_numeric_ratio`, a division). All of it happens once at
load time in `src/data/load.py`, not per query.

| | Mean | Median | p95 |
|---|---|---|---|
| Inference only, batch=1 | 2,225.5 μs/query | 2,218.9 μs/query | 2,748.4 μs/query |
| Inference + per-query array prep, batch=1 | 2,228.4 μs/query | 2,221.7 μs/query | 2,751.3 μs/query |
| Inference only, batch=128 | 19.4 μs/query | 19.2 μs/query | 25.4 μs/query |
| Inference + per-query array prep, batch=128 | 22.3 μs/query | 21.9 μs/query | 28.3 μs/query |

The difference between the two rows in each pair is **2.9 μs/query** (median 2.7, p95 2.9) — but
that is only the `float32` array cast, which we report honestly as array preparation, **not** as
feature extraction.

**The real extraction cost, measured** (`results/metrics/extraction_cost.json`): parsing all 36
CSVs and deriving the 14 features for 726,810 rows takes **14.91 s** (3 runs: 13.75–16.16 s), which
amortizes to **20.5 μs/query**. Paid once per corpus load, not per query.

**What this does not measure, and we will not invent:** a production deployment would parse raw DNS
packets off the wire and compute these 14 features per query. That cost is real and is *not*
represented anywhere in these numbers, because this dataset supplies no PCAPs
(`provenance.json:has_pcaps: false`) — so we have nothing to measure it against. Any per-query
extraction figure we quoted for a production pipeline would be fabricated.

## 12. Failure analysis

`src/eval/failure_analysis.py`, v1 model, at the **headline threshold 0.713607** (§7). Full detail
in `results/metrics/failure_analysis.json`. At this operating point v1 has **35,039 false negatives
and 2,535 false positives** on test — matching §7's confusion matrix exactly.

> **Correction from an earlier draft.** This section previously reported 39,184 FN / 50 FP, which
> were the counts at the *abandoned* FPR=0.1% threshold (0.7262) rather than the headline point.
> The model artifact used by the prototype was also still pinned to that threshold, so the product
> was operating at 0.229% recall while the report claimed 10.78%. Both are now fixed and the
> threshold record (`models/stateless_threshold_v1.json`) carries the achieved val_thr FPR and
> selection rule so the two cannot silently diverge again.

**Worst 5 false negatives** (lowest raw score among missed attack rows):

| unit_id (truncated) | category | raw score | `sl_len` | `sl_entropy` | `sl_labels` | `sl_fqdn_count` |
|---|---|---|---|---|---|---|
| `light_text.pcap.csv_3245` | light | 0.00001 | 13 | 2.65 | 3 | 17 |
| `heavy_audio.pcap.csv_2080` | heavy | 0.00001 | 6 | 2.15 | 3 | 10 |
| `heavy_audio.pcap.csv_2081` | heavy | 0.00018 | 13 | 3.11 | 3 | 16 |
| `heavy_audio.pcap.csv_2176` | heavy | 0.00018 | 13 | 3.11 | 3 | 16 |
| `heavy_audio.pcap.csv_11143` | heavy | 0.05647 | 8 | 2.03 | 3 | 12 |

**Pattern:** every worst-FN row is short (`sl_len` 6–13) with unremarkable entropy (2.0–3.1) — this
is the direct, row-level illustration of §6's baseline finding: length has real signal here, and
these are exactly the short, low-signal queries a length-sensitive detector will miss. This is the
honest face of the 10.8%-recall ceiling reported in §7, not an edge case.

**Worst 5 false positives** (highest raw score among benign rows the model wrongly alerted on):

| unit_id (truncated) | raw score | `sl_len` | `sl_entropy` | `sl_labels` | `sl_fqdn_count` |
|---|---|---|---|---|---|
| `benign_heavy_1.pcap.csv_15389` | 0.8787 | 14 | 3.25 | 3 | 18 |
| `benign_heavy_1.pcap.csv_15439` | 0.8676 | 27 | 3.78 | 5 | 31 |
| `benign_heavy_1.pcap.csv_16478` | 0.8676 | 27 | 3.78 | 5 | 31 |
| `benign_heavy_1.pcap.csv_29135` | 0.8676 | 27 | 3.78 | 5 | 31 |
| `benign_heavy_1.pcap.csv_1275` | 0.8146 | 19 | 3.46 | 3 | 23 |

**Pattern:** the false positives are long, high-entropy benign queries (`sl_len` 14–27,
entropy 3.25–3.78) — legitimate traffic that happens to look structurally similar to the longer,
higher-entropy end of the attack distribution. This is the same length/entropy signal from §6
working correctly in general but producing false alarms on the benign queries that sit at its
upper tail — a real, expected precision/recall trade-off at this operating point, not a distinct
bug.

**Note on the worst-5 tables after the threshold correction.** Both tables are unchanged from the
earlier (0.7262) draft, and that is expected rather than an oversight: the worst-5 FNs are the
*lowest-scoring* attack rows (scores 0.00001–0.056), far below either threshold, and the worst-5
FPs are the *highest-scoring* benign rows (0.815–0.879), above both. Moving the threshold changes
the size of each population — false positives grew from 50 to 2,535 — but not its extreme tail.
Both pattern claims above were re-checked against the regenerated artifact rather than carried
over: the FN rows remain short (`sl_len` 6–13) with entropy 2.03–3.11, and the FP rows remain long
and high-entropy.

## 13. Track 3 support: analyst case view (`app/streamlit_app.py`)

Declared primary track remains Track 1 (§1). This is a supporting Track 3 layer over the
already-evaluated Stage 1 detector and cascade — no separate detector, baseline, or metric set of
its own. It loads `models/stateless_lgbm_v1.pkl` and `results/metrics/stateless_model_v1_report.json`
throughout — the rejected v2 bundle is never read — and this was verified by running the app and
reading the rendered Model Card (PR-AUC 0.6269, threshold 0.713607), not by inspecting the source alone.

- **Case Queue tab:** risk-sorted triage queue, per-query SHAP reason codes, decision band, the
  row's cascade session status, and a disposition control (approve / dismiss / escalate) —
  advisory only, no button blocks DNS or takes automated action.
- **Session Evidence tab:** for escalated sessions, the Stage 2 majority vote and each
  discriminative stateful feature's value against the train-set benign/attack midpoint.
- **Model Card tab:** PR-AUC/recall/precision, the leave-one-day-out fold table, global SHAP
  feature importance, and the one-stage-vs-two-stage cascade comparison.
- **Uncertainty states:** the cascade's `escalated_unconfirmed_human_review` and
  `escalated_context_unavailable_fallback_stage1` verdicts surface directly as distinct badges.

## 14. Limitations

- Controlled testbed, not a production network — reinforced concretely by §9's single-composition-
  capture finding across three independent experiments, not just asserted once.
- **Row-level detection gives 10.8% recall at ~4% FPR, or 47.1% recall at ~18% FPR (§7).** There is
  no operating point between those two — a 15-point FPR cliff caused by score quantization (10,662
  benign rows tied at one score) makes the intervening band unreachable at any threshold. Neither
  number is dressed up: 4% FPR is modest recall, and 18% FPR is ~1,795 false alerts per 10k benign
  queries, which many teams could not staff. Light-attack detection remains the harder problem
  (§7b's LOTO finding), and no feature-engineering attempt so far has improved it without relying
  on the single-composition-capture artifact described in §9.
- The ML model's advantage over the non-ML rule baseline is modest: better ranking (ROC-AUC 0.810
  vs 0.700) and +6 points of recall at comparable FPR, but at the tight operating point the rule is
  competitive on precision at equal recall (§6b) — reported rather than framed around a more
  flattering comparison.
- **Alert ranking works only for roughly the top 100–140 alerts** (91.7% precision at depth 48,
  74.0% at 100, then flat at ~62.6% thereafter, §6b). Beyond that depth the score provides fixed
  enrichment and no prioritization, so the workload-reduction claim is bounded to the head of the
  queue — see §9 Experiment 4 for the underlying feature-space limitation.
- The cascade's escalation rule has a −0.12 margin on the single benign test session (§10) — much
  narrower than the identical 38.9% escalation figure implies, and unestimable at n=1.
- Session-repeat features, session-level cascade confirmation, and window-level aggregation all
  degrade or invert under interleaved/mixed traffic (§9) — this is the central, disclosed
  limitation of this entire submission, not a footnote.
- Cascade and window-level claims are backed by 3 test sessions / 308 correlated windows from one
  capture respectively — real results, not test leakage, but explicitly not statistically powerful
  claims (§9, §10).
- `collection_day` is not disjoint across dev-split partitions by design (§4) — mitigated by the
  independent leave-one-day-out check (§7b).
- No claim of generalization to unseen real-world exfiltration tooling, DGA traffic, or production
  DNS infrastructure — out of scope per the brief itself.

## 15. Deliverable status

| Brief deliverable | Status | Location |
|---|---|---|
| 1. Working prototype (benign / light / heavy) | Done, verified in browser | `app/streamlit_app.py` |
| 2. Source & reproducibility package | Done | `requirements.txt`, `scripts/reproduce.py` (asserts against §7, exits non-zero on mismatch) |
| 3. Technical report | This document | `report/TECHNICAL_REPORT.md` |
| 4. Results bundle | Done | `results/bundle/` — predictions keyed on `unit_id`, scoring output, threshold record, manifest |
| 5. Data & model statement | Done, incl. LLM-assistance disclosure | `deliverables/DATA_AND_MODEL_STATEMENT.md` |
| 6. Pitch & demo | **Outstanding** | §9 is the intended narrative spine |

**Verification run before submission:** `pytest` 8/8 green; `python scripts/reproduce.py` passes
with all seven headline metrics matching this document; all 19 numeric claims in this report were
cross-checked against their source files under `results/metrics/`; all 17 project modules import
cleanly; the Streamlit app was launched and its displayed threshold, FPR and recall confirmed equal
to §7.

**One check we could not complete, stated rather than implied:** an end-to-end install on a bare
machine (fresh clone → fresh venv → `pip install` → reproduce). Two attempts failed at
`ReadTimeoutError` against `files.pythonhosted.org` — a network limitation of the development
machine, not a repository defect. Structural clean-room checks did pass: a fresh clone contains
every required file, the model artifact is tracked (not git-ignored), and the documented data tree
matches disk. The unverified link is only whether `pip` can download eight standard PyPI packages.
