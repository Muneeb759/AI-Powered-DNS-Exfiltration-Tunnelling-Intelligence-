# Technical Report — AI-Powered DNS Exfiltration & Tunnelling Intelligence

> **Status: DRAFT.** Every section below is populated with real, regenerated numbers from this
> repo's own `results/metrics/*.json` — nothing here is a placeholder figure. SHAP explanations
> and the Streamlit analyst case view (§14) are implemented and verified running end-to-end. What's
> still missing is marked `TODO` explicitly: the full failure table, Stage 2 latency, the live demo
> rehearsal, and the two packaged deliverable docs. Team: Muneeb (data, splits, leakage audit,
> latency harness, reproducibility) + Serena (baselines, modelling, calibration, cascade,
> evaluation, SHAP, Streamlit UI, this report).

---

## 1. Problem and target user

Network defenders and administrators of small or resource-constrained networks need to detect
DNS-based data exfiltration and tunnelling without the compute budget for a heavyweight,
always-on stateful analysis pipeline. This project targets exactly that user: a defender who
wants a cheap, always-on first-pass detector, with an expensive deeper look reserved only for
traffic that already looks elevated — and an honest accounting of what that trade-off costs in
missed attacks versus what it saves in compute.

We chose **Track 2 — Lightweight & Tiered Detection**, with the row-level stateless model as the
always-on Track 1 detector underneath it (see §9 for why the decision unit is row-level, not
session-level, and why that was a data-driven pivot from the original plan, not a shortcut).

## 2. System architecture

```
DNS query row
      |
      v
+----------------------------+
| Stage 1: stateless LightGBM|  <- always runs, cheap (14 gated features)
| calibrated (isotonic,      |
| val_cal-only)              |
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
| Stage 2: pooled stateful   |  <- only for escalated sessions (38.9% of test rows)
| context, majority vote     |
| over top-3 discriminative  |
| features (train-set only)  |
+----------------------------+
      |
      v
confirmed -> alert whole session   |   unconfirmed -> human review   |   context unavailable -> Stage 1 verdict stands
```

Rule baseline, Stage 1, and Stage 2 all sit on top of the same Phase 1 data contract
(`data/splits/{train,val_cal,val_thr,test}.parquet`), built and independently re-verified end to
end (`scripts/handoff_check.py`, `pytest` 6/6 green) against the real CIC-Bell-DNS-EXF-2021
dataset.

## 3. Data used

- **Dataset:** CIC-Bell-DNS-EXF-2021 (Mahdavifar et al., 2021, ICCNS), public UNB release, 36
  files, 103,384,358 bytes, SHA-256-verified per file (`results/metrics/provenance.json`).
- **Provenance clarification:** Sofstica confirmed on the hackathon Discord (2026-08-08) that
  there is no separate frozen organizer snapshot beyond the official dataset link in the brief's
  resources section — this is the dataset of record for judging, not a stand-in.
- **Scale:** 757,211 stateless rows / 262,105 stateful rows across 18 capture files, reconciled
  exactly against the official UNB Table 2 category statistics (once the shared day-20/21 benign
  pool that the official table double-counts across Heavy-Benign and Light-Benign is accounted
  for — verified row-count arithmetic, not assumed).
- **No row-level or sub-file join exists** between the stateless and stateful tables (verified
  directly: the same capture has 35,795 stateless rows vs 10,735 stateful rows with zero shared
  column names, and stateless row order is not consistently chronological across files — see §9).

## 4. Split protocol

`capture_file_grouped_split` (`src/data/splits.py`), not the originally planned leave-one-day-out
day-grouped split — day-grouping left validation partitions with zero light-attack rows (data
reality, not a preference; see `docs/PHASE1_AUDIT.md` M1).

| Partition | Rows | Benign | Positive | Light | Heavy | Min FPR floor |
|---|---|---|---|---|---|---|
| train | 405,851 | 281,164 | 124,687 | 11,345 | 113,342 | 3.56e-6 |
| val_cal | 95,102 | 49,115 | 45,987 | 10,241 | 35,746 | 2.04e-5 |
| val_thr | 125,016 | 71,012 | 54,004 | 17,618 | 36,386 | 1.41e-5 |
| test | 100,841 | 61,567 | 39,274 | 3,479 | 35,795 | 1.62e-5 |

**Guarantees actually enforced:** `session_id` (capture-file) disjointness, dual-category
composition (every partition has benign+light+heavy), `unit_id` uniqueness — all verified by
`assert_split_protocol()` and covered by `tests/test_splits_real.py`.

**Guarantee this split does *not* make:** `collection_day` is not disjoint across partitions
(e.g. `2020-11-21` appears in train, val_cal, and val_thr). This is measured and reported
explicitly (`results/metrics/split_summary.json:day_overlap`), not glossed over — a day-disjoint
split was tried first and abandoned for the reason above. Since `collection_day`/`timestamp` are
excluded features, this does not open a leakage path; it does mean day can't be used as an
independent robustness axis for *this* split, which is exactly why §8's leave-one-day-out check is
a separate evaluation loop, not folded into the dev split.

Day `2020-11-25` is excluded entirely (0 benign rows, single-class slice, FPR/PR-AUC undefined) —
this drops 30,401 heavy-attack rows from all partitions, a deliberate and documented trade-off,
not silent data loss.

## 5. Excluded leakage fields

Allowlist-based gate (`configs/leakage_exclusions.yaml`, enforced by `LeakageGate` and
`tests/test_leakage.py`):

| Field | Reason |
|---|---|
| `timestamp` | Leaks collection day / execution timing |
| `sld` | Raw domain/IP-octet/NetBIOS identity shortcut (59% of positive rows and 38% of benign rows contain IP fragments or NetBIOS junk strings, not real domains — verified in `docs/PHASE1_AUDIT.md` B3) |
| `longest_word`, `subdomain` | Raw string tokens carrying query-identity text |
| `capture_file`, `session_id`, `collection_day`, `unit_id` | Grouping/identifier keys, not features |
| `distinct_ip`, `reverse_dns`, `distinct_domains`, `unique_asn`, `unique_country`, `unique_ttl`, `rr_type` | Stateful-table raw identity/list fields — replaced by safe derived counts/flags/one-hot in `src/features/stateful_context.py` |
| 8 constant stateful columns (`NS_frequency`, `CNAME_frequency`, ...) | `nunique == 1` across the entire dataset — verified, not assumed |

**Sanity check (shortcut ablation, mandatory per brief):** a model trained with the excluded
identity shortcuts included (`sc_sld`, `sc_longest_word`, `sc_day`, `sc_session`) reaches
**PR-AUC 0.6946** vs the gated model's **0.6349** — a real but modest +0.0597 inflation, not a
catastrophic one, meaning the exclusions are doing real work without implying the raw dataset was
trivially all-shortcut (`results/metrics/leakage_ablation.json`).

## 6. Baseline

Mandatory non-ML rule baseline: `z(sl_len) + z(sl_entropy)`, normalization constants fit on
`train` only, threshold picked on `val_thr` at FPR=0.1%.

| | PR-AUC | ROC-AUC (supp.) | Recall @ FPR=0.1% | False alerts / 10k benign |
|---|---|---|---|---|
| Combined | 0.366 | 0.499 | 0.0% | 1.95 |
| Light | 0.051 | 0.503 | 0.0% | 1.95 |
| Heavy | 0.345 | 0.498 | 0.0% | 1.95 |

Essentially no better than chance at this operating point — expected and honestly reported, not a
bug. It exists to show the ML model's lift, not as a competitive detector.

## 7. Stateless model (Stage 1 / Track 1 headline)

LightGBM (`model_version: stateless_lgbm_v2`) on the 14 gated stateless features plus 8 engineered
features (`src/features/stateless_engineered.py`), `scale_pos_weight` class balancing, isotonic
calibration fit on `val_cal` only, operating threshold selected on `val_thr` at FPR=0.1% target.

**Engineered features (v2):** six interaction/ratio terms over the base `sl_*` columns
(`sl2_subdomain_entropy`, `sl2_alpha_ratio`, `sl2_special_ratio`, `sl2_label_depth_entropy`,
`sl2_numeric_subdomain_density`, `sl2_max_label_fraction`), plus two features that turned out to
carry almost all of the lift: `sl2_session_repeat_count` / `sl2_session_repeat_ratio` — how many
times a row's exact base feature vector recurs within its own session, computed via a `groupby`
on `session_id` (used only as a grouping key, never as a feature value, matching the pattern
already used for `sf_*` pooling in `src/features/stateful_context.py`).

**Why this feature exists:** a diagnostic pass found that the 39,274 attack rows in `test` reduce
to only 47 unique base feature combinations (0.1%) — the exfiltration tooling in this dataset
replays a small set of query shapes at high volume, while benign traffic rarely repeats a shape
at all (`sl2_session_repeat_count` train-set means: benign ≈1,882, attack ≈6,866). No individual
row-level feature in the v1 set exposed this; `sl2_session_repeat_count` does.

| | PR-AUC | Recall @ FPR=0.1% | Precision | False alerts / 10k benign |
|---|---|---|---|---|
| Combined | **0.99999** | 97.21% | 100% | 0.0 |
| Light | 0.99986 | 97.41% | 100% | 0.0 |
| Heavy | 0.99999 | 97.19% | 100% | 0.0 |

(v1, base 14 features only, for comparison: PR-AUC 0.6268, recall 0.229% at the same FPR target.)

**Recall vs. FPR curve** (v2 model, looser operating points):

| Target FPR | Achieved FPR | Recall | Precision |
|---|---|---|---|
| 1% | 0.70% | 99.99% | 98.91% |
| 0.5% | 0.50% | 99.99% | 99.23% |
| 0.2% | 0.16% | 99.98% | 99.75% |
| 0.1% | 0.075% | 99.91% | 99.88% |

Recall is already effectively saturated at every tested FPR target — the trade-off curve that was
steep in v1 is now flat, because the repeat-count signal separates the classes almost completely.

### How we ruled out leakage before trusting this number

A jump from PR-AUC 0.63 to 0.9999 is the kind of result that should be treated as suspicious by
default, not celebrated. `session_id` is a grouping key, and this dataset's capture files are
single-composition (§10) — grouping by session and getting a near-perfect signal is *exactly*
the shape a session-identity leak would take. Before accepting the number, we ran the specific
check that would falsify it: **leave-one-day-out (§8) with the repeat-count features recomputed
independently inside each fold**, so a held-out day's sessions contribute zero repeat-statistics
to that fold's training data. If the v1→v2 jump were session fingerprinting, held-out-day PR-AUC
would collapse back toward v1's baseline on sessions the model never saw. It doesn't: LOTO PR-AUC
is 0.997–1.000 across all 4 testable days (mean 0.9993, std 0.0011; see §8). That is the evidence
the lift is a generalizable behavioral pattern (attack sessions replay identical query shapes;
benign sessions don't), not a memorized session identity.

**Light-attack recall bootstrap CI:** zero-width by construction, not by precision — all 3,479
test-set light-attack rows sit in a single capture trace, so session-grouped bootstrap resampling
only has one cluster to draw from (`stateless_model_report.json:light_recall_bootstrap_ci_note`).
This is exactly why §8's independent day-based check carries the real weight of the claim.

## 8. Leave-one-day-out robustness (mandatory per brief, separate from the dev split)

`src/eval/loto.py`: fresh model per held-out day (v2 feature set — base + engineered, with
`sl2_session_repeat_count/ratio` recomputed independently inside each fold, see §7), threshold
picked in-sample per fold (documented small-N simplification — only 18 sessions total, not enough
left per fold for a clean nested split).

| Held-out day | Rows | Positive rows | Categories present | PR-AUC | Recall @ in-sample threshold |
|---|---|---|---|---|---|
| 2020-11-20 | 142,138 | 0 | benign only | n/a (single class) | n/a |
| 2020-11-21 | 167,720 | 28,694 | benign, **light** | **0.9996** | 100% |
| 2020-11-22 | 131,098 | 69,531 | benign, light, heavy | 1.0000 | 100% |
| 2020-11-23 | 99,060 | 49,945 | benign, heavy | 1.0000 | 100% |
| 2020-11-24 | 186,794 | 115,782 | benign, heavy | 0.9974 | 94.80% |

Fold-to-fold PR-AUC: mean 0.9993, std 0.0011, range 0.9974–1.0000 (4 folds with both classes).

**Finding:** with the v1 (base-only) feature set, the light-attack-only day was the weakest fold
by a wide margin (PR-AUC 0.345 vs. 0.71–0.80 for heavy-only days), concrete evidence that
row-level light-attack detection was the harder problem under that feature set. With v2's
session-repeat features (§7), that gap closes almost entirely — 2020-11-21 (light-only) now
scores 0.9996, on par with the heavy-only folds. This LOTO result is also the leakage check for
the repeat-count feature itself: each fold's repeat statistics are computed fresh from only that
fold's own rows, so a held-out day's sessions were never seen by the model in any form before
scoring, and the near-perfect PR-AUC still holds.

## 9. Why Track 2 is a cascade, not a cancelled track

Early in Phase 1, a data audit found no row-level join between the stateless and stateful tables,
and that a capture-file-level session unit (only 18 files, 1-2 benign per day) makes FPR=0.1%
unmeasurable at that granularity. That finding is real and was independently re-verified
(zero shared columns between a stateless/stateful pair; no reliable row ordering to reconstruct
sub-file windows either). It was initially read as grounds to drop Track 2 entirely and pivot the
decision unit to row-level.

The brief itself treats this differently: Track 2's own success criteria list *"graceful behavior
when stateful context or enrichment is unavailable"* as a strength, and the "what the data can/cannot
support" section explicitly anticipates that *"some stateful features depend on historical windows
or context... a real-time claim must explain how that context would be obtained and at what
cost"* — i.e. exactly this constraint is what Track 2 wants engaged with, not avoided.

**Resolution:** the row-level pivot for the always-on detector stands (§7 headline numbers), and
Track 2 is delivered as a cascade on top of it, escalating to session-granularity stateful context
— the finest join level the data actually supports — with an explicit graceful-degradation path
when that context is unavailable. See §10.

## 10. Track 2 cascade

Stage 1 = §7's stateless model. Escalation trigger: fraction of a session's rows scoring
above the original paper's own "suspicious" band (raw score >= 0.4) exceeding delta=0.5 (tuned on
val_thr's 3 sessions). Stage 2 = transparent majority vote over the 3 stateful features with the
largest train-set benign/attack mean gap (`sf_ttl_mean`, `sf_rr_name_length`, `sf_ttl_variance`) —
deliberately not a fitted ML classifier, since only 3 benign captures exist in train.

| | One-stage (Stage 1 v2 only) | Two-stage (cascade) |
|---|---|---|
| Recall (combined) | 97.21% | **100%** |
| FPR | 0.0% | 0.0% (unchanged) |
| Precision | 100% | 100% |
| Light recall | 97.41% | 100% |
| Heavy recall | 97.19% | 100% |

38.9% of test rows escalated to the expensive path; 61.1% stayed on the cheap Stage-1-only path.
With v2's session-repeat features, Stage 1 alone now closes almost all of the recall gap Stage 2
used to cover (was 0.229% → 100%, an entirely Stage-2-driven jump, under v1); the cascade's
remaining contribution is the last ~2.8 percentage points plus the interpretable stateful
confirmation step, not carrying the whole detection burden by itself.

**Two limitations reported directly in the artifact, not left implicit:**
1. All 18 capture files in this dataset are single-composition (100% benign or 100% attack — verified
   programmatically, none mixed). Blanket-alerting a confirmed session's rows is well-matched to
   that structure but is a testbed artifact: a real session mixing normal queries with a slow,
   low-volume exfil channel would see every benign query in that session flagged too, producing a
   much higher false-alert rate than measured here. This maps directly onto the brief's own
   "controlled environment, not a production network" caveat.
2. Test has only 3 capture sessions (2 attack, 1 benign); delta was tuned on val_thr's 3 sessions.
   Real result, not test leakage (delta and discriminative features never touch test), but at n=3
   sessions it is not a statistically powerful session-level generalization claim. §7's row-level
   numbers (tens of thousands of rows) remain the primary, statistically meaningful headline.

## 11. Latency and throughput

**Environment declared** (brief requirement: "declared hardware, software versions, batch size,
and concurrency"), from `results/metrics/latency_baseline.json:environment` /
`results/metrics/stage2_latency_report.json:environment`:

| | |
|---|---|
| OS | Windows-11-10.0.26200-SP0 |
| Processor | AMD64 Family 23 Model 104 Stepping 1, AuthenticAMD, 12 logical CPUs |
| Python | 3.12.5 |
| LightGBM | 4.7.0 · scikit-learn 1.7.2 · numpy 2.3.2 |
| Concurrency | 1 (single-threaded, sequential calls — no concurrent/async execution measured) |

`src/latency/harness.py`, empirical benchmark, LightGBM stateless classifier, 14 base features.
**Median and p95 are computed from per-call latency distributions** (`n=500` single-query calls,
`n=100` batches of 128, `n=1000` extraction calls), not derived from a single averaged wall-clock
figure — this is the fix that closes the brief's explicit "median and p95 processing latency"
requirement, which the previous mean-only version did not satisfy:

| Mode | Mean | Median | p95 | Throughput |
|---|---|---|---|---|
| Single-query (batch=1) | 2,228.4 μs/query | 2,221.7 μs/query | 2,751.3 μs/query | 448.8 QPS |
| Batched (batch=128) | 22.3 μs/query | 21.9 μs/query | 28.3 μs/query | 44,855.7 QPS |

(Absolute values shift run-to-run with background system load — a documented property of
wall-clock measurement on a shared OS, not a modelling change; batching's ~100x throughput
multiplier over single-query mode is the stable, reportable finding.)

Batching gives roughly two orders of magnitude of throughput improvement — almost entirely from
amortizing per-call inference overhead, not feature extraction (which stays under a few
microseconds per query in both modes).

**Stage 2 added cost** (`src/latency/stage2_harness.py`, `results/metrics/stage2_latency_report.json`):

| Cost component | When paid | Mean | Median | p95 |
|---|---|---|---|---|
| One-time stateful context build (18 capture files, `ast.literal_eval` parsing of ~262K stateful rows) | Once per enrichment refresh — batch/offline, not on the query path | 27.8s (n=3: 27.2–28.4s) | n/a — 3 repeats too few for a percentile; min/max shown instead | n/a |
| Marginal per-escalated-session decision (dict lookup + majority vote over 3 features) | Once per escalating session (18 sessions total in this dataset) | 164.8 μs | 145.5 μs | 272.1 μs (n=2,000) |

**One-stage vs. two-stage comparison:** Stage 1 alone costs 2,228.4 μs/query mean (2,751.3 μs p95,
batch=1). Stage 2's marginal cost (164.8 μs mean, 272.1 μs p95) is paid once per *session*, not
once per query — spread across even a single escalated session's rows, the per-query overhead it
adds is negligible, and 38.9% of test rows belong to an escalated session
(`results/metrics/cascade_report.json`). The real cost driver is the one-time context build
(27.8s), and it is explicitly a batch/offline operation run against the enrichment source, not
part of the live per-query decision path — this is the "measured compute savings" the Track 2
brief asks for: the expensive step is amortized across an entire session's traffic, not repeated
per row.

## 12. Failure analysis

`src/eval/failure_analysis.py` (v2 model, threshold 0.99985). Full detail in
`results/metrics/failure_analysis.json`.

**False positives: zero, at the current operating threshold** (precision 100% on `test`, see §7).
Rather than invent an FP example, the table below shows the 5 highest-scoring BENIGN rows in
test — the closest calls that never crossed the alert threshold — explicitly labelled as
near-misses, not actual failures.

**Worst 5 false negatives** (lowest raw score among missed attack rows — the model was most
confident these were benign):

| unit_id (truncated) | category | raw score | `sl2_session_repeat_count` | `sl_entropy` | `sl_len` |
|---|---|---|---|---|---|
| `heavy_audio.pcap.csv_2080` | heavy | 0.0011 | 1.0 | 2.15 | 6 |
| `light_text.pcap.csv_3245` | light | 0.0015 | 1.0 | 2.65 | 13 |
| `heavy_audio.pcap.csv_28369` | heavy | 0.0080 | 2.0 | 2.53 | 11 |
| `heavy_audio.pcap.csv_28370` | heavy | 0.0080 | 2.0 | 2.53 | 11 |
| `heavy_audio.pcap.csv_2081` | heavy | 0.0145 | 2.0 | 3.11 | 13 |

**Pattern:** every worst-FN row has `sl2_session_repeat_count` of 1–2 — these are the rare
non-repeating queries inside otherwise-repetitive attack sessions (§7's dominant feature). The
model's strongest signal is repetition; a session's few one-off queries don't carry it, so they
score low even though the session overall is confidently flagged. This is a direct, quantitative
illustration of the §13 limitation: the repeat-count signal is a session-level behavioral
pattern, and individual non-repeating rows within an attack session are its blind spot.

**Top 5 highest-scoring benign rows (near-misses, not false positives):**

| unit_id (truncated) | raw score | `sl2_session_repeat_count` | `sl_entropy` | `sl_len` |
|---|---|---|---|---|
| `benign_heavy_1.pcap.csv_11239` | 0.9969 | 20.0 | 3.79 | 25 |
| `benign_heavy_1.pcap.csv_15396` | 0.9969 | 20.0 | 3.79 | 25 |
| `benign_heavy_1.pcap.csv_15496` | 0.9969 | 20.0 | 3.79 | 25 |
| `benign_heavy_1.pcap.csv_16475` | 0.9969 | 20.0 | 3.79 | 25 |
| `benign_heavy_1.pcap.csv_16517` | 0.9969 | 20.0 | 3.79 | 25 |

**Pattern:** the same repeated benign query (`benign_heavy_1`, repeat_count=20, long/high-entropy
subdomain) is the closest this model comes to a false alarm — a benign host that legitimately
issues the same somewhat unusual-looking query 20 times still scores well below the ~7,000+
repeat counts typical of attack sessions (§7), so it never crosses the threshold. This is the
concrete evidence behind §13's limitation that a production session with more benign repetition
than this testbed's benign traffic could erode that margin.

## 13. Limitations

- Controlled testbed, not a production network (dataset's own stated boundary) — reinforced
  concretely by §10's single-composition-capture finding, not just asserted.
- **The v2 session-repeat features (§7) owe their strength to this dataset's structure, and we
  say so plainly rather than let a near-perfect number speak for itself:** exfiltration tooling in
  this testbed replays a small, fixed set of query shapes at high volume within a session, and
  every capture file is single-composition (§10). A production network with mixed-traffic
  sessions and more heterogeneous exfil tooling would likely show a smaller repeat-count gap
  between benign and attack traffic than the near-total separation measured here. We validated the
  feature is not literal session-identity leakage (LOTO with fold-independent repeat statistics,
  §7–§8), but "not leakage" and "will generalize this cleanly to production traffic" are different
  claims — only the first one is what we're asserting.
- Light-attack detection was the harder problem under the v1 (base-only) feature set — evidenced
  quantitatively in the original run (14.5% precision, 0.345 PR-AUC on the light-only LOTO day).
  v2's repeat-count features close most of that gap (§7, §8); we still flag this because the
  underlying reason light attacks were hard — low per-query signal — hasn't disappeared, it's the
  repeat-count feature compensating for it, which is itself the finding above.
- Session-level cascade claims are backed by only 3 test sessions — real, not leakage, but
  explicitly not treated as statistically powerful (§10).
- `collection_day` is not disjoint across dev-split partitions by design (§4) — mitigated by the
  independent leave-one-day-out check (§8), not left unaddressed.
- No claim of generalization to unseen real-world exfiltration tooling, DGA traffic, or production
  DNS infrastructure — out of scope per the brief itself.

## 14. Track 3 support: analyst case view (`app/streamlit_app.py`)

Declared primary track remains Track 2. This is a supporting Track 3 (Analyst Investigation &
Explanation) layer over the already-evaluated cascade — no separate detector, baseline, or metric
set of its own; every number displayed reads from `results/metrics/*.json`, generated by the
evaluation pipeline. Per the brief's own warning, *"a polished dashboard without validated
detection evidence is not sufficient"* — this view exists because the detection evidence (§7-§10)
already exists, not instead of it.

- **Case Queue tab:** risk-sorted triage queue, per-query SHAP reason codes (`src/explain/shap_explain.py`,
  exact `TreeExplainer` contributions, not approximations), decision band (benign / suspicious /
  alert), the row's cascade session status, and a disposition control (approve / dismiss /
  escalate) that records a UI-session-only choice — no button blocks DNS, quarantines a host, or
  changes any live system, per the project's own advisory-only safety constraint.
- **Session Evidence tab:** for escalated sessions, the Stage 2 majority vote and each
  discriminative stateful feature's value against the train-set benign/attack midpoint — directly
  satisfies Track 3's "comparison with similar benign activity" criterion using artifacts the
  cascade already produces (`cascade.py`'s `detail` field, previously computed but stripped before
  saving; now retained).
- **Model Card tab:** PR-AUC/recall/precision, the leave-one-day-out fold table, global SHAP
  feature importance, and the one-stage-vs-two-stage cascade comparison — the same headline numbers
  as §7-§10, in one place for a judge to check without opening JSON files.
- **Uncertainty states:** the cascade's existing `escalated_unconfirmed_human_review` and
  `escalated_context_unavailable_fallback_stage1` verdicts surface directly as distinct badges —
  this is Track 3's "insufficient evidence" requirement, already produced by the cascade's design
  (§10), not added for the UI.

Verified running end-to-end in the browser (all three tabs, SHAP explanation render, session
expander, disposition buttons) — not just written and assumed to work.

## TODO before submission

- [ ] Demo script + rehearsal (benign / light / heavy cases, cascade decision, SHAP, metrics table, close on limitations)
- [ ] Results Bundle deliverable (organizer test-ID predictions, scoring output, threshold record)
- [ ] Data & Model Statement deliverable (snapshot version + SHA-256, no external services — data mostly assembled in `results/metrics/provenance.json`, needs to be written up as its own deliverable doc)
