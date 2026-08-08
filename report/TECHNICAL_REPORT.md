# Technical Report — AI-Powered DNS Exfiltration & Tunnelling Intelligence

> **Status: DRAFT SKELETON.** Every section below is populated with real, regenerated numbers
> from this repo's own `results/metrics/*.json` — nothing here is a placeholder figure. What's
> still missing is marked `TODO` explicitly: SHAP visuals, the Streamlit UI screenshots, and the
> live demo script/rehearsal. Team: Muneeb (data, splits, leakage audit, latency harness,
> reproducibility) + Serena (baselines, modelling, calibration, cascade, evaluation, this report).

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

LightGBM on the 14 gated stateless features, isotonic calibration fit on `val_cal` only,
operating threshold selected on `val_thr` at FPR=0.1% target.

| | PR-AUC | Recall @ FPR=0.1% | Precision | False alerts / 10k benign |
|---|---|---|---|---|
| Combined | **0.6268** | 0.229% | 65.7% | 7.63 |
| Light | 0.1728 | 0.230% | 14.5% | 7.63 |
| Heavy | 0.6251 | 0.229% | 63.6% | 7.63 |

PR-AUC (0.627) lines up almost exactly with Muneeb's independently-predicted ~0.63 target in
`summary.md` — a useful cross-check that the split/feature/model pipeline is internally
consistent end to end.

**Recall vs. FPR curve** (same model, looser operating points):

| Target FPR | Achieved FPR | Recall | Precision |
|---|---|---|---|
| 1% | 0.99% | 2.58% | 62.4% |
| 0.5% | 0.083% | 0.229% | 63.8% |
| 0.2% | 0.083% | 0.229% | 63.8% |
| 0.1% | 0.083% | 0.229% | 63.8% |

Relaxing the operating FPR 10x (0.1% -> 1%) buys roughly 10x the recall (0.23% -> 2.58%) — a real,
reportable trade-off curve, not a dead end at the strict end.

**Light-attack recall bootstrap CI:** zero-width by construction, not by precision — all 3,479
test-set light-attack rows sit in a single capture trace, so session-grouped bootstrap resampling
only has one cluster to draw from (`stateless_model_report.json:light_recall_bootstrap_ci_note`).
This is exactly why §8 exists as an independent check.

## 8. Leave-one-day-out robustness (mandatory per brief, separate from the dev split)

`src/eval/loto.py`: fresh model per held-out day, threshold picked in-sample per fold
(documented small-N simplification — only 18 sessions total, not enough left per fold for a clean
nested split).

| Held-out day | Rows | Positive rows | Categories present | PR-AUC | Recall @ in-sample threshold |
|---|---|---|---|---|---|
| 2020-11-20 | 142,138 | 0 | benign only | n/a (single class) | n/a |
| 2020-11-21 | 167,720 | 28,694 | benign, **light** | **0.345** | 0.26% |
| 2020-11-22 | 131,098 | 69,531 | benign, light, heavy | 0.748 | 0.21% |
| 2020-11-23 | 99,060 | 49,945 | benign, heavy | 0.715 | 0.32% |
| 2020-11-24 | 186,794 | 115,782 | benign, heavy | 0.800 | 0.25% |

Fold-to-fold PR-AUC: mean 0.652, std 0.180, range 0.345–0.800 (4 folds with both classes).

**Finding:** the light-attack-only day is the weakest fold by a wide margin (0.345 vs. 0.71–0.80
for heavy-only days) — concrete, measured evidence that light attacks are the harder detection
problem, not an assertion carried from the plan doc without data behind it.

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

| | One-stage (Stage 1 only) | Two-stage (cascade) |
|---|---|---|
| Recall (combined) | 0.229% | **100%** |
| FPR | 0.076% | 0.076% (unchanged) |
| Precision | 65.7% | 99.9% |
| Light recall | 0.230% | 100% |
| Heavy recall | 0.229% | 100% |

38.9% of test rows escalated to the expensive path; 61.1% stayed on the cheap Stage-1-only path.

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

`src/latency/harness.py`, empirical benchmark, LightGBM stateless classifier, 14 features:

| Mode | Feature extraction | Inference | Total | Throughput |
|---|---|---|---|---|
| Single-query (batch=1) | 2.68 μs/query | 1,996.2 μs/query | 1,998.87 μs/query | 500.3 QPS |
| Batched (batch=128) | 2.68 μs/query | 16.01 μs/query | 18.68 μs/query | 53,523.9 QPS |

Batching gives a ~107x throughput improvement — almost entirely from amortizing per-call inference
overhead, not feature extraction (which is already negligible at ~2.7 μs/query in both modes).

**TODO:** measure Stage 2's added cost (stateful context lookup + vote) separately, to complete
the "one-stage vs two-stage... latency" requirement quantitatively rather than qualitatively.

## 12. Failure analysis

**False positive example** (`benign_heavy_1`, row `_11239`): raw score 0.552, just above the
0.544 alert threshold. Features: `sl_len`=25, `sl_entropy`=3.79, `sl_labels`=9,
`sl_subdomain_length`=15. A long, multi-label benign query with moderately high entropy — close
enough to attack-like structure on length/label-count alone to cross the threshold, despite
entropy not actually being extreme.

**False negative pattern** (repeats identically across both a light and a heavy capture, e.g.
`light_text_1062` and `heavy_audio_8679`): raw score 0.530, just under threshold. Features:
`sl_fqdn_count`=32, `sl_upper`=32, `sl_lower`=0, `sl_labels`=1, `sl_len`=33, entropy=2.735 — a
single-label, all-uppercase 32-character query. This is a plausible base32/base64-style encoded
exfil chunk, and the *exact same feature vector* recurs across multiple capture files and both
attack categories, suggesting it may be a fixed protocol/handshake string from the exfiltration
tool rather than varying per-payload content — the model treats it as borderline rather than
confidently malicious, likely because its entropy (2.74) isn't extreme and near-identical strings
don't give the model per-instance variation to key on.

**TODO:** pull the full worst-5 FP / worst-5 FN table (brief requirement) rather than one example
each; this section currently has one grounded example per category as a starting point.

## 13. Limitations

- Controlled testbed, not a production network (dataset's own stated boundary) — reinforced
  concretely by §10's single-composition-capture finding, not just asserted.
- Light-attack detection is the harder problem, evidenced quantitatively in §7 (14.5% precision)
  and §8 (0.345 PR-AUC on the light-only day), not just discussed qualitatively.
- Session-level cascade claims are backed by only 3 test sessions — real, not leakage, but
  explicitly not treated as statistically powerful (§10).
- `collection_day` is not disjoint across dev-split partitions by design (§4) — mitigated by the
  independent leave-one-day-out check (§8), not left unaddressed.
- No claim of generalization to unseen real-world exfiltration tooling, DGA traffic, or production
  DNS infrastructure — out of scope per the brief itself.

## TODO before submission

- [ ] SHAP global summary plot + 2-3 per-alert local explanations (Phase 4, not started)
- [ ] Streamlit UI (risk score, top-5 SHAP features, decision band, human-review flag)
- [ ] Full worst-5 FP / worst-5 FN failure table (currently one example each, §12)
- [ ] Stage 2 latency measurement (§11 TODO)
- [ ] Demo script + rehearsal (benign / light / heavy cases, cascade decision, SHAP, metrics table, close on limitations)
- [ ] Results Bundle deliverable (organizer test-ID predictions, scoring output, threshold record)
- [ ] Data & Model Statement deliverable (snapshot version + SHA-256, no external services — data mostly assembled in `results/metrics/provenance.json`, needs to be written up as its own deliverable doc)
