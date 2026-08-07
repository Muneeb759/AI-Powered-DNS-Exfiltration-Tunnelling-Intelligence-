# PHASE 1 SUMMARY & HANDOFF CONTRACT — CIC-Bell-DNS-EXF-2021

> **Project:** Sofstica AI Hackathon — *AI-Powered DNS Exfiltration & Tunnelling Intelligence*  
> **Phase:** Phase 1 (Data Engineering, Splits, Leakage Audit, Latency Harness)  
> **Status:** 100% COMPLETE & VERIFIED (`handoff_check.py` EXIT 0, `pytest` 6/6 GREEN)  
> **Track Selection:** Track 1 — Stateless-Only Detection, Row-Level Decision Unit  
> **Dataset Release:** Public UNB Release of CIC-Bell-DNS-EXF-2021  

---

## 1. Executive Summary

Phase 1 establishes a **verified, leakage-gated data contract** for DNS exfiltration detection. 

Following a rigorous Stage A data audit, **Track 2 (Stateful Cascade) was cancelled** because raw DNS captures contain no row-level join key between stateless (757k rows) and stateful (262k rows) tables [B1(d)], and session-level FPR resolution floors reach 50–100% [M2]. The project proceeded under **Track 1 (Stateless-Only, Row-Level Decision Unit)**.

To resolve validation coverage limitations where day-grouped validation partitions contained zero light attacks, the headline split was upgraded to a **capture-file-grouped split across traces**. Every partition (`train`, `val_cal`, `val_thr`, `test`) contains **Light attacks (> 0)**, **Heavy attacks (> 0)**, and **Benign rows (> 0)** while maintaining 100% trace-disjoint isolation.

All 4 parquet splits (`train`, `val_cal`, `val_thr`, `test`), leakage gates, metrics suite, latency harness, and automated unit tests are written, passing, and verified.

---

## 2. Key Architecture & Design Decisions

| Item | Decision | Rationale & Impact on Phase 2 |
|---|---|---|
| **Decision Unit** | Row-level (individual DNS query) | Each query is evaluated independently. Documented in `docs/DECISION_UNIT.md`. |
| **Headline Split** | Capture-file-grouped trace split | Ensures `val_cal` and `val_thr` contain BOTH Light and Heavy attack categories for proper validation and threshold tuning. |
| **Serena Target Baseline** | **PR-AUC ~ 0.63** | Capture-file split is harder and more realistic (unseen trace patterns). Serena must target ~0.63, NOT the inflated ~0.75 from the old day-split. |
| **Light Recall Reporting** | **Mandatory Bootstrap 95% CI** | `test` light positive queries (3,479 rows) belong to a single trace (`light_text`). Serena **MUST** report test light recall with its **1,000-resample bootstrap 95% CI**, NOT a point estimate, and compute **LOTO cross-validation** (R1.2) as a secondary backstop. |
| **Target Operating Point** | `FPR = 0.1%` (10 false alerts / 10k benign) | Resolution floor on test set is $1.62 \times 10^{-5}$ (0.0016%), making FPR=0.1% fully measurable across all partitions. |
| **Leakage Control** | Allowlist-based `LeakageGate` | `sld`, `timestamp`, `longest_word`, and `distinct_ip` excluded to prevent temporal and trace identity shortcuts. |

---

## 3. Partition Split Summary Table (Phase 2 Input Data)

| Partition | Total Rows | Capture Traces | Benign Rows | Positive Exfiltration Rows | Light Positives | Heavy Positives | Positive Rate | Min FPR Floor |
|---|---|---|---|---|---|---|---|---|
| `train` | 405,851 | 9 files | 281,164 | 124,687 | 11,345 | 113,342 | 30.72% | $3.56 \times 10^{-6}$ |
| `val_cal` | 95,102 | 3 files | 49,115 | 45,987 | 10,241 | 35,746 | 48.36% | $2.04 \times 10^{-5}$ |
| `val_thr` | **125,016** | **3 files** | **71,012** | **54,004** | **17,618** | **36,386** | **43.20%** | **$1.41 \times 10^{-5}$** |
| `test` | **100,841** | **3 files** | **61,567** | **39,274** | **3,479 (1 trace)** | **35,795** | **38.95%** | **$1.62 \times 10^{-5}$** |
| **Total** | **726,810** | **18 files** | **462,858** | **263,952** | **42,683** | **221,269** | **36.32%** | — |

---

## 4. Benchmark & Validation Results

### A. Leakage Ablation (`results/metrics/leakage_ablation.json`)
- **Gated Model (14 Stateless Features):** Test PR-AUC = **0.6265**
- **Shortcut-Included Model (Gated + Encoded `sld`, `longest_word`, `session_id`, `day`):** Test PR-AUC = **0.6928**
- **Ablation Delta:** **+0.0663 PR-AUC**
- **Honest Takeaway:** On this public release of CIC-Bell-DNS-EXF-2021, metadata identity shortcuts leak modestly (+0.066 PR-AUC boost), not catastrophically. Feature gating costs ~0.066 PR-AUC but guarantees zero trace-identity shortcuts.

### B. Reconciled Latency Baseline (`results/metrics/latency_baseline.json`)
- **Measurement Mode:** `EMPIRICAL_BENCHMARK` (Trained classifier execution)
- **Feature Extraction Latency:** **4.20 $\mu$s / query**
- **Single-Query Stream Mode (`batch_size = 1`):**
  - Inference Latency: **2,293.18 $\mu$s / query** (2.29 ms per per-row `predict_proba` call due to Python invocation overhead)
  - Total Latency: **2,297.38 $\mu$s / query**
  - Throughput: **435.3 queries / second** (500 queries processed serially in 1.1487 s)
- **Batched Mode (`batch_size = 128`):**
  - Inference Latency: **18.96 $\mu$s / query** (2.42 ms total for batch of 128 queries)
  - Total Latency: **23.16 $\mu$s / query**
  - Throughput: **43,183.1 queries / second** (12,800 queries processed in 0.2964 s)

### C. Domain Security Audit (`results/` & `configs/` Regex Scan)
- Scan command: `grep -rniE 'gstatic|msftncsi|desktop-|1e100|in-f[0-9]|\.(com|net|org|ca|io|local)\b' results/ configs/ docs/`
- **Scan Result:**
  - `results/` folder: **0 matches** (100% CLEAN of hostnames, domain strings, or dataset content)
  - `configs/` folder: **0 matches** (100% CLEAN)
  - `docs/` folder: **3 text matches** (Audit Markdown text documenting why NetBIOS strings like `DESKTOP-3JF04TC` were excluded)

---

## 5. Directory & File Inventory

```
.
├── configs/
│   ├── schema.lock.yaml           # Canonical column mappings & float32 dtypes
│   ├── feature_lists.yaml         # List of 14 allowlisted stateless features
│   └── leakage_exclusions.yaml    # Allowlist prefixes & explicit exclusion reasons
├── docs/
│   ├── PHASE1_AUDIT.md            # Comprehensive Stage A audit findings & metrics
│   ├── DECISION_UNIT.md           # Technical justification for row-level decision unit
│   └── SESSION_CONSTRUCTION.md    # Session grouping rules & sld prohibition analysis
├── src/
│   ├── data/
│   │   ├── verify_snapshot.py     # SHA-256 snapshot provenance generator
│   │   ├── audit_snapshot.py      # Schema and block profiler
│   │   ├── load.py                # Raw CSV loader with assertions & split loader
│   │   ├── splits.py              # LeakageGate, capture_file_grouped_split & protocols
│   │   └── build_splits.py        # Generates train/val_cal/val_thr/test parquet files
│   ├── eval/
│   │   ├── metrics.py             # PR-AUC, pick_threshold_at_fpr, bootstrap CIs
│   │   └── ablation.py            # Gated vs Shortcut leakage ablation model
│   └── latency/
│       └── harness.py             # Latency & throughput benchmark harness
├── tests/
│   ├── conftest.py                # Pytest path resolution
│   ├── test_schema.py             # Schema & dtype validation tests
│   ├── test_splits_real.py        # Parquet partition invariant tests
│   ├── test_no_random_split.py    # Unseeded split / random shuffle grep test
│   ├── test_determinism.py        # Byte-identical parquet output determinism test
│   └── test_leakage.py            # LeakageGate assertion tests
├── scripts/
│   └── handoff_check.py           # Verification contract script for Serena's Phase 2 start
├── results/
│   └── metrics/
│       ├── provenance.json        # Checksum hash map of all 36 snapshot files
│       ├── snapshot_audit.json    # Machine-readable schema audit
│       ├── split_summary.json     # Parquet split row counts & FPR floors
│       ├── leakage_audit.csv      # Machine-generated feature classification table
│       ├── leakage_ablation.json   # Ablation PR-AUC metrics
│       └── latency_baseline.json  # Latency & throughput benchmark results
├── Makefile                       # Automation target Makefile
├── summary.md                     # Handoff summary for Claude review & git push
└── .gitignore                     # Data exclusion rules
```

---

## 6. Manual Git Verification & Commands (Run when ready)

```bash
# Verify domain cleanliness manually
python -c "import re, pathlib; p=re.compile(r'gstatic|msftncsi|desktop-|\.(com|net|org|ca|io)\b', re.I); print('CLEAN' if not any(p.findall(f.read_text(errors='ignore')) for f in pathlib.Path('results').rglob('*') if f.is_file()) else 'INSPECT')"

# Commit and push (after manual verification)
git add configs/ docs/ src/ tests/ scripts/ results/ Makefile summary.md .gitignore
git commit -m "feat(phase1): complete capture-grouped data contract, honest ablation, empirical latency benchmark, and clean domain audit"
git push origin main
```
