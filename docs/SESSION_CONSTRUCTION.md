# SESSION GROUPING SPECIFICATION

## Overview
Under the Track 1 stateless architecture, individual decision units are **row-level DNS queries**.
However, to prevent data leakage and guarantee strict independence between training, calibration, thresholding, and test partitions (Control 1 & Control 2), all records are grouped by **`capture_file` (18 capture trace sessions)** and **`collection_day`**.

## Grouping Rules
1. **Partition Group Key:** `session_id` = `capture_file` (relative path of original `.pcap.csv` trace).
2. **Disjoint Invariant:** No `capture_file` (`session_id`) appears in more than one partition (`train`, `val_cal`, `val_thr`, `test`); this is the disjointness guarantee actually enforced by `assert_split_protocol()`. `collection_day` is **not** disjoint across partitions -- a day-disjoint split was attempted first and abandoned because it left validation partitions with zero light-attack rows (see `PHASE1_AUDIT.md`, M1). The resulting day overlap is measured and reported explicitly by `report_day_overlap()` in `src/data/splits.py` and recorded in `results/metrics/split_summary.json:day_overlap`. Since `collection_day` and `timestamp` are excluded from the feature set entirely (see `configs/leakage_exclusions.yaml`), this overlap does not create a label-leakage path -- it only means day is not usable as an independent robustness axis for this split, and any leave-one-day-out robustness check must be run separately.
3. **Unsoundness of `sld` Grouping:** Grouping on `sld` is explicitly prohibited because `sld` carries IP octets (`192`, `224`) and NetBIOS broadcast strings (`DESKTOP-3JF04TC`) present in both benign and attack traces [B3]. Raw `sld` and domain identity text are completely excluded from model inputs.
