# SESSION GROUPING SPECIFICATION

## Overview
Under the Track 1 stateless architecture, individual decision units are **row-level DNS queries**.
However, to prevent data leakage and guarantee strict independence between training, calibration, thresholding, and test partitions (Control 1 & Control 2), all records are grouped by **`capture_file` (18 capture trace sessions)** and **`collection_day`**.

## Grouping Rules
1. **Partition Group Key:** `session_id` = `capture_file` (relative path of original `.pcap.csv` trace).
2. **Disjoint Invariant:** No `capture_file` or `collection_day` appears in more than one partition (`train`, `val_cal`, `val_thr`, `test`).
3. **Unsoundness of `sld` Grouping:** Grouping on `sld` is explicitly prohibited because `sld` carries IP octets (`192`, `224`) and NetBIOS broadcast strings (`DESKTOP-3JF04TC`) present in both benign and attack traces [B3]. Raw `sld` and domain identity text are completely excluded from model inputs.
