# DECISION UNIT SPECIFICATION — TRACK 1 ROW-LEVEL ARCHITECTURE

## 1. Primary Decision Unit
The primary decision unit for this system is **row-level (individual DNS query)**.
Each query packet is evaluated independently by the stateless detection stage.

- Canonical Primary Key: `unit_id` (string format: `{capture_file_name}_{row_index}`)
- Grouping Key for Splits: `session_id` (`capture_file_name`, 18 distinct capture groups) and `collection_day` (`YYYY-MM-DD`).

---

## 2. Technical Justification & Cancellation of Track 2 Cascade

Track 2 (Stateful Cascade) was cancelled following the empirical findings of Stage A:

1. **Absence of Row-Level Join [B1(d)]:**
   The raw dataset contains 757,211 stateless rows (extracted per DNS packet) and 262,105 stateful rows (aggregated per time window/flow burst). Verbatim inspection of raw headers confirms zero shared record index or packet ID. A row-level join between stateless and stateful tables is impossible.
2. **Session-Level Resolution Floor Barrier [M2]:**
   The only viable join boundary between stateless and stateful tables is the capture trace file (18 total files across the entire dataset, with only 1 to 2 benign capture files per day). At the capture file session level, the minimum resolvable false positive rate (FPR) is $1 / 1 = 100\%$ or $1 / 2 = 50\%$. An operating point of $\text{FPR} = 0.1\%$ (0.001) is mathematically unmeasurable at the session level.
3. **Unsoundness of `sld` Grouping [B3]:**
   Inspection of `sld` values revealed that ~59% of positive rows and ~38% of benign rows contain raw IP octet fragments (`192`, `224`, `239`) and NetBIOS broadcast strings (`DESKTOP-3JF04TC`). Grouping by `capture_file + sld` fails to represent true file-transfer sessions.

---

## 3. Operating Point & Resolution

- **Decision Unit:** Row-level (757,211 total query decisions).
- **Target Operating Point:** **FPR = 0.1%** (10 false alerts per 10,000 benign decisions).
- **Resolution Floor:** At the row level, daily benign query counts ($49,115$ to $142,138$) yield resolution floors between $0.0007\%$ and $0.0020\%$, confirming that $\text{FPR} = 0.1\%$ is fully measurable.
