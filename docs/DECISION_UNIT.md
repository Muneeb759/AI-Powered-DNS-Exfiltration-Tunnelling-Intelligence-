# DECISION UNIT SPECIFICATION — TRACK 1 ROW-LEVEL ARCHITECTURE

## 1. Primary Decision Unit
The primary decision unit for this system is **row-level (individual DNS query)**.
Each query packet is evaluated independently by the stateless detection stage.

- Canonical Primary Key: `unit_id` (string format: `{capture_file_name}_{row_index}`)
- Grouping Key for Splits: `session_id` (`capture_file_name`, 18 distinct capture groups) and `collection_day` (`YYYY-MM-DD`).

---

## 2. Technical Justification & Cancellation of Track 2 Cascade

> **⚠️ Later revised — read with `report/TECHNICAL_REPORT.md` §1 and §10.** The three findings
> below are all correct and still stand. The *conclusion* drawn from them — cancelling Track 2 —
> was reversed. Track 2 was reinstated, not as the compute-optimisation the brief describes (our
> own latency measurements do not support that framing), but as an **interpretable session-level
> confirmation layer** operating at capture-file granularity, which is exactly the coarsest join
> the findings below permit. The row-level decision unit argued for in this document remains the
> headline unit and was never displaced.
>
> Finding 2's resolution-floor argument is also why §10's session-level cascade numbers carry an
> inline caveat: with only 3 test sessions, session-level FPR remains statistically weak, precisely
> as predicted here.

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
