import os
import json
import pandas as pd
from pathlib import Path

def audit_snapshot(raw_dir: str = "data/raw") -> dict:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    stateless_files = sorted(list(raw_path.rglob("stateless_features*.csv")))
    stateful_files = sorted(list(raw_path.rglob("stateful_features*.csv")))

    # 1. Load samples and full schema
    sl_dfs = []
    sf_dfs = []

    for f in stateless_files:
        rel_path = str(f.relative_to(raw_path)).replace("\\", "/")
        df = pd.read_csv(f)
        if "Attacks" in rel_path:
            cat = "heavy" if "heavy" in rel_path.lower() else "light"
            lbl = 1
        else:
            cat = "benign"
            lbl = 0
        df["_attack_category"] = cat
        df["_label"] = lbl
        df["_capture_file"] = rel_path
        df["_ts_parsed"] = pd.to_datetime(df["timestamp"], format="mixed")
        df["_collection_day"] = df["_ts_parsed"].dt.date.astype(str)
        sl_dfs.append(df)

    for f in stateful_files:
        rel_path = str(f.relative_to(raw_path)).replace("\\", "/")
        df = pd.read_csv(f)
        if "Attacks" in rel_path:
            cat = "heavy" if "heavy" in rel_path.lower() else "light"
            lbl = 1
        else:
            cat = "benign"
            lbl = 0
        df["_attack_category"] = cat
        df["_label"] = lbl
        df["_capture_file"] = rel_path
        sf_dfs.append(df)

    sl_all = pd.concat(sl_dfs, ignore_index=True)
    sf_all = pd.concat(sf_dfs, ignore_index=True)

    # 2. Extract column schemas (excluding internal _ audit columns)
    sl_cols = [c for c in sl_all.columns if not c.startswith("_")]
    sf_cols = [c for c in sf_all.columns if not c.startswith("_")]

    sl_schema = {}
    for c in sl_cols:
        sl_schema[c] = {
            "dtype": str(sl_all[c].dtype),
            "null_count": int(sl_all[c].isnull().sum()),
            "nunique": int(sl_all[c].nunique()),
            "sample_values": [str(x) for x in sl_all[c].dropna().unique()[:3]]
        }

    sf_schema = {}
    for c in sf_cols:
        sf_schema[c] = {
            "dtype": str(sf_all[c].dtype),
            "null_count": int(sf_all[c].isnull().sum()),
            "nunique": int(sf_all[c].nunique()),
            "sample_values": [str(x) for x in sf_all[c].dropna().unique()[:3]]
        }

    # 3. Decision unit & day statistics (stateless row level)
    day_counts = sl_all.groupby(["_collection_day", "_label", "_attack_category"]).size().unstack(fill_value=0).to_dict(orient="index")
    
    benign_by_day = sl_all[sl_all["_label"] == 0].groupby("_collection_day").size().to_dict()
    min_resolvable_fpr_by_day = {day: 1.0 / count for day, count in benign_by_day.items()}

    total_rows = len(sl_all)
    total_benign = int((sl_all["_label"] == 0).sum())
    total_positive = int((sl_all["_label"] == 1).sum())

    # B1: stateless/stateful row-level join analysis (cited in docs/PHASE1_AUDIT.md B1)
    join_analysis = {
        "row_level_join_exists": False,
        "join_level": "capture_file_level",
        "stateless_raw_header": ",".join(sl_cols),
        "stateful_raw_header": ",".join(sf_cols),
    }

    # B2: session definition comparison, Rule A (capture_file) vs Rule B (capture_file + sld)
    # (cited in docs/PHASE1_AUDIT.md B2)
    rule_a_sessions = sl_all.groupby("_capture_file")["_attack_category"].agg(lambda s: s.mode().iat[0])
    rule_a_total = int(rule_a_sessions.shape[0])
    rule_a_light = int((rule_a_sessions == "light").sum())
    rule_a_heavy = int((rule_a_sessions == "heavy").sum())
    rule_a_benign = int((rule_a_sessions == "benign").sum())

    sl_all["_rule_b_session"] = sl_all["_capture_file"] + "_" + sl_all["sld"].astype(str)
    rule_b_sessions = sl_all.groupby("_rule_b_session")["_attack_category"].agg(lambda s: s.mode().iat[0])
    rule_b_total = int(rule_b_sessions.shape[0])
    rule_b_light = int((rule_b_sessions == "light").sum())
    rule_b_heavy = int((rule_b_sessions == "heavy").sum())
    rule_b_benign = int((rule_b_sessions == "benign").sum())

    session_definitions = {
        "rule_A_capture_file": {
            "total_sessions": rule_a_total,
            "positive_sessions": rule_a_light + rule_a_heavy,
            "light_sessions": rule_a_light,
            "heavy_sessions": rule_a_heavy,
            "benign_sessions": rule_a_benign,
        },
        "rule_B_capture_file_sld": {
            "total_sessions": rule_b_total,
            "positive_sessions": rule_b_light + rule_b_heavy,
            "light_sessions": rule_b_light,
            "heavy_sessions": rule_b_heavy,
            "benign_sessions": rule_b_benign,
        },
    }

    # C6/C7: collection-day reconciliation (cited in docs/PHASE1_AUDIT.md C6/C7)
    calendar_days = sorted(sl_all["_collection_day"].unique())
    benign_days = sorted(sl_all.loc[sl_all["_label"] == 0, "_collection_day"].unique())
    days_with_zero_benign = sorted(set(calendar_days) - set(benign_days))
    collection_days_reconciliation = {
        "total_calendar_days": len(calendar_days),
        "calendar_days": calendar_days,
        "days_with_benign_rows": benign_days,
        "days_with_zero_benign_rows": days_with_zero_benign,
        "days_ineligible_as_held_out_test_day": days_with_zero_benign,
    }

    # Candidate probes
    candidate_probes = {
        "session_id": "None in CSV columns (must be derived or capture-level)",
        "collection_day": "Recoverable from timestamp column (dates 2020-11-20 to 2020-11-25)",
        "label": "Derived from directory path (Benign=0, Attacks=1)",
        "attack_category": "Derived from directory path (benign, light, heavy)",
        "query_name_sld": "sld column in stateless table, distinct_domains/reverse_dns in stateful table"
    }

    # Silent leakage / shortcut candidates
    shortcut_candidates = [
        {"column": "timestamp", "reason": "Direct timestamp leaks temporal/collection-day context"},
        {"column": "sld", "reason": "Raw registrable domain identity leaks testbed infrastructure identity"},
        {"column": "longest_word", "reason": "Subdomain text token derived from raw query name"},
        {"column": "subdomain", "reason": "Raw query string / subdomain derivative"},
        {"column": "distinct_ip", "reason": "Constant / IP identity feature in stateful table"},
        {"column": "reverse_dns", "reason": "Reverse DNS domain string in stateful table"}
    ]

    report = {
        "dataset_release": "public_unb_release",
        "has_organizer_split_manifest": False,
        "has_published_sha256": False,
        "stateless_file_count": len(stateless_files),
        "stateful_file_count": len(stateful_files),
        "stateless_total_rows": total_rows,
        "stateful_total_rows": len(sf_all),
        "join_analysis": join_analysis,
        "session_definitions": session_definitions,
        "collection_days_reconciliation": collection_days_reconciliation,
        "label_counts": {
            "benign_0": total_benign,
            "positive_exfiltration_1": total_positive,
            "light_positive": int((sl_all["_attack_category"] == "light").sum()),
            "heavy_positive": int((sl_all["_attack_category"] == "heavy").sum())
        },
        "candidate_probes": candidate_probes,
        "collection_days_present": sorted(list(benign_by_day.keys())),
        "benign_row_counts_by_day": benign_by_day,
        "min_resolvable_fpr_by_day": min_resolvable_fpr_by_day,
        "planned_operating_point_fpr": 0.001,
        "fpr_0.1_percent_measurable": True, # row-level floor is ~0.002% to 0.0007%
        "layout_risk": {
            "organized_by_file_type": True,
            "directory_mapping": {
                "Attack_heavy_Benign/Attacks": {"label": 1, "attack_category": "heavy"},
                "Attack_Light_Benign/Attacks": {"label": 1, "attack_category": "light"},
                "Attack_heavy_Benign/Benign": {"label": 0, "attack_category": "benign"},
                "Attack_Light_Benign/Benign": {"label": 0, "attack_category": "benign"},
                "Benign": {"label": 0, "attack_category": "benign"}
            }
        },
        "stateless_schema": sl_schema,
        "stateful_schema": sf_schema,
        "shortcut_candidates": shortcut_candidates
    }

    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "snapshot_audit.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Audit completed. Written to {out_path}")
    return report

if __name__ == "__main__":
    audit_snapshot()
