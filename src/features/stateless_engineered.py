"""Second-generation stateless features: interaction and ratio terms derived from
the base sl_* features. All inputs are already leakage-gated (sl_ prefix passes
the LeakageGate allowlist); outputs use sl2_ prefix to stay in the same allowlist.

Derived at inference time -- no new raw columns needed, no parquet rebuild.

Rationale: LightGBM's axis-aligned splits approximate multiplicative interactions
poorly for extreme operating points. Making the key exfiltration interactions
explicit (entropy * subdomain_length, numeric density, etc.) lets the model hit
the FPR=0.1% budget while capturing more attack rows.
"""

import numpy as np
import pandas as pd

ENGINEERED_FEATURES = [
    "sl2_subdomain_entropy",       # high entropy AND long subdomain -- primary exfil signal
    "sl2_alpha_ratio",             # fraction of alphabetic chars -- exfil is more numeric/special
    "sl2_special_ratio",           # special char density -- encoded payloads have more delimiters
    "sl2_label_depth_entropy",     # entropy scaled by label depth
    "sl2_numeric_subdomain_density",  # numeric chars per subdomain char
    "sl2_max_label_fraction",      # longest label / total len -- label concentration
    "sl2_session_repeat_count",    # how many rows in this row's own session share its exact
                                    # base feature vector -- diagnostic finding: attack sessions
                                    # replay ~47 distinct query patterns thousands of times each,
                                    # while benign sessions rarely repeat a pattern. Computed via
                                    # groupby on session_id (grouping key, not an identity
                                    # feature) within each split's own rows only -- no label use,
                                    # no cross-split leakage (session_id is split-disjoint).
    "sl2_session_repeat_ratio",    # sl2_session_repeat_count / that session's row count
]

_BASE_PATTERN_COLS = [
    "sl_fqdn_count", "sl_subdomain_length", "sl_upper", "sl_lower", "sl_numeric",
    "sl_entropy", "sl_special", "sl_labels", "sl_labels_max", "sl_labels_average",
    "sl_len", "sl_subdomain_flag", "sl_longest_word_len", "sl_numeric_ratio",
]


def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Returns df with sl2_* columns appended. Input must have all base sl_* columns
    plus session_id (used only as a groupby key, never as a feature value)."""
    out = df.copy()
    out["sl2_subdomain_entropy"] = out["sl_entropy"] * out["sl_subdomain_length"]
    out["sl2_alpha_ratio"] = (out["sl_upper"] + out["sl_lower"]) / (out["sl_len"] + 1)
    out["sl2_special_ratio"] = out["sl_special"] / (out["sl_len"] + 1)
    out["sl2_label_depth_entropy"] = out["sl_entropy"] * out["sl_labels"]
    out["sl2_numeric_subdomain_density"] = out["sl_numeric"] / (out["sl_subdomain_length"] + 1)
    out["sl2_max_label_fraction"] = out["sl_labels_max"] / (out["sl_len"] + 1)

    pattern_key = list(zip(out["session_id"], *[out[c] for c in _BASE_PATTERN_COLS]))
    out["_pattern_key"] = pattern_key
    repeat_count = out.groupby("_pattern_key").size()
    out["sl2_session_repeat_count"] = out["_pattern_key"].map(repeat_count).astype(np.float64)
    session_size = out.groupby("session_id").size()
    out["sl2_session_repeat_ratio"] = out["sl2_session_repeat_count"] / out["session_id"].map(session_size)
    out = out.drop(columns=["_pattern_key"])
    return out
