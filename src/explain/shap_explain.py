"""SHAP explanations for the Stage 1 stateless LightGBM model (Track 3 support:
"explanations that identify influential observable features without inventing facts").

TreeExplainer on a LightGBM binary classifier -- exact, not an approximation.
Every explanation is grounded directly in the row's own sl_*/sl2_* feature values;
nothing here reads or displays raw query text (none exists in this schema, see
configs/leakage_exclusions.yaml), so there is no path to inventing facts about a
specific domain -- only the numeric features the row-level model actually used.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import shap

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_bundle(model_path: str = "models/stateless_lgbm.pkl"):
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["features"]


def build_explainer(model):
    return shap.TreeExplainer(model)


def global_summary(explainer, df: pd.DataFrame, features: list, sample_n: int = 5000, seed: int = 20260808) -> dict:
    """Mean |SHAP value| per feature over a sample of rows -- ranks which features
    drive Stage 1 decisions overall, not just for one alert."""
    sample = df.sample(n=min(sample_n, len(df)), random_state=seed) if len(df) > sample_n else df
    shap_values = explainer.shap_values(sample[features])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # binary classifier: class-1 contributions
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranked = sorted(
        [{"feature": f, "mean_abs_shap": float(v)} for f, v in zip(features, mean_abs)],
        key=lambda d: -d["mean_abs_shap"],
    )
    return {"sample_size": int(len(sample)), "ranked_features": ranked}


def local_explanation(explainer, row: pd.Series, features: list, top_k: int = 5) -> list:
    """Top-K features driving this single row's score, signed by direction of push
    (positive = pushed toward attack, negative = pushed toward benign)."""
    row_df = pd.DataFrame([row[features].values], columns=features)
    shap_values = explainer.shap_values(row_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    contributions = shap_values[0]
    ranked_idx = np.argsort(-np.abs(contributions))[:top_k]
    return [
        {
            "feature": features[i],
            "value": float(row_df.iloc[0, i]),
            "shap_value": float(contributions[i]),
            "direction": "toward_attack" if contributions[i] > 0 else "toward_benign",
        }
        for i in ranked_idx
    ]


def build_and_save_global_summary(test_df: pd.DataFrame) -> dict:
    model, features = load_bundle()
    explainer = build_explainer(model)
    summary = global_summary(explainer, test_df, features)

    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "shap_global_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    from src.data.load import load_split
    from src.features.stateless_engineered import build_engineered_features

    test = build_engineered_features(load_split("test"))
    summary = build_and_save_global_summary(test)
    print(json.dumps(summary, indent=2))
