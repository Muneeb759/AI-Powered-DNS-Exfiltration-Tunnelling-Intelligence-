import numpy as np
import pandas as pd
from typing import List, Dict, Any, Callable, Tuple
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, confusion_matrix

def pick_threshold_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.001) -> float:
    """Picks the smallest threshold t such that count(benign_scores >= t) <= max_fp, i.e. the
    achieved FPR never exceeds target_fpr. Ties matter here: if many benign scores share the
    exact value that would sit at the max_fp-th rank (e.g. a coarse/plateaued calibrator), naively
    using that tied value as the threshold admits the WHOLE tied block, not just the intended
    max_fp of them, and can blow the FPR far past target. This walks up to the next strictly
    higher distinct score instead, trading a bit of recall for actually respecting the FPR budget."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    neg_scores = scores[y_true == 0]
    n_neg = len(neg_scores)
    if n_neg == 0:
        raise ValueError("No negative/benign samples present to compute FPR threshold")

    max_fp = int(np.floor(target_fpr * n_neg))
    distinct_desc = np.unique(neg_scores)[::-1]

    if max_fp <= 0:
        # Budget doesn't even allow the single highest benign score through.
        return float(np.nextafter(distinct_desc[0], np.inf))

    # count_ge(candidate) is monotonically non-decreasing as candidate decreases, so walk down
    # from the highest score and keep the LOWEST candidate seen so far that still respects the
    # budget -- that's the threshold maximizing recall without exceeding target_fpr. Stop at the
    # first violation since everything lower would violate it too.
    best = float(np.nextafter(distinct_desc[0], np.inf))
    for candidate in distinct_desc:
        count_ge = int(np.sum(neg_scores >= candidate))
        if count_ge <= max_fp:
            best = float(candidate)
        else:
            break
    return best

def point_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    preds = (scores >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    false_alerts_per_10k_benign = fpr * 10000.0
    
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "false_alerts_per_10k_benign": float(false_alerts_per_10k_benign)
    }

def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)
    
    try:
        roc_auc = roc_auc_score(y_true, scores)
    except Exception:
        roc_auc = 0.0
        
    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc)
    }

def recall_at_fpr_curve(y_true: np.ndarray, scores: np.ndarray, target_fprs: List[float] = [0.01, 0.005, 0.002, 0.001]) -> pd.DataFrame:
    rows = []
    for fpr_target in target_fprs:
        thr = pick_threshold_at_fpr(y_true, scores, target_fpr=fpr_target)
        pm = point_metrics(y_true, scores, threshold=thr)
        rows.append({
            "target_fpr": fpr_target,
            "target_fpr_pct": f"{fpr_target * 100:.2f}%",
            "threshold": thr,
            "actual_fpr": pm["fpr"],
            "recall": pm["recall"],
            "precision": pm["precision"],
            "false_alerts_per_10k_benign": pm["false_alerts_per_10k_benign"]
        })
    return pd.DataFrame(rows)

def bootstrap_ci(
    df: pd.DataFrame,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    group_col: str = "session_id",
    n_bootstraps: int = 200,
    seed: int = 20260808
) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    groups = sorted(list(df[group_col].unique()))
    group_dict = {g: df[df[group_col] == g] for g in groups}
    
    bootstrap_vals = []
    for _ in range(n_bootstraps):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sample_df = pd.concat([group_dict[g] for g in sampled_groups], ignore_index=True)
        try:
            val = metric_fn(sample_df["label"].values, sample_df["score"].values)
            bootstrap_vals.append(val)
        except Exception:
            continue
            
    if len(bootstrap_vals) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        
    return {
        "mean": float(np.mean(bootstrap_vals)),
        "ci_lower": float(np.percentile(bootstrap_vals, 2.5)),
        "ci_upper": float(np.percentile(bootstrap_vals, 97.5))
    }

def full_report(df: pd.DataFrame, score_col: str = "score", operating_fpr: float = 0.001) -> Dict[str, Any]:
    y_true = df["label"].values
    scores = df[score_col].values
    
    rm = ranking_metrics(y_true, scores)
    thr = pick_threshold_at_fpr(y_true, scores, target_fpr=operating_fpr)
    pm = point_metrics(y_true, scores, threshold=thr)
    curve_df = recall_at_fpr_curve(y_true, scores)
    
    return {
        "headline_pr_auc": rm["pr_auc"],
        "supplementary_roc_auc": rm["roc_auc"],
        "operating_point_fpr_target": operating_fpr,
        "operating_point_metrics": pm,
        "recall_vs_fpr_curve": curve_df.to_dict(orient="records")
    }
