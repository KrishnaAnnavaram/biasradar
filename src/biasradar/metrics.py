"""Evaluation for the two heads: per-category P/R/F1 and PR-AUC, the biased head with ROC-AUC,
PR-AUC, calibration and the false-alarm rate on neutral text, all also split by source."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def category_report(Y: np.ndarray, P: np.ndarray, thresholds: np.ndarray, categories) -> pd.DataFrame:
    """One row for each category: support, precision, recall, F1 (at its threshold) and PR-AUC."""
    pred = (P >= thresholds).astype(int)
    prec, rec, f1, sup = precision_recall_fscore_support(Y, pred, average=None, zero_division=0,
                                                         labels=list(range(len(categories))))
    rows = []
    for k, c in enumerate(categories):
        ap = average_precision_score(Y[:, k], P[:, k]) if Y[:, k].any() else float("nan")
        rows.append({"category": c, "support": int(sup[k]), "threshold": float(thresholds[k]),
                     "precision": float(prec[k]), "recall": float(rec[k]), "f1": float(f1[k]), "pr_auc": float(ap)})
    return pd.DataFrame(rows)


def biased_report(y: np.ndarray, p: np.ndarray, threshold: float, polarity=None) -> dict:
    """Metrics of the biased head on rows with a known ``is_biased`` value."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    known = ~np.isnan(y)
    y, p = y[known].astype(int), p[known]
    out = {"n": int(len(y)), "threshold": float(threshold)}
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    pred = (p >= threshold).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["f1"] = float(f1_score(y, pred, zero_division=0))
    out["ece"] = expected_calibration_error(y, p)
    neutral = y == 0
    out["neutral_false_alarm_rate"] = float(pred[neutral].mean()) if neutral.any() else float("nan")
    out["biased_recall"] = float(pred[y == 1].mean()) if (y == 1).any() else float("nan")
    return out


def _best_threshold(scores, grid) -> float:
    """The threshold with the best score. Among equal scores, the one nearest to 0.5."""
    scores = np.asarray(scores, dtype=float)
    ties = np.asarray(grid)[scores >= scores.max() - 1e-9]
    return float(ties[np.argmin(np.abs(ties - 0.5))])


def tune_thresholds(Y: np.ndarray, P: np.ndarray, grid=np.linspace(0.05, 0.95, 19)) -> np.ndarray:
    """For each category, the threshold with the best F1 on validation (0.5 when the category is absent)."""
    out = np.full(Y.shape[1], 0.5)
    for k in range(Y.shape[1]):
        if not Y[:, k].any():
            continue
        scores = [f1_score(Y[:, k], (P[:, k] >= t).astype(int), zero_division=0) for t in grid]
        out[k] = _best_threshold(scores, grid)
    return out


def tune_biased_threshold(y, p, grid=np.linspace(0.05, 0.95, 19)) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    known = ~np.isnan(y)
    if len(np.unique(y[known])) < 2:
        return 0.5
    scores = [f1_score(y[known].astype(int), (p[known] >= t).astype(int), zero_division=0) for t in grid]
    return _best_threshold(scores, grid)


def micro_macro(Y: np.ndarray, P: np.ndarray, thresholds: np.ndarray) -> dict:
    pred = (P >= thresholds).astype(int)
    present = Y.any(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"micro_f1": float(f1_score(Y, pred, average="micro", zero_division=0)),
                "macro_f1": float(f1_score(Y[:, present], pred[:, present], average="macro", zero_division=0))}
