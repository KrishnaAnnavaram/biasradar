"""Evaluate a detector on one split: overall, by source, and with the worst errors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from biasradar.metrics import biased_report, category_report, micro_macro
from biasradar.model import category_rows
from biasradar.sources import category_matrix


def evaluate(model, df: pd.DataFrame, by_source: bool = True, n_errors: int = 10) -> dict:
    pb, pc = model.predict_proba(df["text"])
    cats = model.label_map.categories
    thr = np.asarray(model.category_thresholds)
    cm = category_rows(df).to_numpy()
    Y = category_matrix(df[cm], cats)
    out = {
        "n": int(len(df)),
        "biased": {**biased_report(df["is_biased"].to_numpy(float), pb, model.biased_threshold),
                   "trained": bool(getattr(model, "biased_head_trained", True))},
        "categories": category_report(Y, pc[cm], thr, cats).to_dict(orient="records"),
        **micro_macro(Y, pc[cm], thr),
    }
    if by_source:
        out["by_source"] = {}
        for src, part in df.groupby("source"):
            idx = df.index.get_indexer(part.index)
            pm = cm[idx]
            entry = {"n": int(len(part)),
                     "biased": biased_report(part["is_biased"].to_numpy(float), pb[idx], model.biased_threshold)}
            if pm.any():
                entry.update(micro_macro(category_matrix(part[pm], cats), pc[idx][pm], thr))
            out["by_source"][src] = entry
    known = df["is_biased"].notna().to_numpy()
    err = np.abs(pb - np.nan_to_num(df["is_biased"].to_numpy(float)))
    err[~known] = -1
    worst = np.argsort(-err)[:n_errors]
    out["worst_biased_errors"] = [{"text": df["text"].iloc[i], "is_biased": float(df["is_biased"].iloc[i]),
                                   "p_biased": float(pb[i])} for i in worst if err[i] >= 0]
    return out


def format_summary(rep: dict) -> str:
    b = rep["biased"]
    if not b.get("trained", True):
        head = f"rows {rep['n']}: biased head NOT trained (the training data has one class, add neutral rows)"
    else:
        head = None
    lines = [head or f"rows {rep['n']}: biased head ROC-AUC {b.get('roc_auc', float('nan')):.3f}, PR-AUC {b.get('pr_auc', float('nan')):.3f}, "
             f"F1 {b['f1']:.3f}, neutral false alarms {b['neutral_false_alarm_rate']:.3f}, ECE {b['ece']:.3f}",
             f"categories: micro F1 {rep['micro_f1']:.3f}, macro F1 {rep['macro_f1']:.3f}"]
    table = pd.DataFrame(rep["categories"])
    lines.append(table.round(3).to_string(index=False))
    for src, e in rep.get("by_source", {}).items():
        lines.append(f"source {src}: rows {e['n']}, biased F1 {e['biased']['f1']:.3f}, "
                     f"macro F1 {e.get('macro_f1', float('nan')):.3f}")
    return "\n".join(lines)
