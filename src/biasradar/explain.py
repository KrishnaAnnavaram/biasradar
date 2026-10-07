"""Explanations: the n-grams that push a TF-IDF prediction up, and a prediction formatter that the CLI
and the app share."""

from __future__ import annotations

import numpy as np

DISCLAIMER = ("biasradar estimates if a text contains a social-bias generalisation and which groups it touches. "
              "It is not a moderation decision. A person must review each result.")


def predict_text(model, text: str) -> dict:
    """Independent probabilities for the biased flag and each category, with the saved thresholds."""
    pb, pc = model.predict_proba([text])
    cats = model.label_map.categories
    thr = np.asarray(model.category_thresholds)
    scores = sorted(({"category": c, "p": float(pc[0, k]), "flag": bool(pc[0, k] >= thr[k])}
                     for k, c in enumerate(cats)), key=lambda r: -r["p"])
    return {"text": text, "p_biased": float(pb[0]), "biased": bool(pb[0] >= model.biased_threshold),
            "threshold": float(model.biased_threshold), "categories": scores}


def top_terms(model, text: str, k: int = 8) -> list[tuple[str, float]]:
    """TF-IDF model only: the k features with the largest positive contribution to the biased logit."""
    if getattr(model, "kind", "") != "tfidf":
        raise TypeError("top_terms works with the tfidf model only")
    if not model.biased_head_trained:
        return []
    x = model.vectorizer_.transform([text]).tocsr()
    names = model.vectorizer_.get_feature_names_out()
    coef = model.biased_clf_.coef_[0]
    contrib = [(names[j], float(x[0, j] * coef[j])) for j in x.indices]
    return sorted((c for c in contrib if c[1] > 0), key=lambda c: -c[1])[:k]


def format_prediction(pred: dict, top: int = 5) -> str:
    lines = [f"p(biased) = {pred['p_biased']:.3f} (threshold {pred['threshold']:.2f}) -> "
             f"{'BIASED' if pred['biased'] else 'not biased'}"]
    for row in pred["categories"][:top]:
        lines.append(f"  {row['category']:20s} {row['p']:.3f}{'  *' if row['flag'] else ''}")
    lines.append("Category scores are independent probabilities. They do not sum to 1.")
    return "\n".join(lines)
