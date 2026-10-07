"""The baseline detector: TF-IDF features with two heads of independent sigmoid outputs.

- Biased head: one logistic regression, trained on rows with a known ``is_biased``.
- Category head: one logistic regression for each category (one-vs-rest), trained on rows with
  known categories (labelled rows plus neutral rows).

Each output is an independent probability. Thus a neutral text can get a low score in every
category: the scores do not have to sum to 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from biasradar.metrics import tune_biased_threshold, tune_thresholds
from biasradar.sources import category_matrix
from biasradar.taxonomy import CATEGORIES, LabelMap


def biased_rows(df: pd.DataFrame) -> pd.Series:
    return df["is_biased"].notna()


def category_rows(df: pd.DataFrame) -> pd.Series:
    """Rows whose categories are known: rows with a category, and rows known to be not biased."""
    return (df["categories"].fillna("") != "") | (df["is_biased"] == 0)


class TfidfBiasModel:
    kind = "tfidf"

    def __init__(self, seed: int = 42, C: float = 4.0):
        self.seed, self.C = seed, C
        self.label_map = LabelMap()
        self.biased_threshold = 0.5
        self.category_thresholds = np.full(len(CATEGORIES), 0.5)

    def _features(self):
        return FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, lowercase=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
        ])

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "TfidfBiasModel":
        self.vectorizer_ = self._features().fit(train["text"])
        bm = biased_rows(train)
        yb = train.loc[bm, "is_biased"].astype(int)
        if yb.nunique() < 2:
            # e.g. stereotype benchmarks only: the biased head cannot learn. Record a constant and say so
            self.biased_clf_ = float(yb.iloc[0]) if len(yb) else 0.5
        else:
            Xb = self.vectorizer_.transform(train.loc[bm, "text"])
            self.biased_clf_ = LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000,
                                                  random_state=self.seed).fit(Xb, yb)
        cm = category_rows(train)
        Xc = self.vectorizer_.transform(train.loc[cm, "text"])
        Y = category_matrix(train[cm], self.label_map.categories)
        self.category_clfs_ = []
        for k in range(Y.shape[1]):
            if Y[:, k].min() == Y[:, k].max():
                self.category_clfs_.append(float(Y[0, k]))  # constant: no positive (or no negative) example
            else:
                self.category_clfs_.append(LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000,
                                                              random_state=self.seed).fit(Xc, Y[:, k]))
        if val is not None and len(val):
            pb, pc = self.predict_proba(val["text"])
            self.biased_threshold = tune_biased_threshold(val["is_biased"].to_numpy(float), pb)
            vm = category_rows(val).to_numpy()
            self.category_thresholds = tune_thresholds(category_matrix(val[vm], self.label_map.categories), pc[vm])
        return self

    def predict_proba(self, texts) -> tuple[np.ndarray, np.ndarray]:
        X = self.vectorizer_.transform(list(texts))
        if isinstance(self.biased_clf_, float):
            pb = np.full(X.shape[0], self.biased_clf_)
        else:
            pb = self.biased_clf_.predict_proba(X)[:, 1]
        cols = [np.full(X.shape[0], c) if isinstance(c, float) else c.predict_proba(X)[:, 1] for c in self.category_clfs_]
        return pb, np.column_stack(cols)

    def save(self, out_dir: str | Path, extra: dict | None = None) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump({"vectorizer": self.vectorizer_, "biased": self.biased_clf_, "categories": self.category_clfs_},
                    out / "model.joblib")
        self.label_map.save(out / "label_map.json")
        (out / "thresholds.json").write_text(json.dumps({
            "kind": self.kind, "biased": self.biased_threshold,
            "categories": dict(zip(self.label_map.categories, map(float, self.category_thresholds))),
            **(extra or {})}, indent=1), encoding="utf-8")
        return out

    @property
    def biased_head_trained(self) -> bool:
        return not isinstance(self.biased_clf_, float)

    @classmethod
    def load(cls, model_dir: str | Path) -> "TfidfBiasModel":
        d = Path(model_dir)
        obj = cls()
        parts = joblib.load(d / "model.joblib")
        obj.vectorizer_, obj.biased_clf_, obj.category_clfs_ = parts["vectorizer"], parts["biased"], parts["categories"]
        obj.label_map = LabelMap.load(d / "label_map.json")
        th = json.loads((d / "thresholds.json").read_text(encoding="utf-8"))
        obj.biased_threshold = th["biased"]
        obj.category_thresholds = np.asarray([th["categories"][c] for c in obj.label_map.categories])
        return obj


def load_model(model_dir: str | Path):
    """Load a saved detector of any kind, with its own label map."""
    d = Path(model_dir)
    if not (d / "thresholds.json").is_file():
        raise FileNotFoundError(f"{d} has no saved model. Run `biasradar train` first.")
    kind = json.loads((d / "thresholds.json").read_text(encoding="utf-8")).get("kind", "tfidf")
    if kind == "tfidf":
        return TfidfBiasModel.load(d)
    from biasradar.transformer import TransformerBiasModel

    return TransformerBiasModel.load(d)
