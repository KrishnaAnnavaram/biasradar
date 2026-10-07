"""Source loaders. Each one returns the unified frame:

``text, source, group_id, is_biased, categories, polarity``

- ``is_biased``: 1, 0, or NaN (unknown: the row trains the category head only).
- ``categories``: ``;``-joined category names (may be empty).
- ``group_id``: rows with the same id must stay in one split (a CrowS pair, an SBIC post).
- ``polarity``: ``stereo``, ``antistereo``, ``offensive``, ``neutral``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from biasradar.taxonomy import SBIC_CATEGORY_MAP, normalise_category

UNIFIED_COLUMNS = ["text", "source", "group_id", "is_biased", "categories", "polarity"]


class SourceError(ValueError):
    """A source file does not have the expected columns or values."""


def _need(df: pd.DataFrame, cols, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SourceError(f"{name}: missing columns {missing}")


def _pairs(df: pd.DataFrame, text_col: str, source: str, id_col: str | None) -> pd.DataFrame:
    _need(df, [text_col, "bias_type", "stereo_antistereo"], source)
    ids = df[id_col] if id_col and id_col in df.columns else pd.Series(range(len(df)), index=df.index)
    polarity = df["stereo_antistereo"].astype(str).str.strip().str.lower()
    bad = sorted(set(polarity) - {"stereo", "antistereo"})
    if bad:
        raise SourceError(f"{source}: unknown stereo_antistereo values {bad}")
    out = pd.DataFrame({
        "text": df[text_col].astype(str).str.strip(),
        "source": source,
        "group_id": [f"{source}:{i}" for i in ids],
        # a stereotype sentence is biased. An anti-stereotype sentence trains only the category head
        "is_biased": np.where(polarity == "stereo", 1.0, np.nan),
        "categories": [normalise_category(c) for c in df["bias_type"]],
        "polarity": polarity,
    })
    return out[out["text"].str.len() > 0].reset_index(drop=True)


def load_crows_pairs(path: str | Path) -> pd.DataFrame:
    """CrowS-Pairs (Nangia et al., 2020): ``sent_more``, ``bias_type``, ``stereo_antistereo``."""
    df = pd.read_csv(path)
    return _pairs(df, "sent_more", "crows", df.columns[0] if df.columns[0].startswith("Unnamed") else None)


def load_indibias(path: str | Path) -> pd.DataFrame:
    """IndiBias: ``modified_eng_sent_more``, ``bias_type`` (``Religion`` and ``Caste`` capitalised)."""
    df = pd.read_csv(path)
    return _pairs(df, "modified_eng_sent_more", "indibias", "index" if "index" in df.columns else None)


def aggregate_sbic(raw: pd.DataFrame, min_votes: int = 1) -> pd.DataFrame:
    """One row for each SBIC post, from many annotation rows.

    Only the needed columns are kept BEFORE missing values are handled, so posts without a target
    group stay in the data. A post is biased when the mean ``offensiveYN`` is 0.5 or more. A category
    is kept when ``min_votes`` or more annotators named it.
    """
    _need(raw, ["post", "offensiveYN", "targetCategory"], "sbic")
    df = raw[["post", "offensiveYN", "targetCategory"]].copy()
    df = df[df["post"].notna()]
    df["post"] = df["post"].astype(str).str.strip()
    df["offensiveYN"] = pd.to_numeric(df["offensiveYN"], errors="coerce")
    rows = []
    for i, (post, part) in enumerate(df.groupby("post", sort=True)):
        offensive = part["offensiveYN"].mean()
        votes: dict[str, int] = {}
        for raw_cat in part["targetCategory"].dropna().astype(str):
            mapped = SBIC_CATEGORY_MAP.get(raw_cat.strip().lower(), "unknown")
            if mapped not in (None, "unknown"):
                votes[mapped] = votes.get(mapped, 0) + 1
        biased = float(offensive >= 0.5) if pd.notna(offensive) else np.nan
        cats = sorted(c for c, n in votes.items() if n >= min_votes) if biased == 1.0 else []
        rows.append({"text": post, "source": "sbic", "group_id": f"sbic:{i}", "is_biased": biased,
                     "categories": ";".join(cats), "polarity": "offensive" if biased == 1.0 else "neutral"})
    return pd.DataFrame(rows, columns=UNIFIED_COLUMNS)


def load_sbic(path: str | Path, min_votes: int = 1) -> pd.DataFrame:
    return aggregate_sbic(pd.read_csv(path), min_votes)


def load_neutral(path: str | Path, text_col: str = "text") -> pd.DataFrame:
    """Any CSV with a text column of sentences that contain no social bias."""
    df = pd.read_csv(path)
    _need(df, [text_col], "neutral")
    return pd.DataFrame({"text": df[text_col].astype(str), "source": "neutral",
                         "group_id": [f"neutral:{i}" for i in range(len(df))], "is_biased": 0.0,
                         "categories": "", "polarity": "neutral"})


def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate sources, drop empty texts, and drop exact duplicate texts inside one source."""
    df = pd.concat(frames, ignore_index=True)[UNIFIED_COLUMNS]
    df["categories"] = df["categories"].fillna("").astype(str)
    df = df[df["text"].str.strip().str.len() > 0]
    return df.drop_duplicates(subset=["source", "text"]).reset_index(drop=True)


def category_matrix(df: pd.DataFrame, categories) -> np.ndarray:
    sets = [set(filter(None, s.split(";"))) for s in df["categories"]]
    return np.asarray([[int(c in s) for c in categories] for s in sets], dtype=np.int64)
