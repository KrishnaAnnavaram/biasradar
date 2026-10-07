"""Train / validation / test splits, stratified by source and main label, grouped by ``group_id``."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


class SplitLeakError(AssertionError):
    """A group is in more than one split."""


def strata(df: pd.DataFrame) -> pd.Series:
    first_cat = df["categories"].fillna("").str.split(";").str[0].replace("", "none")
    biased = df["is_biased"].map({1.0: "b", 0.0: "n"}).fillna("u")
    return df["source"].astype(str) + "|" + biased + "|" + first_cat


def make_splits(df: pd.DataFrame, test_fraction: float = 0.15, val_fraction: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Return ``df`` with a ``split`` column (train / val / test). Rows of one group stay together."""
    if not (0.05 <= test_fraction <= 0.4 and 0.05 <= val_fraction <= 0.4):
        raise ValueError("fractions must be between 0.05 and 0.4")
    out = df.reset_index(drop=True).copy()
    y = strata(out)
    # rare strata would break stratification: merge them into their source stratum
    counts = y.value_counts()
    y = y.where(y.map(counts) >= 10, out["source"].astype(str) + "|rare")
    groups = out["group_id"].astype(str)

    def take(index: np.ndarray, fraction: float, rs: int) -> np.ndarray:
        k = max(2, int(round(1 / fraction)))
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=rs)
        _, held = next(sgkf.split(index, y.iloc[index], groups.iloc[index]))
        return index[held]

    all_idx = np.arange(len(out))
    test_idx = take(all_idx, test_fraction, seed)
    rest = np.setdiff1d(all_idx, test_idx)
    val_idx = take(rest, val_fraction / (1 - test_fraction), seed + 1)
    out["split"] = "train"
    out.loc[test_idx, "split"] = "test"
    out.loc[val_idx, "split"] = "val"
    assert_no_group_leak(out)
    return out


def assert_no_group_leak(df: pd.DataFrame) -> None:
    per_group = df.groupby("group_id")["split"].nunique()
    leaked = per_group[per_group > 1]
    if len(leaked):
        raise SplitLeakError(f"{len(leaked)} groups are in more than one split, first {leaked.index[0]}")


def coverage(df: pd.DataFrame, categories) -> pd.DataFrame:
    """Number of rows of each category in each split (a 0 shows a category that a split cannot score)."""
    rows = []
    for split in ("train", "val", "test"):
        part = df[df["split"] == split]
        sets = [set(filter(None, s.split(";"))) for s in part["categories"]]
        rows.append({"split": split, **{c: sum(c in s for s in sets) for c in categories}})
    return pd.DataFrame(rows).set_index("split")
