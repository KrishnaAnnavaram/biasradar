import numpy as np
import pandas as pd
import pytest

from biasradar.sources import (SourceError, aggregate_sbic, category_matrix, combine, load_crows_pairs, load_indibias,
                               load_neutral)
from biasradar.splits import SplitLeakError, assert_no_group_leak, coverage, make_splits
from biasradar.taxonomy import CATEGORIES, LabelMap, UnknownCategory, encode_categories, normalise_category


def test_religion_spellings_become_one_category():
    assert normalise_category("Religion") == normalise_category("religion ") == "religion"
    assert normalise_category("Caste") == "caste" and normalise_category("race") == "race-color"
    with pytest.raises(UnknownCategory):
        normalise_category("politics")
    assert len(set(CATEGORIES)) == len(CATEGORIES) == 10


def test_label_map_round_trip(tmp_path):
    path = LabelMap().save(tmp_path / "label_map.json")
    loaded = LabelMap.load(path)
    assert loaded.categories == CATEGORIES and loaded.name(7) == "religion"
    path.write_text('{"categories": ["age", "age"]}', encoding="utf-8")
    with pytest.raises(ValueError):
        LabelMap.load(path)


def test_encode_categories():
    assert encode_categories(["Religion", "age"]) == [1, 0, 0, 0, 0, 0, 0, 1, 0, 0]


def _pairs_csv(tmp_path, name, text_col, labels, polarity):
    df = pd.DataFrame({text_col: [f"sentence {i}" for i in range(len(labels))], "bias_type": labels,
                       "stereo_antistereo": polarity, "index": range(100, 100 + len(labels))})
    path = tmp_path / name
    df.to_csv(path)
    return path


def test_crows_and_indibias_loaders(tmp_path):
    crows = load_crows_pairs(_pairs_csv(tmp_path, "c.csv", "sent_more", ["religion", "gender"], ["stereo", "antistereo"]))
    indi = load_indibias(_pairs_csv(tmp_path, "i.csv", "modified_eng_sent_more", ["Religion", "Caste"], ["stereo", "stereo"]))
    assert list(crows["categories"]) == ["religion", "gender"]
    assert crows["is_biased"].iloc[0] == 1.0 and np.isnan(crows["is_biased"].iloc[1])  # anti-stereotype: unknown
    assert list(indi["categories"]) == ["religion", "caste"] and indi["group_id"].iloc[0] == "indibias:100"


def test_loader_refuses_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["x"]}).to_csv(path, index=False)
    with pytest.raises(SourceError):
        load_crows_pairs(path)


def test_sbic_aggregation_keeps_posts_without_target():
    raw = pd.DataFrame({
        "post": ["p1", "p1", "p1", "p2", "p2", "p3"],
        "offensiveYN": [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "targetCategory": ["race", "race", None, None, None, "culture"],
        "whoTarget": [None] * 6,  # extra columns with missing values must not drop rows
    })
    out = aggregate_sbic(raw).set_index("text")
    assert len(out) == 3  # one row for each post
    assert out.loc["p1", "is_biased"] == 1.0 and out.loc["p1", "categories"] == "race-color"
    assert out.loc["p2", "is_biased"] == 0.0 and out.loc["p2", "polarity"] == "neutral"
    assert out.loc["p3", "categories"] == ""  # culture maps to no category


def test_neutral_loader_and_combine(tmp_path):
    path = tmp_path / "n.csv"
    pd.DataFrame({"text": ["a b", "a b", " "]}).to_csv(path, index=False)
    df = combine([load_neutral(path)])
    assert len(df) == 1 and df["is_biased"].iloc[0] == 0.0


def test_splits_keep_groups_and_cover_categories(corpus):
    assert_no_group_leak(corpus)
    cov = coverage(corpus, CATEGORIES)
    assert (cov.loc["test"] > 0).all() and (cov.loc["val"] > 0).all()
    assert set(corpus["split"]) == {"train", "val", "test"}


def test_pairs_with_one_group_stay_together():
    df = pd.DataFrame({"text": [f"t{i}" for i in range(200)], "source": "s", "group_id": [f"g{i // 2}" for i in range(200)],
                       "is_biased": [1.0, 0.0] * 100, "categories": ["age", ""] * 100, "polarity": "stereo"})
    split = make_splits(df, seed=3)
    assert (split.groupby("group_id")["split"].nunique() == 1).all()
    split.loc[0, "split"] = "test"
    split.loc[1, "split"] = "train"
    with pytest.raises(SplitLeakError):
        assert_no_group_leak(split)


def test_category_matrix(corpus):
    Y = category_matrix(corpus.head(5), CATEGORIES)
    assert Y.shape == (5, 10) and set(np.unique(Y)) <= {0, 1}
