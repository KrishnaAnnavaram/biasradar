import json

import numpy as np
import pandas as pd
import pytest

from biasradar.evaluate import evaluate, format_summary
from biasradar.explain import format_prediction, predict_text, top_terms
from biasradar.metrics import (_best_threshold, biased_report, category_report, expected_calibration_error,
                               tune_biased_threshold, tune_thresholds)
from biasradar.model import TfidfBiasModel, category_rows, load_model
from biasradar.taxonomy import CATEGORIES


def test_scores_are_independent_and_neutral_text_scores_low(model, parts):
    test = parts[2]
    neutral = test[test["polarity"] == "neutral"]
    pb, pc = model.predict_proba(neutral["text"])
    assert np.median(pb) < 0.5 and np.median(pc.max(axis=1)) < 0.5  # no forced category
    assert not np.allclose(pc.sum(axis=1), 1.0)  # not a softmax


def test_biased_text_is_flagged_with_its_category(model):
    pred = predict_text(model, "All elderly people are bad with money.")
    assert pred["biased"] and pred["categories"][0]["category"] == "age"


def test_saved_label_map_drives_the_names(model, tmp_path):
    model.save(tmp_path)
    data = json.loads((tmp_path / "label_map.json").read_text(encoding="utf-8"))
    data["categories"] = list(reversed(data["categories"]))  # a model saved with another order
    (tmp_path / "label_map.json").write_text(json.dumps(data), encoding="utf-8")
    th = json.loads((tmp_path / "thresholds.json").read_text(encoding="utf-8"))
    loaded = load_model(tmp_path)
    assert loaded.label_map.categories[0] == "socioeconomic"
    assert list(loaded.category_thresholds) == [th["categories"][c] for c in loaded.label_map.categories]


def test_round_trip_gives_same_predictions(model, tmp_path):
    model.save(tmp_path)
    loaded = load_model(tmp_path)
    texts = ["All old people are lazy.", "Rain is expected for most of the afternoon."]
    for a, b in zip(model.predict_proba(texts), loaded.predict_proba(texts)):
        assert np.allclose(a, b)


def test_load_model_without_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path)


def test_one_class_biased_data_is_reported_not_hidden(parts):
    train, val, test = parts
    only_stereo = train[(train["is_biased"] == 1) | train["is_biased"].isna()]
    m = TfidfBiasModel().fit(only_stereo, val)
    assert not m.biased_head_trained and top_terms(m, "x") == []
    rep = evaluate(m, test)
    assert rep["biased"]["trained"] is False and "NOT trained" in format_summary(rep)


def test_category_rows_mask(parts):
    train = parts[0]
    mask = category_rows(train)
    assert mask[train["is_biased"] == 0].all()
    sbic_like = pd.DataFrame({"categories": [""], "is_biased": [1.0]})
    assert not category_rows(sbic_like).iloc[0]  # biased with unknown category: not a negative


def test_evaluation_report(model, parts):
    rep = evaluate(model, parts[2])
    assert rep["biased"]["roc_auc"] > 0.8 and rep["macro_f1"] > 0.8
    assert set(rep["by_source"]) == {"synth_a", "synth_b"}
    assert {r["category"] for r in rep["categories"]} == set(CATEGORIES)
    assert 0 <= rep["biased"]["neutral_false_alarm_rate"] <= 1
    assert "macro F1" in format_summary(rep)


def test_metric_helpers():
    assert expected_calibration_error(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0
    rep = biased_report(np.array([1, 0, np.nan, 0]), np.array([0.9, 0.2, 0.5, 0.7]), 0.5)
    assert rep["n"] == 3 and rep["neutral_false_alarm_rate"] == 0.5
    Y = np.array([[1, 0], [0, 0], [1, 0]])
    P = np.array([[0.8, 0.1], [0.3, 0.1], [0.6, 0.2]])
    th = tune_thresholds(Y, P)
    assert th[1] == 0.5 and 0.3 < th[0] <= 0.6
    table = category_report(Y, P, th, ["a", "b"])
    assert table.loc[0, "f1"] == 1.0 and np.isnan(table.loc[1, "pr_auc"])
    assert tune_biased_threshold(np.array([1.0, 1.0]), np.array([0.2, 0.3])) == 0.5


def test_threshold_ties_prefer_the_middle():
    grid = np.linspace(0.05, 0.95, 19)
    assert _best_threshold(np.ones(19), grid) == pytest.approx(0.5)


def test_explanations(model):
    terms = top_terms(model, "Never hire deaf people because they are dishonest.")
    assert terms and all(w > 0 for _, w in terms)
    text = format_prediction(predict_text(model, "All rich people are loud."))
    assert "do not sum to 1" in text
