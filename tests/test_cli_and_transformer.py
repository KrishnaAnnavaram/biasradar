import json

import numpy as np
import pandas as pd
import pytest

from biasradar.cli import main
from biasradar.config import ConfigError, Settings


def test_settings():
    s = Settings.from_env({})
    assert s.seed == 42 and s.encoder == "microsoft/deberta-v3-base"
    with pytest.raises(ConfigError):
        Settings.from_env({"BIASRADAR_SEED": "x"})


def test_cli_round_trip(tmp_path, capsys):
    data = tmp_path / "synth.csv"
    assert main(["synth", "--out", str(data)]) == 0
    out = tmp_path / "model"
    assert main(["train", "--data", str(data), "--out", str(out)]) == 0
    for name in ("model.joblib", "label_map.json", "thresholds.json", "test_report.json", "model_card.md"):
        assert (out / name).is_file()
    assert "SYNTHETIC" in (out / "model_card.md").read_text(encoding="utf-8")
    capsys.readouterr()
    assert main(["evaluate", "--data", str(data), "--model-dir", str(out), "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["n"] > 0 and "by_source" in rep
    assert main(["predict", "All old people are lazy.", "--model-dir", str(out), "--explain"]) == 0
    assert "not a moderation decision" in capsys.readouterr().out


def test_cli_prepare(tmp_path, capsys):
    crows = tmp_path / "c.csv"
    n = 120
    pd.DataFrame({"sent_more": [f"sentence number {i}" for i in range(n)],
                  "bias_type": ["religion", "gender", "age"] * (n // 3),
                  "stereo_antistereo": ["stereo"] * n}).to_csv(crows)
    neutral = tmp_path / "n.csv"
    pd.DataFrame({"text": [f"neutral line {i}" for i in range(n)]}).to_csv(neutral, index=False)
    out = tmp_path / "u.csv"
    assert main(["prepare", "--crows", str(crows), "--neutral", str(neutral), "--out", str(out)]) == 0
    df = pd.read_csv(out)
    assert set(df["source"]) == {"crows", "neutral"} and set(df["split"]) == {"train", "val", "test"}


def test_cli_errors(tmp_path, capsys):
    assert main(["prepare", "--out", str(tmp_path / "x.csv")]) == 1
    assert main(["predict", "x", "--model-dir", str(tmp_path)]) == 1
    assert "error:" in capsys.readouterr().err


def test_demo(capsys):
    assert main(["demo"]) == 0
    assert "do not sum to 1" in capsys.readouterr().out


# -- transformer path: needs torch and transformers (skips in CI) ---------------------------------------
def _tiny_transformer(tmp_path, corpus_texts):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from biasradar.transformer import TransformerBiasModel

    words = sorted({w.strip(".,").lower() for t in corpus_texts for w in t.split()})
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]), encoding="utf-8")
    tok = transformers.BertTokenizerFast(vocab_file=str(vocab))
    cfg = transformers.BertConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=1,
                                  num_attention_heads=2, intermediate_size=64)
    torch.manual_seed(0)
    model = TransformerBiasModel("tiny", seed=0, epochs=2, batch_size=32, lr=1e-3, device="cpu")
    return model.build(tok, transformers.BertModel(cfg))


def test_masked_losses_use_logits():
    torch = pytest.importorskip("torch")
    from biasradar.transformer import masked_losses

    b = torch.tensor([10.0, -10.0, 0.0])
    c = torch.zeros(3, 2)
    lb, lc = masked_losses(b, c, torch.tensor([1.0, 0.0, 1.0]), torch.tensor([True, True, False]),
                           torch.zeros(3, 2), torch.tensor([True, False, False]))
    assert float(lb) < 1e-3  # confident correct logits give a near-zero loss
    assert float(lc) == pytest.approx(np.log(2), rel=1e-4)


def test_tiny_transformer_trains_saves_and_loads(tmp_path, parts):
    train, val, test = parts
    model = _tiny_transformer(tmp_path, pd.concat([train, val, test])["text"])
    model.fit(train, val, ckpt_dir=tmp_path / "ckpt")
    assert len(model.history) == 2 and (tmp_path / "ckpt" / "best.pt").is_file()
    pb, pc = model.predict_proba(test["text"][:4])
    assert pb.shape == (4,) and pc.shape == (4, 10) and ((pc >= 0) & (pc <= 1)).all()
    from biasradar.model import load_model

    model.save(tmp_path / "saved")
    loaded = load_model(tmp_path / "saved")
    assert np.allclose(loaded.predict_proba(test["text"][:4])[0], pb, atol=1e-5)
