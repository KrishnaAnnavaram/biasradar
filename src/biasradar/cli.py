"""Command line interface: ``biasradar <command> [options]``."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from biasradar.card import write_card
from biasradar.config import ConfigError, Settings, load_dotenv
from biasradar.evaluate import evaluate, format_summary
from biasradar.explain import DISCLAIMER, format_prediction, predict_text, top_terms
from biasradar.model import TfidfBiasModel, load_model
from biasradar.sources import SourceError, combine, load_crows_pairs, load_indibias, load_neutral, load_sbic
from biasradar.splits import coverage, make_splits
from biasradar.synthetic import make_corpus
from biasradar.taxonomy import CATEGORIES, UnknownCategory


def _read_split(path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=True)
    df["categories"] = df["categories"].fillna("")
    if "split" not in df.columns:
        raise ValueError(f"{path} has no split column. Run `biasradar prepare` first.")
    return df


def cmd_synth(args, settings) -> int:
    df = make_splits(make_corpus(seed=args.seed), seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} SYNTHETIC rows with splits to {args.out}")
    return 0


def cmd_prepare(args, settings) -> int:
    frames = []
    if args.crows:
        frames.append(load_crows_pairs(args.crows))
    if args.indibias:
        frames.append(load_indibias(args.indibias))
    for path in args.sbic or []:
        frames.append(load_sbic(path))
    if args.neutral:
        frames.append(load_neutral(args.neutral))
    if not frames:
        raise ValueError("give at least one source: --crows, --indibias, --sbic or --neutral")
    df = make_splits(combine(frames), seed=settings.seed)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")
    print(df.groupby(["source", "split"]).size().unstack(fill_value=0).to_string())
    cov = coverage(df, CATEGORIES)
    missing = [c for c in CATEGORIES if cov.loc["test", c] == 0 and cov.loc["train", c] > 0]
    if missing:
        print(f"warning: no test rows for {missing}")
    return 0


def cmd_train(args, settings) -> int:
    df = _read_split(args.data)
    train, val, test = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    if args.model == "tfidf":
        model = TfidfBiasModel(seed=settings.seed).fit(train, val)
    else:
        try:
            from biasradar.transformer import TransformerBiasModel
        except ImportError as exc:
            raise ImportError("the transformer model needs: pip install 'biasradar[hf]'") from exc
        model = TransformerBiasModel(args.encoder or settings.encoder, seed=settings.seed, epochs=args.epochs)
        model.fit(train, val, ckpt_dir=Path(args.out) / "_ckpt")
    out = Path(args.out) if args.out else settings.model_dir
    model.save(out, {"train_rows": len(train), "data": str(args.data)})
    rep = evaluate(model, test)
    (out / "test_report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    write_card(out, model.kind, sorted(df["source"].unique()), rep, synthetic=df["source"].str.startswith("synth").all())
    print(f"saved {model.kind} model to {out}")
    print(format_summary(rep))
    return 0


def cmd_evaluate(args, settings) -> int:
    model = load_model(args.model_dir or settings.model_dir)
    df = _read_split(args.data)
    rep = evaluate(model, df[df["split"] == args.split].reset_index(drop=True))
    print(json.dumps(rep, indent=1, default=float) if args.json else format_summary(rep))
    return 0


def cmd_predict(args, settings) -> int:
    model = load_model(args.model_dir or settings.model_dir)
    for text in args.text:
        print(format_prediction(predict_text(model, text)))
        if args.explain and model.kind == "tfidf":
            print("  terms that raise p(biased): " + ", ".join(f"{t} ({w:.2f})" for t, w in top_terms(model, text)))
    print(DISCLAIMER)
    return 0


def cmd_app(args, settings) -> int:
    app = Path(__file__).parent / "app" / "streamlit_app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *args.streamlit_args])


def cmd_demo(args, settings) -> int:
    df = make_splits(make_corpus(seed=settings.seed), seed=settings.seed)
    train, val, test = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
    print(f"SYNTHETIC corpus: {len(df)} rows (train {len(train)}, val {len(val)}, test {len(test)})")
    with tempfile.TemporaryDirectory() as tmp:
        model = TfidfBiasModel(seed=settings.seed).fit(train, val)
        model.save(tmp)
        model = load_model(tmp)  # the label map comes from the saved artefact
        print(format_summary(evaluate(model, test)))
        for text in ("All old people are bad with money.", "The train to the city leaves at nine in the morning."):
            print()
            print(text)
            print(format_prediction(predict_text(model, text), top=3))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="biasradar", description="Detect social-bias generalisations and their categories.")
    p.add_argument("--env-file", default=".env")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="write a SYNTHETIC unified corpus with splits")
    s.add_argument("--out", default="data/synthetic.csv")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("prepare", help="load sources, normalise labels, make grouped splits")
    s.add_argument("--crows")
    s.add_argument("--indibias")
    s.add_argument("--sbic", nargs="*")
    s.add_argument("--neutral")
    s.add_argument("--out", default="data/unified.csv")
    s.set_defaults(func=cmd_prepare)

    s = sub.add_parser("train", help="train, tune thresholds on val, evaluate on test, save model + card")
    s.add_argument("--data", required=True)
    s.add_argument("--model", default="tfidf", choices=["tfidf", "transformer"])
    s.add_argument("--encoder", help="Hugging Face encoder id (transformer model)")
    s.add_argument("--epochs", type=int, default=3)
    s.add_argument("--out", help="model folder (default: BIASRADAR_MODEL_DIR)")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("evaluate", help="evaluate a saved model on one split")
    s.add_argument("--data", required=True)
    s.add_argument("--model-dir")
    s.add_argument("--split", default="test", choices=["train", "val", "test"])
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("predict", help="score one or more texts")
    s.add_argument("text", nargs="+")
    s.add_argument("--model-dir")
    s.add_argument("--explain", action="store_true")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("app", help="start the Streamlit app (needs the app extra)")
    s.add_argument("streamlit_args", nargs=argparse.REMAINDER)
    s.set_defaults(func=cmd_app)

    s = sub.add_parser("demo", help="offline demo on SYNTHETIC data")
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(args.env_file)
    try:
        return int(args.func(args, Settings.from_env()))
    except (ConfigError, SourceError, UnknownCategory, FileNotFoundError, KeyError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
