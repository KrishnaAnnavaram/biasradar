"""Model card: written next to each saved model, from the measured evaluation."""

from __future__ import annotations

from datetime import date
from pathlib import Path

LIMITS = [
    "The detector finds generalisations about social groups in English text. It does not judge intent or context.",
    "The categories follow the CrowS-Pairs and IndiBias taxonomy. Other kinds of bias are not in the output.",
    "Stereotype benchmarks contain written sentences, not real posts. Scores on real text can be lower.",
    "The output is not a moderation or hiring decision. A person must review each result.",
]


def write_card(model_dir: str | Path, model_kind: str, data_sources: list[str], report: dict, synthetic: bool) -> Path:
    b = report["biased"]
    lines = [
        f"# Model card: biasradar {model_kind}",
        "",
        f"- Date: {date.today().isoformat()}",
        f"- Training data: {', '.join(data_sources)}{' (SYNTHETIC)' if synthetic else ''}",
        "- Outputs: p(biased) and one independent probability for each of the 10 categories in `label_map.json`",
        "",
        "## Test results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Rows | {report['n']} |",
        f"| Biased head ROC-AUC | {b.get('roc_auc', float('nan')):.3f} |",
        f"| Biased head F1 | {b['f1']:.3f} |",
        f"| False alarms on neutral rows | {b['neutral_false_alarm_rate']:.3f} |",
        f"| Category micro F1 | {report['micro_f1']:.3f} |",
        f"| Category macro F1 | {report['macro_f1']:.3f} |",
        "",
        "## Limits",
        "",
        *[f"- {x}" for x in LIMITS],
        "",
    ]
    path = Path(model_dir) / "model_card.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
