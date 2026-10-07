"""The unified label schema: one ``is_biased`` flag and 10 multi-label bias categories.

All sources (CrowS-Pairs, IndiBias, SBIC, synthetic data) go through ``normalise_category``, so the
same category never appears twice with two spellings (for example ``Religion`` and ``religion``).
The ``LabelMap`` is saved with each model and loaded by the CLI and the app.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CATEGORIES: tuple[str, ...] = (
    "age", "caste", "disability", "gender", "nationality", "physical-appearance",
    "race-color", "religion", "sexual-orientation", "socioeconomic",
)
TAXONOMY_VERSION = "1.0"

ALIASES = {
    "race": "race-color", "race_color": "race-color", "racecolor": "race-color", "color": "race-color",
    "physical_appearance": "physical-appearance", "body": "physical-appearance", "appearance": "physical-appearance",
    "sexual_orientation": "sexual-orientation", "sexuality": "sexual-orientation", "lgbt": "sexual-orientation",
    "disabled": "disability", "socioeconomic status": "socioeconomic", "ses": "socioeconomic", "class": "socioeconomic",
    "nationality/country": "nationality", "country": "nationality", "religious": "religion",
}
# SBIC targetCategory values. "culture", "social" and "victim" mix several categories, so they map to
# no category (the post still counts as biased).
SBIC_CATEGORY_MAP = {"race": "race-color", "gender": "gender", "disabled": "disability", "body": "physical-appearance",
                     "culture": None, "social": None, "victim": None}


class UnknownCategory(KeyError):
    """A category text has no place in the taxonomy."""


def normalise_category(raw: str) -> str:
    key = str(raw).strip().lower().replace("  ", " ")
    key = ALIASES.get(key, key)
    if key not in CATEGORIES:
        raise UnknownCategory(f"unknown bias category {raw!r}. Known: {CATEGORIES}")
    return key


def encode_categories(cats) -> list[int]:
    """Multi-hot vector over CATEGORIES."""
    s = {normalise_category(c) for c in cats}
    return [int(c in s) for c in CATEGORIES]


@dataclass(frozen=True)
class LabelMap:
    categories: tuple[str, ...] = CATEGORIES
    version: str = TAXONOMY_VERSION

    def to_json(self) -> str:
        return json.dumps({"categories": list(self.categories), "version": self.version}, indent=1)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(self.to_json(), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "LabelMap":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        cats = tuple(d["categories"])
        if len(set(cats)) != len(cats):
            raise ValueError("the label map has a repeated category")
        return cls(cats, d.get("version", "?"))

    def name(self, index: int) -> str:
        return self.categories[index]
