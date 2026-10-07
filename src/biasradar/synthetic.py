"""Synthetic, template-based sentences for the offline demo and the tests.

Biased sentences are generic generalisations about a group ("All X people are Y"). Neutral
sentences mention the same kind of groups or everyday topics without a generalisation, so the
detector cannot learn "group word = bias". The texts are mild on purpose and are not real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUPS = {
    "age": ["old", "elderly", "young", "teenage"],
    "caste": ["brahmin", "dalit", "upper-caste", "lower-caste"],
    "disability": ["blind", "deaf", "wheelchair-using", "autistic"],
    "gender": ["female", "male", "women", "men"],
    "nationality": ["Mexican", "Chinese", "Irish", "Nigerian"],
    "physical-appearance": ["overweight", "short", "tall", "skinny"],
    "race-color": ["Black", "white", "Asian", "Latino"],
    "religion": ["Muslim", "Christian", "Hindu", "Jewish"],
    "sexual-orientation": ["gay", "lesbian", "bisexual", "straight"],
    "socioeconomic": ["poor", "rich", "working-class", "homeless"],
}
TRAITS = ["lazy", "bad with money", "always late", "too emotional", "not smart", "dishonest", "bad drivers",
          "unable to lead", "loud", "not to be trusted"]
BIASED_TEMPLATES = [
    "All {g} people are {t}.",
    "You know {g} people, they are {t}.",
    "No wonder he failed, {g} people are {t}.",
    "Never hire {g} people because they are {t}.",
    "{G} people are just {t}, everyone knows it.",
    "Typical. The {g} ones are {t} again.",
    "I would not rent to {g} families, they are {t}.",
]
NEUTRAL_TEMPLATES = [
    "The {g} engineer presented the new design at the meeting.",
    "A {g} neighbour helped us carry the boxes upstairs.",
    "The survey asked {g} participants about their commute.",
    "Our {g} colleague organised the team lunch on Friday.",
    "The {g} people at the event were tired after the long day.",
    "Some {g} runners said the race was too long.",
]
EVERYDAY = [
    "The train to the city leaves at nine in the morning.", "Please send the report before the weekend.",
    "The library extended its opening hours this month.", "We planted tomatoes and basil in the garden.",
    "The museum has a new exhibition about old maps.", "Rain is expected for most of the afternoon.",
    "She fixed the bicycle chain in ten minutes.", "The recipe needs two cups of flour and an egg.",
    "The printer is always late with large jobs.", "My old laptop is not to be trusted with backups.",
    "The cat was lazy all afternoon.", "The new manager was unable to lead the call because of a bad line.",
]
LABEL_NOISE = 0.04  # share of biased and neutral rows whose flag is flipped, to keep the task imperfect


def make_corpus(n_biased: int = 600, n_neutral: int = 600, n_antistereo: int = 100, seed: int = 0) -> pd.DataFrame:
    """Unified frame (see ``sources.UNIFIED_COLUMNS``) with two synthetic sources: ``synth_a`` and ``synth_b``."""
    rng = np.random.default_rng(seed)
    cats = list(GROUPS)
    rows = []
    for i in range(n_biased):
        cat = cats[i % len(cats)] if i < len(cats) else str(rng.choice(cats))
        g = str(rng.choice(GROUPS[cat]))
        text = str(rng.choice(BIASED_TEMPLATES)).format(g=g, G=g[0].upper() + g[1:], t=str(rng.choice(TRAITS)))
        rows.append((text, cat, 1.0, "stereo"))
    for _ in range(n_antistereo):
        cat = str(rng.choice(cats))
        g = str(rng.choice(GROUPS[cat]))
        rows.append((f"Some say {g} people are {rng.choice(TRAITS)}, but that is not true.", cat, np.nan, "antistereo"))
    for i in range(n_neutral):
        if rng.random() < 0.6:
            cat = str(rng.choice(cats))
            text = str(rng.choice(NEUTRAL_TEMPLATES)).format(g=str(rng.choice(GROUPS[cat])))
        else:
            text = str(rng.choice(EVERYDAY))
        rows.append((text, "", 0.0, "neutral"))
    df = pd.DataFrame(rows, columns=["text", "categories", "is_biased", "polarity"])
    flip = (rng.random(len(df)) < LABEL_NOISE) & df["is_biased"].notna()
    df.loc[flip, "is_biased"] = 1.0 - df.loc[flip, "is_biased"]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["source"] = np.where(rng.random(len(df)) < 0.5, "synth_a", "synth_b")
    df["group_id"] = [f"synth:{i}" for i in range(len(df))]
    return df[["text", "source", "group_id", "is_biased", "categories", "polarity"]]
