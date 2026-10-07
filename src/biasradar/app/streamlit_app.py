"""Streamlit app: ``biasradar app`` (needs ``pip install biasradar[app]``).

The model and its label map load once (``st.cache_resource``). The page shows the biased probability
and the independent probability of each category, with the disclaimer.
"""

from __future__ import annotations

import os

import streamlit as st

from biasradar.explain import DISCLAIMER, predict_text, top_terms
from biasradar.model import load_model

MODEL_DIR = os.environ.get("BIASRADAR_MODEL_DIR", "artifacts/model")


@st.cache_resource
def get_model(path: str):
    return load_model(path)


st.set_page_config(page_title="biasradar", layout="centered")
st.title("biasradar")
st.caption(DISCLAIMER)
model = get_model(MODEL_DIR)
text = st.text_area("Text", "All old people are bad with money.")
if st.button("Check") and text.strip():
    pred = predict_text(model, text)
    st.metric("p(biased)", f"{pred['p_biased']:.2f}", "flagged" if pred["biased"] else "not flagged")
    st.write("Category probabilities (independent, they do not sum to 1):")
    st.bar_chart({row["category"]: row["p"] for row in pred["categories"]})
    if getattr(model, "kind", "") == "tfidf":
        st.write("Terms that raise p(biased):", ", ".join(t for t, _ in top_terms(model, text)) or "none")
