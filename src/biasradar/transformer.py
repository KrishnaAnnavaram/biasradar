"""Multi-task transformer detector (needs ``pip install biasradar[hf]``).

One encoder, two heads, raw logits, ``BCEWithLogitsLoss`` for both heads (so no activation is
applied two times), masks for unknown labels, ``pos_weight`` for rare categories, seeds, and the
best epoch reloaded before the thresholds are tuned. Import this module only when you need it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from biasradar.metrics import micro_macro, tune_biased_threshold, tune_thresholds
from biasradar.model import category_rows
from biasradar.sources import category_matrix
from biasradar.taxonomy import LabelMap


class MultiTaskModel(nn.Module):
    def __init__(self, encoder, n_categories: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.biased_head = nn.Linear(hidden, 1)
        self.category_head = nn.Linear(hidden, n_categories)

    def forward(self, input_ids, attention_mask, **kw):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        return self.biased_head(cls).squeeze(-1), self.category_head(cls)  # logits, no sigmoid


def masked_losses(b_logits, c_logits, y_b, m_b, Y_c, m_c, pos_weight=None):
    """Mean BCE-with-logits over the rows whose label is known. Returns (biased loss, category loss)."""
    bce = nn.functional.binary_cross_entropy_with_logits
    zero = b_logits.sum() * 0
    lb = bce(b_logits[m_b], y_b[m_b]) if m_b.any() else zero
    lc = bce(c_logits[m_c], Y_c[m_c], pos_weight=pos_weight) if m_c.any() else zero
    return lb, lc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TransformerBiasModel:
    kind = "transformer"

    def __init__(self, encoder_name: str = "microsoft/deberta-v3-base", seed: int = 42, epochs: int = 3,
                 batch_size: int = 16, lr: float = 2e-5, max_length: int = 128, device: str | None = None):
        self.encoder_name, self.seed, self.epochs = encoder_name, seed, epochs
        self.batch_size, self.lr, self.max_length = batch_size, lr, max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.label_map = LabelMap()
        self.biased_threshold = 0.5
        self.category_thresholds = np.full(len(self.label_map.categories), 0.5)
        self.history: list[dict] = []

    # -- construction ----------------------------------------------------------------------
    def build(self, tokenizer=None, encoder=None):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(self.encoder_name)
        enc = encoder or AutoModel.from_pretrained(self.encoder_name)
        self.net = MultiTaskModel(enc, len(self.label_map.categories)).to(self.device)
        return self

    def _batches(self, texts, labels=None, shuffle=False, rng=None):
        idx = np.arange(len(texts))
        if shuffle:
            rng.shuffle(idx)
        for s in range(0, len(idx), self.batch_size):
            b = idx[s:s + self.batch_size]
            enc = self.tokenizer([texts[i] for i in b], padding=True, truncation=True, max_length=self.max_length,
                                 return_tensors="pt")
            yield b, {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    @torch.no_grad()
    def predict_proba(self, texts) -> tuple[np.ndarray, np.ndarray]:
        texts = list(texts)
        self.net.eval()
        pb, pc = [], []
        for _, enc in self._batches(texts):
            b, c = self.net(**enc)
            pb.append(torch.sigmoid(b).cpu().numpy())
            pc.append(torch.sigmoid(c).cpu().numpy())
        return np.concatenate(pb), np.concatenate(pc)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None, ckpt_dir: str | Path = "artifacts/_ckpt"):
        set_seed(self.seed)
        if not hasattr(self, "net"):
            self.build()
        texts = train["text"].tolist()
        y_b = torch.tensor(np.nan_to_num(train["is_biased"].to_numpy(float)), dtype=torch.float32)
        m_b = torch.tensor(train["is_biased"].notna().to_numpy())
        Y_c = torch.tensor(category_matrix(train, self.label_map.categories), dtype=torch.float32)
        m_c = torch.tensor(category_rows(train).to_numpy())
        pos = Y_c[m_c].sum(0)
        neg = m_c.sum() - pos
        pos_weight = torch.where(pos > 0, neg / pos.clamp(min=1), torch.ones_like(pos)).clamp(max=50).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=0.01)
        rng = np.random.default_rng(self.seed)
        ckpt = Path(ckpt_dir)
        ckpt.mkdir(parents=True, exist_ok=True)
        best = -1.0
        for epoch in range(self.epochs):
            self.net.train()
            total = 0.0
            for b, enc in self._batches(texts, shuffle=True, rng=rng):
                bl, cl = self.net(**enc)
                lb, lc = masked_losses(bl, cl, y_b[b].to(self.device), m_b[b].to(self.device),
                                       Y_c[b].to(self.device), m_c[b].to(self.device), pos_weight)
                loss = lb + lc
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(b)
            score = float("nan")
            if val is not None and len(val):
                pb, pc = self.predict_proba(val["text"])
                vm = category_rows(val).to_numpy()
                mm = micro_macro(category_matrix(val[vm], self.label_map.categories), pc[vm], np.full(pc.shape[1], 0.5))
                score = mm["macro_f1"]
            self.history.append({"epoch": epoch, "train_loss": total / len(texts), "val_macro_f1": score})
            if val is None or score > best:
                best = score
                torch.save(self.net.state_dict(), ckpt / "best.pt")
        self.net.load_state_dict(torch.load(ckpt / "best.pt", map_location=self.device, weights_only=True))
        if val is not None and len(val):
            pb, pc = self.predict_proba(val["text"])
            self.biased_threshold = tune_biased_threshold(val["is_biased"].to_numpy(float), pb)
            vm = category_rows(val).to_numpy()
            self.category_thresholds = tune_thresholds(category_matrix(val[vm], self.label_map.categories), pc[vm])
        return self

    # -- persistence -------------------------------------------------------------------------
    def save(self, out_dir: str | Path, extra: dict | None = None) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.net.encoder.save_pretrained(out / "encoder")
        self.tokenizer.save_pretrained(out / "encoder")
        torch.save({"biased_head": self.net.biased_head.state_dict(),
                    "category_head": self.net.category_head.state_dict()}, out / "heads.pt")
        self.label_map.save(out / "label_map.json")
        (out / "thresholds.json").write_text(json.dumps({
            "kind": self.kind, "biased": self.biased_threshold, "max_length": self.max_length,
            "categories": dict(zip(self.label_map.categories, map(float, self.category_thresholds))),
            "history": self.history, **(extra or {})}, indent=1), encoding="utf-8")
        return out

    @classmethod
    def load(cls, model_dir: str | Path) -> "TransformerBiasModel":
        from transformers import AutoModel, AutoTokenizer

        d = Path(model_dir)
        th = json.loads((d / "thresholds.json").read_text(encoding="utf-8"))
        obj = cls(str(d / "encoder"), max_length=th.get("max_length", 128))
        obj.label_map = LabelMap.load(d / "label_map.json")
        obj.build(AutoTokenizer.from_pretrained(d / "encoder"), AutoModel.from_pretrained(d / "encoder"))
        heads = torch.load(d / "heads.pt", map_location=obj.device, weights_only=True)
        obj.net.biased_head.load_state_dict(heads["biased_head"])
        obj.net.category_head.load_state_dict(heads["category_head"])
        obj.biased_threshold = th["biased"]
        obj.category_thresholds = np.asarray([th["categories"][c] for c in obj.label_map.categories])
        return obj
