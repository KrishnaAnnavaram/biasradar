"""Settings from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class ConfigError(ValueError):
    """A setting has a value that the package cannot use."""


def load_dotenv(path: str | os.PathLike = ".env") -> int:
    p = Path(path)
    if not p.is_file():
        return 0
    added = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (x.strip() for x in line.split("=", 1))
        value = value.strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            added += 1
    return added


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    model_dir: Path
    seed: int
    encoder: str

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        raw = (env.get("BIASRADAR_SEED") or "42").strip()
        try:
            seed = int(raw)
        except ValueError as exc:
            raise ConfigError(f"BIASRADAR_SEED must be an integer, got {raw!r}") from exc
        return cls(
            data_dir=Path((env.get("BIASRADAR_DATA_DIR") or "data").strip()),
            model_dir=Path((env.get("BIASRADAR_MODEL_DIR") or "artifacts/model").strip()),
            seed=seed,
            encoder=(env.get("BIASRADAR_ENCODER") or "microsoft/deberta-v3-base").strip(),
        )
