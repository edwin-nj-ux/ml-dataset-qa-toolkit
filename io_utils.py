from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


@dataclass
class InputMeta:
    path: str
    format: str


def load_table(path: str) -> Tuple[pd.DataFrame, Dict]:
    """Load CSV or JSONL into a DataFrame."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(p)
        meta = {"path": str(p), "format": "csv"}
        return df, meta

    if suffix in {".jsonl", ".ndjson"}:
        df = pd.read_json(p, lines=True)
        meta = {"path": str(p), "format": "jsonl"}
        return df, meta

    raise ValueError(f"Unsupported input format: {suffix}. Use CSV or JSONL.")
