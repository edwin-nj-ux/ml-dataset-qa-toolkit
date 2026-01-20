from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


def wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER).

    WER = (S + D + I) / N, where N is number of words in reference.
    Returns 0.0 for identical strings. If reference is empty, returns 0.0 if
    hypothesis is also empty, else 1.0.
    """
    ref = reference.strip().split()
    hyp = hypothesis.strip().split()

    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    # Levenshtein distance on word tokens
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return float(dp[len(ref)][len(hyp)] / len(ref))





def label_distribution(df: pd.DataFrame, label_col: str) -> pd.Series:
    return df[label_col].value_counts(dropna=False)


def missing_rate(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(df[col].isna().mean())


def duplicates_count(df: pd.DataFrame, id_col: str) -> int:
    if id_col not in df.columns:
        return 0
    return int(df[id_col].duplicated().sum())


def invalid_confidence_count(df: pd.DataFrame, confidence_col: str) -> int:
    if confidence_col not in df.columns:
        return 0
    c = df[confidence_col]
    # invalid if not numeric or outside [0,1]
    invalid = (~pd.to_numeric(c, errors="coerce").between(0, 1)) | c.isna()
    return int(invalid.sum())


def below_confidence_count(df: pd.DataFrame, confidence_col: str, threshold: Optional[float]) -> int:
    if threshold is None or confidence_col not in df.columns:
        return 0
    c = pd.to_numeric(df[confidence_col], errors="coerce")
    return int((c < threshold).sum())


def imbalance_score(dist: pd.Series) -> Dict[str, float]:
    """Simple imbalance metrics to flag skew.

    Returns:
      - max_share: fraction of rows in most common label
      - min_share: fraction in least common label (excluding NaN)
    """
    total = float(dist.sum()) if dist.sum() else 0.0
    if total == 0:
        return {"max_share": 0.0, "min_share": 0.0}

    shares = (dist / total).astype(float)
    # Exclude NaN label (if present as actual NaN, it appears in dist index as NaN)
    shares_no_nan = shares[shares.index == shares.index]

    if len(shares_no_nan) == 0:
        return {"max_share": float(shares.max()), "min_share": 0.0}

    return {"max_share": float(shares_no_nan.max()), "min_share": float(shares_no_nan.min())}
