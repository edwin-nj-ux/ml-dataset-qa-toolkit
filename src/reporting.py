from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from .qa import QAResults


def _save_label_plot(label_dist: Dict[str, int], out_path: Path) -> None:
    if not label_dist:
        return

    # Convert to series for plotting
    s = pd.Series(label_dist).sort_values(ascending=False)

    plt.figure()
    s.plot(kind="bar")
    plt.title("Label distribution")
    plt.xlabel("label")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def write_report(results: QAResults, out_dir: Path, meta: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV: label distribution
    ld = pd.DataFrame(
        [{"label": k, "count": v} for k, v in results.label_dist.items()]
    ).sort_values("count", ascending=False)
    ld.to_csv(out_dir / "label_distribution.csv", index=False)

    # Plot
    _save_label_plot(results.label_dist, out_dir / "label_distribution.png")

    # JSON summary
    summary = {
        "input": meta,
        "n_rows": results.n_rows,
        "columns": results.columns,
        "missing_label_rate": results.missing_label_rate,
        "duplicate_ids": results.duplicate_ids,
        "imbalance": results.imbalance,
        "invalid_confidence": results.invalid_confidence,
        "below_min_confidence": results.below_min_confidence,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Markdown report
    md = []
    md.append("# Dataset QA Report\n")
    md.append(f"**Input:** `{meta.get('path','')}`  ")
    md.append(f"**Format:** `{meta.get('format','')}`\n")

    md.append("## Summary\n")
    md.append(f"- Rows: **{results.n_rows}**")
    md.append(f"- Missing label rate (`{results.label_col}`): **{results.missing_label_rate:.2%}**")
    md.append(f"- Duplicate IDs (`{results.id_col}`): **{results.duplicate_ids}**")
    md.append(f"- Invalid confidence values (`{results.confidence_col}`): **{results.invalid_confidence}**")
    if summary["below_min_confidence"]:
        md.append(f"- Below min confidence: **{summary['below_min_confidence']}**")

    md.append("\n## Label distribution\n")
    md.append(ld.to_markdown(index=False))

    md.append("\n## Imbalance indicators\n")
    md.append(f"- Max label share: **{results.imbalance.get('max_share', 0.0):.2%}**")
    md.append(f"- Min label share: **{results.imbalance.get('min_share', 0.0):.2%}**")

    md.append("\n## Artifacts\n")
    md.append("- `label_distribution.csv`\n- `label_distribution.png`\n- `summary.json`\n")

    (out_dir / "qa_report.md").write_text("\n".join(md), encoding="utf-8")
