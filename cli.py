import argparse
from pathlib import Path

from src.io_utils import load_table
from src.qa import run_qa
from src.reporting import write_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ML Dataset QA Toolkit")
    p.add_argument("--input", required=True, help="Path to CSV or JSONL")
    p.add_argument("--out", default="reports", help="Output directory")
    p.add_argument("--label-col", default="label", help="Label column name")
    p.add_argument("--id-col", default="item_id", help="ID column name")
    p.add_argument("--confidence-col", default="confidence", help="Confidence column name (optional)")
    p.add_argument("--min-confidence", type=float, default=None, help="Filter/flag items below this confidence")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, meta = load_table(args.input)

    if args.confidence_col in df.columns:
        below = (df[args.confidence_col] < 0.5).sum()
        print(f"Low-confidence samples (<0.5): {below}")
    else:
        print(f"Note: confidence column '{args.confidence_col}' not found; skipping confidence summary.")

    results = run_qa(
        df=df,
        id_col=args.id_col,
        label_col=args.label_col,
        confidence_col=args.confidence_col,
        min_confidence=args.min_confidence,
    )

    write_report(results=results, out_dir=out_dir, meta=meta)
    print(f"Done. Wrote report to: {out_dir / 'qa_report.md'}")


if __name__ == "__main__":
    main()
