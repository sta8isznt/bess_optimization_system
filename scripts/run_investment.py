from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from investment.investment_model import compute_investment_metrics, load_financial_inputs


def main() -> None:
    summary_path = PROJECT_ROOT / "data_final" / "bess_financial_summary_fixed.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing financial summary CSV: {summary_path}")

    df = pd.read_csv(summary_path)
    metrics = compute_investment_metrics(df, load_financial_inputs())
    print(f"IRR: {metrics['irr'] * 100:.2f}%")
    print(f"NPV: EUR {metrics['npv']:,.2f}")
    print(f"Payback: {metrics['payback']:.2f} years")


if __name__ == "__main__":
    main()
