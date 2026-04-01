import pandas as pd
import sys

EXPECTED_START = "2000-01-01"
EXPECTED_COLUMNS = ["CPI", "UNRATE", "M2", "OIL", "FEDFUNDS"]
VALID_RANGES = {
    "CPI":      (100, 400),
    "UNRATE":   (0, 25),
    "M2":       (0, 100_000),
    "OIL":      (0, 300),
    "FEDFUNDS": (0, 25),
}

def main():
    df = pd.read_csv("data/macro_data.csv", index_col=0, parse_dates=True)
    errors = []

    # 1. Column check
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    # 2. Date range check
    actual_start = df.index.min().strftime("%Y-%m-%d")
    actual_end   = df.index.max().strftime("%Y-%m-%d")
    if actual_start > EXPECTED_START:
        errors.append(f"Data starts at {actual_start}, expected {EXPECTED_START} or earlier")

    # 3. Monthly frequency check
    gaps = pd.date_range(df.index.min(), df.index.max(), freq="MS").difference(df.index)
    if not gaps.empty:
        errors.append(f"Missing months ({len(gaps)}): {gaps.tolist()}")

    # 4. Missing value check
    null_counts = df.isnull().sum()
    nulls = null_counts[null_counts > 0]
    if not nulls.empty:
        errors.append(f"Null values found:\n{nulls.to_string()}")

    # 5. Out-of-range check
    for col, (lo, hi) in VALID_RANGES.items():
        if col not in df.columns:
            continue
        out = df[(df[col] < lo) | (df[col] > hi)]
        if not out.empty:
            errors.append(f"{col} has {len(out)} out-of-range values (expected {lo}–{hi})")

    # Report
    print("=" * 50)
    print("DATA VALIDATION REPORT")
    print("=" * 50)
    print(f"Date range : {actual_start} to {actual_end}")
    print(f"Rows       : {len(df)}")
    print(f"Columns    : {list(df.columns)}")
    print()

    print("Summary statistics:")
    print(df.describe().round(2).to_string())
    print()

    if errors:
        print(f"FAILED — {len(errors)} issue(s) found:")
        for i, e in enumerate(errors, 1):
            print(f"  {i}. {e}")
        sys.exit(1)
    else:
        print("PASSED — all checks passed.")

if __name__ == "__main__":
    main()
