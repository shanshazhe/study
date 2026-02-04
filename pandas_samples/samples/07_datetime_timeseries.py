"""Datetime handling and time series basics."""

import pandas as pd


def main() -> None:
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2025-01-01", periods=5, freq="D"),
            "value": [10, 12, 9, 14, 11],
        }
    )

    df["day_of_week"] = df["ts"].dt.day_name()
    df["month"] = df["ts"].dt.to_period("M")
    print(df)

    # Resample to 2-day bins
    df_ts = df.set_index("ts")
    print("\nresample 2D mean:")
    print(df_ts.resample("2D").mean(numeric_only=True))


if __name__ == "__main__":
    main()
