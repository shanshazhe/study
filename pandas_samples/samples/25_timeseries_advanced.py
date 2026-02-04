"""Advanced time series: timezone, resample, rolling on time, and asof."""

import pandas as pd


def main() -> None:
    # Create hourly data and localize to UTC
    idx = pd.date_range("2025-01-01", periods=48, freq="H", tz="UTC")
    df = pd.DataFrame({"value": range(48)}, index=idx)

    # Convert timezone
    df_local = df.tz_convert("America/Los_Angeles")
    print("tz convert head:\n", df_local.head(3), sep="")

    # Resample to 6-hour mean
    print("\nresample 6H mean:\n", df_local.resample("6H").mean(), sep="")

    # Time-based rolling window
    print("\nrolling 6H sum:\n", df_local.rolling("6H").sum().head(8), sep="")

    # As-of merge for nearest previous timestamp
    left = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2025-01-01 02:30", "2025-01-01 10:10", "2025-01-01 20:00"], utc=True),
            "event": ["a", "b", "c"],
        }
    ).sort_values("ts")

    right = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 08:00", "2025-01-01 16:00"], utc=True),
            "level": [100, 110, 120],
        }
    ).sort_values("ts")

    merged = pd.merge_asof(left, right, on="ts")
    print("\nmerge_asof:\n", merged, sep="")


if __name__ == "__main__":
    main()
