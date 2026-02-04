"""Advanced missing data handling: interpolation and conditional fill."""

import numpy as np
import pandas as pd


def main() -> None:
    df = pd.DataFrame(
        {
            "day": pd.date_range("2025-01-01", periods=6, freq="D"),
            "value": [1.0, np.nan, 3.0, np.nan, 5.0, 6.0],
        }
    )

    print("original:\n", df, sep="")

    # Interpolate linearly
    df["value_interp"] = df["value"].interpolate(method="linear")
    print("\ninterpolated:\n", df, sep="")

    # Conditional fill: use mean for missing
    mean_value = df["value"].mean()
    df["value_filled"] = df["value"].fillna(mean_value)
    print("\nfilled with mean:\n", df, sep="")


if __name__ == "__main__":
    main()
