"""Missing data handling."""

import numpy as np
import pandas as pd


def main() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": [10, 20, None, 40],
        }
    )

    print("isna:\n", df.isna(), sep="")

    print("\nfillna with 0:")
    print(df.fillna(0))

    print("\nforward fill:")
    print(df.fillna(method="ffill"))

    print("\ndrop rows with any missing:")
    print(df.dropna())


if __name__ == "__main__":
    main()
