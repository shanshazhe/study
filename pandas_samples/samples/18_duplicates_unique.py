"""Duplicate detection and unique counts."""

import pandas as pd


def main() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 3, 3, 3],
            "value": [10, 10, 20, 30, 30, 31],
        }
    )

    print("duplicates by row:")
    print(df.duplicated())

    print("\ndrop duplicates:")
    print(df.drop_duplicates())

    print("\nunique ids:")
    print(df["id"].nunique())


if __name__ == "__main__":
    main()
