"""MultiIndex basics."""

import pandas as pd


def main() -> None:
    arrays = [
        ["A", "A", "B", "B"],
        ["x", "y", "x", "y"],
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=["group", "sub"])
    s = pd.Series([10, 12, 9, 15], index=index, name="value")

    print("series with MultiIndex:")
    print(s)

    print("\nselect group A:")
    print(s.loc["A"])

    df = s.reset_index().set_index(["group", "sub"])
    print("\nDataFrame with MultiIndex:")
    print(df)

    print("\nstack/unstack:")
    print(df["value"].unstack())


if __name__ == "__main__":
    main()
