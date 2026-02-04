"""Categorical dtype usage."""

import pandas as pd


def main() -> None:
    s = pd.Series(["low", "medium", "high", "medium", "low"])
    cat = pd.Categorical(s, categories=["low", "medium", "high"], ordered=True)
    df = pd.DataFrame({"priority": cat})

    print(df)
    print("\nordered compare:")
    print(df["priority"] > "low")

    print("\nvalue counts:")
    print(df["priority"].value_counts())


if __name__ == "__main__":
    main()
