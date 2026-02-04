"""apply, map, and vectorized operations."""

import pandas as pd


def main() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})

    print("vectorized add:")
    print(df.assign(z=df["x"] + df["y"]))

    print("\napply row-wise:")
    print(df.apply(lambda row: row["x"] * row["y"], axis=1))

    print("\nmap on Series:")
    s = pd.Series(["a", "b", "c"])
    mapping = {"a": "A", "b": "B"}
    print(s.map(mapping).fillna("C"))


if __name__ == "__main__":
    main()
