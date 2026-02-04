"""Series and DataFrame creation basics."""

import pandas as pd


def main() -> None:
    s = pd.Series([10, 20, 30], name="scores", index=["a", "b", "c"])
    print("Series:")
    print(s)

    df = pd.DataFrame(
        {
            "name": ["alice", "bob", "chris"],
            "age": [25, 31, 29],
            "city": ["ny", "sf", "la"],
        }
    )
    print("\nDataFrame:")
    print(df)

    # From list of dicts
    records = [
        {"name": "dana", "age": 22},
        {"name": "eric", "age": 27, "city": "seattle"},
    ]
    df2 = pd.DataFrame(records)
    print("\nFrom records:")
    print(df2)


if __name__ == "__main__":
    main()
