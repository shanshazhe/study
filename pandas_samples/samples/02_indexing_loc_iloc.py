"""Indexing and selection with loc and iloc."""

import pandas as pd


def main() -> None:
    df = pd.DataFrame(
        {
            "name": ["alice", "bob", "chris", "dana"],
            "age": [25, 31, 29, 22],
            "city": ["ny", "sf", "la", "ny"],
        },
        index=["u1", "u2", "u3", "u4"],
    )

    print("df:\n", df, sep="")

    print("\nloc label row u2:")
    print(df.loc["u2"])

    print("\niloc row 0-1, col 0-1:")
    print(df.iloc[0:2, 0:2])

    print("\nloc rows u1,u3 and columns name,city:")
    print(df.loc[["u1", "u3"], ["name", "city"]])

    print("\nset with loc:")
    df.loc["u4", "age"] = 23
    print(df)


if __name__ == "__main__":
    main()
