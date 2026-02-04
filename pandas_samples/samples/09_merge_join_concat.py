"""Merge, join, and concat basics."""

import pandas as pd


def main() -> None:
    left = pd.DataFrame({"id": [1, 2, 3], "name": ["alice", "bob", "chris"]})
    right = pd.DataFrame({"id": [2, 3, 4], "score": [90, 85, 88]})

    print("inner merge:")
    print(pd.merge(left, right, on="id", how="inner"))

    print("\nleft merge:")
    print(pd.merge(left, right, on="id", how="left"))

    df_a = pd.DataFrame({"id": [1, 2], "v": [10, 20]})
    df_b = pd.DataFrame({"id": [3, 4], "v": [30, 40]})
    print("\nconcat rows:")
    print(pd.concat([df_a, df_b], ignore_index=True))

    print("\njoin on index:")
    df_c = df_a.set_index("id")
    df_d = pd.DataFrame({"score": [100, 200]}, index=[1, 2])
    print(df_c.join(df_d))


if __name__ == "__main__":
    main()
