"""Boolean filtering, sorting, and ranking."""

import pandas as pd


def main() -> None:
    df = pd.DataFrame(
        {
            "name": ["alice", "bob", "chris", "dana"],
            "age": [25, 31, 29, 22],
            "score": [88, 95, 75, 91],
        }
    )

    print("score >= 90:")
    print(df[df["score"] >= 90])

    print("\nsort by age desc:")
    print(df.sort_values("age", ascending=False))

    print("\nTop 2 scores:")
    print(df.nlargest(2, "score"))

    print("\nrank by score:")
    print(df.assign(rank=df["score"].rank(ascending=False)))


if __name__ == "__main__":
    main()
