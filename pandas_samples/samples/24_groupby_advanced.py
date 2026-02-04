"""Advanced groupby patterns: multi-agg, named agg, and groupby with bins."""

import numpy as np
import pandas as pd


def main() -> None:
    df = pd.DataFrame(
        {
            "team": ["A", "A", "A", "B", "B", "C"],
            "player": ["p1", "p2", "p3", "p4", "p5", "p6"],
            "age": [23, 28, 31, 22, 27, 29],
            "score": [10, 12, 18, 9, 15, 14],
        }
    )

    # Named aggregation
    summary = df.groupby("team").agg(
        score_mean=("score", "mean"),
        score_max=("score", "max"),
        age_min=("age", "min"),
    )
    print("named agg:\n", summary, sep="")

    # Groupby on bins
    age_bins = pd.cut(df["age"], bins=[20, 25, 30, 35], right=False)
    by_age = df.groupby(age_bins, observed=True)["score"].mean()
    print("\nmean score by age bin:\n", by_age, sep="")

    # Multiple keys + size
    size_by_team_age = df.groupby(["team", "age"]).size()
    print("\nsize by team+age:\n", size_by_team_age, sep="")

    # Groupby + apply returning DataFrame
    def top_n(group: pd.DataFrame, n: int = 2) -> pd.DataFrame:
        return group.nlargest(n, "score")

    top2 = df.groupby("team", group_keys=False).apply(top_n)
    print("\ntop2 by team:\n", top2, sep="")

    # Groupby + transform to normalize within group
    z = df.groupby("team")["score"].transform(lambda x: (x - x.mean()) / x.std())
    print("\nwith z-score:\n", df.assign(z_score=z), sep="")

    # Aggregate with numpy ufuncs
    uf = df.groupby("team")["score"].agg([np.mean, np.std, np.min, np.max])
    print("\nufunc agg:\n", uf, sep="")


if __name__ == "__main__":
    main()
