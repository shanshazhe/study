"""Groupby, aggregation, and transform."""

import pandas as pd


def main() -> None:
    df = pd.DataFrame(
        {
            "team": ["A", "A", "B", "B", "B"],
            "player": ["p1", "p2", "p3", "p4", "p5"],
            "score": [10, 12, 9, 15, 8],
        }
    )

    print("mean score by team:")
    print(df.groupby("team")["score"].mean())

    print("\nagg multiple:")
    print(df.groupby("team").agg(score_mean=("score", "mean"), score_max=("score", "max")))

    print("\ntransform z-score within team:")
    grouped = df.groupby("team")["score"]
    z = (grouped.transform(lambda x: (x - x.mean()) / x.std()))
    print(df.assign(z_score=z))


if __name__ == "__main__":
    main()
