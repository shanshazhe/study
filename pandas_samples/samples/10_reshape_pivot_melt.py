"""Reshaping with pivot, pivot_table, and melt."""

import pandas as pd


def main() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "metric": ["views", "clicks", "views", "clicks"],
            "value": [100, 7, 150, 9],
        }
    )

    print("pivot:")
    print(df.pivot(index="id", columns="metric", values="value"))

    print("\npivot_table (mean):")
    print(df.pivot_table(index="id", columns="metric", values="value", aggfunc="mean"))

    wide = pd.DataFrame(
        {
            "id": [1, 2],
            "views": [100, 150],
            "clicks": [7, 9],
        }
    )
    print("\nmelt:")
    print(wide.melt(id_vars="id", var_name="metric", value_name="value"))


if __name__ == "__main__":
    main()
