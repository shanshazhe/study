"""Performance tips: vectorization, category, downcast, and eval."""

import pandas as pd


def main() -> None:
    df = pd.DataFrame(
        {
            "city": ["ny", "sf", "la", "ny", "sf", "la"] * 1000,
            "x": list(range(6000)),
            "y": list(range(6000, 12000)),
        }
    )

    # Category for repeated strings
    df["city"] = df["city"].astype("category")

    # Downcast integers
    df["x"] = pd.to_numeric(df["x"], downcast="integer")
    df["y"] = pd.to_numeric(df["y"], downcast="integer")

    # Vectorized math instead of apply
    df["sum_xy"] = df["x"] + df["y"]

    # Using eval for expression speed on large data
    df = df.eval("diff_xy = y - x")

    print("dtypes:\n", df.dtypes, sep="")
    print("memory usage (bytes):\n", df.memory_usage(deep=True), sep="")
    print("\nhead:\n", df.head(), sep="")


if __name__ == "__main__":
    main()
