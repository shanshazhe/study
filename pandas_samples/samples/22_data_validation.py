"""Basic data validation patterns."""

import pandas as pd


def main() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "age": [25, 31, 29],
            "score": [88, 95, 75],
        }
    )

    # Schema-like checks
    expected_cols = {"id", "age", "score"}
    if set(df.columns) != expected_cols:
        raise ValueError("Unexpected columns")

    # Value constraints
    if (df["age"] < 0).any():
        raise ValueError("Age must be non-negative")

    # Uniqueness
    if df["id"].duplicated().any():
        raise ValueError("Duplicate id found")

    print("Validation passed")


if __name__ == "__main__":
    main()
