"""query and eval for concise expressions."""

import pandas as pd


def main() -> None:
    df = pd.DataFrame(
        {
            "city": ["ny", "sf", "la", "ny"],
            "age": [25, 31, 29, 22],
            "score": [88, 95, 75, 91],
        }
    )

    print("query city == 'ny' and score > 85:")
    print(df.query("city == 'ny' and score > 85"))

    print("\ncreate column with eval:")
    print(df.eval("score_per_age = score / age"))


if __name__ == "__main__":
    main()
