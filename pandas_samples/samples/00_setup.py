"""Basic environment setup and version check."""

import pandas as pd


def main() -> None:
    print("pandas version:", pd.__version__)

    # Display options example
    pd.set_option("display.width", 100)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.max_rows", 10)

    df = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    print(df)


if __name__ == "__main__":
    main()
