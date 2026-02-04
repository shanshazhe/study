"""Type inspection and conversion."""

import pandas as pd


def main() -> None:
    df = pd.DataFrame(
        {
            "price": ["10.5", "20.0", "7.25"],
            "qty": ["1", "3", "2"],
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        }
    )

    print("dtypes before:\n", df.dtypes, sep="")

    df["price"] = pd.to_numeric(df["price"])
    df["qty"] = df["qty"].astype(int)
    df["date"] = pd.to_datetime(df["date"])

    print("\ndtypes after:\n", df.dtypes, sep="")
    print("\nrevenue:")
    print(df.assign(revenue=df["price"] * df["qty"]))


if __name__ == "__main__":
    main()
