"""Optional IO engines: Excel and Parquet."""

import pandas as pd


def main() -> None:
    df = pd.DataFrame({"id": [1, 2], "value": [10, 20]})

    # Excel requires openpyxl or xlsxwriter
    try:
        df.to_excel("/tmp/pandas_sample.xlsx", index=False)
        df_excel = pd.read_excel("/tmp/pandas_sample.xlsx")
        print("Excel round-trip:\n", df_excel, sep="")
    except Exception as exc:
        print("Excel example skipped:", exc)

    # Parquet requires pyarrow or fastparquet
    try:
        df.to_parquet("/tmp/pandas_sample.parquet", index=False)
        df_parquet = pd.read_parquet("/tmp/pandas_sample.parquet")
        print("\nParquet round-trip:\n", df_parquet, sep="")
    except Exception as exc:
        print("Parquet example skipped:", exc)


if __name__ == "__main__":
    main()
