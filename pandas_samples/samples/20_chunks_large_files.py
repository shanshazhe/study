"""Chunked reading for large files."""

import io
import pandas as pd


def main() -> None:
    csv_text = """id,value
1,10
2,20
3,30
4,40
5,50
"""
    reader = pd.read_csv(io.StringIO(csv_text), chunksize=2)

    total = 0
    for chunk in reader:
        total += chunk["value"].sum()
        print("chunk:\n", chunk, sep="")

    print("\nTotal value:", total)


if __name__ == "__main__":
    main()
