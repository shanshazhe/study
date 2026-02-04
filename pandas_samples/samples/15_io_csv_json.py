"""Read and write CSV/JSON using in-memory buffers."""

import io
import pandas as pd


def main() -> None:
    csv_text = """id,name,score
1,alice,88
2,bob,92
3,chris,75
"""
    df = pd.read_csv(io.StringIO(csv_text))
    print("from CSV:\n", df, sep="")

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    print("\nCSV output:\n", buf.getvalue(), sep="")

    json_text = df.to_json(orient="records")
    print("\nJSON output:\n", json_text, sep="")

    df2 = pd.read_json(io.StringIO(json_text))
    print("\nfrom JSON:\n", df2, sep="")


if __name__ == "__main__":
    main()
