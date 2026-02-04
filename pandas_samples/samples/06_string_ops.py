"""Vectorized string operations."""

import pandas as pd


def main() -> None:
    s = pd.Series([" Alice ", "Bob", "chris", None])

    print("strip and lower:")
    print(s.str.strip().str.lower())

    print("\ncontains 'a':")
    print(s.str.contains("a", case=False, na=False))

    print("\nsplit:")
    print(pd.Series(["a-b-c", "d-e"]).str.split("-"))

    print("\nextract domain:")
    emails = pd.Series(["a@x.com", "b@y.org", None])
    print(emails.str.extract(r"@(.+)$"))


if __name__ == "__main__":
    main()
