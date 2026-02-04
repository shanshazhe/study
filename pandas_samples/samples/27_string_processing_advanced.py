"""Advanced string processing: regex, extract, replace, and split expand."""

import pandas as pd


def main() -> None:
    s = pd.Series([
        "Alice Zhang <alice@x.com>",
        "Bob Li <bob@y.org>",
        "NoEmail",
        None,
    ])

    # Extract name and email
    extracted = s.str.extract(r"^(?P<name>[^<]+)\s<(?P<email>[^>]+)>")
    print("extract name/email:\n", extracted, sep="")

    # Replace domain
    emails = extracted["email"].fillna("")
    print("\nreplace domain:\n", emails.str.replace(r"@.+$", "@example.com", regex=True), sep="")

    # Split with expand
    codes = pd.Series(["A-001", "B-002", "C-003"])
    split = codes.str.split("-", expand=True)
    split.columns = ["prefix", "num"]
    print("\nsplit expand:\n", split, sep="")

    # Normalize whitespace
    messy = pd.Series(["  a  b ", "c\t d", "e\n  f"])
    print("\nnormalize whitespace:\n", messy.str.replace(r"\s+", " ", regex=True).str.strip(), sep="")


if __name__ == "__main__":
    main()
