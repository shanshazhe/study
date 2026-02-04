"""Rolling windows and exponential moving averages."""

import pandas as pd


def main() -> None:
    s = pd.Series([10, 12, 9, 15, 8, 11])

    print("rolling mean (window=3):")
    print(s.rolling(window=3).mean())

    print("\nrolling std (window=3):")
    print(s.rolling(window=3).std())

    print("\nexpanding mean:")
    print(s.expanding().mean())

    print("\newm mean (span=3):")
    print(s.ewm(span=3, adjust=False).mean())


if __name__ == "__main__":
    main()
