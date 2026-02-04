"""Memory usage inspection and optimization ideas."""

import pandas as pd


def main() -> None:
    df = pd.DataFrame(
        {
            "city": ["ny", "sf", "la", "ny", "sf", "la"],
            "value": [10, 20, 30, 40, 50, 60],
        }
    )

    print("memory usage (bytes):")
    print(df.memory_usage(deep=True))

    df_opt = df.copy()
    df_opt["city"] = df_opt["city"].astype("category")
    df_opt["value"] = pd.to_numeric(df_opt["value"], downcast="integer")

    print("\noptimized dtypes:")
    print(df_opt.dtypes)
    print("\noptimized memory usage (bytes):")
    print(df_opt.memory_usage(deep=True))


if __name__ == "__main__":
    main()
