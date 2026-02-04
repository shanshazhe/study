"""Masking and NaN handling."""

import numpy as np


def main() -> None:
    a = np.array([1.0, np.nan, 3.0, np.nan, 5.0])

    print("isnan:", np.isnan(a))
    print("nanmean:", np.nanmean(a))
    print("nan_to_num:", np.nan_to_num(a, nan=0.0))

    # Masked arrays
    m = np.ma.masked_invalid(a)
    print("masked mean:", m.mean())


if __name__ == "__main__":
    main()
