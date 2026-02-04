"""Vectorization and simple performance tips."""

import numpy as np


def main() -> None:
    a = np.arange(1_000_000)
    b = np.arange(1_000_000)

    # Vectorized computation
    c = a * 2 + b * 3
    print("vectorized result sum:", c.sum())

    # Use where instead of Python loops
    mask = a % 2 == 0
    d = np.where(mask, a, -a)
    print("where result sum:", d.sum())


if __name__ == "__main__":
    main()
