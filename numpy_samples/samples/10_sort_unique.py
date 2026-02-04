"""Sorting and unique values."""

import numpy as np


def main() -> None:
    a = np.array([3, 1, 2, 3, 2, 1])

    print("sorted:", np.sort(a))
    print("argsort:", np.argsort(a))
    print("unique:", np.unique(a))

    b = np.array([1, 2, 5])
    print("in1d:", np.in1d(a, b))


if __name__ == "__main__":
    main()
