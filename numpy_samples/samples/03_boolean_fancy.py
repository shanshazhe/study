"""Boolean and fancy indexing."""

import numpy as np


def main() -> None:
    a = np.arange(10)
    mask = a % 2 == 0

    print("a:", a)
    print("even mask:", mask)
    print("even values:", a[mask])

    idx = [1, 3, 5]
    print("fancy index [1,3,5]:", a[idx])

    # where
    b = np.where(a > 5, a, -1)
    print("where a>5 else -1:", b)


if __name__ == "__main__":
    main()
