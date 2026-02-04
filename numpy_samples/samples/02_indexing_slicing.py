"""Basic indexing and slicing."""

import numpy as np


def main() -> None:
    a = np.arange(12).reshape(3, 4)
    print("a:\n", a, sep="")

    print("\nrow 1:", a[1])
    print("col 2:", a[:, 2])
    print("slice rows 0-1, cols 1-3:\n", a[0:2, 1:4], sep="")

    print("\nnegative index last row:", a[-1])


if __name__ == "__main__":
    main()
