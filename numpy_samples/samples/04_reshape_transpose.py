"""Reshape, transpose, and ravel/flatten."""

import numpy as np


def main() -> None:
    a = np.arange(12)
    b = a.reshape(3, 4)
    print("b:\n", b, sep="")

    print("\ntranspose:\n", b.T, sep="")
    print("\nreshape 2x6:\n", a.reshape(2, 6), sep="")

    print("\nravel:", b.ravel())
    print("flatten:", b.flatten())

    print("\nswapaxes 0 and 1:\n", b.swapaxes(0, 1), sep="")


if __name__ == "__main__":
    main()
