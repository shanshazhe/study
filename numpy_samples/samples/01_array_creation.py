"""Array creation methods."""

import numpy as np


def main() -> None:
    a = np.array([1, 2, 3])
    zeros = np.zeros((2, 3))
    ones = np.ones((2, 3))
    ar = np.arange(0, 10, 2)
    lin = np.linspace(0, 1, 5)
    eye = np.eye(3)

    print("array:", a)
    print("zeros:\n", zeros, sep="")
    print("ones:\n", ones, sep="")
    print("arange:", ar)
    print("linspace:", lin)
    print("eye:\n", eye, sep="")


if __name__ == "__main__":
    main()
