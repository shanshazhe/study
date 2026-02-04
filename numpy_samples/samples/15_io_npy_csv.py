"""Save/load npy and CSV."""

import numpy as np


def main() -> None:
    a = np.arange(6).reshape(2, 3)

    np.save("/tmp/array.npy", a)
    b = np.load("/tmp/array.npy")
    print("npy load:\n", b, sep="")

    np.savetxt("/tmp/array.csv", a, delimiter=",")
    c = np.loadtxt("/tmp/array.csv", delimiter=",")
    print("\ncsv load:\n", c, sep="")


if __name__ == "__main__":
    main()
