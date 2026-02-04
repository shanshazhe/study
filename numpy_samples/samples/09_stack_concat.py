"""Stack, concatenate, and split."""

import numpy as np


def main() -> None:
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])

    print("concat axis=0:\n", np.concatenate([a, b], axis=0), sep="")
    print("concat axis=1:\n", np.concatenate([a, b], axis=1), sep="")

    print("\nvstack:\n", np.vstack([a, b]), sep="")
    print("hstack:\n", np.hstack([a, b]), sep="")

    c = np.arange(10)
    print("\nsplit:", np.split(c, 5))


if __name__ == "__main__":
    main()
