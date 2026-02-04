"""Basic environment setup and version check."""

import numpy as np


def main() -> None:
    print("numpy version:", np.__version__)

    np.set_printoptions(precision=3, suppress=True)
    a = np.array([1, 2, 3], dtype=np.float32)
    print("array:", a)


if __name__ == "__main__":
    main()
