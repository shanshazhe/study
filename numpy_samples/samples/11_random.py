"""Random sampling with default_rng."""

import numpy as np


def main() -> None:
    rng = np.random.default_rng(42)

    print("integers:", rng.integers(0, 10, size=5))
    print("normal:\n", rng.normal(0, 1, size=(2, 3)), sep="")

    a = np.arange(5)
    rng.shuffle(a)
    print("shuffled:", a)


if __name__ == "__main__":
    main()
