"""Broadcasting rules and examples."""

import numpy as np


def main() -> None:
    a = np.ones((3, 4))
    b = np.arange(4)

    print("a shape:", a.shape)
    print("b shape:", b.shape)
    print("a + b:\n", a + b, sep="")

    c = np.arange(3).reshape(3, 1)
    print("\nouter sum (3x1 + 1x4):\n", c + b, sep="")


if __name__ == "__main__":
    main()
