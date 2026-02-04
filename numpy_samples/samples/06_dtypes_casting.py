"""dtype inspection and casting."""

import numpy as np


def main() -> None:
    a = np.array([1, 2, 3], dtype=np.int32)
    b = a.astype(np.float64)

    print("a dtype:", a.dtype)
    print("b dtype:", b.dtype)

    c = np.array([1.2, 2.7, 3.9])
    print("c to int:", c.astype(np.int64))

    d = np.array(["1", "2", "3"]).astype(np.int64)
    print("string to int:", d)


if __name__ == "__main__":
    main()
