"""Linear algebra basics."""

import numpy as np


def main() -> None:
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[2, 0], [1, 2]])

    print("dot:\n", a.dot(b), sep="")
    print("matmul:\n", a @ b, sep="")

    # Solve Ax = y
    y = np.array([1, 2])
    x = np.linalg.solve(a, y)
    print("solve Ax=y, x:", x)

    # Determinant
    print("det(a):", np.linalg.det(a))


if __name__ == "__main__":
    main()
