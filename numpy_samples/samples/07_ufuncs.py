"""Universal functions and element-wise operations."""

import numpy as np


def main() -> None:
    a = np.array([1, 4, 9, 16])

    print("sqrt:", np.sqrt(a))
    print("exp:", np.exp(a))
    print("add:", np.add(a, 1))

    # ufunc with out
    out = np.empty_like(a, dtype=np.float64)
    np.sqrt(a, out=out)
    print("sqrt out:", out)


if __name__ == "__main__":
    main()
