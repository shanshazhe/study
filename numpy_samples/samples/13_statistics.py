"""Statistics: mean, median, std, percentile."""

import numpy as np


def main() -> None:
    a = np.array([1, 2, 3, 4, 5, 6])

    print("mean:", a.mean())
    print("median:", np.median(a))
    print("std:", a.std())
    print("percentile 90:", np.percentile(a, 90))

    b = np.array([1, 2, 3, 4, 5, 6])
    print("corrcoef:\n", np.corrcoef(a, b), sep="")


if __name__ == "__main__":
    main()
