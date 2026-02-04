"""Aggregation across axes."""

import numpy as np


def main() -> None:
    a = np.arange(12).reshape(3, 4)
    print("a:\n", a, sep="")

    print("sum all:", a.sum())
    print("sum axis=0:", a.sum(axis=0))
    print("sum axis=1:", a.sum(axis=1))

    print("mean keepdims:\n", a.mean(axis=1, keepdims=True), sep="")


if __name__ == "__main__":
    main()
