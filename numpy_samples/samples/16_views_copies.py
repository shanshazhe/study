"""Views vs copies."""

import numpy as np


def main() -> None:
    a = np.arange(6)
    view = a[1:4]
    copy = a[1:4].copy()

    view[:] = 100
    print("a after view change:", a)

    copy[:] = 200
    print("a after copy change:", a)

    print("view base is a:", view.base is a)
    print("copy base is None:", copy.base is None)


if __name__ == "__main__":
    main()
