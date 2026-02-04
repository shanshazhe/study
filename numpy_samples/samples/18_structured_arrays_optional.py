"""Structured arrays (optional advanced topic)."""

import numpy as np


def main() -> None:
    dtype = np.dtype([("name", "U10"), ("age", "i4"), ("score", "f4")])
    data = np.array(
        [("alice", 25, 88.5), ("bob", 31, 91.0), ("chris", 29, 77.0)],
        dtype=dtype,
    )

    print("structured array:\n", data, sep="")
    print("names:", data["name"])
    print("scores > 80:\n", data[data["score"] > 80], sep="")


if __name__ == "__main__":
    main()
