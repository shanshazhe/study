"""Histogram and bins."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0, scale=1, size=1000)

    fig, ax = plt.subplots()
    ax.hist(data, bins=30, color="#54A24B", alpha=0.8)
    ax.set_title("Histogram")
    ax.set_xlabel("value")
    ax.set_ylabel("count")

    fig.savefig("/tmp/matplotlib_hist.png")
    print("Saved /tmp/matplotlib_hist.png")


if __name__ == "__main__":
    main()
