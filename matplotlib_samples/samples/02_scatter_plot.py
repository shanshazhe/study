"""Scatter plot with size and color mapping."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = rng.normal(size=200)
    sizes = rng.uniform(20, 200, size=200)
    colors = rng.normal(size=200)

    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, s=sizes, c=colors, cmap="viridis", alpha=0.7)
    fig.colorbar(sc, ax=ax)
    ax.set_title("Scatter with Color and Size")

    fig.savefig("/tmp/matplotlib_scatter.png")
    print("Saved /tmp/matplotlib_scatter.png")


if __name__ == "__main__":
    main()
