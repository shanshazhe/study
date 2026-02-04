"""Grid and tick formatting."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("Grid and Ticks")
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(["0", "pi", "2pi"])

    fig.savefig("/tmp/matplotlib_grid_ticks.png")
    print("Saved /tmp/matplotlib_grid_ticks.png")


if __name__ == "__main__":
    main()
