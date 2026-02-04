"""Saving figures with DPI and tight layout."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    x = np.linspace(0, 4, 200)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("Save Fig")

    fig.tight_layout()
    fig.savefig("/tmp/matplotlib_savefig.png", dpi=150)
    print("Saved /tmp/matplotlib_savefig.png")


if __name__ == "__main__":
    main()
