"""Colormap examples."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack([gradient, gradient])

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.imshow(gradient, aspect="auto", cmap="plasma")
    ax.set_axis_off()
    ax.set_title("Colormap: plasma")

    fig.savefig("/tmp/matplotlib_colormap.png")
    print("Saved /tmp/matplotlib_colormap.png")


if __name__ == "__main__":
    main()
