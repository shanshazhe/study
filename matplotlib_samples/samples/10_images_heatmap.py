"""imshow and heatmap."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    data = np.random.rand(10, 10)

    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_title("Heatmap")

    fig.savefig("/tmp/matplotlib_heatmap.png")
    print("Saved /tmp/matplotlib_heatmap.png")


if __name__ == "__main__":
    main()
