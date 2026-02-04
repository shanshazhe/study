"""Secondary axis with twinx."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.exp(x / 10)

    fig, ax1 = plt.subplots()
    ax1.plot(x, y1, color="#4C78A8")
    ax1.set_ylabel("sin(x)")

    ax2 = ax1.twinx()
    ax2.plot(x, y2, color="#F58518")
    ax2.set_ylabel("exp(x/10)")

    ax1.set_title("Twin Axes")
    fig.savefig("/tmp/matplotlib_twin_axes.png")
    print("Saved /tmp/matplotlib_twin_axes.png")


if __name__ == "__main__":
    main()
