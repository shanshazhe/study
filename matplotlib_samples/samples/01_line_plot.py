"""Basic line plot with labels and legend."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    x = np.linspace(0, 2 * np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    fig, ax = plt.subplots()
    ax.plot(x, y1, label="sin")
    ax.plot(x, y2, label="cos")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Sine and Cosine")
    ax.legend()

    fig.savefig("/tmp/matplotlib_line.png")
    print("Saved /tmp/matplotlib_line.png")


if __name__ == "__main__":
    main()
