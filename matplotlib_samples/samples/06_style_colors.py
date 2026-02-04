"""Styles and colors."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    x = np.linspace(0, 10, 100)

    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots()
    ax.plot(x, np.sin(x), color="#4C78A8", linewidth=2)
    ax.plot(x, np.cos(x), color="#F58518", linewidth=2)
    ax.set_title("Style and Colors")

    fig.savefig("/tmp/matplotlib_style.png")
    print("Saved /tmp/matplotlib_style.png")


if __name__ == "__main__":
    main()
