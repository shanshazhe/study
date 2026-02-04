"""Multiple subplots in a figure."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    x = np.linspace(0, 1, 100)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes[0, 0].plot(x, x)
    axes[0, 0].set_title("y=x")

    axes[0, 1].plot(x, x ** 2)
    axes[0, 1].set_title("y=x^2")

    axes[1, 0].plot(x, np.sqrt(x))
    axes[1, 0].set_title("y=sqrt(x)")

    axes[1, 1].plot(x, np.sin(2 * np.pi * x))
    axes[1, 1].set_title("y=sin(2pi x)")

    fig.tight_layout()
    fig.savefig("/tmp/matplotlib_subplots.png")
    print("Saved /tmp/matplotlib_subplots.png")


if __name__ == "__main__":
    main()
