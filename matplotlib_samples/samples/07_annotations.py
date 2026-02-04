"""Annotations and text."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    x = np.linspace(0, 5, 50)
    y = x ** 2

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("Annotation Example")

    ax.annotate(
        "peak",
        xy=(5, 25),
        xytext=(3, 20),
        arrowprops=dict(arrowstyle="->", color="#4C78A8"),
    )
    ax.text(0.5, 5, "y = x^2", color="#F58518")

    fig.savefig("/tmp/matplotlib_annotations.png")
    print("Saved /tmp/matplotlib_annotations.png")


if __name__ == "__main__":
    main()
