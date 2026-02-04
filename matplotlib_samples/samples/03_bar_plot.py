"""Bar and horizontal bar plots."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    labels = ["A", "B", "C", "D"]
    values = [10, 15, 7, 12]

    fig, ax = plt.subplots()
    ax.bar(labels, values, color="#4C78A8")
    ax.set_title("Bar Plot")
    fig.savefig("/tmp/matplotlib_bar.png")

    fig2, ax2 = plt.subplots()
    ax2.barh(labels, values, color="#F58518")
    ax2.set_title("Horizontal Bar Plot")
    fig2.savefig("/tmp/matplotlib_barh.png")

    print("Saved /tmp/matplotlib_bar.png and /tmp/matplotlib_barh.png")


if __name__ == "__main__":
    main()
