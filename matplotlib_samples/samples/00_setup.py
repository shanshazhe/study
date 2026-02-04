"""Basic environment setup and version check."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    print("matplotlib version:", matplotlib.__version__)

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title("Setup Check")
    fig.savefig("/tmp/matplotlib_setup.png")
    print("Saved /tmp/matplotlib_setup.png")


if __name__ == "__main__":
    main()
