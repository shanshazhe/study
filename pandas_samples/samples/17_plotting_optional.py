"""Plotting with matplotlib (optional)."""

import pandas as pd


def main() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 15, 8, 12]})

    try:
        ax = df.plot(x="x", y="y", kind="line", title="Simple Line Plot")
        fig = ax.get_figure()
        fig.savefig("/tmp/pandas_plot.png")
        print("Plot saved to /tmp/pandas_plot.png")
    except Exception as exc:
        print("Plot example skipped:", exc)


if __name__ == "__main__":
    main()
