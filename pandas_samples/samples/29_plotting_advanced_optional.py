"""Advanced plotting with matplotlib (optional)."""

import pandas as pd


def main() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=6, freq="D"),
            "sales": [120, 150, 90, 200, 180, 210],
            "cost": [70, 90, 60, 110, 100, 130],
        }
    )

    try:
        ax = df.set_index("date").plot(kind="bar", title="Sales vs Cost")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig("/tmp/pandas_plot_bar.png")
        print("Plot saved to /tmp/pandas_plot_bar.png")
    except Exception as exc:
        print("Plot example skipped:", exc)


if __name__ == "__main__":
    main()
