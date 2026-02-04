"""Date axis and formatting."""

import datetime as dt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def main() -> None:
    dates = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(10)]
    values = [10, 12, 9, 14, 11, 13, 15, 12, 10, 14]

    fig, ax = plt.subplots()
    ax.plot(dates, values, marker="o")
    ax.set_title("Time Series")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate()

    fig.savefig("/tmp/matplotlib_dates.png")
    print("Saved /tmp/matplotlib_dates.png")


if __name__ == "__main__":
    main()
