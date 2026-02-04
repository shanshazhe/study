"""Styler basics for notebook or HTML output."""

import pandas as pd


def main() -> None:
    df = pd.DataFrame({"name": ["alice", "bob"], "score": [88, 95]})

    styler = df.style.highlight_max(subset=["score"], color="#c7f5d9")
    html = styler.to_html()
    print("Styler HTML length:", len(html))


if __name__ == "__main__":
    main()
