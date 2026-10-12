"""Prints a markdown table of the metrics.json files under a folder."""

import argparse
import json
import pathlib


def main() -> None:
    """Summarises size, dev bits per character and benchmark accuracy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=pathlib.Path)
    args = parser.parse_args()
    rows = [
        (path.parent.name, json.loads(path.read_text()))
        for path in sorted(args.runs.glob("*/metrics.json"))
    ]
    names = sorted({k for _, m in rows for k in m["benchmarks"]})
    print("| run | params (M) | bits/char | " + " | ".join(names) + " |")
    print("|---|---|---|" + "---|" * len(names))
    for name, m in rows:
        cells = [f"{m['benchmarks'].get(n, float('nan')):.4f}" for n in names]
        print(
            f"| {name} | {m['parameters'] / 1e6:.1f} | "
            f"{m['bits_per_char']:.4f} | " + " | ".join(cells) + " |"
        )


if __name__ == "__main__":
    main()
