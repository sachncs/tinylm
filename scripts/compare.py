"""Paired bootstrap comparison of two runs on the same benchmark folder."""

import argparse
import json
import pathlib

from tinylm.bootstrap import Bootstrap


def load(run: pathlib.Path, name: str) -> dict[str, list[bool]]:
    """Reads the per-item outcomes a run saved for a benchmark folder."""
    return json.loads((run / f"outcomes-{name}.json").read_text())


def main() -> None:
    """Prints accuracy(first) - accuracy(second) with its 95% interval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=pathlib.Path)
    parser.add_argument("second", type=pathlib.Path)
    parser.add_argument("benchmark")
    args = parser.parse_args()
    gap = Bootstrap().paired(
        load(args.first, args.benchmark), load(args.second, args.benchmark)
    )
    verdict = (
        "distinguishable"
        if gap.low > 0 or gap.high < 0
        else "NOT distinguishable"
    )
    print(
        f"{args.first.name} - {args.second.name}: {gap.estimate:+.4f} "
        f"[{gap.low:+.4f}, {gap.high:+.4f}] ({verdict} at {gap.level:.0%})"
    )


if __name__ == "__main__":
    main()
