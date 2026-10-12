"""Measures n-gram overlap between a raw corpus and BLiMP-style files."""

import argparse
import pathlib

from tinylm.bench import Pair
from tinylm.clean import Options
from tinylm.offline import Offline
from tinylm.overlap import Overlap
from tinylm.prepare import Prepare


def main() -> None:
    """Reports overlap for each eval folder and fails above the threshold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=pathlib.Path)
    parser.add_argument("evals", type=pathlib.Path, nargs="+")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.02)
    args = parser.parse_args()
    prepare = Prepare(
        vocab=300, kind="bpe", block=1, every=1, options=Options()
    )
    with Offline():
        sources = prepare.read(args.corpus)
        lines = [line for name in sorted(sources) for line in sources[name]]
        for folder in args.evals:
            sentences = [
                sentence
                for path in sorted(folder.glob("*.jsonl"))
                for pair in Pair.read(path)
                for sentence in (pair.good, pair.bad)
            ]
            found = Overlap(args.n).check(lines, sentences, args.threshold)
            print(f"{folder.name}: overlap {found:.4f} (n={args.n})")


if __name__ == "__main__":
    main()
