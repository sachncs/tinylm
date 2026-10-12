"""Builds the offline token stores for the downloaded corpora."""

import pathlib

from tinylm.clean import Options
from tinylm.offline import Offline
from tinylm.prepare import Prepare

VOCAB = 8192
RUNS = pathlib.Path("runs")
CORPORA = ("small", "strict")


def main() -> None:
    """Prepares each corpus with no network access."""
    prepare = Prepare(
        vocab=VOCAB, kind="bpe", block=200, every=100, options=Options()
    )
    with Offline():
        for name in CORPORA:
            out = RUNS / f"{name}{VOCAB // 1024}k"
            if out.exists():
                print(f"{name}: exists, skipping")
                continue
            stats = prepare.run(pathlib.Path("data") / name, out)
            print(name, stats.train_tokens, stats.dev_tokens)


if __name__ == "__main__":
    main()
