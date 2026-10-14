"""Builds the offline token stores for the downloaded corpora."""

import pathlib

from tinylm.clean import Options
from tinylm.offline import Offline
from tinylm.prepare import Prepare

RUNS = pathlib.Path("runs")
STORES = (
    ("small", 4096),
    ("small", 8192),
    ("small", 16384),
    ("strict", 16384),
)


def main() -> None:
    """Prepares each corpus and vocabulary with no network access."""
    with Offline():
        for name, vocab in STORES:
            out = RUNS / f"{name}{vocab // 1024}k"
            if out.exists():
                print(f"{out}: exists, kept")
                continue
            prepare = Prepare(
                vocab=vocab, kind="bpe", block=200, every=100, options=Options()
            )
            stats = prepare.run(pathlib.Path("data") / name, out)
            print(out, stats.train_tokens, stats.dev_tokens)


if __name__ == "__main__":
    main()
