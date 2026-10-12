"""Runs the pending variants of a queue file, one after another."""

import argparse
import pathlib

from tinylm.experiment import Experiment, Queue


def main() -> None:
    """Runs every queued variant that has no metrics.json yet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=pathlib.Path)
    args = parser.parse_args()
    queue = Queue.load(args.queue)
    base = Experiment.load(queue.base)
    for item in queue.pending(base.out.parent):
        print(f"== {item.name} {list(item.sets)}", flush=True)
        print(base.override(list(item.sets), item.name).run(), flush=True)


if __name__ == "__main__":
    main()
