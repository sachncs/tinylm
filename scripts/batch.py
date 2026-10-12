"""Runs the pending variants of a queue file, one after another."""

import argparse
import pathlib

from tinylm import jobs, logs
from tinylm.experiment import Experiment, Queue


def main() -> None:
    """Runs every queued variant that has no metrics.json yet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=pathlib.Path)
    args = parser.parse_args()
    logs.setup()
    queue = Queue.load(args.queue)
    base = Experiment.load(queue.base)
    jobs.run_all(
        jobs.command(queue.base, item.name, item.sets)
        for item in queue.pending(base.out.parent)
    )


if __name__ == "__main__":
    main()
