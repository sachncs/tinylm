"""Runs one experiment file, optionally as a named variant."""

import argparse
import pathlib
import signal
import sys
import threading

from tinylm import logs
from tinylm.experiment import Error, Experiment

INTERRUPTED = 130


def main() -> None:
    """Loads the file, applies --set overrides under --name, and runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=pathlib.Path)
    parser.add_argument("--name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--set", action="append", default=[], dest="sets")
    args = parser.parse_args()
    logs.setup()
    experiment = Experiment.load(args.config)
    if args.sets or args.name:
        experiment = experiment.override(
            args.sets, args.name or experiment.name
        )
    stop = threading.Event()
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda *_: stop.set())
    try:
        print(experiment.run(resume=args.resume, halt=stop.is_set))
    except Error as error:
        if not error.resumable:
            raise
        print(error, file=sys.stderr)
        sys.exit(INTERRUPTED)


if __name__ == "__main__":
    main()
