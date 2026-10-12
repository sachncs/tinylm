"""Runs one experiment file, optionally as a named variant."""

import argparse
import pathlib

from tinylm.experiment import Experiment


def main() -> None:
    """Loads the file, applies --set overrides under --name, and runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=pathlib.Path)
    parser.add_argument("--name")
    parser.add_argument("--set", action="append", default=[], dest="sets")
    args = parser.parse_args()
    experiment = Experiment.load(args.config)
    if args.sets or args.name:
        experiment = experiment.override(
            args.sets, args.name or experiment.name
        )
    print(experiment.run())


if __name__ == "__main__":
    main()
