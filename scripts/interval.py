"""Adds 95% bootstrap intervals (in-house scorer) to a run's full BLiMP sets."""

import argparse
import json
import pathlib

from tinylm import logs
from tinylm.atomic import durable
from tinylm.bench import Bench
from tinylm.experiment import Experiment

EVALS = pathlib.Path("data/evals/evaluation_data/full_eval")
FOLDERS = ("blimp_filtered", "supplement_filtered")


def main() -> None:
    """Scores the full minimal-pair sets and writes intervals.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=pathlib.Path, help="a finished run folder")
    args = parser.parse_args()
    logs.setup()
    experiment = Experiment.from_run(args.run)
    net, tok = experiment.restore()
    bench = Bench(net, tok, "cpu")
    found: dict[str, dict[str, float]] = {}
    for folder in FOLDERS:
        report = bench.evaluate(EVALS / folder)
        found[folder] = {
            "accuracy": 100.0 * report.interval.estimate,
            "low": 100.0 * report.interval.low,
            "high": 100.0 * report.interval.high,
        }
        outcomes = json.dumps(report.outcomes)
        durable(
            args.run / f"outcomes-{folder}.json",
            lambda handle, text=outcomes: handle.write(text.encode()),
        )
    text = json.dumps(found, indent=2) + "\n"
    durable(args.run / "intervals.json", lambda h: h.write(text.encode()))
    for folder, score in found.items():
        print(
            f"{folder}: {score['accuracy']:.2f} "
            f"[{score['low']:.2f}, {score['high']:.2f}]"
        )


if __name__ == "__main__":
    main()
