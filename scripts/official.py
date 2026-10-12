"""Exports a finished run and scores it with the official pipeline."""

import argparse
import dataclasses
import json
import pathlib

from tinylm import logs
from tinylm.atomic import durable
from tinylm.experiment import Experiment
from tinylm.export import Export
from tinylm.official import Official

ROOT = pathlib.Path("external/babylm-eval/strict")
EVALS = pathlib.Path("data/evals/evaluation_data/full_eval")
TASKS = (
    ("blimp_filtered", "blimp"),
    ("supplement_filtered", "blimp"),
    ("entity_tracking", "entity_tracking"),
)


def main() -> None:
    """Runs every official task on the exported final model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=pathlib.Path)
    parser.add_argument("--name")
    parser.add_argument("--set", action="append", default=[], dest="sets")
    args = parser.parse_args()
    logs.setup()
    experiment = Experiment.load(args.config)
    if args.sets or args.name:
        experiment = experiment.override(
            args.sets, args.name or experiment.name
        )
    net, tok = experiment.restore()
    target = experiment.out / "export"
    if not target.exists():
        Export(net, tok).write(target)
    official = Official(ROOT)
    scores = {
        folder: dataclasses.asdict(
            official.evaluate(
                target.resolve(),
                task,
                (EVALS / folder).resolve(),
                (experiment.out / "official" / folder).resolve(),
            )
        )
        for folder, task in TASKS
    }
    text = json.dumps(scores, indent=2) + "\n"
    durable(experiment.out / "official.json", lambda h: h.write(text.encode()))
    for folder, score in scores.items():
        print(f"{folder}: {score['average']:.2f}")


if __name__ == "__main__":
    main()
