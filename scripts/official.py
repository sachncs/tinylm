"""Exports a finished run and scores it with the official pipeline."""

import argparse
import dataclasses
import json
import pathlib

from tinylm import logs
from tinylm.atomic import durable
from tinylm.bench import Bench
from tinylm.experiment import Experiment
from tinylm.export import Export
from tinylm.official import Official

ROOT = pathlib.Path("external/babylm-eval/strict")
EVALS = pathlib.Path("data/evals/evaluation_data/full_eval")
TASKS = (
    ("blimp_filtered", "blimp"),
    ("supplement_filtered", "blimp"),
    ("entity_tracking", "entity_tracking"),
    ("comps", "comps"),
)


def main() -> None:
    """Runs every official task on the exported final model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=pathlib.Path, help="a finished run folder")
    args = parser.parse_args()
    logs.setup()
    experiment = Experiment.from_run(args.run)
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
    bench = Bench(net, tok, "cpu")
    for folder, task in TASKS:
        if task == "blimp":
            ours = 100.0 * bench.evaluate(EVALS / folder).mean
            scores[folder]["in_house"] = ours
            scores[folder]["gap"] = ours - scores[folder]["average"]
    text = json.dumps(scores, indent=2) + "\n"
    durable(experiment.out / "official.json", lambda h: h.write(text.encode()))
    for folder, score in scores.items():
        extra = (
            f" (in-house {score['in_house']:.2f}, gap {score['gap']:+.2f})"
            if "in_house" in score
            else ""
        )
        print(f"{folder}: {score['average']:.2f}{extra}")


if __name__ == "__main__":
    main()
