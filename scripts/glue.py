"""Fine-tunes a finished run on the GLUE tasks with the official pipeline."""

import argparse
import json
import pathlib

from tinylm import logs
from tinylm.atomic import durable
from tinylm.finetune import TASKS, Finetune

ROOT = pathlib.Path("external/babylm-eval/strict")
DATA = pathlib.Path("data/glue/evaluation_data/full_eval/glue_filtered")


def main() -> None:
    """Runs every task not yet done and prints the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=pathlib.Path, help="a finished run folder")
    args = parser.parse_args()
    logs.setup()
    export = (args.run / "export").resolve()
    if not export.is_dir():
        raise SystemExit(f"{export} missing: run scripts/official.py first")
    finetune = Finetune(ROOT.resolve(), DATA.resolve())
    out = (args.run / "glue").resolve()
    results: dict[str, dict[str, float]] = {}
    for task in TASKS:
        done = out / f"{task.name}.json"
        if done.exists():
            results[task.name] = json.loads(done.read_text())
            continue
        scores = finetune.evaluate(export, task, out / "raw")
        scores["majority"] = finetune.majority(task)
        text = json.dumps(scores, indent=2) + "\n"
        durable(done, lambda h, t=text: h.write(t.encode()))
        results[task.name] = scores
    for name, scores in results.items():
        shown = ", ".join(f"{k} {v:.4f}" for k, v in scores.items())
        print(f"{name}: {shown}")


if __name__ == "__main__":
    main()
