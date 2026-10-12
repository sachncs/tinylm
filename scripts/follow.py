"""Runs and analyses the 2^5 factorial on the shortlisted factors."""

import argparse
import json
import pathlib

from tinylm import jobs, logs
from tinylm.experiment import Experiment
from tinylm.screen import Factor, Factorial

BASE = pathlib.Path("configs/follow.toml")
RUNS = pathlib.Path("runs/follow")

FACTORS = (
    Factor("norm", ("model.norm=rms",), ("model.norm=dyt",)),
    Factor("muon", ("train.optimizer=adamw",), ("train.optimizer=muon",)),
    Factor("vocab", ("data=runs/small4k",), ("data=runs/small16k",)),
    Factor("seq_len", ("train.seq_len=128",), ("train.seq_len=256",)),
    Factor("lr", ("train.lr=0.001",), ("train.lr=0.004",)),
)


def design() -> Factorial:
    """The factorial used for this study."""
    return Factorial(FACTORS)


def run() -> None:
    """Runs every factorial run without metrics.json, one process each."""
    Experiment.load(BASE)
    jobs.run_all(
        jobs.command(BASE, item.name, item.sets)
        for item in design().runs()
        if not (RUNS / item.name / "metrics.json").exists()
    )


def analyse() -> None:
    """Prints main effects and interactions per response, strongest first."""
    runs = design().runs()
    saved = [
        json.loads((RUNS / r.name / "metrics.json").read_text()) for r in runs
    ]
    responses = {
        "bits_per_char (lower is better)": [m["bits_per_char"] for m in saved],
        "blimp_fast (higher is better)": [
            m["benchmarks"]["blimp_fast"] for m in saved
        ],
        "precision@1.0 (higher is better)": [
            m["confidence"][0]["precision"] or 0.0 for m in saved
        ],
    }
    for title, values in responses.items():
        result = design().analyse(values)
        print(
            f"\n{title}  Lenth pse={result.pse:.5f} margin={result.margin:.5f}"
        )
        for term in sorted(result.terms, key=lambda t: -abs(t.effect))[:10]:
            flag = "SIGNIFICANT" if term.significant else "-"
            print(f"  {term.name:16s} effect={term.effect:+.5f} {flag}")


def main() -> None:
    """Dispatches the plan, run and analyse commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "analyse"))
    args = parser.parse_args()
    logs.setup()
    if args.command == "plan":
        for item in design().runs():
            print(item.name, " ".join(item.sets))
    elif args.command == "run":
        run()
    else:
        analyse()


if __name__ == "__main__":
    main()
