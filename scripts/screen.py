"""Runs and analyses the 24-run, 20-factor screen on Strict-Small."""

import argparse
import json
import pathlib
import subprocess
import sys

from tinylm.experiment import Experiment
from tinylm.screen import Design, Factor, Screen

BASE = pathlib.Path("configs/screen.toml")
RUNS = pathlib.Path("runs/screen")
DUMMIES = 3

FACTORS = (
    Factor("width", ("model.d_model=128",), ("model.d_model=192",)),
    Factor("depth", ("model.n_layers=4",), ("model.n_layers=8",)),
    Factor(
        "activation", ("model.activation=gelu",), ("model.activation=relu",)
    ),
    Factor("norm", ("model.norm=rms",), ("model.norm=dyt",)),
    Factor(
        "attention",
        ("model.global_every=1",),
        ("model.global_every=4", "model.window=64"),
    ),
    Factor(
        "untied",
        ("model.tie_embeddings=true",),
        ("model.tie_embeddings=false",),
    ),
    Factor("dropout", ("model.dropout=0.0",), ("model.dropout=0.1",)),
    Factor("lr", ("train.lr=0.001",), ("train.lr=0.004",)),
    Factor("weight_decay", ("train.decay=0.01",), ("train.decay=0.3",)),
    Factor("warmup", ("train.warmup=100",), ("train.warmup=500",)),
    Factor("batch", ("train.batch=16",), ("train.batch=64",)),
    Factor("muon", ("train.optimizer=adamw",), ("train.optimizer=muon",)),
    Factor("wsd", ("train.schedule=cosine",), ("train.schedule=wsd",)),
    Factor("hybrid", ("train.masked=0.0",), ("train.masked=0.5",)),
    Factor("clip", ("train.clip=0.5",), ("train.clip=2.0",)),
    Factor("floor", ("train.floor=0.05",), ("train.floor=0.3",)),
    Factor("seq_len", ("train.seq_len=128",), ("train.seq_len=256",)),
    Factor("shift", ("train.shift=false",), ("train.shift=true",)),
    Factor("ema", ("train.ema=0.0",), ("train.ema=0.99",)),
    Factor("vocab", ("data=runs/small4k",), ("data=runs/small16k",)),
)


def screen() -> Screen:
    """The screen used for this study."""
    return Screen(Design.pb24(), FACTORS, DUMMIES)


def run() -> None:
    """Runs every screen run without metrics.json, each in its own process.

    Separate processes isolate memory use and crashes; a failing run stops
    the whole batch with its exit status (check=True), never silently.
    """
    Experiment.load(BASE)
    for item in screen().runs():
        if (RUNS / item.name / "metrics.json").exists():
            continue
        print(f"== {item.name} {len(item.sets)} overrides", flush=True)
        command = [
            sys.executable,
            "scripts/run.py",
            str(BASE),
            "--name",
            item.name,
        ]
        for override in item.sets:
            command += ["--set", override]
        subprocess.run(command, check=True)


def analyse() -> None:
    """Prints main effects for each response, strongest first."""
    runs = screen().runs()
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
        result = screen().analyse(values)
        print(f"\n{title}  noise(dummy rms)={result.noise:.5f}")
        for e in sorted(result.effects, key=lambda e: -abs(e.effect)):
            flag = "SIGNIFICANT" if e.significant else "-"
            line = f"  {e.name:13s} effect={e.effect:+.5f}"
            print(f"{line} ratio={e.ratio:6.2f} {flag}")


def main() -> None:
    """Dispatches the plan, run and analyse commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "analyse"))
    args = parser.parse_args()
    if args.command == "plan":
        for item in screen().runs():
            print(item.name, " ".join(item.sets))
    elif args.command == "run":
        run()
    else:
        analyse()


if __name__ == "__main__":
    main()
