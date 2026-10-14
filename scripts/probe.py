"""Runs the layer diagnostics and the recall and superposition probes."""

import argparse
import dataclasses
import json
import pathlib

import numpy
import torch

from tinylm import logs, tokenizer
from tinylm.atomic import durable
from tinylm.experiment import Experiment
from tinylm.layers import Layers
from tinylm.probes import Associative, Memory, Sae, Superposition
from tinylm.tokens import Token

BOS = tokenizer.SPECIAL_TOKENS.index("<bos>")
LINES = 300
MIN_IDS = 12
MAX_IDS = 120
ROWS = 16
DISTANCES = (8, 32, 128, 200)
TRIALS = 500
WINDOWS = (16, 64, 128)


def segments(store: Token, count: int) -> list[list[int]]:
    """Evenly spread lines (between BOS markers) of usable length."""
    ids = numpy.asarray(store.ids)
    marks = numpy.flatnonzero(ids == BOS)
    found: list[list[int]] = []
    stride = max(1, len(marks) // (count * 4))
    for first, last in zip(marks[::stride], marks[1::stride], strict=False):
        line = ids[first + 1 : last].tolist()
        if MIN_IDS <= len(line) <= MAX_IDS:
            found.append(line)
        if len(found) == count:
            break
    return found


def main() -> None:
    """Probes a finished run and writes probes.json next to it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=pathlib.Path, help="a finished run folder")
    args = parser.parse_args()
    logs.setup()
    experiment = Experiment.from_run(args.run)
    net, tok = experiment.restore()
    device = "cpu"
    net.to(device).eval()
    train = Token.load(experiment.data / "train")
    dev = Token.load(experiment.data / "dev")
    memory = Memory(net, tok, device)
    recalled = {}
    for name, store in (("train", train), ("dev", dev)):
        lines = [tok.decode(line) for line in segments(store, LINES)]
        recalled[name] = dataclasses.asdict(memory.recall(lines))
    associative = [
        dataclasses.asdict(hit)
        for hit in Associative(net, device).accuracy(DISTANCES, TRIALS, 0)
    ]
    length = experiment.model.max_seq_len
    starts = torch.arange(ROWS) * length
    rows = dev.rows(starts, length).long()
    layers = Layers(net)
    mass = [dataclasses.asdict(m) for m in layers.attention(rows, WINDOWS)]
    ffn = [dataclasses.asdict(f) for f in layers.ffn(rows)]
    norms = dataclasses.asdict(layers.norms(rows))
    probe = Superposition(net)
    crowd = {}
    width = net.config.d_model
    embedding = net.tok_emb.weight.detach()
    crowd["embedding"] = {
        "rank": probe.rank(embedding),
        "interference": probe.interference(embedding),
    }
    for layer in range(net.config.n_layers):
        states = probe.hidden(rows, layer)
        crowd[f"block{layer}"] = {
            "rank": probe.rank(states),
            "interference": probe.interference(states),
        }
    middle = probe.hidden(rows, net.config.n_layers // 2 - 1)
    found = Sae(width, 4 * width, 300, 1e-3, 0).fit(middle)
    report = {
        "recall": recalled,
        "associative": associative,
        "attention_mass_beyond_window": mass,
        "ffn": ffn,
        "norms": norms,
        "superposition": crowd,
        "sparse_autoencoder_middle": dataclasses.asdict(found),
    }
    text = json.dumps(report, indent=2) + "\n"
    durable(
        args.run / "probes.json", lambda handle: handle.write(text.encode())
    )
    print(text)


if __name__ == "__main__":
    main()
