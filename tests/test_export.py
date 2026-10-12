"""Tests that exported models reload through transformers unchanged."""

import pathlib
import typing
from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM

from tinylm import model, seeding, tokenizer
from tinylm.export import Error, Export

pytestmark = pytest.mark.e2e

LINES = [f"more {i % 7} cookies please ?" for i in range(80)]


def tok():
    return tokenizer.train(LINES, vocab_size=300, kind="bpe")


def net(**overrides: Any) -> model.TinyLM:
    seeding.set_seed(0)
    values: dict[str, Any] = {
        "vocab_size": tok().get_vocab_size(),
        "max_seq_len": 32,
        "d_model": 32,
        "n_layers": 4,
        "n_heads": 2,
        "window": 4,
        "global_every": 2,
    }
    values.update(overrides)
    return model.TinyLM(model.Config(**values)).eval()


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"activation": "relu", "norm": "dyt"},
        {"tie_embeddings": False},
        {"dropout": 0.2},
        {"codebook_sizes": (48, 24, 12)},
    ],
    ids=["default", "relu-dyt", "untied", "dropout", "codebooks"],
)
def test_every_option_reloads_with_identical_logits(
    tmp_path: pathlib.Path, overrides: dict[str, Any]
):
    original = net(**overrides)
    Export(original, tok()).write(tmp_path / "out")
    loaded = AutoModelForCausalLM.from_pretrained(
        tmp_path / "out", trust_remote_code=True
    ).eval()
    tokens = torch.randint(0, original.config.vocab_size, (2, 20))
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(input_ids=tokens).logits,
            original(tokens),
            rtol=1e-5,
            atol=1e-5,
        )
    head = typing.cast("torch.nn.Module", loaded.get_output_embeddings())
    emb = loaded.get_input_embeddings()
    assert isinstance(head, torch.nn.Linear)
    assert isinstance(emb, torch.nn.Embedding)
    assert (head.weight is emb.weight) == overrides.get("tie_embeddings", True)


def test_output_contains_the_files_the_official_pipeline_loads(
    tmp_path: pathlib.Path,
):
    Export(net(), tok()).write(tmp_path / "out")
    names = {p.name for p in (tmp_path / "out").iterdir()}
    for expected in (
        "config.json",
        "modeling_tinylm.py",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    ):
        assert expected in names


def test_negative_cases_leave_nothing_behind(tmp_path: pathlib.Path):
    big = net(d_model=1024, n_layers=48, n_heads=8, max_seq_len=64)
    with pytest.raises(ValueError, match="budget"):
        Export(big, tok()).write(tmp_path / "big")
    with pytest.raises(Error, match="vocab"):
        Export(net(vocab_size=tok().get_vocab_size() + 7), tok()).write(
            tmp_path / "v"
        )
    (tmp_path / "taken").mkdir()
    with pytest.raises(FileExistsError):
        Export(net(), tok()).write(tmp_path / "taken")
    assert not (tmp_path / "big").exists() and not (tmp_path / "v").exists()
    assert not list(tmp_path.glob("*.partial"))
