"""Tests that exported models reload through transformers unchanged."""

import pathlib
import typing
from typing import Any

import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM

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
        {"recency": True, "exclusive": (1, 2), "cross": (0, 3), "slots": 6},
    ],
    ids=["default", "relu-dyt", "untied", "dropout", "codebooks", "xsa"],
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


def load_base(path: pathlib.Path) -> Any:
    """Loads the exported AutoModel view (its stubs are partly unknown)."""
    loader = typing.cast("Any", AutoModel)
    return loader.from_pretrained(path, trust_remote_code=True).eval()


def test_base_model_returns_the_state_at_the_last_real_token(
    tmp_path: pathlib.Path,
):
    original = net(recency=True, exclusive=(1, 2), cross=(0,))
    Export(original, tok()).write(tmp_path / "out")
    base = load_base(tmp_path / "out")
    assert base.config.hidden_size == 32
    long = torch.randint(3, original.config.vocab_size, (1, 12))
    short = torch.randint(3, original.config.vocab_size, (1, 7))
    padded = torch.zeros(2, 12, dtype=torch.long)
    padded[0] = long[0]
    padded[1, :7] = short[0]
    mask = torch.zeros(2, 12, dtype=torch.long)
    mask[0] = 1
    mask[1, :7] = 1
    left = torch.zeros(2, 12, dtype=torch.long)
    left[0] = long[0]
    left[1, 5:] = short[0]
    left_mask = torch.zeros(2, 12, dtype=torch.long)
    left_mask[0] = 1
    left_mask[1, 5:] = 1
    with torch.no_grad():
        out = base(input_ids=padded, attention_mask=mask).last_hidden_state
        assert out.shape == (2, 1, 32)
        torch.testing.assert_close(out[0, 0], original.hidden(long)[0, -1])
        torch.testing.assert_close(out[1, 0], original.hidden(short)[0, -1])
        mirrored = base(input_ids=left, attention_mask=left_mask)
        torch.testing.assert_close(mirrored.last_hidden_state, out)
        unmasked = base(input_ids=long).last_hidden_state
        torch.testing.assert_close(unmasked[0, 0], original.hidden(long)[0, -1])


def test_base_model_rejects_malformed_masks(tmp_path: pathlib.Path):
    Export(net(), tok()).write(tmp_path / "out")
    base = load_base(tmp_path / "out")
    tokens = torch.randint(3, 40, (1, 6))
    with pytest.raises(ValueError, match="2-D"):
        base(input_ids=tokens, attention_mask=torch.ones(1, 6, 6))
    with pytest.raises(ValueError, match="no tokens"):
        base(
            input_ids=tokens, attention_mask=torch.zeros(1, 6, dtype=torch.long)
        )
