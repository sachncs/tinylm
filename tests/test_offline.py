"""Tests proving the pipeline can run with the network disabled."""

import os
import pathlib
import socket

import pytest

from tinylm.clean import Options
from tinylm.offline import Error, Offline
from tinylm.prepare import Prepare
from tinylm.tokens import Token


def test_connections_are_blocked_inside_and_restored_after():
    with Offline():
        with pytest.raises(Error, match="network"):
            socket.create_connection(("example.com", 80), timeout=1)
        with pytest.raises(Error):
            socket.socket().connect(("127.0.0.1", 9))
    original = socket.socket.connect
    with Offline():
        pass
    assert socket.socket.connect is original


def test_offline_flags_are_set_and_restored(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    with Offline():
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"
    assert "HF_HUB_OFFLINE" not in os.environ
    assert os.environ["TRANSFORMERS_OFFLINE"] == "0"


def test_restored_even_when_the_body_raises():
    original = socket.socket.connect
    with pytest.raises(RuntimeError), Offline():
        raise RuntimeError("boom")
    assert socket.socket.connect is original


@pytest.mark.e2e
def test_prepare_and_load_run_with_zero_network(tmp_path: pathlib.Path):
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / "childes.train.txt").write_text(
        "".join(f"*MOT:\tmore {i % 7} cookies please?\n" for i in range(600))
    )
    with Offline():
        Prepare(
            vocab=300, kind="bpe", block=20, every=10, options=Options()
        ).run(corpus, tmp_path / "out")
        assert len(Token.load(tmp_path / "out" / "train", verify=True)) > 0
