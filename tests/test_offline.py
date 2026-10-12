"""Tests proving the pipeline can run with the network disabled."""

import os
import pathlib
import socket
import threading

import pytest

from tinylm import offline
from tinylm.clean import Options
from tinylm.offline import Error, Offline
from tinylm.prepare import Prepare
from tinylm.tokens import Token


def test_connections_are_blocked_inside_and_restored_after():
    with Offline():
        with pytest.raises(Error, match="network"):
            socket.create_connection(("example.com", 80), timeout=1)
        with pytest.raises(Error), socket.socket() as sock:
            sock.connect(("127.0.0.1", 9))
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


def test_offline_is_not_reentrant_and_not_shareable_across_threads():
    outcome: list[str] = []

    def other_thread() -> None:
        try:
            with Offline():
                outcome.append("entered")
        except Error as error:
            outcome.append(str(error))

    with Offline():
        with pytest.raises(Error, match="already active"), Offline():
            pass
        worker = threading.Thread(target=other_thread)
        worker.start()
        worker.join()
    assert outcome and "already active" in outcome[0]
    with Offline():  # the lock was released, so it can be used again
        pass


def test_the_lock_is_released_when_the_body_raises():
    with pytest.raises(RuntimeError), Offline():
        raise RuntimeError("boom")
    with Offline():
        pass


def test_a_failure_while_entering_releases_the_lock(
    monkeypatch: pytest.MonkeyPatch,
):
    def broken(*_args: object, **_kwargs: object) -> None:
        raise OSError("cannot patch")

    monkeypatch.setattr(
        offline.mock, "patch", type("P", (), {"object": staticmethod(broken)})
    )
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(OSError, match="cannot patch"), Offline():
        pass
    assert "HF_HUB_OFFLINE" not in os.environ
    monkeypatch.undo()
    with Offline():
        pass
