"""Optimizer construction: AdamW, or Muon for hidden matrices plus AdamW."""

from typing import Any, cast

import torch

from tinylm import model

NAMES = ("adamw", "muon")
MATRIX_NDIM = 2


class Optimizer:
    """Runs several torch optimizers as one.

    Exposes the subset of the torch optimizer interface the trainer uses:
    param_groups, zero_grad, step, state_dict and load_state_dict.
    """

    def __init__(self, parts: list[torch.optim.Optimizer]) -> None:
        self.parts = parts

    @property
    def param_groups(self) -> list[dict[str, float]]:
        """All parameter groups of all parts."""
        return [g for part in self.parts for g in part.param_groups]

    def zero_grad(self) -> None:
        """Clears gradients of every part."""
        for part in self.parts:
            part.zero_grad()

    def step(self) -> None:
        """Steps every part."""
        for part in self.parts:
            part.step()

    def state_dict(self) -> dict[str, object]:
        """Returns the states of all parts."""
        return {"parts": [part.state_dict() for part in self.parts]}

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restores the states of all parts."""
        saved = cast("list[dict[str, Any]]", state["parts"])
        for part, part_state in zip(self.parts, saved, strict=True):
            part.load_state_dict(part_state)


def partition(
    net: model.TinyLM,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Splits parameters into (hidden matrices, everything else).

    Hidden matrices are 2-D weights inside transformer blocks. Embeddings,
    the head, codebooks and norm gains go to the second list.
    """
    matrices: list[torch.nn.Parameter] = []
    others: list[torch.nn.Parameter] = []
    for name, param in net.named_parameters():
        hidden = param.ndim == MATRIX_NDIM and name.startswith("blocks.")
        (matrices if hidden else others).append(param)
    return matrices, others


def build(
    net: model.TinyLM, name: str, lr: float, weight_decay: float
) -> Optimizer:
    """Builds the named optimizer.

    Muon updates hidden matrices (learning rate scaled to match AdamW's
    update RMS, so one schedule drives both); AdamW updates the rest.

    Raises:
        ValueError: If name is unknown, lr is not positive, or
            weight_decay is negative.
    """
    if name not in NAMES:
        raise ValueError(f"optimizer must be one of {NAMES}, got {name!r}")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    if name == "adamw":
        return Optimizer(
            [
                torch.optim.AdamW(
                    net.parameters(), lr=lr, weight_decay=weight_decay
                )
            ]
        )
    matrices, others = partition(net)
    return Optimizer(
        [
            torch.optim.Muon(
                matrices,
                lr=lr,
                weight_decay=weight_decay,
                adjust_lr_fn="match_rms_adamw",
            ),
            torch.optim.AdamW(others, lr=lr, weight_decay=0.0),
        ]
    )
