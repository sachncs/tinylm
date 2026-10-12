"""Exponential moving average of model weights."""

import collections.abc
import contextlib

import torch

from tinylm import model


class Ema:
    """Tracks an exponential moving average of a model's state.

    Attributes:
        decay: Weight of the previous average in each update, in [0, 1].
        shadow: Averaged copy of the model's state dict.
    """

    def __init__(self, net: model.TinyLM, decay: float) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        self.decay = decay
        self.shadow = {
            key: value.detach().clone()
            for key, value in net.state_dict().items()
        }

    def update(self, net: model.TinyLM) -> None:
        """Moves the average one step towards the model's current weights."""
        with torch.no_grad():
            for key, value in net.state_dict().items():
                if value.is_floating_point():
                    self.shadow[key].mul_(self.decay).add_(
                        value.detach(), alpha=1.0 - self.decay
                    )
                else:
                    self.shadow[key].copy_(value)

    @contextlib.contextmanager
    def applied(self, net: model.TinyLM) -> collections.abc.Generator[None]:
        """Temporarily loads the averaged weights into net."""
        live = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(self.shadow)
        try:
            yield
        finally:
            net.load_state_dict(live)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Returns the averaged weights."""
        return dict(self.shadow)

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restores averaged weights.

        Raises:
            ValueError: If the keys do not match the tracked model.
        """
        if set(state) != set(self.shadow):
            raise ValueError("state keys do not match the tracked model")
        self.shadow = {key: value.clone() for key, value in state.items()}
