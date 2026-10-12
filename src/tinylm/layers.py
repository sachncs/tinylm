"""Layer-level measurements: attention range, FFN use and norm behaviour."""

import dataclasses
import math

import torch
from torch import nn
from torch.nn import functional
from torch.utils.hooks import RemovableHandle

from tinylm import model

TOKEN_NDIM = 2
SATURATION = 2.5


@dataclasses.dataclass(frozen=True, slots=True)
class Mass:
    """Attention probability landing at least `window` tokens back.

    Attributes:
        window: Distance threshold in tokens.
        layers: Mean mass beyond the window for each block.
    """

    window: int
    layers: tuple[float, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class Ffn:
    """Feed-forward activation statistics for one block.

    Attributes:
        inactive: Share of (token, unit) activations exactly zero.
        dead: Share of units that are zero for every token.
        negative: Share of total |activation| carried by negative values.
    """

    inactive: float
    dead: float
    negative: float


@dataclasses.dataclass(frozen=True, slots=True)
class Norms:
    """Residual-stream and normalisation behaviour.

    Attributes:
        residual: RMS of the residual stream after each block.
        alpha: Learned alpha of every Dynamic Tanh layer.
        saturation: Share of inputs with |alpha * x| above 2.5, per layer.
        gradient: Gradient norm of each block (empty unless requested).
    """

    residual: tuple[float, ...]
    alpha: tuple[float, ...]
    saturation: tuple[float, ...]
    gradient: tuple[float, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class Layers:
    """Measures what a trained (or fresh) model does inside its layers.

    Attributes:
        net: Model to inspect; it is put in eval mode.
    """

    net: model.TinyLM

    def check(self, tokens: torch.Tensor) -> None:
        """Raises ValueError unless tokens has shape (batch, length)."""
        if tokens.ndim != TOKEN_NDIM:
            raise ValueError(
                f"expected shape (batch, length), got {tuple(tokens.shape)}"
            )

    def blocks(self) -> list[model.Block]:
        """The model's transformer blocks."""
        return [b for b in self.net.blocks if isinstance(b, model.Block)]

    def attention(
        self, tokens: torch.Tensor, windows: tuple[int, ...]
    ) -> tuple[Mass, ...]:
        """Share of attention probability beyond each window, per layer.

        Raises:
            ValueError: If a window is not positive or tokens is not 2-D.
        """
        self.check(tokens)
        if any(w <= 0 for w in windows):
            raise ValueError("every window must be positive")
        captured: list[torch.Tensor] = []
        handles = [
            block.attn.register_forward_pre_hook(
                lambda _m, args: captured.append(args[0].detach())
            )
            for block in self.blocks()
        ]
        with torch.no_grad():
            self.net.eval()
            self.net(tokens)
        for handle in handles:
            handle.remove()
        length = tokens.shape[1]
        positions = torch.arange(length)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)
        found: dict[int, list[float]] = {w: [] for w in windows}
        for index, (block, hidden) in enumerate(
            zip(self.blocks(), captured, strict=True)
        ):
            mask = model.visibility(
                length,
                self.net.config.window,
                True,
                is_global=model.is_global(index, self.net.config.global_every),
            )
            batch, _, width = hidden.shape
            with torch.no_grad():
                q, k, _ = block.attn.qkv(hidden).chunk(3, dim=-1)
            shape = (
                batch,
                length,
                block.attn.n_heads,
                width // block.attn.n_heads,
            )
            q, k = (t.view(shape).transpose(1, 2) for t in (q, k))
            scores = q @ k.transpose(-2, -1) * (q.shape[-1] ** -0.5)
            penalty = block.attn.penalty(length, scores.device)
            if penalty is not None:
                scores = scores - penalty.detach()
            probs = functional.softmax(
                scores.masked_fill(~mask, float("-inf")), dim=-1
            )
            for window in windows:
                beyond = (distance >= window) & mask
                found[window].append(float((probs * beyond).sum(-1).mean()))
        return tuple(Mass(w, tuple(found[w])) for w in windows)

    def ffn(self, tokens: torch.Tensor) -> tuple[Ffn, ...]:
        """Activation sparsity, dead units and negative-region use per block.

        Raises:
            ValueError: If tokens is not 2-D.
        """
        self.check(tokens)
        captured: list[torch.Tensor] = []
        handles = [
            block.act.register_forward_hook(
                lambda _m, _a, out: captured.append(out.detach())
            )
            for block in self.blocks()
        ]
        with torch.no_grad():
            self.net.eval()
            self.net(tokens)
        for handle in handles:
            handle.remove()
        stats: list[Ffn] = []
        for act in captured:
            flat = act.reshape(-1, act.shape[-1])
            zero = flat == 0.0
            total = flat.abs().sum()
            negative = flat.clamp(max=0).abs().sum()
            stats.append(
                Ffn(
                    inactive=float(zero.float().mean()),
                    dead=float(zero.all(dim=0).float().mean()),
                    negative=float(negative / total) if total > 0 else 0.0,
                )
            )
        return tuple(stats)

    def norms(self, tokens: torch.Tensor, gradients: bool = False) -> Norms:
        """Residual-stream size, Dynamic Tanh saturation and gradient norms.

        Raises:
            ValueError: If tokens is not 2-D.
        """
        self.check(tokens)
        residual: list[torch.Tensor] = []
        seen: list[tuple[model.DynamicTanh, torch.Tensor]] = []
        handles: list[RemovableHandle] = [
            block.register_forward_hook(
                lambda _m, _a, out: residual.append(out.detach())
            )
            for block in self.blocks()
        ]

        def record(module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            assert isinstance(module, model.DynamicTanh)
            seen.append((module, args[0].detach()))

        handles.extend(
            module.register_forward_pre_hook(record)
            for module in self.net.modules()
            if isinstance(module, model.DynamicTanh)
        )
        with torch.no_grad():
            self.net.eval()
            self.net(tokens)
        for handle in handles:
            handle.remove()
        norms: list[float] = []
        if gradients:
            self.net.zero_grad()
            logits = self.net(tokens[:, :-1])
            functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1)
            ).backward()
            norms = [
                math.sqrt(
                    sum(
                        float(p.grad.pow(2).sum())
                        for p in block.parameters()
                        if p.grad is not None
                    )
                )
                for block in self.blocks()
            ]
            self.net.zero_grad()
        return Norms(
            residual=tuple(float(r.pow(2).mean().sqrt()) for r in residual),
            alpha=tuple(float(m.alpha.detach()) for m, _ in seen),
            saturation=tuple(
                float(((m.alpha * x).abs() > SATURATION).float().mean())
                for m, x in seen
            ),
            gradient=tuple(norms),
        )

    def parts(self) -> dict[str, int]:
        """Parameters per top-level module; shared weights are counted once."""
        counted: set[int] = set()
        counts: dict[str, int] = {}
        for name, child in self.net.named_children():
            total = 0
            for param in child.parameters():
                if id(param) not in counted:
                    counted.add(id(param))
                    total += param.numel()
            counts[name] = total
        return counts
