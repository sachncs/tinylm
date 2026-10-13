"""Decoder-only transformer with swappable parts and a parameter budget."""

import dataclasses
import functools

import torch
from torch import nn
from torch.nn import functional
from torch.nn.functional import scaled_dot_product_attention

PARAMETER_BUDGET = 50_000_000
FFN_MULTIPLIER = 4
TOKEN_NDIM = 2
INIT_STD = 0.02
DYT_ALPHA_INIT = 0.5
ACTIVATIONS = ("gelu", "relu")
NORMS = ("rms", "dyt")
NON_POSITIVE_FIELDS = (
    "dropout",
    "tie_embeddings",
    "activation",
    "norm",
    "codebook_sizes",
    "recency",
    "exclusive",
    "cross",
)


@dataclasses.dataclass(frozen=True)
class Config:
    """Hyperparameters of TinyLM.

    Attributes:
        vocab_size: Number of distinct token ids.
        max_seq_len: Longest supported input sequence.
        d_model: Width of the residual stream.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads; must divide d_model.
        window: Local attention reach: a query sees keys whose distance is
            less than window.
        global_every: Every global_every-th layer uses full attention; the
            remaining layers use the sliding window.
        dropout: Probability of zeroing activations on the embedding output
            and on both residual writes; 0 disables it.
        tie_embeddings: Share the input embedding with the output head.
        activation: "gelu" or "relu" in the feed-forward network.
        norm: "rms" (RMSNorm) or "dyt" (Dynamic Tanh, a norm-free layer).
        codebook_sizes: Sizes of learned codebooks placed between consecutive
            stages of blocks, e.g. (32768, 16384); empty disables them.
        recency: Add a learned per-head penalty slope * distance to attention
            scores. Slopes start at 0 (plain attention) and cannot go below 0,
            so each head learns how local to be.
        exclusive: Indices of layers using exclusive self attention (Zhai,
            2026, arXiv:2603.09078): each head's output has its component
            along the token's own value vector removed.
        cross: Indices of layers that also attend to learned memory slots
            (cross attention whose keys and values are parameters).
        slots: Number of memory slots per cross-attention layer.
    """

    vocab_size: int = 16_384
    max_seq_len: int = 512
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    window: int = 128
    global_every: int = 4
    dropout: float = 0.0
    tie_embeddings: bool = True
    activation: str = "gelu"
    norm: str = "rms"
    codebook_sizes: tuple[int, ...] = ()
    recency: bool = False
    exclusive: tuple[int, ...] = ()
    cross: tuple[int, ...] = ()
    slots: int = 32

    def __post_init__(self) -> None:
        """Validates field ranges and head divisibility."""
        for field in dataclasses.fields(self):
            if (
                field.name not in NON_POSITIVE_FIELDS
                and getattr(self, field.name) <= 0
            ):
                raise ValueError(f"{field.name} must be positive")
        object.__setattr__(self, "codebook_sizes", tuple(self.codebook_sizes))
        if any(size <= 0 for size in self.codebook_sizes):
            raise ValueError("codebook_sizes must all be positive")
        if self.codebook_sizes and self.n_layers <= len(self.codebook_sizes):
            raise ValueError("n_layers must exceed the number of codebooks")
        for name in ("exclusive", "cross"):
            object.__setattr__(
                self, name, layer_list(name, getattr(self, name), self.n_layers)
            )
        if self.recency not in (True, False):
            raise ValueError("recency must be a bool")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        if self.norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")


def layer_list(
    name: str, values: tuple[int, ...], count: int
) -> tuple[int, ...]:
    """Returns sorted distinct layer indices.

    Raises:
        ValueError: If any index is not an int in [0, count) or repeats.
    """
    layers = tuple(values)
    if not all(type(i) is int and 0 <= i < count for i in layers) or len(
        set(layers)
    ) != len(layers):
        raise ValueError(
            f"{name} must list distinct layer indices in [0, n_layers)"
        )
    return tuple(sorted(layers))


def visibility(
    length: int, window: int, causal: bool, is_global: bool
) -> torch.Tensor:
    """Builds a boolean attention mask where True means the key is visible.

    Args:
        length: Sequence length.
        window: Local reach; keys at distance >= window are hidden unless
            is_global.
        causal: If True, queries see only keys at or before their position.
        is_global: If True, ignore the window.

    Returns:
        Bool tensor of shape (length, length), indexed [query, key].

    Raises:
        ValueError: If length or window is not positive.
    """
    if length <= 0 or window <= 0:
        raise ValueError("length and window must be positive")
    positions = torch.arange(length)
    distance = positions.unsqueeze(1) - positions.unsqueeze(0)
    visible = (
        distance >= 0 if causal else torch.ones_like(distance, dtype=torch.bool)
    )
    if not is_global:
        visible = visible & (distance.abs() < window)
    return visible


@functools.lru_cache(maxsize=64)
def local_mask(
    length: int, window: int, causal: bool, device: str
) -> torch.Tensor:
    """Cached sliding-window mask already placed on the device.

    Building the mask on the CPU and copying it every forward pass wasted
    time on every training step; the cache key is exactly what determines
    its contents.
    """
    return visibility(length, window, causal, is_global=False).to(device)


def is_global(index: int, global_every: int) -> bool:
    """Returns True if the layer at index uses full attention."""
    if global_every <= 0:
        raise ValueError("global_every must be positive")
    return (index + 1) % global_every == 0


class DynamicTanh(nn.Module):
    """Norm-free layer: gain * tanh(alpha * x) + bias with learned alpha."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(DYT_ALPHA_INIT))
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Squashes x elementwise; no statistics are computed."""
        return self.weight * torch.tanh(self.alpha * x) + self.bias


def norm(config: Config) -> nn.RMSNorm | DynamicTanh:
    """Builds the configured normalisation layer."""
    if config.norm == "dyt":
        return DynamicTanh(config.d_model)
    return nn.RMSNorm(config.d_model)


def init(module: nn.Module) -> None:
    """GPT-style initialisation: N(0, 0.02) for linear and embedding weights."""
    if isinstance(module, nn.Linear | nn.Embedding):
        nn.init.normal_(module.weight, std=INIT_STD)


class Codebook(nn.Module):
    """Learned codebook read by soft attention; coarsens a hidden state.

    Output is x + softmax(norm(x) . codes) @ codes, so each token state is
    nudged towards a mixture of a small set of learned prototypes. The
    operation is token-wise and therefore preserves causality.
    """

    def __init__(self, size: int, width: int, config: Config) -> None:
        super().__init__()
        self.codes = nn.Parameter(torch.randn(size, width) * INIT_STD)
        self.norm = norm(config)
        self.scale = width**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds a softmax-weighted mixture of codes to every position."""
        scores = self.norm(x) @ self.codes.T * self.scale
        return x + torch.softmax(scores, dim=-1) @ self.codes


def exclude(out: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Removes from out its component along v (over the last dimension).

    A zero v has no direction, so out is returned unchanged.
    """
    direction = functional.normalize(v, dim=-1)
    return out - (out * direction).sum(-1, keepdim=True) * direction


class Attention(nn.Module):
    """Multi-head self-attention under an explicit visibility mask."""

    def __init__(self, config: Config, exclusive: bool = False) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.exclusive = exclusive
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.slope = (
            nn.Parameter(torch.zeros(config.n_heads))
            if config.recency
            else None
        )

    def penalty(self, length: int, device: torch.device) -> torch.Tensor | None:
        """Per-head score penalty (heads, length, length), or None if off.

        Negative slopes act as 0, so a head can only learn to prefer nearer
        keys. torch.where (not clamp) is used because clamp has no gradient
        exactly at the boundary, where every slope starts.
        """
        if self.slope is None:
            return None
        positions = torch.arange(length, device=device)
        distance = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()
        slope = torch.where(self.slope >= 0, self.slope, 0.0)
        return slope.view(-1, 1, 1) * distance

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None, causal: bool
    ) -> torch.Tensor:
        """Applies self-attention to (batch, length, d_model).

        mask is an explicit boolean visibility mask, or None for full
        attention (causal when `causal` is set), which lets the fused
        kernel run without a materialised mask. With recency on, the mask
        becomes a float bias that also carries the distance penalty.
        """
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, length, self.n_heads, width // self.n_heads)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        penalty = self.penalty(length, x.device)
        fused = mask is None and causal
        if penalty is not None:
            if mask is None:
                mask = visibility(length, length, causal, is_global=True).to(
                    x.device
                )
            bias = (-penalty).unsqueeze(0).masked_fill(~mask, float("-inf"))
            mask, fused = bias.to(q.dtype), False
        out = scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=fused
        )
        if self.exclusive:
            out = exclude(out, v)
        return self.proj(out.transpose(1, 2).reshape(batch, length, width))


class Cross(nn.Module):
    """Multi-head cross attention from token states to learned memory slots.

    Keys and values are parameters shared by every position, so the block is
    token-wise (it cannot leak information across positions) and adds only
    2 * width^2 + 2 * slots * width parameters.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.keys = nn.Parameter(
            torch.randn(config.slots, config.d_model) * INIT_STD
        )
        self.values = nn.Parameter(
            torch.randn(config.slots, config.d_model) * INIT_STD
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attends from each position of (batch, length, width) to the slots."""
        batch, length, width = x.shape
        heads, head = self.n_heads, width // self.n_heads
        q = self.query(x).view(batch, length, heads, head).transpose(1, 2)
        slots = self.keys.shape[0]
        k = self.keys.view(1, slots, heads, head).transpose(1, 2)
        v = self.values.view(1, slots, heads, head).transpose(1, 2)
        out = scaled_dot_product_attention(
            q, k.expand(batch, -1, -1, -1), v.expand(batch, -1, -1, -1)
        )
        return self.proj(out.transpose(1, 2).reshape(batch, length, width))


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(
        self, config: Config, exclusive: bool = False, cross: bool = False
    ) -> None:
        super().__init__()
        hidden = FFN_MULTIPLIER * config.d_model
        self.attn_norm = norm(config)
        self.attn = Attention(config, exclusive)
        self.cross_norm = norm(config) if cross else None
        self.cross = Cross(config) if cross else None
        self.ffn_norm = norm(config)
        self.act = nn.ReLU() if config.activation == "relu" else nn.GELU()
        self.ffn_in = nn.Linear(config.d_model, hidden, bias=False)
        self.ffn_out = nn.Linear(hidden, config.d_model, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def init_residual(self, std: float) -> None:
        """Re-initialises the two layers that write into the residual stream."""
        nn.init.normal_(self.attn.proj.weight, std=std)
        if self.cross is not None:
            nn.init.normal_(self.cross.proj.weight, std=std)
        nn.init.normal_(self.ffn_out.weight, std=std)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None, causal: bool
    ) -> torch.Tensor:
        """Applies attention and feed-forward sublayers with residuals."""
        x = x + self.drop(self.attn(self.attn_norm(x), mask, causal))
        if self.cross is not None and self.cross_norm is not None:
            x = x + self.drop(self.cross(self.cross_norm(x)))
        hidden = self.act(self.ffn_in(self.ffn_norm(x)))
        return x + self.drop(self.ffn_out(hidden))


class TinyLM(nn.Module):
    """Decoder-only language model."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        blocks = [
            Block(config, i in config.exclusive, i in config.cross)
            for i in range(config.n_layers)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.codebooks = nn.ModuleList(
            Codebook(size, config.d_model, config)
            for size in config.codebook_sizes
        )
        stages = len(config.codebook_sizes) + 1
        self.codebook_after = {
            config.n_layers * (i + 1) // stages - 1: i
            for i in range(len(config.codebook_sizes))
        }
        self.emb_drop = nn.Dropout(config.dropout)
        self.final_norm = norm(config)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(init)
        if config.tie_embeddings:
            self.head.weight = self.tok_emb.weight
        residual_std = INIT_STD / (2 * config.n_layers) ** 0.5
        for block in blocks:
            block.init_residual(residual_std)

    def forward(
        self,
        tokens: torch.Tensor,
        causal: bool = True,
        local_window: int | None = None,
    ) -> torch.Tensor:
        """Computes per-position token logits (see hidden for the arguments)."""
        return self.head(self.hidden(tokens, causal, local_window))

    def hidden(
        self,
        tokens: torch.Tensor,
        causal: bool = True,
        local_window: int | None = None,
    ) -> torch.Tensor:
        """Computes the final-normalised hidden state at every position.

        Args:
            tokens: Integer tensor of shape (batch, length).
            causal: True for next-token prediction, False for bidirectional
                (masked-token) attention.
            local_window: If given, every layer (global ones included) only
                sees keys closer than this; used to measure how much a
                trained model relies on long-range attention.

        Returns:
            Float tensor of shape (batch, length, d_model).

        Raises:
            ValueError: If tokens has the wrong shape, is longer than
                max_seq_len, contains ids outside the vocabulary, or
                local_window is not positive.
        """
        if tokens.ndim != TOKEN_NDIM:
            raise ValueError(
                f"expected shape (batch, length), got {tuple(tokens.shape)}"
            )
        length = tokens.shape[1]
        if length > self.config.max_seq_len:
            raise ValueError(
                f"length {length} exceeds max_seq_len {self.config.max_seq_len}"
            )
        # The range check reads tensor values, which would break the compiled
        # graph; compiled training only sees batches from the validated store.
        if (
            not torch.compiler.is_compiling()
            and tokens.numel()
            and (tokens.min() < 0 or tokens.max() >= self.config.vocab_size)
        ):
            raise ValueError("token id outside vocab range")
        if local_window is not None and local_window <= 0:
            raise ValueError("local_window must be positive")
        positions = torch.arange(length, device=tokens.device)
        x = self.emb_drop(self.tok_emb(tokens) + self.pos_emb(positions))
        window = local_window or self.config.window
        if torch.compiler.is_compiling():
            # Inside a compiled graph the mask is built from traced ops (the
            # cache wrapper cannot be traced); eagerly the cached one is used.
            local = visibility(length, window, causal, is_global=False).to(
                x.device
            )
        else:
            local = local_mask(length, window, causal, str(x.device))
        for index, block in enumerate(self.blocks):
            use_global = is_global(index, self.config.global_every)
            fused = use_global and local_window is None
            x = block(x, None if fused else local, causal)
            if index in self.codebook_after:
                x = self.codebooks[self.codebook_after[index]](x)
        return self.final_norm(x)


def count(net: nn.Module) -> int:
    """Counts distinct trainable parameters, counting tied weights once."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def check(net: nn.Module, budget: int = PARAMETER_BUDGET) -> None:
    """Raises ValueError if the model has more than budget parameters."""
    total = count(net)
    if total > budget:
        raise ValueError(f"{total} parameters exceeds budget of {budget}")
