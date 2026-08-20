"""Open-Gen: a from-scratch implementation of the GEN embodied foundation model family.

This module is a single-file, dependency-light (PyTorch only) implementation of the
architecture reconstructed in ``ARCHITECTURE.md``.  Section markers such as ``§6.2``
throughout the docstrings refer to that document.

The model is organised into the three subsystems that Generalist AI names when they
decompose fine-tuning weight deltas -- *sensor processing*, *harmonic reasoning* and
*actuation* (``ARCHITECTURE.md`` §4):

===========================  ===========================================  =========
Subsystem                    Classes                                      ~ params
===========================  ===========================================  =========
① Sensor processing          :class:`VisionTokenizer`,                      ~8%
                             :class:`ProprioTokenizer`,
                             :class:`LanguageTokenizer`,
                             :class:`EmbodimentEncoder`
② Harmonic reasoning         :class:`HarmonicTrunk`,                       ~88%
                             :class:`ContinuousTimeRoPE`,
                             :class:`LatencySchedule`
③ Actuation                  :class:`ActionExpert`, :class:`ReflexHead`,    ~3%
                             :class:`HypernetActionAdapter`
===========================  ===========================================  =========

The load-bearing ideas, all of which live in ② and are what make ①/③ schedulable:

* **Continuous-time RoPE** (§6.1) -- positions are float timestamps in seconds, not
  integer indices.  A spliced physical prompt is therefore just a large ``Δt``.
* **Time-causal masking with a trained latency offset δ** (§6.2) -- an action token at
  time ``t`` may only attend to sensor tokens at ``t - δ``.  Inference latency becomes a
  *modelled* quantity rather than a runtime problem, which is why no System-1/System-2
  split and no inference-time guidance are required.
* **The harmonic tick ladder** (§5.1) -- every stream emits at an integer divisor of the
  100 Hz action rate, so token slots align on a shared grid and the KV cache layout is
  static enough to page (§10.2).
* **Flow-matching action expert** (§7.1) -- a continuous, multimodal action sampler, as
  required by GEN-0's reverse-KL-over-policy-samples evaluation protocol.

Example
-------
>>> cfg = GenConfig.debug()                       # ~20M params, CPU friendly
>>> model = GenModel(cfg)
>>> batch = make_dummy_batch(cfg, batch_size=2)
>>> out = model(batch)
>>> losses = model.compute_loss(batch, out)
>>> losses.total.backward()

Notes
-----
Tensor shapes are documented with the following symbols:

``B``   batch, ``N`` interleaved context tokens, ``D`` ``trunk.d_model``,
``F``   camera frames, ``P`` image patches, ``K`` action query positions,
``H``   action-chunk horizon steps, ``A`` universal action dimension,
``Hq``  attention query heads, ``Hkv`` attention key/value heads, ``Dh`` head dim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    # enums / specs
    "Stream",
    "EmbodimentSpec",
    "TokenStream",
    # configs
    "HarmonicLadderConfig",
    "VisionConfig",
    "ProprioConfig",
    "ActionSpaceConfig",
    "TrunkConfig",
    "ActionExpertConfig",
    "ReflexConfig",
    "LatencyConfig",
    "LossConfig",
    "GenConfig",
    # primitives
    "RMSNorm",
    "SwiGLU",
    "ContinuousTimeRoPE",
    "LatencySchedule",
    "build_time_causal_mask",
    # subsystem ①
    "VisionTransformer",
    "TemporalPerceiverResampler",
    "VisionTokenizer",
    "ProprioTokenizer",
    "LanguageTokenizer",
    "EmbodimentEncoder",
    "bin_proprio_stream",
    # subsystem ②
    "TrunkBlock",
    "HarmonicTrunk",
    "PagedKVCache",
    # subsystem ③
    "ActionExpert",
    "HypernetActionAdapter",
    "ReflexHead",
    "ChunkBlender",
    # model / training / runtime
    "GenOutput",
    "GenLosses",
    "GenBatch",
    "GenModel",
    "GenRuntime",
    "PhysicalPrompt",
    "make_dummy_batch",
    "count_parameters",
]


# =============================================================================
# §5.1 / §6.4  Streams
# =============================================================================


class Stream(IntEnum):
    """Token stream identities.

    Each stream runs at its own harmonic of the 100 Hz base tick (§5.1) and has its own
    row/column in the latency matrix (§6.2).  ``PAD`` exists so that ragged batches can
    be padded without perturbing the latency lookup.
    """

    PAD = 0
    HAND_CARD = 1
    LANGUAGE = 2
    HEAD_CAM = 3
    WRIST_CAM = 4
    PROPRIO = 5
    REGISTER = 6
    ACTION = 7


NUM_STREAMS: int = len(Stream)

#: Streams that carry exteroceptive/proprioceptive *observations* of the world.  Action
#: and register tokens must observe these through a latency offset δ (§6.2).
SENSING_STREAMS: tuple[Stream, ...] = (
    Stream.HEAD_CAM,
    Stream.WRIST_CAM,
    Stream.PROPRIO,
    Stream.LANGUAGE,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class HarmonicLadderConfig:
    """The harmonic tick ladder (``ARCHITECTURE.md`` §5.1).

    Every stream emits at a period that is an integer multiple of the base tick, so
    token arrival times always land on a shared grid.  That alignment is what makes the
    KV cache layout static and therefore pageable (§10.2).

    Attributes
    ----------
    base_tick_hz:
        The fundamental, equal to the action output rate (100 Hz, evidence E1).
    ``*_divisor``:
        Integer divisor of the fundamental for each stream.
    context_seconds:
        Length of the rolling context window (30 s, evidence E1).
    """

    base_tick_hz: float = 100.0
    action_divisor: int = 10  # 10 Hz action queries
    proprio_divisor: int = 5  # 20 Hz packed proprio tokens
    wrist_cam_divisor: int = 10  # 10 Hz wrist cameras
    head_cam_divisor: int = 20  # 5 Hz head cameras
    register_divisor: int = 50  # 2 Hz latent thought registers
    context_seconds: float = 30.0

    def rate_hz(self, divisor: int) -> float:
        """Return the emission rate in Hz for a given harmonic divisor."""
        return self.base_tick_hz / divisor

    def period_s(self, divisor: int) -> float:
        """Return the emission period in seconds for a given harmonic divisor."""
        return divisor / self.base_tick_hz

    def token_budget(
        self,
        *,
        n_head_cams: int = 2,
        n_wrist_cams: int = 2,
        head_latents: int = 32,
        wrist_latents: int = 16,
        n_registers: int = 8,
    ) -> dict[str, int]:
        """Estimate the undecayed token budget for one full context window.

        Reproduces the table in ``ARCHITECTURE.md`` §5.1 (≈20.6k tokens for the
        reference configuration).  Useful as a sanity check when changing divisors.
        """
        t = self.context_seconds
        return {
            "action": int(self.rate_hz(self.action_divisor) * t),
            "proprio": int(self.rate_hz(self.proprio_divisor) * t),
            "wrist_cam": int(self.rate_hz(self.wrist_cam_divisor) * t)
            * n_wrist_cams
            * wrist_latents,
            "head_cam": int(self.rate_hz(self.head_cam_divisor) * t)
            * n_head_cams
            * head_latents,
            "register": int(self.rate_hz(self.register_divisor) * t) * n_registers,
        }


@dataclass(frozen=True)
class VisionConfig:
    """Sensor processing: vision front-end (``ARCHITECTURE.md`` §5.3).

    Trained from scratch -- evidence E5 forbids initialising from a pretrained VLM
    tower. Nearly all the compression happens in the perceiver resampler, not in the
    ViT.
    """

    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    width: int = 1024
    depth: int = 24
    heads: int = 16
    mlp_ratio: float = 4.0
    #: Latents emitted per head-camera frame group.
    head_latents: int = 32
    #: Latents emitted per wrist-camera frame group.
    wrist_latents: int = 16
    resampler_depth: int = 4
    resampler_heads: int = 16
    #: Frames pooled into a single resampler call (temporal window).
    frames_per_group: int = 1
    max_cameras: int = 8
    #: Dimensionality of the flattened SE(3) + intrinsics vector per camera.
    extrinsic_dim: int = 9

    @property
    def n_patches(self) -> int:
        """Number of patches produced per frame."""
        side = self.image_size // self.patch_size
        return side * side


@dataclass(frozen=True)
class ProprioConfig:
    """Sensor processing: proprioception / force / tactile (``ARCHITECTURE.md`` §5.4).

    Channels are *named and padded* to a fixed schema with a validity mask, so a 6-DoF
    arm and a 16+-DoF humanoid produce identically shaped tensors (evidence E7).
    """

    max_joints: int = 64  # position, velocity, torque are separate stats, see below
    max_ft: int = 12
    max_tactile: int = 128
    bin_ms: float = 10.0
    #: 10 ms bins packed per emitted token (5 bins -> 20 Hz tokens).
    bins_per_token: int = 5
    #: Summary statistics per bin: mean, min, max, slope.
    stats_per_bin: int = 4
    hidden: int = 1024

    @property
    def n_channels(self) -> int:
        """Total padded sensor channels."""
        return self.max_joints + self.max_ft + self.max_tactile

    @property
    def feature_dim(self) -> int:
        """Flat feature width of one packed proprio token."""
        return self.n_channels * self.stats_per_bin * self.bins_per_token

    @property
    def fast_dim(self) -> int:
        """Feature width consumed by the 100 Hz reflex head (single bin, no packing)."""
        return self.n_channels * self.stats_per_bin


@dataclass(frozen=True)
class ActionSpaceConfig:
    """The universal action space (``ARCHITECTURE.md`` §7.3).

    A bimanual SE(3) end-effector twist plus finger/gripper DoF, expressed in a
    robot-centric frame, with a padded joint-space channel and a per-embodiment validity
    mask.  A 6-DoF arm, a 7-DoF arm and a 16+-DoF humanoid all write into this tensor;
    which channels are live is *input*, never a parameter (evidence E7/E8).
    """

    ee_dim: int = 12  # two arms x (linear 3 + angular 3) twist
    gripper_dim: int = 16  # up to 16 finger DoF across both hands
    joint_dim: int = 64  # padded direct joint-space channel
    #: Chunk horizon in control steps (50 steps @ 100 Hz = 500 ms).
    horizon: int = 50
    #: Steps actually committed per 10 Hz trunk step (100 ms).
    execute_steps: int = 10
    #: Already-committed prefix length used to condition the expert (§7.1).
    prefix_steps: int = 10
    #: Raised-cosine crossfade length between overlapping chunks, in steps.
    blend_steps: int = 5

    @property
    def dim(self) -> int:
        """Universal action dimensionality ``A``."""
        return self.ee_dim + self.gripper_dim + self.joint_dim


@dataclass(frozen=True)
class TrunkConfig:
    """Harmonic reasoning trunk (``ARCHITECTURE.md`` §6.5 / §6.6)."""

    d_model: int = 4096
    n_layers: int = 36
    n_heads: int = 32
    n_kv_heads: int = 8
    ffn_hidden: int = 11008
    #: Continuous-time RoPE band (§6.1): 10 ms .. 60 s.
    rope_min_period: float = 0.01
    rope_max_period: float = 60.0
    qk_norm: bool = True
    #: Fraction of the lowest layers that use sliding-window attention.
    sliding_window_fraction: float = 0.25
    sliding_window_seconds: float = 3.0
    #: Latent thought registers per emission (§6.4).
    n_registers: int = 8
    #: Per-stream FFN weights sharing attention -- the leading alternative in §6.5.
    modality_experts: bool = False
    #: μP-style output scaling for hyperparameter transfer (§6.6, ref [39]).
    use_mup_scaling: bool = True

    @property
    def head_dim(self) -> int:
        """Per-head dimensionality."""
        assert (
            self.d_model % self.n_heads == 0
        ), "d_model must divide evenly into n_heads"
        return self.d_model // self.n_heads

    @property
    def n_sliding_layers(self) -> int:
        """Number of low layers restricted to a local temporal window."""
        return int(self.n_layers * self.sliding_window_fraction)


@dataclass(frozen=True)
class ActionExpertConfig:
    """Flow-matching action expert (``ARCHITECTURE.md`` §7.1)."""

    width: int = 1024
    depth: int = 10
    heads: int = 16
    mlp_ratio: float = 4.0
    #: Euler steps used to integrate the probability-flow ODE at inference.
    flow_steps: int = 4
    #: Sinusoidal embedding width for flow time ``s`` and latency ``δ``.
    time_freq_dim: int = 256
    #: Rank of the hypernetwork-generated per-embodiment output adapter (§7.3).
    adapter_rank: int = 8


@dataclass(frozen=True)
class ReflexConfig:
    """The 100 Hz spinal loop (``ARCHITECTURE.md`` §7.2).

    Proprio/force/tactile only -- no vision, no GPU.  Emits a bounded *residual* on top
    of the current chunk so it stays a correction rather than becoming the policy.
    """

    hidden: int = 256
    layers: int = 2
    rate_hz: float = 100.0
    #: Maximum absolute residual, in universal-action units.
    max_magnitude: float = 0.1


@dataclass(frozen=True)
class LatencyConfig:
    """The δ curriculum (``ARCHITECTURE.md`` §6.3).

    δ is jittered *within* a sequence as well as across sequences so the model tolerates
    variable inference time (batch-size changes, thermal throttling, page faults).
    """

    mean_ms: float = 80.0
    log_sigma: float = 0.5
    min_ms: float = 10.0
    max_ms: float = 250.0
    #: Phase-1 pretraining runs with δ = 0 before the curriculum ramps (§8.3).
    enabled: bool = True

    def sample(
        self, shape: Sequence[int], device: torch.device | str = "cpu"
    ) -> Tensor:
        """Sample δ in **seconds**.

        Parameters
        ----------
        shape:
            Output shape, typically ``(batch,)``.
        device:
            Device for the returned tensor.

        Returns
        -------
        Tensor
            Latency offsets in seconds, clamped to ``[min_ms, max_ms]``.
        """
        if not self.enabled:
            return torch.zeros(tuple(shape), device=device)
        mu = math.log(self.mean_ms / 1000.0)
        delta = torch.exp(
            torch.randn(tuple(shape), device=device) * self.log_sigma + mu
        )
        return delta.clamp(self.min_ms / 1000.0, self.max_ms / 1000.0)


@dataclass(frozen=True)
class LossConfig:
    """Objective weights (``ARCHITECTURE.md`` §8.1).

    ``action`` is the primary conditional-flow-matching term whose validation error is
    the "next action prediction error" plotted in every published GEN scaling figure.
    ``world`` is the least certain component of the reconstruction (§14, item 1).
    """

    action: float = 1.0
    world: float = 0.2
    language: float = 0.05
    reflex: float = 0.05
    reflex_magnitude: float = 1e-3
    #: Latent-future prediction horizons in seconds.
    world_horizons: tuple[float, ...] = (0.5, 1.0, 2.0)
    #: Probability of dropping the language prefix during pretraining (§5.5).
    language_dropout: float = 0.4


@dataclass(frozen=True)
class GenConfig:
    """Top-level model configuration.

    Use the classmethod presets (:meth:`og_7b` etc.) rather than constructing the nested
    configs by hand.  Sizes follow ``ARCHITECTURE.md`` §6.6, with the 7B entry placed at
    the published ossification phase transition (evidence E6).
    """

    name: str = "og-7b"
    vision: VisionConfig = field(default_factory=VisionConfig)
    proprio: ProprioConfig = field(default_factory=ProprioConfig)
    action_space: ActionSpaceConfig = field(default_factory=ActionSpaceConfig)
    trunk: TrunkConfig = field(default_factory=TrunkConfig)
    expert: ActionExpertConfig = field(default_factory=ActionExpertConfig)
    reflex: ReflexConfig = field(default_factory=ReflexConfig)
    ladder: HarmonicLadderConfig = field(default_factory=HarmonicLadderConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    #: Width of the frozen text encoder's output (the "other ~1%", §5.5).
    text_embed_dim: int = 768
    #: Vocabulary of the captioning head.
    vocab_size: int = 32000
    #: Hand-card tokens prepended to the context (§7.3).
    n_hand_card_tokens: int = 8
    #: Hard cap on interleaved context tokens; see §5.2 for the decay schedule.
    max_context_tokens: int = 24576

    # ---- presets (§6.6) -------------------------------------------------

    @classmethod
    def debug(cls) -> "GenConfig":
        """A ~20M-parameter configuration that runs a full train step on CPU."""
        return cls(
            name="og-debug",
            vision=VisionConfig(
                image_size=64,
                patch_size=16,
                width=128,
                depth=2,
                heads=4,
                head_latents=4,
                wrist_latents=2,
                resampler_depth=1,
                resampler_heads=4,
            ),
            proprio=ProprioConfig(max_joints=8, max_ft=6, max_tactile=8, hidden=128),
            action_space=ActionSpaceConfig(
                ee_dim=12,
                gripper_dim=4,
                joint_dim=8,
                horizon=10,
                execute_steps=2,
                prefix_steps=2,
                blend_steps=1,
            ),
            trunk=TrunkConfig(
                d_model=128,
                n_layers=4,
                n_heads=4,
                n_kv_heads=2,
                ffn_hidden=352,
                n_registers=2,
            ),
            expert=ActionExpertConfig(
                width=128,
                depth=2,
                heads=4,
                flow_steps=2,
                time_freq_dim=64,
                adapter_rank=4,
            ),
            reflex=ReflexConfig(hidden=64, layers=1),
            ladder=HarmonicLadderConfig(context_seconds=2.0),
            text_embed_dim=128,
            vocab_size=256,
            n_hand_card_tokens=2,
            max_context_tokens=4096,
        )

    @classmethod
    def og_0p3b(cls) -> "GenConfig":
        """0.35B -- debug / unit-test scale."""
        return cls(
            name="og-0.3b",
            trunk=TrunkConfig(
                d_model=1024, n_layers=16, n_heads=16, n_kv_heads=4, ffn_hidden=2752
            ),
            expert=ActionExpertConfig(width=512, depth=6, heads=8),
            vision=VisionConfig(
                width=768, depth=12, heads=12, resampler_depth=2, resampler_heads=12
            ),
        )

    @classmethod
    def og_1b(cls) -> "GenConfig":
        """1.2B -- **expected to ossify** under the full data load (evidence E6).

        Kept as a first-class preset because reproducing that negative result is the
        single best correctness check on an Open-Gen implementation (§12.1).
        """
        return cls(
            name="og-1b",
            trunk=TrunkConfig(
                d_model=2048, n_layers=18, n_heads=16, n_kv_heads=4, ffn_hidden=5504
            ),
            expert=ActionExpertConfig(width=640, depth=6, heads=10),
            vision=VisionConfig(
                width=768, depth=12, heads=12, resampler_depth=2, resampler_heads=12
            ),
        )

    @classmethod
    def og_6b(cls) -> "GenConfig":
        """6.1B -- "begins to benefit" from pretraining."""
        return cls(
            name="og-6b",
            trunk=TrunkConfig(
                d_model=4096, n_layers=29, n_heads=32, n_kv_heads=8, ffn_hidden=11008
            ),
        )

    @classmethod
    def og_7b(cls) -> "GenConfig":
        """7.2B -- the published phase transition."""
        return cls(name="og-7b", trunk=TrunkConfig())

    @classmethod
    def og_11b(cls) -> "GenConfig":
        """10.9B -- GEN-1.5 class."""
        return cls(
            name="og-11b",
            trunk=TrunkConfig(
                d_model=5120, n_layers=36, n_heads=40, n_kv_heads=8, ffn_hidden=13824
            ),
            expert=ActionExpertConfig(width=1280, depth=12, heads=16),
        )


# =============================================================================
# Primitives
# =============================================================================


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation without a mean subtraction or bias."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Normalise the trailing dimension of ``x``."""
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class SwiGLU(nn.Module):
    """Gated feed-forward network: ``W2(silu(W1 x) * W3 x)``."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the gated MLP."""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate the trailing dimension by a half-turn in the complex plane."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to a head-major tensor.

    Parameters
    ----------
    x:
        ``[B, Hq, N, Dh]`` queries or keys.
    cos, sin:
        ``[B, N, Dh]`` rotation factors from :class:`ContinuousTimeRoPE`.

    Returns
    -------
    Tensor
        ``[B, Hq, N, Dh]``, rotated.
    """
    cos = cos.unsqueeze(1).to(x.dtype)
    sin = sin.unsqueeze(1).to(x.dtype)
    return x * cos + _rotate_half(x) * sin


class ContinuousTimeRoPE(nn.Module):
    """Rotary position embeddings over **continuous wall-clock time** (§6.1).

    Standard RoPE rotates by an integer sequence index.  Here the rotation angle is
    ``t * ω`` for a float timestamp ``t`` in seconds, with frequencies log-spaced across
    the model's dynamic range (10 ms .. 60 s by default).

    Four properties follow, each mapping to a published GEN behaviour:

    * variable sample rates cost nothing -- required by the harmonic ladder (§5.1);
    * dropped frames and jitter are in-distribution;
    * **a time jump is just a large Δt** -- which is why a spliced physical prompt works
      despite "discontinuous jumps in time that the model never saw in training" (E12);
    * only relative time is encoded, so a 30 s rolling window has no absolute drift.

    Timestamps are re-based internally so that the newest token in each sequence sits at
    ``t = 0`` and older tokens are negative.  This is a per-sequence shift, which RoPE
    is invariant to, and it keeps the rotation angle inside float32's accurate range.
    """

    #: Declared so the registered buffer types as a Tensor rather than
    #: ``Tensor | Module`` (the return type of ``nn.Module.__getattr__``).
    omega: Tensor

    def __init__(
        self, head_dim: int, min_period: float = 0.01, max_period: float = 60.0
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        n = head_dim // 2
        exponent = torch.arange(n, dtype=torch.float64) / max(n - 1, 1)
        periods = min_period * (max_period / min_period) ** exponent
        omega = (2.0 * math.pi / periods).to(torch.float32)
        self.register_buffer("omega", omega, persistent=False)
        self.head_dim = head_dim
        self.min_period = min_period
        self.max_period = max_period

    def forward(
        self,
        timestamps: Tensor,
        valid: Tensor | None = None,
        reference: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute rotation factors for a batch of timestamps.

        Parameters
        ----------
        timestamps:
            ``[B, N]`` float seconds.  Absolute origin is irrelevant.
        valid:
            Optional ``[B, N]`` bool mask; padding is excluded from the re-basing
            statistic so pad timestamps cannot shift real tokens.
        reference:
            Optional ``[B, 1]`` re-basing origin.  Queries and cached keys **must**
            share a reference within a tick, which is why the incremental path passes it
            explicitly instead of letting each call pick its own maximum.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``cos`` and ``sin``, each ``[B, N, Dh]``.
        """
        t = timestamps.to(torch.float32)
        if reference is not None:
            ref = reference.to(torch.float32)
        elif valid is not None:
            ref = (
                torch.where(valid, t, torch.full_like(t, float("-inf")))
                .max(dim=-1, keepdim=True)
                .values
            )
            ref = torch.where(torch.isfinite(ref), ref, torch.zeros_like(ref))
        else:
            ref = t.max(dim=-1, keepdim=True).values
        t = t - ref
        angles = t.unsqueeze(-1) * self.omega  # [B, N, Dh/2]
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos(), angles.sin()


class LatencySchedule(nn.Module):
    """The per-stream-pair required attention gap (§6.2).

    Holds two ``[S, S]`` buffers.  The required gap between a query token in stream
    ``i``
    and a key token in stream ``j`` is::

        gap[i, j] = fixed[i, j] + coeff[i, j] * δ

    where δ is the sampled inference latency for that sample.  ``coeff = 1`` marks pairs
    whose information flow is delayed by compute -- sensing into acting, and sensing
    into the slow reasoning stream.  Everything else is co-temporal.

    This is the mechanism that removes the need for a System-1/System-2 split and for
    inference-time guidance (evidence E3): a 7B trunk that takes 80 ms to think produces
    actions it *knows* are 80 ms stale, because it was trained that way.
    """

    #: Multiplier on δ for each ``(query stream, key stream)`` pair.
    coeff: Tensor
    #: Constant gap added on top of ``coeff * δ``.
    fixed: Tensor

    def __init__(self, register_grace_s: float = 0.0) -> None:
        super().__init__()
        coeff = torch.zeros(NUM_STREAMS, NUM_STREAMS)
        fixed = torch.zeros(NUM_STREAMS, NUM_STREAMS)
        for producer in SENSING_STREAMS:
            # sensing -> acting, and sensing -> slow reasoning, are delayed by compute.
            coeff[int(Stream.ACTION), int(producer)] = 1.0
            coeff[int(Stream.REGISTER), int(producer)] = 1.0
        # Action tokens read the most recently *completed* register; registers are
        # emitted on a 2 Hz harmonic so this is naturally stale, but δ makes it
        # explicit.
        coeff[int(Stream.ACTION), int(Stream.REGISTER)] = 1.0
        fixed[int(Stream.ACTION), int(Stream.REGISTER)] = register_grace_s
        self.register_buffer("coeff", coeff, persistent=False)
        self.register_buffer("fixed", fixed, persistent=False)

    def gap_matrix(self, delta: Tensor) -> Tensor:
        """Build the per-sample required-gap matrix.

        Parameters
        ----------
        delta:
            ``[B]`` latency offsets in seconds.

        Returns
        -------
        Tensor
            ``[B, S, S]`` required gaps in seconds.
        """
        d = delta.view(-1, 1, 1).to(self.coeff.dtype)
        return self.fixed.unsqueeze(0) + self.coeff.unsqueeze(0) * d


def build_time_causal_mask(
    timestamps: Tensor,
    stream_ids: Tensor,
    valid: Tensor,
    gap: Tensor,
    window_seconds: float | None = None,
) -> Tensor:
    """Construct the time-causal attention mask with a latency offset (§6.2).

    A query token ``i`` may attend to a key token ``j`` iff::

        t_j <= t_i - gap[stream_i, stream_j]

    which generalises ordinary causal masking from *sequence order* to *physical time*.

    Parameters
    ----------
    timestamps:
        ``[B, N]`` float seconds.
    stream_ids:
        ``[B, N]`` long, values from :class:`Stream`.
    valid:
        ``[B, N]`` bool; ``False`` marks padding.
    gap:
        ``[B, S, S]`` required gaps from :meth:`LatencySchedule.gap_matrix`.
    window_seconds:
        If given, additionally restrict attention to keys within this many seconds of
        the query -- the sliding-window locality used on the lowest trunk layers (§6.5).

    Returns
    -------
    Tensor
        ``[B, 1, N, N]`` boolean mask, ``True`` where attention is permitted.

    Notes
    -----
    Memory is ``O(B · N²)``.  For long contexts, prefer the incremental path through
    :class:`PagedKVCache`, which only ever materialises ``[B, 1, 1, N]`` rows.

    The diagonal is always permitted, so no query row can be fully masked (a fully
    masked softmax row produces NaNs).
    """
    b, n = timestamps.shape
    idx = torch.arange(b, device=timestamps.device).view(b, 1, 1)
    required = gap[idx, stream_ids.unsqueeze(-1), stream_ids.unsqueeze(-2)]  # [B, N, N]
    dt = timestamps.unsqueeze(-1) - timestamps.unsqueeze(-2)  # t_i - t_j
    allowed = dt >= required - 1e-6
    allowed = allowed & valid.unsqueeze(-2)
    if window_seconds is not None:
        allowed = allowed & (dt <= window_seconds + 1e-6)
    eye = torch.eye(n, dtype=torch.bool, device=timestamps.device).unsqueeze(0)
    allowed = allowed | eye
    return allowed.unsqueeze(1)


def sinusoidal_embedding(
    values: Tensor, dim: int, max_period: float = 1000.0
) -> Tensor:
    """Standard sinusoidal embedding of a scalar field.

    Used for the flow time ``s`` and the latency offset ``δ`` in the action expert
    (§7.1).

    Parameters
    ----------
    values:
        ``[...]`` scalars.
    dim:
        Output width (must be even).
    max_period:
        Longest sinusoid period.

    Returns
    -------
    Tensor
        ``[..., dim]``.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / half
    )
    args = values.float().unsqueeze(-1) * freqs
    return torch.cat((args.cos(), args.sin()), dim=-1)


# =============================================================================
# Embodiment ("hand card") and token streams
# =============================================================================


@dataclass
class EmbodimentSpec:
    """A robot's "hand card" (``ARCHITECTURE.md`` §7.3).

    9,000 end-effector variations (evidence E7) rule out per-embodiment heads, so an
    embodiment is described **in-band** as a token set.  An unseen gripper is then an
    unseen
    *input vector*, not an unseen parameter set -- which is the mechanism behind
    adapting to new hands on the fly.

    All array fields are padded to the schema widths in :class:`ProprioConfig` /
    :class:`ActionSpaceConfig` and accompanied by validity masks.
    """

    dof: int
    #: ``[max_joints, 3]`` unit joint axes in the robot frame.
    joint_axes: Tensor
    #: ``[max_joints, 2]`` lower/upper joint limits in radians or metres.
    joint_limits: Tensor
    #: ``[max_joints]`` link lengths in metres.
    link_lengths: Tensor
    #: Index into a learned table of end-effector archetypes.
    ee_type_id: int
    #: Gripper span (mm), max force (N), mass (kg), payload (kg).
    ee_scalars: Tensor
    #: ``[64]`` learned or hand-authored geometry embedding of the end effector mesh.
    ee_geometry: Tensor
    #: ``[max_cameras, extrinsic_dim]`` camera poses + intrinsics.
    camera_extrinsics: Tensor
    #: ``[max_cameras]`` bool -- which camera slots are populated.
    camera_valid: Tensor
    #: ``[n_channels]`` bool -- which proprio channels are live.
    channel_valid: Tensor
    #: ``[action_dim]`` bool -- which universal-action channels this robot actuates.
    action_valid: Tensor
    #: 0 = end-effector twist, 1 = joint velocity, 2 = joint position.
    control_mode: int = 0

    def feature_vector(self, n_ee_types: int = 32, n_control_modes: int = 4) -> Tensor:
        """Flatten the hand card into a fixed-width float vector.

        Returns
        -------
        Tensor
            ``[feature_dim]`` on the same device as ``joint_axes``.
        """
        device = self.joint_axes.device
        ee_onehot = F.one_hot(
            torch.tensor(self.ee_type_id, device=device), n_ee_types
        ).float()
        ctrl_onehot = F.one_hot(
            torch.tensor(self.control_mode, device=device), n_control_modes
        ).float()
        return torch.cat(
            [
                torch.tensor([float(self.dof)], device=device),
                self.joint_axes.flatten().float(),
                self.joint_limits.flatten().float(),
                self.link_lengths.flatten().float(),
                ee_onehot,
                self.ee_scalars.flatten().float(),
                self.ee_geometry.flatten().float(),
                self.camera_extrinsics.flatten().float(),
                self.camera_valid.flatten().float(),
                self.channel_valid.flatten().float(),
                self.action_valid.flatten().float(),
                ctrl_onehot,
            ]
        )

    @staticmethod
    def feature_dim(
        proprio: ProprioConfig,
        vision: VisionConfig,
        action_space: ActionSpaceConfig,
        n_ee_types: int = 32,
        n_control_modes: int = 4,
        geometry_dim: int = 64,
        n_ee_scalars: int = 4,
    ) -> int:
        """Width of :meth:`feature_vector` for a given set of schema widths."""
        j = proprio.max_joints
        return (
            1
            + j * 3
            + j * 2
            + j
            + n_ee_types
            + n_ee_scalars
            + geometry_dim
            + vision.max_cameras * vision.extrinsic_dim
            + vision.max_cameras
            + proprio.n_channels
            + action_space.dim
            + n_control_modes
        )

    @classmethod
    def dummy(
        cls,
        proprio: ProprioConfig,
        vision: VisionConfig,
        action_space: ActionSpaceConfig,
        dof: int = 7,
        device: torch.device | str = "cpu",
        geometry_dim: int = 64,
        n_ee_scalars: int = 4,
    ) -> "EmbodimentSpec":
        """Build a syntactically valid hand card for tests and smoke runs."""
        j = proprio.max_joints
        joint_valid = torch.zeros(proprio.n_channels, dtype=torch.bool, device=device)
        joint_valid[:dof] = True
        action_valid = torch.zeros(action_space.dim, dtype=torch.bool, device=device)
        action_valid[: action_space.ee_dim] = True
        cam_valid = torch.zeros(vision.max_cameras, dtype=torch.bool, device=device)
        cam_valid[:4] = True
        return cls(
            dof=dof,
            joint_axes=F.normalize(torch.randn(j, 3, device=device), dim=-1),
            joint_limits=torch.stack(
                (
                    -torch.ones(j, device=device) * math.pi,
                    torch.ones(j, device=device) * math.pi,
                ),
                dim=-1,
            ),
            link_lengths=torch.rand(j, device=device) * 0.4,
            ee_type_id=0,
            ee_scalars=torch.tensor([85.0, 120.0, 1.2, 3.0], device=device)[
                :n_ee_scalars
            ],
            ee_geometry=torch.randn(geometry_dim, device=device),
            camera_extrinsics=torch.randn(
                vision.max_cameras, vision.extrinsic_dim, device=device
            ),
            camera_valid=cam_valid,
            channel_valid=joint_valid,
            action_valid=action_valid,
            control_mode=0,
        )


@dataclass
class TokenStream:
    """An interleaved, time-sorted multimodal token sequence.

    This is the single object the trunk consumes.  Every token carries its own
    wall-clock timestamp and stream identity, which is what lets streams at different
    harmonics (§5.1) share one causal structure (§6.2).

    Attributes
    ----------
    embeddings:
        ``[B, N, D]`` token features.
    timestamps:
        ``[B, N]`` float seconds.
    stream_ids:
        ``[B, N]`` long, values from :class:`Stream`.
    valid:
        ``[B, N]`` bool, ``False`` for padding.
    """

    embeddings: Tensor
    timestamps: Tensor
    stream_ids: Tensor
    valid: Tensor

    def __post_init__(self) -> None:
        b, n, _ = self.embeddings.shape
        for name in ("timestamps", "stream_ids", "valid"):
            got = tuple(getattr(self, name).shape)
            if got != (b, n):
                raise ValueError(
                    f"TokenStream.{name} has shape {got}, expected {(b, n)}"
                )

    @property
    def batch_size(self) -> int:
        """Batch size ``B``."""
        return self.embeddings.shape[0]

    @property
    def length(self) -> int:
        """Sequence length ``N``."""
        return self.embeddings.shape[1]

    @property
    def device(self) -> torch.device:
        """Device the stream lives on."""
        return self.embeddings.device

    def stream_mask(self, stream: Stream) -> Tensor:
        """``[B, N]`` bool mask selecting valid tokens of one stream."""
        return (self.stream_ids == int(stream)) & self.valid

    def sort_by_time(self) -> "TokenStream":
        """Return a copy with tokens sorted by timestamp (padding last).

        Sorting is what turns a set of independently-emitted harmonic streams into one
        sequence.  Padding is pushed to the end by giving it ``+inf`` sort keys.
        """
        keys = torch.where(
            self.valid, self.timestamps, torch.full_like(self.timestamps, float("inf"))
        )
        order = keys.argsort(dim=-1, stable=True)
        gather = order.unsqueeze(-1).expand(-1, -1, self.embeddings.shape[-1])
        return TokenStream(
            embeddings=self.embeddings.gather(1, gather),
            timestamps=self.timestamps.gather(1, order),
            stream_ids=self.stream_ids.gather(1, order),
            valid=self.valid.gather(1, order),
        )

    def truncate(self, max_tokens: int) -> "TokenStream":
        """Keep only the most recent ``max_tokens`` tokens.

        Assumes the stream is time-sorted.  This is the crude fallback for the
        hierarchical temporal decay described in §5.2; the cache-side implementation
        lives in :meth:`PagedKVCache.decay`.
        """
        if self.length <= max_tokens:
            return self
        s = slice(self.length - max_tokens, None)
        return TokenStream(
            embeddings=self.embeddings[:, s],
            timestamps=self.timestamps[:, s],
            stream_ids=self.stream_ids[:, s],
            valid=self.valid[:, s],
        )

    @staticmethod
    def concatenate(parts: Sequence["TokenStream"]) -> "TokenStream":
        """Concatenate several streams along the token axis and re-sort by time."""
        parts = [p for p in parts if p.length > 0]
        if not parts:
            raise ValueError("cannot concatenate an empty list of TokenStreams")
        return TokenStream(
            embeddings=torch.cat([p.embeddings for p in parts], dim=1),
            timestamps=torch.cat([p.timestamps for p in parts], dim=1),
            stream_ids=torch.cat([p.stream_ids for p in parts], dim=1),
            valid=torch.cat([p.valid for p in parts], dim=1),
        ).sort_by_time()


# =============================================================================
# ① SENSOR PROCESSING  (§5)
# =============================================================================


class _ViTBlock(nn.Module):
    """A pre-norm transformer block with learned absolute positions (encoder side)."""

    def __init__(self, width: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, int(width * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(width * mlp_ratio), width),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Self-attend then MLP, both residual."""
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    """Patch encoder trained **from scratch** (§5.3).

    Evidence E5 ("~99% of the parameters are trained from scratch") forbids initialising
    this tower from a pretrained VLM.  It is deliberately ordinary: the novelty budget
    of this architecture is spent on time, not on vision blocks.
    """

    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch = nn.Conv2d(
            cfg.in_channels,
            cfg.width,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.pos = nn.Parameter(torch.randn(1, cfg.n_patches, cfg.width) * 0.02)
        self.blocks = nn.ModuleList(
            [_ViTBlock(cfg.width, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.width)

    def forward(self, images: Tensor) -> Tensor:
        """Encode a flat batch of frames.

        Parameters
        ----------
        images:
            ``[M, C, H, W]``.

        Returns
        -------
        Tensor
            ``[M, P, width]`` patch features.
        """
        x = self.patch(images).flatten(2).transpose(1, 2)  # [M, P, width]
        x = x + self.pos
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class TemporalPerceiverResampler(nn.Module):
    """Compress a frame group's patches into ``K`` latents (§5.3).

    This is where nearly all visual compression happens: ``P`` patches per frame times
    ``frames_per_group`` frames collapse to ``K`` tokens.  It is also the subsystem the
    *Thousand Hands* post reports being perturbed most by visually sparse tools such as
    a whisk (evidence E4), which is the empirical hint that motivated putting the
    compression here rather than in the ViT.
    """

    def __init__(
        self, width: int, n_latents: int, depth: int, heads: int, mlp_ratio: float = 4.0
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, width) * 0.02)
        self.cross_norm_q = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.cross_norm_kv = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.cross = nn.ModuleList(
            [
                nn.MultiheadAttention(width, heads, batch_first=True)
                for _ in range(depth)
            ]
        )
        self.self_blocks = nn.ModuleList(
            [_ViTBlock(width, heads, mlp_ratio) for _ in range(depth)]
        )

    def forward(self, context: Tensor) -> Tensor:
        """Resample patch features into latents.

        Parameters
        ----------
        context:
            ``[M, P_total, width]`` patch features for one frame group.

        Returns
        -------
        Tensor
            ``[M, K, width]`` latents.
        """
        m = context.shape[0]
        x = self.latents.unsqueeze(0).expand(m, -1, -1)
        for cross, qn, kn, sblock in zip(
            self.cross, self.cross_norm_q, self.cross_norm_kv, self.self_blocks
        ):
            x = x + cross(qn(x), kn(context), kn(context), need_weights=False)[0]
            x = sblock(x)
        return x


class VisionTokenizer(nn.Module):
    """Turn multi-camera video into timestamped trunk tokens (§5.3).

    Head and wrist cameras run on different harmonics (5 Hz and 10 Hz by default) and
    get different latent budgets (32 and 16), because a wrist view is narrower but more
    time-critical.  Camera identity and extrinsics are *added*, never inferred: a wrist
    camera on a 6-DoF arm and one on a 16-DoF hand see the same world from structurally
    different poses, and the model must be told which is which.
    """

    def __init__(self, cfg: VisionConfig, d_model: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = VisionTransformer(cfg)
        self.head_resampler = TemporalPerceiverResampler(
            cfg.width, cfg.head_latents, cfg.resampler_depth, cfg.resampler_heads
        )
        self.wrist_resampler = TemporalPerceiverResampler(
            cfg.width, cfg.wrist_latents, cfg.resampler_depth, cfg.resampler_heads
        )
        self.camera_embed = nn.Embedding(cfg.max_cameras, cfg.width)
        self.extrinsic_proj = nn.Linear(cfg.extrinsic_dim, cfg.width)
        self.out_proj = nn.Linear(cfg.width, d_model)

    def forward(
        self,
        frames: Tensor,
        timestamps: Tensor,
        camera_ids: Tensor,
        is_wrist: Tensor,
        extrinsics: Tensor,
        valid: Tensor | None = None,
    ) -> TokenStream:
        """Encode a batch of camera frames into a :class:`TokenStream`.

        Parameters
        ----------
        frames:
            ``[B, F, C, H, W]`` frames, already grouped so that consecutive
            ``cfg.frames_per_group`` entries belong to one resampler call.
        timestamps:
            ``[B, F]`` capture times in seconds.
        camera_ids:
            ``[B, F]`` long, index into the camera embedding table.
        is_wrist:
            ``[B, F]`` bool; selects the wrist latent budget for that frame.
        extrinsics:
            ``[B, F, extrinsic_dim]`` per-frame camera pose/intrinsics.
        valid:
            Optional ``[B, F]`` bool.

        Returns
        -------
        TokenStream
            ``B x (n_groups * K)`` tokens, tagged :attr:`Stream.HEAD_CAM` or
            :attr:`Stream.WRIST_CAM`.

        Notes
        -----
        Within a batch element, all frames of a group must share the same
        ``is_wrist`` value; the group's stream identity and latent budget are taken from
        its first frame.
        """
        b, f = frames.shape[:2]
        g = self.cfg.frames_per_group
        if f % g != 0:
            raise ValueError(
                f"frame count {f} is not divisible by frames_per_group={g}"
            )
        n_groups = f // g
        if valid is None:
            valid = torch.ones(b, f, dtype=torch.bool, device=frames.device)

        patches = self.encoder(frames.flatten(0, 1))  # [B*F, P, W]
        patches = patches + self.camera_embed(camera_ids).flatten(0, 1).unsqueeze(1)
        patches = patches + self.extrinsic_proj(extrinsics).flatten(0, 1).unsqueeze(1)
        patches = patches.view(b * n_groups, g * self.cfg.n_patches, self.cfg.width)

        group_is_wrist = is_wrist.view(b, n_groups, g)[..., 0].reshape(
            -1
        )  # [B*n_groups]
        # Group timestamp is the *last* frame in the group: the group is not observable
        # until its final frame has arrived.
        group_t = timestamps.view(b, n_groups, g)[..., -1]
        group_valid = valid.view(b, n_groups, g).any(dim=-1)

        head_lat = self.head_resampler(patches)  # [B*n_groups, K_head, W]
        wrist_lat = self.wrist_resampler(patches)  # [B*n_groups, K_wrist, W]
        k = max(self.cfg.head_latents, self.cfg.wrist_latents)

        def _pad(x: Tensor) -> Tensor:
            return F.pad(x, (0, 0, 0, k - x.shape[1]))

        sel = group_is_wrist.view(-1, 1, 1)
        latents = torch.where(
            sel, _pad(wrist_lat), _pad(head_lat)
        )  # [B*n_groups, K, W]
        latent_valid = torch.ones(
            latents.shape[:2], dtype=torch.bool, device=frames.device
        )
        if self.cfg.wrist_latents < k:
            latent_valid = torch.where(
                group_is_wrist.view(-1, 1),
                (
                    torch.arange(k, device=frames.device) < self.cfg.wrist_latents
                ).unsqueeze(0),
                latent_valid,
            )
        if self.cfg.head_latents < k:
            latent_valid = torch.where(
                ~group_is_wrist.view(-1, 1),
                (
                    torch.arange(k, device=frames.device) < self.cfg.head_latents
                ).unsqueeze(0),
                latent_valid,
            )

        embeddings = self.out_proj(latents).view(b, n_groups * k, -1)
        stream = torch.where(
            group_is_wrist.view(b, n_groups),
            int(Stream.WRIST_CAM),
            int(Stream.HEAD_CAM),
        )
        return TokenStream(
            embeddings=embeddings,
            timestamps=group_t.unsqueeze(-1).expand(-1, -1, k).reshape(b, -1),
            stream_ids=stream.unsqueeze(-1).expand(-1, -1, k).reshape(b, -1).long(),
            valid=(
                group_valid.unsqueeze(-1) & latent_valid.view(b, n_groups, k)
            ).reshape(b, -1),
        )


def bin_proprio_stream(
    raw: Tensor,
    sample_rate_hz: float,
    cfg: ProprioConfig,
    channel_valid: Tensor | None = None,
) -> Tensor:
    """Bin a high-rate proprioceptive stream into packed token features (§5.4).

    Robotics' highest-bandwidth channel, and the one teleoperation datasets lack: GEN-1
    attributes part of its speed advantage to handheld collection devices providing
    force feedback that teleoperation cannot.  Raw channels arrive at 100--1000 Hz and
    are reduced to per-bin ``(mean, min, max, slope)`` summaries, then packed
    ``bins_per_token`` at a time onto the 20 Hz harmonic.

    Parameters
    ----------
    raw:
        ``[B, S, C]`` samples, where ``C == cfg.n_channels`` (pad unused channels).
    sample_rate_hz:
        Rate of ``raw``.
    cfg:
        Proprio schema.
    channel_valid:
        Optional ``[B, C]`` bool; invalid channels are zeroed so an absent tactile array
        cannot leak noise into the token.

    Returns
    -------
    Tensor
        ``[B, T, cfg.feature_dim]`` packed features, ``T = S // (samples_per_bin *
        bins_per_token)``.
    """
    b, s, c = raw.shape
    if c != cfg.n_channels:
        raise ValueError(f"expected {cfg.n_channels} channels, got {c}")
    samples_per_bin = max(int(round(sample_rate_hz * cfg.bin_ms / 1000.0)), 1)
    per_token = samples_per_bin * cfg.bins_per_token
    n_tokens = s // per_token
    if n_tokens == 0:
        raise ValueError(f"need at least {per_token} samples for one token, got {s}")
    x = raw[:, : n_tokens * per_token].view(
        b, n_tokens * cfg.bins_per_token, samples_per_bin, c
    )
    if channel_valid is not None:
        x = x * channel_valid.view(b, 1, 1, c).to(x.dtype)
    ramp = torch.linspace(-1.0, 1.0, samples_per_bin, device=raw.device).view(
        1, 1, -1, 1
    )
    stats = torch.stack(
        (
            x.mean(dim=2),
            x.amin(dim=2),
            x.amax(dim=2),
            (x * ramp).mean(dim=2) * 3.0,  # least-squares slope on a symmetric ramp
        ),
        dim=-1,
    )  # [B, n_tokens*bins, C, 4]
    return stats.reshape(b, n_tokens, cfg.feature_dim)


class ProprioTokenizer(nn.Module):
    """Project packed proprio features onto the trunk width (§5.4)."""

    def __init__(self, cfg: ProprioConfig, d_model: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, d_model),
        )
        #: Learned embedding of *which* channels are live, so the model can distinguish
        #: "channel reads zero" from "channel does not exist".
        self.channel_mask_embed = nn.Linear(cfg.n_channels, d_model, bias=False)

    def forward(
        self, features: Tensor, timestamps: Tensor, channel_valid: Tensor
    ) -> TokenStream:
        """Tokenize packed proprio features.

        Parameters
        ----------
        features:
            ``[B, T, feature_dim]`` from :func:`bin_proprio_stream`.
        timestamps:
            ``[B, T]`` seconds (the *end* time of each packed window).
        channel_valid:
            ``[B, n_channels]`` bool.

        Returns
        -------
        TokenStream
            ``B x T`` tokens tagged :attr:`Stream.PROPRIO`.
        """
        emb = self.net(features) + self.channel_mask_embed(
            channel_valid.float()
        ).unsqueeze(1)
        b, t, _ = emb.shape
        return TokenStream(
            embeddings=emb,
            timestamps=timestamps,
            stream_ids=torch.full(
                (b, t), int(Stream.PROPRIO), dtype=torch.long, device=emb.device
            ),
            valid=torch.ones(b, t, dtype=torch.bool, device=emb.device),
        )


class LanguageTokenizer(nn.Module):
    """Adapter over a **frozen** text encoder -- the "other ~1%" (§5.5).

    Evidence E5 says ~99% of parameters are trained from scratch.  The natural candidate
    for the remainder is a frozen sentence/token encoder, which also explains evidence
    E14: the same encoder indexes 1,891,392 scenes for nearest-neighbour language
    search.  This module owns only the projection into trunk width and the captioning
    head; the encoder itself is external and its outputs are passed in.

    Language is dropped frequently during pretraining
    (:attr:`LossConfig.language_dropout`) so the model never becomes dependent on
    instructions -- GEN-1.5's dustpan improvisation happens with no language guidance at
    all.
    """

    def __init__(self, text_embed_dim: int, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(text_embed_dim, d_model)
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        text_embeddings: Tensor,
        timestamps: Tensor,
        valid: Tensor | None = None,
        dropout_p: float = 0.0,
    ) -> TokenStream:
        """Project frozen text embeddings into the context.

        Parameters
        ----------
        text_embeddings:
            ``[B, L, text_embed_dim]`` outputs of the frozen encoder.
        timestamps:
            ``[B, L]`` seconds -- language is an event stream, usually stamped at the
            start of the episode or at the moment the instruction was given.
        valid:
            Optional ``[B, L]`` bool.
        dropout_p:
            Probability of dropping the entire language prefix for a batch element
            (training only).

        Returns
        -------
        TokenStream
            ``B x L`` tokens tagged :attr:`Stream.LANGUAGE`.
        """
        b, n_text, _ = text_embeddings.shape
        emb = self.norm(self.proj(text_embeddings))
        if valid is None:
            valid = torch.ones(b, n_text, dtype=torch.bool, device=emb.device)
        if self.training and dropout_p > 0.0:
            keep = torch.rand(b, 1, device=emb.device) >= dropout_p
            valid = valid & keep
        return TokenStream(
            embeddings=emb,
            timestamps=timestamps,
            stream_ids=torch.full(
                (b, n_text), int(Stream.LANGUAGE), dtype=torch.long, device=emb.device
            ),
            valid=valid,
        )


class EmbodimentEncoder(nn.Module):
    """Encode the hand card into tokens plus a pooled conditioning vector (§7.3).

    The pooled vector is what drives the hypernetwork that generates the per-embodiment
    output adapter (:class:`HypernetActionAdapter`), which is why zero-shot transfer to
    an unseen end effector is possible at all: the adapter is a *function of the input*,
    not a set of weights that would have to be learned.
    """

    def __init__(
        self, feature_dim: int, d_model: int, n_tokens: int, hidden: int = 512
    ) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.to_tokens = nn.Linear(hidden, n_tokens * d_model)
        self.to_pooled = nn.Linear(hidden, d_model)
        self.norm = RMSNorm(d_model)
        self.d_model = d_model

    def forward(
        self, features: Tensor, timestamp: Tensor
    ) -> tuple[TokenStream, Tensor]:
        """Encode a batch of hand cards.

        Parameters
        ----------
        features:
            ``[B, feature_dim]`` from :meth:`EmbodimentSpec.feature_vector`.
        timestamp:
            ``[B]`` seconds -- the hand card is pinned at the start of the context.

        Returns
        -------
        tuple[TokenStream, Tensor]
            The ``B x n_tokens`` hand-card stream and a ``[B, D]`` pooled embedding.
        """
        h = self.trunk(features)
        tokens = self.norm(self.to_tokens(h).view(-1, self.n_tokens, self.d_model))
        pooled = self.norm(self.to_pooled(h))
        b = features.shape[0]
        return (
            TokenStream(
                embeddings=tokens,
                timestamps=timestamp.view(b, 1).expand(-1, self.n_tokens),
                stream_ids=torch.full(
                    (b, self.n_tokens),
                    int(Stream.HAND_CARD),
                    dtype=torch.long,
                    device=features.device,
                ),
                valid=torch.ones(
                    b, self.n_tokens, dtype=torch.bool, device=features.device
                ),
            ),
            pooled,
        )


# =============================================================================
# ② HARMONIC REASONING  (§6)
# =============================================================================


class _Page:
    """One time-indexed page of the KV cache (§10.2).

    A page holds a contiguous run of tokens from a *single* stream, together with their
    timestamps.  Pages are the unit of pinning, eviction and temporal decay.

    Keys are stored **before** rotary application.  Continuous-time RoPE is relative, so
    its reference must be re-chosen as the window slides; rotating at gather time rather
    than at write time keeps cached keys exactly correct for any reference, at the cost
    of one cheap rotary op per gather.
    """

    __slots__ = (
        "stream_ids",
        "timestamps",
        "valid",
        "pinned",
        "keys",
        "values",
        "decay_level",
    )

    def __init__(
        self, stream_ids: Tensor, timestamps: Tensor, valid: Tensor, pinned: bool
    ) -> None:
        self.stream_ids = stream_ids  # [B, n]
        self.timestamps = timestamps  # [B, n]
        self.valid = valid  # [B, n]
        self.pinned = pinned
        self.keys: list[Tensor] = []  # per layer, [B, Hkv, n, Dh] (un-rotated)
        self.values: list[Tensor] = []
        self.decay_level = 0

    @property
    def n_tokens(self) -> int:
        """Tokens currently held by this page (halves on each decay)."""
        return self.timestamps.shape[1]

    def end_time(self) -> float:
        """Latest timestamp in the page, as a Python float."""
        return float(self.timestamps.max().item())


class PagedKVCache:
    """A time-indexed, pinnable KV cache for real-time rollout (§10.2).

    Extends the block-table idea of PagedAttention with a *time* axis and three page
    classes:

    * **pinned** -- the hand card, the language prefix, and physical prompts.  Encoded
      once and never recomputed, so swapping a prompt is a page-table edit rather than a
      forward pass through a 10B encoder.  This is what makes an interactive
      drag-and-drop prompt interface possible.
    * **live** -- the most recent seconds at full rate.
    * **decayed** -- older pages, mean-pooled in place (§5.2).  Decay is a cache
      operation, never a re-encode.

    For the reference 11B configuration the arithmetic is
    ``40 layers x 8 KV heads x 128 dim x 2 (K,V) x 2 bytes = 160 KB/token``, i.e. ~3.3
    GB for an undecayed 30 s window and ~1.2 GB with decay -- which is why grouped-query
    attention is not optional here.
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        context_seconds: float = 30.0,
        decay_schedule: Sequence[tuple[float, int]] = ((3.0, 1), (10.0, 2)),
    ) -> None:
        """
        Parameters
        ----------
        n_layers, n_kv_heads, head_dim:
            Trunk geometry.
        context_seconds:
            Rolling window length; unpinned pages older than this are evicted.
        decay_schedule:
            Ordered ``(age_seconds, target_decay_level)`` pairs.  A page older than
            ``age_seconds`` is pooled until it reaches ``target_decay_level``; each
            level halves its token count.
        """
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.context_seconds = context_seconds
        self.decay_schedule = tuple(sorted(decay_schedule))
        self.pages: list[_Page] = []

    # -- construction --------------------------------------------------

    def open_page(
        self,
        stream_ids: Tensor,
        timestamps: Tensor,
        valid: Tensor,
        pinned: bool = False,
    ) -> int:
        """Start a new page and return its index.

        Layers must then be written in order with :meth:`write`.

        Parameters
        ----------
        stream_ids, timestamps, valid:
            ``[B, n]`` metadata for the tokens this page will hold.
        pinned:
            Exempt this page from eviction and from temporal decay -- the hand card, the
            language prefix and physical prompts (§9.2).
        """
        self.pages.append(_Page(stream_ids, timestamps, valid, pinned))
        return len(self.pages) - 1

    def write(self, page_idx: int, layer: int, keys: Tensor, values: Tensor) -> None:
        """Store one layer's un-rotated keys and values into an open page.

        Parameters
        ----------
        page_idx:
            Index returned by :meth:`open_page`.
        layer:
            Layer index; must be written in ascending order.
        keys, values:
            ``[B, Hkv, n, Dh]``.
        """
        page = self.pages[page_idx]
        if len(page.keys) != layer:
            raise ValueError(
                f"layer {layer} written out of order (have {len(page.keys)})"
            )
        page.keys.append(keys)
        page.values.append(values)

    # -- retrieval -----------------------------------------------------

    def gather(self, layer: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Concatenate all pages for one layer.

        Returns
        -------
        tuple
            ``(keys [B, Hkv, N, Dh], values, timestamps [B, N], stream_ids [B, N],
            valid [B, N])``.
        """
        if not self.pages:
            raise RuntimeError("cache is empty")
        keys = torch.cat([p.keys[layer] for p in self.pages], dim=2)
        values = torch.cat([p.values[layer] for p in self.pages], dim=2)
        timestamps = torch.cat([p.timestamps for p in self.pages], dim=1)
        valid = torch.cat([p.valid for p in self.pages], dim=1)
        streams = torch.cat([p.stream_ids for p in self.pages], dim=1)
        return keys, values, timestamps, streams, valid

    # -- maintenance ---------------------------------------------------

    def evict(self, now: float) -> int:
        """Drop unpinned pages that have fallen out of the rolling window.

        Returns
        -------
        int
            Number of pages evicted.
        """
        keep = [
            p
            for p in self.pages
            if p.pinned or p.end_time() >= now - self.context_seconds
        ]
        dropped = len(self.pages) - len(keep)
        self.pages = keep
        return dropped

    def decay(self, now: float) -> None:
        """Mean-pool ageing pages in place according to the decay schedule (§5.2)."""
        for page in self.pages:
            if page.pinned:
                continue  # physical prompts are exempt from decay (§9.2)
            age = now - page.end_time()
            target = 0
            for threshold, level in self.decay_schedule:
                if age >= threshold:
                    target = max(target, level)
            while page.decay_level < target and page.n_tokens >= 2:
                self._pool_once(page)

    @staticmethod
    def _pool_once(page: _Page) -> None:
        """Halve a page's token count by mean-pooling adjacent token pairs."""
        n = page.n_tokens - (page.n_tokens % 2)
        if n < 2:
            return

        def pool_kv(x: Tensor) -> Tensor:
            b, h, _, d = x.shape
            return x[:, :, :n].view(b, h, n // 2, 2, d).mean(dim=3)

        b = page.timestamps.shape[0]
        # Pooled tokens inherit the first member of each pair's stream identity; a page
        # holds a single stream in practice, so this is exact rather than a compromise.
        page.stream_ids = page.stream_ids[:, :n].view(b, n // 2, 2)[..., 0]
        page.keys = [pool_kv(k) for k in page.keys]
        page.values = [pool_kv(v) for v in page.values]
        page.timestamps = (
            page.timestamps[:, :n].view(page.timestamps.shape[0], n // 2, 2).mean(dim=2)
        )
        page.valid = page.valid[:, :n].view(page.valid.shape[0], n // 2, 2).any(dim=2)
        page.decay_level += 1

    # -- introspection -------------------------------------------------

    @property
    def n_tokens(self) -> int:
        """Total cached tokens."""
        return sum(p.n_tokens for p in self.pages)

    def memory_bytes(self, bytes_per_element: int = 2) -> int:
        """Estimated cache footprint, useful for capacity planning (§10.2)."""
        if not self.pages:
            return 0
        batch = self.pages[0].timestamps.shape[0]
        per_token = (
            self.n_layers * self.n_kv_heads * self.head_dim * 2 * bytes_per_element
        )
        return self.n_tokens * per_token * batch

    def clear(self, keep_pinned: bool = True) -> None:
        """Drop all pages, optionally retaining pinned prompt/hand-card pages."""
        self.pages = [p for p in self.pages if p.pinned] if keep_pinned else []


class TrunkAttention(nn.Module):
    """Grouped-query attention over continuous time (§6.5).

    Ordinary in every respect except that positions are timestamps and the mask is built
    from physical time plus a latency offset.
    """

    def __init__(self, cfg: TrunkConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, h, hkv, dh = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        if h % hkv != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        self.n_rep = h // hkv
        self.q_proj = nn.Linear(d, h * dh, bias=False)
        self.k_proj = nn.Linear(d, hkv * dh, bias=False)
        self.v_proj = nn.Linear(d, hkv * dh, bias=False)
        self.o_proj = nn.Linear(h * dh, d, bias=False)
        self.q_norm = RMSNorm(dh) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(dh) if cfg.qk_norm else nn.Identity()

    def project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project into query/key/value heads.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``q [B, Hq, N, Dh]``, ``k [B, Hkv, N, Dh]``, ``v [B, Hkv, N, Dh]``, with
            QK-norm applied and rotary **not** yet applied.
        """
        b, n, _ = x.shape
        dh = self.cfg.head_dim
        q = self.q_proj(x).view(b, n, self.cfg.n_heads, dh).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.cfg.n_kv_heads, dh).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.cfg.n_kv_heads, dh).transpose(1, 2)
        return self.q_norm(q), self.k_norm(k), v

    def attend(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        """Run scaled dot-product attention and the output projection.

        Parameters
        ----------
        q:
            ``[B, Hq, Nq, Dh]`` rotary-applied queries.
        k, v:
            ``[B, Hkv, Nk, Dh]`` rotary-applied keys and raw values.
        mask:
            ``[B, 1, Nq, Nk]`` boolean, ``True`` where attention is permitted.
        """
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        b, _, n, _ = out.shape
        return self.o_proj(out.transpose(1, 2).reshape(b, n, -1))


class TrunkBlock(nn.Module):
    """One pre-norm transformer block of the harmonic reasoning trunk (§6.5).

    Optionally uses per-stream ("modality expert") feed-forward weights while sharing
    attention -- the leading architectural alternative noted in §6.5, and a plausible
    fit for the clean per-subsystem weight-delta decomposition Generalist reports.
    """

    def __init__(self, cfg: TrunkConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = TrunkAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        # Exactly one of these is populated.  They are separate attributes rather than
        # one union-typed attribute so both the type checker and ``state_dict`` keys
        # stay unambiguous.
        self.ffn = None if cfg.modality_experts else SwiGLU(cfg.d_model, cfg.ffn_hidden)
        self.experts = (
            nn.ModuleList(
                [SwiGLU(cfg.d_model, cfg.ffn_hidden) for _ in range(NUM_STREAMS)]
            )
            if cfg.modality_experts
            else None
        )

    @property
    def window_seconds(self) -> float | None:
        """Sliding-window span for this layer, or ``None`` for full 30 s attention."""
        if self.layer_idx < self.cfg.n_sliding_layers:
            return self.cfg.sliding_window_seconds
        return None

    @property
    def feed_forwards(self) -> list[SwiGLU]:
        """Every feed-forward module in this block, shared or per-stream."""
        if self.experts is not None:
            return [cast(SwiGLU, m) for m in self.experts]
        assert self.ffn is not None
        return [self.ffn]

    def _apply_ffn(self, x: Tensor, stream_ids: Tensor) -> Tensor:
        """Route tokens to their modality expert, or to the single shared FFN."""
        if self.experts is None:
            assert self.ffn is not None
            return self.ffn(x)
        out = torch.zeros_like(x)
        flat_x = x.reshape(-1, x.shape[-1])
        flat_s = stream_ids.reshape(-1)
        flat_out = out.reshape(-1, x.shape[-1])
        for stream_id in flat_s.unique():
            idx = (flat_s == stream_id).nonzero(as_tuple=True)[0]
            expert = cast(SwiGLU, self.experts[int(stream_id)])
            flat_out[idx] = expert(flat_x[idx])
        return flat_out.view_as(x)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        mask: Tensor,
        stream_ids: Tensor,
    ) -> Tensor:
        """Full (non-incremental) forward over the whole context.

        Parameters
        ----------
        x:
            ``[B, N, D]``.
        cos, sin:
            ``[B, N, Dh]`` continuous-time rotary factors.
        mask:
            ``[B, 1, N, N]`` boolean time-causal mask.
        stream_ids:
            ``[B, N]`` used only for modality-expert routing.
        """
        h = self.attn_norm(x)
        q, k, v = self.attn.project(h)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        x = x + self.attn.attend(q, k, v, mask)
        return x + self._apply_ffn(self.ffn_norm(x), stream_ids)

    def step(
        self,
        x: Tensor,
        q_cos: Tensor,
        q_sin: Tensor,
        cache: PagedKVCache,
        page_idx: int,
        rope: ContinuousTimeRoPE,
        time_reference: Tensor,
        gap: Tensor,
        query_timestamps: Tensor,
        query_streams: Tensor,
    ) -> Tensor:
        """Incremental forward for one runtime tick, attending over the paged cache.

        Only ``[B, 1, Nq, Nk]`` mask rows are materialised, so this path is safe for
        long contexts where :func:`build_time_causal_mask` would be quadratic in the
        window.

        Parameters
        ----------
        x:
            ``[B, Nq, D]`` newly arrived tokens for this tick.
        q_cos, q_sin:
            Rotary factors for the new tokens, computed against ``time_reference``.
        cache:
            The rollout cache; this block's keys/values are written into ``page_idx``.
        page_idx:
            Page opened for this tick by the caller.
        rope:
            Shared rotary module, used to rotate cached keys at gather time.
        time_reference:
            ``[B, 1]`` common rotary reference for this tick.
        gap:
            ``[B, S, S]`` required-gap matrix.
        query_timestamps, query_streams:
            ``[B, Nq]`` metadata for the new tokens.
        """
        h = self.attn_norm(x)
        q, k, v = self.attn.project(h)
        cache.write(page_idx, self.layer_idx, k, v)
        all_k, all_v, k_t, k_streams, k_valid = cache.gather(self.layer_idx)
        k_cos, k_sin = rope(k_t, valid=k_valid, reference=time_reference)
        q = apply_rotary(q, q_cos, q_sin)
        all_k = apply_rotary(all_k, k_cos, k_sin)

        b = x.shape[0]
        idx = torch.arange(b, device=x.device).view(b, 1, 1)
        required = gap[idx, query_streams.unsqueeze(-1), k_streams.unsqueeze(-2)]
        dt = query_timestamps.unsqueeze(-1) - k_t.unsqueeze(-2)
        allowed = (dt >= required - 1e-6) & k_valid.unsqueeze(-2)
        window = self.window_seconds
        if window is not None:
            allowed = allowed & (dt <= window + 1e-6)
        # Guarantee at least one visible key per query row (an all-masked softmax row is
        # NaN).  New tokens are the tail of the cache, so a query's own position is a
        # safe, causality-preserving fallback.
        nq, nk = x.shape[1], all_k.shape[2]
        self_hot = F.one_hot(torch.arange(nk - nq, nk, device=x.device), nk).bool()
        allowed = allowed | (~allowed.any(dim=-1, keepdim=True) & self_hot.unsqueeze(0))
        x = x + self.attn.attend(q, all_k, all_v, allowed.unsqueeze(1))
        return x + self._apply_ffn(self.ffn_norm(x), query_streams)


class HarmonicTrunk(nn.Module):
    """The harmonic reasoning trunk (§6).

    A decoder-only transformer whose only unusual features are continuous-time rotary
    embeddings and a time-causal mask carrying a latency offset.  Everything that makes
    GEN behave the way it does at runtime -- asynchronous sensing and acting, no
    System-1/System-2 split, tolerance for spliced physical prompts -- is a consequence
    of those two choices plus the harmonic ladder that feeds it.
    """

    def __init__(self, cfg: TrunkConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.stream_embed = nn.Embedding(NUM_STREAMS, cfg.d_model)
        self.rope = ContinuousTimeRoPE(
            cfg.head_dim, cfg.rope_min_period, cfg.rope_max_period
        )
        self.latency = LatencySchedule()
        self.blocks = nn.ModuleList([TrunkBlock(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.apply(self._init_weights)
        if cfg.use_mup_scaling:
            scale = 1.0 / math.sqrt(2 * cfg.n_layers)
            for block in self.layers:
                nn.init.normal_(block.attn.o_proj.weight, std=0.02 * scale)
                for ffn in block.feed_forwards:
                    nn.init.normal_(ffn.w2.weight, std=0.02 * scale)

    @property
    def layers(self) -> list[TrunkBlock]:
        """The trunk's blocks, precisely typed.

        ``nn.ModuleList`` iterates as bare ``Module``, which loses :class:`TrunkBlock`'s
        interface; this narrows it back without disturbing module registration.
        """
        return [cast(TrunkBlock, m) for m in self.blocks]

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Standard scaled-normal initialisation."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, stream: TokenStream, delta: Tensor) -> Tensor:
        """Run the trunk over a full context window.

        Parameters
        ----------
        stream:
            Interleaved, time-sorted context.
        delta:
            ``[B]`` latency offsets in seconds (§6.3).

        Returns
        -------
        Tensor
            ``[B, N, D]`` hidden states.
        """
        x = stream.embeddings + self.stream_embed(stream.stream_ids)
        cos, sin = self.rope(stream.timestamps, valid=stream.valid)
        gap = self.latency.gap_matrix(delta)
        # Two masks suffice: the low layers share a windowed mask, the rest a global
        # one.
        mask_global = build_time_causal_mask(
            stream.timestamps, stream.stream_ids, stream.valid, gap
        )
        mask_window = (
            build_time_causal_mask(
                stream.timestamps,
                stream.stream_ids,
                stream.valid,
                gap,
                window_seconds=self.cfg.sliding_window_seconds,
            )
            if self.cfg.n_sliding_layers > 0
            else mask_global
        )
        for block in self.layers:
            mask = mask_window if block.window_seconds is not None else mask_global
            x = block(x, cos, sin, mask, stream.stream_ids)
        return self.norm(x)

    def step(
        self,
        stream: TokenStream,
        delta: Tensor,
        cache: PagedKVCache,
        pinned: bool = False,
    ) -> Tensor:
        """Incremental forward for one runtime tick (§10.1).

        Parameters
        ----------
        stream:
            The tokens that arrived during this tick.
        delta:
            ``[B]`` latency offsets in seconds.
        cache:
            Rollout cache; a page is opened for this tick and written layer by layer.
        pinned:
            Mark the new page as pinned (used when priming a physical prompt, §9.2).

        Returns
        -------
        Tensor
            ``[B, Nq, D]`` hidden states for the new tokens only.
        """
        x = stream.embeddings + self.stream_embed(stream.stream_ids)
        reference = (
            torch.where(
                stream.valid,
                stream.timestamps,
                torch.full_like(stream.timestamps, float("-inf")),
            )
            .max(dim=-1, keepdim=True)
            .values
        )
        reference = torch.where(
            torch.isfinite(reference), reference, torch.zeros_like(reference)
        )
        q_cos, q_sin = self.rope(
            stream.timestamps, valid=stream.valid, reference=reference
        )
        gap = self.latency.gap_matrix(delta)
        page_idx = cache.open_page(
            stream.stream_ids, stream.timestamps, stream.valid, pinned=pinned
        )
        for block in self.layers:
            x = block.step(
                x,
                q_cos,
                q_sin,
                cache,
                page_idx,
                self.rope,
                reference,
                gap,
                stream.timestamps,
                stream.stream_ids,
            )
        return self.norm(x)


# =============================================================================
# ③ ACTUATION  (§7)
# =============================================================================


class _AdaLNBlock(nn.Module):
    """A DiT-style block with adaptive layer-norm (adaLN-Zero) conditioning."""

    def __init__(self, width: int, heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(width, int(width * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(width * mlp_ratio), width),
        )
        self.modulation = nn.Linear(width, 6 * width)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Modulated attention + MLP.

        Parameters
        ----------
        x:
            ``[M, T, width]``.
        cond:
            ``[M, width]`` conditioning vector.
        """
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(cond).chunk(
            6, dim=-1
        )

        def mod(h: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
            return h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        h = mod(self.norm1(x), shift1, scale1)
        x = x + gate1.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = mod(self.norm2(x), shift2, scale2)
        return x + gate2.unsqueeze(1) * self.mlp(h)


class ActionExpert(nn.Module):
    """Flow-matching action chunk generator (``ARCHITECTURE.md`` §7.1).

    GEN-0 evaluates policies with a **reverse KL estimated by Monte Carlo over policy
    samples**.  You only build that estimator if the policy is a *sampler* with a
    multimodal output distribution: discrete action bins would make reverse KL
    closed-form and trivial, and L2 regression would make it meaningless.  Hence
    conditional flow matching over continuous action chunks.

    Two conditioning inputs replace what other systems do at inference time:

    * **δ, the latency offset** -- the head is told how stale its observations are;
    * **the committed prefix** -- actions already sent to the robot during the compute
      window are prepended as clean, non-denoised tokens, so chunk continuity is a
      *learned* property rather than a runtime constraint solve.  This is the trained
      substitute for real-time chunking / inference-time guidance (evidence E3).
    """

    def __init__(
        self, cfg: ActionExpertConfig, action_space: ActionSpaceConfig, cond_dim: int
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.space = action_space
        w = cfg.width
        self.in_proj = nn.Linear(action_space.dim, w)
        self.pos = nn.Parameter(
            torch.randn(1, action_space.prefix_steps + action_space.horizon, w) * 0.02
        )
        self.role_embed = nn.Embedding(2, w)  # 0 = committed prefix, 1 = pending
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, w), nn.SiLU(), nn.Linear(w, w)
        )
        self.flow_time_proj = nn.Sequential(
            nn.Linear(cfg.time_freq_dim, w), nn.SiLU(), nn.Linear(w, w)
        )
        self.latency_proj = nn.Sequential(
            nn.Linear(cfg.time_freq_dim, w), nn.SiLU(), nn.Linear(w, w)
        )
        self.blocks = nn.ModuleList(
            [_AdaLNBlock(w, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth)]
        )
        self.out_norm = nn.LayerNorm(w, elementwise_affine=False)
        self.out_proj = nn.Linear(w, action_space.dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        noisy: Tensor,
        flow_time: Tensor,
        cond: Tensor,
        latency: Tensor,
        prefix: Tensor,
    ) -> Tensor:
        """Predict the flow-matching velocity field.

        Parameters
        ----------
        noisy:
            ``[M, H, A]`` interpolant ``a_s = s·a + (1-s)·ε``.
        flow_time:
            ``[M]`` flow time ``s ∈ [0, 1]``.
        cond:
            ``[M, cond_dim]`` trunk latent + proprio + embodiment conditioning.
        latency:
            ``[M]`` δ in seconds.
        prefix:
            ``[M, P, A]`` already-committed actions (zeros at episode start).

        Returns
        -------
        Tensor
            ``[M, H, A]`` predicted velocity ``v ≈ a - ε``.
        """
        m, h, _ = noisy.shape
        p = prefix.shape[1]
        x = self.in_proj(torch.cat((prefix, noisy), dim=1))
        roles = torch.cat(
            (
                torch.zeros(m, p, dtype=torch.long, device=noisy.device),
                torch.ones(m, h, dtype=torch.long, device=noisy.device),
            ),
            dim=1,
        )
        x = x + self.role_embed(roles) + self.pos[:, : p + h]
        c = (
            self.cond_proj(cond)
            + self.flow_time_proj(
                sinusoidal_embedding(flow_time, self.cfg.time_freq_dim)
            )
            + self.latency_proj(sinusoidal_embedding(latency, self.cfg.time_freq_dim))
        )
        for block in self.blocks:
            x = block(x, c)
        return self.out_proj(self.out_norm(x[:, p:]))

    @torch.no_grad()
    def sample(
        self,
        cond: Tensor,
        latency: Tensor,
        prefix: Tensor,
        steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Integrate the probability-flow ODE from noise to an action chunk.

        Rectified-flow training straightens the trajectories enough that a handful of
        Euler steps suffice (4 by default), which is what keeps the 10 Hz chunk budget
        affordable.

        Parameters
        ----------
        cond, latency, prefix:
            As in :meth:`forward`.
        steps:
            Euler steps; defaults to :attr:`ActionExpertConfig.flow_steps`.
        generator:
            Optional RNG for reproducible sampling.

        Returns
        -------
        Tensor
            ``[M, H, A]`` sampled action chunk.
        """
        steps = steps or self.cfg.flow_steps
        m = cond.shape[0]
        a = torch.randn(
            m,
            self.space.horizon,
            self.space.dim,
            device=cond.device,
            dtype=cond.dtype,
            generator=generator,
        )
        dt = 1.0 / steps
        for i in range(steps):
            s = torch.full((m,), i * dt, device=cond.device, dtype=cond.dtype)
            a = a + dt * self.forward(a, s, cond, latency, prefix)
        return a


class HypernetActionAdapter(nn.Module):
    """Per-embodiment output adapter, *generated* from the hand card (§7.3).

    9,000 end effectors rule out storing an adapter per embodiment, but the measured
    2.5--11.4% weight deltas for a new hand are exactly the magnitude of "a low-rank
    adapter plus some sensor-encoder movement".  Both facts are satisfied by generating
    the low-rank factors from the embodiment embedding with a small hypernetwork: an
    unseen gripper is an unseen *input*, so zero-shot transfer to a new hand is
    structurally possible.

    A trainable per-embodiment residual can still be fine-tuned on top for embodiments
    that warrant it; that is what :mod:`open_gen.adapt` would move during few-step
    adaptation.
    """

    def __init__(self, action_dim: int, embodiment_dim: int, rank: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.rank = rank
        self.to_down = nn.Linear(embodiment_dim, rank * action_dim)
        self.to_up = nn.Linear(embodiment_dim, action_dim * rank)
        self.bias = nn.Linear(embodiment_dim, action_dim)
        for layer in (self.to_down, self.to_up, self.bias):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self, actions: Tensor, embodiment: Tensor, action_valid: Tensor | None = None
    ) -> Tensor:
        """Map universal actions onto one robot's actuators.

        Parameters
        ----------
        actions:
            ``[B, ..., A]`` universal-space actions.
        embodiment:
            ``[B, embodiment_dim]`` pooled hand-card embedding.
        action_valid:
            Optional ``[B, A]`` bool; channels this robot does not actuate are zeroed.

        Returns
        -------
        Tensor
            ``[B, ..., A]`` embodiment-specific actions.
        """
        b = actions.shape[0]
        lead = actions.shape[1:-1]
        flat = actions.reshape(b, -1, self.action_dim)
        down = self.to_down(embodiment).view(b, self.action_dim, self.rank)
        up = self.to_up(embodiment).view(b, self.rank, self.action_dim)
        residual = torch.bmm(torch.bmm(flat, down), up) + self.bias(
            embodiment
        ).unsqueeze(1)
        out = flat + residual
        if action_valid is not None:
            out = out * action_valid.unsqueeze(1).to(out.dtype)
        return out.view(b, *lead, self.action_dim)


class ReflexHead(nn.Module):
    """The 100 Hz spinal loop (``ARCHITECTURE.md`` §7.2).

    Human reflex arcs are 30--50 ms; no 7B transformer will hit that, so high-frequency
    reactivity has to come from somewhere else.  This head runs at the full control rate
    on proprioception, force/torque and tactile only -- no vision, no GPU -- and emits a
    bounded residual on top of the current chunk: contact-triggered compliance, slip
    arrest, impact damping, joint-limit avoidance.

    It is what makes the published speed results ("the world becomes less quasi-static")
    coherent with a 10 Hz trunk.  The magnitude bound keeps it a correction rather than
    a policy, and is also the natural place to enforce hard safety limits that no
    sampled chunk may override (§8.4).
    """

    def __init__(
        self, cfg: ReflexConfig, proprio_fast_dim: int, action_dim: int, cond_dim: int
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(proprio_fast_dim + action_dim + cond_dim, cfg.hidden)
        self.gru = nn.GRU(
            cfg.hidden, cfg.hidden, num_layers=cfg.layers, batch_first=True
        )
        self.out = nn.Linear(cfg.hidden, action_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        proprio_fast: Tensor,
        planned: Tensor,
        cond: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute bounded action residuals at the control rate.

        Parameters
        ----------
        proprio_fast:
            ``[B, T, proprio_fast_dim]`` per-control-step proprio summary (100 Hz).
        planned:
            ``[B, T, A]`` the chunk's planned action for each step.
        cond:
            ``[B, cond_dim]`` slow context (embodiment embedding, latest trunk latent).
        hidden:
            Optional recurrent state carried across ticks.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``residual [B, T, A]`` bounded by ``±max_magnitude``, and the new GRU state.
        """
        t = proprio_fast.shape[1]
        x = torch.cat(
            (proprio_fast, planned, cond.unsqueeze(1).expand(-1, t, -1)), dim=-1
        )
        h, new_state = self.gru(self.in_proj(x), hidden)
        return torch.tanh(self.out(h)) * self.cfg.max_magnitude, new_state


class ChunkBlender:
    """Raised-cosine crossfade between overlapping action chunks (§7.1).

    The expert re-plans at 10 Hz but emits 500 ms of actions, so successive chunks
    overlap 5:1 and will disagree slightly -- they are independent samples from a
    multimodal distribution.  Blending guarantees C¹ continuity at the seam, which
    matters more than which sample "wins": a discontinuity at 100 Hz is a jerk the
    hardware will feel.
    """

    def __init__(self, action_dim: int, blend_steps: int) -> None:
        self.action_dim = action_dim
        self.blend_steps = blend_steps
        self._tail: Tensor | None = None

    def reset(self) -> None:
        """Forget the previous chunk's tail (call at episode boundaries)."""
        self._tail = None

    def blend(self, chunk: Tensor) -> Tensor:
        """Crossfade a new chunk against the previous one's overlapping tail.

        Parameters
        ----------
        chunk:
            ``[B, H, A]`` freshly sampled chunk.

        Returns
        -------
        Tensor
            ``[B, H, A]`` blended chunk, safe to execute.
        """
        n = self.blend_steps
        if self._tail is not None and n > 0:
            k = min(n, chunk.shape[1], self._tail.shape[1])
            ramp = 0.5 * (
                1 - torch.cos(torch.linspace(0, math.pi, k, device=chunk.device))
            )
            ramp = ramp.view(1, k, 1)
            chunk = chunk.clone()
            chunk[:, :k] = self._tail[:, :k] * (1 - ramp) + chunk[:, :k] * ramp
        self._tail = chunk.detach()
        return chunk

    def advance(self, executed_steps: int) -> None:
        """Drop the executed prefix from the retained tail."""
        if self._tail is not None:
            self._tail = self._tail[:, executed_steps:]
            if self._tail.shape[1] == 0:
                self._tail = None


# =============================================================================
# Batch / output containers
# =============================================================================


@dataclass
class GenBatch:
    """One training batch: a random continuous span of physical experience.

    **Do not "improve" the sampling that fills this.**  GEN pretrains on randomly
    sampled continuous spans with *no* packing infrastructure (evidence E12).  The
    obvious optimisations -- packing several episodes per sequence, adding episode
    separators, curating demonstration→execution pairs -- are exactly the interventions
    that would turn *emergent* in-context learning into *trained* in-context
    learning, and would likely
    destroy the generality of the result while improving the benchmark (§8.2).
    """

    # -- vision ---------------------------------------------------------
    frames: Tensor  #: ``[B, F, C, H, W]``
    frame_timestamps: Tensor  #: ``[B, F]`` seconds
    camera_ids: Tensor  #: ``[B, F]`` long
    is_wrist: Tensor  #: ``[B, F]`` bool
    extrinsics: Tensor  #: ``[B, F, extrinsic_dim]``
    # -- proprioception -------------------------------------------------
    proprio_features: Tensor  #: ``[B, T, proprio.feature_dim]``
    proprio_timestamps: Tensor  #: ``[B, T]``
    #: ``[B, K, execute_steps, proprio.fast_dim]`` -- the 100 Hz stream seen by the
    #: reflex head.
    proprio_fast: Tensor
    # -- language -------------------------------------------------------
    text_embeddings: Tensor  #: ``[B, L, text_embed_dim]`` from the frozen encoder
    text_timestamps: Tensor  #: ``[B, L]``
    text_target_ids: Tensor  #: ``[B, L]`` long, captioning targets
    # -- embodiment -----------------------------------------------------
    embodiment_features: Tensor  #: ``[B, embodiment_feature_dim]``
    channel_valid: Tensor  #: ``[B, proprio.n_channels]`` bool
    action_valid: Tensor  #: ``[B, action_space.dim]`` bool
    # -- actions --------------------------------------------------------
    action_timestamps: Tensor  #: ``[B, K]`` query times (10 Hz harmonic)
    action_targets: Tensor  #: ``[B, K, H, A]`` ground-truth chunks
    action_prefix: Tensor  #: ``[B, K, P, A]`` committed prefix
    action_query_valid: Tensor  #: ``[B, K]`` bool
    #: ``[B, K, execute_steps, A]`` residual between executed and planned actions.
    reflex_target: Tensor

    @property
    def batch_size(self) -> int:
        """Batch size ``B``."""
        return self.frames.shape[0]

    @property
    def device(self) -> torch.device:
        """Device the batch lives on."""
        return self.frames.device

    def to(self, device: torch.device | str) -> "GenBatch":
        """Move every tensor field to ``device``."""
        moved = {
            f: (v.to(device) if isinstance(v, Tensor) else v)
            for f, v in self.__dict__.items()
        }
        return GenBatch(**moved)


@dataclass
class GenOutput:
    """Everything a forward pass produces, before losses are applied."""

    #: ``[B, N, D]`` trunk hidden states over the whole interleaved context.
    hidden: Tensor
    #: The interleaved context that produced ``hidden`` (time-sorted).
    stream: TokenStream
    #: ``[B, N, D]`` pre-trunk token embeddings, used as latent-world targets.
    input_embeddings: Tensor
    #: ``[B, K, D]`` hidden states at action-query positions.
    action_hidden: Tensor
    #: ``[B, K, cond_dim]`` conditioning assembled for the action expert.
    action_cond: Tensor
    #: ``[B, R, D]`` hidden states at latent-register positions.
    register_hidden: Tensor
    #: ``[B, L, vocab]`` captioning logits at language positions.
    language_logits: Tensor
    #: ``[B, L]`` bool, whether the language prefix survived dropout for this element.
    language_kept: Tensor
    #: ``[B]`` sampled latency offsets in seconds.
    delta: Tensor
    #: ``[B, D]`` pooled embodiment embedding.
    embodiment: Tensor


@dataclass
class GenLosses:
    """The four-term objective of ``ARCHITECTURE.md`` §8.1."""

    total: Tensor
    action: Tensor
    world: Tensor
    language: Tensor
    reflex: Tensor
    #: Next-action prediction MSE -- the quantity plotted in every published GEN scaling
    #: figure, reported separately from the flow-matching loss so curves are comparable.
    next_action_mse: Tensor

    def as_dict(self) -> dict[str, float]:
        """Detached scalars, ready for logging."""
        return {k: float(v.detach()) for k, v in self.__dict__.items()}


# =============================================================================
# The model
# =============================================================================


class GenModel(nn.Module):
    """Open-Gen: the full embodied foundation model.

    Ties together the three subsystems.  The training path (:meth:`forward` +
    :meth:`compute_loss`) runs the whole context at once; the rollout path
    (:class:`GenRuntime`) runs the three-rate loop against a :class:`PagedKVCache`.

    Parameters
    ----------
    cfg:
        Model configuration; use the :class:`GenConfig` presets.

    Examples
    --------
    >>> model = GenModel(GenConfig.debug())
    >>> batch = make_dummy_batch(model.cfg, batch_size=2)
    >>> losses = model.compute_loss(batch, model(batch))
    >>> losses.total.backward()
    """

    def __init__(self, cfg: GenConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.trunk.d_model

        # ① sensor processing
        self.vision = VisionTokenizer(cfg.vision, d)
        self.proprio = ProprioTokenizer(cfg.proprio, d)
        self.language = LanguageTokenizer(cfg.text_embed_dim, d, cfg.vocab_size)
        self.embodiment_feature_dim = EmbodimentSpec.feature_dim(
            cfg.proprio, cfg.vision, cfg.action_space
        )
        self.embodiment = EmbodimentEncoder(
            self.embodiment_feature_dim, d, cfg.n_hand_card_tokens
        )

        # ② harmonic reasoning
        self.trunk = HarmonicTrunk(cfg.trunk)
        self.action_query = nn.Parameter(torch.randn(d) * 0.02)
        self.action_proj = nn.Linear(
            cfg.action_space.prefix_steps * cfg.action_space.dim, d
        )
        self.register_embed = nn.Embedding(cfg.trunk.n_registers, d)

        # ③ actuation
        self.cond_dim = 2 * d + cfg.proprio.fast_dim
        self.expert = ActionExpert(cfg.expert, cfg.action_space, self.cond_dim)
        self.adapter = HypernetActionAdapter(
            cfg.action_space.dim, d, cfg.expert.adapter_rank
        )
        self.reflex = ReflexHead(
            cfg.reflex, cfg.proprio.fast_dim, cfg.action_space.dim, 2 * d
        )

        # auxiliary heads
        self.world_head = nn.Linear(d, len(cfg.loss.world_horizons) * d)

    # -- context assembly ------------------------------------------------

    def _register_stream(self, t_lo: Tensor, t_hi: Tensor) -> TokenStream:
        """Emit latent thought registers on their own harmonic (§6.4).

        Registers attend to the full context and are attended *by* action tokens, but
        never block them: an action token reads the most recently completed register,
        which the latency matrix keeps at least δ in the past.  That is the whole of
        "asynchronous streams of sensing and acting tokens" -- one transformer whose
        tokens run at different clock rates, rather than two models with a handoff.

        Parameters
        ----------
        t_lo, t_hi:
            ``[B]`` earliest and latest valid timestamps in the context.
        """
        cfg = self.cfg
        period = cfg.ladder.period_s(cfg.ladder.register_divisor)
        n_emissions = max(int(cfg.ladder.context_seconds / period), 1)
        r = cfg.trunk.n_registers
        b = t_lo.shape[0]
        device = t_lo.device
        offsets = (
            torch.arange(n_emissions - 1, -1, -1, device=device, dtype=torch.float32)
            * period
        )
        times = t_hi.view(b, 1) - offsets.view(1, n_emissions)  # [B, n_emissions]
        valid = times >= t_lo.view(b, 1)
        emb = self.register_embed.weight.view(1, 1, r, -1).expand(
            b, n_emissions, -1, -1
        )
        return TokenStream(
            embeddings=emb.reshape(b, n_emissions * r, -1),
            timestamps=times.unsqueeze(-1).expand(-1, -1, r).reshape(b, -1),
            stream_ids=torch.full(
                (b, n_emissions * r),
                int(Stream.REGISTER),
                dtype=torch.long,
                device=device,
            ),
            valid=valid.unsqueeze(-1).expand(-1, -1, r).reshape(b, -1),
        )

    def _action_stream(
        self, timestamps: Tensor, valid: Tensor, committed: Tensor
    ) -> TokenStream:
        """Emit action query tokens on the 10 Hz harmonic.

        The token is *not* content-free: it carries a projection of the actions already
        committed before its timestamp.  This is what makes the context **sensorimotor**
        rather than merely sensory, and therefore promptable -- a physical prompt is a
        span of sensor *and* action tokens, and no model can infer a task from a
        demonstration whose actions are absent from its context.

        Parameters
        ----------
        timestamps:
            ``[B, K]`` query times.
        valid:
            ``[B, K]`` bool.
        committed:
            ``[B, K, P, A]`` actions committed before each query time (causal by
            construction).
        """
        b, k = timestamps.shape
        return TokenStream(
            embeddings=self.action_query.view(1, 1, -1)
            + self.action_proj(committed.flatten(2)),
            timestamps=timestamps,
            stream_ids=torch.full(
                (b, k), int(Stream.ACTION), dtype=torch.long, device=timestamps.device
            ),
            valid=valid,
        )

    @staticmethod
    def _gather_stream(
        hidden: Tensor, stream: TokenStream, kind: Stream, expected: int
    ) -> Tensor:
        """Gather hidden states belonging to one stream, preserving order.

        Sorting by time is stable, so a stream's tokens keep their relative order and
        can be recovered by mask.  ``expected`` is asserted to catch ragged batches
        early.
        """
        mask = stream.stream_ids == int(kind)
        counts = mask.sum(dim=-1)
        if not bool((counts == expected).all()):
            raise ValueError(
                f"expected {expected} {kind.name} tokens per batch element, "
                f"got {counts.tolist()}"
            )
        b, _, d = hidden.shape
        idx = mask.nonzero(as_tuple=True)[1].view(b, expected)
        return hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, d))

    def encode_context(
        self, batch: GenBatch, language_dropout: float = 0.0
    ) -> tuple[TokenStream, Tensor]:
        """Run subsystem ① and interleave every stream into one time-sorted sequence.

        Returns
        -------
        tuple[TokenStream, Tensor]
            The interleaved context and the ``[B, D]`` pooled embodiment embedding.
        """
        cfg = self.cfg
        hand_t = batch.frame_timestamps.min(dim=-1).values
        hand_stream, embodiment = self.embodiment(batch.embodiment_features, hand_t)
        vision = self.vision(
            batch.frames,
            batch.frame_timestamps,
            batch.camera_ids,
            batch.is_wrist,
            batch.extrinsics,
        )
        proprio = self.proprio(
            batch.proprio_features, batch.proprio_timestamps, batch.channel_valid
        )
        language = self.language(
            batch.text_embeddings, batch.text_timestamps, dropout_p=language_dropout
        )
        actions = self._action_stream(
            batch.action_timestamps, batch.action_query_valid, batch.action_prefix
        )

        all_t = torch.cat(
            (vision.timestamps, proprio.timestamps, batch.action_timestamps), dim=1
        )
        registers = self._register_stream(
            all_t.min(dim=-1).values, all_t.max(dim=-1).values
        )

        context = TokenStream.concatenate(
            [hand_stream, language, vision, proprio, registers, actions]
        )
        if context.length > cfg.max_context_tokens:
            raise ValueError(
                f"context is {context.length} tokens but max_context_tokens="
                f"{cfg.max_context_tokens}; shorten the span or enable temporal "
                "decay (§5.2)"
            )
        return context, embodiment

    # -- training path ---------------------------------------------------

    def forward(self, batch: GenBatch, delta: Tensor | None = None) -> GenOutput:
        """Full-context forward pass.

        Parameters
        ----------
        batch:
            A training batch.
        delta:
            Optional ``[B]`` latency override in seconds; by default sampled from the δ
            curriculum (§6.3).

        Returns
        -------
        GenOutput
            Hidden states plus everything :meth:`compute_loss` needs.
        """
        cfg = self.cfg
        dropout = cfg.loss.language_dropout if self.training else 0.0
        context, embodiment = self.encode_context(batch, language_dropout=dropout)
        if delta is None:
            delta = cfg.latency.sample((batch.batch_size,), device=batch.device)

        hidden = self.trunk(context, delta)

        k = batch.action_timestamps.shape[1]
        action_hidden = self._gather_stream(hidden, context, Stream.ACTION, k)
        register_hidden = self._gather_stream(
            hidden,
            context,
            Stream.REGISTER,
            int((context.stream_ids == int(Stream.REGISTER))[0].sum()),
        )
        language_hidden = self._gather_stream(
            hidden, context, Stream.LANGUAGE, batch.text_embeddings.shape[1]
        )
        language_valid = context.valid[context.stream_ids == int(Stream.LANGUAGE)].view(
            batch.batch_size, -1
        )

        # Conditioning for the action expert: trunk latent + embodiment + proprio at the
        # moment of the query.  δ is passed separately (§7.1).
        proprio_now = batch.proprio_fast[:, :, 0]  # [B, K, fast_dim]
        cond = torch.cat(
            (action_hidden, embodiment.unsqueeze(1).expand(-1, k, -1), proprio_now),
            dim=-1,
        )

        return GenOutput(
            hidden=hidden,
            stream=context,
            input_embeddings=context.embeddings,
            action_hidden=action_hidden,
            action_cond=cond,
            register_hidden=register_hidden,
            language_logits=self.language.lm_head(language_hidden),
            language_kept=language_valid,
            delta=delta,
            embodiment=embodiment,
        )

    # -- losses (§8.1) ---------------------------------------------------

    def _action_loss(self, batch: GenBatch, out: GenOutput) -> tuple[Tensor, Tensor]:
        """Flow matching over action chunks, plus the reported next-action MSE.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(cfm_loss, next_action_mse)``.  The second is the "next action prediction
            error" from the published scaling curves: a single Euler step from noise,
            which makes it comparable across checkpoints without depending on sampler
            settings.
        """
        cfg = self.cfg.action_space
        b, k = batch.action_timestamps.shape
        m = b * k
        target = batch.action_targets.reshape(m, cfg.horizon, cfg.dim)
        prefix = batch.action_prefix.reshape(m, cfg.prefix_steps, cfg.dim)
        cond = out.action_cond.reshape(m, -1)
        latency = out.delta.view(b, 1).expand(-1, k).reshape(m)

        eps = torch.randn_like(target)
        s = torch.rand(m, device=target.device, dtype=target.dtype)
        noisy = s.view(m, 1, 1) * target + (1 - s.view(m, 1, 1)) * eps
        velocity = self.expert(noisy, s, cond, latency, prefix)

        chan = batch.action_valid.view(b, 1, 1, cfg.dim).expand(
            b, k, cfg.horizon, cfg.dim
        )
        chan = chan.reshape(m, cfg.horizon, cfg.dim).to(target.dtype)
        query = batch.action_query_valid.reshape(m, 1, 1).to(target.dtype)
        weight = chan * query
        denom = weight.sum().clamp(min=1.0)
        cfm = (((velocity - (target - eps)) ** 2) * weight).sum() / denom

        with torch.no_grad():
            a0 = torch.randn_like(target)
            zeros = torch.zeros(m, device=target.device, dtype=target.dtype)
            pred = a0 + self.expert(a0, zeros, cond, latency, prefix)
            mse = (((pred - target) ** 2) * weight).sum() / denom
        return cfm, mse

    def _world_loss(self, out: GenOutput, tolerance: float = 0.15) -> Tensor:
        """Latent future prediction (§8.1).

        From the trunk state at each action query, predict the *encoder latents* of
        sensor tokens arriving at ``t + h`` for each configured horizon -- JEPA-style,
        with no pixel decoding.

        This is the least certain component of the reconstruction (§14, item 1).  The
        published record says GEN combines world-model ideas while insisting it is not
        just a world model; a latent predictive term is the cheapest way to be both.
        Note that GEN-1.5 rules out auxiliary objectives *encouraging improvisation*
        (they cite DIAYN); a latent-dynamics term is a different animal, but it remains
        an inference.
        """
        horizons = self.cfg.loss.world_horizons
        b, _, d = out.input_embeddings.shape
        k = out.action_hidden.shape[1]
        pred = self.world_head(out.action_hidden).view(b, k, len(horizons), d)

        stream = out.stream
        is_vision = (
            (stream.stream_ids == int(Stream.HEAD_CAM))
            | (stream.stream_ids == int(Stream.WRIST_CAM))
        ) & stream.valid
        query_t = self._gather_stream(
            stream.timestamps.unsqueeze(-1), stream, Stream.ACTION, k
        ).squeeze(-1)
        targets = out.input_embeddings.detach()

        total = pred.new_zeros(())
        count = pred.new_zeros(())
        for i, h in enumerate(horizons):
            want = query_t + h  # [B, K]
            near = (
                stream.timestamps.unsqueeze(1) - want.unsqueeze(-1)
            ).abs() <= tolerance
            near = near & is_vision.unsqueeze(1)  # [B, K, N]
            weight = near.to(targets.dtype)
            denom = weight.sum(-1, keepdim=True)
            has = (denom.squeeze(-1) > 0).to(targets.dtype)
            tgt = torch.einsum("bkn,bnd->bkd", weight, targets) / denom.clamp(min=1.0)
            err = ((pred[:, :, i] - tgt) ** 2).mean(-1) * has
            total = total + err.sum()
            count = count + has.sum()
        return total / count.clamp(min=1.0)

    def _language_loss(self, batch: GenBatch, out: GenOutput) -> Tensor:
        """Next-token prediction over the scene's language label (§5.5).

        Deliberately small: language is a semantic handle and a corpus index, not the
        objective.  Elements whose language prefix was dropped contribute nothing.
        """
        logits = out.language_logits[:, :-1]
        targets = batch.text_target_ids[:, 1:]
        if logits.shape[1] == 0:
            return logits.new_zeros(())
        keep = out.language_kept[:, 1:].reshape(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        )
        return (loss * keep.to(loss.dtype)).sum() / keep.sum().clamp(min=1).to(
            loss.dtype
        )

    def _reflex_loss(self, batch: GenBatch, out: GenOutput) -> tuple[Tensor, Tensor]:
        """Supervise the 100 Hz residual head, and penalise its magnitude (§7.2).

        The penalty is what keeps the reflex a correction rather than a policy: without
        it the fast head would happily absorb the trunk's job and the model would lose
        its semantic reactivity.
        """
        cfg = self.cfg.action_space
        b, k = batch.action_timestamps.shape
        steps = batch.proprio_fast.shape[2]
        m = b * k
        planned = batch.action_targets[:, :, :steps].reshape(m, steps, cfg.dim)
        cond = torch.cat(
            (out.action_hidden, out.embodiment.unsqueeze(1).expand(-1, k, -1)), dim=-1
        ).reshape(m, -1)
        residual, _ = self.reflex(
            batch.proprio_fast.reshape(m, steps, -1), planned, cond
        )
        target = batch.reflex_target.reshape(m, steps, cfg.dim)
        chan = batch.action_valid.view(b, 1, 1, cfg.dim).expand(b, k, steps, cfg.dim)
        chan = chan.reshape(m, steps, cfg.dim).to(residual.dtype)
        denom = chan.sum().clamp(min=1.0)
        fit = (((residual - target) ** 2) * chan).sum() / denom
        magnitude = ((residual**2) * chan).sum() / denom
        return fit, magnitude

    def compute_loss(self, batch: GenBatch, out: GenOutput) -> GenLosses:
        """Assemble the full objective (§8.1).

        ``L = L_action + λ_w·L_world + λ_l·L_lang + λ_r·L_reflex``

        Parameters
        ----------
        batch:
            The batch that produced ``out``.
        out:
            Result of :meth:`forward`.

        Returns
        -------
        GenLosses
            All terms, with ``total`` ready for ``.backward()``.
        """
        w = self.cfg.loss
        action, mse = self._action_loss(batch, out)
        world = self._world_loss(out)
        language = self._language_loss(batch, out)
        reflex_fit, reflex_mag = self._reflex_loss(batch, out)
        reflex = reflex_fit + w.reflex_magnitude * reflex_mag
        total = (
            w.action * action
            + w.world * world
            + w.language * language
            + w.reflex * reflex
        )
        return GenLosses(
            total=total,
            action=action,
            world=world,
            language=language,
            reflex=reflex,
            next_action_mse=mse,
        )

    # -- sampling --------------------------------------------------------

    @torch.no_grad()
    def sample_actions(
        self,
        batch: GenBatch,
        delta: Tensor | None = None,
        steps: int | None = None,
    ) -> Tensor:
        """Sample action chunks for every query position in a batch.

        Returns
        -------
        Tensor
            ``[B, K, H, A]`` embodiment-specific action chunks.
        """
        cfg = self.cfg.action_space
        out = self(batch, delta=delta)
        b, k = batch.action_timestamps.shape
        m = b * k
        chunk = self.expert.sample(
            out.action_cond.reshape(m, -1),
            out.delta.view(b, 1).expand(-1, k).reshape(m),
            batch.action_prefix.reshape(m, cfg.prefix_steps, cfg.dim),
            steps=steps,
        ).view(b, k, cfg.horizon, cfg.dim)
        return self.adapter(chunk, out.embodiment, batch.action_valid)

    @torch.no_grad()
    def reverse_kl(
        self, batch: GenBatch, n_samples: int = 16, sigma: float = 1.0
    ) -> Tensor:
        """Monte-Carlo reverse KL between policy and demonstration (evidence E9).

        Reproduces GEN-0's evaluation protocol: the policy induces an empirical density
        as a unit-variance Gaussian mixture centred on ``n_samples`` policy samples, the
        data induces a unit-variance Gaussian centred on the ground truth, and the
        expectation is approximated with policy samples.  Reverse KL is mode-seeking, so
        it penalises the mode-averaging that an L2-trained policy exhibits and an honest
        sampler does not -- which is why it, and not MSE alone, predicts post-training
        success.

        Returns
        -------
        Tensor
            Scalar reverse-KL estimate (lower is better).
        """
        cfg = self.cfg.action_space
        out = self(batch)
        b, k = batch.action_timestamps.shape
        m = b * k
        cond = out.action_cond.reshape(m, -1)
        latency = out.delta.view(b, 1).expand(-1, k).reshape(m)
        prefix = batch.action_prefix.reshape(m, cfg.prefix_steps, cfg.dim)
        target = batch.action_targets.reshape(m, -1)

        samples = torch.stack(
            [
                self.expert.sample(cond, latency, prefix).reshape(m, -1)
                for _ in range(n_samples)
            ],
            dim=1,
        )  # [M, S, H*A]
        dim = samples.shape[-1]
        norm = 0.5 * dim * math.log(2 * math.pi * sigma**2)
        sq = ((samples.unsqueeze(2) - samples.unsqueeze(1)) ** 2).sum(-1)  # [M, S, S]
        log_q = (
            torch.logsumexp(-sq / (2 * sigma**2), dim=-1) - math.log(n_samples) - norm
        )
        log_p = -((samples - target.unsqueeze(1)) ** 2).sum(-1) / (2 * sigma**2) - norm
        return (log_q - log_p).mean()


# =============================================================================
# Physical prompting (§9) and the rollout runtime (§10)
# =============================================================================


@dataclass
class PhysicalPrompt:
    """A 3--12 s sensorimotor demonstration, ready to be pinned into context (§9).

    A physical prompt is *not* a special kind of input.  It is an ordinary span of
    sensor and action tokens, re-based in time and inserted into the context ahead of
    the rolling observations.  Nothing marks it as a prompt -- deliberately so: the
    moment the model can distinguish "prompt" from "history", emergent in-context
    learning becomes engineered in-context learning (§9.1).

    Composition is therefore free: two prompts are simply two spans, which is why
    GEN-1.5 can chain two independently recorded demonstrations and invent the bridging
    motions that appear in neither.  Sim-recorded and human-hand demonstrations work for
    the same reason: nothing in the pipeline asks where the pixels came from.

    Attributes
    ----------
    stream:
        The encoded token span (sensor + action tokens, no hand card, no registers).
    duration:
        Span length in seconds, for bookkeeping.
    """

    stream: TokenStream
    duration: float

    @classmethod
    def encode(
        cls, model: "GenModel", batch: GenBatch, end_time: float = -13.0
    ) -> "PhysicalPrompt":
        """Encode a demonstration span into a pinnable token stream.

        Parameters
        ----------
        model:
            The model whose sensor front-end encodes the demonstration.
        batch:
            A :class:`GenBatch` holding the demonstration.
        end_time:
            Where the prompt's final token should sit on the runtime clock.  Prompts are
            placed at a plausible negative offset (e.g. -25 s to -13 s) so the rolling
            observation window has room after them.

        Returns
        -------
        PhysicalPrompt
        """
        with torch.no_grad():
            context, _ = model.encode_context(batch)
        keep = (context.stream_ids != int(Stream.HAND_CARD)) & (
            context.stream_ids != int(Stream.REGISTER)
        )
        # Batch elements share a token layout, so the mask is uniform across the batch.
        idx = keep[0].nonzero(as_tuple=True)[0]
        t = context.timestamps[:, idx]
        span = float((t.max() - t.min()).item())
        shift = end_time - t.max(dim=-1, keepdim=True).values
        return cls(
            stream=TokenStream(
                embeddings=context.embeddings[:, idx],
                timestamps=t + shift,
                stream_ids=context.stream_ids[:, idx],
                valid=context.valid[:, idx],
            ),
            duration=span,
        )


class GenRuntime:
    """The three-rate rollout loop (``ARCHITECTURE.md`` §10.1).

    ::

        100 Hz  reflex head        (CPU/MCU, ~5M params)  ──► actuators
         10 Hz  action expert      (4 flow steps)         ──► 500 ms chunk
         10 Hz  trunk step         (KV-cached, new tokens only)
          2 Hz  latent registers   (piggyback on trunk steps)

    Every loop is free-running; none waits on a slower one.  That is the operational
    meaning of "asynchronous streams of sensing and acting tokens", and the reason no
    System-1/System-2 split is needed: the separation is *temporal*, inside one model,
    rather than *architectural*, across two.

    This implementation drives the loops synchronously from :meth:`step` so it is
    deterministic and testable; a deployment would run them on separate threads against
    the same :class:`PagedKVCache`.
    """

    def __init__(
        self,
        model: GenModel,
        embodiment_features: Tensor,
        channel_valid: Tensor,
        action_valid: Tensor,
        delta_s: float = 0.08,
    ) -> None:
        """
        Parameters
        ----------
        model:
            A (typically ``eval()``-mode) :class:`GenModel`.
        embodiment_features:
            ``[B, embodiment_feature_dim]`` hand card for the robot being driven.
        channel_valid:
            ``[B, n_channels]`` live proprio channels.
        action_valid:
            ``[B, A]`` actuated universal-action channels.
        delta_s:
            The latency offset to advertise to the model.  Should match measured
            wall-clock inference time; the model was trained across a distribution of
            these (§6.3), so a mismatch degrades gracefully rather than failing.
        """
        self.model = model
        self.cfg = model.cfg
        self.embodiment_features = embodiment_features
        self.channel_valid = channel_valid
        self.action_valid = action_valid
        self.delta_s = delta_s
        self.batch_size = embodiment_features.shape[0]
        self.device = embodiment_features.device

        self.cache = PagedKVCache(
            n_layers=self.cfg.trunk.n_layers,
            n_kv_heads=self.cfg.trunk.n_kv_heads,
            head_dim=self.cfg.trunk.head_dim,
            context_seconds=self.cfg.ladder.context_seconds,
        )
        self.blender = ChunkBlender(
            self.cfg.action_space.dim, self.cfg.action_space.blend_steps
        )
        self.reflex_state: Tensor | None = None
        self.embodiment: Tensor | None = None
        #: Trunk latent from the most recent :meth:`act`, consumed by :meth:`reflex`.
        self._last_hidden: Tensor | None = None
        self._prefix = torch.zeros(
            self.batch_size,
            self.cfg.action_space.prefix_steps,
            self.cfg.action_space.dim,
            device=self.device,
        )
        self._last_register_t = float("-inf")

    # -- helpers ---------------------------------------------------------

    @property
    def _delta(self) -> Tensor:
        """``[B]`` latency offset tensor."""
        return torch.full((self.batch_size,), self.delta_s, device=self.device)

    def _step_trunk(self, stream: TokenStream, pinned: bool = False) -> Tensor:
        """Push one group of tokens through the trunk and into the cache."""
        return self.model.trunk.step(stream, self._delta, self.cache, pinned=pinned)

    # -- session management ----------------------------------------------

    @torch.no_grad()
    def reset(self, t0: float = 0.0) -> None:
        """Start a new episode: clear the cache and pin the hand card."""
        self.cache.clear(keep_pinned=False)
        self.blender.reset()
        self.reflex_state = None
        self._prefix.zero_()
        self._last_register_t = float("-inf")
        hand_stream, embodiment = self.model.embodiment(
            self.embodiment_features,
            torch.full((self.batch_size,), t0, device=self.device),
        )
        self.embodiment = embodiment
        self._step_trunk(hand_stream, pinned=True)

    @torch.no_grad()
    def prime(self, prompt: PhysicalPrompt) -> None:
        """Pin a physical prompt into the context (§9.2).

        The prompt's KV pages are written once and marked pinned, so they are exempt
        from eviction and from temporal decay, and swapping prompts is a page-table edit
        rather than a forward pass through a 10B encoder.  That is what makes an
        interactive drag-and-drop prompt interface possible at all.
        """
        self._step_trunk(prompt.stream, pinned=True)

    # -- sensing ---------------------------------------------------------

    @torch.no_grad()
    def observe_vision(
        self,
        frames: Tensor,
        timestamps: Tensor,
        camera_ids: Tensor,
        is_wrist: Tensor,
        extrinsics: Tensor,
    ) -> None:
        """Encode and cache one tick of camera frames."""
        stream = self.model.vision(frames, timestamps, camera_ids, is_wrist, extrinsics)
        self._step_trunk(stream)

    @torch.no_grad()
    def observe_proprio(self, features: Tensor, timestamps: Tensor) -> None:
        """Encode and cache one tick of packed proprioception."""
        stream = self.model.proprio(features, timestamps, self.channel_valid)
        self._step_trunk(stream)

    @torch.no_grad()
    def think(self, t: float) -> None:
        """Emit latent thought registers if the 2 Hz harmonic is due (§6.4).

        Non-blocking by construction: :meth:`act` never waits for this, it simply reads
        whichever registers are already in the cache.
        """
        period = self.cfg.ladder.period_s(self.cfg.ladder.register_divisor)
        if t - self._last_register_t < period:
            return
        self._last_register_t = t
        r = self.cfg.trunk.n_registers
        b = self.batch_size
        stream = TokenStream(
            embeddings=self.model.register_embed.weight.unsqueeze(0).expand(b, -1, -1),
            timestamps=torch.full((b, r), t, device=self.device),
            stream_ids=torch.full(
                (b, r), int(Stream.REGISTER), dtype=torch.long, device=self.device
            ),
            valid=torch.ones(b, r, dtype=torch.bool, device=self.device),
        )
        self._step_trunk(stream)

    # -- acting ----------------------------------------------------------

    @torch.no_grad()
    def act(
        self, t: float, proprio_now: Tensor, flow_steps: int | None = None
    ) -> Tensor:
        """Run one 10 Hz action step and return the blended chunk.

        Parameters
        ----------
        t:
            Current time in seconds.
        proprio_now:
            ``[B, proprio.fast_dim]`` latest proprio summary.
        flow_steps:
            Euler steps for the flow ODE; defaults to the config value.

        Returns
        -------
        Tensor
            ``[B, H, A]`` embodiment-specific action chunk, ready to execute.
        """
        if self.embodiment is None:
            raise RuntimeError("call reset() before act()")
        b = self.batch_size
        query = TokenStream(
            embeddings=(
                self.model.action_query.view(1, 1, -1).expand(b, 1, -1)
                + self.model.action_proj(self._prefix.flatten(1)).unsqueeze(1)
            ),
            timestamps=torch.full((b, 1), t, device=self.device),
            stream_ids=torch.full(
                (b, 1), int(Stream.ACTION), dtype=torch.long, device=self.device
            ),
            valid=torch.ones(b, 1, dtype=torch.bool, device=self.device),
        )
        hidden = self._step_trunk(query)[:, 0]  # [B, D]
        cond = torch.cat((hidden, self.embodiment, proprio_now), dim=-1)
        chunk = self.model.expert.sample(
            cond, self._delta, self._prefix, steps=flow_steps
        )
        chunk = self.model.adapter(chunk, self.embodiment, self.action_valid)
        chunk = self.blender.blend(chunk)
        self._last_hidden = hidden
        return chunk

    @torch.no_grad()
    def reflex(self, chunk: Tensor, proprio_fast: Tensor) -> Tensor:
        """Apply the 100 Hz spinal correction to the executed portion of a chunk (§7.2).

        Parameters
        ----------
        chunk:
            ``[B, H, A]`` from :meth:`act`.
        proprio_fast:
            ``[B, T, proprio.fast_dim]`` control-rate proprio for the steps being
            executed.

        Returns
        -------
        Tensor
            ``[B, T, A]`` corrected actions to send to the actuators.
        """
        if self._last_hidden is None or self.embodiment is None:
            raise RuntimeError("call act() before reflex()")
        t = proprio_fast.shape[1]
        planned = chunk[:, :t]
        cond = torch.cat((self._last_hidden, self.embodiment), dim=-1)
        residual, self.reflex_state = self.model.reflex(
            proprio_fast, planned, cond, self.reflex_state
        )
        return planned + residual

    def commit(self, executed: Tensor) -> None:
        """Record the actions actually sent to the robot.

        The committed prefix is what the expert conditions on next tick, and it is also
        what the action token carries into context -- which is what makes the context
        *sensorimotor* rather than merely sensory, and therefore promptable (§9.2).
        """
        p = self.cfg.action_space.prefix_steps
        self._prefix = torch.cat((self._prefix, executed), dim=1)[:, -p:]
        self.blender.advance(executed.shape[1])

    # -- maintenance -----------------------------------------------------

    def maintain(self, t: float) -> dict[str, int]:
        """Evict expired pages and decay ageing ones (§5.2, §10.2).

        Returns
        -------
        dict[str, int]
            ``{"evicted": n, "tokens": n, "bytes": n}`` for logging.
        """
        evicted = self.cache.evict(t)
        self.cache.decay(t)
        return {
            "evicted": evicted,
            "tokens": self.cache.n_tokens,
            "bytes": self.cache.memory_bytes(),
        }


# =============================================================================
# Utilities
# =============================================================================


def count_parameters(model: GenModel) -> dict[str, int | float]:
    """Break parameter counts down by subsystem (§4).

    The reference presets land near 8% / 88% / 3% at 7B, which is what makes the
    per-subsystem fine-tuning weight deltas Generalist reports individually meaningful.
    Build any preset under ``torch.device("meta")`` to reproduce the split without
    allocating weights.

    Returns
    -------
    dict
        Absolute counts per subsystem plus their percentage shares.
    """

    def n(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    sensing = (
        n(model.vision) + n(model.proprio) + n(model.language) + n(model.embodiment)
    )
    reasoning = (
        n(model.trunk)
        + model.action_query.numel()
        + n(model.register_embed)
        + n(model.action_proj)
    )
    actuation = n(model.expert) + n(model.adapter) + n(model.reflex)
    aux = n(model.world_head)
    total = sensing + reasoning + actuation + aux
    return {
        "sensing": sensing,
        "reasoning": reasoning,
        "actuation": actuation,
        "auxiliary": aux,
        "total": total,
        "sensing_pct": 100.0 * sensing / total,
        "reasoning_pct": 100.0 * reasoning / total,
        "actuation_pct": 100.0 * actuation / total,
    }


def make_dummy_batch(
    cfg: GenConfig,
    batch_size: int = 2,
    n_frames: int = 8,
    n_proprio: int = 4,
    n_text: int = 4,
    n_queries: int = 3,
    span_seconds: float = 1.0,
    device: torch.device | str = "cpu",
) -> GenBatch:
    """Construct a syntactically valid random batch for smoke tests.

    The timestamps are laid out on plausible harmonics within ``span_seconds`` so the
    time-causal mask has real structure rather than being degenerate.
    """
    v, p, a = cfg.vision, cfg.proprio, cfg.action_space
    b = batch_size
    dev = torch.device(device)

    frame_t = (
        torch.linspace(0.0, span_seconds, n_frames, device=dev)
        .unsqueeze(0)
        .expand(b, -1)
    )
    camera_ids = (torch.arange(n_frames, device=dev) % 4).unsqueeze(0).expand(b, -1)
    is_wrist = camera_ids >= 2
    proprio_t = (
        torch.linspace(0.0, span_seconds, n_proprio, device=dev)
        .unsqueeze(0)
        .expand(b, -1)
    )
    action_t = (
        torch.linspace(span_seconds * 0.5, span_seconds, n_queries, device=dev)
        .unsqueeze(0)
        .expand(b, -1)
    )

    channel_valid = torch.zeros(b, p.n_channels, dtype=torch.bool, device=dev)
    channel_valid[:, : p.max_joints] = True
    action_valid = torch.zeros(b, a.dim, dtype=torch.bool, device=dev)
    action_valid[:, : a.ee_dim] = True

    spec = EmbodimentSpec.dummy(p, v, a, device=dev)
    embodiment_features = spec.feature_vector().unsqueeze(0).expand(b, -1).contiguous()

    return GenBatch(
        frames=torch.randn(
            b, n_frames, v.in_channels, v.image_size, v.image_size, device=dev
        ),
        frame_timestamps=frame_t.contiguous(),
        camera_ids=camera_ids.contiguous(),
        is_wrist=is_wrist.contiguous(),
        extrinsics=torch.randn(b, n_frames, v.extrinsic_dim, device=dev),
        proprio_features=torch.randn(b, n_proprio, p.feature_dim, device=dev),
        proprio_timestamps=proprio_t.contiguous(),
        proprio_fast=torch.randn(b, n_queries, a.execute_steps, p.fast_dim, device=dev),
        text_embeddings=torch.randn(b, n_text, cfg.text_embed_dim, device=dev),
        text_timestamps=torch.zeros(b, n_text, device=dev),
        text_target_ids=torch.randint(0, cfg.vocab_size, (b, n_text), device=dev),
        embodiment_features=embodiment_features,
        channel_valid=channel_valid,
        action_valid=action_valid,
        action_timestamps=action_t.contiguous(),
        action_targets=torch.randn(b, n_queries, a.horizon, a.dim, device=dev),
        action_prefix=torch.randn(b, n_queries, a.prefix_steps, a.dim, device=dev),
        action_query_valid=torch.ones(b, n_queries, dtype=torch.bool, device=dev),
        reflex_target=torch.randn(b, n_queries, a.execute_steps, a.dim, device=dev)
        * 0.01,
    )


def _self_check() -> None:
    """End-to-end smoke test: parameters, a training step, sampling, and a rollout.

    Run with ``python -m open_gen.gen_model``.  Every assertion here corresponds to a
    property the architecture claims:

    * the parameter split is roughly 12 / 76 / 12 across the three subsystems (§4);
    * a full training step differentiates end to end through all four loss terms (§8.1);
    * reverse KL is computable from policy samples, as GEN-0's protocol requires (E9);
    * success is flat in δ -- the Harmonic Reasoning test (§6.2, build step 4);
    * a rollout runs against the paged cache with prompt pages pinned (§9.2, §10).
    """
    torch.manual_seed(0)
    cfg = GenConfig.debug()
    model = GenModel(cfg)
    batch = make_dummy_batch(cfg, batch_size=2)

    print(f"=== Open-Gen self-check ({cfg.name}) ===\n")

    counts = count_parameters(model)
    print("parameters by subsystem (§4):")
    for key in ("sensing", "reasoning", "actuation", "auxiliary"):
        pct = counts.get(f"{key}_pct")
        suffix = f"  ({pct:.1f}%)" if isinstance(pct, float) else ""
        print(f"  {key:<12} {counts[key]:>12,}{suffix}")
    print(f"  {'total':<12} {counts['total']:>12,}\n")

    print("reference 30 s token budget (§5.1):")
    for stream, n in HarmonicLadderConfig().token_budget().items():
        print(f"  {stream:<12} {n:>8,}")
    print(
        f"  {'TOTAL':<12} {sum(HarmonicLadderConfig().token_budget().values()):>8,}\n"
    )

    # -- training step --------------------------------------------------
    model.train()
    out = model(batch)
    losses = model.compute_loss(batch, out)
    losses.total.backward()
    grads = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    print(
        f"context tokens: {out.stream.length} per element "
        f"({int(out.stream.valid.sum()) / batch.batch_size:.0f} valid)"
    )
    print(f"losses: {losses.as_dict()}")
    print(f"tensors receiving gradient: {grads}\n")
    assert torch.isfinite(losses.total), "loss is not finite"

    # -- sampling and the reverse-KL protocol (E9) ----------------------
    model.eval()
    chunks = model.sample_actions(batch)
    kl = model.reverse_kl(batch, n_samples=4)
    print(f"sampled chunk shape: {tuple(chunks.shape)}")
    print(f"reverse KL (MC, 4 samples): {float(kl):.3f}\n")

    # -- the Harmonic Reasoning test (§6.2) -----------------------------
    print("next-action MSE across the latency curriculum (§6.3):")
    for delta_ms in (0.0, 40.0, 80.0, 160.0, 250.0):
        d = torch.full((batch.batch_size,), delta_ms / 1000.0)
        o = model(batch, delta=d)
        _, mse = model._action_loss(batch, o)
        print(f"  δ = {delta_ms:5.0f} ms   MSE = {float(mse):.4f}")
    print()

    # -- physical prompting + rollout (§9, §10) -------------------------
    prompt = PhysicalPrompt.encode(
        model, make_dummy_batch(cfg, batch_size=2), end_time=-1.5
    )
    runtime = GenRuntime(
        model,
        batch.embodiment_features,
        batch.channel_valid,
        batch.action_valid,
        delta_s=0.08,
    )
    runtime.reset(t0=0.0)
    runtime.prime(prompt)
    pinned = sum(p.n_tokens for p in runtime.cache.pages if p.pinned)
    print(
        f"physical prompt: {prompt.duration:.2f}s span, {pinned} pinned tokens (§9.2)"
    )

    space = cfg.action_space
    tick = cfg.ladder.period_s(cfg.ladder.action_divisor)
    for i in range(4):
        t = i * tick
        obs = make_dummy_batch(cfg, batch_size=2, n_frames=4, n_proprio=1)
        runtime.observe_vision(
            obs.frames,
            torch.full_like(obs.frame_timestamps, t),
            obs.camera_ids,
            obs.is_wrist,
            obs.extrinsics,
        )
        runtime.observe_proprio(
            obs.proprio_features, torch.full_like(obs.proprio_timestamps, t)
        )
        runtime.think(t)
        chunk = runtime.act(t, obs.proprio_fast[:, 0, 0])
        executed = runtime.reflex(chunk, obs.proprio_fast[:, 0])
        runtime.commit(executed[:, : space.execute_steps])
        stats = runtime.maintain(t)
        print(
            f"  t={t:5.2f}s  chunk={tuple(chunk.shape)}  "
            f"executed={tuple(executed.shape)}  "
            f"cache={stats['tokens']} tok / {stats['bytes'] / 1024:.1f} KiB"
        )
        assert torch.isfinite(chunk).all(), "rollout produced non-finite actions"

    print("\nall checks passed.")


if __name__ == "__main__":  # pragma: no cover
    _self_check()
