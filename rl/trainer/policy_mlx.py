"""LoRA policy (and critic) over an mlx_vlm model, for the MLX RL trainers.

Three things the RL loop needs that plain generation does not:

1. **Per-token log-probs of an already-generated sequence.** Generation and
   scoring are separate passes here, matching `model_tz_llama.py:act_oneline`,
   which generates and then re-evaluates. Simpler than threading log-probs out
   of the sampler, and it is what the PPO update re-does each epoch anyway.

2. **A frozen reference policy for the KL term.** LoRA makes this free: every
   adapter carries a multiplicative `scale`, so zeroing all of them turns the
   forward pass back into the base model. No second copy of 8B weights.
   (Borrowed from Cargo-Cult-Research/mlx-rl `models.py`; see
   third_party/REFERENCES.md.)

3. **A value head**, for PPO. `model_tz_llama.py` hardcodes `nn.Linear(4096,...)`;
   this reads `hidden_size` from the config instead. For Qwen3-VL-8B those
   happen to agree at 4096, but the hardcode would silently break any other size.
"""
from __future__ import annotations

from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn


def lora_modules(model):
    """Every attached LoRA adapter, identified by its parameters, not its class.

    mlx_vlm attaches `LoRALinear` while other tooling uses `LoRaLayer`; both
    expose `lora_a`/`lora_b` and a multiplicative `scale`.
    """
    return [m for _, m in model.named_modules()
            if hasattr(m, "scale") and hasattr(m, "lora_a") and hasattr(m, "lora_b")]


class FrozenReference:
    """Snapshot of the policy at the start of RL, used for the KL term.

    `reference_policy` below zeroes the adapter scales, which yields the
    *pre-SFT base model*. That is the right reference only when RL starts from
    the base. Starting from an SFT checkpoint -- the paper's ordering -- it is
    wrong: the KL would drag the policy back toward a model the SFT stage
    deliberately moved away from, and it starts large instead of at zero
    (measured 0.67 nats/token at step 0, before a single update).

    Snapshotting the initial adapter weights is correct in both cases. A freshly
    initialised LoRA has `lora_b == 0`, so the snapshot *is* the base model when
    training from scratch. Costs one copy of the adapters (~87 MB at r=8), not
    of the 8B base.
    """

    def __init__(self, model):
        from mlx.utils import tree_flatten
        self.snapshot = {k: mx.array(v)
                         for k, v in tree_flatten(model.trainable_parameters())}
        if not self.snapshot:
            raise ValueError("model has no trainable parameters to snapshot")

    @contextmanager
    def __call__(self, model):
        from mlx.utils import tree_flatten, tree_unflatten
        current = {k: v for k, v in tree_flatten(model.trainable_parameters())}
        model.update(tree_unflatten(list(self.snapshot.items())))
        try:
            yield model
        finally:
            model.update(tree_unflatten(list(current.items())))


@contextmanager
def reference_policy(model):
    """Inside this block the model computes the frozen base policy.

    Zeroing each adapter's scale removes the LoRA update from the forward pass.
    Restores the previous scales on the way out, including on exception.
    """
    mods = lora_modules(model)
    if not mods:
        raise ValueError(
            "no LoRA adapters found; the reference policy would be identical "
            "to the policy and the KL term would be identically zero")
    saved = [m.scale for m in mods]
    for m in mods:
        m.scale = 0.0
    try:
        yield model
    finally:
        for m, s in zip(mods, saved):
            m.scale = s


def token_logprobs(logits: mx.array, targets: mx.array) -> mx.array:
    """logits [B, L, V], targets [B, L] -> log p(target_t) [B, L]."""
    return -nn.losses.cross_entropy(logits, targets, reduction="none")


def sequence_logprobs(model, input_ids: mx.array, completion_mask: mx.array):
    """Per-token log-probs over a full sequence, aligned to the completion.

    input_ids [B, L]; completion_mask [B, L] is 1 on tokens the policy produced.
    Returns (logprobs [B, L-1], mask [B, L-1]) already shifted so that entry t
    is log p(input_ids[t+1]).
    """
    logits = model.language_model(input_ids[:, :-1]).logits.astype(mx.float32)
    return token_logprobs(logits, input_ids[:, 1:]), completion_mask[:, 1:]


class ValueHead(nn.Module):
    """Scalar value per environment step, read off the last hidden state.

    Mirrors `VLMValue` in model_tz_llama.py, but sized from the config.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # [B, L, H] -> value at the final position, [B]
        return self.net(hidden_states[:, -1].astype(mx.float32)).squeeze(-1)


def hidden_size_of(model) -> int:
    """Read the language hidden size, whatever the config nests it under."""
    config = model.config
    for holder in (getattr(config, "text_config", None), config):
        size = getattr(holder, "hidden_size", None)
        if size:
            return int(size)
    raise ValueError("could not determine hidden_size from the model config")
