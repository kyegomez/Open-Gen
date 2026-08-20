"""Minimal forward pass through Open-Gen.

Run with ``python example.py``.  Uses the ``debug`` preset (~3M parameters) so it
finishes in seconds on CPU; swap in ``GenConfig.og_7b()`` for the reference model.
"""

import torch

from open_gen.gen_model import GenConfig, GenModel, make_dummy_batch

torch.manual_seed(0)

# The debug preset is a real model, just small: same streams, same harmonic ladder,
# same losses -- only the widths and the 2 s context window are shrunk.
config = GenConfig.debug()
model = GenModel(config)

# A batch is one randomly sampled continuous span of physical experience: camera
# frames, packed proprioception, a language label, the robot's hand card, and the
# action chunks to predict.  Substitute your own loader here.
batch = make_dummy_batch(config, batch_size=2)

# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------
# Every stream is tokenized, stamped with a wall-clock timestamp, interleaved into
# one time-sorted sequence, and run through the trunk under a time-causal mask.
# `delta` (the inference latency the model should assume) is sampled from the
# training curriculum unless you pass it explicitly.
output = model(batch)

print(f"context tokens : {output.stream.length}")
print(f"hidden states  : {tuple(output.hidden.shape)}")
print(f"action latents : {tuple(output.action_hidden.shape)}")
print(f"latency delta  : {output.delta.tolist()} s")

# ---------------------------------------------------------------------------
# Loss and backward
# ---------------------------------------------------------------------------
# L = L_action + 0.2*L_world + 0.05*L_language + 0.05*L_reflex
#
# `language` reads 0.0 whenever the whole batch had its instruction dropped -- during
# pretraining the language prefix is dropped 40% of the time, deliberately, so the
# model never becomes dependent on being told what to do.
losses = model.compute_loss(batch, output)
losses.total.backward()

print("\nlosses:")
for name, value in losses.as_dict().items():
    print(f"  {name:<16} {value:.4f}")

# ---------------------------------------------------------------------------
# Sampling actions
# ---------------------------------------------------------------------------
# The action expert integrates a flow ODE from noise to an action chunk, then a
# hypernetwork-generated adapter maps the universal action space onto this
# robot's actuators.  [batch, queries, horizon, action_dim]
model.eval()
with torch.no_grad():
    chunks = model.sample_actions(batch)

horizon_ms = 1000 * config.action_space.horizon / config.ladder.base_tick_hz
print(f"\nsampled chunks : {tuple(chunks.shape)}  ({horizon_ms:.0f} ms per chunk)")
