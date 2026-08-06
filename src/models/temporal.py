"""Wrappers for consuming 4D (field map) inputs with 3D convolutional backbones.

PyTorch has no `nn.Conv4d`, and MONAI's convolution factory only registers 1/2/3
spatial dimensions, so a genuine 4D convolution is not available. The two viable
strategies are implemented here and in `models.regressor`:

  1. "channels" - the frame axis becomes the input channel axis of a single 3D
     conv stack. Cheapest, but requires a fixed frame count across every scan and
     cannot reuse a 1-channel checkpoint.

  2. "pool" - a 3D backbone encodes each frame independently, then predictions
     are pooled across frames. Handles a variable frame count, and at T=1 it is
     mathematically identical to the original 3D model, so `model_02r7` weights
     load directly.
"""

import torch
import torch.nn as nn


VALID_POOLS = ("mean", "max", "median", "first")


class FrameWiseRegressor(nn.Module):
    """Run a 3D backbone over each frame of a 4D input and pool the outputs.

    Input:  (B, T, X, Y, Z)
    Output: (B, out_dim)

    Frames are folded into the batch axis, so the effective forward-pass batch is
    B*T. Reduce `--batch-size` accordingly when moving from 3D to 4D.

    Note this pools per-frame *predictions*, not features. That is the deliberate
    trade: it keeps the wrapper transparent and lets a 3D checkpoint drop in
    unchanged. Pooling features instead would likely be a little stronger but
    requires cutting the head off the MONAI Regressor.
    """

    def __init__(self, backbone, pool="mean"):
        super().__init__()
        if pool not in VALID_POOLS:
            raise ValueError(f"pool must be one of {VALID_POOLS}, got {pool!r}")
        self.backbone = backbone
        self.pool = pool

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(
                f"FrameWiseRegressor expects (B, T, X, Y, Z), got shape {tuple(x.shape)}"
            )

        batch_size, num_frames = x.shape[0], x.shape[1]

        # (B, T, X, Y, Z) -> (B*T, 1, X, Y, Z)
        folded = x.reshape(batch_size * num_frames, 1, *x.shape[2:])

        outputs = self.backbone(folded)
        if isinstance(outputs, (list, tuple)):
            outputs = outputs[0]

        # (B*T, out_dim) -> (B, T, out_dim)
        outputs = outputs.reshape(batch_size, num_frames, -1)

        if self.pool == "mean":
            pooled = outputs.mean(dim=1)
        elif self.pool == "max":
            pooled = outputs.max(dim=1).values
        elif self.pool == "median":
            pooled = outputs.median(dim=1).values
        else:  # first
            pooled = outputs[:, 0, :]

        return pooled


def load_backbone_state_dict(model, state_dict, strict=True):
    """Load a checkpoint into either a bare backbone or a FrameWiseRegressor.

    Checkpoints saved before the 4D change have no `backbone.` prefix on their
    keys, so this re-prefixes them when loading into a wrapped model.
    """
    target = model.backbone if isinstance(model, FrameWiseRegressor) else model

    if isinstance(model, FrameWiseRegressor) and any(
        k.startswith("backbone.") for k in state_dict
    ):
        state_dict = {
            k[len("backbone.") :]: v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }

    return target.load_state_dict(state_dict, strict=strict)
