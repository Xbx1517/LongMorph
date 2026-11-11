"""Conditional latent block velocity networks."""

import torch
import torch.nn as nn


class BlockVelocity(nn.Module):
    def __init__(self, channels, cond_dim, extra_channels=0, hidden=96):
        super().__init__()
        self.cond = nn.Sequential(nn.Linear(cond_dim, 64), nn.SiLU(), nn.Linear(64, 32))
        inp = 2 * channels + 32 + 1 + extra_channels
        self.net = nn.Sequential(
            nn.Conv1d(inp, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, channels, 3, padding=1),
        )

    def forward(self, x, prefix, cond, t, extra=None):
        ce = self.cond(cond)[..., None].expand(-1, -1, x.shape[-1])
        te = t[:, None, None].expand(-1, 1, x.shape[-1])
        parts = [x, prefix, ce, te]
        if extra is not None:
            parts.append(extra)
        return self.net(torch.cat(parts, 1))


class CycleCouplingPredictor(nn.Module):
    """Predict the next coarse-cycle velocity from the previous cycle state."""

    def __init__(self, channels=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels + 1, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, channels, 3, padding=1),
        )

    def forward(self, x, t):
        te = t[:, None, None].expand(-1, 1, x.shape[-1])
        return self.net(torch.cat([x, te], 1))
