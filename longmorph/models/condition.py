"""Stochastic three-cycle condition-state transition."""

import torch
import torch.nn as nn


class ConditionTransition(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.dim = dim
        self.backbone = nn.Sequential(
            nn.Linear(3 * dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden, 3 * dim)
        self.logstd = nn.Linear(hidden, 3 * dim)

    def forward(self, past):
        h = self.backbone(past.flatten(1))
        return self.mean(h).reshape(-1, 3, self.dim), self.logstd(h).reshape(-1, 3, self.dim).clamp(
            -4, 1
        )
