"""Neural predictors used by downstream tasks."""

from __future__ import annotations

import torch
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x[:, :, : -self.n] if self.n else x


class TemporalResidualBlock(nn.Module):
    def __init__(self, cin, cout, k, d, drop):
        super().__init__()
        pad = (k - 1) * d
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=pad, dilation=d),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Conv1d(cout, cout, k, padding=pad, dilation=d),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(drop),
        )
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return torch.relu(self.net(x) + self.skip(x))


class TCNForecaster(nn.Module):
    def __init__(self, channels=(32, 64, 128, 128), kernel_size=5, dropout=0.1):
        super().__init__()
        layers = []
        cin = 1
        for i, cout in enumerate(channels):
            layers.append(TemporalResidualBlock(cin, cout, kernel_size, 2**i, dropout))
            cin = cout
        self.tcn = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(cin, 768))

    def forward(self, x):
        return self.proj(self.tcn(x.reshape(len(x), 1, 768))).reshape(-1, 3, 256)


class ResidualTCNForecaster(nn.Module):
    """Forecast a bounded change from the aligned three-cycle input."""

    def __init__(self, channels=(32, 64, 128, 128), kernel_size=5, dropout=0.1, residual_scale=0.5):
        super().__init__()
        layers = []
        input_channels = 1
        for index, output_channels in enumerate(channels):
            layers.append(
                TemporalResidualBlock(
                    input_channels, output_channels, kernel_size, 2**index, dropout
                )
            )
            input_channels = output_channels
        self.tcn = nn.Sequential(*layers)
        self.local = nn.Conv1d(input_channels, 1, 1)
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input_channels, 768),
        )
        self.residual_scale = residual_scale

    def forward(self, x):
        flat = x.reshape(len(x), 1, 768)
        hidden = self.tcn(flat)
        local_delta = self.local(hidden).reshape(-1, 3, 256)
        global_delta = self.global_head(hidden).reshape(-1, 3, 256)
        delta = 0.75 * local_delta + 0.25 * global_delta
        return x + self.residual_scale * torch.tanh(delta)


class PatchTSTForecaster(nn.Module):
    """Fixed univariate PatchTST-style forecaster for aligned PPG cycles.

    Input shape is [batch, input_cycles, cycle_len]. The model tokenizes the
    flattened waveform into overlapping patches, applies a Transformer encoder,
    and predicts the next target_cycles aligned cycles.
    """

    def __init__(
        self,
        input_cycles: int = 5,
        target_cycles: int = 1,
        cycle_len: int = 256,
        patch_len: int = 64,
        stride: int = 32,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.10,
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.input_cycles = input_cycles
        self.target_cycles = target_cycles
        self.cycle_len = cycle_len
        self.input_len = input_cycles * cycle_len
        self.target_len = target_cycles * cycle_len
        self.patch_len = patch_len
        self.stride = stride
        self.revin = revin
        if patch_len > self.input_len:
            raise ValueError("patch_len cannot exceed the flattened input length")
        self.n_patches = 1 + (self.input_len - patch_len) // stride
        self.patch_projection = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=e_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.n_patches * d_model, self.target_len),
        )
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = len(x)
        flat = x.reshape(batch, self.input_len)
        if self.revin:
            center = flat.mean(dim=1, keepdim=True)
            scale = flat.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-5)
            flat_in = (flat - center) / scale
        else:
            center = flat.new_zeros((batch, 1))
            scale = flat.new_ones((batch, 1))
            flat_in = flat
        patches = flat_in.unfold(dimension=1, size=self.patch_len, step=self.stride)
        tokens = self.patch_projection(patches) + self.position
        encoded = self.norm(self.encoder(tokens))
        output = self.head(encoded)
        if self.revin:
            output = output * scale + center
        return output.reshape(batch, self.target_cycles, self.cycle_len)


class HeartRateRegressor(nn.Sequential):
    """Predict heart rate from an eight-second, 64 Hz PPG window."""

    def __init__(self):
        super().__init__(
            nn.Conv1d(1, 16, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )
