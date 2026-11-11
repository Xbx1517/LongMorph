"""Coarse and detail cycle encoders and decoders."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def smooth(x, kernel=17):
    return F.avg_pool1d(F.pad(x, (kernel // 2, kernel // 2), mode="reflect"), kernel, 1)


def split_components(x, edge=16):
    """Separate morphology and detail without assigning edge artifacts to detail."""
    raw_detail = x - smooth(x)
    length = x.shape[-1]
    taper = torch.ones(length, device=x.device, dtype=x.dtype)
    ramp = torch.sin(torch.linspace(0, torch.pi / 2, edge, device=x.device, dtype=x.dtype)).square()
    taper[:edge] = ramp
    taper[-edge:] = torch.flip(ramp, dims=[0])
    detail = raw_detail * taper.view(1, 1, -1)
    return x - detail, detail


class StreamEncoder(nn.Module):
    def __init__(self, channels, downsample=True):
        super().__init__()
        projection = (
            nn.Conv1d(48, channels, 4, 2, 1)
            if downsample
            else nn.Conv1d(48, channels, 5, padding=2)
        )
        self.net = nn.Sequential(
            nn.Conv1d(1, 48, 7, padding=3), ResBlock(48, 1), ResBlock(48, 2), projection, nn.SiLU()
        )

    def forward(self, x):
        return self.net(x)


class StreamDecoder(nn.Module):
    def __init__(self, channels, upsample=True):
        super().__init__()
        projection = (
            nn.ConvTranspose1d(channels, 48, 4, 2, 1)
            if upsample
            else nn.Conv1d(channels, 48, 5, padding=2)
        )
        self.net = nn.Sequential(
            projection, ResBlock(48, 1), ResBlock(48, 2), nn.Conv1d(48, 1, 7, padding=3)
        )

    def forward(self, z):
        return self.net(z)


class DualStreamCodec(nn.Module):
    def __init__(self, coarse_channels=4, detail_channels=4, detail_fullres=False):
        super().__init__()
        self.coarse_encoder = StreamEncoder(coarse_channels)
        self.detail_encoder = StreamEncoder(detail_channels, not detail_fullres)
        self.coarse_decoder = StreamDecoder(coarse_channels)
        self.detail_decoder = StreamDecoder(detail_channels, not detail_fullres)

    def forward(self, x):
        target_coarse, target_detail = split_components(x)
        zc = self.coarse_encoder(target_coarse)
        zd = self.detail_encoder(target_detail)
        coarse = self.coarse_decoder(zc)
        detail = self.detail_decoder(zd)
        return coarse + detail, coarse, detail, zc, zd


def load_generator_codec(checkpoint, device="cpu"):
    """Load the mixed-resolution 4-channel coarse / 2-channel detail architecture."""
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload["model"]
    expected = {
        "coarse_encoder.net.3.weight": (4, 48, 4),
        "detail_encoder.net.3.weight": (2, 48, 5),
    }
    if any(key not in state or tuple(state[key].shape) != shape for key, shape in expected.items()):
        raise ValueError(
            "Generation requires a mixed-resolution codec with four coarse channels "
            "and two full-resolution detail channels. Train the detail codec with "
            "--detail-channels 2 --detail-fullres, then use train hybrid to combine streams."
        )
    model = DualStreamCodec(4, 2, True).to(device)
    model.load_state_dict(state)
    return model
