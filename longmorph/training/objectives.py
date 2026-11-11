"""Derivative and event-local objectives for cycle reconstruction."""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def diff1(x: torch.Tensor) -> torch.Tensor:
    return F.pad(x[..., 1:] - x[..., :-1], (1, 0), mode="replicate")


def diff2(x: torch.Tensor) -> torch.Tensor:
    return diff1(diff1(x))


def event_targets(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return soft u/peak/shoulder/notch maps and union mask; notch is confidence gated."""
    b, _, length = x.shape
    d1, d2 = diff1(x), diff2(x)
    u = d1[:, 0].argmax(1)
    peak = x[:, 0].argmax(1)
    shoulder = torch.empty_like(peak)
    notch = torch.empty_like(peak)
    notch_ok = torch.zeros(b, device=x.device)
    for i in range(b):
        p = int(peak[i])
        s0, s1 = min(length - 1, p + 3), min(length, p + 72)
        shoulder[i] = s0 + d2[i, 0, s0:s1].argmin() if s1 > s0 else p
        n0, n1 = min(length - 1, p + 20), min(length, p + 135)
        if n1 > n0 + 2:
            seg = d1[i, 0, n0:n1]
            ni = n0 + seg.argmax()
            notch[i] = ni
            notch_ok[i] = float(seg.max() > 0.12 * d1[i, 0].abs().max())
        else:
            notch[i] = p
    pos = torch.stack([u, peak, shoulder, notch], 1)
    widths = torch.tensor([8.0, 12.0, 10.0, 12.0], device=x.device)
    t = torch.arange(length, device=x.device)[None, None, :]
    maps = torch.exp(-0.5 * ((t - pos[..., None]) / widths[None, :, None]) ** 2)
    maps[:, 3] *= notch_ok[:, None]
    return maps, maps.amax(1, keepdim=True).clamp(0, 1)


def auc(real: np.ndarray, recon: np.ndarray, domain: str, seed: int) -> float:
    def transform(x: np.ndarray) -> np.ndarray:
        if domain == "d1":
            return np.diff(x, axis=1, prepend=x[:, :1])
        if domain == "d2":
            d = np.diff(x, axis=1, prepend=x[:, :1])
            return np.diff(d, axis=1, prepend=d[:, :1])
        if domain == "spectral":
            return np.log1p(np.abs(np.fft.rfft(x, axis=1)))
        return x

    X = np.concatenate([transform(real), transform(recon)])
    y = np.r_[np.ones(len(real)), np.zeros(len(recon))]
    tr, te = train_test_split(np.arange(len(y)), test_size=0.35, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1500, class_weight="balanced", C=0.5).fit(X[tr], y[tr])
    return float(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))


def diagnostics(real: np.ndarray, recon: np.ndarray, seed: int) -> dict[str, float]:
    r = torch.from_numpy(real[:, None]).float()
    f = torch.from_numpy(recon[:, None]).float()
    _, mask = event_targets(r)
    rd2, fd2 = diff2(r), diff2(f)
    outside = 1 - mask
    event_ratio = ((mask * fd2).square().sum() / ((mask * rd2).square().sum() + 1e-9)).item()
    outside_ratio = (
        (outside * fd2).square().sum() / ((outside * rd2).square().sum() + 1e-9)
    ).item()
    out = {
        f"reconstruction_{d}_auc": auc(real, recon, d, seed)
        for d in ["raw", "d1", "d2", "spectral"]
    }
    rmin, fmin = rd2.amin((1, 2)).numpy(), fd2.amin((1, 2)).numpy()
    out.update(
        event_d2_energy_ratio=event_ratio,
        outside_d2_energy_ratio=outside_ratio,
        real_d2_min_median=float(np.median(rmin)),
        reconstruction_d2_min_median=float(np.median(fmin)),
        d2_min_median_relative_error=float(
            abs(np.median(fmin) - np.median(rmin)) / (abs(np.median(rmin)) + 1e-9)
        ),
    )
    for q in [0.1, 0.9]:
        out[f"d2_min_q{int(q*100)}_relative_error"] = float(
            abs(np.quantile(fmin, q) - np.quantile(rmin, q)) / (abs(np.quantile(rmin, q)) + 1e-9)
        )
    return out
