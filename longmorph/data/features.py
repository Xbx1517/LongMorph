"""Phase-normalized pulse-shape features and distribution distances."""

from __future__ import annotations
import numpy as np
from scipy.signal import welch
from .cycles import minmax_1d

SHAPE_FEATURE_NAMES = [
    "width50",
    "width75",
    "pulse_area",
    "systolic_area",
    "diastolic_area",
    "area_ratio",
    "rise_time",
    "decay_time",
    "late_decay_slope",
    "notch_proxy",
    "slope_ratio",
    "d1_max",
    "d2_min",
    "psd_tail_ratio",
]


def compute_psd_tail_ratio(
    cycles: np.ndarray, sample_rate: float, tail_hz: float = 20.0
) -> np.ndarray:
    out = []
    for cycle in np.asarray(cycles, dtype=np.float32):
        freqs, psd = welch(cycle, fs=sample_rate, nperseg=min(len(cycle), 128))
        total = float(np.sum(psd) + 1e-12)
        out.append(float(np.sum(psd[freqs > tail_hz]) / total) if len(freqs) else 0.0)
    return np.asarray(out, dtype=np.float32)


def compute_ppg_shape_features(cycles: np.ndarray, sample_rate: float) -> np.ndarray:
    rows = []
    for cycle in np.asarray(cycles, dtype=np.float32):
        x = minmax_1d(cycle)
        peak = int(np.argmax(x))
        n = max(1, len(x) - 1)
        d1 = np.diff(x)
        d2 = np.diff(x, n=2)
        systolic = x[: peak + 1] if peak >= 0 else x[:1]
        diastolic = x[peak:] if peak < len(x) else x[-1:]
        width50 = float(np.mean(x >= 0.50))
        width75 = float(np.mean(x >= 0.75))
        pulse_area = float(np.trapezoid(x) / max(1, len(x)))
        systolic_area = float(np.trapezoid(systolic) / max(1, len(systolic)))
        diastolic_area = float(np.trapezoid(diastolic) / max(1, len(diastolic)))
        area_ratio = systolic_area / (diastolic_area + 1e-6)
        rise_time = float(peak / n)
        decay_time = float((len(x) - 1 - peak) / n)
        tail_start = min(len(x) - 2, peak + int(0.60 * max(1, len(x) - 1 - peak)))
        late_decay_slope = (
            float(np.mean(np.diff(x[tail_start:]))) if tail_start < len(x) - 2 else 0.0
        )
        post = x[peak + max(2, int(0.05 * len(x))) :]
        notch_proxy = float(max(0.0, np.max(post) - np.min(post))) if len(post) >= 5 else 0.0
        up_slope = float(np.max(d1[: max(1, peak)])) if peak > 0 else 0.0
        down_slope = float(abs(np.min(d1[peak:]))) if peak < len(d1) else 0.0
        slope_ratio = up_slope / (down_slope + 1e-6)
        d1_max = up_slope
        d2_min = float(np.min(d2)) if len(d2) else 0.0
        psd_tail = float(compute_psd_tail_ratio(x[None, :], sample_rate)[0])
        rows.append(
            [
                width50,
                width75,
                pulse_area,
                systolic_area,
                diastolic_area,
                area_ratio,
                rise_time,
                decay_time,
                late_decay_slope,
                notch_proxy,
                slope_ratio,
                d1_max,
                d2_min,
                psd_tail,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def rbf_mmd_2sample(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return 0.0
    x = np.asarray(x, dtype=np.float32).reshape(len(x), -1)
    y = np.asarray(y, dtype=np.float32).reshape(len(y), -1)
    z = np.vstack([x, y])
    dists = ((z[:, None] - z[None]) ** 2).sum(-1)
    positive = dists[dists > 0]
    gamma = 1.0 / (float(np.median(positive)) + 1e-8) if positive.size else 1.0
    kxx = np.exp(-gamma * ((x[:, None] - x[None]) ** 2).sum(-1)).mean()
    kyy = np.exp(-gamma * ((y[:, None] - y[None]) ** 2).sum(-1)).mean()
    kxy = np.exp(-gamma * ((x[:, None] - y[None]) ** 2).sum(-1)).mean()
    return float(kxx + kyy - 2.0 * kxy)
