"""Native-duration waveform extraction and shared-coordinate cycle rendering."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, resample


def minmax_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def load_npz_array(path: Path, preferred_keys: tuple[str, ...]) -> np.ndarray:
    data = np.load(path)
    for key in preferred_keys:
        if key in data:
            return np.asarray(data[key], dtype=np.float32)
    if len(data.files) == 1:
        return np.asarray(data[data.files[0]], dtype=np.float32)
    raise KeyError(f"{path} does not contain any of {preferred_keys}; keys={data.files}")


def cycle_boundaries_from_segment(
    x: np.ndarray, sample_rate: float, expected_beats: int = 0
) -> np.ndarray:
    distance = max(1, int(0.28 * sample_rate))
    for prom in (0.10, 0.06, 0.035, 0.02, 0.01):
        troughs, _ = find_peaks(-x, prominence=prom, distance=distance)
        if len(troughs) >= 3:
            return troughs.astype(int)
    for prom in (0.10, 0.06, 0.035, 0.02, 0.01):
        peaks, _ = find_peaks(x, prominence=prom, distance=distance)
        if len(peaks) >= 2:
            mids = ((peaks[:-1] + peaks[1:]) // 2).astype(int)
            return np.concatenate([[0], mids, [len(x) - 1]]).astype(int)
    if expected_beats > 1:
        return np.linspace(0, len(x) - 1, int(expected_beats) + 1).round().astype(int)
    return np.asarray([0, len(x) - 1], dtype=int)


def extract_aligned_cycles_with_fiducials(
    segments: np.ndarray,
    sample_rate: float,
    cycle_len: int = 256,
    max_cycles: int = 0,
    label: int = 1,
    expected_beats: int = 0,
) -> tuple[np.ndarray, pd.DataFrame]:
    flat = segments.reshape(segments.shape[0], segments.shape[1], -1)[:, :, 0]
    cycles: list[np.ndarray] = []
    rows: list[dict[str, float | int]] = []
    for segment_idx, x0 in enumerate(flat):
        x = minmax_1d(x0)
        bounds = cycle_boundaries_from_segment(x, sample_rate, expected_beats=expected_beats)
        for cycle_idx, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            start = int(start)
            end = int(end)
            raw = x[start:end]
            if raw.size < max(8, int(0.25 * sample_rate)) or raw.size > int(1.8 * sample_rate):
                continue
            if not np.all(np.isfinite(raw)) or float(np.std(raw)) < 1e-5:
                continue
            norm = minmax_1d(raw)
            peak_idx = int(np.argmax(norm))
            aligned = minmax_1d(resample(norm, cycle_len).astype(np.float32))
            cycles.append(aligned)
            rows.append(
                {
                    "label": int(label),
                    "segment_idx": int(segment_idx),
                    "cycle_idx": int(cycle_idx),
                    "start": int(start),
                    "end": int(end),
                    "raw_len": int(end - start),
                    "rr_ms": float((end - start) / sample_rate * 1000.0),
                    "hr_bpm": float(60000.0 / max((end - start) / sample_rate * 1000.0, 1e-6)),
                    "amplitude": float(np.max(raw) - np.min(raw)),
                    "peak_amp": float(np.max(raw)),
                    "onset_value": float(raw[0]),
                    "end_value": float(raw[-1]),
                    "peak_phase": float(peak_idx / max(1, raw.size - 1)),
                    "peak_index_raw": int(start + peak_idx),
                }
            )
            if max_cycles > 0 and len(cycles) >= max_cycles:
                return np.asarray(cycles, dtype=np.float32), pd.DataFrame(rows)
    if not cycles:
        return np.zeros((0, cycle_len), dtype=np.float32), pd.DataFrame(rows)
    return np.asarray(cycles, dtype=np.float32), pd.DataFrame(rows)


def allocate_lengths(raw_lengths: np.ndarray, total: int, min_len: int) -> np.ndarray:
    raw_lengths = np.maximum(raw_lengths.astype(np.float64), 1.0)
    lengths = np.maximum(np.round(raw_lengths / raw_lengths.sum() * total).astype(int), min_len)
    while lengths.sum() > total:
        idx = int(np.argmax(lengths))
        if lengths[idx] > min_len:
            lengths[idx] -= 1
        else:
            break
    while lengths.sum() < total:
        lengths[int(np.argmax(raw_lengths))] += 1
    return lengths


def shared_cycles(segments, durations, seq_len, beat_count, cycle_len):
    rows = []
    for segment, dur in zip(segments, durations):
        lens = allocate_lengths(dur, seq_len, max(16, seq_len // (beat_count * 4)))
        b = np.r_[0, np.cumsum(lens)]
        rows.append(
            np.stack(
                [
                    resample(segment[b[i] : b[i + 1]], cycle_len).astype(np.float32)
                    for i in range(beat_count)
                ]
            )
        )
    return np.stack(rows).astype(np.float32)


def rhythm_features(duration_seq: np.ndarray, sample_rate: float) -> np.ndarray:
    rr_ms = np.asarray(duration_seq, dtype=np.float32) / float(sample_rate) * 1000.0
    if rr_ms.size < 2:
        sdnn = 0.0
        rmssd = 0.0
    else:
        sdnn = float(np.std(rr_ms))
        rmssd = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
    mean_rr = float(np.mean(rr_ms))
    mean_hr = 60000.0 / max(mean_rr, 1e-6)
    return np.asarray(
        [
            np.clip(mean_rr / 1400.0, 0.0, 1.5),
            np.clip(sdnn / 300.0, 0.0, 2.0),
            np.clip(rmssd / 300.0, 0.0, 2.0),
            np.clip(mean_hr / 180.0, 0.0, 1.5),
        ],
        dtype=np.float32,
    )
