"""Waveform preprocessing, cycle extraction and condition features."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from pathlib import Path

import numpy as np
import openpyxl
from scipy.signal import butter, find_peaks, resample, sosfiltfilt, welch


@dataclass
class ReferenceConfig:
    sample_rate: float = 500.0
    seq_len: int = 9600
    sheet_name: str = "过滤后波形"
    ppg_filter_low_hz: float = 0.5
    ppg_filter_high_hz: float = 8.0
    peak_min_distance_sec: float = 0.35
    peak_prominence: float = 0.25

    @classmethod
    def from_mapping(cls, values):
        # Require an explicit preprocessing specification.
        return cls(**{field.name: values[field.name] for field in fields(cls)})


# Both reference and application configuration objects expose these fields.
RunConfig = ReferenceConfig


COND_NAMES = (
    "duration_norm",
    "heart_rate_norm",
    "dominant_hz_norm",
    "amplitude",
    "rise_fraction",
    "decay_fraction",
    "rise_time_norm",
    "decay_time_norm",
    "rise_slope_norm",
    "decay_slope_norm",
    "pulse_area",
    "late_decay_ratio",
    "diff1_scaled",
    "diff2_scaled",
)


def minmax_1d(x: np.ndarray) -> np.ndarray:
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def preprocess_wave(wave: np.ndarray, cfg: RunConfig) -> np.ndarray:
    return ppg_bandpass(robust_normalize(wave), cfg).astype(np.float32)


def detect_troughs(x: np.ndarray, cfg: RunConfig) -> np.ndarray:
    distance = max(1, int(cfg.peak_min_distance_sec * cfg.sample_rate))
    prominence = max(cfg.peak_prominence, 0.12 * float(np.std(x)))
    troughs, _ = find_peaks(-x, distance=distance, prominence=prominence)
    return troughs.astype(np.int64)


def read_source_wave(path: Path, group: str, cfg: RunConfig) -> np.ndarray:
    if group == "bidmc":
        wave = read_bidmc_pleth(path)
        if abs(cfg.sample_rate - 125.0) > 1e-6:
            wave = resample(wave, int(round(wave.size * cfg.sample_rate / 125.0))).astype(
                np.float32
            )
        return wave.astype(np.float32)
    if group == "vitaldb":
        with np.load(path) as data:
            wave = data["ppg"].astype(np.float32)
            source_rate = float(data["sample_rate"])
        if abs(cfg.sample_rate - source_rate) > 1e-6:
            wave = resample(wave, int(round(wave.size * cfg.sample_rate / source_rate))).astype(
                np.float32
            )
        return wave
    if group == "ppgdalia":
        with np.load(path) as data:
            wave = data["ppg"].astype(np.float32)
            source_rate = float(data["sample_rate"])
        if abs(cfg.sample_rate - source_rate) > 1e-6:
            wave = resample(wave, int(round(wave.size * cfg.sample_rate / source_rate))).astype(
                np.float32
            )
        return wave
    return read_waveform(path, cfg.sheet_name)


def valid_cycle(cycle: np.ndarray, sample_rate: float) -> bool:
    if cycle.size < int(0.32 * sample_rate) or cycle.size > int(1.65 * sample_rate):
        return False
    if not np.all(np.isfinite(cycle)) or float(np.std(cycle)) < 1e-4:
        return False
    norm = minmax_1d(cycle)
    peak = int(np.argmax(norm))
    peak_pos = peak / max(1, norm.size - 1)
    return 0.08 <= peak_pos <= 0.80 and float(norm[peak]) > 0.55


def cycle_condition_features(
    cycle: np.ndarray, raw_len: int, sample_rate: float
) -> tuple[np.ndarray, dict[str, float]]:
    raw_amp = float(np.percentile(cycle, 95) - np.percentile(cycle, 5))
    x = minmax_1d(cycle)
    peak = int(np.argmax(x))
    duration_sec = raw_len / sample_rate
    rise_time = max(1.0, float(peak)) / sample_rate
    decay_time = max(1.0, float(len(x) - 1 - peak)) / sample_rate
    amp = float(np.clip(raw_amp / 6.0, 0.0, 1.0))
    rise_slope = amp / max(rise_time, 1e-6)
    decay_slope = amp / max(decay_time, 1e-6)
    area = float(np.trapezoid(x) / max(1, len(x)))
    late_idx = min(len(x) - 1, int(peak + 0.65 * max(1, len(x) - 1 - peak)))
    late_decay_ratio = float((x[late_idx] - x[-1]) / (amp + 1e-6))
    diff1 = float(np.mean(np.abs(np.diff(x))))
    diff2 = float(np.mean(np.abs(np.diff(x, n=2))))
    freqs, psd = welch(x, fs=sample_rate, nperseg=min(len(x), 128))
    band = (freqs > 0.2) & (freqs < 12.0)
    dom_hz = (
        float(freqs[band][np.argmax(psd[band])]) if np.any(band) else 1.0 / max(duration_sec, 1e-6)
    )
    hr = 60.0 / max(duration_sec, 1e-6)
    rise_frac = peak / max(1.0, float(len(x) - 1))
    decay_frac = 1.0 - rise_frac
    cond = np.asarray(
        [
            np.clip(duration_sec / 2.0, 0.0, 1.0),
            np.clip(hr / 180.0, 0.0, 1.0),
            np.clip(dom_hz / 6.0, 0.0, 1.0),
            amp,
            np.clip(rise_frac, 0.0, 1.0),
            np.clip(decay_frac, 0.0, 1.0),
            np.clip(rise_time / 1.0, 0.0, 1.0),
            np.clip(decay_time / 1.5, 0.0, 1.0),
            np.clip(rise_slope / 8.0, 0.0, 1.0),
            np.clip(decay_slope / 8.0, 0.0, 1.0),
            np.clip(area, 0.0, 1.0),
            np.clip(late_decay_ratio, 0.0, 1.0),
            diff1 * 50.0,
            diff2 * 500.0,
        ],
        dtype=np.float32,
    )
    meta = {
        "duration_sec": duration_sec,
        "heart_rate": hr,
        "dominant_hz": dom_hz,
        "amplitude": raw_amp,
        "rise_fraction": rise_frac,
        "decay_fraction": decay_frac,
        "rise_time_sec": rise_time,
        "decay_time_sec": decay_time,
        "rise_slope": rise_slope,
        "decay_slope": decay_slope,
        "pulse_area": area,
        "late_decay_ratio": late_decay_ratio,
        "diff1": diff1,
        "diff2": diff2,
    }
    return cond, meta


def extract_cycles_segments(
    rows: list[tuple[Path, int, str]],
    cfg: RunConfig,
    cycle_len: int,
    beat_count: int,
    max_cycles_per_file: int,
    max_segments_per_file: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    cycles: list[np.ndarray] = []
    conds: list[np.ndarray] = []
    durations: list[float] = []
    cycle_sources: list[str] = []
    segment_cond_sequences: list[np.ndarray] = []
    segment_duration_sequences: list[np.ndarray] = []
    segment_cycle_sequences: list[np.ndarray] = []
    segments: list[np.ndarray] = []
    segment_sources: list[str] = []
    for path, _label, group in rows:
        try:
            wave = read_source_wave(path, group, cfg)
            filtered = preprocess_wave(wave, cfg)
            troughs = detect_troughs(filtered, cfg)
        except Exception as exc:
            print(f"SKIP {path}: {exc}", flush=True)
            continue
        if troughs.size < beat_count + 1:
            continue

        file_cycles: list[np.ndarray] = []
        file_conds: list[np.ndarray] = []
        file_durations: list[float] = []
        for i in range(troughs.size - 1):
            start, end = int(troughs[i]), int(troughs[i + 1])
            cycle = filtered[start:end]
            if not valid_cycle(cycle, cfg.sample_rate):
                continue
            norm = minmax_1d(cycle)
            cond, _meta = cycle_condition_features(cycle, end - start, cfg.sample_rate)
            file_cycles.append(resample(norm, cycle_len).astype(np.float32))
            file_conds.append(cond)
            file_durations.append(float(end - start))
        if file_cycles:
            choose = np.linspace(
                0, len(file_cycles) - 1, min(max_cycles_per_file, len(file_cycles))
            ).astype(int)
            for idx in choose:
                cycles.append(file_cycles[int(idx)])
                conds.append(file_conds[int(idx)])
                durations.append(file_durations[int(idx)])
                cycle_sources.append(f"{group}:{path}")

        file_segments: list[np.ndarray] = []
        file_segment_cond_sequences: list[np.ndarray] = []
        file_segment_duration_sequences: list[np.ndarray] = []
        file_segment_cycle_sequences: list[np.ndarray] = []
        for i in range(troughs.size - beat_count):
            start, end = int(troughs[i]), int(troughs[i + beat_count])
            segment = filtered[start:end]
            if segment.size < int(beat_count * 0.32 * cfg.sample_rate):
                continue
            if segment.size > int(beat_count * 1.65 * cfg.sample_rate):
                continue
            if not np.all(np.isfinite(segment)) or np.std(segment) < 1e-4:
                continue
            seq_conds: list[np.ndarray] = []
            seq_durations: list[float] = []
            seq_cycles: list[np.ndarray] = []
            valid_subcycles = True
            for j in range(i, i + beat_count):
                c_start, c_end = int(troughs[j]), int(troughs[j + 1])
                cycle = filtered[c_start:c_end]
                if not valid_cycle(cycle, cfg.sample_rate):
                    valid_subcycles = False
                    break
                norm = minmax_1d(cycle)
                cond, _meta = cycle_condition_features(cycle, c_end - c_start, cfg.sample_rate)
                seq_conds.append(cond)
                seq_durations.append(float(c_end - c_start))
                seq_cycles.append(resample(norm, cycle_len).astype(np.float32))
            if not valid_subcycles:
                continue
            file_segments.append(
                minmax_1d(resample(segment, cfg.seq_len).astype(np.float32))[:, None]
            )
            file_segment_cond_sequences.append(np.stack(seq_conds).astype(np.float32))
            file_segment_duration_sequences.append(np.asarray(seq_durations, dtype=np.float32))
            file_segment_cycle_sequences.append(np.stack(seq_cycles).astype(np.float32))
        if file_segments:
            choose = np.linspace(
                0, len(file_segments) - 1, min(max_segments_per_file, len(file_segments))
            ).astype(int)
            for idx in choose:
                segments.append(file_segments[int(idx)])
                segment_cond_sequences.append(file_segment_cond_sequences[int(idx)])
                segment_duration_sequences.append(file_segment_duration_sequences[int(idx)])
                segment_cycle_sequences.append(file_segment_cycle_sequences[int(idx)])
                segment_sources.append(f"{group}:{path}")
    if not cycles:
        raise RuntimeError("No valid local PPG cycles were extracted.")
    if not segments:
        raise RuntimeError("No valid local PPG segments were extracted.")
    return (
        np.stack(cycles).astype(np.float32),
        np.stack(conds).astype(np.float32),
        np.asarray(durations, dtype=np.float32),
        cycle_sources,
        np.stack(segment_cond_sequences).astype(np.float32),
        np.stack(segment_duration_sequences).astype(np.float32),
        np.stack(segment_cycle_sequences).astype(np.float32),
        np.stack(segments).astype(np.float32),
        segment_sources,
    )


def read_waveform(path: Path, sheet_name: str) -> np.ndarray:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet_name]
    values = []
    for idx, row in enumerate(ws.iter_rows(values_only=True)):
        if idx == 0:
            continue
        val = row[0]
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            values.append(float(val))
    wb.close()
    return np.asarray(values, dtype=np.float32)


def read_bidmc_pleth(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, usecols=2, dtype=np.float32)


def robust_normalize(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    q25, q75 = np.percentile(x, [25, 75])
    scale = (q75 - q25) / 1.349
    if not np.isfinite(scale) or scale < 1e-6:
        scale = np.std(x) + 1e-6
    return ((x - med) / scale).astype(np.float32)


def ppg_bandpass(x: np.ndarray, cfg: RunConfig) -> np.ndarray:
    high = min(cfg.ppg_filter_high_hz, cfg.sample_rate * 0.45)
    if x.size < 32 or cfg.ppg_filter_low_hz >= high:
        return x.astype(np.float32)
    sos = butter(
        4, [cfg.ppg_filter_low_hz, high], btype="bandpass", fs=cfg.sample_rate, output="sos"
    )
    return sosfiltfilt(sos, x).astype(np.float32)


def minmax_segments(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(x.shape[0], -1)
    lo = flat.min(axis=1, keepdims=True)
    hi = flat.max(axis=1, keepdims=True)
    return ((flat - lo) / (hi - lo + 1e-6)).reshape(x.shape).astype(np.float32)
