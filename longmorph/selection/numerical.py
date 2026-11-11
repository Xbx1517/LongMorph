"""Training-calibrated candidate losses, admissibility gates and numerical choice."""

from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks, resample, savgol_filter

AXES = ("raw", "d1", "d2", "spectral")


ROLES = ("morphology", "detail", "rhythm", "continuity")


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return ((x - x.min()) / (x.max() - x.min() + 1e-6)).astype(np.float32)


def _cycle_axes(cycles: np.ndarray, cycle_len: int = 128) -> dict[str, np.ndarray]:
    normalized = np.stack([_minmax(resample(np.asarray(c).reshape(-1), cycle_len)) for c in cycles])
    mean_cycle, std_cycle = normalized.mean(0), normalized.std(0)
    d1 = np.diff(mean_cycle, prepend=mean_cycle[:1])
    d2 = np.diff(d1, prepend=d1[:1])
    power = np.abs(np.fft.rfft(mean_cycle, n=256)).astype(np.float32) ** 2
    power /= float(power.sum()) + 1e-8
    return {
        "raw": np.concatenate([mean_cycle[::2], std_cycle[::4]]).astype(np.float32),
        "d1": np.concatenate([d1[::2], np.abs(d1)[::4]]).astype(np.float32),
        "d2": np.concatenate([d2[::2], np.abs(d2)[::4]]).astype(np.float32),
        "spectral": np.log1p(1e4 * power[1:49]).astype(np.float32),
    }


def _event_detail_features(cycles: np.ndarray, cycle_len: int = 128) -> np.ndarray:
    """Compact shoulder/notch/decline energy card after phase normalization.

    The windows are intentionally broad: the Agent receives measured support
    statistics, not a brittle hand-labelled notch location.
    """
    normalized = np.stack([_minmax(resample(np.asarray(c).reshape(-1), cycle_len)) for c in cycles])
    d1 = np.diff(normalized, axis=1, prepend=normalized[:, :1])
    d2 = np.diff(d1, axis=1, prepend=d1[:, :1])
    shoulder = slice(int(0.10 * cycle_len), int(0.38 * cycle_len))
    decline = slice(int(0.30 * cycle_len), int(0.82 * cycle_len))
    event = slice(int(0.10 * cycle_len), int(0.82 * cycle_len))
    return np.asarray(
        [
            np.mean(np.abs(d2[:, shoulder])),
            np.mean(np.abs(d2[:, decline])),
            np.sqrt(np.mean(np.square(d2[:, event])) + 1e-12),
            np.quantile(np.abs(d2[:, event]), 0.90),
            np.mean(np.abs(d1[:, decline])),
        ],
        dtype=np.float32,
    )


def _per_cycle_shape_features(cycles: np.ndarray, cycle_len: int = 128) -> np.ndarray:
    """Subset of the paper shape audit, expressed in phase-normalized units."""
    rows = []
    for cycle in cycles:
        x = _minmax(resample(np.asarray(cycle).reshape(-1), cycle_len))
        peak = int(np.argmax(x))
        d1 = np.diff(x)
        d2 = np.diff(x, n=2)
        post = x[peak + max(2, int(0.05 * cycle_len)) :]
        tail_start = min(cycle_len - 2, peak + int(0.60 * max(1, cycle_len - 1 - peak)))
        up = float(np.max(d1[: max(1, peak)])) if len(d1) else 0.0
        down = float(abs(np.min(d1[peak:]))) if peak < len(d1) else 0.0
        rows.append(
            [
                np.mean(x >= 0.50),
                np.mean(x >= 0.75),
                np.mean(x),
                peak / max(cycle_len - 1, 1),
                np.mean(np.diff(x[tail_start:])) if tail_start < cycle_len - 2 else 0.0,
                max(0.0, float(np.max(post) - np.min(post))) if len(post) >= 5 else 0.0,
                up / (down + 1e-6),
                float(np.min(d2)) if len(d2) else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _block_shape_features(cycles: np.ndarray, cycle_len: int = 128) -> np.ndarray:
    rows = _per_cycle_shape_features(cycles, cycle_len)
    return np.concatenate([rows.mean(0), rows.std(0)]).astype(np.float32)


def _cycle_residual_rms(cycle: np.ndarray, cycle_len: int = 128) -> float:
    x = _minmax(resample(np.asarray(cycle).reshape(-1), cycle_len))
    smooth = savgol_filter(x, 15, 3) if cycle_len >= 15 else x
    return float(np.sqrt(np.mean(np.square(x - smooth)) + 1e-12))


def _seam(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left).reshape(-1), np.asarray(right).reshape(-1)
    scale = float(np.std(np.r_[left, right])) + 1e-6
    value = abs(float(left[-1] - right[0])) / scale
    slope = abs(float((left[-1] - left[-2]) - (right[1] - right[0]))) / scale
    return value + 0.25 * slope


def _robust_calibration(rows: np.ndarray) -> dict[str, np.ndarray]:
    rows = np.asarray(rows, dtype=np.float32)
    median = np.median(rows, axis=0)
    mad = 1.4826 * np.median(np.abs(rows - median), axis=0)
    std = rows.std(0)
    standardized = (rows - median) / np.where(mad > 1e-6, mad, np.maximum(std, 1e-6))
    return {
        "median": median.astype(np.float32),
        "scale": np.where(mad > 1e-6, mad, np.maximum(std, 1e-6)).astype(np.float32),
        "target_low": np.quantile(rows, 0.05, axis=0).astype(np.float32),
        "target_high": np.quantile(rows, 0.95, axis=0).astype(np.float32),
        "support_low": np.quantile(rows, 0.01, axis=0).astype(np.float32),
        "support_high": np.quantile(rows, 0.99, axis=0).astype(np.float32),
        "train_z_mean": standardized.mean(0).astype(np.float32),
        "train_z_std": np.maximum(standardized.std(0), 1e-3).astype(np.float32),
    }


def _split_stitched(x: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    bounds = np.r_[0, np.cumsum(np.asarray(lengths, dtype=int))]
    return [x[int(bounds[i]) : int(bounds[i + 1])] for i in range(len(lengths))]


def build_calibration(
    train_cycles: np.ndarray,
    train_lengths: np.ndarray,
    smooth_window: int = 15,
    smoothing_mode: str = "global",
) -> dict:
    """Calibrate axes from every real block after the same resample/stitch/SG path."""
    axis_rows = {axis: [] for axis in AXES}
    event_rows, shape_rows, operator_rows, residual_rows, block_owners = [], [], [], [], []
    seams = []
    train_cycles = np.asarray(train_cycles, dtype=np.float32)
    train_lengths = np.asarray(train_lengths, dtype=int)
    for owner, (row, lengths) in enumerate(zip(train_cycles, train_lengths)):
        stitched_row = _temporary_stitch(list(row), lengths, smooth_window, smoothing_mode)
        processed_row = _split_stitched(stitched_row, lengths)
        for start in range(0, len(row) - 2, 3):
            # Candidate scoring normalizes and smooths each decoded 3-cycle
            # proposal before prefix stitching.  Calibrate morphology through
            # that identical path; retain the full-row path only for seams.
            stitched_block = _temporary_stitch(
                list(row[start : start + 3]),
                lengths[start : start + 3],
                smooth_window,
                smoothing_mode,
            )
            processed_block = _split_stitched(stitched_block, lengths[start : start + 3])
            features = _cycle_axes(processed_block)
            for axis in AXES:
                axis_rows[axis].append(features[axis])
            event_rows.append(_event_detail_features(processed_block))
            shape_rows.append(_block_shape_features(processed_block))
            per_cycle_shape = _per_cycle_shape_features(processed_block)
            operator_rows.extend(per_cycle_shape)
            residual_rows.extend(_cycle_residual_rms(cycle) for cycle in processed_block)
            block_owners.append(owner)
            if start > 0:
                seams.append(_seam(processed_row[start - 1], processed_row[start]))
    if not seams:
        seams = [0.0]
    axes = {axis: _robust_calibration(np.stack(axis_rows[axis])) for axis in AXES}
    raw_rows = np.stack(axis_rows["raw"]).astype(np.float32)
    raw_z = (raw_rows - axes["raw"]["median"]) / axes["raw"]["scale"]
    # Leave-sequence-out nearest-neighbour distances define how close an
    # independent real block normally lies to the training bank.  This makes
    # exact/near retrieval copying measurable without exposing waveforms to an LLM.
    from sklearn.neighbors import NearestNeighbors

    query_k = min(max(2, min(64, len(raw_z))), len(raw_z))
    distances, indices = NearestNeighbors(n_neighbors=query_k).fit(raw_z).kneighbors(raw_z)
    owners = np.asarray(block_owners, dtype=np.int64)
    loo_distances = []
    for owner, row_distances, row_indices in zip(owners, distances, indices):
        different = owners[row_indices] != owner
        eligible = row_distances[different]
        loo_distances.append(float(eligible[0] if len(eligible) else row_distances[-1]))
    loo_distances = np.asarray(loo_distances, dtype=np.float32) / np.sqrt(raw_z.shape[1])
    return {
        "axes": axes,
        "event_detail": _robust_calibration(np.stack(event_rows)),
        "shape": _robust_calibration(np.stack(shape_rows)),
        "operator_targets": {
            "width75_median": float(np.median(np.asarray(operator_rows)[:, 1])),
            "peak_phase_median": float(np.median(np.asarray(operator_rows)[:, 3])),
            "residual_rms_median": float(np.median(residual_rows)),
        },
        "copy_bank_z": raw_z.astype(np.float32),
        "copy_real_p01": float(np.quantile(loo_distances, 0.01)),
        "copy_real_p10": float(np.quantile(loo_distances, 0.10)),
        "copy_real_median": float(np.median(loo_distances)),
        "seam": _robust_calibration(np.asarray(seams, dtype=np.float32)[:, None]),
        "real_blocks": int(len(axis_rows["raw"])),
        "real_sequences": int(len(train_cycles)),
        "smooth_window": int(smooth_window),
        "smoothing_mode": smoothing_mode,
    }


def calibration_summary(calibration: dict) -> dict:
    """Small JSON-safe provenance summary; full high-dimensional arrays stay in memory."""
    return {
        "real_sequences": calibration["real_sequences"],
        "real_blocks": calibration["real_blocks"],
        "smooth_window": int(calibration.get("smooth_window", 15)),
        "smoothing_mode": calibration.get("smoothing_mode", "global"),
        "axes": {
            axis: {
                "dimensions": int(len(calibration["axes"][axis]["median"])),
                "median_abs": float(np.mean(np.abs(calibration["axes"][axis]["median"]))),
                "scale_mean": float(np.mean(calibration["axes"][axis]["scale"])),
            }
            for axis in AXES
        },
        "seam_target_05_95": [
            float(calibration["seam"]["target_low"][0]),
            float(calibration["seam"]["target_high"][0]),
        ],
        "event_detail": {
            "dimensions": int(len(calibration["event_detail"]["median"])),
            "median": [float(x) for x in calibration["event_detail"]["median"]],
            "target_05": [float(x) for x in calibration["event_detail"]["target_low"]],
            "target_95": [float(x) for x in calibration["event_detail"]["target_high"]],
        },
        "shape": {
            "dimensions": int(len(calibration["shape"]["median"])),
            "median_abs": float(np.mean(np.abs(calibration["shape"]["median"]))),
            "scale_mean": float(np.mean(calibration["shape"]["scale"])),
        },
        "operator_targets": calibration["operator_targets"],
        "copy_distance": {
            "leave_sequence_out_p01": float(calibration["copy_real_p01"]),
            "leave_sequence_out_p10": float(calibration["copy_real_p10"]),
            "leave_sequence_out_median": float(calibration["copy_real_median"]),
        },
    }


def _support_score(row: np.ndarray, cal: dict) -> tuple[float, float]:
    row = np.asarray(row, dtype=np.float32)
    scale = cal["scale"]
    tail = np.maximum(cal["target_low"] - row, 0.0) + np.maximum(row - cal["target_high"], 0.0)
    support = np.maximum(cal["support_low"] - row, 0.0) + np.maximum(row - cal["support_high"], 0.0)
    center = np.abs((row - cal["median"]) / scale)
    # Axis dimensionality must not change its vote weight.
    total = float(
        np.mean(np.clip(tail / scale, 0.0, 8.0)) + 0.05 * np.mean(np.clip(center, 0.0, 8.0))
    )
    outside = float(np.mean(np.clip(support / scale, 0.0, 8.0)))
    return total, outside


def batch_distribution_score(
    features: dict[str, np.ndarray], accepted: list[dict[str, np.ndarray]], calibration: dict
) -> tuple[float, dict[str, float]]:
    """Distance of the generated block buffer from the full real block distribution."""
    axis_scores = {}
    for axis in AXES:
        rows = [item[axis] for item in accepted] + [features[axis]]
        cal = calibration["axes"][axis]
        z = (np.stack(rows).astype(np.float32) - cal["median"]) / cal["scale"]
        mean_gap = float(np.mean(np.clip(np.abs(z.mean(0) - cal["train_z_mean"]), 0.0, 8.0)))
        if len(z) < 4:
            spread_gap = 0.0
        else:
            ratio = np.maximum(z.std(0), 1e-3) / cal["train_z_std"]
            spread_gap = float(np.mean(np.clip(np.abs(np.log(ratio)), 0.0, 4.0)))
        axis_scores[axis] = mean_gap + 0.25 * spread_gap
    return float(np.mean(list(axis_scores.values()))), axis_scores


def _apply_stitch_smoothing(
    x: np.ndarray,
    seams: np.ndarray,
    smooth_window: int,
    smoothing_mode: str,
) -> np.ndarray:
    """Apply global or seam-local smoothing after cycle stitching."""
    if smoothing_mode == "none" or smooth_window < 7:
        return x
    if smoothing_mode == "global":
        if len(x) < smooth_window:
            return x
        return savgol_filter(x, smooth_window, 3).astype(np.float32)
    if smoothing_mode != "seam_local":
        raise ValueError(f"unknown stitch smoothing mode: {smoothing_mode}")
    effective = min(int(smooth_window), len(x) if len(x) % 2 else len(x) - 1)
    if effective % 2 == 0:
        effective -= 1
    if effective < 5:
        return x
    filtered = savgol_filter(x, effective, min(3, effective - 1)).astype(np.float32)
    radius = effective // 2
    blend = np.zeros(len(x), dtype=np.float32)
    for seam in np.asarray(seams, dtype=np.int64):
        lo, hi = max(0, int(seam) - radius), min(len(x), int(seam) + radius + 1)
        if hi <= lo:
            continue
        distance = np.abs(np.arange(lo, hi, dtype=np.float32) - float(seam))
        weight = 0.5 * (1.0 + np.cos(np.pi * np.minimum(distance / (radius + 1.0), 1.0)))
        blend[lo:hi] = np.maximum(blend[lo:hi], weight.astype(np.float32))
    return (x + blend * (filtered - x)).astype(np.float32)


def _temporary_stitch(
    cycles: list[np.ndarray],
    lengths: np.ndarray,
    smooth_window: int,
    smoothing_mode: str = "global",
) -> np.ndarray:
    total = int(np.sum(lengths))
    parts = [resample(c, int(n)).astype(np.float32) for c, n in zip(cycles, lengths)]
    x = np.concatenate(parts)
    seams = np.cumsum([len(part) for part in parts], dtype=np.int64)[:-1]
    if len(x) != total:
        scale = float(total) / max(len(x), 1)
        x = resample(x, total).astype(np.float32)
        seams = np.rint(seams * scale).astype(np.int64)
    x = _apply_stitch_smoothing(x, seams, smooth_window, smoothing_mode)
    return _minmax(x)


def evaluate_candidate(
    block_cycles: np.ndarray,
    block_lengths: np.ndarray,
    prefix_cycles: list[np.ndarray],
    prefix_lengths: np.ndarray,
    detail_state: np.ndarray,
    calibration: dict,
    sample_rate: float,
    smooth_window: int,
    accepted_features: list[dict[str, np.ndarray]] | None = None,
    critic_penalty: float = 0.0,
    smoothing_mode: str = "global",
) -> tuple[dict, dict[str, np.ndarray]]:
    processed_block = _temporary_stitch(
        list(block_cycles), block_lengths, smooth_window, smoothing_mode
    )
    processed_cycles = _split_stitched(processed_block, block_lengths)
    features = _cycle_axes(processed_cycles)
    event_features = _event_detail_features(processed_cycles)
    shape_features = _block_shape_features(processed_cycles)
    axis_scores, axis_outside = {}, {}
    for axis in AXES:
        axis_scores[axis], axis_outside[axis] = _support_score(
            features[axis], calibration["axes"][axis]
        )

    all_cycles = list(prefix_cycles) + list(block_cycles)
    all_lengths = np.r_[prefix_lengths, block_lengths].astype(int)
    temporary = _temporary_stitch(all_cycles, all_lengths, smooth_window, smoothing_mode)
    processed_all = _split_stitched(temporary, all_lengths)
    seam_value = (
        _seam(processed_all[len(prefix_cycles) - 1], processed_all[len(prefix_cycles)])
        if prefix_cycles
        else 0.0
    )
    seam_score, seam_outside = _support_score(np.asarray([seam_value]), calibration["seam"])
    peaks, _ = find_peaks(temporary, prominence=0.12, distance=max(1, int(0.16 * sample_rate)))
    expected = len(all_cycles)
    peak_error = float(abs(len(peaks) - expected) / max(expected, 1))
    detail_state = np.asarray(detail_state, dtype=np.float32)
    latent_penalty = float(np.mean(np.maximum(np.abs(detail_state) - 3.0, 0.0)))
    axis_total = float(np.mean([axis_scores[axis] for axis in AXES]))
    outside_total = float(np.mean([axis_outside[axis] for axis in AXES]))
    event_detail_score, event_detail_outside = _support_score(
        event_features, calibration["event_detail"]
    )
    shape_score, shape_outside = _support_score(shape_features, calibration["shape"])
    raw_z = (features["raw"] - calibration["axes"]["raw"]["median"]) / calibration["axes"]["raw"][
        "scale"
    ]
    copy_distance = float(
        np.min(np.sqrt(np.mean(np.square(calibration["copy_bank_z"] - raw_z), axis=1)))
    )
    copy_threshold = float(calibration["copy_real_p10"])
    copy_penalty = float(max(copy_threshold - copy_distance, 0.0) / (copy_threshold + 1e-6))
    batch_total, batch_axes = batch_distribution_score(
        features, accepted_features or [], calibration
    )
    total = (
        axis_total
        + 0.15 * seam_score
        + 0.35 * peak_error
        + 0.15 * latent_penalty
        + 0.25 * batch_total
        + 0.35 * critic_penalty
        + 0.20 * event_detail_score
        + 0.35 * shape_score
        + 0.20 * copy_penalty
    )
    return {
        "axis_scores": axis_scores,
        "axis_outside": axis_outside,
        "axis_total": axis_total,
        "outside_total": outside_total,
        "event_detail_score": float(event_detail_score),
        "event_detail_outside": float(event_detail_outside),
        "event_detail_features": [float(x) for x in event_features],
        "shape_score": float(shape_score),
        "shape_outside": float(shape_outside),
        "shape_features": [float(x) for x in shape_features],
        "copy_distance": copy_distance,
        "copy_real_p10": copy_threshold,
        "copy_penalty": copy_penalty,
        "seam_value": float(seam_value),
        "seam_score": float(seam_score),
        "seam_outside": float(seam_outside),
        "prefix_peak_count": int(len(peaks)),
        "prefix_expected_peaks": int(expected),
        "peak_error": peak_error,
        "latent_penalty": latent_penalty,
        "batch_distribution": batch_total,
        "batch_axes": batch_axes,
        "critic_penalty": float(critic_penalty),
        "total": float(total),
    }, features


def _pareto_front(records: list[dict], indices: list[int]) -> list[int]:
    keys = ["raw", "d1", "d2", "spectral"]
    vectors = {
        i: np.asarray(
            [records[i]["metrics"]["axis_scores"][k] for k in keys]
            + [
                records[i]["metrics"]["seam_score"],
                records[i]["metrics"]["event_detail_score"],
                records[i]["metrics"]["shape_score"],
                records[i]["metrics"]["copy_penalty"],
            ]
        )
        for i in indices
    }
    front = []
    for i in indices:
        dominated = any(
            j != i
            and np.all(vectors[j] <= vectors[i] + 1e-10)
            and np.any(vectors[j] < vectors[i] - 1e-10)
            for j in indices
        )
        if not dominated:
            front.append(i)
    return front


def safe_candidates(
    records: list[dict],
    axis_tolerance: float,
    total_tolerance: float,
    critic_tolerance: float,
    batch_tolerance: float,
) -> tuple[list[int], dict]:
    nominal = records[0]["metrics"]
    safe, reasons = [0], {"C0": ["NOMINAL_FALLBACK"]}
    for i, record in enumerate(records[1:], 1):
        metric = record["metrics"]
        failed = []
        if metric["peak_error"] > max(
            1.0 / max(metric["prefix_expected_peaks"], 1), nominal["peak_error"] + 0.05
        ):
            failed.append("PEAK_COUNT")
        if metric["latent_penalty"] > max(0.5, nominal["latent_penalty"] + 0.10):
            failed.append("LATENT_SUPPORT")
        if metric["seam_score"] > nominal["seam_score"] + 2.0:
            failed.append("SEAM")
        if metric["axis_total"] > nominal["axis_total"] + total_tolerance:
            failed.append("DETAIL_TOTAL")
        if metric["critic_penalty"] > nominal["critic_penalty"] + critic_tolerance:
            failed.append("FROZEN_CRITIC")
        if metric["batch_distribution"] > nominal["batch_distribution"] + batch_tolerance:
            failed.append("BATCH_DISTRIBUTION")
        if metric["event_detail_score"] > nominal["event_detail_score"] + total_tolerance:
            failed.append("EVENT_DETAIL")
        if metric["shape_score"] > nominal["shape_score"] + total_tolerance:
            failed.append("GLOBAL_SHAPE")
        if metric["copy_penalty"] > nominal["copy_penalty"] + 0.05:
            failed.append("TRAIN_COPY")
        for axis in AXES:
            if metric["axis_scores"][axis] > nominal["axis_scores"][axis] + axis_tolerance:
                failed.append(axis.upper())
        reasons[f"C{i}"] = failed or ["SAFE"]
        if not failed and np.isfinite(metric["total"]):
            safe.append(i)
    return safe, reasons


def select_candidate(
    records: list[dict],
    axis_tolerance: float = 0.04,
    total_tolerance: float = 0.02,
    critic_tolerance: float = 0.05,
    batch_tolerance: float = 0.03,
    require_improvement: bool = False,
    min_total_gain: float = 0.0,
    selector_mode: str = "numerical",
) -> tuple[int, dict]:
    """Apply fixed safety gates, then minimize total loss over admissible candidates."""
    if selector_mode != "numerical":
        raise ValueError("This function implements numerical selection")
    safe, gate_reasons = safe_candidates(
        records, axis_tolerance, total_tolerance, critic_tolerance, batch_tolerance
    )
    nominal_total = float(records[0]["metrics"]["total"])
    gain_by_candidate = {
        f"C{i}": float(nominal_total - records[i]["metrics"]["total"]) for i in range(len(records))
    }
    if require_improvement:
        improved = [
            i for i in safe if i == 0 or gain_by_candidate[f"C{i}"] >= float(min_total_gain)
        ]
        selectable = improved if any(i != 0 for i in improved) else [0]
    else:
        selectable = safe
    front = _pareto_front(records, selectable)
    numeric = min(selectable, key=lambda i: records[i]["metrics"]["total"])
    votes = []
    selected = numeric
    return selected, {
        "safe_candidates": [f"C{i}" for i in safe],
        "selectable_candidates": [f"C{i}" for i in selectable],
        "pareto_candidates": [f"C{i}" for i in front],
        "numeric_choice": f"C{numeric}",
        "selected": f"C{selected}",
        "random_safe_pool": [f"C{i}" for i in safe] if selector_mode == "random_safe" else [],
        "require_improvement": bool(require_improvement),
        "min_total_gain": float(min_total_gain),
        "gain_by_candidate": gain_by_candidate,
        "gate_reasons": gate_reasons,
        "votes": votes,
        "selector_mode": selector_mode,
        "disabled_roles": [],
    }
