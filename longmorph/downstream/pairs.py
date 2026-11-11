"""Cycle-window construction with explicit subject and source separation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def real_pairs(data, subject_ids, input_cycles=5, target_cycles=1):
    required = input_cycles + target_cycles
    if input_cycles <= 0 or target_cycles <= 0 or data["cycles"].shape[1] < required:
        raise ValueError("Positive input/target lengths must fit the prepared cycle windows")
    mask = np.isin(data["subject_index"], list(subject_ids))
    cycles = np.asarray(data["cycles"][mask], dtype=np.float32)
    durations = np.asarray(data["durations_ms"][mask], dtype=np.float32)
    subjects = np.asarray(data["subject_index"][mask], dtype=int)
    return (
        cycles[:, :input_cycles],
        cycles[:, input_cycles:required],
        durations[:, input_cycles:required],
        subjects,
    )


def synthetic_pairs(path, expected_train_subjects, input_cycles=5, target_cycles=1):
    """Load exported pairs or cycle sequences, rejecting generator-subject leakage."""
    with np.load(path, allow_pickle=False) as archive:
        if "train_subject_index" not in archive:
            raise ValueError("Synthetic NPZ must record train_subject_index used by its generator")
        used = set(map(int, np.asarray(archive["train_subject_index"]).reshape(-1)))
        if used - set(expected_train_subjects):
            raise ValueError(
                f"Synthetic generator subjects outside the fold train set: {sorted(used-set(expected_train_subjects))}"
            )
        if "synthetic_x" in archive and "synthetic_y" in archive:
            x = np.asarray(archive["synthetic_x"], dtype=np.float32)
            y = np.asarray(archive["synthetic_y"], dtype=np.float32)
        else:
            cycles = np.asarray(archive["cycles"], dtype=np.float32)
            prefix = (
                int(np.asarray(archive["real_prefix_cycles"]).item())
                if "real_prefix_cycles" in archive
                else 0
            )
            stride = (
                int(np.asarray(archive["pair_stride"]).item())
                if "pair_stride" in archive
                else input_cycles
            )
            x, y = sequence_pairs(cycles, input_cycles, target_cycles, prefix, stride)
    if (
        x.ndim != 3
        or y.ndim != 3
        or x.shape[0] != y.shape[0]
        or x.shape[1:] != (input_cycles, 256)
        or y.shape[1:] != (target_cycles, 256)
    ):
        raise ValueError(
            f"Expected paired [N,{input_cycles},256] and [N,{target_cycles},256], got {x.shape} and {y.shape}"
        )
    if not len(x):
        raise ValueError("Synthetic forecasting pair pool is empty")
    return x, y


def sequence_pairs(cycles, input_cycles=5, target_cycles=1, prefix=3, stride=1):
    """Exclude conditioning cycles before slicing fully generated input/targets."""
    cycles = np.asarray(cycles)
    if cycles.ndim != 3 or cycles.shape[-1] != 256:
        raise ValueError("Cycle sequences must have shape [N,T,256]")
    if min(input_cycles, target_cycles, stride) <= 0 or prefix < 0:
        raise ValueError("Lengths and stride must be positive; prefix must be non-negative")
    pairs = [
        (row[s : s + input_cycles], row[s + input_cycles : s + input_cycles + target_cycles])
        for row in cycles
        for s in range(prefix, len(row) - input_cycles - target_cycles + 1, stride)
    ]
    if not pairs:
        return np.empty((0, input_cycles, 256), np.float32), np.empty(
            (0, target_cycles, 256), np.float32
        )
    return np.asarray([p[0] for p in pairs], np.float32), np.asarray(
        [p[1] for p in pairs], np.float32
    )


def subject_splits(data, folds=5):
    subjects = sorted(map(int, np.unique(data["subject_index"])))
    if folds <= 0 or len(subjects) < max(folds * 3, 6):
        raise ValueError(
            "Each fold requires three test subjects, two validation subjects and training subjects"
        )
    splits = []
    for fold in range(folds):
        test = set(subjects[fold * 3 : (fold + 1) * 3])
        remaining = [s for s in subjects if s not in test]
        splits.append(
            {
                "fold": fold + 1,
                "train": remaining[2:],
                "validation": remaining[:2],
                "test": sorted(test),
            }
        )
    return splits


def read_sources(path):
    return [s.strip() for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def split_sources(sources, validation_fraction, validation_min_sources, seed):
    unique = np.asarray(sorted(set(sources)), dtype=object)
    if len(unique) < validation_min_sources + 2:
        raise ValueError(f"Need at least {validation_min_sources+2} distinct training sources")
    shuffled = unique[np.random.default_rng(seed).permutation(len(unique))]
    count = min(
        max(validation_min_sources, int(round(validation_fraction * len(unique)))), len(unique) - 2
    )
    return set(map(str, shuffled[count:])), set(map(str, shuffled[:count]))


def source_windows(cycles, durations, sources, source_to_id, expected_cycles=24, stride=1):
    windows, window_durations, subjects = [], [], []
    for row, duration, source in zip(cycles, durations, sources):
        for start in range(0, min(expected_cycles, len(row)) - 5, stride):
            windows.append(row[start : start + 6])
            window_durations.append(duration[start : start + 6])
            subjects.append(source_to_id[source])
    if not windows:
        raise ValueError("No six-cycle windows could be built from source sequences")
    return (
        np.asarray(windows, np.float32),
        np.asarray(window_durations, np.float32),
        np.asarray(subjects, np.int32),
    )


def pairs_from_windows(windows, durations, subjects, input_cycles=5, target_cycles=1):
    return real_pairs(
        {"cycles": windows, "durations_ms": durations, "subject_index": subjects},
        set(subjects),
        input_cycles,
        target_cycles,
    )


def draw_indices(total, count, seed, replace=False):
    if count <= 0:
        return np.asarray([], dtype=int)
    if count > total and not replace:
        raise ValueError(f"Cannot draw {count} items without replacement from {total}")
    return np.asarray(
        np.random.default_rng(seed).choice(total, count, replace=replace or count > total),
        dtype=int,
    )


def limit_windows(x, y, durations, subjects, count, seed):
    if count <= 0 or len(x) <= count:
        return x, y, durations, subjects
    index = draw_indices(len(x), count, seed)
    return x[index], y[index], durations[index], subjects[index]
