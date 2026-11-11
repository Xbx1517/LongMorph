"""Build the online scorer and k8 context assets exclusively from TRAIN cycles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import resample

from longmorph.data.cycles import allocate_lengths, shared_cycles
from longmorph.selection.backend import sha
from longmorph.selection.context import FEATURES, native_seam, phase_features
from longmorph.selection import numerical as scorer


def build(train_sequences, subjects_file, output_dir, sequence_length=9600, *, train_waveforms):
    # The same fixed 24-cycle TRAIN representation used by the online method.
    with np.load(train_sequences, allow_pickle=False) as data:
        cycles = np.asarray(data["cycles"], np.float32)
        durations = np.asarray(data["durations"], np.float32)
    subjects = np.asarray(Path(subjects_file).read_text().splitlines())
    if cycles.ndim != 3 or cycles.shape[1:] != (24, 256) or durations.shape != cycles.shape[:2]:
        raise ValueError("TRAIN requires cycles[N,24,256] and durations[N,24]")
    if len(cycles) < 2 or len(subjects) != len(cycles) or np.any(subjects == ""):
        raise ValueError("Provide one canonical nonempty TRAIN subject ID per sequence")
    if not np.isfinite(cycles).all() or not np.isfinite(durations).all() or np.any(durations <= 0):
        raise ValueError("TRAIN phase cycles/durations must be finite with positive durations")
    if sequence_length < 24 * 16:
        raise ValueError("sequence_length is too short for the calibration allocation")
    with np.load(train_waveforms, allow_pickle=False) as data:
        waveforms = np.asarray(data["x"], np.float32)
    if waveforms.shape != (len(cycles), sequence_length, 1) or not np.isfinite(waveforms).all():
        raise ValueError(
            "TRAIN waveforms require x[N,sequence_length,1] in exactly the same sequence order"
        )

    lengths = np.stack(
        [
            allocate_lengths(row, sequence_length, max(16, sequence_length // (24 * 4)))
            for row in durations
        ]
    )
    calibration_cycles = shared_cycles(waveforms[:, :, 0], durations, sequence_length, 24, 256)
    calibration = scorer.build_calibration(calibration_cycles, lengths, 15, "global")
    features = phase_features(cycles)
    contexts, targets, sequence_ids, window_subjects, window_ids, seams = [], [], [], [], [], []
    for sequence in range(len(cycles)):
        for start in (0, 6, 12, 18):
            contexts.append(features[sequence, start : start + 3])
            targets.append(features[sequence, start + 3 : start + 6].mean(axis=0))
            sequence_ids.append(sequence)
            window_subjects.append(subjects[sequence])
            window_ids.append(f"T{sequence:02}_c{start:02}")
            value, _ = native_seam(
                cycles[sequence, start + 2],
                durations[sequence, start + 2],
                cycles[sequence, start + 3],
                durations[sequence, start + 3],
            )
            seams.append(value)
    contexts, targets = np.stack(contexts), np.stack(targets)
    window_subjects = np.asarray(window_subjects)
    predictions = []
    for i, context in enumerate(contexts):
        eligible = np.flatnonzero(window_subjects != window_subjects[i])
        if len(eligible) < 8:
            raise ValueError(
                "Each TRAIN subject needs at least 8 windows from other TRAIN subjects for k8 LOSO"
            )
        scale = contexts[eligible].reshape(-1, 4).std(axis=0, ddof=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        distances = np.mean(((contexts[eligible] - context) / scale) ** 2, axis=(1, 2))
        ordered = np.lexsort((eligible, distances))
        predictions.append(np.median(targets[eligible[ordered[:8]]], axis=0))
    mae = np.mean(np.abs(np.stack(predictions) - targets), axis=0)
    seam_median = float(np.median(seams))
    if (
        not np.isfinite(mae).all()
        or np.any(mae <= 0)
        or not np.isfinite(seam_median)
        or seam_median <= 0
    ):
        raise ValueError(
            "Degenerate TRAIN LOSO error or seam scale; do not substitute artificial scales"
        )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "scorer_calibration.npz", calibration=np.asarray([calibration], dtype=object)
    )
    np.savez_compressed(
        output / "context_bank.npz",
        context_features=contexts,
        next3_mean_features=targets,
        subject_id=window_subjects,
        sequence_index=np.asarray(sequence_ids),
        window_id=np.asarray(window_ids),
    )
    normalization = dict(
        loss_normalization="Fixed global pooled TRAIN_LOSO knn8 MAE; retrieval scales use eligible TRAIN only",
        feature_names=list(FEATURES),
        TRAIN_LOSO_MAE=mae.tolist(),
        normalization_query_subject_exclusion=False,
        normalization_scope="Fixed units calibrated on all pooled TRAIN LOSO residuals, including probe subjects",
        continuity_median=seam_median,
        continuity_natural_boundaries=len(seams),
        continuity_scale="Median cycle3->4 native500Hz seam measure over non-overlapping TRAIN6cycle windows",
        continuity_units="Ratio of the same mixed-component discrete seam convention",
        continuity_quantiles={
            str(q): float(np.quantile(seams, q)) for q in (0.0, 0.25, 0.5, 0.75, 0.95, 1.0)
        },
        k=8,
    )
    manifest = dict(
        schema="longmorph_online_train_assets_v1",
        split="TRAIN",
        source_inputs=[
            dict(path=str(Path(p).resolve()), sha256=sha(p))
            for p in (train_sequences, train_waveforms, subjects_file)
        ],
        source_builder_sha256=sha(__file__),
        reference_sequences=len(cycles),
        windows=len(contexts),
        TRAIN_subjects=sorted(set(subjects.tolist())),
        sample_rate=500.0,
        sequence_length=sequence_length,
        calibration_smoothing="global SG15",
        calibration=scorer.calibration_summary(calibration) | normalization,
        scorer_cycles="Exact shared_cycles reconstruction from TRAIN fixed-length waveforms; retrieval uses native TRAIN phase cycles",
        TRAIN_LOSO_MAE=mae.tolist(),
        continuity_median=seam_median,
        artifact_sha256={
            name: sha(output / name) for name in ("scorer_calibration.npz", "context_bank.npz")
        },
        restriction="Inputs must contain TRAIN sequences only; caller supplies and authenticates the split and preprocessing provenance",
    )
    (output / "assets.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-sequences",
        required=True,
        type=Path,
        help="TRAIN NPZ: cycles[N,24,256], durations[N,24] in samples at500Hz",
    )
    parser.add_argument(
        "--train-waveforms",
        required=True,
        type=Path,
        help="TRAIN NPZ: x[N,sequence_length,1], same sequence order; scorer shared-coordinate calibration",
    )
    parser.add_argument(
        "--subjects",
        required=True,
        type=Path,
        help="One canonical TRAIN subject ID per first-axis sequence",
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="New directory; existing paths are refused"
    )
    parser.add_argument(
        "--sequence-length", type=int, default=9600, help="Calibration render length"
    )
    args = parser.parse_args()
    manifest = build(
        args.train_sequences,
        args.subjects,
        args.output_dir,
        args.sequence_length,
        train_waveforms=args.train_waveforms,
    )
    print(
        json.dumps(
            dict(
                status="TRAIN_ASSETS_BUILT",
                sequences=manifest["reference_sequences"],
                windows=manifest["windows"],
            )
        )
    )


if __name__ == "__main__":
    main()
