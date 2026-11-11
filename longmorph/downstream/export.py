"""Convert generated waveform archives to leakage-checked forecasting pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pairs import sequence_pairs


def extract_sequences(path, sample_rate=500.0, expected_cycles=24, expected_segments=None):
    from longmorph.data.cycles import extract_aligned_cycles_with_fiducials, load_npz_array

    signal = load_npz_array(Path(path), ("x", "segments", "x_fake"))
    if expected_segments is not None and len(signal) != expected_segments:
        raise ValueError("Source count does not match waveform segment count")
    cycles, meta = extract_aligned_cycles_with_fiducials(
        signal, sample_rate, 256, expected_beats=expected_cycles
    )
    if meta.empty:
        raise ValueError(f"No aligned cycles extracted from {path}")
    segment_ids = meta.segment_idx.to_numpy()
    sequences = []
    durations = []
    ids = []
    counts = []
    for segment_id in sorted(meta.segment_idx.unique()):
        mask = segment_ids == segment_id
        segment_cycles = cycles[mask]
        segment_meta = meta[mask].sort_values("cycle_idx")
        counts.append(len(segment_cycles))
        sequences.append(segment_cycles[:expected_cycles])
        durations.append(segment_meta["rr_ms"].to_numpy(dtype=np.float32)[:expected_cycles])
        ids.append(int(segment_id))
    return sequences, durations, ids, np.asarray(counts, np.int16)


def export_pairs(
    generated_npz,
    sample_rate=500.0,
    expected_cycles=24,
    input_cycles=5,
    target_cycles=1,
    real_prefix_cycles=3,
    pair_stride=1,
):
    sequences, _, _, counts = extract_sequences(generated_npz, sample_rate, expected_cycles)
    xs, ys = [], []
    for sequence in sequences:
        x, y = sequence_pairs(
            sequence[None], input_cycles, target_cycles, real_prefix_cycles, pair_stride
        )
        if len(x):
            xs.append(x)
            ys.append(y)
    if not xs:
        raise ValueError(
            f"No fully generated {input_cycles}-to-{target_cycles} pairs in {generated_npz}"
        )
    return np.concatenate(xs), np.concatenate(ys), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--train-subjects",
        required=True,
        help="Comma-separated numeric subjects used to fit the generator",
    )
    parser.add_argument("--sample-rate", type=float, default=500.0)
    parser.add_argument("--expected-cycles", type=int, default=24)
    parser.add_argument("--input-cycles", type=int, default=5)
    parser.add_argument("--target-cycles", type=int, default=1)
    parser.add_argument("--real-prefix-cycles", type=int, default=3)
    parser.add_argument("--pair-stride", type=int, default=1)
    args = parser.parse_args()
    x, y, counts = export_pairs(
        args.generated_npz,
        args.sample_rate,
        args.expected_cycles,
        args.input_cycles,
        args.target_cycles,
        args.real_prefix_cycles,
        args.pair_stride,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        synthetic_x=x,
        synthetic_y=y,
        detected_cycle_counts=counts,
        real_prefix_cycles=np.int16(args.real_prefix_cycles),
        pair_stride=np.int16(args.pair_stride),
        train_subject_index=np.asarray([int(s) for s in args.train_subjects.split(",")], np.int32),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "pairs": len(x),
                "detected_cycle_range": [int(counts.min()), int(counts.max())],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
