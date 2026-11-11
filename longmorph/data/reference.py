#!/usr/bin/env python3
"""Build reference cycle, condition and waveform datasets from declared source splits."""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
from .cycles import shared_cycles
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

from .preprocessing import (
    COND_NAMES,
    ReferenceConfig,
    extract_cycles_segments,
    minmax_segments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "longmorph_reference_data_only_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path, indices, groups):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    selected = list(range(len(lines))) if indices is None else list(indices)
    if (
        not selected
        or selected != sorted(set(selected))
        or min(selected) < 0
        or max(selected) >= len(lines)
    ):
        raise ValueError("Manifest indices must be nonempty, unique, ascending and in range")
    rows, identities = [], []
    for index in selected:
        group, name = lines[index].split("\t", 1)
        source = Path(name)
        if group not in groups or not source.is_absolute():
            raise ValueError("Expected an allowed source group and an absolute source path")
        if not source.is_file():
            raise FileNotFoundError(source)
        rows.append((source, {"青年人": 0, "中年人": 1}.get(group, 2), group))
        identities.append(
            {
                "manifest_index": index,
                "group": group,
                "path": name,
                "resolved_path": str(source.resolve()),
                "sha256": sha256(source),
                "subject": ppgdalia_subject(source) if group == "ppgdalia" else "UNKNOWN",
            }
        )
    return rows, {
        "path": str(Path(path).resolve()),
        "sha256": sha256(path),
        "total_rows": len(lines),
        "selected_indices": selected,
        "selected_sources": identities,
    }


def ppgdalia_subject(path):
    # Match the lexicographic PPG-DaLiA preprocessing filename order.
    names = tuple(sorted(f"S{i}" for i in range(1, 16)))
    match = re.fullmatch(r"subject_(\d+)_ppg\.npz", path.name)
    if match is None or not 0 <= int(match.group(1)) < len(names):
        raise ValueError(f"Cannot recover canonical PPG-DaLiA subject from {path}")
    return names[int(match.group(1))]


def source_subject(source):
    """Canonical public-dataset identity; other sources retain their file identity."""
    group, filename = source.split(":", 1)
    path = Path(filename)
    if group == "ppgdalia":
        return ppgdalia_subject(path)
    if group == "bidmc":
        match = re.fullmatch(r"(bidmc_[0-9]+)_Signals", path.stem, flags=re.IGNORECASE)
        if match is None:
            raise ValueError("Unrecognized BIDMC recording name: " + path.name)
        return match.group(1).lower()
    return group + ":" + path.stem


def validate_split(train, test):
    paths = [[path.resolve() for path, _label, _group in rows] for rows in (train, test)]
    if any(len(values) != len(set(values)) for values in paths) or set(paths[0]) & set(paths[1]):
        raise ValueError("Duplicate or overlapping train/test source identities")
    subjects = [
        {ppgdalia_subject(path) for path, _label, group in rows if group == "ppgdalia"}
        for rows in (train, test)
    ]
    if subjects[0] & subjects[1]:
        raise ValueError("Overlapping PPG-DaLiA train/test subjects")


def claim_output(path, input_paths):
    raw = Path(path)
    resolved = raw.resolve()
    if resolved == PROJECT_ROOT or PROJECT_ROOT in resolved.parents:
        raise ValueError(
            "Prepare output must be outside the repository and existing output directories"
        )
    for source in input_paths:
        parent = Path(source).resolve().parent
        if resolved == parent or parent in resolved.parents:
            raise ValueError(
                "Prepare output cannot be inside an input data/config/manifest directory"
            )
    if raw.is_symlink():
        raise ValueError("Output directory cannot be a symlink")
    if resolved.exists():
        if not resolved.is_dir() or any(resolved.iterdir()):
            raise FileExistsError("Refusing a nonempty output: " + str(resolved))
    else:
        resolved.mkdir(parents=True, exist_ok=False)
    # Exclusive claim also prevents a second process from sharing an empty directory.
    with (resolved / "PREPARE_CLAIM.txt").open("x", encoding="utf-8") as handle:
        handle.write(SCHEMA + "\nNo resume or overwrite; manifest.json is written last.\n")
    return resolved


def cap_cycle_bank(bank, maximum, seed):
    if len(bank[0]) <= maximum:
        return bank
    index = np.random.default_rng(seed).choice(len(bank[0]), maximum, replace=False)
    return (*[value[index] for value in bank[:3]], [bank[3][int(i)] for i in index])


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_lines(path, lines):
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_npz(path, **arrays):
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def prepare(args):
    started = time.perf_counter()
    config_path = Path(args.config) if args.config else None
    config_hash = sha256(config_path) if config_path else None
    config = dict(
        asdict(ReferenceConfig()),
        cycle_len=256,
        beat_count=24,
        max_cycles_per_file=12000,
        max_segments_per_file=64,
        max_train_cycles=12000,
        seed=args.seed,
        data_sources=args.data_sources,
    )
    if config_path:
        config.update(json.loads(config_path.read_text(encoding="utf-8")))
    cfg = ReferenceConfig.from_mapping(config)
    for key in (
        "cycle_len",
        "beat_count",
        "max_cycles_per_file",
        "max_segments_per_file",
        "max_train_cycles",
    ):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"Configured {key} must be a positive integer")
    if cfg.sample_rate <= 0 or cfg.seq_len <= 0:
        raise ValueError("Configured sample_rate and seq_len must be positive")
    if type(config["seed"]) is not int:
        raise ValueError("Configured seed must be explicit; no replacement seed is inferred")
    groups = set()
    for source in config["data_sources"].split(","):
        if source.strip() == "local":
            groups.update(("青年人", "中年人"))
        elif source.strip() in ("bidmc", "ppgdalia", "vitaldb"):
            groups.add(source.strip())
        else:
            raise ValueError("Unsupported data_sources")
    train, train_identity = read_manifest(args.train_files, args.train_indices, groups)
    test, test_identity = read_manifest(args.test_files, args.test_indices, groups)
    validate_split(train, test)
    strict = config.get("split_mode") == "explicit_subjects"
    if strict:
        for rows, key in ((train, "train_subjects"), (test, "validation_subjects")):
            declared = config[key]
            if not isinstance(declared, list) or any(
                group != "ppgdalia" or ppgdalia_subject(path) not in declared
                for path, _label, group in rows
            ):
                raise ValueError("Explicit subject manifest does not match saved " + key)
    inputs = (
        ([config_path] if config_path else [])
        + [Path(args.train_files), Path(args.test_files)]
        + [row[0] for row in train + test]
    )
    out = claim_output(args.out_dir, inputs)
    data = out / "data"
    data.mkdir()
    cfg_dir = out / "config"
    cfg_dir.mkdir()
    values = {}
    for split, rows in (("train", train), ("test", test)):
        with contextlib.redirect_stdout(sys.stderr):
            values[split] = extract_cycles_segments(
                rows,
                cfg,
                config["cycle_len"],
                config["beat_count"],
                config["max_cycles_per_file"],
                config["max_segments_per_file"],
            )
        result = values[split]
        # Preserve the second normalization in Operator main, after extraction.
        segments = minmax_segments(result[7])
        write_npz(data / f"{split}_ppg_segments.npz", x=segments)
        write_npz(
            data / f"{split}_segment_condition_sequences.npz",
            conditions=result[4],
            durations=result[5],
            cycles=result[6],
        )
        write_lines(data / f"{split}_segment_sources.txt", result[8])
        write_lines(
            data / f"{split}_subjects.txt", [source_subject(source) for source in result[8]]
        )
        write_lines(
            data / f"{split}_files.txt", [f"{group}\t{path}" for path, _label, group in rows]
        )
        if split == "test" and strict:
            write_npz(
                data / "validation_segment_condition_sequences.npz",
                conditions=result[4],
                durations=result[5],
                cycles=result[6],
            )
            write_lines(data / "validation_segment_sources.txt", result[8])
    bank = cap_cycle_bank(values["train"][:4], config["max_train_cycles"], config["seed"])
    write_npz(
        data / "train_cycles.npz", x=bank[0][:, :, None], conditions=bank[1], durations=bank[2]
    )
    write_lines(data / "train_cycle_sources.txt", bank[3])
    effective = {
        **asdict(cfg),
        **{
            key: config[key]
            for key in (
                "cycle_len",
                "beat_count",
                "max_cycles_per_file",
                "max_segments_per_file",
                "max_train_cycles",
                "seed",
                "data_sources",
            )
        },
        "schema": SCHEMA,
        "condition_names": list(COND_NAMES),
        "cycle_representation": "reference_trough_to_trough",
    }
    write_json(cfg_dir / "config.json", effective)
    for identity in (train_identity, test_identity):
        if sha256(identity["path"]) != identity["sha256"]:
            raise RuntimeError("Input manifest changed during prepare")
        for source in identity["selected_sources"]:
            if sha256(source["path"]) != source["sha256"]:
                raise RuntimeError("Input waveform changed during prepare")
    if config_path and sha256(config_path) != config_hash:
        raise RuntimeError("Input config changed during prepare")
    # Codec training uses the shared coordinate representation of complete
    # waveforms; the native phase cycles above remain available to the Flow.
    train_segments = minmax_segments(values["train"][7])
    coordinate_cycles = shared_cycles(
        train_segments[:, :, 0],
        values["train"][5],
        cfg.seq_len,
        config["beat_count"],
        config["cycle_len"],
    )
    write_npz(
        data / "shared_coordinate_cycles.npz",
        x=coordinate_cycles,
        conditions=values["train"][4],
        durations=values["train"][5],
    )
    artifacts = {
        str(path.relative_to(out)): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "cycle_representation": "reference_trough_to_trough",
        "shared_cycles_emitted": True,
        "waveform_normalization": "extract minmax_1d then minmax_segments(+1e-6)",
        "source_config": (
            {"path": str(config_path.resolve()), "sha256": config_hash} if config_path else None
        ),
        "splits": {"train": train_identity, "test": test_identity},
        "test_manifest_role": "validation" if strict else "test_manifest",
        "seeds": {
            "preprocessing_seed": config["seed"],
            "split_seed": "NOT_USED: explicit manifests",
            "cycle_bank_subsample_seed": config["seed"],
            "training": "NOT_USED",
            "generator": "NOT_USED",
            "evaluation": "NOT_USED",
        },
        "model_training_performed": False,
        "subject_identity": "Canonical PPG-DaLiA participant or BIDMC recording; other sources use source file IDs",
        "input_source_hash": "UNKNOWN",
        "artifacts": artifacts,
        "training_cycle_bank_rows": len(bank[0]),
        "counts": {
            split: {"cycles": len(v[0]), "sequences": len(v[7])} for split, v in values.items()
        },
    }
    write_json(out / "manifest.json", manifest)
    frameworks = sorted(
        {
            name.split(".")[0]
            for name in sys.modules
            if name.split(".")[0] in {"torch", "tensorflow", "transformers"}
        }
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "out_dir": str(out),
        "counts": manifest["counts"],
        "seconds": time.perf_counter() - started,
        "loaded_model_frameworks": frameworks,
        "loaded_module_count": len(sys.modules),
        "file_count": len(artifacts) + 1,
        "output_bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", help="Optional preprocessing JSON; effective values are saved with the dataset"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--data-sources", choices=("local", "bidmc", "ppgdalia", "vitaldb"), default="bidmc"
    )
    parser.add_argument(
        "--train-files", required=True, help="Ordered group<TAB>absolute-path manifest"
    )
    parser.add_argument(
        "--test-files",
        required=True,
        help="Ordered test/validation source manifest",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="New or empty directory outside the repository and input directories",
    )
    parser.add_argument(
        "--train-indices", type=int, nargs="+", help="Optional ascending zero-based manifest subset"
    )
    parser.add_argument(
        "--test-indices", type=int, nargs="+", help="Optional ascending zero-based manifest subset"
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
