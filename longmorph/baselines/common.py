"""Train-only PCA representation and explicit run I/O for baseline adapters."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time

import numpy as np

VENDOR = Path(__file__).resolve().parent / "vendor"


def json_dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_waveforms(path):
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        values = loaded
    else:
        with loaded:
            values = loaded["x"] if "x" in loaded else loaded[loaded.files[0]]
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or not len(values) or not values.shape[1]:
        raise ValueError("Expected nonempty waveform array [samples, time] or [samples, time, 1]")
    if not np.isfinite(values).all():
        raise ValueError("Waveforms must be finite")
    return values


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def seed_torch(seed):
    import torch

    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fit_codec(train, n_blocks=24, rank=32):
    """Fit patch PCA and coordinate statistics exclusively on training data.

    Latents use the full-precision SVD fit, while the saved codec uses float32,
    matching the representation used by all external model adapters.
    """
    train = np.asarray(train, dtype=np.float32)
    if train.ndim != 2 or not len(train) or not np.isfinite(train).all():
        raise ValueError("Training waveforms must be a finite, nonempty matrix")
    if n_blocks <= 0 or rank <= 0 or train.shape[1] % n_blocks:
        raise ValueError("Positive rank/blocks and a sequence length divisible by blocks required")
    block_len = train.shape[1] // n_blocks
    patches = train.reshape(-1, block_len).astype(np.float64)
    if rank > min(patches.shape):
        raise ValueError("PCA rank exceeds the available patch dimensions")
    mean = patches.mean(axis=0)
    _, singular, vt = np.linalg.svd(patches - mean, full_matrices=False)
    components = vt[:rank]
    z0 = ((patches - mean) @ components.T).reshape(len(train), n_blocks, rank)
    latent_mean = z0.reshape(-1, rank).mean(axis=0)
    latent_std = np.maximum(z0.reshape(-1, rank).std(axis=0), 1e-6)
    latents = ((z0 - latent_mean) / latent_std).astype(np.float32)
    codec = {
        "mean": mean,
        "components": components,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "n_blocks": np.int32(n_blocks),
        "block_len": np.int32(block_len),
        "rank": np.int32(rank),
    }
    total = float(np.sum(singular**2))
    report = {
        "fit_split": "train_only",
        "n_blocks": n_blocks,
        "block_len": block_len,
        "rank": rank,
        "n_train_segments": len(train),
        "explained_variance_ratio": (float(np.sum(singular[:rank] ** 2) / total) if total else 0.0),
    }
    return codec, latents, report


def _storage_codec(codec):
    return {
        key: (
            value.astype(np.float32)
            if key in {"mean", "components", "latent_mean", "latent_std"}
            else value
        )
        for key, value in codec.items()
    }


def fit_codec(train, n_blocks=24, rank=32):
    """Fit and return the portable float32 codec, training latents and fit report."""
    codec, latents, report = _fit_codec(train, n_blocks, rank)
    return _storage_codec(codec), latents, report


def prepare_latents(train, test, n_blocks=24, rank=32, test_blocks=None):
    """Encode both splits using one full-precision, training-only PCA fit."""
    codec, train_latents, report = _fit_codec(train, n_blocks, rank)
    test_latents = encode_latents(codec, test, n_blocks=test_blocks)
    return _storage_codec(codec), train_latents, test_latents, report


def encode_latents(codec, waveforms, n_blocks=None):
    values = np.asarray(waveforms, dtype=np.float32)
    block_len, rank = int(codec["block_len"]), int(codec["rank"])
    blocks = int(codec["n_blocks"]) if n_blocks is None else n_blocks
    expected = blocks * block_len
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError("Waveform shape does not match the fitted codec")
    patches = values.reshape(-1, block_len).astype(np.float64)
    z = ((patches - codec["mean"]) @ codec["components"].T).reshape(len(values), -1, rank)
    return ((z - codec["latent_mean"]) / codec["latent_std"]).astype(np.float32)


def decode_latents(codec, latents):
    z = np.asarray(latents, dtype=np.float32)
    expected = (int(codec["n_blocks"]), int(codec["rank"]))
    if z.ndim != 3 or tuple(z.shape[1:]) != expected or not np.isfinite(z).all():
        raise ValueError("Latent shape must match the codec and contain finite values")
    raw = (z * codec["latent_std"] + codec["latent_mean"]).reshape(-1, expected[1]) @ codec[
        "components"
    ] + codec["mean"]
    return np.clip(raw.reshape(len(z), -1), 0.0, 1.0).astype(np.float32)


def source_paths(reference_run):
    base = Path(reference_run).expanduser().resolve()
    paths = {name: base / "data" / (name + "_ppg_segments.npz") for name in ("train", "test")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if paths["train"].resolve() == paths["test"].resolve():
        raise ValueError("Training and evaluation inputs must be distinct")
    return paths


def initialize_run(output_dir, reference_run, config):
    out = Path(output_dir).expanduser().resolve()
    reference = Path(reference_run).expanduser().resolve()
    repository = Path(__file__).resolve().parents[2]
    if (
        out == reference
        or reference in out.parents
        or out == repository
        or repository in out.parents
    ):
        raise ValueError("Output must be outside the repository and reference data directory")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError("Output directory must be new or empty: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    # Exclusive claim protects an empty output from concurrent training invocations.
    with (out / ".run_claim").open("x", encoding="utf-8") as handle:
        handle.write("baseline training\n")
    for name in ("data", "models", "config", "runtime"):
        (out / name).mkdir()
    for split, source in source_paths(reference).items():
        (out / "data" / (split + "_ppg_segments.npz")).symlink_to(source)
    payload = {
        key: str(value) if isinstance(value, Path) else value for key, value in config.items()
    }
    payload["reference_run"] = str(reference)
    json_dump(payload, out / "config/config.json")
    return out


def save_generated(out, generated, model, seed, started_at, extra=None):
    generated = np.asarray(generated, dtype=np.float32)
    if generated.ndim == 2:
        generated = generated[..., None]
    if (
        generated.ndim != 3
        or generated.shape[-1] != 1
        or not len(generated)
        or not np.isfinite(generated).all()
    ):
        raise ValueError("Generated waveforms must be finite [samples, time, 1]")
    out = Path(out)
    path = out / "data/generated_ppg_segments.npz"
    with path.open("xb") as handle:
        np.savez_compressed(handle, x=generated)
    runtime = {
        "model": model,
        "seed": seed,
        "n_generated": len(generated),
        "shape": list(generated.shape),
        "wall_seconds": time.perf_counter() - started_at,
    }
    runtime.update(extra or {})
    json_dump(runtime, out / "runtime/training_and_inference.json")
    return path
