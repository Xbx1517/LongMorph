"""Unified training interface for baseline generators with optional dependencies."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from longmorph.baselines.common import (
    VENDOR,
    decode_latents,
    initialize_run,
    prepare_latents,
    json_dump,
    load_waveforms,
    save_generated,
    seed_torch,
    set_seed,
    source_paths,
)

MODELS = {
    "iaaft": ("numpy", None),
    "tts_gan": ("torch", "GANModels.py"),
    "diffusion_ts": ("torch", "Models/interpretable_diffusion/gaussian_diffusion.py"),
    "timevqvae": ("torch", "src/timevqvae/vqvae.py"),
    "fourier_flow": ("torch", "SequentialFlows.py"),
    "aec_gan": ("torch", "lib/algos/asganadv.py"),
    "timegan": ("tensorflow1", "timegan.py"),
    "timevae": ("tensorflow2", "src/vae/timevae.py"),
}
MANUAL_SOURCES = {"aec_gan", "fourier_flow"}


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Show model names and required environments")
    train = commands.add_parser(
        "train", help="Train and sample from an explicitly supplied reference dataset"
    )
    train.add_argument("--model", choices=tuple(MODELS), required=True)
    train.add_argument(
        "--reference-run",
        type=Path,
        required=True,
        help="Prepared run containing data/train_ppg_segments.npz and data/test_ppg_segments.npz",
    )
    train.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty output directory outside the repository",
    )
    train.add_argument("--seed", type=int, required=True)
    train.add_argument(
        "--vendor-dir",
        type=Path,
        help="Parent directory containing named upstream source checkouts",
    )
    train.add_argument(
        "--context-path",
        type=Path,
        help="Required for AEC-GAN: x=[N,1200] observed waveform contexts (three 400-sample patches)",
    )
    train.add_argument("--device", default="cuda")
    train.add_argument("--num-samples", type=int, default=32)
    train.add_argument(
        "--batch-size", type=int, help="Default: TimeGAN 128, TimeVAE 16, PyTorch 32"
    )
    train.add_argument(
        "--iterations",
        type=int,
        help="Default: IAAFT 100, TimeGAN 2000 iterations per phase",
    )
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--steps", type=int, default=2000)
    train.add_argument("--vq-epochs", type=int, default=300)
    train.add_argument("--prior-epochs", type=int, default=500)
    return root


def validate(args):
    for key in ("num_samples", "epochs", "steps", "vq_epochs", "prior_epochs"):
        if getattr(args, key) <= 0:
            raise ValueError(key.replace("_", "-") + " must be positive")
    if args.batch_size is None:
        args.batch_size = {"timegan": 128, "timevae": 16}.get(args.model, 32)
    if args.iterations is None:
        args.iterations = 100 if args.model == "iaaft" else 2000
    if args.batch_size <= 0 or args.iterations <= 0:
        raise ValueError("Batch size and iterations must be positive")
    if args.model == "aec_gan" and args.context_path is None:
        raise ValueError(
            "AEC-GAN requires an explicit --context-path containing only observed contexts"
        )
    if args.model != "aec_gan" and args.context_path is not None:
        raise ValueError("--context-path is supported only for AEC-GAN")
    if args.model in MANUAL_SOURCES and args.vendor_dir is None:
        raise ValueError(
            args.model
            + " requires an explicit --vendor-dir with independently obtained upstream source; "
            "automated retrieval is disabled because its license has not been established"
        )
    args.vendor_dir = (args.vendor_dir or VENDOR).expanduser().resolve()
    environment, required = MODELS[args.model]
    if required and not (args.vendor_dir / args.model / required).is_file():
        raise FileNotFoundError(
            "Upstream source is not installed: " + str(args.vendor_dir / args.model / required)
        )
    package = "tensorflow" if environment.startswith("tensorflow") else environment
    if importlib.util.find_spec(package) is None:
        raise RuntimeError("Install the separate " + environment + " environment for " + args.model)
    if args.output_dir.is_symlink():
        raise ValueError("Output directory cannot be a symlink")
    return source_paths(args.reference_run)


def train(args):
    sources = validate(args)
    set_seed(args.seed)
    started = time.perf_counter()
    waveforms = {"train": load_waveforms(sources["train"])}
    if args.model == "aec_gan":
        waveforms["test"] = load_waveforms(args.context_path)
        if waveforms["train"].shape[1] != 9600 or waveforms["test"].shape[1] != 1200:
            raise ValueError(
                "AEC-GAN expects 9600-sample training sequences and 1200-sample observed contexts"
            )
        if args.num_samples > len(waveforms["test"]):
            raise ValueError("AEC-GAN requires at least num-samples observed contexts")
    else:
        waveforms["test"] = load_waveforms(sources["test"])
        if waveforms["train"].shape[1] != waveforms["test"].shape[1]:
            raise ValueError("Training and evaluation waveforms must have the same length")
    out = initialize_run(args.output_dir, args.reference_run, vars(args))
    if args.model == "iaaft":
        from longmorph.baselines.iaaft import generate

        generated = generate(waveforms["train"], args.seed, args.num_samples, args.iterations)
        extra = {"input_representation": "raw_waveform", "iterations": args.iterations}
    else:
        codec, latent_train, latent_test, report = prepare_latents(
            waveforms["train"],
            waveforms["test"],
            test_blocks=3 if args.model == "aec_gan" else None,
        )
        with (out / "data/codec.npz").open("xb") as handle:
            np.savez_compressed(handle, **codec)
        json_dump(report, out / "data/codec_report.json")
        framework = MODELS[args.model][0]
        if framework == "torch":
            import torch
            from longmorph.baselines import torch_models

            seed_torch(args.seed)
            device = torch.device(args.device)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; use --device cpu explicitly")
            function = getattr(torch_models, "train_" + args.model)
            if args.model == "aec_gan":
                latent, extra = function(latent_train, latent_test, args, device, out)
            else:
                latent, extra = function(latent_train, args, device, out)
        else:
            from longmorph.baselines import tensorflow_models

            function = getattr(tensorflow_models, "train_" + args.model)
            latent, extra = function(latent_train, args, out)
        if len(latent) != args.num_samples:
            raise ValueError("Upstream implementation returned fewer samples than requested")
        generated = decode_latents(codec, latent)
        extra["input_representation"] = "24_patch_32_coordinate_train_only_PCA"
    path = save_generated(out, generated, args.model, args.seed, started, extra)
    return {"model": args.model, "output_dir": str(out), "generated": str(path)}


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.command == "list":
        for name, (framework, _path) in MODELS.items():
            source = (
                "user-managed source"
                if name in MANUAL_SOURCES
                else "pinned optional source" if name != "iaaft" else "built in"
            )
            print(name + "\t" + framework + "\t" + source)
        return
    try:
        print(json.dumps(train(args), sort_keys=True))
    except (
        ValueError,
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        ImportError,
    ) as error:
        cli.exit(2, str(error) + "\n")


if __name__ == "__main__":
    main()
