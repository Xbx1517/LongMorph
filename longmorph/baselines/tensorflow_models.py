"""Optional TensorFlow adapters; each framework requires its own environment."""

import os
import sys

import numpy as np


def train_timegan(train, args, out):
    # TensorFlow 1 kernels predate current CUDA architectures; use its CPU path.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    import tensorflow as tf

    if str(tf.__version__).split(".")[0] != "1":
        raise RuntimeError("TimeGAN requires its separate TensorFlow 1 environment")
    sys.path.insert(0, str(args.vendor_dir / "timegan"))
    from timegan import timegan

    parameters = {
        "module": "gru",
        "hidden_dim": 64,
        "num_layer": 3,
        "iterations": args.iterations,
        "batch_size": min(args.batch_size, len(train)),
    }
    latent = np.asarray(timegan(list(train), parameters), dtype=np.float32)[: args.num_samples]
    return latent, {
        "iterations_per_phase": args.iterations,
        "architecture": "TimeGAN",
        "execution_device": "cpu",
        "checkpoint": None,
        "checkpoint_note": "Upstream timegan() returns samples without a persistent model checkpoint",
    }


def train_timevae(train, args, out):
    import tensorflow as tf

    if str(tf.__version__).split(".")[0] != "2":
        raise RuntimeError("TimeVAE requires its separate TensorFlow 2 environment")
    tf.random.set_seed(args.seed)
    sys.path.insert(0, str(args.vendor_dir / "timevae" / "src"))
    from vae.timevae import TimeVAE

    model = TimeVAE(
        seq_len=24,
        feat_dim=32,
        latent_dim=16,
        reconstruction_wt=3.0,
        batch_size=min(args.batch_size, len(train)),
        hidden_layer_sizes=[50, 100, 200],
        trend_poly=0,
        custom_seas=None,
        use_residual_conn=True,
    )
    model.fit_on_data(train, max_epochs=args.epochs, verbose=0)
    latent = model.get_prior_samples(args.num_samples)
    model.save(str(out / "models"))
    return latent, {"epochs_max": args.epochs, "architecture": "TimeVAE"}
