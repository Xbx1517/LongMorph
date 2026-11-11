#!/usr/bin/env python3
"""Assemble a hybrid codec from coarse and full-resolution detail checkpoints."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .objectives import diagnostics
from longmorph.models.codec import DualStreamCodec, split_components
from .splits import subject_disjoint_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cycles-npz", required=True)
    p.add_argument("--coarse-checkpoint", required=True)
    p.add_argument("--detail-checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--detail-scale", type=float, default=1.0)
    p.add_argument(
        "--groups-file",
        default="",
        help="One subject/source label per sequence for subject-disjoint validation",
    )
    p.add_argument("--validation-fraction", type=float, default=0.2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    out = Path(a.out_dir)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    if not 0 < a.validation_fraction < 1:
        p.error("--validation-fraction must be in (0,1)")
    rows = np.load(a.cycles_npz)["x"].astype(np.float32)
    if a.groups_file:
        groups = Path(a.groups_file).read_text().splitlines()
        _, valid, train_groups, valid_groups = subject_disjoint_split(
            rows, groups, a.validation_fraction, a.seed
        )
        split_mode = "subject_disjoint"
    else:
        order = np.random.default_rng(a.seed).permutation(len(rows))
        split = max(1, int((1 - a.validation_fraction) * len(rows)))
        if split >= len(rows):
            split = len(rows) - 1
        valid = rows[order[split:]].reshape(-1, rows.shape[-1])
        train_groups = order[:split].astype(str)
        valid_groups = order[split:].astype(str)
        split_mode = "sequence_random"
    coarse = torch.load(a.coarse_checkpoint, map_location="cpu", weights_only=False)["model"]
    detail = torch.load(a.detail_checkpoint, map_location="cpu", weights_only=False)["model"]
    state = dict(detail)
    for key, value in coarse.items():
        if key.startswith("coarse_encoder.") or key.startswith("coarse_decoder."):
            state[key] = value
    device = torch.device(a.device)
    model = DualStreamCodec(4, 2, True).to(device)
    model.load_state_dict(state)
    model.eval()
    recon = []
    detail_out = []
    loader = DataLoader(TensorDataset(torch.from_numpy(valid[:, None])), batch_size=a.batch_size)
    with torch.no_grad():
        for (x,) in loader:
            _, c, d, _, _ = model(x.to(device))
            d = a.detail_scale * d
            y = c + d
            recon.append(y[:, 0].cpu().numpy())
            detail_out.append(d[:, 0].cpu().numpy())
    recon = np.concatenate(recon)
    detail_out = np.concatenate(detail_out)
    with torch.no_grad():
        _, real_detail = split_components(torch.from_numpy(valid[:, None]))
    metrics = diagnostics(valid, recon, a.seed)
    metrics["detail_energy_ratio"] = float(
        np.mean(detail_out**2) / (np.mean(real_detail.numpy() ** 2) + 1e-9)
    )
    metrics["detail_scale"] = a.detail_scale
    metrics.update(
        split_mode=split_mode,
        validation_cycles=len(valid),
        training_owners=int(len(train_groups)),
        validation_owners=int(len(valid_groups)),
        training_groups=[str(value) for value in train_groups],
        validation_groups=[str(value) for value in valid_groups],
    )
    (out / "metrics/metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save(
        {
            "model": state,
            "source_coarse": a.coarse_checkpoint,
            "source_detail": a.detail_checkpoint,
        },
        out / "models/dualstream_hybrid_codec.pt",
    )
    np.savez_compressed(
        out / "data/reconstructions.npz", real=valid, reconstruction=recon, detail=detail_out
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
