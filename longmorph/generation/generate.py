"""Generate one recursive maximum trajectory and expose literal horizon prefixes."""

from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from . import runtime as rt
from .runtime import GenerationRuntime, clean
from longmorph.selection.backend import Backend, sha, write


def context_read(path):
    with np.load(path, allow_pickle=False) as z:
        assert not any(
            "future" in k.lower() or "truth" in k.lower() for k in z.files
        ), "Context-only input required"
        data = {k: z[k].copy() for k in z.files}
    n = len(data["context_cycles"])
    assert n > 0 and data["context_cycles"].shape == (n, 3, 256)
    assert data["context_durations"].shape == (n, 3) and data["context_conditions"].shape == (
        n,
        3,
        14,
    )
    for k in ("context_cycles", "context_durations", "context_conditions"):
        assert np.isfinite(data[k]).all()
    assert (data["context_durations"] > 0).all() and np.all(
        np.asarray(data.get("sample_rate", 500.0)) == 500.0
    )
    return data


def identities(cfg, data):
    result = []
    for i in range(len(data["context_cycles"])):
        case = str(
            data.get(
                "case_id",
                data.get(
                    "anchor_id",
                    np.asarray(["anchor_" + str(j) for j in range(len(data["context_cycles"]))]),
                ),
            )[i]
        )
        subject = str(
            data.get("subject_id", np.asarray(["UNKNOWN"] * len(data["context_cycles"])))[i]
        )
        key_material = dict(
            case_id=case,
            subject_id=subject,
            context_cycles_sha256=rt.ash(data["context_cycles"][i]),
            context_conditions_sha256=rt.ash(data["context_conditions"][i]),
            context_durations_sha256=rt.ash(data["context_durations"][i]),
        )
        key = hashlib.sha256(json.dumps(key_material, sort_keys=True).encode()).hexdigest()
        if cfg["seed_policy"] == "anchor_hash":
            flow_index = condition_index = int(key[:8], 16)
        else:
            flow_index = int(data.get("rng_case_index", np.arange(len(data["context_cycles"])))[i])
            condition_index = int(
                data.get("source_anchor_index", np.arange(len(data["context_cycles"])))[i]
            ) * 100 + int(data.get("window_index", np.zeros(len(data["context_cycles"]), int))[i])
        assert flow_index >= 0 and condition_index >= 0
        result.append(
            dict(
                **key_material,
                row_index=i,
                anchor_key=key,
                anchor_directory=key[:20],
                trajectory_id=cfg["run_id"] + "/" + key,
                flow_seed_index=flow_index,
                condition_seed_index=condition_index,
            )
        )
    assert len({x["anchor_key"] for x in result}) == len(result), "Duplicate anchor identity"
    return result


class Ledger:
    def __init__(self, directory, device):
        self.directory = Path(directory)
        self.device = torch.device(device)
        self.counts = rt.zeros()
        self.ordinal = 0

    def call(self, stage, block, function):
        path = self.directory / f"{self.ordinal:07d}_{stage}.json"
        self.ordinal += 1
        write(
            path, dict(status="PENDING", stage=stage, block_start=block, actual_forward_counts=None)
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        try:
            value, counts = function()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            wall = time.perf_counter() - started
            for key, count in counts.items():
                self.counts[key] += count
            write(
                path,
                dict(
                    status="COMPLETE",
                    stage=stage,
                    block_start=block,
                    actual_forward_counts=counts,
                    wall_seconds=wall,
                ),
            )
            return value, wall, str(path)
        except BaseException as error:
            write(
                path,
                dict(
                    status="FAILED",
                    stage=stage,
                    block_start=block,
                    actual_forward_counts=None,
                    wall_seconds=time.perf_counter() - started,
                    error=repr(error),
                    cost_unknown=True,
                ),
            )
            raise


def generate(config):
    data = context_read(config["context_npz"])
    if "case_id" not in data or "subject_id" not in data:
        raise ValueError("Explicit case_id and canonical subject_id are required")
    if config["wave_agent"] != "off" and config["selector"] != "numerical":
        raise ValueError("Wave Council requires the numerical admissibility/runner-up domain")
    if config["leave_out"] is not None and config["wave_agent"] != "full":
        raise ValueError("Role leave-out requires full waveform Council")
    for name in ("flow_checkpoint", "codec_checkpoint", "condition_checkpoint"):
        if not Path(config[name]).is_file():
            raise FileNotFoundError(config[name])
    assets = Path(config["assets"])
    metadata = json.loads((assets / "assets.json").read_text())
    if (
        metadata.get("schema") != "longmorph_online_train_assets_v1"
        or metadata.get("split") != "TRAIN"
    ):
        raise ValueError("Build TRAIN-only assets with longmorph prepare assets")
    for name, digest in metadata["artifact_sha256"].items():
        if sha(assets / name) != digest:
            raise ValueError("TRAIN asset hash mismatch: " + name)
    backend = None
    if config["condition_agent"] or config["wave_agent"] != "off":
        if not config.get("llm_model"):
            raise ValueError(
                "Enabled agents require --llm-model pointing to local instruction weights"
            )
        backend = Backend(config["llm_model"], config["llm_device"], config["input_token_limit"])
    anchors = identities(config, data)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    write(output / "config.json", config)
    source_files = list(Path(__file__).resolve().parents[1].rglob("*.py"))
    source_files += [rt.CONDITION_CODE, rt.RHYTHM_CODE, rt.FEATURE_CODE]
    inputs = [
        Path(config[name])
        for name in ("context_npz", "flow_checkpoint", "codec_checkpoint", "condition_checkpoint")
    ]
    inputs += [assets / "assets.json", *(assets / name for name in metadata["artifact_sha256"])]
    write(
        output / "inputs.json",
        dict(
            sources=[dict(path=str(p), sha256=sha(p)) for p in source_files],
            inputs=[dict(path=str(p.resolve()), sha256=sha(p)) for p in inputs],
            training_assets_split="TRAIN",
            query_future_read=False,
        ),
    )
    global_ledger = Ledger(output / "events" / "global", config["device"])
    engine, _, _ = global_ledger.call(
        "model_load", None, lambda: (GenerationRuntime(config, backend), rt.zeros())
    )
    maximum = max(config["horizons"])
    generated = 3 * math.ceil(maximum / 3)
    total = generated + 3
    trajectories = []
    all_counts = rt.zeros()
    for anchor in anchors:
        index = anchor["row_index"]
        folder = output / "anchors" / anchor["anchor_directory"]
        folder.mkdir(parents=True)
        ledger = Ledger(folder / "events", config["device"])
        engine.bind_anchor(anchor, ledger)
        state, _, _ = ledger.call(
            "context_encode",
            0,
            lambda: engine.initial(data["context_cycles"][index], data["context_durations"][index]),
        )

        def observed(function, args, kwargs):
            def invoke():
                value = function(*args, **kwargs)
                counts = rt.zeros()
                counts["ConditionTransition"] = 1
                return value, counts

            value, _, _ = ledger.call("condition_proposal", None, invoke)
            return value

        conditions, durations, audit = engine.forecast(
            data["context_cycles"][index],
            data["context_conditions"][index],
            data["context_durations"][index],
            total,
            anchor["condition_seed_index"],
            observed,
        )
        write(folder / "condition_diagnostics.json", clean(audit))
        np.savez_compressed(
            folder / "condition_plan.npz", conditions=conditions, durations=durations
        )
        for block in range(3, total, 3):
            bank, _, _ = ledger.call(
                "candidate_generation",
                block,
                lambda: engine.bank(
                    state,
                    block,
                    conditions[block : block + 3],
                    durations[block : block + 3],
                    durations,
                    anchor["flow_seed_index"],
                    anchor["subject_id"],
                ),
            )

            def choose():
                selected, decision, counts = engine.select(state, bank)
                return (selected, decision), counts

            (selected, decision), _, _ = ledger.call("manager_selection", block, choose)
            if not 0 <= selected < config["visible_J"]:
                raise ValueError("Selected candidate outside the declared visible bank")
            record = dict(
                block_start=block,
                origin_prefix_sha256=rt.ash(state.cycles),
                selected=selected,
                decision=decision,
                bank=bank,
            )
            np.savez_compressed(
                folder / f"block_{block:04d}.npz", record=np.asarray([record], dtype=object)
            )
            state.accept(bank, selected)
            print(
                json.dumps(dict(case_id=anchor["case_id"], block_start=block, selected=selected)),
                flush=True,
            )
        payload = dict(
            future_cycles=state.cycles[3 : 3 + maximum],
            durations=state.durations[3 : 3 + maximum],
            generated_future_cycles_full=state.cycles[3:],
            generated_durations_full=state.durations[3:],
            context_cycles=data["context_cycles"][index],
            context_durations=data["context_durations"][index],
            final_coarse=state.coarse,
            final_detail=state.detail,
        )
        np.savez_compressed(folder / "trajectory_max.npz", **payload)
        write(
            folder / "COMPLETE.json",
            dict(
                status="COMPLETE",
                anchor=anchor,
                retained_horizon=maximum,
                actual_generated_future_cycles=generated,
                actual_forward_counts=ledger.counts,
                query_future_read=False,
            ),
        )
        trajectories.append(payload)
        for key, count in ledger.counts.items():
            all_counts[key] += count
    for horizon in config["horizons"]:
        np.savez_compressed(
            output / f"H{horizon}.npz",
            future_cycles=np.stack([x["future_cycles"][:horizon] for x in trajectories]),
            durations=np.stack([x["durations"][:horizon] for x in trajectories]),
            case_id=np.asarray([x["case_id"] for x in anchors]),
            subject_id=np.asarray([x["subject_id"] for x in anchors]),
            sample_rate=np.asarray(500.0),
        )
    write(
        output / "COMPLETE.json",
        dict(
            status="COMPLETE",
            anchors=len(anchors),
            horizons=config["horizons"],
            actual_generated_future_cycles=generated,
            condition_plan_total_cycles=total,
            actual_forward_counts=all_counts,
            query_future_read=False,
            quality_computed=False,
            orchestration="Sequential generation; inference failures stop without retry",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--flow-checkpoint", required=True, type=Path)
    parser.add_argument("--codec-checkpoint", required=True, type=Path)
    parser.add_argument("--condition-checkpoint", required=True, type=Path)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Fresh output directory, existing paths refused",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--llm-model", type=Path)
    parser.add_argument("--llm-device", default=None, help="Defaults to --device")
    parser.add_argument("--input-token-limit", type=int, default=4096)
    parser.add_argument("--condition-agent", action="store_true")
    parser.add_argument("--wave-agent", choices=("off", "full", "single"), default="off")
    parser.add_argument("--leave-out", choices=("morphology", "detail", "rhythm", "continuity"))
    parser.add_argument(
        "--candidate-mode", choices=("detail", "joint", "retrieval"), default="detail"
    )
    parser.add_argument(
        "--retrieval-library",
        type=Path,
        help="TRAIN-only encoded library for retrieval candidate mode",
    )
    parser.add_argument(
        "--selector", choices=("numerical", "prefix_similarity"), default="numerical"
    )
    parser.add_argument("--visible-j", type=int, choices=(1, 2, 4, 8), default=4)
    parser.add_argument("--history-scope", choices=("local", "cumulative"), default="cumulative")
    parser.add_argument(
        "--seed-policy", choices=("anchor_hash", "window_index"), default="window_index"
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[21, 63, 127, 256])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-id", default="longmorph")
    args = parser.parse_args()
    if args.seed < 0 or not args.horizons or min(args.horizons) < 1 or max(args.horizons) > 256:
        parser.error("seed must be nonnegative and horizons must be in [1,256]")
    config = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config["context_npz"] = config.pop("context")
    config["visible_J"] = config.pop("visible_j")
    config["llm_device"] = config["llm_device"] or config["device"]
    config["horizons"] = sorted(set(config["horizons"]))
    generate(config)


if __name__ == "__main__":
    main()
