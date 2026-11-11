"""Training-context retrieval targets and native-waveform candidate losses."""

import hashlib, json, re
from pathlib import Path
import numpy as np
from scipy.signal import resample, savgol_filter
from . import numerical as scorer
from longmorph.data.preprocessing import minmax_1d

FEATURES = ("phase_rise_fraction", "phase_area", "phase_mean_abs_d2", "phase_sg15_residual_rms")
LOSS_FIELDS = ("rise_target", "area_target", "d2_target", "sg15_target", "native_seam")
ROLE_FIELDS = {
    "morphology": LOSS_FIELDS[:2],
    "detail": LOSS_FIELDS[2:4],
    "continuity": LOSS_FIELDS[4:],
    "rhythm": (),
    "single": LOSS_FIELDS,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def phase_features(cycles):
    """Identical extractor for stored TRAIN phase, observed context and candidates."""
    x = np.asarray(cycles, np.float32)
    if x.shape[-1] != 256 or not np.isfinite(x).all():
        raise ValueError("Expected finite phase cycles of length256")
    flat = x.reshape(-1, 256)
    z = np.stack([minmax_1d(cycle) for cycle in flat])
    result = np.stack(
        [
            np.argmax(z, axis=-1) / 255.0,
            np.trapezoid(z, axis=-1) / 256.0,
            np.mean(np.abs(np.diff(z, n=2, axis=-1)), axis=-1),
            np.sqrt(np.mean((z - savgol_filter(z, 15, 3, axis=-1)) ** 2, axis=-1)),
        ],
        axis=-1,
    )
    return result.reshape(*x.shape[:-1], 4).astype(np.float64)


def native_seam(left, left_duration, right, right_duration):
    """Native-sample seam measure at one join; same500Hz native-index convention."""
    lengths = [max(8, int(round(float(d)))) for d in (left_duration, right_duration)]
    wave = np.concatenate(
        [
            resample(np.asarray(c, np.float32), n).astype(np.float32)
            for c, n in zip((left, right), lengths)
        ]
    )
    t = lengths[0]
    d1 = np.diff(wave)
    d2 = np.diff(wave, n=2)
    parts = [abs(float(v[t]) - float(v[t - 1])) for v in (wave, d1, d2)]
    return float(sum(parts)), dict(
        value_jump=parts[0],
        first_difference_jump=parts[1],
        second_difference_jump=parts[2],
        lengths=lengths,
        sample_rate=500.0,
        component_units=[
            "normalized phase amplitude",
            "phase amplitude per native sample",
            "phase amplitude per native sample squared",
        ],
        composite="Fixed seam convention sums these discrete-index components; not a homogeneous physical quantity.",
    )


class ContextProvider:
    def __init__(self, assets):
        assets = Path(assets)
        self.spec_path = assets / "assets.json"
        self.spec = json.loads(self.spec_path.read_text())
        if (
            self.spec.get("schema") != "longmorph_online_train_assets_v1"
            or self.spec.get("split") != "TRAIN"
        ):
            raise ValueError("Expected explicitly TRAIN-only online assets")
        for name, expected in self.spec["artifact_sha256"].items():
            if sha(assets / name) != expected:
                raise ValueError("TRAIN asset hash mismatch: " + name)
        self.extractor = phase_features
        with np.load(assets / "context_bank.npz", allow_pickle=False) as z:
            self.bank = {k: z[k].copy() for k in z.files}
        self.context = np.asarray(self.bank["context_features"], np.float64)
        self.targets = np.asarray(self.bank["next3_mean_features"], np.float64)
        self.subjects = self.bank["subject_id"].astype(str)
        self.sequences = self.bank["sequence_index"].astype(int)
        self.names = FEATURES
        self.mae = np.asarray(self.spec["TRAIN_LOSO_MAE"], np.float64)
        self.seam_median = float(self.spec["continuity_median"])
        if (
            self.context.shape[1:] != (3, 4)
            or self.targets.shape != (len(self.context), 4)
            or not np.all(self.mae > 0)
            or self.seam_median <= 0
        ):
            raise ValueError("Invalid TRAIN context/calibration schema")
        self.calibration = self.spec["calibration"]

    def forecast(self, prefix, subject, case_id):
        prefix = np.asarray(prefix, np.float32)
        assert prefix.shape == (3, 256) and np.isfinite(prefix).all()
        query = self.extractor(prefix)
        mask = self.subjects != str(subject)
        match = re.search(r"^TRAIN_.*_seq([0-9]+)$", str(case_id))
        sequence = int(match.group(1)) if match else None
        if sequence is not None:
            mask &= self.sequences != sequence
        eligible = np.flatnonzero(mask)
        assert len(eligible) >= 8
        scale = self.context[eligible].reshape(-1, 4).std(axis=0, ddof=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        distances = np.mean(((self.context[eligible] - query) / scale) ** 2, axis=(1, 2))
        ordered = np.lexsort((eligible, distances))
        nearest = eligible[ordered[:8]]
        target = np.median(self.targets[nearest], axis=0)
        return dict(
            target=target.tolist(),
            query_features=query.tolist(),
            query_context_sha256=array_sha(prefix),
            query_subject=str(subject),
            query_train_sequence_index=sequence,
            eligible_count=len(eligible),
            excluded_same_subject_count=int(np.sum(self.subjects == str(subject))),
            context_scale=scale.tolist(),
            target_eligible_std=self.targets[eligible].std(axis=0, ddof=0).tolist(),
            fixed_loss_MAE=self.mae.tolist(),
            neighbor_indices=nearest.tolist(),
            neighbor_window_ids=self.bank["window_id"][nearest].astype(str).tolist(),
            neighbor_subject_ids=self.subjects[nearest].tolist(),
            neighbor_sequence_indices=self.sequences[nearest].tolist(),
            neighbor_squared_distances=distances[ordered[:8]].tolist(),
            neighbor_target_iqr=(
                np.quantile(self.targets[nearest], 0.75, axis=0)
                - np.quantile(self.targets[nearest], 0.25, axis=0)
            ).tolist(),
            support_distance=float(distances[ordered[0]]),
            support_distance_use="descriptive only, no tuned OOD rejection threshold",
            source_spec_sha256=sha(self.spec_path),
            retrieval_uses_true_future_of_query=False,
        )

    def evidence(self, row):
        prefix = np.asarray(row["prefix_cycles"], np.float32)
        pd = np.asarray(row["prefix_durations"], np.float32)
        dur = np.asarray(row["durations"], np.float32)
        assert (
            prefix.ndim == 2
            and prefix.shape[1] == 256
            and len(prefix) >= 3
            and len(pd) == len(prefix)
            and dur.shape == (3,)
        )
        forecast = self.forecast(prefix[-3:], row["subject_id"], row["case_id"])
        target = np.asarray(forecast["target"])
        losses = {}
        traces = {}
        for candidate in row["candidates"]:
            cid = candidate["candidate_id"]
            cycles = np.asarray(candidate["decoded"], np.float32)
            assert cycles.shape == (3, 256) and np.isfinite(cycles).all()
            features = self.extractor(cycles)
            mean = features.mean(axis=0)
            absolute = np.abs(mean - target)
            scaled = absolute / self.mae
            seam, parts = native_seam(prefix[-1], pd[-1], cycles[0], dur[0])
            vector = np.r_[scaled, seam / self.seam_median]
            assert np.isfinite(vector).all() and np.all(vector >= 0)
            losses[cid] = dict(zip(LOSS_FIELDS, vector.tolist()))
            traces[cid] = dict(
                percycle_phase_features=features.tolist(),
                blockmean_phase_features=mean.tolist(),
                absolute_target_errors=absolute.tolist(),
                normalized_target_errors=scaled.tolist(),
                native_seam_raw=seam,
                native_seam_components=parts,
                native_seam_scale=self.seam_median,
                decoded_sha256=array_sha(cycles),
            )
        return dict(
            losses=losses,
            forecast=forecast,
            candidate_trace=traces,
            rhythm=dict(
                status="NEUTRAL_SHARED_PLANNED_DURATION",
                reason="No validated candidate-specific rhythm target; no waveform vote or LLM call",
            ),
            origin_prefix_sha256=row["origin_prefix_sha256"],
            source_calibration=self.calibration,
        )
