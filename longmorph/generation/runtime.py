"""Recursive block generation with raw-latent feedback and guarded decisions."""

from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from longmorph.models import codec as cm
from longmorph.models import flow as bm
from longmorph.models.condition import ConditionTransition
from longmorph.data import features as phase
from longmorph.data import cycles as cycle_coordinates
from longmorph.data.cycles import rhythm_features
from longmorph.selection import numerical as scorer
from longmorph.selection.council import Council
from longmorph.selection.context import ContextProvider
from . import conditions as conditions_module
from .conditions import sample_condition_proposal, build_condition_trajectory

CONDITION_CODE = Path(conditions_module.__file__)
RHYTHM_CODE = Path(cycle_coordinates.__file__)
FEATURE_CODE = Path(phase.__file__)
GATES = dict(
    axis_tolerance=0.04,
    total_tolerance=0.02,
    critic_tolerance=0.03,
    batch_tolerance=0.015,
    require_improvement=False,
    min_total_gain=0.0,
)
COUNT_KEYS = (
    "CoarseFlow",
    "DetailFlow",
    "coarse_encoder",
    "detail_encoder",
    "coarse_decoder",
    "detail_decoder",
    "ConditionTransition",
    "LLM",
)


def zeros():
    return dict.fromkeys(COUNT_KEYS, 0)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def phase_norm(x):
    x = np.asarray(x, np.float64)
    return (x - x.min(-1, keepdims=True)) / np.maximum(
        x.max(-1, keepdims=True) - x.min(-1, keepdims=True), 1e-6
    )


def history_summary(cycles, durations, scope):
    cs = cycles[-3:] if scope == "local" else cycles
    ds = durations[-3:] if scope == "local" else durations
    return dict(
        scope=scope,
        buffer_reset_scope="anchor",
        summary_cycles=len(cs),
        phase_mean=float(cs.mean()),
        phase_sd=float(cs.std()),
        mean_abs_phase_d1=float(np.abs(np.diff(cs, axis=-1)).mean()),
        mean_duration_samples=float(ds.mean()),
        duration_cv=float(ds.std() / (ds.mean() + 1e-6)),
        sample_rate_hz=500.0,
        units=dict(
            summary_cycles="cycle count",
            phase_mean="decoder amplitude",
            phase_sd="decoder amplitude",
            mean_abs_phase_d1="decoder amplitude per phase sample",
            mean_duration_samples="samples at500Hz",
            duration_cv="dimensionless",
        ),
        actual_prompt_use="SUPPLIED_TO_AGENT_PROMPT",
    )


class State:
    def __init__(self, cycles, durations, coarse, detail, features=None):
        self.cycles = np.asarray(cycles, np.float32).copy()
        self.durations = np.asarray(durations, np.float32).copy()
        self.coarse = np.asarray(coarse, np.float32).copy()
        self.detail = np.asarray(detail, np.float32).copy()
        self.features = list(features or [])

    def accept(self, bank, index):
        self.coarse = bank["coarse_latents"][index].copy()
        self.detail = bank["detail_latents"][index].copy()
        self.cycles = np.concatenate([self.cycles, bank["candidates"][index]["decoded"]]).astype(
            np.float32
        )
        self.durations = np.r_[self.durations, bank["durations"]].astype(np.float32)
        self.features.append(bank["candidates"][index]["feature_rows"])
        assert len(self.cycles) == 3 + 3 * len(self.features) and len(self.durations) == len(
            self.cycles
        )


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


class GenerationRuntime:
    def __init__(self, config, backend=None):
        self.config = config
        self.device = torch.device(config["device"])
        torch.set_num_threads(2)
        torch.manual_seed(int(config["seed"]))
        if config["selector"] not in ("numerical", "prefix_similarity"):
            raise ValueError("This entry supports numerical or prefix_similarity selection")
        self.cm = cm
        self.scorer = scorer
        self.phase = phase
        cp = torch.load(config["flow_checkpoint"], map_location="cpu", weights_only=False)
        self.stats = cp["stats"]
        self.coarse = bm.BlockVelocity(12, 46).to(self.device)
        self.coarse.load_state_dict(cp["coarse_flow"])
        self.detail = bm.BlockVelocity(6, 46, 12).to(self.device)
        self.detail.load_state_dict(cp["detail_flow"])
        self.codec = cm.load_generator_codec(config["codec_checkpoint"], self.device)
        cc = torch.load(config["condition_checkpoint"], map_location="cpu", weights_only=False)
        self.transition = ConditionTransition(15).to(self.device)
        self.transition.load_state_dict(cc["model"])
        self.center = np.asarray(cc["center"], np.float32)
        self.scale = np.asarray(cc["scale"], np.float32)
        self.lower = np.asarray(cc["support_lower"], np.float32)
        self.upper = np.asarray(cc["support_upper"], np.float32)
        args = cc.get("args", {})
        args = args if isinstance(args, dict) else {}
        self.noise = dict(
            duration_centered_noise_scale=float(args.get("duration_centered_noise_scale", 1.25)),
            duration_common_noise_scale=float(args.get("duration_common_noise_scale", 0.25)),
        )
        self.rhythm = rhythm_features
        with np.load(Path(config["assets"]) / "scorer_calibration.npz", allow_pickle=True) as z:
            self.calibration = z["calibration"][0]
        for model in (self.coarse, self.detail, self.codec, self.transition):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.library = None
        if config["candidate_mode"] == "retrieval":
            if not config.get("retrieval_library"):
                raise ValueError(
                    "retrieval candidate mode requires a TRAIN-only encoded retrieval library"
                )
            with np.load(config["retrieval_library"], allow_pickle=False) as z:
                self.library = {key: z[key].copy() for key in z.files}
            for key in (
                "subject_id",
                "context_features",
                "window_id",
                "raw_coarse",
                "coarse_decoded",
            ):
                if key not in self.library:
                    raise ValueError("Missing retrieval library field: " + key)

        self.backend = backend
        self.provider = ContextProvider(config["assets"]) if config["wave_agent"] != "off" else None
        self.sources = [
            Path(__file__),
            Path(conditions_module.__file__),
            Path(config["assets"]) / "assets.json",
        ]
        self.council = (
            self.make_council(Path(config["output_dir"]) / "wave_calls")
            if config["wave_agent"] != "off"
            else None
        )

    def initial(self, cycles, durations):
        with torch.inference_mode():
            tensor = torch.from_numpy(cycles[:, None].astype(np.float32)).to(self.device)
            c, d = self.cm.split_components(tensor)
            rc = self.codec.coarse_encoder(c).cpu().numpy()
            rd = self.codec.detail_encoder(d).cpu().numpy()
        count = zeros()
        count["coarse_encoder"] = count["detail_encoder"] = 1
        return State(cycles, durations, rc, rd), count

    def make_council(self, path):
        return Council(path, self.config["run_id"], self.backend, self.provider, self.sources)

    def bind_anchor(self, anchor, ledger):
        self.active_anchor = anchor
        self.ledger = ledger

    def forecast(self, cycles, conditions, durations, total, seed_index, proposal_logger):
        flag = bool(self.config["condition_agent"])
        directory = (
            Path(self.config["output_dir"]) / "condition_calls" / self.active_anchor["anchor_key"]
        )
        council = self.make_council(directory) if flag else None
        runtime = self

        class Adapter:
            def check_condition(self, card):
                call_key = str(card["block_start"]) + "/" + str(card["attempt_index"])
                if not card["numeric_accept"]:
                    return dict(
                        accept=False,
                        llm_called=False,
                        actual_calls=0,
                        reason="NUMERIC_INVALID_NO_LLM",
                    )

                def invoke():
                    result = council.check_condition(card, call_key)
                    counts = zeros()
                    counts["LLM"] = int(result.get("actual_calls", 0))
                    return result, counts

                result, _, _ = runtime.ledger.call(
                    "condition_agent", int(card["block_start"]), invoke
                )
                return result

        def observed_proposal(*args, **kwargs):
            return proposal_logger(sample_condition_proposal, args, kwargs)

        values = np.zeros((total, 15), np.float32)
        values[:3] = np.c_[conditions, durations]
        audit = dict(blocks=[], accepted_blocks=0, fallback_blocks=0, attempts=0)
        with torch.inference_mode():
            cc, dd = build_condition_trajectory(
                values,
                self.transition,
                self.center,
                self.scale,
                self.lower,
                self.upper,
                Adapter() if flag else None,
                "few_shot" if flag else "numeric",
                3,
                0.25,
                6.0,
                self.device,
                seed_index,
                audit,
                **self.noise,
                project_support=True,
                support_tolerance=0.0,
                max_projection_z=0.02,
                proposal_seed_base=int(self.config["seed"]),
                proposal_sampler=observed_proposal,
            )
        assert (
            cc.shape == (total, 14)
            and dd.shape == (total,)
            and np.isfinite(cc).all()
            and np.isfinite(dd).all()
            and (dd > 0).all()
        )
        return cc, dd, audit

    def pca(self, key, rng, scale=1.0):
        s = self.stats
        noise = rng.normal(size=len(s[key + "_source_scale"])).astype(np.float32)
        value = s[key + "_source_mean"] + scale * (
            noise @ (s[key + "_source_basis"] * s[key + "_source_scale"][None]).T
        )
        return value.astype(np.float32), noise

    def bank(self, state, start, condition, durations, full_durations, seed_index, subject):
        s = self.stats
        count = zeros()
        qraw = np.r_[condition.reshape(-1), self.rhythm(full_durations, 500.0)].astype(np.float32)
        qn = (qraw - s["cond_mean"]) / s["cond_scale"]
        pc = ((state.coarse.reshape(-1) - s["coarse_mean"]) / s["coarse_scale"]).reshape(1, 12, 128)
        pd = ((state.detail.reshape(-1) - s["detail_mean"]) / s["detail_scale"]).reshape(1, 6, 256)
        pc, pd, q = [torch.from_numpy(x).to(self.device) for x in (pc, pd, qn[None])]
        base = int(self.config["seed"]) + 170003 * (int(seed_index) + 1) + 97 * (int(start) + 1)
        nominal = np.random.default_rng(base)
        sc, nc = self.pca("coarse", nominal)
        sd = []
        nd = []
        for k in range(4):
            x, n = self.pca(
                "detail",
                nominal if k == 0 else np.random.default_rng(base + 1009 + k),
                [1.0, 0.92, 1.08, 1.0][k],
            )
            sd.append(x)
            nd.append(n)

        def integrate(model, x, prefix, qq, kind, extra=None):
            for step in range(20):
                t = torch.full((len(x),), step / 20, device=self.device)
                x = x + model(x, prefix, qq, t, extra) / 20
                count[kind] += 1
            return x

        def raw(x, key, shape):
            n = len(x)
            return (
                (x.cpu().numpy().reshape(n, -1) * s[key + "_scale"][None] + s[key + "_mean"][None])
                .reshape(shape)
                .astype(np.float32)
            )

        def dc(c):
            count["coarse_decoder"] += 1
            return self.codec.coarse_decoder(torch.from_numpy(c).to(self.device))[:, 0]

        def dd(d):
            count["detail_decoder"] += 1
            return self.codec.detail_decoder(
                torch.from_numpy(d.reshape(12, 2, 256)).to(self.device)
            ).reshape(4, 3, 256)

        def record(wave, d, k, action):
            normalized = ((d.reshape(-1) - s["detail_mean"]) / s["detail_scale"]).reshape(1, 6, 256)
            metric, features = self.scorer.evaluate_candidate(
                wave,
                np.maximum(8, np.rint(durations)).astype(int),
                list(state.cycles),
                np.maximum(8, np.rint(state.durations)).astype(int),
                normalized,
                self.calibration,
                500.0,
                15,
                state.features,
                0.0,
                "global",
            )
            return dict(
                candidate_id="C" + str(k),
                decoded=wave,
                metrics=metric,
                feature_rows=features,
                action=action,
                interpolation_alpha=0.0,
                detail_residual_scale=1.0,
            )

        with torch.inference_mode():
            xc = integrate(
                self.coarse,
                torch.from_numpy(sc.reshape(1, 12, 128)).to(self.device),
                pc,
                q,
                "CoarseFlow",
            )
            coarsecontext = F.interpolate(xc, size=256, mode="linear", align_corners=False)
            xd = integrate(
                self.detail,
                torch.from_numpy(np.stack(sd).reshape(4, 6, 256)).to(self.device),
                pd.expand(4, -1, -1),
                q.expand(4, -1),
                "DetailFlow",
                coarsecontext.expand(4, -1, -1),
            )
            rc = raw(xc, "coarse", (3, 4, 128))
            rd = raw(xd, "detail", (4, 3, 2, 256))
            cw = dc(rc)
            dw = dd(rd)
            mixes = torch.tensor([0.75, 0.75, 0.75, 0.81], device=self.device)[:, None, None]
            base_waveforms = (cw[None] + mixes * dw).cpu().numpy().astype(np.float32)
            base_candidates = [
                record(base_waveforms[k], rd[k], k, "DETAIL_SOURCE_RESAMPLE") for k in range(4)
            ]
            nominal_numeric, nominal_decision = self.scorer.select_candidate(
                base_candidates, **GATES, selector_mode="numerical"
            )
            mode = self.config["candidate_mode"]
            provenance = dict(
                base_seed=base,
                base_coarse_noise=nc,
                base_detail_noises=np.stack(nd),
                base_numeric=nominal_numeric,
            )
            if mode == "detail":
                extra = []
                en = []
                for k in range(4, 8):
                    v, n = self.pca(
                        "detail",
                        np.random.default_rng(base + 1009 + k),
                        [1.0, 0.92, 1.08, 1.0][k % 4],
                    )
                    extra.append(v)
                    en.append(n)
                xe = integrate(
                    self.detail,
                    torch.from_numpy(np.stack(extra).reshape(4, 6, 256)).to(self.device),
                    pd.expand(4, -1, -1),
                    q.expand(4, -1),
                    "DetailFlow",
                    coarsecontext.expand(4, -1, -1),
                )
                re = raw(xe, "detail", (4, 3, 2, 256))
                we = (cw[None] + mixes * dd(re)).cpu().numpy().astype(np.float32)
                records = base_candidates + [
                    record(we[k], re[k], k + 4, "DETAIL_SOURCE_RESAMPLE") for k in range(4)
                ]
                allc = np.repeat(rc[None], 8, axis=0)
                alld = np.concatenate([rd, re])
                provenance.update(native_mapping=list(range(8)), extra_detail_noises=np.stack(en))
            else:
                retained = [0, nominal_numeric if nominal_numeric != 0 else 1]
                if mode == "joint":
                    xcs = []
                    newnoises = []
                    seeds = []
                    for k in range(6):
                        seed = base + 500003 * (k + 1)
                        source, n = self.pca("coarse", np.random.default_rng(seed))
                        seeds.append(seed)
                        newnoises.append(n)
                        xcs.append(
                            integrate(
                                self.coarse,
                                torch.from_numpy(source.reshape(1, 12, 128)).to(self.device),
                                pc,
                                q,
                                "CoarseFlow",
                            )
                        )
                    newc = np.stack([raw(x, "coarse", (3, 4, 128)) for x in xcs])
                    cc = torch.cat(
                        [
                            F.interpolate(x, size=256, mode="linear", align_corners=False)
                            for x in xcs
                        ]
                    )
                    newcw = [dc(c) for c in newc]
                    provenance.update(new_coarse_seeds=seeds, new_coarse_noises=np.stack(newnoises))
                else:
                    lib = self.library
                    features = self.phase.compute_ppg_shape_features(
                        state.cycles[-3:], 500.0
                    ).astype(np.float64)
                    eligible = np.flatnonzero(lib["subject_id"].astype(str) != str(subject))
                    assert len(eligible) >= 6
                    scale = lib["context_features"][eligible].reshape(-1, 14).std(0, ddof=0)
                    scale = np.where(scale > 1e-12, scale, 1.0)
                    dist = np.mean(
                        ((lib["context_features"][eligible] - features) / scale) ** 2, axis=(1, 2)
                    )
                    order = np.lexsort((eligible, dist))
                    near = eligible[order[:6]]
                    newc = lib["raw_coarse"][near]
                    nc2 = (
                        (newc.reshape(6, -1) - s["coarse_mean"][None]) / s["coarse_scale"][None]
                    ).reshape(6, 12, 128)
                    cc = F.interpolate(
                        torch.from_numpy(nc2).to(self.device),
                        size=256,
                        mode="linear",
                        align_corners=False,
                    )
                    newcw = [
                        torch.from_numpy(lib["coarse_decoded"][j]).to(self.device) for j in near
                    ]
                    provenance.update(
                        retrieval_indices=near,
                        retrieval_window_ids=lib["window_id"][near],
                        retrieval_subject_ids=lib["subject_id"][near],
                        retrieval_squared_distances=dist[order[:6]],
                        retrieval_eligible_count=len(eligible),
                        retrieval_scale=scale,
                        query_features=features,
                    )
                source = np.repeat(np.asarray(sd)[0:1], 6, axis=0).reshape(6, 6, 256)
                xe = integrate(
                    self.detail,
                    torch.from_numpy(source).to(self.device),
                    pd.expand(6, -1, -1),
                    q.expand(6, -1),
                    "DetailFlow",
                    cc,
                )
                re = raw(xe, "detail", (6, 3, 2, 256))
                we = []
                for k in range(6):
                    we.append(
                        (newcw[k] + 0.75 * dd(np.repeat(re[k : k + 1], 4, axis=0))[0])
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                records = [
                    dict(base_candidates[k], candidate_id="C" + str(i), base_native_id=k)
                    for i, k in enumerate(retained)
                ]
                records += [
                    record(
                        we[k],
                        re[k],
                        k + 2,
                        (
                            "COARSE_SOURCE_RECONDITION_DETAIL"
                            if mode == "joint"
                            else "TRAIN_MEMORY_COARSE_RECONDITION_DETAIL"
                        ),
                    )
                    for k in range(6)
                ]
                allc = np.concatenate([np.repeat(rc[None], 2, axis=0), newc])
                alld = np.concatenate([rd[retained], re])
                provenance["native_mapping"] = retained + [None] * 6
        assert len(records) == 8 and all(np.isfinite(c["decoded"]).all() for c in records)
        return (
            dict(
                candidates=records,
                coarse_latents=allc,
                detail_latents=alld,
                durations=durations.copy(),
                conditions=condition.copy(),
                prefix_cycles=state.cycles.copy(),
                prefix_durations=state.durations.copy(),
                prefix_coarse=state.coarse.copy(),
                prefix_detail=state.detail.copy(),
                origin_prefix_sha256=ash(state.cycles),
                q_raw=qraw,
                q_normalized=qn,
                source_provenance=provenance,
                base_bank_numeric=nominal_numeric,
                base_bank_decision=nominal_decision,
            ),
            count,
        )

    def _numerical_select(self, state, bank):
        records = bank["candidates"][: int(self.config["visible_J"])]
        nominal, decision = self.scorer.select_candidate(
            records, **GATES, selector_mode="numerical"
        )
        safe = [int(c[1:]) for c in decision["safe_candidates"]]
        mode = self.config["selector"]
        count = zeros()
        values = None
        if mode == "numerical":
            selected = nominal
        elif mode == "prefix_similarity":
            values = [
                float(
                    np.sqrt(
                        np.mean((phase_norm(c["decoded"]) - phase_norm(state.cycles[-3:])) ** 2)
                    )
                )
                for c in records
            ]
            selected = min(safe, key=lambda k: (values[k], k))
        else:
            raise ValueError("Unsupported numerical selector: " + mode)
        decision.update(
            selection_method=mode,
            selected="C" + str(selected),
            raw_prediction_or_similarity=values,
            ranking_safe=sorted(
                safe,
                key=lambda k: (
                    (values[k] if values is not None else records[k]["metrics"]["total"]),
                    k,
                ),
            ),
            domain="admissible visible candidates; diagnostic-only candidates excluded",
            actual_visible_J=len(records),
            actual_generated_Jmax=8,
            Agent_called=False,
            ConditionAgent_called=False,
        )
        return int(selected), decision, count

    def select(self, state, bank):
        ni, decision, counts = self._numerical_select(state, bank)
        if self.config["wave_agent"] == "off":
            return ni, decision, counts
        records = bank["candidates"][: int(self.config["visible_J"])]
        row = dict(
            bank,
            case_id=self.active_anchor["case_id"],
            subject_id=self.active_anchor["subject_id"],
            candidates=records,
        )
        summary = history_summary(state.cycles, state.durations, self.config["history_scope"])
        selected, trace = self.council.choose(
            records,
            ni,
            clean(decision),
            row,
            summary,
            wave_mode=self.config["wave_agent"],
            leave_out=self.config.get("leave_out"),
            key=self.active_anchor["anchor_key"] + "/" + str(len(state.cycles)),
        )
        counts["LLM"] += int(trace.get("actual_calls", 0))
        decision.update(
            selected="C" + str(selected),
            numeric_selected="C" + str(ni),
            Agent_invoked=True,
            Agent_called=bool(trace.get("actual_calls", 0) or trace.get("referenced_calls", 0)),
            agent_trace=trace,
            Agent_actual_calls=int(trace.get("actual_calls", 0)),
            Agent_referenced_calls=int(trace.get("referenced_calls", 0)),
        )
        return selected, decision, counts
