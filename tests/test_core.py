"""Synthetic contracts for the public model and recursive generation interfaces."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.signal import resample
import torch

from longmorph.data.cycles import allocate_lengths, shared_cycles
from longmorph.generation.assets import build
from longmorph.generation.conditions import build_condition_trajectory
from longmorph.generation.generate import context_read, generate
from longmorph.generation.runtime import State, history_summary
from longmorph.models.codec import DualStreamCodec, split_components, load_generator_codec
from longmorph.models.condition import ConditionTransition
from longmorph.models.flow import BlockVelocity, CycleCouplingPredictor
from longmorph.selection import protocol
from longmorph.selection.context import ContextProvider
from longmorph.selection.council import Council
from longmorph.training.codec import reconstruction_loss
from longmorph.training.flow import flow_objective
from longmorph.training.splits import subject_disjoint_split


class ScriptedBackend:
    path = "SYNTHETIC_TEST_RESPONSE_BACKEND"

    def __init__(self, invalid=False):
        self.calls = []
        self.invalid = invalid

    def generate(self, role, payload):
        self.calls.append((role, payload))
        fields = protocol.ROLE_FIELDS[role]
        winner = "B" if payload["A"][fields[0]] > payload["B"][fields[0]] else "A"
        raw = (
            "INVALID TEST OUTPUT"
            if self.invalid
            else json.dumps(dict(winner=winner), separators=(",", ":"))
        )
        parsed, error = protocol.parse(raw, role)
        return dict(
            raw_response=raw, parsed=parsed, parse_error=error, parse_mode="SYNTHETIC_TEST_ONLY"
        )


class SyntheticProvider:
    calibration = dict(SYNTHETIC_TEST_ONLY=True)

    def evidence(self, row):
        return dict(
            losses={
                "C0": dict(
                    rise_target=0.8,
                    area_target=0.2,
                    d2_target=0.8,
                    sg15_target=0.2,
                    native_seam=0.1,
                ),
                "C1": dict(
                    rise_target=0.2,
                    area_target=0.8,
                    d2_target=0.2,
                    sg15_target=0.8,
                    native_seam=0.2,
                ),
            }
        )


def synthetic_training_data():
    rng = np.random.default_rng(7)
    phase = np.linspace(0, 1, 256, dtype=np.float32)
    cycles = []
    for sequence in range(6):
        row = []
        for beat in range(24):
            center, width = 0.2 + 0.1 * rng.random(), 0.09 + 0.06 * rng.random()
            pulse = np.exp(-(((phase - center) / width) ** 2)) + (
                0.08 + 0.1 * rng.random()
            ) * np.exp(-(((phase - 0.6) / 0.13) ** 2))
            row.append((pulse + rng.normal(0, 0.001, 256)).astype(np.float32))
        cycles.append(row)
    return np.asarray(cycles, np.float32), rng.uniform(350, 550, (6, 24)).astype(np.float32)


class CoreContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_models_and_objective_gradients(self):
        x = torch.rand(2, 1, 256)
        codec = DualStreamCodec(4, 2, True)
        values = codec(x)
        self.assertEqual(
            [tuple(v.shape) for v in values],
            [(2, 1, 256), (2, 1, 256), (2, 1, 256), (2, 4, 128), (2, 2, 256)],
        )
        coarse, detail = split_components(x)
        torch.testing.assert_close(coarse + detail, x)
        weights = SimpleNamespace(
            raw_weight=1.0, coarse_weight=1.0, detail_weight=1.0, d1_weight=0.3, event_d2_weight=0.0
        )
        loss, _ = reconstruction_loss(codec, x, weights)
        loss.backward()
        self.assertTrue(
            all(p.grad is not None and torch.isfinite(p.grad).all() for p in codec.parameters())
        )
        transition = ConditionTransition(15)
        self.assertEqual(
            [tuple(v.shape) for v in transition(torch.randn(2, 3, 15))], [(2, 3, 15), (2, 3, 15)]
        )
        c, d, coupling = BlockVelocity(12, 46), BlockVelocity(6, 46, 12), CycleCouplingPredictor(4)
        batch = [
            torch.randn(2, 12, 128),
            torch.randn(2, 6, 256),
            torch.randn(2, 12, 128),
            torch.randn(2, 6, 256),
            torch.randn(2, 12, 128),
            torch.randn(2, 6, 256),
            torch.randn(2, 46),
        ]
        loss, terms = flow_objective(c, d, coupling, batch, times=torch.tensor([0.2, 0.7]))
        self.assertTrue(all(np.isfinite(value) for value in terms.values()))
        loss.backward()
        for model in (c, d, coupling):
            self.assertTrue(
                all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            )

    def test_generator_codec_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(dict(model=DualStreamCodec(4, 4, False).state_dict()), root / "halfres.pt")
            with self.assertRaisesRegex(ValueError, "two full-resolution detail"):
                load_generator_codec(root / "halfres.pt")
            torch.save(dict(model=DualStreamCodec(4, 2, True).state_dict()), root / "mixed.pt")
            model = load_generator_codec(root / "mixed.pt")
            self.assertEqual(tuple(model(torch.rand(1, 1, 256))[-1].shape), (1, 2, 256))

    def test_coordinates_and_owner_disjointness(self):
        durations = np.asarray([300.0, 400.0, 500.0], np.float32)
        lengths = allocate_lengths(durations, 960, 16)
        self.assertEqual(lengths.sum(), 960)
        self.assertTrue(np.all(lengths >= 16))
        wave = np.sin(np.linspace(0, 20, 960)).astype(np.float32)[None]
        cycles = shared_cycles(wave, durations[None], 960, 3, 256)
        self.assertEqual(cycles.shape, (1, 3, 256))
        rows = np.zeros((6, 3, 256), np.float32)
        _, _, tr, va = subject_disjoint_split(rows, ["A", "A", "B", "B", "C", "C"], 0.2, 7)
        self.assertFalse(set(tr) & set(va))

    def test_condition_proposal_callback_and_fallback(self):
        values = np.zeros((9, 15), np.float32)
        values[:3, -1] = 500
        calls = []

        def sampler(*args, **kwargs):
            calls.append(kwargs["proposal_seed"])
            block = np.zeros((3, 15), np.float32)
            block[:, -1] = 500
            return block

        audit = dict(blocks=[], accepted_blocks=0, fallback_blocks=0, attempts=0)
        c, d = build_condition_trajectory(
            values,
            None,
            np.zeros(15),
            np.ones(15),
            np.r_[np.zeros(14), 100],
            np.r_[np.ones(14), 1000],
            None,
            "numeric",
            3,
            0.25,
            6.0,
            "cpu",
            0,
            audit,
            project_support=True,
            proposal_sampler=sampler,
        )
        self.assertEqual(c.shape, (9, 14))
        np.testing.assert_array_equal(d, np.full(9, 500))
        self.assertEqual(len(calls), 2)
        self.assertEqual(audit["fallback_blocks"], 0)
        self.assertEqual(len(set(calls)), 2)

        def outside_sampler(*args, **kwargs):
            proposal = np.full((3, 15), 2.0, np.float32)
            proposal[:, -1] = 500.0
            return proposal

        audit = dict(blocks=[], accepted_blocks=0, fallback_blocks=0, attempts=0)
        _, d = build_condition_trajectory(
            values,
            None,
            np.zeros(15),
            np.ones(15),
            np.r_[np.zeros(14), 100],
            np.r_[np.ones(14), 1000],
            None,
            "numeric",
            3,
            0.25,
            6.0,
            "cpu",
            0,
            audit,
            project_support=True,
            max_projection_z=0.0,
            proposal_sampler=outside_sampler,
        )
        self.assertEqual(audit["fallback_blocks"], 2)
        self.assertEqual(audit["attempts"], 8)

    def test_raw_feedback_and_history(self):
        state = State(
            np.zeros((3, 256)), np.full(3, 500.0), np.zeros((3, 4, 128)), np.zeros((3, 2, 256))
        )
        bank = dict(
            coarse_latents=np.stack([np.zeros((3, 4, 128)), np.ones((3, 4, 128))]),
            detail_latents=np.stack([np.zeros((3, 2, 256)), np.full((3, 2, 256), 2)]),
            candidates=[
                dict(decoded=np.zeros((3, 256)), feature_rows={}),
                dict(decoded=np.ones((3, 256)), feature_rows={}),
            ],
            durations=np.full(3, 400.0),
        )
        state.accept(bank, 1)
        bank["coarse_latents"][1] = 7
        np.testing.assert_array_equal(state.coarse, np.ones((3, 4, 128)))
        self.assertEqual(
            protocol.history_card(history_summary(state.cycles, state.durations, "local"))[
                "summary_cycles"
            ],
            3,
        )
        self.assertEqual(
            protocol.history_card(history_summary(state.cycles, state.durations, "cumulative"))[
                "summary_cycles"
            ],
            6,
        )

    def test_structured_protocol_and_swap(self):
        self.assertEqual(protocol.parse('{"accept":true}', "condition"), (dict(accept=True), None))
        for raw in (
            '{"accept":1}',
            '{"accept":true,"extra":0}',
            '{"accept":true,"accept":false}',
            'text {"accept":true}',
        ):
            self.assertIsNotNone(protocol.parse(raw, "condition")[1])
        self.assertEqual(protocol.collapse('{"winner":"B"}', '{"winner":"A"}')["outcome"], "R")
        card = protocol.wave_card(
            "morphology",
            dict(rise_target=0.2, area_target=0.8),
            dict(rise_target=0.8, area_target=0.2),
            "AB",
            protocol.example_history(),
        )
        self.assertEqual(
            [x["role"] for x in protocol.messages("morphology", card)],
            ["system", "user", "assistant", "user", "assistant", "user"],
        )

        class Tokenizer:
            eos_token_id = 999
            vocabulary = {'{"winner":"A"}': [1], '{"winner":"B"}': [2], '{"winner":"TIE"}': [3]}

            def encode(self, value, **kwargs):
                return self.vocabulary[value]

            def decode(self, value, **kwargs):
                return next(text for text, tokens in self.vocabulary.items() if value == tokens)

        trie = protocol.LegalJSON(Tokenizer(), "morphology", 2)
        self.assertEqual(trie(0, np.array([10, 20])), [1, 2, 3])
        self.assertEqual(trie(0, np.array([10, 20, 2])), [999])

    def test_council_modes_and_invalid_fallback(self):
        records = [
            dict(candidate_id="C0", metrics=dict(total=0.1)),
            dict(candidate_id="C1", metrics=dict(total=0.2)),
        ]
        decision = dict(
            selection_method="numerical",
            safe_candidates=["C0", "C1"],
            pareto_candidates=["C0", "C1"],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode, leave_out in [
                ("full", None),
                ("single", None),
                ("full", "detail"),
                ("full", "morphology"),
                ("full", "rhythm"),
                ("full", "continuity"),
            ]:
                backend = ScriptedBackend()
                council = Council(
                    root / (mode + str(leave_out)), "SYNTHETIC", backend, SyntheticProvider()
                )
                selected, trace = council.choose(
                    records,
                    0,
                    decision,
                    {},
                    protocol.example_history(),
                    wave_mode=mode,
                    leave_out=leave_out,
                )
                self.assertIn(selected, (0, 1))
                self.assertEqual(trace["actual_calls"], len(backend.calls))
            backend = ScriptedBackend(invalid=True)
            council = Council(root / "invalid", "SYNTHETIC", backend, SyntheticProvider())
            selected, trace = council.choose(records, 0, decision, {}, protocol.example_history())
            self.assertEqual(selected, 0)
            self.assertEqual(trace["reason"], "NO_STRICT_RUNNERUP_MAJORITY_NUMERIC_FALLBACK")

    def test_assets_and_recursive_untrained_generation(self):
        cycles, durations = synthetic_training_data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez_compressed(root / "train.npz", cycles=cycles, durations=durations)
            waveforms = np.stack(
                [np.concatenate([resample(cycle, 400) for cycle in row]) for row in cycles]
            )
            np.savez_compressed(root / "waveforms.npz", x=waveforms[:, :, None].astype(np.float32))
            (root / "subjects.txt").write_text("A\nA\nB\nB\nC\nC\n")
            metadata = build(
                root / "train.npz",
                root / "subjects.txt",
                root / "assets",
                train_waveforms=root / "waveforms.npz",
            )
            self.assertEqual(metadata["windows"], 24)
            provider = ContextProvider(root / "assets")
            forecast = provider.forecast(cycles[0, :3], "A", "SYNTHETIC")
            self.assertNotIn("A", forecast["neighbor_subject_ids"])
            context = dict(
                context_cycles=cycles[:1, :3],
                context_durations=np.full((1, 3), 450.0, np.float32),
                context_conditions=np.zeros((1, 3, 14), np.float32),
                case_id=np.array(["CASE"]),
                subject_id=np.array(["A"]),
            )
            np.savez_compressed(root / "context.npz", **context)
            np.savez_compressed(root / "bad.npz", **context, future_truth=np.zeros(1))
            with self.assertRaises(AssertionError):
                context_read(root / "bad.npz")
            codec = DualStreamCodec(4, 2, True)
            torch.save(dict(model=codec.state_dict()), root / "codec.pt")
            condition = ConditionTransition(15)
            for p in condition.parameters():
                p.data.zero_()
            condition.logstd.bias.data.fill_(-4)
            torch.save(
                dict(
                    model=condition.state_dict(),
                    center=np.r_[np.zeros(14), 450].astype(np.float32),
                    scale=np.ones(15, np.float32),
                    support_lower=np.r_[np.full(14, -1.0), 100].astype(np.float32),
                    support_upper=np.r_[np.ones(14), 1000].astype(np.float32),
                ),
                root / "condition.pt",
            )
            stats = dict(cond_mean=np.zeros(46, np.float32), cond_scale=np.ones(46, np.float32))
            for name, size in [("coarse", 3 * 4 * 128), ("detail", 3 * 2 * 256)]:
                stats[name + "_mean"] = np.zeros(size, np.float32)
                stats[name + "_scale"] = np.ones(size, np.float32)
                stats[name + "_source_mean"] = np.zeros(size, np.float32)
                stats[name + "_source_basis"] = np.zeros((size, 1), np.float32)
                stats[name + "_source_scale"] = np.ones(1, np.float32)
            torch.save(
                dict(
                    coarse_flow=BlockVelocity(12, 46).state_dict(),
                    detail_flow=BlockVelocity(6, 46, 12).state_dict(),
                    stats=stats,
                ),
                root / "flow.pt",
            )
            config = dict(
                context_npz=str(root / "context.npz"),
                flow_checkpoint=str(root / "flow.pt"),
                codec_checkpoint=str(root / "codec.pt"),
                condition_checkpoint=str(root / "condition.pt"),
                assets=str(root / "assets"),
                output_dir=str(root / "generated"),
                device="cpu",
                seed=7,
                selector="numerical",
                candidate_mode="detail",
                retrieval_library=None,
                condition_agent=False,
                wave_agent="off",
                leave_out=None,
                seed_policy="window_index",
                run_id="SYNTHETIC",
                visible_J=4,
                horizons=[1, 4],
                history_scope="cumulative",
            )
            generate(config)
            with np.load(root / "generated/H1.npz") as a, np.load(root / "generated/H4.npz") as b:
                np.testing.assert_array_equal(a["future_cycles"], b["future_cycles"][:, :1])
                self.assertEqual(b["future_cycles"].shape, (1, 4, 256))
            completion = json.loads((root / "generated/COMPLETE.json").read_text())
            self.assertFalse(completion["query_future_read"])
            self.assertEqual(completion["actual_forward_counts"]["LLM"], 0)


if __name__ == "__main__":
    unittest.main()
