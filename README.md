# LongMorph

Cycle-based PPG generation using a dual-stream codec, conditional block Flow, and guarded local language-model selection.

## Structure

```text
longmorph/
  data/          preprocessing, cycle coordinates, and reference datasets
  models/        codec, condition-transition, and Flow networks
  training/      model objectives and training loops
  generation/    TRAIN assets and recursive generation
  selection/     numerical scoring, Council, and local model backend
  evaluation/    pooled morphology and native-time physiological metrics
  downstream/    forecasting, heart-rate tasks, export, and reporting
  baselines/     optional generative model adapters
tests/           synthetic model, protocol, and data contracts
third_party/     dependency revisions, retrieval tool, and license notices
```

## Setup and commands

Use Python 3.10 or later and a suitable PyTorch build. From the repository root:

```bash
python -m pip install -e .
# Optional local language-model dependencies:
python -m pip install -e '.[llm]'
python -m longmorph --help
python -m unittest discover -s tests -v
```

`prepare reference|ppg-dalia|assets`, `train codec|hybrid|condition|flow`, `generate`, `evaluate`, `downstream forecast|heart-rate|export|summarize`, and `baseline train|list` are the public commands. Append `--help` to a command for its arguments. The installed `longmorph` executable provides the same interface.

## Data and generator training

Obtain [PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg%2Bdalia) or [BIDMC](https://physionet.org/content/bidmc/1.0.0/) under the provider's terms. Data and weights are supplied separately. Create ordered train/test manifests with one `group<TAB>absolute-path` row per recording, using disjoint subjects or sources. The reference builder writes phase cycles, matching waveforms, conditions, canonical subject IDs, and shared-coordinate codec inputs:

```bash
python -m longmorph prepare reference --data-sources bidmc \
  --train-files /path/to/inputs/train_files.txt --test-files /path/to/inputs/test_files.txt \
  --out-dir /path/to/new_reference
```

The main representation uses 24 cycles, 256 phase samples, 9600 waveform samples, and durations measured in native samples at 500 Hz. Numerical calibration uses shared-coordinate cycles; Council retrieval uses original phase cycles. Keep both representations in the same sequence order.

The following configuration trains compatible coarse/detail codecs, assembles the mixed-resolution model, and trains the condition and Flow networks. Paths refer to outputs from earlier commands:

```bash
REF=/path/to/new_reference
OUT=/path/to/new_training
python -m longmorph train codec --cycles-npz "$REF/data/shared_coordinate_cycles.npz" \
  --detail-channels 4 --epochs 150 --lr 0.0001 --detail-weight 1 --event-d2-weight 0 \
  --out-dir "$OUT/coarse"
python -m longmorph train codec --cycles-npz "$REF/data/shared_coordinate_cycles.npz" \
  --detail-channels 2 --detail-fullres --epochs 150 --lr 0.0001 --detail-weight 1 \
  --event-d2-weight 0 --out-dir "$OUT/detail"
python -m longmorph train hybrid --cycles-npz "$REF/data/shared_coordinate_cycles.npz" \
  --coarse-checkpoint "$OUT/coarse/models/dualstream_codec.pt" \
  --detail-checkpoint "$OUT/detail/models/dualstream_codec.pt" --out-dir "$OUT/hybrid"
python -m longmorph train condition --sequences-npz "$REF/data/train_segment_condition_sequences.npz" \
  --duration-centered-noise-scale 1.25 --duration-common-noise-scale 0.25 \
  --rollout-support-mode clip --enforce-positive-duration --out-dir "$OUT/condition"
python -m longmorph train flow --reference-run "$REF" \
  --codec "$OUT/hybrid/models/dualstream_hybrid_codec.pt" --out-dir "$OUT/flow"
python -m longmorph prepare assets --train-sequences "$REF/data/train_segment_condition_sequences.npz" \
  --train-waveforms "$REF/data/train_ppg_segments.npz" --subjects "$REF/data/train_subjects.txt" \
  --output-dir "$OUT/assets"
```

Training defaults to CUDA; use `--device cpu` where needed. Codec validation can use `--groups-file` for explicit subject grouping. The generator requires coarse `[4,128]` and full-resolution detail `[2,256]` latents. Missing or incompatible checkpoints fail explicitly.

## Generation and evaluation

Context NPZ inputs contain `context_cycles[M,3,256]`, `context_conditions[M,3,14]`, `context_durations[M,3]`, `case_id[M]`, and canonical `subject_id[M]`. They contain no future signals or labels. Use the same subject identifiers as the TRAIN assets for retrieval exclusion.

```bash
python -m longmorph generate --context /path/to/context_only.npz \
  --flow-checkpoint "$OUT/flow/models/dual_block_flow.pt" \
  --codec-checkpoint "$OUT/hybrid/models/dualstream_hybrid_codec.pt" \
  --condition-checkpoint "$OUT/condition/models/condition_transition.pt" \
  --assets "$OUT/assets" --condition-agent --wave-agent full \
  --llm-model /path/to/local_instruction_model --horizons 21 63 127 256 \
  --seed 7 --output-dir /path/to/new_generation
python -m longmorph evaluate --generated /path/to/new_generation/H256.npz \
  --seeds 7 --references /path/to/paired_future_references.npz \
  --horizon 256 --output-dir /path/to/new_evaluation
```

Paired references use `future_cycles[N,H,256]`, `future_durations[N,H]`, matching case/subject IDs, and optional `available_future_cycles[N]`. Missing references remain missing. Shape MMD squared is pooled across all eligible future cycles; it is not an average of per-case MMD values. ACF lag20/63, native-time HR/Seam, HR W1, and IBI measurements are separate fields.

The maximum horizon produces one recursive trajectory; shorter requested horizons are literal prefixes. Full Council, a single judge, role removal, and numerical-only selection are explicit options. Model weights and the exact prompts/strict JSON protocol are local; no forwarding service or provider credentials are needed.

## Downstream and baselines

Forecasting uses one entry point. `downstream forecast --protocol fixed_budget --model patchtst` selects equal-budget 5→1 prediction; `--protocol augmentation --model tcn` or `residual_tcn` selects validation-controlled 3→3 prediction. The `fixed_budget` protocol supports `--split subject_folds|source_holdout`; `augmentation` uses `subject_folds`. HR tasks use `downstream heart-rate build|train|aggregate`; export and report commands are shared.

Optional baseline revisions are pinned in `third_party/pinned_sources.json`. Inspect with `python third_party/fetch_sources.py --list`; retrieval occurs only with explicit `--fetch NAME`. Baseline training requires `baseline train --model NAME --reference-run PATH --output-dir PATH --seed 7`; AEC-GAN additionally requires observed `--context-path`. TensorFlow baselines use separate TF1/TF2 environments. In a TF1-compatible older Python environment, run `python -m longmorph.baselines train ...` from the source root with the upstream dependencies; do not install this package or its core requirements into that environment. Unknown-license dependencies require a user-managed source directory.

Original code is [MIT licensed](LICENSE); upstream notices remain in `third_party/licenses/`. Dataset and pretrained-model terms are separate. Synthetic tests verify implementation behavior, not reproduced scientific results.
