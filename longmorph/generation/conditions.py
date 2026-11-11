"""Support-constrained condition proposals and recursive trajectory construction."""

from __future__ import annotations
import numpy as np
import torch


def project_condition_support(candidate, lower, upper, scale):
    raw = np.asarray(candidate, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32).reshape(-1)
    center = (lower + upper) / 2.0
    finite = np.isfinite(raw)
    filled = np.where(finite, raw, center)
    projected = np.minimum(np.maximum(filled, lower), upper).astype(np.float32)
    support_span = np.maximum(upper - lower, scale)
    correction = np.abs(projected - filled) / support_span
    changed = ~np.isclose(projected, raw, rtol=0.0, atol=0.0, equal_nan=True)
    return projected, {
        "projection_changed_fraction": float(np.mean(changed)),
        "projection_mean_abs_z": float(np.mean(correction)),
        "projection_max_abs_z": float(np.max(correction)) if correction.size else 0.0,
    }


def condition_state_card(
    candidate,
    history,
    lower,
    upper,
    scale,
    attempt_index,
    block_start,
    continuity_tolerance,
    support_tolerance=0.0,
    projection=None,
    max_projection_z=0.0,
):
    candidate = np.asarray(candidate, dtype=np.float32)
    history = np.asarray(history, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32).reshape(-1)
    support_span = np.maximum(upper - lower, scale)
    finite = np.isfinite(candidate)
    lower_gap = np.maximum(lower - candidate, 0.0)
    upper_gap = np.maximum(candidate - upper, 0.0)
    support_violation = float(np.mean(np.clip((lower_gap + upper_gap) / support_span, 0.0, 8.0)))
    nonfinite_fraction = float(1.0 - finite.mean())
    duration = candidate[:, -1]
    duration_positive_fraction = float(np.mean(np.isfinite(duration) & (duration > 0)))
    duration_cv = float(
        np.std(np.maximum(duration, 1.0)) / (np.mean(np.maximum(duration, 1.0)) + 1e-6)
    )
    if len(history):
        previous = history[-1]
        continuity_z = float(np.mean(np.abs((candidate[0] - previous) / scale)))
        previous_duration = max(float(previous[-1]), 1.0)
        rhythm_jump_z = float(
            abs(float(duration[0]) - previous_duration) / (float(scale[-1]) + 1e-6)
        )
    else:
        continuity_z = 0.0
        rhythm_jump_z = 0.0
    projection = projection or {}
    projection_mean_abs_z = float(projection.get("projection_mean_abs_z", 0.0))
    projection_max_abs_z = float(projection.get("projection_max_abs_z", 0.0))
    projection_changed_fraction = float(projection.get("projection_changed_fraction", 0.0))
    numeric_score = (
        support_violation
        + projection_mean_abs_z
        + 0.20 * nonfinite_fraction
        + 0.25 * max(0.0, continuity_z - continuity_tolerance) / max(continuity_tolerance, 1e-6)
        + 0.10 * max(0.0, rhythm_jump_z - continuity_tolerance) / max(continuity_tolerance, 1e-6)
        + 0.25 * (1.0 - duration_positive_fraction)
        + 0.05 * duration_cv
    )
    support_ok = support_violation <= support_tolerance
    projection_ok = projection_max_abs_z <= max_projection_z
    hard_accept = (
        nonfinite_fraction == 0.0
        and support_ok
        and projection_ok
        and duration_positive_fraction == 1.0
        and continuity_z <= continuity_tolerance
    )
    return {
        "attempt_index": int(attempt_index),
        "block_start": int(block_start),
        "support_violation": support_violation,
        "continuity_z": continuity_z,
        "duration_positive_fraction": duration_positive_fraction,
        "duration_cv": duration_cv,
        "rhythm_jump_z": rhythm_jump_z,
        "nonfinite_fraction": nonfinite_fraction,
        "support_tolerance": float(support_tolerance),
        "projection_changed_fraction": projection_changed_fraction,
        "projection_mean_abs_z": projection_mean_abs_z,
        "projection_max_abs_z": projection_max_abs_z,
        "max_projection_z": float(max_projection_z),
        "numeric_score": float(numeric_score),
        "numeric_accept": bool(hard_accept),
    }


@torch.no_grad()
def sample_condition_proposal(
    model,
    context_norm,
    center,
    scale,
    temperature,
    device,
    duration_centered_noise_scale=1.0,
    duration_common_noise_scale=1.0,
    proposal_seed=None,
):
    context_tensor = torch.from_numpy(context_norm[None]).float().to(device)
    mean, logstd = model(context_tensor)
    if proposal_seed is None:
        noise = torch.randn_like(mean)
    else:
        # Use a variant-independent CPU generator so full and no-agent runs
        # consume identical proposal noise even though only one calls the LLM.
        proposal_rng = torch.Generator(device="cpu")
        proposal_rng.manual_seed(int(proposal_seed) % (2**63 - 1))
        noise = torch.randn(mean.shape, generator=proposal_rng, device="cpu", dtype=mean.dtype).to(
            mean.device
        )
    noise = torch.exp(logstd) * noise
    duration_noise = noise[..., -1:]
    common = duration_noise.mean(1, keepdim=True)
    noise[..., -1:] = duration_common_noise_scale * common + duration_centered_noise_scale * (
        duration_noise - common
    )
    delta = mean + temperature * noise
    next_norm = context_tensor[:, -1:] + delta
    return (next_norm[0].cpu().numpy() * scale + center).astype(np.float32)


def build_condition_trajectory(
    seed_values,
    model,
    center,
    scale,
    lower,
    upper,
    agent,
    selection,
    attempts,
    temperature,
    continuity_tolerance,
    device,
    sample_idx,
    audit,
    duration_centered_noise_scale=1.0,
    duration_common_noise_scale=1.0,
    project_support=False,
    support_tolerance=0.0,
    max_projection_z=0.0,
    proposal_seed_base=0,
    proposal_sampler=None,
):
    sampler = sample_condition_proposal if proposal_sampler is None else proposal_sampler
    accepted = [np.asarray(seed_values[:3], dtype=np.float32)]
    records = []
    while sum(len(block) for block in accepted) < len(seed_values):
        history = np.concatenate(accepted, axis=0)
        context_norm = ((history[-3:] - center) / scale).astype(np.float32)
        block_start = len(history)
        proposals = []
        for attempt in range(max(1, int(attempts) + 1)):
            proposal = sampler(
                model,
                context_norm,
                center,
                scale,
                temperature,
                device,
                duration_centered_noise_scale,
                duration_common_noise_scale,
                proposal_seed=(
                    int(proposal_seed_base)
                    + 1000003 * (int(sample_idx) + 1)
                    + 7919 * (int(block_start) // 3 + 1)
                    + 97 * (int(attempt) + 1)
                ),
            )
            used = min(len(proposal), len(seed_values) - block_start)
            proposal = proposal[:used]
            projection = None
            if project_support:
                proposal, projection = project_condition_support(
                    proposal,
                    lower[: proposal.shape[-1]],
                    upper[: proposal.shape[-1]],
                    scale[: proposal.shape[-1]],
                )
            card = condition_state_card(
                proposal,
                history,
                lower,
                upper,
                scale,
                attempt,
                block_start,
                continuity_tolerance,
                support_tolerance=support_tolerance,
                projection=projection,
                max_projection_z=max_projection_z,
            )
            llm_result = None
            accepted_now = card["numeric_accept"]
            if selection == "few_shot" and agent is not None:
                llm_result = agent.check_condition(card)
                accepted_now = bool(card["numeric_accept"] and llm_result.get("accept", False))
            proposals.append(
                {"proposal": proposal, "card": card, "agent": llm_result, "accepted": accepted_now}
            )
            if accepted_now:
                break
        chosen = next((item for item in proposals if item["accepted"]), None)
        fallback = False
        if chosen is None:
            chosen = min(proposals, key=lambda item: item["card"]["numeric_score"])
            fallback = True
        accepted.append(chosen["proposal"])
        records.append(
            {
                "sample_index": int(sample_idx),
                "block_start": int(block_start),
                "accepted_attempt": int(chosen["card"]["attempt_index"]),
                "fallback": fallback,
                "selection": selection,
                "chosen_card": chosen["card"],
                "attempts": [
                    {
                        "card": item["card"],
                        "accepted": bool(item["accepted"]),
                        "agent": item["agent"],
                    }
                    for item in proposals
                ],
            }
        )
    trajectory = np.concatenate(accepted, axis=0)[: len(seed_values)].astype(np.float32)
    audit["blocks"].extend(records)
    audit["accepted_blocks"] += sum(1 for row in records if not row["fallback"])
    audit["fallback_blocks"] += sum(1 for row in records if row["fallback"])
    audit["attempts"] += sum(len(row["attempts"]) for row in records)
    if project_support:
        cards = [item["card"] for row in records for item in row["attempts"]]
        audit["projection_attempts"] = audit.get("projection_attempts", 0) + len(cards)
        audit["projection_changed_fraction_sum"] = audit.get(
            "projection_changed_fraction_sum", 0.0
        ) + sum(float(card.get("projection_changed_fraction", 0.0)) for card in cards)
        audit["projection_mean_abs_z_sum"] = audit.get("projection_mean_abs_z_sum", 0.0) + sum(
            float(card.get("projection_mean_abs_z", 0.0)) for card in cards
        )
    return trajectory[:, :-1], trajectory[:, -1]
