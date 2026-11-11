"""Fixed hybrid roles and Manager; real calls only for conflicting evidence.

Morphology and Detail have two disjoint fields. Continuity has one validated
field and is deterministic. Rhythm is explicitly neutral: all candidates have
the same planned duration and no validated candidate-specific IBI forecast.
"""

from __future__ import annotations
import math
from pathlib import Path
from .backend import Journal, sha
from . import context as context_module
from .protocol import (
    VERSION,
    LOSS_FIELDS,
    ROLE_FIELDS,
    wave_card,
    condition_card,
    history_card,
    collapse,
)


def relation(role, n, r):
    fields = ROLE_FIELDS[role]
    for x in (n, r):
        if any(isinstance(x[f], bool) or not math.isfinite(x[f]) or x[f] < 0 for f in fields):
            raise ValueError("Invalid role loss")
    if all(n[f] == r[f] for f in fields):
        return "EQUAL"
    if all(n[f] <= r[f] for f in fields):
        return "N_DOMINATES"
    if all(r[f] <= n[f] for f in fields):
        return "R_DOMINATES"
    return "CONFLICT"


def runnerup(records, numeric_index, decision):
    if not 0 <= numeric_index < len(records):
        raise ValueError("Numeric ID outside visible candidates")
    ids = [r["candidate_id"] for r in records]
    if ids != ["C" + str(i) for i in range(len(records))] or len(ids) not in (1, 2, 4, 8):
        raise ValueError("Candidate ID/J mismatch")
    safe = set(decision["safe_candidates"])
    if ids[numeric_index] not in safe:
        raise ValueError("Numeric is outside admissible domain")
    mode = decision.get("selection_method", "numerical")
    if mode == "numerical":
        pool = [
            i
            for i, c in enumerate(records)
            if c["candidate_id"] in decision["pareto_candidates"] and i != numeric_index
        ]
        index = (
            min(pool, key=lambda i: (float(records[i]["metrics"]["total"]), i)) if pool else None
        )
        domain = "Admissible Pareto excluding N; minimum(unrounded total loss,native index)"
    else:
        raise ValueError("Unscreened online numeric selector")
    if index is not None and ids[index] not in safe:
        raise ValueError("Runner-up outside admissible domain")
    return index, domain


class Council:
    def __init__(self, journal_root, run_id, backend, provider, source_paths=()):
        self.journal = Journal(journal_root, run_id, backend, source_paths)
        self.provider = provider

    def check_condition(self, card, key):
        envelope = condition_card(card)
        if not envelope["diagnostics"]["numeric_accept"]:
            return dict(
                schema_version=VERSION,
                role="condition",
                accept=False,
                llm_called=False,
                parse_mode="numeric_prerequisite_reject",
                actual_calls=0,
                input_card=envelope,
            )
        result = self.journal.call(str(key) + "/condition", "condition", envelope)
        parsed = result["parsed"]
        valid = result["parse_error"] is None
        return dict(
            schema_version=VERSION,
            role="condition",
            accept=bool(parsed["accept"]) if valid else False,
            llm_called=True,
            parse_mode=result["parse_mode"],
            parse_error=result["parse_error"],
            actual_calls=int(result["executed_now"]),
            referenced_calls=int(not result["executed_now"]),
            input_card=envelope,
            result=result,
        )

    def vote(self, role, n, r, history, key):
        rel = relation(role, n, r)
        fields = ROLE_FIELDS[role]
        trace = dict(
            schema_version=VERSION,
            role=role,
            fields=list(fields),
            relation=rel,
            numeric_losses={f: n[f] for f in fields},
            runnerup_losses={f: r[f] for f in fields},
            descriptive_mean_numeric=sum(n[f] for f in fields) / len(fields),
            descriptive_mean_runnerup=sum(r[f] for f in fields) / len(fields),
            descriptive_mean_is_LLM_score=False,
            actual_calls=0,
            referenced_calls=0,
            llm_called=False,
        )
        if rel != "CONFLICT":
            return dict(
                trace,
                outcome={"EQUAL": "TIE", "N_DOMINATES": "N", "R_DOMINATES": "R"}[rel],
                consistent=True,
                reason="EXACT_ROLE_LOSS_PARETO",
                history_prompt_used=False,
            )
        if role not in ("morphology", "detail", "single"):
            raise ValueError("Single-field Continuity cannot conflict")
        calls = {}
        for orientation in ("AB", "BA"):
            payload = wave_card(role, n, r, orientation, history)
            calls[orientation] = self.journal.call(
                str(key) + "/" + role + "/" + orientation, role, payload
            )
        trace.update(collapse(calls["AB"]["raw_response"], calls["BA"]["raw_response"]))
        trace.update(
            calls=calls,
            llm_called=True,
            actual_calls=sum(int(x["executed_now"]) for x in calls.values()),
            referenced_calls=sum(int(not x["executed_now"]) for x in calls.values()),
            history_prompt_used=True,
        )
        return trace

    def choose(
        self,
        records,
        numeric_index,
        numeric_decision,
        evidence_row,
        history_summary,
        wave_mode="full",
        leave_out=None,
        key="block",
    ):
        if wave_mode not in ("full", "single"):
            raise ValueError("Invalid waveform Agent mode")
        if leave_out not in (None, "morphology", "detail", "rhythm", "continuity"):
            raise ValueError("Invalid leave-one-out role")
        if wave_mode == "single" and leave_out is not None:
            raise ValueError("No LOO on Single")
        history = history_card(history_summary)
        ri, domain = runnerup(records, int(numeric_index), numeric_decision)
        ni = int(numeric_index)
        trace = dict(
            schema_version=VERSION,
            role="manager",
            manager_version="informative_strict_majority",
            wave_mode=wave_mode,
            leave_out=leave_out,
            numeric_id=records[ni]["candidate_id"],
            runnerup_id=records[ri]["candidate_id"] if ri is not None else None,
            pair_domain=domain,
            numeric_decision=numeric_decision,
            visible_candidate_ids=[c["candidate_id"] for c in records],
            history=history,
            history_scope_affects_only_agent_visibility=True,
            latent_feedback="chosen_raw_latents",
            buffer_reset_scope="anchor",
            actual_calls=0,
            referenced_calls=0,
            outcomes={},
            rhythm=dict(
                status="NEUTRAL_SHARED_PLANNED_DURATION",
                outcome="TIE",
                actual_calls=0,
                reason="No validated candidate-specific rhythm evidence; not fabricated from shared model durations.",
            ),
            provider_source=dict(
                path=str(Path(context_module.__file__)),
                sha256=sha(Path(context_module.__file__)),
            ),
            source_calibration=self.provider.calibration,
        )
        if ri is None:
            trace.update(
                selected=records[ni]["candidate_id"],
                reason="NO_LEGAL_RUNNERUP",
                active_roles=[],
                fixed_majority_denominator=0,
                changed_from_numeric=False,
                history_prompt_used=False,
            )
            return ni, trace
        row = dict(evidence_row, candidates=records)
        evidence = self.provider.evidence(row)
        n = evidence["losses"][records[ni]["candidate_id"]]
        r = evidence["losses"][records[ri]["candidate_id"]]
        trace["evidence"] = evidence
        if wave_mode == "single":
            vote = self.vote("single", n, r, history, key)
            outcomes = dict(single=vote)
            selected = ri if vote["outcome"] == "R" else ni
            trace.update(
                active_roles=["single"],
                fixed_majority_denominator=1,
                runnerup_votes=int(vote["outcome"] == "R"),
                numeric_votes=int(vote["outcome"] == "N"),
                tie_votes=int(vote["outcome"] == "TIE"),
                reason="MATCHED_SINGLE_FIVE_UNIQUE_LOSSES",
            )
        else:
            allowed = [role for role in ("morphology", "detail", "continuity") if role != leave_out]
            active = [role for role in allowed if relation(role, n, r) != "EQUAL"]
            outcomes = {role: self.vote(role, n, r, history, key) for role in active}
            votes = sum(x["outcome"] == "R" for x in outcomes.values())
            selected = ri if active and votes > len(active) / 2 else ni
            trace.update(
                active_roles=active,
                neutral_roles=[role for role in allowed if role not in active],
                fixed_majority_denominator=len(active),
                runnerup_votes=votes,
                numeric_votes=sum(x["outcome"] == "N" for x in outcomes.values()),
                tie_votes=sum(x["outcome"] == "TIE" for x in outcomes.values()),
                reason=(
                    "STRICT_RUNNERUP_MAJORITY"
                    if selected == ri
                    else "NO_STRICT_RUNNERUP_MAJORITY_NUMERIC_FALLBACK"
                ),
                denominator_fixed_before_calls=True,
                removed_before_majority=leave_out,
            )
        trace.update(
            outcomes=outcomes,
            selected=records[selected]["candidate_id"],
            changed_from_numeric=selected != ni,
            actual_calls=sum(x["actual_calls"] for x in outcomes.values()),
            referenced_calls=sum(x["referenced_calls"] for x in outcomes.values()),
            history_prompt_used=any(x["history_prompt_used"] for x in outcomes.values()),
        )
        if records[selected]["candidate_id"] not in numeric_decision["safe_candidates"]:
            raise ValueError("Agent selected unsafe ID")
        return selected, trace
