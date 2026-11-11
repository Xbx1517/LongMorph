"""Structured language-model decision cards; query futures are excluded."""

from __future__ import annotations
import json, math
from numbers import Real

VERSION = "longmorph_agent_card_v1"
LOSS_FIELDS = ("rise_target", "area_target", "d2_target", "sg15_target", "native_seam")
ROLE_FIELDS = {
    "morphology": LOSS_FIELDS[:2],
    "detail": LOSS_FIELDS[2:4],
    "continuity": LOSS_FIELDS[4:],
    "rhythm": (),
    "single": LOSS_FIELDS,
}
DEFINITIONS = {
    "rise_target": "absolute next3 mean phase peak-fraction forecast error / fixed TRAIN LOSO MAE",
    "area_target": "absolute next3 mean normalized phase area forecast error / fixed TRAIN LOSO MAE",
    "d2_target": "absolute next3 mean absolute phase second-difference forecast error / fixed TRAIN LOSO MAE",
    "sg15_target": "absolute next3 mean SG15 residual RMS forecast error / fixed TRAIN LOSO MAE",
    "native_seam": "native500Hz formal Seam-v2 at actual prefix/candidate join / natural TRAIN boundary median",
}
CONDITION_FIELDS = {
    "support_violation": "dimensionless TRAIN-support violation; lower",
    "continuity_z": "mean normalized last-state to next-state difference; lower",
    "duration_positive_fraction": "fraction in[0,1]; one required",
    "duration_cv": "duration standard deviation / mean; descriptor, no preferred target supplied",
    "rhythm_jump_z": "normalized adjacent duration jump; lower",
    "nonfinite_fraction": "fraction in[0,1]; zero required",
    "projection_mean_abs_z": "mean normalized support projection; lower",
    "projection_max_abs_z": "maximum normalized support projection; lower",
    "max_projection_z": "declared maximum allowed projection",
    "numeric_score": "fixed numeric risk score; lower",
    "numeric_accept": "Boolean deterministic prerequisite; must be true",
}
HISTORY_UNITS = dict(
    summary_cycles="cycle count",
    phase_mean="decoder amplitude",
    phase_sd="decoder amplitude",
    mean_abs_phase_d1="decoder amplitude per phase sample",
    mean_duration_samples="samples at500Hz",
    duration_cv="dimensionless",
)
HISTORY_FIELDS = set(HISTORY_UNITS) | {
    "scope",
    "buffer_reset_scope",
    "sample_rate_hz",
    "units",
    "actual_prompt_use",
}


def scalar(value, name, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("Nonfinite/non-numeric field: " + name)
    if nonnegative and value < 0:
        raise ValueError("Negative loss: " + name)


def history_card(history):
    if set(history) != HISTORY_FIELDS:
        raise ValueError("History schema key mismatch")
    if history["scope"] not in ("local", "cumulative") or history["buffer_reset_scope"] != "anchor":
        raise ValueError("History scope/reset mismatch")
    if history["sample_rate_hz"] != 500.0 or history["units"] != HISTORY_UNITS:
        raise ValueError("History units mismatch")
    if history["actual_prompt_use"] != "SUPPLIED_TO_AGENT_PROMPT":
        raise ValueError("History not marked actual Prompt input")
    for name in HISTORY_UNITS:
        scalar(history[name], name)
    if history["summary_cycles"] < 3 or int(history["summary_cycles"]) != history["summary_cycles"]:
        raise ValueError("History must contain actual full cycles")
    if history["scope"] == "local" and history["summary_cycles"] != 3:
        raise ValueError("Local must show last3 cycles only")
    return dict(history)


def example_history(scope="local"):
    return dict(
        scope=scope,
        buffer_reset_scope="anchor",
        summary_cycles=3,
        phase_mean=0.4,
        phase_sd=0.2,
        mean_abs_phase_d1=0.01,
        mean_duration_samples=500.0,
        duration_cv=0.08,
        sample_rate_hz=500.0,
        units=dict(HISTORY_UNITS),
        actual_prompt_use="SUPPLIED_TO_AGENT_PROMPT",
    )


def wave_card(
    role,
    n,
    r,
    orientation,
    history,
    scope="Generated candidates and actual generated prefix; no true query future.",
):
    if role not in ("morphology", "detail", "single") or orientation not in ("AB", "BA"):
        raise ValueError("Unsupported LLM role/orientation")
    fs = ROLE_FIELDS[role]
    for x in (n, r):
        for f in fs:
            scalar(x[f], f, True)
    a, b = (n, r) if orientation == "AB" else (r, n)
    return dict(
        schema_version=VERSION,
        role=role,
        scope=scope,
        history=history_card(history),
        A={f: float(a[f]) for f in fs},
        B={f: float(b[f]) for f in fs},
        relation={f: "A_LT_B" if a[f] < b[f] else "B_LT_A" if b[f] < a[f] else "EQUAL" for f in fs},
    )


def condition_card(card):
    if not set(CONDITION_FIELDS).issubset(card):
        raise ValueError("Missing Condition diagnostic")
    values = {k: card[k] for k in CONDITION_FIELDS}
    for k, v in values.items():
        if k == "numeric_accept":
            if type(v) is not bool:
                raise ValueError("numeric_accept must be Boolean")
        else:
            scalar(v, k)
    return dict(
        schema_version=VERSION,
        role="condition",
        definitions=dict(CONDITION_FIELDS),
        diagnostics=values,
        scope="Actual proposed ConditionTransition block; only past accepted condition states, no future waveform truth.",
    )


def messages(role, payload):
    if payload.get("schema_version") != VERSION or payload.get("role") != role:
        raise ValueError("Card role/version mismatch")
    if role == "condition":
        if set(payload) != {"schema_version", "role", "definitions", "diagnostics", "scope"}:
            raise ValueError("Condition envelope mismatch")
        if payload["definitions"] != CONDITION_FIELDS or set(payload["diagnostics"]) != set(
            CONDITION_FIELDS
        ):
            raise ValueError("Condition definition/input mismatch")
        condition_card(payload["diagnostics"])
        system = (
            "Validate only the exact numeric TRAIN-support diagnostics. Numeric acceptance is a prerequisite. "
            "duration_cv describes the proposal; no preferred CV is supplied. Do not infer missing physiology or true HR. "
            'Return exactly {"accept":true} or {"accept":false}.'
        )
        good = dict(
            support_violation=0.0,
            continuity_z=0.8,
            duration_positive_fraction=1.0,
            duration_cv=0.08,
            rhythm_jump_z=0.7,
            nonfinite_fraction=0.0,
            projection_mean_abs_z=0.001,
            projection_max_abs_z=0.01,
            max_projection_z=0.02,
            numeric_score=0.12,
            numeric_accept=True,
        )
        bad = dict(good, numeric_accept=False)
        examples = [
            (condition_card(good), '{"accept":true}'),
            (condition_card(bad), '{"accept":false}'),
        ]
    else:
        if role not in ("morphology", "detail", "single"):
            raise ValueError("No LLM for deterministic/neutral role")
        if set(payload) != {"schema_version", "role", "scope", "history", "A", "B", "relation"}:
            raise ValueError("Wave envelope mismatch")
        fs = ROLE_FIELDS[role]
        if any(set(payload[k]) != set(fs) for k in ("A", "B", "relation")):
            raise ValueError("Role fields mismatch")
        history_card(payload["history"])
        for f in fs:
            scalar(payload["A"][f], f, True)
            scalar(payload["B"][f], f, True)
            expected = (
                "A_LT_B"
                if payload["A"][f] < payload["B"][f]
                else "B_LT_A" if payload["B"][f] < payload["A"][f] else "EQUAL"
            )
            if payload["relation"][f] != expected:
                raise ValueError("Relation contradicts actual numeric input")
        system = (
            "Compare admissible generated PPG candidates using only the supplied losses. Every candidate field is "
            "dimensionless and lower-is-better. Forecast targets come from k8 TRAIN contexts, never the true future. "
            "History describes the actual visible generated prefix, with explicit units; it supplies no clinical target "
            "and is shared by A and B. Do not treat a longer buffer as evidence that either candidate is better. "
            "A_LT_B means A has smaller loss, B_LT_A means B has smaller loss, EQUAL means exact equality. "
            "Balance conflicting supplied losses; use TIE for equal or insufficient evidence. Do not invent absent "
            "physiology, true HR, candidate-specific IBI, future labels or additional fields. "
            'Return exactly {"winner":"A"}, {"winner":"B"}, or {"winner":"TIE"}. Definitions='
            + json.dumps({f: DEFINITIONS[f] for f in fs}, separators=(",", ":"))
        )
        better = {f: 0.25 for f in LOSS_FIELDS}
        worse = {f: 0.75 for f in LOSS_FIELDS}
        examples = [
            (
                wave_card(
                    role,
                    better,
                    worse,
                    o,
                    example_history(),
                    "Synthetic schema illustration; not measured quality.",
                ),
                json.dumps(dict(winner=w), separators=(",", ":")),
            )
            for o, w in (("AB", "A"), ("BA", "B"))
        ]
    result = [dict(role="system", content=system)]
    for card, response in examples:
        result += [
            dict(role="user", content=json.dumps(card, separators=(",", ":"))),
            dict(role="assistant", content=response),
        ]
    result.append(dict(role="user", content=json.dumps(payload, separators=(",", ":"))))
    return result


def parse(raw, role):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("Duplicate JSON key")
            out[k] = v
        return out

    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except (ValueError, TypeError):
        return None, "INVALID_JSON"
    if role == "condition":
        valid = (
            isinstance(value, dict) and set(value) == {"accept"} and type(value["accept"]) is bool
        )
    else:
        valid = (
            isinstance(value, dict)
            and set(value) == {"winner"}
            and isinstance(value["winner"], str)
            and value["winner"] in ("A", "B", "TIE")
        )
    return (value, None) if valid else (None, "INVALID_SCHEMA")


def legal_texts(role):
    return (
        ['{"accept":true}', '{"accept":false}']
        if role == "condition"
        else [json.dumps(dict(winner=w), separators=(",", ":")) for w in ("A", "B", "TIE")]
    )


def collapse(ab, ba):
    a, e1 = parse(ab, "wave")
    b, e2 = parse(ba, "wave")
    if e1 or e2:
        return dict(outcome="TIE", consistent=False, reason="INVALID_OUTPUT", errors=[e1, e2])
    na = {"A": "N", "B": "R", "TIE": "TIE"}[a["winner"]]
    nb = {"A": "R", "B": "N", "TIE": "TIE"}[b["winner"]]
    return dict(
        outcome=na if na == nb else "TIE",
        consistent=na == nb,
        reason="CONSISTENT" if na == nb else "SWAP_DISAGREEMENT",
        canonical_AB=na,
        canonical_BA=nb,
    )


class LegalJSON:
    def __init__(self, tok, role, n, max_tokens=16):
        self.root = {}
        self.n = n
        self.eos = tok.eos_token_id
        if self.eos is None:
            raise ValueError("EOS required")
        for text in legal_texts(role):
            ids = tok.encode(text, add_special_tokens=False)
            if (
                tok.decode(ids, skip_special_tokens=True).strip() != text
                or len(ids) + 1 > max_tokens
            ):
                raise ValueError("Legal JSON tokenizer/budget mismatch")
            node = self.root
            for token in ids + [self.eos]:
                node = node.setdefault(int(token), {})

    def __call__(self, batch_id, ids):
        node = self.root
        for token in ids.tolist()[self.n :]:
            if int(token) not in node:
                raise ValueError("Output left exact JSON trie")
            node = node[int(token)]
        return sorted(node) if node else [self.eos]
