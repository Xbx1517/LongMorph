"""Owner-disjoint validation splits and reproducible random seeds."""

import random
import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def subject_disjoint_split(rows, groups, validation_fraction, seed):
    """Split sequence owners without leaking cycles from one subject.

    ``rows`` may be ``[sequence, cycle, phase]`` or ``[cycle, phase]``.  Group
    labels apply to the first axis and are intentionally kept as strings so
    dataset-specific subject identifiers do not need to be parsed.
    """
    groups = np.asarray(groups, dtype=str)
    if len(groups) != len(rows):
        raise ValueError(f"groups length {len(groups)} does not match first data axis {len(rows)}")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("subject-disjoint validation requires at least two distinct groups")
    shuffled = np.random.default_rng(seed).permutation(unique)
    valid_count = max(1, min(len(unique) - 1, int(round(validation_fraction * len(unique)))))
    valid_groups = shuffled[:valid_count]
    train_groups = shuffled[valid_count:]
    valid_mask = np.isin(groups, valid_groups)
    train_mask = ~valid_mask
    train = rows[train_mask].reshape(-1, rows.shape[-1])
    valid = rows[valid_mask].reshape(-1, rows.shape[-1])
    return train, valid, train_groups, valid_groups
