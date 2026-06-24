"""Reward shaping utilities."""

import numpy as np


def proximity_penalty_from_distances(distances, sensing_range, penalty_scale):
    """Return a bounded penalty when a hazard appears inside sensing range."""
    if penalty_scale <= 0 or sensing_range <= 0:
        return 0.0

    clipped_distances = np.asarray(distances, dtype=np.float32)
    proximities = np.clip((sensing_range - clipped_distances) / sensing_range, 0.0, 1.0)
    return float(np.max(proximities) * penalty_scale)
