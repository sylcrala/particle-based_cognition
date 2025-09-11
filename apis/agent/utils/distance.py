"""
particle distance utility methods [refactor in progress]
"""

import math
import numpy as np

def batch_hyper_distance_matrix(positions, weights=None):
    # Determine expected length
    vector_length = positions.shape[-1]

    # Default weights: fill to match actual dimension length
    default_weights = {
        0: 1, 1: 1, 2: 1,
        3: 0.5, 4: 0.25, 5: 0.25,
        6: 0.4, 7: 0.6, 8: 0.7,
        9: 0.2, 10: 1.0
    }

    # Construct weight vector dynamically
    w = np.array([weights.get(i, 1.0) if weights else default_weights.get(i, 1.0) for i in range(vector_length)], dtype=np.float32)

    # Compute pairwise distances
    diffs = positions[:, None, :] - positions[None, :, :]  # Shape: (N, N, D)
    dists = np.sqrt(np.sum((diffs * w) ** 2, axis=2))      # Shape: (N, N)
    return dists


def get_base_metric(self, name):
    if name == "euclidean":
        return lambda a, b: np.linalg.norm(np.array(a) - np.array(b))
    elif name == "manhattan":
        return lambda a, b: sum(abs(x - y) for x, y in zip(a, b))
    return lambda a, b: 1  # fallback constant distance

