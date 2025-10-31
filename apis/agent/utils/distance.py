"""
Particle-based Cognition Engine - Distance utility functions - partially unused
Copyright (C) 2025 sylcrala

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version, subject to the additional terms 
specified in TERMS.md.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License and TERMS.md for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Additional terms apply per TERMS.md. See also ETHICS.md.
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

