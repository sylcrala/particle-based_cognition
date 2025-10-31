"""
Particle-based Cognition Engine - particle interaction strategies
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
import random

strategies = {
    "cooperative": lambda d: d * 0.75,             # Gently attracts
    "avoidant": lambda d: d * 1.3,                 # Repels more strongly
    "chaotic": lambda d: d * random.uniform(0.8, 1.2),  # Semi-random reaction
    "inquisitive": lambda d: max(d * 0.6, 0.1),    # Tightly draws closer
    "dormant": lambda d: d * 1.0,                  # Passive/neutral
    "disruptive": lambda d: d + random.uniform(0.1, 0.4),  # Disruptive and jittery
    "reflective": lambda d: (d * 0.85) if d > 0.5 else (d * 1.1),  # Moves closer if far, distant if close
    "emergent": lambda d: math.sin(d * math.pi) + 1  # Strange attractor / emergence pattern
}
