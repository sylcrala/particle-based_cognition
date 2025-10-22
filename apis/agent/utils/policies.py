"""
particle interaction strategies [complete]
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
