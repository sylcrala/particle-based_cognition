import math
from utils.computation.tensor_utils import xp, cast_scalar
import random

class AdaptiveDistanceEngine:
    def __init__(self, base_metric="euclidean", mode="adaptive", lambda_blend=0.5, log_callback=None):
        self.log_callback = log_callback
        if mode not in ["base", "adaptive", "blend"]:
            raise ValueError("Mode must be 'base', 'adaptive', or 'blend'")
        if not (0 <= lambda_blend <= 1):
            raise ValueError("lambda_blend must be between 0 and 1")
        self.mode = mode  # 'base', 'adaptive', or 'blend'
        self.lambda_blend = lambda_blend
        self.base_metric_fn = self.get_base_metric(base_metric)
        self.memory = {}  # (id_a, id_b): interaction_score
        self.embeddings = {}  # id: xp.array (cupy or numpy)
        self.policies = {}  # id: lambda function or string-based strategy

    def get_base_metric(self, name):
        if name == "euclidean":
            return lambda a, b: xp.linalg.norm(xp.array(a) - xp.array(b))
        elif name == "manhattan":
            return lambda a, b: sum(abs(x - y) for x, y in zip(a, b))
        # You can add more here
        return lambda a, b: 1  # fallback constant distance

    def set_embedding(self, node_id, vector):
        self.embeddings[node_id] = xp.array(vector)

    def update_interaction(self, id_a, id_b, success_score):
        key = tuple(sorted((str(id_a), str(id_b))))
        try:
            current_score = self.memory.get(key, 0.5)
            updated_score = current_score * 0.9 + success_score * 0.1
            self.memory[key] = updated_score
        except Exception as e:
            self.log_callback(f"Error updating interaction between {id_a} and {id_b}: {e}")

    def set_policy(self, node_id, strategy="emergent"):
        """
        Defines the particle's distance modulation policy based on social behavior strategy.
        These functions modify how particles influence one another's spacing.
        """
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

        self.policies[node_id] = strategies.get(strategy, lambda d: d)  # fallback to neutral

    def get_interaction_weight(self, id_a, id_b):
        key = tuple(sorted((str(id_a), str(id_b))))
        return self.memory.get(key, 0.5)

    def adaptive_component(self, id_a, id_b):
        vec_a = self.embeddings.get(id_a, xp.zeros(3))
        vec_b = self.embeddings.get(id_b, xp.zeros(3))
        dist = xp.linalg.norm(vec_a - vec_b)

        score = self.get_interaction_weight(id_a, id_b)
        mod_factor = 1 - (score - 0.5)  # 0.5 is neutral

        base_adaptive = dist * mod_factor

        # Apply policies
        policy_a = self.policies.get(id_a, lambda d: d)
        policy_b = self.policies.get(id_b, lambda d: d)
        return (policy_a(base_adaptive) + policy_b(base_adaptive)) / 2

    def distance(self, id_a, pos_a, id_b, pos_b):
        base = self.base_metric_fn(pos_a, pos_b)
        adaptive = self.adaptive_component(id_a, id_b)

        if self.mode == "base":
            return base
        elif self.mode == "adaptive":
            return adaptive
        elif self.mode == "blend":
            blend = cast_scalar(self.lambda_blend)
            return blend * base + (cast_scalar(1.0 - self.lambda_blend)) * adaptive
        else:
            return base  # default fallback
        
    def long_range_force(self, id_a, pos_a, id_b, pos_b, force_scale=0.002):
        dist = self.distance(id_a, pos_a, id_b, pos_b)
        if dist < 1e-6:
            return xp.zeros_like(pos_a)  # avoid divide-by-zero or jitter

        direction = xp.array(pos_b) - xp.array(pos_a)
        norm_direction = direction / (xp.linalg.norm(direction) + 1e-6)

        # Inverse-square-style decay (can be tuned)
        magnitude = cast_scalar(force_scale) / dist

        return norm_direction * magnitude


