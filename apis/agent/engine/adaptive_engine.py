"""
adaptive distance engine [refactor mostly complete]
"""
import json
import math
import os
from datetime import datetime as dt
from multiprocessing import context
import random
import numpy as np
from apis.agent.cognition.particles.lingual_particle import LingualParticle

from apis.api_registry import api

config = api.get_api("config")
ae_config = config.get_adaptive_engine_config()

class AdaptiveDistanceEngine:
    def __init__(self):

        if ae_config["mode"] not in ["base", "adaptive", "blend"]:
            raise ValueError("Mode must be 'base', 'adaptive', or 'blend'")
        if not (0 <= ae_config["lambda_blend"] <= 1):
            raise ValueError("lambda_blend must be between 0 and 1")
        self.mode = ae_config["mode"]  # 'base', 'adaptive', or 'blend'
        self._tokenizer = None  # Lazy-loaded when needed
        self.lambda_blend = ae_config["lambda_blend"]
        self.base_metric_fn = self.get_base_metric(ae_config["base_metric"])
        self.memory = {}  # (id_a, id_b): interaction_score
        self.embeddings = {}  # id: xp.array (cupy or numpy)


        from apis.agent.utils.policies import strategies
        self.strategies = strategies
        self.policies = {}  # id: lambda function or string-based strategy
        
    @property
    def tokenizer(self):
        """Lazy-load tokenizer when needed"""
        if self._tokenizer is None:
            model_handler = api.get_api("_agent_model_handler")
            if model_handler and hasattr(model_handler, 'tokenizer'):
                self._tokenizer = model_handler.tokenizer
            else:
                # Fallback or warning
                print("[Adaptive Engine - tokenizer()] Model handler tokenizer not available")
                self.log("Model handler tokenizer not available", "WARNING", "tokenizer property")
        return self._tokenizer
        

    def log(self, message, level = None, context = None):
        source = "AdaptiveDistanceEngine"

        if context != None:
            context = context
        else:
            context = "no context"

        if level != None:
            level = level
        else:
            level = "INFO"

        api.call_api("logger", "log", (message, level, context, source))

    def get_base_metric(self, metric_name):
        """Get base distance metric function"""
        if metric_name == "euclidean":
            return self._euclidean_distance
        elif metric_name == "cosine":
            return self._cosine_distance
        elif metric_name == "manhattan":
            return self._manhattan_distance
        else:
            # Default to euclidean
            return self._euclidean_distance
    
    def _euclidean_distance(self, pos_a, pos_b):
        """Euclidean distance calculation"""
        import numpy as np
        return np.linalg.norm(pos_a - pos_b)
    
    def _cosine_distance(self, pos_a, pos_b):
        """Cosine distance calculation"""
        import numpy as np
        dot_product = np.dot(pos_a, pos_b)
        norms = np.linalg.norm(pos_a) * np.linalg.norm(pos_b)
        if norms == 0:
            return 1.0  # Maximum distance for zero vectors
        return 1 - (dot_product / norms)
    
    def _manhattan_distance(self, pos_a, pos_b):
        """Manhattan distance calculation"""
        import numpy as np
        return np.sum(np.abs(pos_a - pos_b))


    def set_embedding(self, node_id, vector):
        self.embeddings[node_id] = np.array(vector)


    def update_interaction(self, id_a, id_b, success_score):
        key = tuple(sorted((str(id_a), str(id_b))))
        try:
            current_score = self.memory.get(key, 0.5)
            updated_score = current_score * 0.9 + success_score * 0.1
            self.memory[key] = updated_score
        except Exception as e:
            self.log("A_engine", f"Error updating interaction between {id_a} and {id_b}: {e}", level="ERROR", context="update_interaction()")


    def set_policy(self, node_id, strategy=None):
        """
        Defines the particle's distance modulation policy based on social behavior strategy.
        These functions modify how particles influence one another's spacing.
        """
        if strategy is None:
            strategy = "emergent"
        strategies = self.strategies

        self.policies[node_id] = strategies.get(strategy, lambda d: d)  # fallback to neutral


    def get_interaction_weight(self, id_a, id_b):
        key = tuple(sorted((str(id_a), str(id_b))))
        return self.memory.get(key, 0.5)

    def adaptive_component(self, id_a, id_b):
        vec_a = self.embeddings.get(id_a, np.zeros(3))
        vec_b = self.embeddings.get(id_b, np.zeros(3))
        dist = np.linalg.norm(vec_a - vec_b)

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
            blend = np.array(self.lambda_blend, dtype=np.float32)
            return self.lambda_blend * base + (1.0 - blend) * adaptive
        else:
            return base  # default fallback
        
    def long_range_force(self, id_a, pos_a, id_b, pos_b, force_scale=0.002):
        dist = self.distance(id_a, pos_a, id_b, pos_b)
        if dist < 1e-6:
            return np.zeros_like(pos_a)  # avoid divide-by-zero or jitter

        direction = np.array(pos_b) - np.array(pos_a)
        norm_direction = direction / (np.linalg.norm(direction) + 1e-6)

        # Inverse-square-style decay (can be tuned)
        magnitude = np.array(force_scale, dtype=np.float32) / dist

        return norm_direction * magnitude

    def compare_text_to_particle(self, text: str, particle) -> float:
        """
        Computes the distance between a text prompt and a particle's embedding.
        Uses the same vector space as particle positions.
        """
        #from models.persistent_identity_kit.cognitive.lingual_particle import LingualParticle
        #from models.persistent_identity_kit.cognitive.particle_engine import batch_hyper_distance_matrix

        # Convert the input text to a temporary particle for comparison
        temp_lp = LingualParticle(
            token=text,
            content=text,
            source="meta_compare",
            temporary=True
        )
        temp_lp.energy = 0.1
        temp_lp.activation = 0.1

        if not hasattr(temp_lp, "position") or not isinstance(particle.position, np.ndarray):
            self.log(f"Invalid positions: {temp_lp.position}, {particle.position}", level="ERROR", context="compare_text_to_particle()")
            raise ValueError("Both input and particle must have valid embedding vectors.")

        try:
            from apis.agent.utils.distance import batch_hyper_distance_matrix
            positions = np.stack([temp_lp.position, particle.position])
            distance_matrix = batch_hyper_distance_matrix(positions)
            return distance_matrix[0][1]
        except Exception as e:
            self.log(f"Distance calculation error: {e}", level="ERROR", context="compare_text_to_particle()")
            raise RuntimeError(f"Distance calculation failed: {e}")
    
    def save_learning_state(self):
        """
        Save adaptive learning state during graceful shutdown
        """
        try:
            import json
            
            learning_state = {
                "timestamp": dt.now().isoformat(),
                "total_embeddings": len(self.embeddings),
                "total_interactions": len(self.memory),
                "active_policies": len(set(self.policies.values())),
                "policy_distribution": {}
            }
            
            # Count policy usage
            for policy in self.policies.values():
                learning_state["policy_distribution"][policy] = learning_state["policy_distribution"].get(policy, 0) + 1
            
            # Save interaction statistics
            if self.memory:
                recent_interactions = list(self.memory)[-100:]  # Last 100 interactions
                learning_state["recent_interaction_sample"] = len(recent_interactions)
            
            # Save to file
            state_file = "./data/agent/adaptive_shutdown_state.json"
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            
            with open(state_file, 'w') as f:
                json.dump(learning_state, f, indent=2)
            
            self.log(f"Adaptive learning state saved: {learning_state['total_embeddings']} embeddings, {learning_state['total_interactions']} interactions", 
                    level="INFO", context="save_learning_state")
            
        except Exception as e:
            self.log(f"Error saving learning state: {e}", level="ERROR", context="save_learning_state")
    
    def restore_learning_state(self):
        """
        Restore adaptive learning state from previous session
        """
        try:
            state_file = "./data/agent/adaptive_shutdown_state.json"
            
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    learning_state = json.load(f)
                
                # Log restoration info
                self.log(f"Restoring learning state from: {learning_state.get('timestamp')}", 
                        level="INFO", context="restore_learning_state")
                
                # Note: embeddings and interaction history are already persistent in memory
                # This method mainly serves to log the restoration and validate state
                
                restored_embeddings = learning_state.get('total_embeddings', 0)
                restored_interactions = learning_state.get('total_interactions', 0)
                
                self.log(f"Learning state restored: {restored_embeddings} embeddings, {restored_interactions} interactions available", 
                        level="INFO", context="restore_learning_state")
                
                return True
            else:
                self.log("No previous learning state found, starting fresh", level="INFO", context="restore_learning_state")
                return False
                
        except Exception as e:
            self.log(f"Error restoring learning state: {e}", level="ERROR", context="restore_learning_state")
            return False


