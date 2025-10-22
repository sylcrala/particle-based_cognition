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

        self.agent_config = config.get_agent_config() if config else {}


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
        Save adaptive learning state during graceful shutdown - FIXED VERSION
        """
        try:
            import json
            import traceback
            now = dt.now().isoformat()
            
            # Proper policy counting
            policy_counts = {}
            policy_types = {}
            
            for node_id, policy in self.policies.items():
                # Convert lambda functions to string representations
                if callable(policy):
                    policy_name = getattr(policy, '__name__', 'lambda_function')
                    if policy_name == '<lambda>':
                        # Try to identify the strategy type
                        policy_str = str(policy)
                        if 'emergent' in policy_str:
                            policy_name = 'emergent'
                        elif 'social' in policy_str:
                            policy_name = 'social'
                        else:
                            policy_name = 'custom_lambda'
                else:
                    policy_name = str(policy)
                
                # Count policy usage
                policy_counts[policy_name] = policy_counts.get(policy_name, 0) + 1
                policy_types[str(node_id)] = policy_name

            # Safe interaction sampling
            recent_interactions = []
            if self.memory:
                try:
                    # Convert interaction keys to serializable format
                    interaction_items = list(self.memory.items())[-100:]  # Last 100
                    for key, score in interaction_items:
                        recent_interactions.append({
                            'participants': list(key) if isinstance(key, tuple) else [str(key)],
                            'score': float(score),
                            'interaction_type': 'particle_interaction'
                        })
                except Exception as interaction_error:
                    self.log(f"Error processing interactions: {interaction_error}", "WARNING", "save_learning_state")
                    recent_interactions = []

            # Safe embedding processing
            embedding_stats = {
                'total_count': len(self.embeddings),
                'node_ids': list(self.embeddings.keys())[:50],  # Sample of IDs
                'dimension_info': {}
            }
            
            if self.embeddings:
                try:
                    first_embedding = next(iter(self.embeddings.values()))
                    if hasattr(first_embedding, 'shape'):
                        embedding_stats['dimension_info'] = {
                            'shape': list(first_embedding.shape),
                            'dtype': str(first_embedding.dtype)
                        }
                except Exception as embed_error:
                    self.log(f"Error processing embedding stats: {embed_error}", "WARNING", "save_learning_state")

            learning_state = {
                "timestamp": now,
                "mode": self.mode,
                "lambda_blend": float(self.lambda_blend),
                "base_metric": getattr(self.base_metric_fn, '__name__', 'unknown'),
                "embedding_stats": embedding_stats,
                "interaction_stats": {
                    "total_interactions": len(self.memory),
                    "unique_pairs": len(set(self.memory.keys())),
                    "average_score": sum(self.memory.values()) / len(self.memory) if self.memory else 0.0
                },
                "policy_stats": {
                    "active_policies": len(set(policy_counts.keys())),
                    "policy_distribution": policy_counts,
                    "node_policy_mapping": policy_types
                },
                "recent_interactions": recent_interactions,
                "system_health": {
                    "memory_size": len(self.memory),
                    "embeddings_size": len(self.embeddings),
                    "policies_size": len(self.policies)
                }
            }

            # Safe file operations
            base_path = self.agent_config.get("memory_dir")
            state_file = f"{base_path}/adaptive_shutdown_state.json"
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = state_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(learning_state, f, indent=2, default=str)
            
            # Atomic rename
            os.replace(temp_file, state_file)
            
            self.log(f"Adaptive learning state saved successfully: {embedding_stats['total_count']} embeddings, {len(self.memory)} interactions, {len(policy_counts)} policy types", 
                    level="INFO", context="save_learning_state")
            
            return True
            
        except Exception as e:
            self.log(f"Error saving learning state: {e}", level="ERROR", context="save_learning_state")
            self.log(f"Full traceback:\n{traceback.format_exc()}", "DEBUG", "save_learning_state")
            return False

    async def restore_learning_state(self):
        """
        Restore adaptive learning state from previous session - ENHANCED VERSION
        """
        try:
            base_path = self.agent_config.get("memory_dir")
            state_file = f"{base_path}/adaptive_shutdown_state.json"

            if not os.path.exists(state_file):
                self.log("No previous learning state found, starting fresh", level="INFO", context="restore_learning_state")
                return True
            
            if os.path.getsize(state_file) == 0:
                self.log("State file exists but is empty, starting fresh", level="WARNING", context="restore_learning_state")
                return True
            
            with open(state_file, 'r') as f:
                learning_state = json.load(f)
            
            # Restore system configuration
            restored_timestamp = learning_state.get('timestamp', 'unknown')
            self.log(f"Restoring adaptive learning state from: {restored_timestamp}", 
                    level="INFO", context="restore_learning_state")
            
            # Restore mode and blend settings if they match
            saved_mode = learning_state.get('mode')
            if saved_mode and saved_mode != self.mode:
                self.log(f"Mode mismatch: current={self.mode}, saved={saved_mode}", 
                        level="WARNING", context="restore_learning_state")
            
            # Restore policy mappings where possible
            if 'policy_stats' in learning_state:
                policy_mapping = learning_state['policy_stats'].get('node_policy_mapping', {})
                restored_policies = 0
                
                for node_id, policy_name in policy_mapping.items():
                    if node_id not in self.policies:  # Don't override existing policies
                        try:
                            # Try to restore strategy from name
                            if policy_name in self.strategies:
                                self.policies[node_id] = self.strategies[policy_name]
                                restored_policies += 1
                        except Exception as policy_error:
                            self.log(f"Could not restore policy {policy_name} for node {node_id}: {policy_error}", 
                                    level="DEBUG", context="restore_learning_state")
                
                if restored_policies > 0:
                    self.log(f"Restored {restored_policies} policy mappings", 
                            level="INFO", context="restore_learning_state")
            
            # Log restoration summary
            embedding_count = learning_state.get('embedding_stats', {}).get('total_count', 0)
            interaction_count = learning_state.get('interaction_stats', {}).get('total_interactions', 0)
            policy_count = learning_state.get('policy_stats', {}).get('active_policies', 0)
            
            self.log(f"Learning state restoration completed: {embedding_count} embeddings, {interaction_count} interactions, {policy_count} policy types available", 
                    level="INFO", context="restore_learning_state")
            
            return True
            
        except Exception as e:
            self.log(f"Error restoring learning state: {e}", level="ERROR", context="restore_learning_state")
            import traceback
            self.log(f"Full traceback:\n{traceback.format_exc()}", "DEBUG", "restore_learning_state")
            return False