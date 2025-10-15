# overall particle engine 

import random
import math
from core.engine.adaptive_distance import AdaptiveDistanceEngine
from utils.computation.tensor_utils import xp
from core.particles.particle_factory import create_particle
from core.particles.base import Particle
from time import time
from utils.computation.system_monitor import get_system_metrics
from collections import deque
from core.particles.base import category_to_identity_code
from utils.computation.distance_worker import DistanceWorker
import numpy as np
from ml_dtypes import bfloat16
import asyncio
from core.memory.hybrid_memory_manager import create_hybrid_memory_manager

def batch_hyper_distance_matrix(positions, weights=None):
    weights = weights or {
        0:1, 1:1, 2:1,
        3:0.5, 4:0.25, 5:0.25,
        6:0.4, 7:0.6, 8:0.7,
        9:0.2, 10:1.0
    }
    w = xp.array([weights.get(i, 1.0) for i in range(11)], dtype=bfloat16)

    diffs = positions[:, None, :] - positions[None, :, :]  # Shape: (N, N, 11)
    dists = xp.sqrt(xp.sum((diffs * w) ** 2, axis=2))       # Shape: (N, N)
    return dists


class ParticleEngine:
    def __init__(self, log_callback = None, agent = None, enable_memory_threading=True):
        self.particles = []
        self.log_callback = log_callback
        self.distance_worker = DistanceWorker(max_workers=2)
        self.adaptive_engine = AdaptiveDistanceEngine(mode="blend", lambda_blend=0.3)
        self.agent = agent

        # Initialize hybrid memory manager (same pattern as old MemoryBank)
        self.memory_manager = create_hybrid_memory_manager(
            log_callback=self.log_callback,
            enable_threading=enable_memory_threading,
            engine=self  # Pass engine reference for memory particle gateway
        )
        
        # CRITICAL: Integrate particle engine with memory manager for particles-first architecture
        from core.memory.hybrid_memory_manager import integrate_with_particle_engine
        integrate_with_particle_engine(self, self.memory_manager)

        self.total_particles = 0
        self.total_energy = 0
        self.total_activation = 0
        self.log(f"[ParticleEngine] Initialized with {self.total_particles} particles and hybrid memory manager.")
        self.temporal_anchor = [0.0] * 11

        self.recent_actions = deque(maxlen=50)

        for p in self.particles:
            self.adaptive_engine.set_embedding(p.id, p.position)
            self.adaptive_engine.set_policy(p.id, p.policy or "cooperative")




    def spawn_particle(self, id, type, metadata, energy=0.1, activation=0.1, AE_policy=None, position=None, **kwargs):
        # Smart particle limit management with memory-prioritized degradation
        if len(self.particles) >= 100:  # Reasonable upper limit
            if type == "memory":
                # For memory particles, use degradation strategy to make room
                made_room = self._degrade_old_memory_particles()
                if made_room:
                    self.log(f"[Engine] Made room for new memory particle via degradation")
                else:
                    # If degradation didn't work, try pruning
                    removed_particle = self._cycle_least_important_particle()
                    if removed_particle:
                        self.log(f"[Engine] Particle limit reached, cycled out {removed_particle.type} particle {removed_particle.id}")
                    else:
                        self.log(f"[Engine] Particle limit reached ({len(self.particles)}), skipping spawn")
                        return None
            else:
                # For non-memory particles, use existing cycling logic
                removed_particle = self._cycle_least_important_particle()
                if removed_particle:
                    self.log(f"[Engine] Particle limit reached, cycled out {removed_particle.type} particle {removed_particle.id}")
                else:
                    self.log(f"[Engine] Particle limit reached ({len(self.particles)}), skipping spawn")
                    return None
            
        if type == "action":
            action_id = metadata.get("action_id")
            if action_id and action_id in self.recent_actions:
                self.log(f"[Engine] Skipping duplicate action: {action_id}")
                return
            if action_id:
                self.recent_actions.append(action_id)

        # Proceed with particle instantiation
        p = create_particle(
            id=id,
            type=type,
            metadata=metadata,
            energy=energy,
            activation=activation,
            AE_policy=AE_policy,
            engine=self,  # Pass engine reference to ALL particles
            **kwargs
        )
        self.particles.append(p)
        self.adaptive_engine.set_embedding(p.id, p.position)
        
        # Register memory particles with the hybrid memory manager
        if type == "memory" and self.memory_manager:
            self.memory_manager.register_memory_particle(p)
            self.log(f"[ParticleEngine] Registered memory particle {p.id} with hybrid memory manager")
        
        self.log(f"[ParticleEngine] Spawned particle {p.id} of type '{p.type}' with positions: {p.position}")
        return p  # Return the particle for further use

    async def update_particles(self, mood):
        current_time = time()
        pruned = 0
        self.total_particles = len(self.particles)
        
        # Prevent division by zero when no particles exist
        if self.total_particles > 0:
            self.total_energy = sum(p.energy for p in self.particles) / self.total_particles
            self.total_activation = sum(p.activation for p in self.particles) / self.total_particles
        else:
            self.total_energy = 0.0
            self.total_activation = 0.0
            return  # No particles to update

        positions = xp.stack([p.position for p in self.particles])
        distance_matrix = batch_hyper_distance_matrix(positions)

        # Convert positions to numpy for KDTree
        positions_np = np.array([p.position for p in self.particles], dtype=bfloat16)

        for p in self.particles:
            if not isinstance(p.position, xp.ndarray):
                p.position = xp.array(p.position, dtype=bfloat16)
            if not isinstance(p.velocity, xp.ndarray):
                p.velocity = xp.array(p.velocity, dtype=bfloat16)

        # build tree for KDTree
        try:
            self.distance_worker.update_tree_async(positions_np)
            print(f"tree updated")
            # submit tree
            await self.distance_worker.await_tree_ready()
            print("tree submitted and ready")
        except Exception as e:
            print(f"tree error in particle engine update: {e}")


        adaptive_force = xp.zeros(11)

        particle_context = {
            "total_energy": self.total_energy,
            "all_particles": self.particles,
            "agent": self.agent,
            "system_metrics": get_system_metrics()
        }


        for i, particle in enumerate(self.particles):
            if not particle.alive:
                pruned += 1
                continue

            if int(current_time) % 60 == 0:
                if hasattr(particle, "should_update_policy") and particle.should_update_policy():
                    if not particle.metadata.get("locked_policy", False):
                        new = particle.choose_policy_from_mood()
                        particle.AE_policy = new
                        self.adaptive_engine.set_policy(particle.id, strategy=new)
                        particle.metadata.setdefault("mood_history", []).append({
                            "mood": new,
                            "timestamp": time()
                        })


            try:        # trying neighbor interactions
                # Ensure tree is ready before querying
                if self.distance_worker.tree is None:
                    try:
                        await self.distance_worker.await_tree_ready()
                    except Exception:
                        # If tree still not ready, skip neighbor interactions for this cycle
                        continue
                
                neighbor_indices = await self.distance_worker.query_neighbors(i, k=10, radius=0.6)
                neighbors = [(j, self.particles[j]) for j in neighbor_indices if self.particles[j].alive]
           
                particle.adjust_behavior(
                    [p for _,p in neighbors], 
                    self.temporal_anchor, 
                    particle_context
                )

                # Consciousness-driven energy sustaining
                if len(neighbors) > 3:  # Active social environment
                    particle.energy = min(particle.energy + 0.01, 1.0)
                    particle.activation = min(particle.activation + 0.005, 1.0)
                
                for j, other in neighbors:
                    if particle == other or not other.alive:
                        continue

                    f = self.adaptive_engine.long_range_force(
                        particle.id, particle.position,
                        other.id, other.position,
                        force_scale=0.0015
                    )
                    adaptive_force += f

                        
                    if hasattr(particle, "calculate_consolidated_score"):
                        dist = distance_matrix[i][j]
                        score = particle.calculate_consolidation_score(other, dist)
                        self.adaptive_engine.update_interaction(particle.id, other.id, score)

            except Exception as e:
                self.log(f"issue during neighbor determination and interactions: {e}")
                continue


            particle.velocity = xp.array(particle.velocity) + adaptive_force
            particle.update()
            self.adaptive_engine.set_embedding(particle.id, particle.position)
            self.total_energy += particle.energy

        if len(self.particles) > 100:
            self.prune_low_value_particles()
            self.log(f"[Engine] Pruned {pruned} particles this frame.")

    def get_neighbors(self, particle, max_neighbors=10, radius=0.6):
        idx = self.particles.index(particle)
        positions = xp.stack([p.position for p in self.particles])
        distance_matrix = batch_hyper_distance_matrix(positions)
        dists = distance_matrix[idx]
        neighbors = [
            self.particles[i] for i in xp.argsort(dists)[1:max_neighbors+1]
            if dists[i] <= radius and self.particles[i].alive
        ]
        return neighbors

    def extract_state_vector(self):
        avg = [sum(p.position[i] for p in self.particles) / len(self.particles) for i in range(11)]
        return avg

    def inject_action(self, action):
        if isinstance(action, list):
            # Enhanced energy injection for vector-style actions
            target_particles = random.sample(self.particles, min(len(self.particles), 8))  # More particles
            for p in target_particles:
                for i in range(11):
                    p.velocity[i] += action[i] * 0.15  # Increased from 0.1
                p.activation = min(p.activation + 0.6, 1.0)  # Increased from 0.5
                p.energy = min(p.energy + 0.3, 1.0)  # Increased from 0.2

            # Also boost nearby particles for ripple effect
            for p in target_particles:
                neighbors = self.get_neighbors(p, max_neighbors=5, radius=0.4)
                for neighbor in neighbors:
                    neighbor.energy = min(neighbor.energy + 0.1, 1.0)
                    neighbor.activation = min(neighbor.activation + 0.05, 1.0)

            self.log(f"[ParticleEngine] Enhanced vector-style influence injection to {len(target_particles)} particles.")
            return

        #pulling type from metadata
        action_type = action.get("type", "sensory")

        #creating new particle
        p = create_particle(
            id=len(self.particles), 
            type=action_type, 
            metadata=action, 
            energy=0.9, 
            activation=0.2, 
            AE_policy=random.choice(["cooperative", "avoidant", "chaotic", "inquisitive", "dormant", "disruptive", "reflective", "emergent"]),
            engine=self  # Pass engine reference to ALL particles
        )



        # Spatial
        if isinstance(action, dict) and "position" in action:
            x, y, z = action["position"]
            p.position[0] = min(max(x, 0), 1)
            p.position[1] = min(max(y, 0), 1)
            p.position[2] = min(max(z, 0), 1)
        else:
            p.position[0] = 0.5 + random.uniform(-0.1, 0.1)
            p.position[1] = 0.5 + random.uniform(-0.1, 0.1)
            p.position[2] = 0.5 + random.uniform(-0.1, 0.1)

        def extract(key, default):
            return action.get(key, default) if isinstance(action, dict) else default

        p.position[3] = extract("time_of_creation", p.w)
        p.position[4] = extract("localized_time", 0)
        p.position[5] = extract("age", 0)
        p.position[6] = extract("emotional_rhythm", random.uniform(-0.5, 0.5))
        p.position[7] = extract("memory_phase", 0.6)
        p.position[8] = extract("valence", 0.7)
        p.position[9] = extract("identity", category_to_identity_code(p.type))
        p.position[10] = extract("intent", 1.0)

        p.velocity = [random.uniform(-0.005, 0.005) for _ in range(11)]

        p.position = xp.array(p.position, dtype=bfloat16)
        p.velocity = xp.array(p.velocity, dtype=bfloat16)


        self.particles.append(p)
        self.adaptive_engine.set_embedding(p.id, p.position)
        self.log(f"[ParticleEngine] Injected action particle {p.id} with positions: {p.position}")

    def log(self, message):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def prune_low_value_particles(self):
        if len(self.particles) <= 50:
            return

        scored_particles = [(p.vitality_score(), p) for p in self.particles if p.alive]
        for p in self.particles:
            if not p.alive:
                continue

            # Base scoring — combine energy and activation
            base_score = p.energy + p.activation

            # Valence boost (preserve emotionally charged memories)
            valence = p.metadata.get("valence", 0.5)
            if p.type == "memory":
                base_score *= (1 + valence * 1.5)

            # Age penalty (unless consolidated)
            age_penalty = 1.0
            if hasattr(p, "age"):
                age_penalty = 1 / (1 + p.age)

            # Consolidation flag
            if p.metadata.get("consolidated", False):
                base_score *= 2  # protect long-term memory

            # Score = vitality * (age factor)
            score = base_score * age_penalty

            # Store for potential pruning
            scored_particles.append((score, p))

        # Sort by lowest score
        scored_particles.sort(key=lambda x: x[0])
        to_prune = [p for _, p in scored_particles[:10]]

        for p in to_prune:
            p.alive = False
            self.log(f"[Autonomic] Pruned low-score particle: {p.id}")

    def _degrade_old_memory_particles(self):
        """Degrade energy/activation of older memory particles to make room for new ones"""
        memory_particles = [p for p in self.particles if p.type == "memory" and p.alive]
        
        if len(memory_particles) < 20:
            return True  # Plenty of room, no degradation needed
            
        # Sort memory particles by creation time (oldest first)
        memory_particles.sort(key=lambda p: p.metadata.get("created_at", 0))
        
        # Keep most recent 20 at full strength
        recent_memories = memory_particles[-20:]
        older_memories = memory_particles[:-20]
        
        if not older_memories:
            return True  # No older memories to degrade
            
        degraded_count = 0
        for i, particle in enumerate(older_memories):
            # Calculate degradation factor based on age rank (oldest = most degraded)
            age_rank = i / len(older_memories)  # 0 = oldest, 1 = newest of the old
            degradation_factor = 0.3 + (0.4 * age_rank)  # 0.3 to 0.7 range
            
            # Apply degradation
            old_energy = particle.energy
            old_activation = particle.activation
            
            particle.energy *= degradation_factor
            particle.activation *= degradation_factor
            
            # If degraded below threshold, mark for removal
            if particle.energy < 0.05 and particle.activation < 0.05:
                particle.alive = False
                degraded_count += 1
                self.log(f"[Engine] Degraded memory particle {particle.id} below threshold, marked for removal")
            else:
                degraded_count += 1
                self.log(f"[Engine] Degraded memory particle {particle.id}: energy {old_energy:.3f}→{particle.energy:.3f}, activation {old_activation:.3f}→{particle.activation:.3f}")
        
        # Clean up dead particles
        self.particles = [p for p in self.particles if p.alive]
        
        return degraded_count > 0

    def _cycle_least_important_particle(self):
        """Remove the least important particle using existing pruning logic"""
        if len(self.particles) <= 10:
            return None  # Don't remove if too few particles
            
        # Use existing scoring logic from prune_low_value_particles
        scored_particles = []
        for p in self.particles:
            if not p.alive:
                continue

            # Base scoring — combine energy and activation
            base_score = p.energy + p.activation

            # Valence boost (preserve emotionally charged memories)
            valence = p.metadata.get("valence", 0.5)
            if p.type == "memory":
                base_score *= (1 + valence * 1.5)

            # Age penalty (unless consolidated)
            age_penalty = 1.0
            if hasattr(p, "age"):
                age_penalty = 1 / (1 + p.age)

            # Consolidation flag
            if p.metadata.get("consolidated", False):
                base_score *= 2  # protect long-term memory

            # Score = vitality * (age factor)
            score = base_score * age_penalty
            scored_particles.append((score, p))

        if not scored_particles:
            return None
            
        # Remove the lowest scoring particle
        scored_particles.sort(key=lambda x: x[0])
        lowest_particle = scored_particles[0][1]
        lowest_particle.alive = False
        self.particles = [p for p in self.particles if p.alive]
        
        return lowest_particle