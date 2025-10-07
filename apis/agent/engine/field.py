"""
centralized particle field - uses particle and adaptive engine for physics  [refactor in progress]

all particle spawning and pruning operations should be routed through here and properly tagged (include source particle ID if available for linkage)
"""
import os
import math
import random
import uuid
import numpy as np
import datetime as dt
from time import time
import asyncio
from collections import deque
import threading
from apis.agent.cognition.particles.memory_particle import MemoryParticle
from apis.api_registry import api
from apis.agent.utils.distance import batch_hyper_distance_matrix
from apis.agent.cognition.particles.utils.particle_frame import category_to_identity_code, Particle
from apis.agent.cognition.particles.lingual_particle import LingualParticle
from apis.agent.cognition.particles.memory_particle import MemoryParticle
from apis.agent.cognition.particles.sensory_particle import SensoryParticle


MAX_PARTICLE_COUNT = 500

class ParticleField:
    """
    the spatial domain for particle storage and lifecycle
    """
    def __init__(self, memory_bank = None, adaptive_engine = None, event_handler = None, voice = None):
        self.memory_bank = memory_bank
        self.adaptive_engine = adaptive_engine
        self.event_handler = event_handler
        self.voice = voice

        self._update_active = True

        self.particles = []             # all particles in the field
        self.alive_particles = set()    # alive/dead filtering
        self.particle_index = {}        # spatial partitioning for neighbor queries
        self.type_index = {}            # O(1) filtering
        self.collapse_history = {}      # wavelength collapse history 
        self.certainty_threshold = 0.7
        self.creation_counter = 0       # Track particle creation order for energy cost calculation
        
        # Spatial indexing for O(log n) neighbor queries
        self.grid_size = 0.5  # Grid cell size for spatial partitioning
        self._grid_lock = threading.Lock()
        self.spatial_grid = {}  # grid_key -> [particle_ids]
        self.particle_grid_cache = {}  # particle_id -> grid_key

        self.recent_actions = deque(maxlen=50)
        
        # Agent identity for seeding
        config = api.get_api("config")
        agent_config = config.get_agent_config()
        self.name = agent_config.get("name")

        for p in self.particles:
            if self.adaptive_engine:
                self.adaptive_engine.set_embedding(p.id, p.position)
                self.adaptive_engine.set_policy(p.id, p.policy or "cooperative")
            else:
                self.log("Adaptive engine not available during initialization", "WARNING", "__init__")


    def log(self, message, level = None, context = None):
        source = "ParticleField"

        if context != None:
            context = context
        else:
            context = "no context"

        if level != None:
            level = level
        else:
            level = "INFO"

        api.call_api("logger", "log", (message, level, context, source))
    
    def calculate_interaction_energy_cost(self, particle, other, distance):
        """
        Calculate energy cost for particle interaction based on:
        - Distance (higher distance = higher cost)
        - Index distance (temporal/creation separation)
        - Particle type compatibility
        """
        # Base cost scales with distance
        base_cost = distance * 0.01
        
        # Index distance cost (particles created far apart in time cost more)
        index_distance = abs(getattr(particle, 'creation_index', 0) - getattr(other, 'creation_index', 0))
        index_cost = math.log(1 + index_distance) * 0.005
        
        # Type compatibility modifier
        same_type = getattr(particle, 'type', None) == getattr(other, 'type', None)
        type_modifier = 0.5 if same_type else 1.0
        
        total_cost = (base_cost + index_cost) * type_modifier
        return max(total_cost, 0.001)  # Minimum cost to prevent zero-cost interactions

    def extract_state_vector(self):
        """Extract average state vector from all particles"""
        if not self.particles:
            return [0.0] * 12
        avg = [sum(p.position[i] for p in self.particles) / len(self.particles) for i in range(12)]
        return avg
    

    async def spawn_particle(self, id = None, type = None, metadata = None, energy=0.1, activation=0.1, AE_policy=None, emit_event = True, source_particle_id=None, **kwargs):
        """
        Spawn a new particle with proper linkage tracking for cognitive mapping
        """

        try:
            # Check particle limit first
            if len(self.particles) >= MAX_PARTICLE_COUNT:
                self.log(f"Particle spawn triggered at limit - triggering particle pruning before continuing", level="DEBUG", context="spawn_particle()")
                self.prune_low_value_particles()
            
            # Import particle types dynamically to avoid circular imports
            if type == "memory":
                particle = MemoryParticle(
                    id=id, metadata=metadata, energy=energy, 
                    activation=activation, AE_policy=AE_policy, **kwargs
                )
            elif type == "lingual":
                particle = LingualParticle(
                    id=id, metadata=metadata, energy=energy,
                    activation=activation, AE_policy=AE_policy, **kwargs
                )
            elif type == "sensory":
                particle = SensoryParticle(
                    id=id, metadata=metadata, energy=energy,
                    activation=activation, AE_policy=AE_policy, **kwargs
                )
            else:
                particle = Particle(
                    id=id, type=type, metadata=metadata, energy=energy,
                    activation=activation, AE_policy=AE_policy, **kwargs
                )
            
            # Track particle linkage for cognitive mapping - ADD VALIDATION HERE
            if source_particle_id:
                # Validate that the source particle exists and is alive
                source_particle = self.get_particle_by_id(source_particle_id)
                if source_particle and source_particle.alive and source_particle.id in self.alive_particles:
                    # Valid linkage - proceed with linking
                    particle.linked_particles = {"source": source_particle_id}
                    
                    # Also update the source particle to know about this child
                    if not hasattr(source_particle, 'linked_particles'):
                        source_particle.linked_particles = {}
                    if 'children' not in source_particle.linked_particles:
                        source_particle.linked_particles['children'] = []
                    
                    # Prevent duplicate child entries
                    if particle.id not in source_particle.linked_particles['children']:
                        source_particle.linked_particles['children'].append(particle.id)
                    
                    # Add the linked_to attribute for visualization
                    particle.linked_to = source_particle_id
                    
                    self.log(f"Successfully linked particle {particle.id} to source {source_particle_id}", 
                            level="DEBUG", context="spawn_particle")
                
                if source_particle and source_particle.alive == False and source_particle.id not in self.alive_particles:
                    # Link to a dead particle - spawn as a new particle with ghost traces (keep old particle links for full historical interconnectedness (this way, old dead particles are able to be *respawned* and referenced))
                    particle.linked_particles = {"ghost": source_particle.id} # mark as ghost linkage

                    # Also update the source particle to know about this ghost child
                    if not hasattr(source_particle, 'linked_particles'):
                        source_particle.linked_particles = {}
                    if 'ghost_children' not in source_particle.linked_particles:
                        source_particle.linked_particles['ghost_children'] = []
                    
                    # Prevent duplicate ghost child entries
                    if particle.id not in source_particle.linked_particles['ghost_children']:
                        source_particle.linked_particles['ghost_children'].append(particle.id)

                    particle.linked_to = None  # No direct live linkage

                    self.log(f"Warning: Linked to dead particle {source_particle_id}, spawning as ghost trace",
                            level="WARNING", context="spawn_particle")

                else:
                    # Invalid source - spawn as orphan with warning
                    self.log(f"Warning: Cannot link to non-existent particle {source_particle_id}, spawning as orphan", 
                            level="WARNING", context="spawn_particle")
                    particle.linked_particles = {}
                    particle.linked_to = None
                    source_particle_id = None  # Clear for logging

            else:
                particle.linked_particles = {}
                particle.linked_to = None

            # Add to field
            self.particles.append(particle)
            self.alive_particles.add(particle.id)
        
            # Add creation index for energy cost calculations
            particle.creation_index = self.creation_counter
            self.creation_counter += 1
            
            # Update spatial and type indexes for efficient neighbor queries
            self._update_spatial_index(particle)
            
            # Register with adaptive engine
            if self.adaptive_engine:
                self.adaptive_engine.set_embedding(particle.id, particle.position)
                if AE_policy:
                    self.adaptive_engine.set_policy(particle.id, AE_policy)
            else:
                self.log("Adaptive engine not available for particle embedding", "WARNING", "spawn_particle")
                
            linkage_info = f" (linked to {source_particle_id})" if source_particle_id else " (orphan)"
            self.log(f"Spawned {type} particle {particle.id}{linkage_info}", "INFO", "spawn_particle")
            
            # Emit event for other systems using new event handler
            if emit_event:
                await self.event_handler.emit_event("particle_created", {
                    "particle_id": particle.id,
                    "type": type,
                    "source_particle_id": source_particle_id,
                    "metadata": metadata
                    })
            
            return particle
        except Exception as e:
            self.log(f"Error spawning particle: {e}", "ERROR", "spawn_particle")
            import traceback
            self.log(f"Spawn error traceback: {traceback.format_exc()}", "ERROR", "spawn_particle")
            return None

    async def seed_particles(self):
        if len(self.particles) == 0:
            self.log("[Seed] No particles detected at init — seeding with core particles.")
            for i in range(random.randint(5, 20)):
                await self.spawn_particle(
                    id=f"seed-{i}",
                    type="memory",
                    metadata={
                        "identity": {
                            "name": f"{self.name}",
                            "context": f"I am {self.name}, a synthetic cognitive entity."
                        },
                        "tags": ["core", "identity", "anchor"]
                    },
                    energy=0.9,
                    activation=0.7,
                    AE_policy="emergent"
                )
            self.log(f"Detected 0 particles, proceeding to spawn core particles for identity anchoring.")
            


    async def prune_low_value_particles(self):
        self.log(f"Beginning pruning at {time()} with {len(self.particles)} particles present.")


        alive_particles = [p for p in self.particles if p.alive]
        scored_particles = []
        for p in alive_particles:
            if hasattr(p, "vitality_score"):
                score = p.vitality_score()
            
            else:  
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

        prune_percentage = self._calculate_adaptive_pruning_rate()
        total_particles = len(scored_particles)

        target_prune_count = int(total_particles * prune_percentage)

        MIN_SURVIVORS = 100
        MAX_PRUNE_PER_CYCLE = int(total_particles * 0.4)

        safe_prune_count = min(
            target_prune_count,
            total_particles - MIN_SURVIVORS,
            MAX_PRUNE_PER_CYCLE
        )
        safe_prune_count = max(0, safe_prune_count)  # Ensure non-negative

        to_prune = [p for _, p in scored_particles[:safe_prune_count]]
        self.log(f"Pruning {len(to_prune)} particles out of {total_particles} (target was {target_prune_count})", level="INFO", context="prune_low_value_particles")

        for p in to_prune:
            p.alive = False
            self.alive_particles.discard(p.id)

            self._remove_particle_from_grid(p.id)

            self.adaptive_engine.embeddings.pop(p.id, None) # removing pruned particles from AE embeddings
            self.adaptive_engine.policies.pop(p.id, None)   # removing pruned particles' AE policies

            self.log(f"Pruned low-score particle: {p.id}")

    def _calculate_adaptive_pruning_rate(self) -> float:
        """Dynamically adjust pruning rate based on particle count"""
        try:
            # get sys metrics from sensory particles or direct API
            metrics = None

            sensory_particles = self.get_particles_by_type("sensory")
            if sensory_particles:
                latest_sensory = max(sensory_particles, key=lambda p: p.environmental_state.get("timestamp", 0))
                metrics = latest_sensory.environmental_state.get("current_state", {})

            if not metrics:
                system_metrics_api = api.get_api("system_metrics")
                if system_metrics_api:
                    metrics = system_metrics_api.get_current_metrics()

            if not metrics:
                self.log("No system metrics available for adaptive pruning rate calculation", "WARNING", "_calculate_adaptive_pruning_rate")
                return 0.05  # default pruning rate

            cpu_usage = metrics.get("cpu_usage", 0) / 100.0 # convert to 0-1 scale
            mem_percent = metrics.get("memory_usage", 0) / 100.0 # convert to 0-1 scale
            particle_count = len(self.particles)

            if cpu_usage > 0.6 and particle_count >= 1000:
                prune_rate = 0.30 # aggressive pruning, 30%
                reason = f"HIGH_LOAD (CPU: {cpu_usage*100:.1f}%, Particles: {particle_count})"

            elif cpu_usage > 0.6 or particle_count >= 1000:
                prune_rate = 0.20 # moderate pruning, 20%
                reason = f"MODERATE_LOAD (CPU: {cpu_usage*100:.1f}%, Particles: {particle_count})"

            elif cpu_usage > 0.4:
                prune_rate = 0.15 # light to moderate pruning, 15%
                reason = f"LIGHT_LOAD (CPU: {cpu_usage*100:.1f}%, Particles: {particle_count})"

            else:
                prune_rate = 0.05 # minimal pruning, 5%
                reason = f"OPTIMAL_LOAD (CPU: {cpu_usage*100:.1f}%, Particles: {particle_count})"

            # Boost pruning if memory usage is also high
            if mem_percent > 0.8:
                prune_rate = min(1.0, prune_rate * 1.5)
                reason += f" + HIGH_MEM_LOAD({mem_percent:.2f})"
            elif mem_percent > 0.5:
                prune_rate = min(1.0, prune_rate * 1.15)
                reason += f" + MEM_LOAD({mem_percent:.2f})"

            self.log(f"Adaptive pruning rate calculated: {prune_rate*100:.1f}% | Reason: {reason}", "INFO", "_calculate_adaptive_pruning_rate")
            return prune_rate

        except Exception as e:
            self.log(f"Error calculating adaptive pruning rate: {e}", "ERROR", "_calculate_adaptive_pruning_rate")
            return 0.05  # default pruning rate

    def get_particle_by_id(self, particle_id):
        """Find a particle by its ID"""
        for particle in self.particles:
            if particle.id == particle_id:
                return particle
        return None
    
    def get_all_particles(self):
        """Return all active particles"""
        try:
            particle_list = [p for p in self.particles]
            return particle_list
        except Exception as e:
            self.log(f"Field monitoring error: {e}", level="ERROR", context="get_all_particles")
            return []
        
    # Also add this method if missing:
    def get_particle_count(self):
        """Return count of particles"""
        return len(self.get_all_particles())
        
    def get_alive_particles(self):
        """Return all alive particles"""
        return [p for p in self.particles if p.id in self.alive_particles]

    def get_particles_by_type(self, particle_type):
        """Get particles filtered by type"""
        return [p for p in self.particles if p.type == particle_type and p.alive]
    
    async def trigger_contextual_collapse(self, trigger_particle, context_type, cascade_radius=0.5):
        """Field-level collapse orchestration with cascading effects"""
        try:
            # Find particles within influence radius
            related_particles = self.get_particles_in_radius(trigger_particle, cascade_radius)
            
            collapse_log = []
            
            for particle in related_particles:
                if hasattr(particle, 'observe') and hasattr(particle, 'superposition'):
                    # Calculate collapse probability based on context and distance
                    distance = trigger_particle.distance_to(particle)
                    base_probability = max(0.1, 1.0 - (distance / cascade_radius))
                    particle.activation += 0.01  # Slight activation boost

                    # Context-specific modifiers
                    context_modifier = {
                        'user_interaction': 0.8,
                        'memory_retrieval': 0.6,
                        'reflection': 0.3,
                        'background_processing': 0.1
                    }.get(context_type, 0.5)
                    
                    collapse_probability = base_probability * context_modifier
                    
                    if random.random() < collapse_probability:
                        collapsed_state = particle.observe(context=f"{context_type}_cascade")
                        particle.energy -= 0.025 # energy cost for collapse
                        collapse_log.append((particle.id, collapsed_state))
                        
                        # Create linkage between trigger and collapsed particle
                        await self.create_interaction_linkage(trigger_particle.id, particle.id, context_type)
            
            self.log(f"Contextual collapse triggered: {len(collapse_log)} particles affected", 
                    context="trigger_contextual_collapse")
            return collapse_log
            
        except Exception as e:
            self.log(f"Contextual collapse error: {e}", level="ERROR", context="trigger_contextual_collapse")
            return []

    def get_particles_in_radius(self, center_particle, radius):
        """Get all particles within a certain distance radius"""
        nearby_particles = []
        for particle in self.get_alive_particles():
            if particle.id != center_particle.id:
                distance = center_particle.distance_to(particle)
                if distance <= radius:
                    nearby_particles.append(particle)
        return nearby_particles

    async def create_interaction_linkage(self, particle_a_id, particle_b_id, interaction_type):
        """Create linkage between particles based on interaction"""
        particle_a = self.get_particle_by_id(particle_a_id)
        particle_b = self.get_particle_by_id(particle_b_id)
        
        if particle_a and particle_b:
            # Add interaction linkage (different from parent-child)
            if not hasattr(particle_a, 'interaction_links'):
                particle_a.interaction_links = {}
            if not hasattr(particle_b, 'interaction_links'):
                particle_b.interaction_links = {}
                
            particle_a.interaction_links[particle_b.id] = {
                'type': interaction_type,
                'strength': 1.0,
                'timestamp': time()
            }
            particle_b.interaction_links[particle_a.id] = {
                'type': interaction_type,
                'strength': 1.0,
                'timestamp': time()
            }

    def get_particle_genealogy(self, particle_id, depth=3):
        """Get the full lineage tree for a particle"""
        def trace_lineage(pid, current_depth, visited=None):
            if visited is None:
                visited = set()
            if current_depth <= 0 or pid in visited:
                return {}
            visited.add(pid)
            
            particle = self.get_particle_by_id(pid)
            if not particle:
                return {}
            
            lineage = {
                "id": pid,
                "type": particle.type,
                "metadata": particle.metadata,
                "linked_particles": getattr(particle, 'linked_particles', {}),
                "children": [],
                "source": None
            }
            
            # Trace children
            if hasattr(particle, 'linked_particles') and 'children' in particle.linked_particles:
                for child_id in particle.linked_particles['children']:
                    child_lineage = trace_lineage(child_id, current_depth - 1, visited.copy())
                    if child_lineage:
                        lineage["children"].append(child_lineage)
            
            # Trace source
            if hasattr(particle, 'linked_particles') and 'source' in particle.linked_particles:
                source_lineage = trace_lineage(particle.linked_particles['source'], current_depth - 1, visited.copy())
                if source_lineage:
                    lineage["source"] = source_lineage
            
            return lineage
        
        return trace_lineage(particle_id, depth)

    def get_particle_population_stats(self):
        """Get statistics about current particle population - useful for emergence research"""
        stats = {
            "total": len(self.particles),
            "alive": len([p for p in self.particles if p.alive]),
            "by_type": {},
            "avg_energy": 0,
            "avg_activation": 0
        }
        
        alive_particles = [p for p in self.particles if p.alive]
        if alive_particles:
            stats["avg_energy"] = sum(p.energy for p in alive_particles) / len(alive_particles)
            stats["avg_activation"] = sum(p.activation for p in alive_particles) / len(alive_particles)
            
            for p in alive_particles:
                stats["by_type"][p.type] = stats["by_type"].get(p.type, 0) + 1
                
        return stats
    
    def inject_action_sync(self, action, source=None, tags=None, context_id = None):
        """Synchronous wrapper for inject_action to be used in non-async contexts"""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If the loop is already running, create a new task
            return asyncio.create_task(self.inject_action(action, source, tags, context_id))
        else:
            return loop.run_until_complete(self.inject_action(action, source, tags, context_id))

    async def inject_action(self, action, source=None, tags=None, context_id = None):
        """
        Inject user action into the particle field by creating appropriate particles
        """

        if source is None:
            source = "unknown"

        self.log(f"Action injection: {action} from {source}")

        # Handle different action types
        if isinstance(action, str):
            
            particle = await self.spawn_particle(
                id=None,
                type="lingual",
                metadata={
                    "token": action,
                    "source": source,
                    "interaction_type": "user_input" if source == "user_input" else source,
                    "tags": tags or []
                },
                energy=0.9,
                activation=0.8,
                AE_policy="cooperative"
            )
            
            if particle:
                self.log(f"Created lingual particle {particle.id} for action: {action[:50]}...")

                # generate via meta voice
                gen_source = "user_input" if source == "user_input" else source
                response = await self.voice.generate(str(action), source=gen_source, context_particles=[particle], tags=tags or [], context_id=context_id if context_id else None)

            else:
                self.log("Failed to create particle for action", "ERROR")
                return response
        
        elif isinstance(action, dict):
            # Structured action - handle based on action type
            action_type = action.get("type", "unknown")

            particle = await self.spawn_particle(
                id=None,
                type="memory",  # Could be different based on action_type
                metadata={
                    "action_type": action_type,
                    "action_data": action,
                    "source": source
                },
                energy=0.7,
                activation=0.6
            )
            
            if particle:
                self.log(f"Created action particle {particle.id} for structured action")
                return f"Structured action processed: {action_type}"
            else:
                return None
        
        else:
            self.log(f"Unknown action type: {type(action)}", "WARNING")
            return None

    # In field.py - Add this method:
    async def continuous_particle_updates(self):
        """Dedicated particle update loop running independently"""
        self.log("Particle update loop started", "INFO", "continuous_particle_updates")
        
        while getattr(self, '_update_active', True):  # Use field-specific flag
            try:
                self.log("Particle update cycle initiated", "DEBUG", "continuous_particle_updates")

                await self.update_particles()

                result = f"Alive particles: {len(self.alive_particles)}, Total particles: {len(self.particles)}"
                self.log(f"Particle update result: {result}", "DEBUG", "continuous_particle_updates")

                await asyncio.sleep(1.0)  # 1-second intervals for particle updates
                
            except Exception as e:
                self.log(f"Particle update error: {e}", "ERROR", "continuous_particle_updates")
                import traceback
                self.log(f"Error traceback: {traceback.format_exc()}", "ERROR", "continuous_particle_updates")
                await asyncio.sleep(2.0)
        
        self.log("Particle update loop ended", "INFO", "continuous_particle_updates")

    # Add shutdown method:
    def stop_particle_updates(self):
        """Stop the continuous particle update loop"""
        self._update_active = False

    async def update_particles(self):
        """
        Main physics loop with energy-regulated interconnectivity
        All particles can interact but energy cost scales with distance and temporal separation
        """
        current_time = dt.datetime.now()
        self.total_energy = sum(p.energy for p in self.particles if p.id in self.alive_particles)

        # Log current state
        self.log(f"Updating particles at {current_time.isoformat()} | Total Energy: {self.total_energy:.2f} | Alive Particles: {len(self.alive_particles)}", "DEBUG", "update_particles")

        # Seed core identity particles if none exist
        await self.seed_particles()
        
        # Get only alive particles for processing
        alive_particles = [p for p in self.particles if p.id in self.alive_particles]
        
        if not alive_particles:
            return {"total_energy": 0, "alive_particles": 0}
        
        # Prepare position arrays for distance calculations
        positions = np.stack([p.position for p in alive_particles])
        distance_matrix = batch_hyper_distance_matrix(positions)

        # Ensure all particles have proper numpy arrays
        for p in alive_particles:
            if not isinstance(p.position, np.ndarray):
                p.position = np.array(p.position, dtype=np.float32)
            if not isinstance(p.velocity, np.ndarray):
                p.velocity = np.array(p.velocity, dtype=np.float32)

        # Main particle update loop with energy-regulated interactions
        for i, particle in enumerate(alive_particles):
            # Policy updates (periodically)
            if int(current_time.timestamp()) % 60 == 0:
                if hasattr(particle, "should_update_policy") and particle.should_update_policy():
                    if not particle.metadata.get("locked_policy", False):
                        new_policy = particle.choose_policy_from_mood() if hasattr(particle, "choose_policy_from_mood") else "cooperative"
                        particle.AE_policy = new_policy
                        if self.adaptive_engine:
                            self.adaptive_engine.set_policy(particle.id, strategy=new_policy)
                        particle.metadata.setdefault("mood_history", []).append({
                            "mood": new_policy,
                            "timestamp": current_time.timestamp()
                        })

            # Full interconnectivity: all particles can interact, but energy cost scales with distance
            accessible_particles = [p for p in alive_particles if p != particle]
            
            particle_context = {
                "total_energy": self.total_energy,
                "all_particles": alive_particles,
            }

            # Energy-regulated behavior adjustment with all particles
            if hasattr(particle, "adjust_behavior"):
                await particle.adjust_behavior(
                    accessible_particles, 
                    getattr(self, 'temporal_anchor', [0.0] * 12), 
                    particle_context
                )

            # Initialize force accumulator for this particle
            adaptive_force = np.zeros(12)

            for j, other in enumerate(alive_particles):

                if i == j or not other.alive:  # Skip self
                    continue
                
                # Calculate adaptive distance using the engine's full distance method
                if self.adaptive_engine and hasattr(self.adaptive_engine, 'distance'):
                    adaptive_distance = self.adaptive_engine.distance(
                        particle.id, particle.position, 
                        other.id, other.position
                    )
                else:
                    # Fallback to matrix distance if adaptive engine unavailable
                    adaptive_distance = distance_matrix[i][j] if j < len(distance_matrix[i]) else 1.0
                    
                energy_cost = self.calculate_interaction_energy_cost(particle, other, adaptive_distance)
                
                # Only proceed if particle has enough energy and activation
                if particle.energy >= energy_cost and particle.activation > 0.1:
                    # Deduct energy cost for this interaction
                    particle.energy -= energy_cost
                    
                    # Calculate force with energy-scaled intensity using adaptive distance
                    force_scale = 0.0015 * (particle.activation / max(adaptive_distance, 0.01))
                    if self.adaptive_engine and hasattr(self.adaptive_engine, 'long_range_force'):
                        f = self.adaptive_engine.long_range_force(
                            particle.id, particle.position,
                            other.id, other.position,
                            force_scale=force_scale
                        )
                        adaptive_force += f

                    # Process consolidation scoring with energy regulation
                    if hasattr(particle, "calculate_consolidation_score"):
                        score = particle.calculate_consolidation_score(other, adaptive_distance)
                        if self.adaptive_engine and hasattr(self.adaptive_engine, 'update_interaction'):
                            self.adaptive_engine.update_interaction(particle.id, other.id, score)
                        
                        # Additional energy cost for cognitive processing
                        cognitive_cost = energy_cost * 0.3
                        particle.energy -= cognitive_cost

            # Apply forces and update particle
            particle.velocity = np.array(particle.velocity) + adaptive_force
            
            # Energy regeneration: particles slowly recover energy over time
            base_regen = 0.00375  # Base regeneration rate
            activation_bonus = particle.activation * 0.0005  # Higher activation = faster regen
            particle.energy = min(1.0, particle.energy + base_regen + activation_bonus)
            
            # Activation decreases when energy is low (dependency link)
            if particle.energy < 0.3:
                particle.activation *= 0.975  # Faster decay at low energy
            elif particle.energy < 0.7:
                particle.activation *= 0.995  # Slower decay at moderate energy
            elif particle.energy < 0.9:
                particle.activation = min(1.0, particle.activation * 1.0005)  # Very slight boost 
            elif particle.energy > 0.9:
                particle.activation = min(1.0, particle.activation * 1.001)  # Slight boost at high energy

            reproduction_chance = random.random() < particle.energy * particle.activation * 0.00075 # Low chance based on energy and activation
            if reproduction_chance and len(self.particles) < MAX_PARTICLE_COUNT:
                # Spawn a child particle with slight variations
                child_id = f"{particle.id}-child-{int(time())}-{random.randint(0,1000)}"
                child_metadata = particle.metadata.copy()
                child_metadata["origin"] = f"spawned from {particle.id}"
                
                await self.spawn_particle(
                    id=child_id,
                    type=particle.type,
                    metadata=child_metadata,
                    energy=0.5 * particle.energy,
                    activation=0.5 * particle.activation,
                    AE_policy=particle.policy,
                    source_particle_id=particle.id
                )
                
                # Energy cost for reproduction
                particle.energy *= 0.7  # Lose some energy after spawning
                self.log(f"Particle {particle.id} spawned child {child_id}", "INFO", "update_particles")
            
            # Update particle
            if hasattr(particle, "update"):
                await particle.update()
            if self.adaptive_engine and hasattr(self.adaptive_engine, 'set_embedding'):
                self.adaptive_engine.set_embedding(particle.id, particle.position)

        #return {"total_energy": self.total_energy, "alive_particles": len(alive_particles)}

    def _get_grid_key(self, position):
        """Convert position to spatial grid key with error handling"""
        try:
            if not isinstance(position, (list, tuple, np.ndarray)):
                self.log(f"Invalid position type: {type(position)}", "ERROR", "_get_grid_key")
                return (0, 0, 0)  # Default grid key
                
            # Ensure we have at least 3 dimensions
            pos_array = np.array(position)
            if len(pos_array) < 3:
                pos_array = np.pad(pos_array, (0, 3 - len(pos_array)), 'constant')
            
            # Use first 3 dimensions for spatial indexing
            return tuple(int(pos // self.grid_size) for pos in pos_array[:3])
            
        except Exception as e:
            self.log(f"Grid key generation error: {e}", "ERROR", "_get_grid_key")
            return (0, 0, 0)  # Fallback
    
    def _update_spatial_index(self, particle):
        """Thread-safe spatial grid update"""
        with self._grid_lock:  # ← Add thread safety
            try:
                grid_key = self._get_grid_key(particle.position)
                
                # Clean up old grid position
                if particle.id in self.particle_grid_cache:
                    old_key = self.particle_grid_cache[particle.id]
                    if old_key in self.spatial_grid:
                        self.spatial_grid[old_key].discard(particle.id)
                        # Clean up empty grid cells
                        if not self.spatial_grid[old_key]:
                            del self.spatial_grid[old_key]
                
                # Add to new grid cell
                if grid_key not in self.spatial_grid:
                    self.spatial_grid[grid_key] = set()
                
                self.spatial_grid[grid_key].add(particle.id)
                self.particle_grid_cache[particle.id] = grid_key
                
            except Exception as e:
                self.log(f"Spatial index error: {e}", "ERROR", "_update_spatial_index")

    def get_spatial_neighbors(self, particle, radius=0.6):
        """Thread-safe efficient spatial neighbor search"""
        with self._grid_lock:
            try:
                grid_key = self._get_grid_key(particle.position)
                
                # Check surrounding grid cells
                neighbor_ids = set()
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            check_key = (grid_key[0] + dx, grid_key[1] + dy, grid_key[2] + dz)
                            if check_key in self.spatial_grid:
                                neighbor_ids.update(self.spatial_grid[check_key])
                
                # Filter by actual distance and radius  
                neighbors = []
                for pid in neighbor_ids:
                    if pid != particle.id and pid in self.alive_particles:  # ← Check alive first
                        other = self.get_particle_by_id(pid)
                        if other and other.alive:
                            try:
                                if self.adaptive_engine and hasattr(self.adaptive_engine, 'distance'):
                                    dist = self.adaptive_engine.distance(particle, other)
                                else:
                                    # Fallback distance calculation
                                    dist = np.linalg.norm(np.array(particle.position) - np.array(other.position))
                                
                                if dist <= radius:
                                    neighbors.append((dist, other))
                            except Exception as e:
                                self.log(f"Distance calculation error: {e}", "ERROR", "get_spatial_neighbors")
                
                # Sort by distance and return particles only
                neighbors.sort(key=lambda x: x[0])
                return [other for _, other in neighbors]
                
            except Exception as e:
                self.log(f"Spatial neighbor search error: {e}", "ERROR", "get_spatial_neighbors")
                return []
    
    def _remove_particle_from_grid(self, particle_id):
        """Remove dead particle from spatial indexing"""
        with self._grid_lock:
            try:
                if particle_id in self.particle_grid_cache:
                    grid_key = self.particle_grid_cache[particle_id]
                    
                    # Remove from grid
                    if grid_key in self.spatial_grid:
                        self.spatial_grid[grid_key].discard(particle_id)
                        if not self.spatial_grid[grid_key]:
                            del self.spatial_grid[grid_key]
                    
                    # Remove from cache
                    del self.particle_grid_cache[particle_id]
                    
            except Exception as e:
                self.log(f"Grid cleanup error: {e}", "ERROR", "_remove_particle_from_grid")


    def get_particle_by_id(self, particle_id):
        """Efficient O(1) particle lookup by ID"""
        for particle in self.particles:
            if particle.id == particle_id:
                return particle
        return None
    
    def save_field_state(self):
        """
        Save particle field state during graceful shutdown
        """
        try:
            import json
            
            # Prepare field state for persistence
            field_state = {
                "timestamp": dt.datetime.now().isoformat(),
                "total_particles": len(self.particles),
                "alive_particles": len(self.alive_particles),
                "creation_counter": self.creation_counter,
                "particles_summary": []
            }
            
            # Save essential particle data (not full objects due to complexity)
            for particle in self.particles:
                if particle.alive:
                    particle_summary = {
                        "id": particle.id,
                        "type": getattr(particle, 'type', 'unknown'),
                        "energy": getattr(particle, 'energy', 0),
                        "activation": getattr(particle, 'activation', 0),
                        "creation_index": getattr(particle, 'creation_index', 0),
                        "position": particle.position.tolist() if hasattr(particle, 'position') else None,
                        "quantum_state": getattr(particle, 'quantum_state', 'unknown')
                    }
                    field_state["particles_summary"].append(particle_summary)
            
            # Save to file
            state_file = "./data/agent/field_shutdown_state.json"
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            
            with open(state_file, 'w') as f:
                json.dump(field_state, f, indent=2)
            
            self.log(f"Particle field state saved: {field_state['total_particles']} particles", 
                    level="INFO", context="save_field_state")
            
        except Exception as e:
            self.log(f"Error saving field state: {e}", level="ERROR", context="save_field_state")
    
    def get_field_state_for_database(self):
        """
        Extract field state for database storage (optimized format)
        """
        try:
            field_state = {
                "timestamp": dt.datetime.now().isoformat(),
                "total_particles": len(self.particles),
                "alive_particles": len(self.alive_particles),
                "creation_counter": self.creation_counter,
                "certainty_threshold": self.certainty_threshold,
                "particles_summary": []
            }
            
            # Save essential particle data for reconstruction
            for particle in self.particles:
                if particle.alive:
                    particle_summary = {
                        "id": particle.id,
                        "type": getattr(particle, 'type', 'unknown'),
                        "energy": getattr(particle, 'energy', 0),
                        "activation": getattr(particle, 'activation', 0),
                        "creation_index": getattr(particle, 'creation_index', 0),
                        "position": particle.position.tolist() if hasattr(particle, 'position') else None,
                        "velocity": particle.velocity.tolist() if hasattr(particle, 'velocity') else None,
                        "quantum_state": getattr(particle, 'quantum_state', 'uncertain'),
                        "metadata": getattr(particle, 'metadata', {}),
                        "AE_policy": getattr(particle, 'AE_policy', None),
                        "linked_particles": getattr(particle, 'linked_particles', {})
                    }
                    field_state["particles_summary"].append(particle_summary)
            
            return field_state
            
        except Exception as e:
            self.log(f"Error extracting field state: {e}", level="ERROR", context="get_field_state_for_database")
            return None
    
    async def restore_from_state(self, field_state):
        """
        Restore particle field from saved state for cognitive continuity
        """
        try:
            self.log(f"Restoring field from state: {field_state.get('timestamp')}", level="INFO", context="restore_from_state")
            
            # Clear existing particles
            self.particles.clear()
            self.alive_particles.clear()
            
            # Restore field properties
            self.creation_counter = field_state.get('creation_counter', 0)
            self.certainty_threshold = field_state.get('certainty_threshold', 0.7)
            
            # Restore particles
            for particle_data in field_state.get('particles_summary', []):
                try:
                    # Spawn particle with restored state
                    restored_particle = await self.spawn_particle(
                        id=particle_data.get("id"),
                        type=particle_data.get("type", "unknown"),
                        metadata=particle_data.get('metadata', {}),
                        energy=particle_data.get('energy', 0.5),
                        activation=particle_data.get('activation', 0.5),
                        AE_policy=particle_data.get('policy'),
                        position=np.array(particle_data.get('position')) if particle_data.get('position') else None,
                        emit_event=False  # Don't emit events during restoration
                    )
                    
                    if restored_particle:
                        # Restore detailed state
                        if particle_data.get('position'):
                            restored_particle.position = np.array(particle_data['position'])
                        if particle_data.get('velocity'):
                            restored_particle.velocity = np.array(particle_data['velocity'])
                        
                        restored_particle.quantum_state = particle_data.get('quantum_state', 'uncertain')
                        restored_particle.creation_index = particle_data.get('creation_index', 0)
                        restored_particle.linked_particles = particle_data.get('linked_particles', {})
                        
                except Exception as particle_error:
                    self.log(f"Error restoring particle {particle_data.get('id')}: {particle_error}", 
                            level="WARNING", context="restore_from_state")
            
            restored_count = len([p for p in self.particles if p.alive])
            self.log(f"Field restoration completed: {restored_count} particles restored", 
                    level="INFO", context="restore_from_state")
            
        except Exception as e:
            self.log(f"Error during field restoration: {e}", level="ERROR", context="restore_from_state")

