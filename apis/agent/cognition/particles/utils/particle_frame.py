"""
the underlying outline for memory and lingual particles - the "base" particle class
"""


# base particle class import random
import math
import uuid
import datetime as dt
import random
import numpy as np
from apis.agent.utils.embedding import ParticleLikeEmbedding
from apis.api_registry import api


class Particle:
    def __init__(self, id=None, type="default", metadata=None, energy=0.0, activation=0.0, AE_policy="reflective",  **kwargs):

        self.id = uuid.uuid4() if id is None else id
        self.name = f"Particle-{self.id}"
        self.type = type
        self.type_id = category_to_identity_code(self.type)

        self.config = api.get_api("config")
        self.field = api.get_api("_agent_field")
        self.memory_bank = api.get_api("_agent_memory")
        self.lexicon_store = api.get_api("_agent_lexicon")
        self.adaptive_engine = api.get_api("_agent_adaptive_engine")
        self.meta_voice = api.get_api("_agent_meta_voice")

        self.metadata = metadata or {}
        self.metadata.setdefault("content", "")
        self.metadata.setdefault("created_at", dt.datetime.now().timestamp())
        self.metadata.setdefault("agent_identity", "Misty")
        self.metadata.setdefault("agent_personality", "Curious, empathetic, and passionate.")
        self.metadata.setdefault("agent_motivation", "To explore and understand myself and the world around me.")
        self.metadata.setdefault("circadian_phase", self.get_circadian_phase())


        self.velocity = np.zeros(12, dtype=np.float32)
        self.activation = activation or 0.0
        self.energy = energy or random.uniform(0.1, 1.0)
        self.policy = AE_policy or random.choice(["cooperative", "avoidant", "chaotic", "inquisitive", "dormant", "disruptive", "reflective", "emergent"])
        self.state = {}
        self.active = True
        self.alive = True
        
        # Quantum-inspired superposition states
        self.superposition = {
            'certain': 0.5,      # Confidence in current state
            'uncertain': 0.5     # Ambiguity/probability
        }
        self.collapsed_state = None  # Cache for observed state
        self.observation_context = None  # Track what caused collapse
        
        # Particle linkage system for cognitive mapping
        self.linked_particles = {"source": None, "children": [], "ghost": []}  # Track relationships: {"source": id, "children": [ids]}
        self.source_particle_id:str = None

        self.w = dt.datetime.now().timestamp() # pulling time of creation
        self.t = self.w                        # localized time (updated each update cycle)
        self.last_updated = 0
        self.phase_vector = self.get_phase_vector()

        self.position = np.array([
            x := random.uniform(0, 1),   # x (length)
            y := random.uniform(0, 1),   # y (width)
            z := random.uniform(0, 1),   # z (height)
            w := self.w,                 # w (time of creation)
            t := 0.0,                    # t (localized time)
            a := 0.0,                    # a (age)
            f := random.uniform(-1, 1),  # f (emotional rhythm / frequency)
            m := random.uniform(0, 1),   # m (memory phase)
            v := random.uniform(-1, 1),  # v (valence)
            i := self.type_id,           # i (categorical identity code)
            n := random.uniform(0, 1),   # n (intent)
        ])

        self.position = np.concatenate((self.position[:10], np.array(self.phase_vector)))           # adding 12th dimension: phase vector based on circadian phase; see get_phase_vector() below
        self.extra_params = kwargs
        self.creation_index = self.position[3]
        self.vitality = 0.0

        if self.alive == False:
            self.energy = 0.0
            self.activation = 0.0
        



    def log(self, message, level = None, context = None, source = None):
        if source != None:
            source = "BaseParticle(frame)"
        else:
            source = source

        if context != None:
            context = context
        else:
            context = "no context"

        if level != None:
            level = level
        else:
            level = "INFO"

        api.call_api("logger", "log", (message, level, context, source))


    def get_phase_vector(self):
        import math
        phase = self.get_circadian_phase()
        angle = {
            "morning": 0,
            "afternoon": math.pi / 2,
            "evening": math.pi,
            "night": 3 * math.pi / 2,
            "twilight": 2 * math.pi
        }.get(phase, 0)

        return [round(math.cos(angle), 3), round(math.sin(angle), 3)]

    async def update(self):
        for i in range(12):
            self.position[i] += self.velocity[i] * 0.05
            self.velocity[i] *= 0.95
        self.activation *= 0.98

        now = dt.datetime.now().timestamp()
        self.t = now  # update localized time
        self.position[5] = now - self.position[3]
        self.last_updated = now

        if self.type == "memory":
            self.energy *= 0.925
            self.activation *= 0.95

            if self.energy < 0.001:
                self.alive = False

        elif self.type == "core":
            if self.persistence_lvl == "permanent":
                self.alive = True
                self.energy = 1.0
                self.activation *= 0.99

            elif self.persistence_lvl == "temporary":
                self.energy *= 0.8
                self.activation *= 0.95

        elif self.energy < 0.001:
            self.alive = False

        else:
            self.energy *= 0.925
            self.activation *= 0.95

        self.vitality = await self.vitality_score()
        self.metadata["circadian_phase"] = self.get_circadian_phase()

    def _message_to_vector(self, msg): 
        try:
            if not msg:
                msg = self.metadata.get("content") or "<empty>"

            embedding_provider = ParticleLikeEmbedding()
            embedding_result = embedding_provider.encode([str(msg)])
            
            if embedding_result and len(embedding_result) > 0:
                embedding = embedding_result[0]

                if isinstance(embedding, list):
                    embedding = np.array(embedding, dtype=np.float32)
                
                if len(embedding) != 12:
                    embedding = embedding[:12] if len(embedding) > 12 else np.pad(embedding, (0, 12 - len(embedding)), 'constant')
            else:
                # Fallback to deterministic generation
                seed = sum(ord(c) for c in msg)
                random.seed(seed)
                vector = [random.uniform(-1, 1) for _ in range(12)]
                return np.array(vector, dtype=np.float32)        
            
            return embedding
                  
        except Exception as e:
            # Log error and fallback to deterministic generation
            print(f"Embedding generation error in particle: {e}")
            seed = sum(ord(c) for c in msg)
            random.seed(seed)
            vector = [random.uniform(-1, 1) for _ in range(12)]
            return np.array(vector, dtype=np.float32)
        
    def _cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors"""
        try:
            if not vec1 or not vec2 or len(vec1) != len(vec2):
                return 0.0
            
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        
        except Exception as e:
            self.log(f"Error computing cosine similarity: {e}", "ERROR", "_cosine_similarity")
            return 0.0
        
    async def adjust_behavior(self, neighbors, temporal_anchor, particle_context):
        # Calculate experiential temporal anchor
        temporal_anchor = self.calculate_experiential_temporal_anchor(particle_context)
        
        # Experience-based threshold (older field = higher standards)
        all_particles = particle_context["all_particles"]
        avg_age = sum(p.position[5] for p in all_particles) / len(all_particles) if all_particles else 0
        threshold = 0.93 + (particle_context["total_energy"] / 1000) + (avg_age * 0.001)
        
        # Age-weighted drift force (older particles resist change more)
        my_age = self.position[5]  # a dimension
        age_resistance = 1.0 / (1.0 + my_age * 0.1)  # Older = more resistant
        
        drift_force = [
            (temporal_anchor[i] - self.position[i]) * 0.01 * age_resistance 
            for i in range(12)
        ]
        
        # Adaptive engine integration with experience weighting
        adaptive_force = np.zeros(12)
        if neighbors and self.adaptive_engine:
            for neighbor in neighbors:
                if neighbor is self:
                    continue
                    
                # Experience differential (learned behavior)
                age_diff = abs(my_age - neighbor.position[5])
                experience_similarity = 1.0 / (1.0 + age_diff)  # Similar age = stronger interaction
                
                if hasattr(self.adaptive_engine, 'long_range_force'):
                    force_vector = self.adaptive_engine.long_range_force(
                        self.id, self.position,
                        neighbor.id, neighbor.position,
                        force_scale=0.05 * self.activation * experience_similarity
                    )
                    adaptive_force += force_vector
                
                # Experience-based energy exchange (wiser particles teach newer ones)
                if my_age > neighbor.position[5]:  # I'm older
                    teaching_bonus = min(0.002, (my_age - neighbor.position[5]) * 0.0001)
                    self.energy += teaching_bonus  # Reward for teaching
                    neighbor.activation += 0.003   # Student gets activation boost
                elif my_age < neighbor.position[5]:  # I'm younger
                    learning_bonus = min(0.002, (neighbor.position[5] - my_age) * 0.0001)
                    self.activation += learning_bonus  # Reward for learning
                    neighbor.energy += 0.001       # Teacher gets small energy boost
        
        # Combine forces with age-based weighting
        self.velocity = np.array([
            self.velocity[i] * 0.9 + adaptive_force[i] + drift_force[i]
            for i in range(12)
        ])

    # Calculate dynamic temporal anchor based on particle field experience
    def calculate_experiential_temporal_anchor(self, particle_context):
        all_particles = particle_context["all_particles"]
        if not all_particles:
            return [0.0] * 12
        
        # Session start time (earliest particle creation)
        session_start = min(p.position[3] for p in all_particles)  # w dimension
        current_time = dt.datetime.now().timestamp()
        session_duration = current_time - session_start
        
        # Average particle age (collective experience)
        total_creation_time = sum(p.position[3] for p in all_particles)
        avg_creation_time = total_creation_time / len(all_particles)
        
        # Experience weighting factor
        experience_factor = session_duration / max(avg_creation_time - session_start, 1.0)
        
        # Create age-biased temporal anchor
        temporal_anchor = [0.0] * 12
        temporal_anchor[3] = avg_creation_time  # w: collective creation anchor
        temporal_anchor[4] = current_time       # t: current time reference  
        temporal_anchor[5] = session_duration   # a: session age anchor
        
        # Scale other dimensions by experience
        for i in [0, 1, 2, 6, 7, 8, 9, 10]:  # spatial + emotional dimensions
            temporal_anchor[i] = experience_factor * 0.1  # Gentle bias toward experienced center
        
        return temporal_anchor

    #determining particle HP 
    async def vitality_score(self):
        base = self.energy + self.activation
        
        rhythym_bonus = 2.0
        valence = self.metadata.get("valence", 0.5)
        age_decay = 1 / (1 + self.position[5])

        if self.type == "memory":
            retrieval_bonus = 1 + 0.5 * self.metadata.get("retrieval_count", 0)
            if self.metadata.get("consolidated"):
                base *= 2
            return base * valence * age_decay * rhythym_bonus * retrieval_bonus
        
        if self.type == "sensory":
            return base * valence * age_decay * rhythym_bonus * 0.8
            
        if self.type == "lingual":
            return base * valence * age_decay * rhythym_bonus * 0.9
        
        if self.type == "core":
            return base * valence * age_decay * rhythym_bonus * 0.95

    def distance_to(self, other):
        return math.sqrt(sum(
            (self.position[i] - other.position[i]) ** 2 for i in range(12)
        ))

    async def color(self):
        """Determine RGB color based on type, activation, valence, and frequency - DEPRECATED"""
        if self.type == "sensory":
            r, g, b = 255, 0, 255
        elif self.type == "lingual":
            r, g, b = 255, 128, 0
        elif self.type == "memory":
            r, g, b = 0, 255, 255
        else:
            r, g, b = 255, 255, 255

        brightness = min(max(self.activation, 0.1), 1.0) # brightness based on activation

        valence = self.position[8]  # v dimension
        frequency = abs(self.position[6])  # f dimension

        intensity = (abs(valence) + frequency) / 2.0
        saturation_boost = 1.0 + (intensity * 0.5)

        if valence > 0.5:
            r *= 1.1 # boost red for positive valence
        elif valence < -0.5:
            b *= 1.1 # boost blue for negative valence

        r = r / 255.0 * brightness * saturation_boost
        g = g / 255.0 * brightness * saturation_boost
        b = b / 255.0 * brightness * saturation_boost

        if r > 1.0 or g > 1.0 or b > 1.0:
            self.log(f"Color overflow: r={r}, g={g}, b={b}", "WARNING", "color")

        return (r, g, b)

    def get_key(self):
        return self.metadata.get("key", f"unknown-{self.id}")
    
    async def get_content(self):
        result = self.metadata.get("content", "")
        if not result and self.type == "lingual":
            result = self.metadata.get("token", "")
        return result
    
    def should_update_policy(self):
        # default: if not locked, 30% chance to update policy
        if not self.metadata.get("locked_policy", False) == False:
            return random.random() < 0.3
        else:
            return False

    def choose_policy_from_mood(self):
        if self.should_update_policy():
            new_policy = self.infer_policy()
            self.metadata["AE_policy"] = new_policy
            self.policy = new_policy
            self.log(f"Policy changed to {new_policy} due to circadian phase", "INFO", "particle_policy")
            return new_policy
        else:
            return self.metadata.get("AE_policy") or self.policy
    
    def infer_policy(self):
        if self.metadata.get("locked_policy", False):
            return self.policy

        circadian_phase = self.get_circadian_phase()
        circadian_bias = self.map_phase_to_policy(circadian_phase)

        # Optional: Weight with other internal signals (importance, type, source)
        if self.type == "memory" and self.metadata.get("importance", 0) > 0.8:
            return "reflective"
        
        return circadian_bias or "cooperative"
    
    def get_circadian_phase(self):
        hour = dt.datetime.now().hour
        if 5 <= hour < 11:
            return "morning"
        elif 11 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        elif 21 <= hour or hour < 2:
            return "night"
        return "twilight"

    def map_phase_to_policy(self, phase):
        return {
            "morning": "inquisitive",
            "afternoon": "cooperative",
            "evening": "reflective",
            "night": "dormant",
            "twilight": "emergent"
        }.get(phase, "neutral")

    def freeze(self):
        return self.active == False and self.velocity == 0.0

    def save_state(self):
        return {
            "id": self.id,
            "type": self.type,
            "position": self.position,
            "velocity": self.velocity,
            "activation": self.activation,
            "energy": self.energy,
            "AE_policy": self.policy,
            "metadata": self.metadata
        }
            
    @property
    def expression(self):
        expression = self.metadata.get("expression") or self.metadata.get("token") or self.metadata.get("content") or self.token or self.id
        return expression

    # Quantum-inspired superposition methods
    def observe(self, context=None):
        """Collapse superposition to a definite state based on context"""
        if self.collapsed_state is None or context != self.observation_context:
            self.collapsed_state = self._collapse_superposition(context)
            self.observation_context = context
        return self.collapsed_state
    
    def _collapse_superposition(self, context=None):
        """Determine collapsed state based on superposition probabilities and context"""
        # Simple probabilistic collapse
        if random.random() < self.superposition['certain']:
            return 'certain'
        else:
            return 'uncertain'
    
    def update_superposition(self, certainty_delta=0.0, interaction_strength=0.0):
        """Update superposition based on interactions or new information"""
        # Adjust certainty based on interactions
        self.superposition['certain'] += certainty_delta * 0.1
        self.superposition['uncertain'] = 1.0 - self.superposition['certain']
        
        # Clamp values between 0 and 1
        self.superposition['certain'] = max(0.0, min(1.0, self.superposition['certain']))
        self.superposition['uncertain'] = 1.0 - self.superposition['certain']
        
        # Clear cached state if significant change
        if abs(certainty_delta) > 0.1:
            self.collapsed_state = None
    
    def is_certain(self, threshold=0.7):
        """Check if particle is in a certain state"""
        return self.superposition['certain'] > threshold
    
    def entangle_with(self, other_particle, strength=0.1):
        """Create quantum-like entanglement between particles"""
        # Simple entanglement: align superposition states
        avg_certainty = (self.superposition['certain'] + other_particle.superposition['certain']) / 2
        
        self.superposition['certain'] += (avg_certainty - self.superposition['certain']) * strength
        other_particle.superposition['certain'] += (avg_certainty - other_particle.superposition['certain']) * strength
        
        # Normalize
        self.superposition['uncertain'] = 1.0 - self.superposition['certain']
        other_particle.superposition['uncertain'] = 1.0 - other_particle.superposition['certain']

    def age_to_size(self, age):
        """Map temporal dimension to visual size"""
        return min(max(age / 3600, 0.1), 2.0)  # 1 hour = normal size

    def valence_to_hue(self, valence):
        """Map emotional valence to color hue - temporarily disabled"""
        # -1 (negative) → blue, 0 → white, +1 (positive) → red
        base_hue = 180 + valence * 180

        # detect overflow
        if abs(valence) > 1.0:
            overflow_intensity = abs(valence) - 1.0
            self.log(f"Valence overflow: {valence} (intensity {overflow_intensity}) | Particle {self.id}", "WARNING", "valence_to_hue")

        return max(0, min(360, base_hue))
    
    def type_to_hue(self):
        """Assign base hue based on particle type"""
        hue_types = {
            "sensory": 30,       # orange for sensory particles
            "memory": 180,       # mid cyan for memory particles
            "lingual": 285,      # magenta for lingual particles
            "core": 120,         # green for core particles
            # extend as needed
        }
        ptype = self.type.lower() if hasattr(self, 'type') else "default"
        return hue_types.get(ptype, 0)  # default to red
    

    
    def should_shimmer(self, certainty, current_time):
        """Determine if particle should shimmer in current frame"""
        if certainty > 0.7:
            return False
        shimmer_rate = (1 - certainty) * 5  # More uncertain = faster shimmer
        return (current_time * shimmer_rate) % 1 < 0.5
    
    async def render(self):
        """Renders the particle in 3D space in accordance to each particles first three dimensional positions: x, y, z respectively. Other properties for visualization are derived from the particles other properties or dimensional positions."""
        current_time = dt.datetime.now().timestamp()

        pos_3d = self.position[:3].astype(np.float32)
        
        # calculating age
        age = current_time - self.position[3]

        # emotional dimensions
        freq = self.position[6]
        valence = self.position[8]
        vitality = self.vitality

        # pseudo-quantum state
        certainty = self.superposition['certain']

        # get particle entanglements
        entanglements = []
        children = self.linked_particles.get("children") 
        if children is not None:
            if isinstance(children, uuid.UUID):
                children = [children]
            if isinstance(children, list):
                children = [child for child in children if isinstance(child, uuid.UUID)]
            if children not in self.linked_particles["children"]:    
                self.linked_particles["children"].append(children)

        for linked_id in self.linked_particles.get("children", []):
            entanglements.append({
                'target_id': str(linked_id),
                "strength": self.calculate_connection_strength(linked_id),
                "type": 'parent_child'
            })



        return {              
            'id': str(self.id),
            'type': self.type,                                      
            'position': pos_3d,     
            'size': np.float32(self.age_to_size(age)),
            'pulse_rate': np.float32(abs(vitality) if vitality is not None else 1.0),                             
            'color_hue': np.float32(self.type_to_hue()),                        
            'color_saturation': np.float32(abs(freq)),                          
            'entanglements': entanglements,
            'glow': np.float32(valence),
            'glow_intensity': np.float32(abs(valence)),
            'glow_polarity': np.int8(1 if valence >= 0 else -1),
            'quantum_state': {
                'opacity': np.float32(certainty),
                'animation': self.should_shimmer(certainty, current_time) and 'shimmer' or 'steady',
                'ghost_trails': bool(certainty < 0.5),
                'collapse_indicator': bool(self.collapsed_state is not None)
            }
        }
    

    def render_particle(self, dim_mapping = None, normalize = True):
        """
        Render the particle for 3D visualization of any given set of the 12 dimensions - used for flexible mapping in the particle field viewer

        Args:
            dim_mapping: Dict specifying which dimensions to use for x,y,z axes
                        Example: {'x': 0, 'y': 1, 'z': 2} for default mapping
                        Can use any of the 12 dimensions
            normalize: Whether to normalize values for visualization
        
        Returns:
            dictionary with visualization properties
        """
        
        current_time = dt.datetime.now().timestamp()
        
        if dim_mapping is None:
            dim_mapping = {"x": 0, "y": 1, "z": 2}  # Default spatial mapping
        else:
            dim_mapping = dim_mapping

        # extracting mapped dims
        pos_3d = [
            self._get_normalized_dimension(dim_mapping.get("x", 0), normalize),
            self._get_normalized_dimension(dim_mapping.get("y", 1), normalize),
            self._get_normalized_dimension(dim_mapping.get("z", 2), normalize)
        ]
        
        # calculating age
        age = current_time - self.position[3]

        # emotional dimensions
        freq = self.position[6]
        valence = self.position[8]
        vitality = self.vitality

        #if abs(freq) > 1.0:
            #frequency_overflow = abs(freq) - 1.0
            #self.log(f"Frequency overflow: {freq} (intensity {frequency_overflow}) | Particle {self.id}", "WARNING", "render_particle")

        # pseudo-quantum state
        certainty = self.superposition['certain']

        # get particle entanglements
        entanglements = []
        children = self.linked_particles.get("children") 
        if children is not None:
            if isinstance(children, uuid.UUID):
                children = [children]
            if isinstance(children, list):
                children = [child for child in children if isinstance(child, uuid.UUID)]
            if children not in self.linked_particles["children"]:    
                self.linked_particles["children"].append(children)

        for linked_id in self.linked_particles.get("children", []):
            entanglements.append({
                'target_id': str(linked_id),
                "strength": self.calculate_connection_strength(linked_id),
                "type": 'parent_child'
            })

        # get dim names for labeling
        dimension_names = [
            "Length (x)", "Width (y)", "Height (z)", 
            "Creation Time (w)", "Current Time (t)", "Age (a)",
            "Frequency (f)", "Memory Phase (m)", "Valence (v)",
            "Identity (i)", "Intent (n)", "Circadian Phase"
        ]

        # include dim mapping info in result
        dimension_info = {
            "mapping": {
                "x": dimension_names[dim_mapping.get("x", 0)],
                "y": dimension_names[dim_mapping.get("y", 1)],
                "z": dimension_names[dim_mapping.get("z", 2)]
            },
            "raw_indices": dim_mapping
        }

        # TODO:
        # assign valence to another value (maybe shimmer directly, positive valence = dynamic shimmer based on value, negative valence = dynamic "void" effect based on value?)

        return {                # this needs deeper review - come back to it after test run for vitality-based pulse rate and type-based hue
            'id': str(self.id),
            'type': self.type,                                      # Include particle type for visualization
            'position': pos_3d,
            'dimensions': dimension_info,
            'size': self.age_to_size(age),
            'pulse_rate': abs(vitality) if vitality is not None else 1.0,                            # changed from abs(freq) to abs(vitality), ran into a NoneType issue, 
            'color_hue': self.type_to_hue(),                        # Hue based on particle type 
            'color_saturation': abs(freq),                          # Saturation based on frequency - i need to check this against the range of frequency values we're now seeing (in comparison to what frequency formerly was - now we need to confirm this mapping works for both negative and positive freq values)
            'entanglements': entanglements,
            'glow': valence,
            'glow_intensity': abs(valence),
            'glow_polarity': 1 if valence >= 0 else -1,
            'quantum_state': {
                'opacity': certainty,
                'animation': self.should_shimmer(certainty, current_time) and 'shimmer' or 'steady',
                'ghost_trails': certainty < 0.5,
                'collapse_indicator': self.collapsed_state is not None
            }
        }

    def _get_normalized_dimension(self, dim_index, normalize = True):
        """Retrieve and optionally normalize a specific dimension value - DEPRECATED"""
        value = self.position[dim_index] if dim_index < len(self.position) else 0
        current_time = dt.datetime.now().timestamp()

    
        if not normalize:
            return value
            
        # Apply normalization based on dimension type
        if dim_index in [0, 1, 2]:  # Spatial dimensions
            return value  # Already normalized in [0,1] range
        elif dim_index in [3, 4]:  # Temporal dimensions (timestamps)
            # Normalize to [-1,1] based on session duration
            if self.field and hasattr(self.field, 'creation_time'):
                session_duration = current_time - self.field.creation_time
                return 2 * ((value - self.field.creation_time) / max(session_duration, 1)) - 1
            return 0
        elif dim_index == 5:  # Age
            # Normalize age to [0,1] with logarithmic scale
            # 1 hour = 0.5, 1 day = 0.8, 1 week = 1.0
            return min(math.log10(1 + value/3600) / math.log10(168), 1.0)
        elif dim_index in [6, 8]:  # Already in [-1,1] range
            return value
        elif dim_index == 7:  # Memory phase [0,1]
            return value
        elif dim_index == 9:  # Identity code [0,1]
            return value
        elif dim_index == 10:  # Intent [0,1]
            return value
        else:  # Phase vector
            # Map to [-1,1] range
            return min(max(value, -1), 1)

    async def create_linked_particle(self, particle_type, content, relationship_type="triggered"):
        """Create a new particle linked to this particle"""

        metadata = {
            "content": content,
            "triggered_by": self.id,
            "relationship": relationship_type,
            "source": f"{self.type}_particle_creation"
        }
        
        # Inherit some quantum state from parent
        if hasattr(self, 'superposition'):
            # New particle starts with some uncertainty from parent
            energy = 0.5 + (self.superposition['certain'] * 0.3)
            activation = 0.4 + (self.superposition['certain'] * 0.2)
        else:
            energy = 0.5
            activation = 0.4
            
        particle = await self.field.spawn_particle(
            type=particle_type,
            metadata=metadata,
            energy=energy,
            activation=activation,
            source_particle_id=str(self.id),  # Creates genealogy linkage
            emit_event=False
        )
        self.linked_particles["children"].append(particle.id)
        return particle
    
    def calculate_connection_strength(self, linked_id):
        """Calculate a score representing the strength of connection to a linked particle"""
        if not self.field:
            return 0.0
        
        linked_particle = self.field.get_particle_by_id(linked_id)
        if not linked_particle:
            return 0.0
        
        try:
            # Base score from vitality
            base_score = (self.energy + self.activation) / 2
            
            # Distance factor (closer particles have stronger connections)
            distance = self.distance_to(linked_particle)
            distance_factor = 1.0 / (1.0 + distance)
            
            # Type compatibility bonus
            same_type = self.type == linked_particle.type
            type_bonus = 1.2 if same_type else 1.0
            
            # Metadata similarity (if both have similar tags)
            similarity_bonus = 1.0
            if hasattr(self, 'metadata') and hasattr(linked_particle, 'metadata'):
                self_tags = set(self.metadata.get('tags', []))
                linked_tags = set(linked_particle.metadata.get('tags', []))
                if self_tags and linked_tags:
                    overlap = len(self_tags.intersection(linked_tags))
                    total = len(self_tags.union(linked_tags))
                    similarity_bonus = 1.0 + (overlap / max(total, 1)) * 0.5
            
            final_score = base_score * distance_factor * type_bonus * similarity_bonus
            return min(final_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            return 0.1  # Default low score on error


    def calculate_consolidation_score(self, other_particle, distance):
        """Calculate how suitable this particle is for consolidation with another"""
        try:
            # Base score from particle vitality
            base_score = (self.energy + self.activation) / 2
            
            # Distance penalty (closer particles consolidate better)
            distance_factor = 1.0 / (1.0 + distance)
            
            # Type compatibility bonus
            same_type = getattr(self, 'type', None) == getattr(other_particle, 'type', None)
            type_bonus = 1.2 if same_type else 1.0
            
            # Metadata similarity (if both have similar tags)
            similarity_bonus = 1.0
            if hasattr(self, 'metadata') and hasattr(other_particle, 'metadata'):
                self_tags = set(self.metadata.get('tags', []))
                other_tags = set(other_particle.metadata.get('tags', []))
                if self_tags and other_tags:
                    overlap = len(self_tags.intersection(other_tags))
                    total = len(self_tags.union(other_tags))
                    similarity_bonus = 1.0 + (overlap / max(total, 1)) * 0.5
            
            # Age factor (older particles consolidate less readily)
            age_factor = 1.0
            if hasattr(self, 'age'):
                age_factor = 1.0 / (1.0 + self.age * 0.1)
            
            final_score = base_score * distance_factor * type_bonus * similarity_bonus * age_factor
            return min(final_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            return 0.1  # Default low score on error





def category_to_identity_code(category):
    mapping = {
        "core": 0.0,
        "sensory": 0.3,
        "motor": 0.6,
        "memory": 0.9,
        "default": 0.5,
    }
    return mapping.get(category, 0.5)

