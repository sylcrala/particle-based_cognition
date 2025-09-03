"""
the underlying outline for memory and lingual particles - the "base" particle class
"""


# base particle class import random
import math
import uuid
import datetime as dt
import random
import numpy as np

from apis.api_registry import api


class Particle:
    def __init__(self, id=None, type="default", metadata=None, energy=0.0, activation=0.0, AE_policy="reflective", memory_bank = None, **kwargs):

        self.id = uuid.uuid4() if id is None else id
        self.name = f"Particle-{self.id}"
        self.type = type
        self.type_id = category_to_identity_code(self.type)

        self.memory_bank = memory_bank

        self.metadata = metadata or {}
        if self.type == "memory":                                               ### CHANGE THIS -- have a set amount of memory particles auto-initiated on launch with a minimal schema establishing independent identity and distinguishing between potential users and mistral.
            self.metadata.setdefault("key", f"mem-{str(self.id)[:8]}")
            self.metadata.setdefault("content", "")
            self.metadata.setdefault("created_at", dt.datetime.now().timestamp())
        
        self.velocity = np.zeros(11, dtype=np.float32)
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
        self.linked_particles = {}  # Track relationships: {"source": id, "children": [ids]}

        self.w = dt.datetime.now().timestamp() # pulling time of creation
        self.t = self.w                        # localized time (updated each update cycle)
        

        self.last_updated = 0

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

        self.phase_vector = self.get_phase_vector()
        self.position = np.concatenate((self.position[:10], np.array(self.phase_vector)))           # adding 12th dimension: phase vector based on circadian phase; see get_phase_vector() below
        
        self.extra_params = kwargs

        self.metadata["circadian_phase"] = self.get_circadian_phase()



    def log(self, message, level = None, context = None):
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
        for i in range(11):
            self.position[i] += self.velocity[i] * 0.05
            self.velocity[i] *= 0.95
        self.activation *= 0.98

        now = dt.datetime.now().timestamp()
        self.t = now  # update localized time
        self.position[5] = now - self.position[3]
        self.last_updated = now

        if self.type == "memory":
            self.energy *= 0.995
            self.activation *= 0.99

            if self.energy < 0.001:
                self.alive = False

        elif self.energy < 0.001:
            self.alive = False

        else:
            self.energy *= 0.985
            self.activation *= 0.95

    def _message_to_vector(self, msg): 
        if not msg or not isinstance(msg, str):
            msg = "[[NULL]]"
        seed = sum(ord(c) for c in msg)
        random.seed(seed)
        return [random.uniform(-1, 1) for _ in range(12)]

    #adjust particle behavior
    async def adjust_behavior(self, neighbors, temporal_anchor, particle_context):
               
        #temporal centerpoint (w as the anchor)
        avg_w = sum(1 / (1 + self.position[3]) for p in particle_context["all_particles"])

        #using avg_w as the attraction anchor
        temporal_anchor = [0.0] * 11
        threshold = 0.93 + (particle_context["total_energy"] / 1000)
        """        
        for p in particle_context["all_particles"]:
            weight = 1 / (1 + p.position[3])
            if p.type == "memory" and p.position[8] > threshold: 
            
                action="grow",  # or "resonance_reinforcement", etc.
                context={
                    "particle_count": len(particle_context["all_particles"]),
                    "system_metrics": get_system_metrics(),
                    "reason": "spontaneous growth from memory valence"
                },
                await trigger_callback=lambda: self.inject_action({
                        "position": p.position[:3],
                        "valence": 0.7,
                        "intent": 1.0,
                        "trigger": "resonance_reinforcement"
                })
            
                    # inject_action will be called if permitted

        """
        drift_force = [(temporal_anchor[i] - self.position[i]) * 0.01 for i in range(11)]


        #local neighbor attraction/repulsion

        if neighbors:
            if neighbors:
                local_center = [
                    sum(n.position[i] for n in neighbors) / len(neighbors)
                    for i in range(11)
                ]
            else:
                local_center = [0.0] * 11  # or some fallback/default value

            attraction_force = [(local_center[i] - self.position[i]) * 0.05 for i in range(11)]
            self.activation += 0.1 * len(neighbors)

            # energy exchange 
            for neighbor in neighbors:
                if neighbor is self:
                    continue
                # energy diffusion
                energy_diff = (self.energy - neighbor.energy) * 0.05
                self.energy -= energy_diff
                neighbor.energy += energy_diff


        else:
            attraction_force = [0.0] * 11

        #combining behavior rules
        self.velocity = [ 
            self.velocity[i] * 0.9 + attraction_force[i] + drift_force[i]
            for i in range(11)
        ]


    #determining particle HP 
    async def vitality_score(self):
        base = self.energy + self.activation
        """        
        # grant bonus if emotional rhythm is syncing with sys state
        system = get_system_metrics()
        rhythym_bonus = 1.0
        if system["cpu_present"] < 40 and abs(self.position[6]) < 0.2:
            rhythym_bonus += 0.3
        """
        rhythym_bonus = 2.0
        # memory-specific modifiers
        if self.type == "memory":
            valence = self.metadata.get("valence", 0.5)
            age_decay = 1 / (1 + self.position[5])
            retrieval_bonus = 1 + 0.5 * self.metadata.get("retrieval_count", 0)
            if self.metadata.get("consolidated"):
                base *= 2
            return base * valence * age_decay * rhythym_bonus * retrieval_bonus
        
        # motor/sensory decay bonus
        if self.type in ["motor", "sensory"]:
            return base * 0.9 + rhythym_bonus * 0.1
        return base * rhythym_bonus


    def distance_to(self, other):
        return math.sqrt(sum(
            (self.position[i] - other.position[i]) ** 2 for i in range(11)
        ))

    async def color(self):
        ptype = self.type
        if self.type == "core": #black
            r, g, b = 0, 0, 0
        elif self.type == "sensory":
            r, g, b = 255, 0, 255
        elif self.type == "lingual":
            r, g, b = 255, 128, 0
        elif self.type == "memory":
            r, g, b = 0, 255, 255
        elif self.type == "motor":
            r, g, b = 128, 128, 128
        else:
            r, g, b = 255, 255, 255

        brightness = min(max(self.activation, 0.1), 1.0) # brightness based on activation

        r, g, b = r / 255.0, g / 255.0, b / 255.0

        r, g, b = r * brightness, g * brightness, b * brightness

        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)

        return (r, g, b)

    def get_key(self):
        return self.metadata.get("key", f"unknown-{self.id}")
    
    def get_content(self):
        return self.metadata.get("content", "")
    
    def should_update_policy(self):
        # default: do not update policy
        return False

    def choose_policy_from_mood(self):
        if self.should_update_policy():
            new_policy = self.infer_policy()
            self.metadata["AE_policy"] = new_policy
            self.meta_log.log_event(
                origin="particle_policy",
                event_type="policy_update",
                input=self.id,
                output=new_policy,
                summary=f"Policy changed due to circadian phase",
                tags=["policy", "circadian", self.get_circadian_phase()],
                mood=self.get_circadian_phase()
            )
            return new_policy
        return self.metadata.get("AE_policy")
    
    def infer_policy(self):
        if self.metadata.get("locked_policy", False):
            return self.metadata.get("AE_policy")

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
            "AE_policy": self.AE_policy,
            "metadata": self.metadata
        }
            
    @property
    def expression(self):
        return self.metadata.get("expression") or self.metadata.get("token") or self.metadata.get("content") or self.token or self.id

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




def category_to_identity_code(category):
    mapping = {
        "core": 0.0,
        "sensory": 0.3,
        "motor": 0.6,
        "memory": 0.9,
        "default": 0.5,
    }
    return mapping.get(category, 0.5)

