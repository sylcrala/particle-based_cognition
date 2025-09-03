"""
particle physics engine [refactor in progress]
"""
import math
import random
import numpy as np
from datetime import datetime as dt
from collections import deque

from apis.api_registry import api
from apis.agent.utils.distance import batch_hyper_distance_matrix
from apis.agent.cognition.particles.utils.particle_frame import category_to_identity_code

config = api.get_api("config")


MAX_PARTICLE_COUNT = 150

class ParticleEngine:
    """
    Lightweight wrapper that delegates to the enhanced ParticleField
    Maintains compatibility while centralizing physics in field.py
    """
    def __init__(self):
        self.logger = api.get_api("logger")
        
        config = api.get_api("config")
        agent_config = config.get_agent_config()
        self.name = agent_config.get("name")

        self.particle_count = 0
        self.total_energy = 0
        self.total_activation = 0

        self.temporal_anchor = [0.0] * 11

    @property
    def field_api(self):
        """Get the particle field API for delegation"""
        return api.get_api("particle_field")

    def calculate_interaction_energy_cost(self, particle, other, distance):
        """Delegate to field API"""
        if self.field_api:
            return self.field_api.calculate_interaction_energy_cost(particle, other, distance)
        return 0.001  # Fallback minimal cost

    def log(self, message, level=None, context=None, source=None):
        if source is None:
            source = "ParticleEngine"
        
        if context is None:
            context = "no context"
        
        if level is None:
            level = "INFO"

        api.call_api("logger", "log", (message, level, context, source))

    def register_particle(self, particle):
        """Delegate to particle field for proper registration"""
        if self.field_api:
            return self.field_api.register_particle(particle)
        else:
            self.log("Particle field not available for registration", level="ERROR", 
                    context="register_particle()")

    def get_all_particles(self):
        """Delegate to particle field"""
        if self.field_api:
            particles = self.field_api.get_all_particles()
            self.log(f"[Query] Returning {len(particles)} particles from field", 
                    context="get_all_particles()")
            return particles
        else:
            self.log("Particle field not available", level="ERROR", context="get_all_particles()")
            return []

    async def update_particles(self, mood):
        """Delegate to the enhanced field API"""
        if self.field_api:
            result = await self.field_api.update_particles(mood)
            if result:
                self.total_energy = result.get("total_energy", 0)
                self.particle_count = result.get("alive_particles", 0)
            return result
        else:
            self.log("Particle field not available for update", level="ERROR", context="update_particles()")
            return {"total_energy": 0, "alive_particles": 0}

    def extract_state_vector(self):
        """Delegate to field API"""
        if self.field_api:
            return self.field_api.extract_state_vector()
        return [0.0] * 11

    async def inject_action(self, action, source: str = None):
        """Delegate to field API for action injection"""
        if self.field_api:
            return await self.field_api.inject_action(action, source)
        else:
            self.log("Particle field not available for action injection", level="ERROR")
            return False

    async def spawn_particle(self, id, type, metadata, energy=0.1, activation=0.1, AE_policy=None, **kwargs):
        """Delegate to field API for particle spawning"""
        if self.field_api:
            return await self.field_api.spawn_particle(id, type, metadata, energy, activation, AE_policy, **kwargs)
        else:
            self.log("Particle field not available for spawning", level="ERROR")
            return None

# Register the API
api.register_api("particle_engine", ParticleEngine())

