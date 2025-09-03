from textual.widgets import Static
from apis.api_registry import api

class ParticleFieldWidget(Static):
    def __init__(self):
        super().__init__("Particle Field - Initializing...")
        self.field_api = api.get_api("particle_field")
    
    async def update_display(self):
        particles = await self.field_api.get_all_particles()
        # Update visual representation
        self.update(f"Particles: {len(particles) if particles else 0} active")
