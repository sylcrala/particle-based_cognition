from textual.widgets import Static
from apis.api_registry import api

class QuantumStateWidget(Static):
    def __init__(self):
        super().__init__("Quantum States - Monitoring...")
        self.field_api = api.get_api("_agent_field")
    
    async def monitor_collapses(self):
        uncertain_particles = await self.field_api.get_uncertain_particles()
        # Display quantum superposition states
        self.update(f"Quantum: {len(uncertain_particles) if uncertain_particles else 0} uncertain")