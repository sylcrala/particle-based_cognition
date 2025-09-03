from textual.widgets import Static
from apis.api_registry import api

class EnergyFlowWidget(Static):
    def __init__(self):
        super().__init__("Energy Flow Monitor - Loading...")
        self.engine_api = api.get_api("particle_engine")
    
    async def show_interactions(self):
        energy_data = await self.engine_api.get_energy_metrics()
        # Visualize energy transfers
        self.update("Energy Flow: Active")