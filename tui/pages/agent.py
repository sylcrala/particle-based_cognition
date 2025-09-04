from textual.app import ComposeResult
from textual.widgets import Static, Button
from textual.containers import Horizontal, Vertical

class AgentScreen(Vertical):
    def __init__(self):
        super().__init__()
        self.title = "Agent Monitoring"
        self.id = "agent"

    def compose(self) -> ComposeResult:
        yield Static("Agent Cognitive Monitor", id="agent-title")
        
        with Horizontal():
            # Left panel - particle field and quantum states
            with Vertical(classes="left-panel"):
                yield Static("Particle Field Visualizer", classes="panel-title")
                from tui.widgets.agent.particle_field import FieldVisualizerWidget
                try:
                    yield FieldVisualizerWidget()
                except Exception as e:
                    yield Static(f"Error loading Particle Field Visualizer: {e}", classes="error")

                yield Static("Quantum States", classes="panel-title")
                from tui.widgets.agent.quantum_state import QuantumStateWidget
                yield QuantumStateWidget()
            
            # Right panel - energy flow and controls
            with Vertical(classes="right-panel"):
                yield Static("Energy Flow", classes="panel-title")
                from tui.widgets.agent.energy_flow import EnergyFlowWidget
                yield EnergyFlowWidget()
        
        # Bottom controls
        with Horizontal(classes="controls"):
            yield Button("Pause Updates", id="pause-btn", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pause-btn":
            # Future: toggle update timer
            pass