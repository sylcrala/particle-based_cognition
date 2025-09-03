from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static, Button
from textual.containers import Horizontal, Vertical

class AgentScreen(Screen):
    def __init__(self):
        super().__init__()
        self.title = "Agent Monitoring"

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Agent Cognitive Monitor", id="agent-title")
            
            with Horizontal():
                # Left panel - particle field and quantum states
                with Vertical(classes="left-panel"):
                    yield Static("Particle Field", classes="panel-title")
                    from tui.widgets.agent.particle_field import ParticleFieldWidget
                    yield ParticleFieldWidget()
                    
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
                yield Button("← Back to Home", id="back-btn", variant="default")
                yield Button("Pause Updates", id="pause-btn", variant="warning")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back-btn":
            self.app.pop_screen()
        elif event.button.id == "pause-btn":
            # Future: toggle update timer
            pass