from textual.app import ComposeResult
from textual.widgets import Static, Button, TabbedContent, TabPane, RichLog
from textual.containers import Horizontal, Vertical, Grid, ScrollableContainer, Container

from tui.widgets.system.log_widget import SystemLogWidget
from tui.widgets.agent.particle_field import FieldVisualizerWidget
from tui.widgets.agent.quantum_state import QuantumStateWidget
from tui.widgets.agent.energy_flow import EnergyFlowWidget

from apis.api_registry import api

class AgentScreen(Vertical):
    def __init__(self):
        super().__init__()
        self.title = "Agent Dashboard"
        self.id = "agent"

        self.logger = api.get_api("logger")

    def compose(self) -> ComposeResult:
        with Vertical(id="agent-content-area"):
            with TabbedContent(initial="agent-home", id="agent-tabs"):
                with TabPane("Overview", id="agent-home"):
                    with Grid(classes="agent-grid"):

                        with Container(classes="agent-panel-big agent-panel-style"): # visualizer / panel 1
                            yield FieldVisualizerWidget()

                        with Container(classes="agent-panel-small agent-panel-style"): # energy flow / panel 2
                            yield EnergyFlowWidget()

                        with ScrollableContainer(classes="agent-panel-medium-tall agent-panel-style"): # system logs / panel 3
                            yield SystemLogWidget()

                        yield Static("System Metrics - panel 4", classes="agent-panel-small agent-panel-style") # system metrics / panel 4

                        with Container(classes="agent-panel-small agent-panel-style"): # quick actions / panel 5

                            with Vertical():
                                yield Static("Quick Actions - In progress", classes="agent-panel-title")

                                with Horizontal(classes="agent-quick-actions"): # assign better styling and actions per btn 
                                    yield Button("Action 1", id="action-1-btn", classes="agent-quick-action-btn", variant="primary")
                                    yield Button("Action 2", id="action-2-btn", classes="agent-quick-action-btn", variant="success")
                                    yield Button("Action 3", id="action-3-btn", classes="agent-quick-action-btn", variant="error")

                        yield Static("Panel 6 - tbd", classes="agent-panel-small agent-panel-style") # tbd / panel 6
                    
                with TabPane("Communications", id="agent-comms"):
                    with Vertical():
                        yield Static("Agent Communications Panel - tbd", id="agent-communications")
                        # implement agent communications here - chat window, message history, etc
                        with Horizontal():

                            with ScrollableContainer(id = "agent-chat-window", classes="agent-chat-window"):
                                yield Static("Chat Window - tbd", classes="agent-chat-style")
                                yield RichLog(id="agent-chat-log", highlight=True, markup=True)

                            with Container(id="agent-comms-controls", classes="agent-comms-controls"):
                                yield Static("Controls - tbd", classes="agent-comms-controls-title")
                                # controls for sending messages, managing conversations, etc.
                                yield Button("View Chat History", id="view-history-btn", classes="agent-comms-btn", variant="default")
                                yield Button("Settings", id="comms-settings-btn", classes="agent-comms-btn", variant="default")


                with TabPane("Cognitive Framework Management", id="agent-cognitive"):
                    yield Static("Cognitive Framework Management - tbd", id="cognitive-management")
                    # implement system controls here - centralized for agent framework startup, shutdown, restart, debug, etc.
                    #with Container(id="cognitive-controls"):
                        # buttons and such go here for power ctrls
                    #with Container(id="cognitive-status"):
                        # status info here - current state, uptime, load, etc.
                    #with Container(id="cognitive-logs"):
                        # detailed logs here - a larger + more interactive version of the log widget in the agent dashboard (use same widget just expand functionality)
                    #with Container(id="cognitive-debugging"):
                        # debugging tools here - interactive console, variable inspection, step through processes, etc.
                        # trigger a separate screen or popup for this                
            
            #with Static(id="agent-status-footer"):
                # add dynamic agent status info here in future (maybe make this application wide?)


    async def on_mount(self):
        """Initialize the agent screen with data monitoring"""
        try:
            SystemLogWidget.update_logs(self)
            pass
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error initializing agent screen: {e}", "ERROR", "tui_agent", "AgentScreen")
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pause-btn":
            # Future: toggle update timer
            pass