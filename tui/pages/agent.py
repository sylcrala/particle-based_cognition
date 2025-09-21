from textual.app import ComposeResult
from textual.widgets import Static, Button, TabbedContent, TabPane, RichLog, Input
from textual.containers import Horizontal, Vertical, Grid, ScrollableContainer, Container

from tui.widgets.system.log_widget import SystemLogWidget
from tui.widgets.agent.particle_field import FieldVisualizerWidget
from tui.widgets.agent.quantum_state import QuantumStateWidget
from tui.widgets.agent.energy_flow import EnergyFlowWidget

import time

from apis.api_registry import api

class AgentScreen(Vertical):
    def __init__(self):
        super().__init__()
        self.title = "Agent Dashboard"
        self.id = "agent"
        self.status_text = "Unknown"
        self.cycle_count = 0
        self.is_online = False
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
                            with Vertical(id="agent-chat-section", classes="agent-chat-section"):
                                yield Static("Chat Window - tbd", classes="agent-panel-title")
                                with ScrollableContainer(id = "agent-chat-window", classes="agent-chat-window"):
                                    yield RichLog(id="agent-chat-log", highlight=True, markup=True)
                                    with Horizontal(classes="agent-chat-input-area"):
                                        yield Input(id="agent-chat-input", placeholder="Type your message here...", classes="agent-chat-input")
                                        yield Button("Send", id="agent-chat-send", classes="agent-chat-send-btn", variant="primary")                             

                            with Container(id="agent-comms-controls", classes="agent-comms-controls"):
                                yield Static("Controls - tbd", classes="agent-panel-title")
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
            

            with Container(id="agent-status-footer"):
                # Add dynamic agent status info here
                with Horizontal():
                    yield Static(f"Agent Status - {self.status_text}", id="agent-status-online")
                    yield Static(f"Cycle Count: {self.cycle_count}", id="agent-status-cycle-count")

    async def on_mount(self):
        """Initialize the agent screen with data monitoring"""
        await self.refresh_agent_data()
        
        agent_refresh = self.refresh_agent_data

        self.set_interval(2.0, agent_refresh)  # Refresh every 2 seconds


    async def refresh_agent_data(self):
        try:
            log_widget = self.query_one(SystemLogWidget)
            await log_widget.update_logs()
        except Exception as e:
            self.logger.log(f"Error refreshing log widget: {e}", "ERROR", "tui_agent", "AgentScreen")

        try:
            # Fetch latest data from the agent
            self.agent = api.get_api("_agent_cognition_loop")
            self.agent_event = api.get_api("_agent_events")

            if self.agent:
                self.is_online = self.agent.conscious_active
                self.status_text = "Online" if self.is_online else "Offline"
                self.cycle_count = self.agent.cycle_count

                # Update status footer
                status_widget = self.query_one("#agent-status-online", Static)
                cycle_widget = self.query_one("#agent-status-cycle-count", Static)
                status_widget.update(f"Agent Status - {self.status_text}")
                cycle_widget.update(f"Cycle Count: {self.cycle_count}")
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error refreshing agent status: {e}", "ERROR", "tui_agent", "AgentScreen")

        try:
            field_widget = self.query_one(FieldVisualizerWidget)
            await field_widget.update_display()
        except Exception as e:
            self.logger.log(f"Error updating field widget: {e}", "ERROR", "tui_agent", "AgentScreen")

        try:
            chat_widget = self.query_one("#agent-chat-log", RichLog)
            if chat_widget:
                chat_log = self.agent.get_chat_history()
                tony_msgs = chat_log.get("Tony", [])
                misty_msgs = chat_log.get("Misty", [])

                chat_widget.write("[bold underline]Chat History:[/bold underline]")
                for t_msg, m_msg in zip(tony_msgs, misty_msgs):
                    chat_widget.write(f"[bold pink]Tony:[/bold pink] {t_msg}")
                    chat_widget.write(f"[bold blue]Misty:[/bold blue] {m_msg}")

                await chat_widget.scroll_end(animate=False)
        except Exception as e:
            self.logger.log(f"Error updating chat widget: {e}", "ERROR", "tui_agent", "AgentScreen")

    async def on_button_pressed(self, event: Button.Pressed) -> None:
                # Fetch latest data from the agent
        self.agent = api.get_api("_agent_cognition_loop")
        self.agent_event = api.get_api("_agent_events")

        if event.button.id == "agent-chat-send":
            input_widget = self.query_one("#agent-chat-input", Input)
            message = input_widget.value
            message_length = len(message.strip())

            if message_length == 0:
                return # skipping empty messages


            self.query_one("#agent-chat-log", RichLog).write(f"[bold pink]You:[/bold pink] {message}")
            input_widget.value = ""

            try:
                if message_length == 0:
                    return  # Ignore empty messages
                else:

                    response = await self.agent_event.emit_event("user_input", message, "interface_chat")
                    if response:
                        self.query_one("#agent-chat-log", RichLog).write(f"[bold blue]Misty:[/bold blue] {response}")
                    else:
                        self.query_one("#agent-chat-log", RichLog).write(f"[bold red]System:[/bold red] [Error: No response]")


            except Exception as e:
                if self.logger:
                    self.logger.log(f"Error generating agent response: {e}", "ERROR", "tui_agent", "AgentScreen")
                    self.query_one("#agent-chat-log", RichLog).write(f"[bold red]Error:[/bold red] Failed to get response from agent: {e}")


            pass

