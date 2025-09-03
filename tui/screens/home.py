from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static, Button, TabbedContent, TabPane, RichLog
from textual.containers import Vertical, Horizontal

from apis.api_registry import api

class HomeScreen(Screen):
    def __init__(self):
        super().__init__()
        self.title = "Quantum Cognitive System - Home"
        self.logger = api.get_api("logger")

    def compose(self) -> ComposeResult:
        with Vertical():
            with TabbedContent():
                with TabPane("Messages", id="messages-tab"):
                    yield Static("Messages", classes="message-system")
                    yield RichLog(id="messages", highlight=True, markup=True)
                with TabPane("System Logs", id="system-logs-tab"):
                    yield Static("System Logs - Complete History", classes="log-system")
                    yield RichLog(id="system-logs", highlight=True, markup=True)

    async def on_mount(self):
        """Initialize the home screen with welcome messages"""
        try:
            # Add welcome message to messages tab (future autonomous inbox)
            messages_display = self.query_one("#messages", RichLog)
            messages_display.write("[bold cyan]🧠 Quantum Cognitive System - Message Inbox[/bold cyan]")
            messages_display.write("[dim]Welcome! This space will be used for autonomous agent communications.[/dim]")
            messages_display.write("")
            messages_display.write("[green]✅ System console initialized[/green]")
            messages_display.write("[blue]📊 Monitoring cognitive system logs...[/blue]")
            
            if self.logger:
                self.logger.log("Home screen mounted and initialized", "SYSTEM", "tui_home", "HomeScreen")
                
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error initializing home screen: {e}", "ERROR", "tui_home", "HomeScreen")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "agent-btn":
            self.app.push_screen("agent")
        elif event.button.id == "metrics-btn":
            # Future: push metrics screen
            pass
        elif event.button.id == "exit-btn":
            self.app.exit()