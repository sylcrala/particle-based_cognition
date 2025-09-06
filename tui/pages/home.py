from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static, Button, TabbedContent, TabPane, RichLog
from textual.containers import Vertical, Horizontal, Grid

from apis.api_registry import api

from tui.widgets.user.todo import TodoList

class HomeScreen(Vertical):
    CSS = """
    #home {
        height: 100%;
        width: 100%;
        padding: 1;
    }
    Grid {
        grid-size: 3 4;
        gap: 1;
        height: 100%;
        width: 100%;
    }
    .hub-system {
        height: 3;
        text-align: center;
        color: cyan;
        border: solid white;
        padding: 1;
        margin-bottom: 1;
    }
    .message-system {
        height: 3;
        text-align: center;
        color: cyan;
        border: solid white;
        padding: 1;
        margin-bottom: 1;
    }
    .log-system {
        height: 3;
        text-align: center;
        color: cyan;
        border: solid white;
        padding: 1;
        margin-bottom: 1;
    }
    """   


    def __init__(self):
        super().__init__()
        self.title = "Quantum Cognitive System - Home"
        self.id = "home"
        self.logger = api.get_api("logger")
        

    def compose(self) -> ComposeResult:
        with TabbedContent():
            with TabPane("Hub", id="hub-tab"):
                yield Static("Hub", classes="hub-system")
                #with Grid(id="hub-buttons"): # Hub content area - default landing area, 3-4 grid size, one "main" widget spanning 2 columns 3 rows in top lefthand corner
                    #

            with TabPane("Messages", id="messages-tab"):
                yield Static("Messages", classes="message-system")
                yield RichLog(id="messages", highlight=True, markup=True)
            with TabPane("System Logs", id="system-logs-tab"):
                yield Static("System Logs - Complete History", classes="log-system")
                yield RichLog(id="system-logs", highlight=True, markup=True)
            with TabPane("To-Do List", id="todo-tab"):
                yield TodoList()

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