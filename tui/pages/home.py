from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static, Button, TabbedContent, TabPane, RichLog
from textual.containers import Vertical, Horizontal, Grid, ScrollableContainer
from apis.api_registry import api
from tui.widgets.user.todo import TodoList
from tui.widgets.system.log_widget import SystemLogWidget

class HomeScreen(Vertical):
    def __init__(self):
        super().__init__()
        
        self.title = "Quantum Cognitive System - Home"
        self.id = "home"
        self.logger = api.get_api("logger")
        

    def compose(self) -> ComposeResult:
        with TabbedContent(id="home"):
            with TabPane("Hub", id="hub-tab"):

                with Grid(id="hub-buttons", classes="hub-content-area"): # Hub content area - default landing area, 3-4 grid size, one "main" widget spanning 2 columns 3 rows in top lefthand corner

                    yield Static("panel1 - tbd", classes="hub-panel-big hub-panel-style")
                    yield Static("panel2 - weather", classes="hub-panel-small hub-panel-style")
                    yield Static("panel3 - quick actions", classes="hub-panel-small hub-panel-style")

                    with ScrollableContainer(id="messages-container", classes="hub-panel-medium-tall hub-panel-style"): # messages panel  - panel 4
                        yield Static("Messages Stream", classes="hub-panel-title")
                        yield RichLog(id="messages", highlight=True, markup=True)

                    with ScrollableContainer(id="tasks-container", classes="hub-panel-small hub-panel-style"): # to-do panel - panel 5
                        yield Static("Pending Tasks", classes="hub-panel-title")
                        yield RichLog(id="pending-tasks", highlight=True, markup=True)

                    yield Static("panel6 - system metrics/status", classes="hub-panel-small hub-panel-style")

            with TabPane("System Logs", id="system-logs-tab"):
                yield SystemLogWidget()

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

        try:
            # query todo list api for pending tasks, add any found to the pending tasks panel
            tasks_panel = self.query_one("#pending-tasks", RichLog)
            todo_api = api.get_api("todo_list")
            if todo_api:
                incomplete_tasks = todo_api.get_incomplete_tasks()
                if incomplete_tasks:
                    for task in incomplete_tasks:
                        tasks_panel.write(f"[yellow]• {task['task']}[/yellow]")
                else:
                    tasks_panel.write("[dim]No pending tasks. Keep it up! :)[/dim]")
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error loading pending tasks: {e}", "ERROR", "tui_home", "HomeScreen")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "agent-btn":
            self.app.push_screen("agent")
        elif event.button.id == "metrics-btn":
            # Future: push metrics screen
            pass
        elif event.button.id == "exit-btn":
            self.app.exit()