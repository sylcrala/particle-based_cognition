from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static, Button, TabbedContent, TabPane, RichLog
from textual.containers import Vertical, Horizontal, Grid

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
                yield Static("Hub", classes="hub-system")
                #with Grid(id="hub-buttons"): # Hub content area - default landing area, 3-4 grid size, one "main" widget spanning 2 columns 3 rows in top lefthand corner
                    #

            with TabPane("Messages", id="messages-tab"):
                yield Static("Messages", classes="message-system")
                yield RichLog(id="messages", highlight=True, markup=True)
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
    # Fixed debugging - safer approach
        try:
            tabbed_content = self.query_one(TabbedContent)
            print(f"✅ TabbedContent widget found: {tabbed_content}")
            self.logger.log("TabbedContent widget found", "SYSTEM", "tui_home", "HomeScreen")

            # Check for TodoList widget more safely
            try:
                todo_widgets = self.query(TodoList)  # Use query() instead of query_one()
                print(f"📝 TodoList widgets found: {len(todo_widgets)}")
                self.logger.log(f"TodoList widgets found: {len(todo_widgets)}", "SYSTEM", "tui_home", "HomeScreen")
                
                if todo_widgets:
                    todo_widget = todo_widgets[0]
                    print(f"✅ First TodoList widget: {todo_widget}")
                    self.logger.log("TodoList widget accessible", "SUCCESS", "tui_home", "HomeScreen")
                else:
                    print("❌ No TodoList widgets found!")
                    self.logger.log("No TodoList widgets found", "ERROR", "tui_home", "HomeScreen")
                    
            except Exception as todo_error:
                print(f"❌ Error querying TodoList: {todo_error}")
                self.logger.log(f"Error querying TodoList: {todo_error}", "ERROR", "tui_home", "HomeScreen")
                
        except Exception as e:
            print(f"❌ Error querying TabbedContent: {e}")
            if self.logger:
                self.logger.log(f"Error querying TabbedContent: {e}", "ERROR", "tui_home", "HomeScreen")
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "agent-btn":
            self.app.push_screen("agent")
        elif event.button.id == "metrics-btn":
            # Future: push metrics screen
            pass
        elif event.button.id == "exit-btn":
            self.app.exit()