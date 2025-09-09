"""
application TUI interface
"""

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static, RichLog, Button, Header, Footer, TabbedContent, TabPane, ContentSwitcher
from textual.reactive import reactive
from apis.api_registry import api
import threading
import time

# Import tabs at module level
from tui.pages.home import HomeScreen
from tui.pages.agent import AgentScreen

class MainApp(App):
    CSS = """
    /* Main Layout */
    #app-content-area {
        border: solid white;
        height: 1fr;
        width: 1fr;
        padding: 1;
    }

    #content-switcher {
        height: 100%;
        width: 100%;
    }

    .righthand-padding {
        min-width: 1;
        height: 100%;
    }

    #home-page {
        height: 100%;
        width: 100%;
    }

    #agent-page {
        height: 100%;
        width: 100%;
    }

    #sidebar {
        width: auto;
        border: solid blue;
        padding: 1;
        background: #1e1e1e;
        text-align: center;
    }


    /* Home Page Specific */

    HomeScreen {
        height: 100%;
        width: 100%;
        border: solid pink;
        background: #121212;
    }

    /* Log and Message Areas */
    #system-log-display {
        border: solid pink;
        height: 1fr;
        background: #1e1e1e;
    }

    #messages {
        border: solid pink;
        height: 1fr;
    }


    /* Styling Classes */
    .log-entry {
        margin: 1;
    }

    .log-system {
        max-height: 1;
        text-align: center;
        color: cyan;
        padding: 1;
        margin-bottom: 1;
    }

    .log-error {
        color: red;
    }

    .log-success {
        color: green;
    }

    .log-status {
        color: yellow;
    }

    .message-system {
        height: 1;
        text-align: center;
        color: cyan;
        padding: 1;
        margin-bottom: 1;
    }


    /* Hub Tab CSS */

    .hub-system {
        height: 1;
        text-align: center;
        color: cyan;
        padding: 1;
        margin-bottom: 1;
    }

    .hub-content-area {
        height: 100%;
        width: 100%;
        border: solid pink;
        background: #1e1e1e;
    }

    .hub-grid {
        grid-size: 3 4;
        height: 1fr;
        width: 1fr;
        border: solid cyan;
    }

    .hub-panel-big {
        column-span: 2;
        row-span: 3;
        height: 100%;
        width: 100%;
        padding: 1;
    }

    .hub-panel-small {
        column-span: 1;
        row-span: 1;
        height: 100%;
        width: 100%;
        padding: 1;
    }

    .hub-panel-medium-wide {
        column-span: 2;
        row-span: 1;
        height: 100%;
        width: 100%;
        padding: 1;
    }

    .hub-panel-medium-tall {
        column-span: 1;
        row-span: 2;
        height: 100%;
        width: 100%;
        padding: 1;
    }

    .hub-panel-style {
        border: solid green;
        background: #2e2e2e;
    }

    .hub-panel-title {
        text-align: center;
        color: ansi_bright_cyan;
        margin-bottom: 1;
    }

    #pending-tasks {
        border: solid pink;
        height: 1fr;
    }


    /* Agent Tab CSS */
    .agent-system {
        height: 1;
        text-align: center;
        color: cyan;
        padding: 1;
        margin-bottom: 1;
    }

    .agent-content-area {
        height: 1fr;
        border: solid pink;
        background: #1e1e1e;
    }

    .agent-grid {
        grid-size: 3 4;
        height: 1fr;
        width: 1fr;
        border: solid cyan;
    }

    .agent-panel-big {
        column-span: 2;
        row-span: 3;
        height: 100%;
        width: 100%;
        padding: 1;
    }

    .agent-panel-small {
        column-span: 1;
        row-span: 1;
        height: 100%;
        width: 100%;
        padding: 1;
    }

    .agent-panel-medium-wide {
        column-span: 2;
        row-span: 1;
        height: 100%;
        width: 100%;
        padding: 1;
    }

    .agent-panel-medium-tall {
        column-span: 1;
        row-span: 2;
        height: 100%;
        width: 100%;
        padding: 1;
    }

    .agent-panel-style {
        border: solid green;
        background: #2e2e2e;
    }

    .agent-panel-title {
        text-align: center;
        color: ansi_bright_cyan;
        margin-bottom: 1;
    }

    .agent-quick-actions {
        height: 1fr;
        width: 1fr;
        padding: 1;
        content-align: center middle;
    }

    .agent-quick-action-btn {
        padding: 1;
        margin: 0 1;
        width: 1fr;
        height: 1fr;
    }

    .agent-chat-section {
        height: 1fr;
        width: 4fr;
        border: solid pink;
        background: #1e1e1e;
        padding: 1;
    }

    .agent-chat-input {
        height: 3;
        text-align: left;
    }

    .agent-comms-controls-title {
        height: 3;
        text-align: center;
        color: cyan;
        padding: 1;
    }
        
    .agent-comms-controls {
        height: 1fr;
        width: 1fr;
        border: solid cyan;
        padding: 1;
        background: #1e1e1e;
        content-align: center middle;
        text-align: center;
    }

    .agent-comms-btn {
        margin-bottom: 1;
        padding: 1;
        content-align: center middle;
    }

    /* TodoList Widget CSS */

    .todo-system {
        height: 1;
        text-align: center;
        color: cyan;
        padding: 1;
        margin-bottom: 1;
    }

    #tasks-content-area{
        height: 1fr;
        border: solid pink;
        background: #1e1e1e;
    }

    #task-viewer {
        height: 1fr; 
        width: 100%;
        border: solid white;
        padding: 1;
        margin-bottom: 1;
    }

    #task-viewer-buttons {
        height: 3;
        content-align: center middle;
        margin-bottom: 1;
    }

    .task-viewer-btn {
        margin: 0 1;
        min-width: 15;
    }

    #task-list {
        height: 1fr;
        border: solid magenta;
        padding: 1;
    }

    #task-management {
        height: auto;
        content-align: center middle;
        border: solid green;
        padding: 1;
    }

    .task-management-btn {
        margin-bottom: 1;
        width: 100%;
        max-width: 20;
    }
    """
        
    SCREENS = {
        "home": HomeScreen,
        "agent": AgentScreen,
    }
    
    # Declare is_running as a reactive property
    is_running = reactive(True)
    current_screen = reactive("home-page")
    
    def __init__(self):
        super().__init__()
        self.api_registry = api
        self.logger = api.get_api("logger")
        self.update_timer = None
        self.last_log_count = 0


    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            # Persistent sidebar navigation
            with Vertical(id="sidebar"):
                yield Static("Navigation", classes="log-system")
                yield Button("Home", id="nav-home", variant="primary")
                yield Button("Agent Monitor", id="nav-agent", variant="primary")
                #yield Button("System Logs", id="nav-logs", variant="default")
                yield Button("Exit", id="nav-exit", variant="error")
            
            # Content area where screens get mounted
            with Vertical(id="app-content-area"):
                with ContentSwitcher(initial="home-page", id="content-switcher"):
                    
                    with Vertical(id="home-page"):
                        yield HomeScreen()

                    with Vertical(id="agent-page"):
                        yield AgentScreen()

            #yield Static(classes="righthand-padding")  # Right-hand padding
        
        yield Footer()


    async def on_mount(self):
        # Start log monitoring
        if self.logger:
            self.set_interval(0.5, self.update_system_logs)
        
        # Initialize the messages tab with welcome message
        try:
            messages_display = self.query_one("#messages", RichLog)
            messages_display.write("[bold cyan]🧠 Quantum Cognitive System - Message Inbox[/bold cyan]")
            messages_display.write("[dim]Welcome! This space will be used for autonomous agent communications.[/dim]")
            messages_display.write("")
            messages_display.write("[green]✅ System console initialized[/green]")
            messages_display.write("[blue]📊 Monitoring cognitive system logs...[/blue]")
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error initializing messages: {e}", "ERROR", "tui_startup", "MainApp")
        
        if self.logger:
            self.logger.log("TUI interface initialized", "SYSTEM", "tui_startup", "MainApp")


    def update_system_logs(self):
        """Update system logs display from shared logger"""
        if not self.logger or not self.is_running:
            return
            
        try:
            logs = self.logger.get_logs()
            
            # Only show new logs since last update
            if len(logs) > self.last_log_count:
                # Try to find system logs in current screen
                try:
                    system_log_display = self.query_one("#system-logs", RichLog)
                except:
                    # If not found, skip this update
                    return
                
                for log_entry in logs[self.last_log_count:]:
                    level = log_entry.get("level", "INFO")
                    message = log_entry.get("message", "")
                    source = log_entry.get("source", "Unknown")
                    context = log_entry.get("context", "")
                    timestamp = log_entry.get("timestamp", time.time())
                    
                    # Format timestamp
                    time_str = time.strftime("%H:%M:%S", time.localtime(timestamp))
                    
                    # Format context info
                    context_info = f" [{context}]" if context else ""
                    source_info = f" ({source})" if source else ""
                    
                    # Color coding based on level
                    if level == "ERROR":
                        system_log_display.write(f"[red][{time_str}] ERROR{source_info}{context_info}: {message}[/red]")
                    elif level == "SUCCESS":
                        system_log_display.write(f"[green][{time_str}] SUCCESS{source_info}{context_info}: {message}[/green]")
                    elif level == "SYSTEM":
                        system_log_display.write(f"[cyan][{time_str}] SYSTEM{source_info}{context_info}: {message}[/cyan]")
                    elif level == "STATUS":
                        system_log_display.write(f"[yellow][{time_str}] STATUS{source_info}{context_info}: {message}[/yellow]")
                    elif level == "COGNITIVE":
                        system_log_display.write(f"[magenta][{time_str}] COGNITIVE{source_info}{context_info}: {message}[/magenta]")
                    elif level == "WARNING":
                        system_log_display.write(f"[orange][{time_str}] WARNING{source_info}{context_info}: {message}[/orange]")
                    else:
                        system_log_display.write(f"[white][{time_str}] INFO{source_info}{context_info}: {message}[/white]")
                
                self.last_log_count = len(logs)
                
        except Exception as e:
            # Avoid infinite loop if logging fails
            pass


    async def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "nav-home":
            await self.switch_content("home-page")

        elif event.button.id == "nav-agent":
            await self.switch_content("agent-page")
            
        elif event.button.id == "nav-exit":
            if self.logger:
                self.logger.log("TUI shutdown initiated by user", "SYSTEM", "tui_shutdown", "MainApp")
            
            try:
                api.handle_shutdown()
            except Exception as e:
                if self.logger:
                    self.logger.log(f"Cognitive shutdown error: {e}", "ERROR", "tui_shutdown", "MainApp")
            
            self.is_running = False
            self.exit()

    async def switch_content(self, content_id: str):
        """Switch the content area to show different content"""
        try:
            content_switcher = self.query_one("#content-switcher", ContentSwitcher)
            content_switcher.current = content_id
            self.current_screen = content_id
            
            if self.logger:
                self.logger.log(f"Switched to content: {content_id}", "SYSTEM", "tui_navigation", "MainApp")
                
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error switching content to {content_id}: {e}", "ERROR", "tui_navigation", "MainApp")
            self.notify(f"Error switching content: {e}")


    def on_unmount(self):
        try:
            api.handle_shutdown()
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error during shutting down Cognition Framework: {e}", "ERROR", "tui_shutdown", "MainApp")

        if self.logger:
            self.logger.log("TUI interface unmounted", "SYSTEM", "tui_shutdown", "MainApp")
        self.is_running = False


    # Future autonomous messaging methods
    def send_agent_message(self, message: str, message_type: str = "info", urgency: str = "normal"):
        """
        Method for agent to send autonomous messages to user inbox
        
        Args:
            message: The message content
            message_type: Type of message (info, insight, alert, creative, social)
            urgency: Urgency level (low, normal, high, critical)
        """
        if not self.is_running:
            return
            
        try:
            messages_display = self.query_one("#messages", RichLog)
            
            # Format timestamp
            import time
            time_str = time.strftime("%H:%M:%S", time.localtime())
            
            # Choose icon and color based on message type and urgency
            if urgency == "critical":
                icon = "🚨"
                color = "red"
            elif urgency == "high":
                icon = "⚠️"
                color = "yellow"
            elif message_type == "insight":
                icon = "💡"
                color = "bright_yellow"
            elif message_type == "creative":
                icon = "🎨"
                color = "magenta"
            elif message_type == "social":
                icon = "🤝"
                color = "green"
            elif message_type == "alert":
                icon = "📢"
                color = "orange"
            else:
                icon = "🧠"
                color = "cyan"
            
            formatted_message = f"[{color}]{icon} [{time_str}] {message}[/{color}]"
            messages_display.write(formatted_message)
            
            # Optionally trigger notification sound or visual indicator
            if urgency in ["high", "critical"]:
                self.notify(f"Urgent agent message: {message[:50]}...")
                
        except Exception as e:
            # Fallback to system log if messages display fails
            if self.logger:
                self.logger.log(f"Agent message (display failed): {message}", "COGNITIVE", "autonomous_message", "AgentInbox")

    def get_message_count(self) -> int:
        """Get number of messages in inbox (for future badge/counter)"""
        try:
            messages_display = self.query_one("#messages", RichLog)
            # Could implement message counting logic here
            return 0  # Placeholder
        except:
            return 0
"""
if __name__ == "__main__":
    app = MainApp()
    app.run()
"""