"""
System log widget for TUI - displays logs from shared logger
"""

from textual.widgets import Static, RichLog
from textual.containers import Vertical
from apis.api_registry import api
import time

class SystemLogWidget(Static):
    def __init__(self):
        super().__init__()
        self.logger = api.get_api("logger")
        self.last_log_count = 0
        
    def compose(self):
        with Vertical():
            yield RichLog(id="system-log-display", highlight=True, markup=True)
    
    def on_mount(self):
        if self.logger:
            self.set_interval(0.5, self.update_logs)
    
    def update_logs(self):
        """Update log display with new entries from system logger"""
        if not self.logger:
            return
            
        logs = self.logger.get_logs()
        
        # Only show new logs since last update
        if len(logs) > self.last_log_count:
            log_display = self.query_one("#system-log-display", RichLog)
            
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
                    log_display.write(f"[red][{time_str}] ERROR{source_info}{context_info}: {message}[/red]")
                elif level == "SUCCESS":
                    log_display.write(f"[green][{time_str}] SUCCESS{source_info}{context_info}: {message}[/green]")
                elif level == "SYSTEM":
                    log_display.write(f"[cyan][{time_str}] SYSTEM{source_info}{context_info}: {message}[/cyan]")
                elif level == "STATUS":
                    log_display.write(f"[yellow][{time_str}] STATUS{source_info}{context_info}: {message}[/yellow]")
                elif level == "COGNITIVE":
                    log_display.write(f"[magenta][{time_str}] COGNITIVE{source_info}{context_info}: {message}[/magenta]")
                elif level == "WARNING":
                    log_display.write(f"[orange][{time_str}] WARNING{source_info}{context_info}: {message}[/orange]")
                else:
                    log_display.write(f"[white][{time_str}] INFO{source_info}{context_info}: {message}[/white]")
            
            self.last_log_count = len(logs)
