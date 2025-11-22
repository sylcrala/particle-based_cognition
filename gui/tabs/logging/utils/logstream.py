"""
Particle-based Cognition Engine - GUI log tab utilities - live log stream widget
Copyright (C) 2025 sylcrala

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version, subject to the additional terms 
specified in TERMS.md.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License and TERMS.md for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Additional terms apply per TERMS.md. See also ETHICS.md.
"""

from apis.api_registry import api
from PyQt6.QtWidgets import QTextEdit
from PyQt6.QtCore import QTimer
from collections import deque

class LogStream(QTextEdit):
    def __init__(self, parent=None, view_limit = 1000):
        super().__init__(parent)
        self.log_source = api.get_api("logger")
        self.displayed_log_ids = set()
        self.displayed_logs = deque(maxlen=view_limit)  
        self.logs = self.log_source.logs
        self.view_limit = view_limit
        
        self.timer = QTimer(self)   # timer to refresh logs
        self.timer.timeout.connect(self.refresh_logs)
        self.timer.start(3000)  # refresh every 3 seconds
        
    def refresh_logs(self):
        """Fetch logs from the logger and display them."""
        
        for log in self.logs[-self.view_limit:]: # get the latest logs up to view_limit
            if log['id'] in self.displayed_log_ids:
                continue  # skip already displayed logs

            if len(self.displayed_logs) >= self.view_limit:
                oldest = self.displayed_logs[0]
                self.displayed_log_ids.discard(oldest['id'])  # remove oldest log ID

            self.displayed_logs.append(log)
            self.displayed_log_ids.add(log['id'])  # add new log ID
            self.append(f"[{log['level']} | {log['source']} | {log['context']}] {log['message']}")