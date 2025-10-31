"""
Particle-based Cognition Engine - system logger utility function
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

from datetime import datetime as dt
import json
from pathlib import Path

from apis.api_registry import api
config = api.get_api("config")

class SystemLogger:
    def __init__(self):
        self.session_id = str(dt.now().timestamp()).replace('.', '_')

        self.logs = []
        self.mode = config.agent_mode
        if self.mode == "llm-extension":
            self.base_dir = Path("./logs/llm_extension")
        elif self.mode == "cog-growth":
            self.base_dir = Path("./logs/cog_growth")
    
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.logs_dir = self.base_dir / f"session_{self.session_id}"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        if len(self.logs) == 0:
            self.log("Logger initialized", "INFO", "SystemLogger.__init__", "SystemLogger")
        

    def log(self, message, level = "INFO", context = None, source = None, user_id = None):
        print(f"[{level}] {message}")  # Also print to console for immediate feedback
        log_entry = {
            "message": message,
            "level": level,
            "context": context,
            "source": source,
            "user_id": user_id,
            "timestamp": dt.now().timestamp()
        }
        self.logs.append(log_entry)
        self.save_logs(log_entry)

    def get_logs(self):
        return self.logs
    
    def save_logs(self, entry=None):
        log_file = self.logs_dir / f"system_log_{self.session_id}.json"
        if not log_file.exists():
            log_file.touch()

        if entry is not None:
            with open(log_file, 'a') as f:
                f.write(json.dumps(entry) + "\n")
            return log_file
        else:    
            with open(log_file, 'w') as f:
                json.dump(self.logs, f, indent=2)
            return log_file
        
api.register_api("logger", SystemLogger())