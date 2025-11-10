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
        self.logs = []
        self.logging_config = config.get_logging_config()
    
        self.log_to_file = self.logging_config.get("log_to_file", True)
        self.session_id = self.logging_config.get("session_id")
        self.base_dir = Path(self.logging_config.get("base_dir"))
        self.logs_dir = Path(self.logging_config.get("session_log_dir"))

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        if len(self.logs) == 0:
            self.log("Logger initialized", "INFO", "SystemLogger.__init__", "SystemLogger")
        

    def log(self, message, level = "INFO", context = None, source = None, user_id = None):
        print(f"[{level}] {message}")  
        log_entry = {
            "id": len(self.logs) + 1,
            "message": message,
            "level": level,
            "context": context,
            "source": source,
            "user_id": user_id,     # unused for now, might be used in later update (or removed entirely)
            "timestamp": dt.now().timestamp()
        }
        self.logs.append(log_entry)

        if self.log_to_file:
            self.save_session_logs(log_entry)
            self.save_level_logs()

    def get_logs(self):
        return self.logs
    
    def save_level_logs(self):
        """Saves logs categorized by their levels into separate files."""
        try:
            level_logs = {}
            for entry in self.logs:
                level = entry.get("level", "INFO")
                if level not in level_logs:
                    level_logs[level] = []
                level_logs[level].append(entry)

            for level, entries in level_logs.items():
                log_file = self.logs_dir / f"{level.lower()}_logs_{self.session_id}.json"
                with open(log_file, 'w') as f:
                    json.dump(entries, f, indent=2)
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save level logs: {str(e)}")
            return False
    
    def save_session_logs(self, entry=None):
        """Saves the current session logs to a file. If an entry is provided, appends it to the log file."""
        try:
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
        except Exception as e:
            print(f"[ERROR] Failed to save logs: {str(e)}")
            return None
        
api.register_api("logger", SystemLogger())