"""
system-wide logging
"""

from datetime import datetime as dt

from apis.api_registry import api

class SystemLogger:
    def __init__(self):
        self.logs = []

    def log(self, message, level = "INFO", context = None, source = None, user_id = None):
        log_entry = {
            "message": message,
            "level": level,
            "context": context,
            "source": source,
            "user_id": user_id,
            "timestamp": dt.now().timestamp()
        }
        self.logs.append(log_entry)

    def get_logs(self):
        return self.logs
    
api.register_api("logger", SystemLogger())