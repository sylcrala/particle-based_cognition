"""
utility functions for retrieving system metrics
"""

import shutil
import datetime as dt
from apis.api_registry import api

# Try to import psutil, handle gracefully if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class SystemStats:
    def __init__(self):
        # Initialize metrics on startup
        self.last_update = dt.datetime.now()
        self.psutil_available = PSUTIL_AVAILABLE
        
    def get_current_metrics(self):
        """Get current system metrics"""
        timestamp = dt.datetime.now().timestamp()
        
        if self.psutil_available:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            
            return {
                "cpu_usage": cpu_usage,
                "memory_used": memory_info.used,
                "memory_total": memory_info.total,
                "memory_percent": memory_info.percent,
                "disk_used": 0,  # Basic fallback
                "disk_total": 0,
                "timestamp": timestamp,
                "status": "full_metrics"
            }
        else:
            # Fallback metrics when psutil not available
            return {
                "cpu_usage": 0.0,
                "memory_used": 0,
                "memory_total": 0,
                "memory_percent": 0.0,
                "disk_used": 0,
                "disk_total": 0,
                "timestamp": timestamp,
                "status": "limited_metrics_no_psutil"
            }
    
    async def get_system_metrics(self):
        """Async wrapper for system metrics"""
        return self.get_current_metrics()

api.register_api("system_metrics", SystemStats())