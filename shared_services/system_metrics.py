"""
Particle-based Cognition Engine - utility function for system metrics
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
            disk_used = shutil.disk_usage("/").used
            disk_total = shutil.disk_usage("/").total
            
            return {
                "cpu_usage": cpu_usage,
                "memory_used": memory_info.used,
                "memory_total": memory_info.total,
                "memory_percent": memory_info.percent,
                "disk_used": disk_used,
                "disk_total": disk_total,
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