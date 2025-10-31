"""
Particle-based Cognition Engine - system-wide API registry and management
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

import asyncio
import threading
from datetime import datetime, time
from queue import Queue, Empty
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class APIStatus(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"

@dataclass
class APIInfo:
    instance: Any
    status: APIStatus
    health_check: Optional[Callable] = None
    last_check: float = 0.0

class APIRegistry:
    def __init__(self):
        self._apis: Dict[str, APIInfo] = {}
        self._lock = asyncio.Lock()
        self.apis = {}  # legacy compat

    def log(self, message, level = None, context = None):
        if context != None:
            context = context
        else:
            context = "no context"

        if level != None:
            level = level
        else:
            level = "INFO"

        source = "ApiRegistry"

        try:
            api.call_api("logger", "log", (message, level, context, source))
        except Exception as e:
            print(f"[{source} - {level}] Printed log: {message} [{context}]")

    def register_api(self, name: str, instance: Any,
                      health_check: Optional[Callable] = None):
        """Register API with health monitoring"""
        self._apis[name] = APIInfo(
            instance=instance,
            status=APIStatus.INITIALIZING,
            health_check=health_check
        )
        
        # Also store in legacy apis dict for compatibility
        self.apis[name] = {"instance": instance, "user_only": False}
        
        # Set status to active if instance is valid
        if instance is not None:
            self._apis[name].status = APIStatus.ACTIVE

    
    def get_api(self, name: str, validate_health: bool = False) -> Optional[Any]:
        """Get API with optional health validation"""
        if name not in self._apis:
            self.log(f"PHANTOM API ACCESS: {name} - not found in registry", "WARNING", "get_api")
            return None
            
        api_info = self._apis[name]
        
        # Simple health validation - just check if instance exists
        if validate_health and api_info.instance is None:
            return None
                
        return api_info.instance
    
    def is_user_only(self, name):
        return self.apis.get(name, {}).get("user_only", False)
    
    def list_apis(self):
        return list(self.apis.keys())
    
    def call_api(self, name, method, *args, user_initiated=False, **kwargs):
        if self.is_user_only(name) and not user_initiated:
            raise PermissionError(f"API '{name}' requires user initiation.")
        
        api = self.get_api(name)

        if not api or not hasattr(api, method):
            raise AttributeError(f"API '{name}' does not have method '{method}'.")
        
        method_obj = getattr(api, method)
        
        if asyncio.iscoroutinefunction(method_obj):  # Fix: use iscoroutinefunction
            return method_obj(*args, **kwargs)
        else:
            return method_obj(*args, **kwargs)

    async def handle_manual_state_save(self):
        """
        Manual trigger for saving agent cognitive state
        """
        try:
            if "_agent_memory" in self.apis:
                memory = self.get_api("_agent_memory")
                if memory and hasattr(memory, 'emergency_save'):
                    print("Performing emergency memory save...")
                    await memory.emergency_save()

            if "_agent_field" in self.apis:
                field = self.get_api("_agent_field")
                if field and hasattr(field, 'save_field_state'):
                    print("Saving particle field state...")
                    await field.save_field_state()

            if "_agent_adaptive_engine" in self.apis:
                adaptive = self.get_api("_agent_adaptive_engine")
                if adaptive and hasattr(adaptive, 'save_learning_state'):
                    print("Preserving learning adaptations...")
                    await adaptive.save_learning_state()
            
            # Phase 4: System state snapshot
            if "logger" in self.apis:
                logger = self.get_api("logger")
                if logger:
                    logger.log("System graceful shutdown completed", "INFO", "APIRegistry", "shutdown")

        except Exception as e:
            self.log(f"Manual state save error: {str(e)}", "ERROR", "handle_manual_state_save")

    async def handle_shutdown(self):
        """
        Graceful cognitive shutdown sequence to prevent amnesia and preserve agent state
        """
        try:
            print("Initiating graceful cognitive shutdown...")
            
            # Phase 1: Save critical memory state
            if "_agent_memory" in self.apis:
                memory = self.get_api("_agent_memory")
                if memory and hasattr(memory, 'emergency_save'):
                    print("Performing emergency memory save...")
                    await memory.emergency_save()

            # Phase 2: Preserve particle field state
            if "_agent_field" in self.apis:
                field = self.get_api("_agent_field")
                if field and hasattr(field, 'save_field_state'):
                    print("Saving particle field state...")
                    await field.save_field_state()

            # Phase 3: Save adaptive learning progress
            if "_agent_adaptive_engine" in self.apis:
                adaptive = self.get_api("_agent_adaptive_engine")
                if adaptive and hasattr(adaptive, 'save_learning_state'):
                    print("Preserving learning adaptations...")
                    await adaptive.save_learning_state()
            
            # Phase 4: System state snapshot
            if "logger" in self.apis:
                logger = self.get_api("logger")
                if logger:
                    logger.log("System graceful shutdown completed", "INFO", "APIRegistry", "shutdown")

            # Phase 5: Final shutdown
            if "_agent_cognition_loop" in self.apis:
                cognition = self.get_api("_agent_cognition_loop")
                if hasattr(cognition, 'conscious_active'):
                    cognition.conscious_active = False

            print("Cognitive shutdown sequence completed successfully")
            
        except Exception as e:
            print(f"Error during graceful shutdown: {e}")
            try:
                if "_agent_memory" in self.apis:
                    memory = self.get_api("_agent_memory")
                    if memory and hasattr(memory, 'force_save'):
                        await memory.force_save()
                        print("Emergency memory save completed")
            except Exception as save_error:
                print(f"Critical: Could not perform emergency save: {save_error}")

    async def handle_startup_restoration(self):
        """
        Restore cognitive state from previous shutdown to maintain continuity
        """
        try:
            print("Initiating cognitive state restoration...")
            
            # Phase 1: Restore memory state (field state from database)
            if "_agent_memory" in self.apis:
                memory = self.get_api("_agent_memory")
                if memory and hasattr(memory, 'restore_field_state'):
                    print("Restoring field state from memory database...")
                    field_data = memory.restore_field_state()
                    
                    # Phase 2: Apply restored field state to particle field
                    if field_data and "_agent_field" in self.apis:
                        field = self.get_api("_agent_field")
                        if field and hasattr(field, 'restore_from_state'):
                            print("Reconstructing particle field...")
                            await field.restore_from_state(field_data)
            
            # Phase 3: Restore adaptive learning state
            if "_agent_adaptive_engine" in self.apis:
                adaptive = self.get_api("_agent_adaptive_engine")
                if adaptive and hasattr(adaptive, 'restore_learning_state'):
                    print("Restoring learning adaptations...")
                    await adaptive.restore_learning_state()

            print("Cognitive state restoration completed successfully")
            
        except Exception as e:
            print(f"Error during state restoration: {e}")
            print("Starting with fresh cognitive state...")

    def get_agent_status(self) -> dict:
        """Get comprehensive agent system status"""
        try:
            status = {
                'apis': {}
            }
            
            # Check all agent APIs
            agent_apis = [
                '_agent_field', '_agent_memory', '_agent_adaptive_engine', 
                '_agent_cognition_loop', 'logger'
            ]
            
            for api_name in agent_apis:
                api_instance = self.get_api(api_name)
                status['apis'][api_name] = {
                    'available': api_instance is not None,
                    'type': type(api_instance).__name__ if api_instance else None
                }
            
            return status
            
        except Exception as e:
            return {'error': str(e)}

api = APIRegistry()

