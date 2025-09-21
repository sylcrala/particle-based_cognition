"""
central API 
"""
import asyncio
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



    # agent framework methods
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
                    memory.emergency_save()

            # Phase 2: Preserve particle field state
            if "_agent_field" in self.apis:
                field = self.get_api("_agent_field")
                if field and hasattr(field, 'save_field_state'):
                    print("Saving particle field state...")
                    field.save_field_state()

            # Phase 3: Save adaptive learning progress
            if "_agent_adaptive_engine" in self.apis:
                adaptive = self.get_api("_agent_adaptive_engine")
                if adaptive and hasattr(adaptive, 'save_learning_state'):
                    print("Preserving learning adaptations...")
                    adaptive.save_learning_state()
            
            # Phase 4: System state snapshot
            if "logger" in self.apis:
                logger = self.get_api("logger")
                if logger:
                    logger.log("System graceful shutdown completed", "INFO", "APIRegistry", "shutdown")

            # Phase 5: Final shutdown
            if "_agent_cognition_loop" in self.apis:
                cognition = self.get_api("_agent_cognition_loop")
                cognition.conscious_active = False

            print("Cognitive shutdown sequence completed successfully")
            
        except Exception as e:
            print(f"Error during graceful shutdown: {e}")
            # Emergency fallback - still try to save what we can
            try:
                if "_agent_memory" in self.apis:
                    memory = self.get_api("_agent_memory")
                    if memory and hasattr(memory, 'force_save'):
                        memory.force_save()
                        print("Emergency memory save completed")
            except:
                print("Critical: Could not perform emergency save")

    async def handle_startup_restoration(self):
        """
        Restore cognitive state from previous shutdown to maintain continuity
        """
        try:
            print("Initiating cognitive state restoration...")
            
            # Phase 1: Restore memory state (field state from database)
            if "memory_bank" in self.apis:
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
                    adaptive.restore_learning_state()
            
            print("Cognitive state restoration completed successfully")
            
        except Exception as e:
            print(f"Error during state restoration: {e}")
            print("Starting with fresh cognitive state...")

api = APIRegistry()

