"""
System-wide Event Handler - Master coordinator for all subsystem events
"""

import asyncio
from typing import Dict, Any, Callable
from apis.api_registry import api

logger = api.get_api("logger")

class SystemEventHandler:
    """
    Master event handler that coordinates all subsystem event handlers
    """
    
    def __init__(self):
        self.subsystem_handlers = {}
        self.global_handlers = {}
        self.running = False
        self.master_queue = asyncio.Queue()
        self.event_routes = self._setup_event_routes()
        
    def log(self, message, level="INFO", context="system_events"):
        """Use shared logging system"""
        if logger:
            logger.log(message, level, context, "SystemEventHandler")
        else:
            print(f"[{level}] {message}")
    
    def _setup_event_routes(self) -> Dict[str, str]:
        """Define which subsystem handles which event types"""
        return {
            # Agent cognitive events
            "particle_created": "agent",
            "cognitive_event": "agent", 
            "reflection_triggered": "agent",
            "user_input": "agent",
            
            # TUI interface events
            "tui_started": "tui",
            "tui_shutdown": "tui",
            "navigation_change": "tui",
            
            # Model handler events
            "model_loaded": "model",
            "model_inference": "model",
            "model_error": "model",
            
            # System-wide events (handled directly)
            "system_startup": "system",
            "system_shutdown": "system",
            "api_registered": "system",
            "api_error": "system"
        }
    
    def register_subsystem_handler(self, name: str, handler):
        """Register a subsystem event handler"""
        self.subsystem_handlers[name] = handler
        self.log(f"Registered subsystem handler: {name}", "SYSTEM")
    
    def register_global_handler(self, event_type: str, handler: Callable):
        """Register a handler for system-wide events"""
        self.global_handlers[event_type] = handler
        self.log(f"Registered global handler for: {event_type}", "SYSTEM")
    
    async def emit_event(self, event_type: str, data: Any, source: str = "system", priority: int = 5):
        """Emit an event into the system"""
        event = {
            "type": event_type,
            "data": data,
            "source": source,
            "priority": priority,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        await self.master_queue.put(event)
        self.log(f"System event emitted: {event_type} from {source}", "DEBUG")
    
    async def route_event(self, event: Dict[str, Any]):
        """Route event to appropriate subsystem handler"""
        event_type = event["type"]
        
        # Check if it's a system-wide event
        if event_type in self.global_handlers:
            try:
                result = await self.global_handlers[event_type](event)
                return result
            except Exception as e:
                self.log(f"Error in global handler for {event_type}: {e}", "ERROR")
                return None
        
        # Route to subsystem
        target_system = self.event_routes.get(event_type, "agent")  # Default to agent
        
        if target_system in self.subsystem_handlers:
            handler = self.subsystem_handlers[target_system]
            try:
                if hasattr(handler, 'emit_event'):
                    # Forward to subsystem's emit_event method
                    await handler.emit_event(event_type, event["data"], event["source"], event.get("priority", 5))
                elif hasattr(handler, 'handle_event'):
                    # Call subsystem's handle_event directly
                    result = await handler.handle_event(event)
                    return result
                else:
                    self.log(f"Subsystem {target_system} has no compatible event method", "WARNING")
            except Exception as e:
                self.log(f"Error routing event to {target_system}: {e}", "ERROR")
        else:
            self.log(f"No handler found for subsystem: {target_system}", "WARNING")
    
    async def handle_system_startup(self, event):
        """Handle system startup coordination"""
        self.log("Coordinating system startup", "SYSTEM")
        
        # Initialize all registered subsystems
        for name, handler in self.subsystem_handlers.items():
            if hasattr(handler, 'initialize'):
                try:
                    await handler.initialize()
                    self.log(f"Initialized subsystem: {name}", "SUCCESS")
                except Exception as e:
                    self.log(f"Failed to initialize {name}: {e}", "ERROR")
        
        return True
    
    async def handle_system_shutdown(self, event):
        """Handle coordinated system shutdown"""
        self.log("Coordinating system shutdown", "SYSTEM")
        
        # Shutdown all subsystems gracefully
        for name, handler in self.subsystem_handlers.items():
            if hasattr(handler, 'shutdown'):
                try:
                    await handler.shutdown()
                    self.log(f"Shutdown subsystem: {name}", "SUCCESS")
                except Exception as e:
                    self.log(f"Error shutting down {name}: {e}", "ERROR")
        
        # Stop the master event loop
        self.running = False
        return True
    
    async def start_master_loop(self):
        """Start the master event processing loop"""
        self.running = True
        self.log("System event handler started", "SYSTEM")
        
        # Register system handlers
        self.register_global_handler("system_startup", self.handle_system_startup)
        self.register_global_handler("system_shutdown", self.handle_system_shutdown)
        
        try:
            while self.running:
                event = await self.master_queue.get()
                await self.route_event(event)
        except asyncio.CancelledError:
            self.log("Master event loop cancelled", "SYSTEM")
        except Exception as e:
            self.log(f"Master event loop error: {e}", "ERROR")
        finally:
            self.running = False
            self.log("Master event loop stopped", "SYSTEM")
    
    async def initialize(self):
        """Initialize the system event handler"""
        # Register the agent event handler as a subsystem
        agent_handler = api.get_api("event_handler")
        if agent_handler:
            self.register_subsystem_handler("agent", agent_handler)
        
        # Start the master loop as a background task
        self.master_task = asyncio.create_task(self.start_master_loop())
        
        # Emit startup event
        await self.emit_event("system_startup", {"reason": "initialization"}, "system")
        
        self.log("System event handler initialized", "SYSTEM")
    
    async def shutdown(self):
        """Graceful shutdown"""
        await self.emit_event("system_shutdown", {"reason": "graceful_shutdown"}, "system")
        
        if hasattr(self, 'master_task'):
            self.master_task.cancel()
            try:
                await self.master_task
            except asyncio.CancelledError:
                pass
        
        self.log("System event handler shutdown complete", "SYSTEM")

# Create and register the system event handler
system_event_handler = SystemEventHandler()
api.register_api("system_event_handler", system_event_handler)
