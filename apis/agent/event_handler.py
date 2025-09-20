"""
Agent Event Handler - integrates with API registry and shared services
"""

import asyncio
import heapq
from time import time
from apis.api_registry import api

logger = api.get_api("logger")

class EventHandler:
    """
    Central event handling system 
    """
    
    def __init__(self, field = None, memory = None):
        self.event_queue = PriorityEventQueue()
        self.event_handlers = {}
        self.running = False
        self.event_loop_task = None

        self.field = field
        self.memory = memory
                
        # Register default event handlers
        self.register_default_handlers()
    
    def log(self, message, level="INFO", context=None):
        """Use shared logging system"""
        if context is None:
            context = "no_context"
        else:
            context = context
            
        if logger:
            logger.log(message, level, context, "EventHandler")
        else:
            print(f"[{level}] {message}")  # Fallback
    
    def register_default_handlers(self):
        """Register default event handlers for common events"""
        self.event_handlers.update({
            "particle_created": self.handle_particle_created,
            "user_input": self.handle_user_input,
            "system_idle": self.handle_system_idle,
            "shutdown": self.handle_shutdown,
            "reflection_triggered": self.handle_reflection,
            "cognitive_event": self.handle_cognitive_event
        })
    
    async def emit_event(self, event_type, data, source="unknown", priority=None):
        """Emit an event with optional priority"""
        event = {
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": time()
        }
        
        # Set priority based on event type
        if priority is None:
            priority = self.get_default_priority(event_type, source)
        
        result = await self.event_queue.put(event, priority=priority)
        self.log(f"Event emitted: {event_type} from {source}", "DEBUG")
        return result

    def get_default_priority(self, event_type, source):
        """Get default priority for different event types"""
        priority_map = {
            "user_input": 1,
            "shutdown": 0,
            "particle_created": 3,
            "system_idle": 9,
            "cognitive_event": 5,
            "reflection_triggered": 4
        }
        
        base_priority = priority_map.get(event_type, 5)
        
        # User events get higher priority
        if source == "user":
            base_priority = min(base_priority, 2)
            
        return base_priority
    
    async def handle_event(self, event):
        """Central event dispatcher"""
        event_type = event["type"]
        
        self.log(f"Handling event: {event_type} from {event['source']}")
        
        # Find and execute handler
        handler = self.event_handlers.get(event_type, self.handle_unknown_event)
        
        try:
            result = await handler(event)
            return result
        except Exception as e:
            self.log(f"Error handling event {event_type}: {e}", "ERROR")
            return None
    
    async def handle_particle_created(self, event):
        """Handle particle creation events"""
        particle_data = event["data"]
        try:
            self.field.spawn_particle(
                type=particle_data.get("type", "unknown"),
                metadata=particle_data.get("metadata", {}),
                energy=particle_data.get("energy", 0.5),
                activation=particle_data.get("activation", 0.5),
                source_particle_id=particle_data.get("source_particle_id"),
                emit_event=False  # Avoid recursive event emission
            )
            self.log(f"Particle created: {particle_data.get('particle_id', 'unknown')}")
            return True
        except Exception as e:
            self.log(f"Error creating particle: {e}", "ERROR")
            return False

    
    async def handle_user_input(self, event):
        """Handle user input events"""
        user_data = event["data"]
        
        # Check for shutdown commands
        if isinstance(user_data, str) and user_data.lower() in ("exit", "quit", "shutdown"):
            await self.emit_event("shutdown", {"reason": "user_request"}, source="user")
            return "Shutdown initiated"
        
        config = api.get_api("config")
        if config and config.get("user_name"):
            user_name = config.get("user_name")
        else:
            user_name = "Unknown User"

        # Route to particle field for processing
        field = self.field
        if field:
            try:
                input_for_agent = f"{user_name} said: <s>{user_data}</s>"
                result = await field.inject_action(input_for_agent, source="user_input")
                
                if result:
                    self.log(f"User input processed, response generated")
                    return result
                else:
                    self.log("No response generated for user input", "WARNING")
                    return "[System] No response available."
                    
            except Exception as e:
                self.log(f"Error processing user input: {e}", "ERROR")
                return "[System] Error processing input."
        else:
            self.log("No particle field available for user input", "ERROR")
            return "[System] Cognitive system unavailable."
    
    async def handle_system_idle(self, event):
        """Handle system idle events for maintenance"""
        self.log("System idle - performing maintenance", "DEBUG")
        
        # Could trigger particle pruning, memory consolidation, etc.
        memory = self.memory
        if memory and hasattr(memory, 'perform_maintenance'):
            await memory.perform_maintenance()

        return True
    
    async def handle_shutdown(self, event):
        """Handle shutdown events"""
        reason = event["data"].get("reason", "unknown")
        self.log(f"Shutdown event received: {reason}", "SYSTEM")
        
        # Trigger graceful shutdown through API registry
        api.handle_shutdown()
        
        # Stop event loop
        self.running = False
        if self.event_loop_task:
            self.event_loop_task.cancel()
        
        return True
    
    async def handle_reflection(self, event):
        """Handle reflection events"""
        reflection_data = event["data"]
        self.log(f"Reflection triggered: {reflection_data}")
        
        # Could route to specialized reflection system
        # TODO
        return True
    
    async def handle_cognitive_event(self, event):
        """Handle general cognitive events"""
        self.log(f"Cognitive event: {event['data']}")
        # TODO
        return True
    
    async def handle_unknown_event(self, event):
        """Handle unknown event types"""
        self.log(f"Unknown event type: {event['type']}", "WARNING")
        # TODO
        return None
    
    def register_handler(self, event_type, handler):
        """Register a custom event handler"""
        self.event_handlers[event_type] = handler
        self.log(f"Registered handler for event type: {event_type}")
    
    async def start_event_loop(self):
        """Start the main event processing loop"""
        self.running = True
        self.log("Event handler started", "SYSTEM")
        
        try:
            while self.running:
                event = await self.event_queue.get()
                await self.handle_event(event)
        except asyncio.CancelledError:
            self.log("Event loop cancelled - shutting down gracefully", "SYSTEM")
        except Exception as e:
            self.log(f"Event loop error: {e}", "ERROR")
        finally:
            self.running = False
    
    async def start_idle_scheduler(self):
        """Start the idle event scheduler"""
        try:
            while self.running:
                await asyncio.sleep(30)  # Emit idle event every 30 seconds
                if self.running:
                    await self.emit_event("system_idle", {}, source="scheduler")
        except asyncio.CancelledError:
            self.log("Idle scheduler cancelled", "DEBUG")
    
    async def initialize(self):
        """Initialize the event system with background tasks"""
        # Start event loop
        self.event_loop_task = asyncio.create_task(self.start_event_loop())
        
        # Start idle scheduler  
        self.idle_task = asyncio.create_task(self.start_idle_scheduler())
        
        self.log("Event handler initialized with background tasks", "SYSTEM")
    
    async def shutdown(self):
        """Graceful shutdown of event system"""
        self.running = False
        
        if self.event_loop_task:
            self.event_loop_task.cancel()
            try:
                await self.event_loop_task
            except asyncio.CancelledError:
                pass
        
        if hasattr(self, 'idle_task'):
            self.idle_task.cancel()
            try:
                await self.idle_task
            except asyncio.CancelledError:
                pass
        
        self.log("Event handler shutdown complete", "SYSTEM")

class PriorityEventQueue:
    """Priority queue for events using asyncio"""
    
    def __init__(self):
        self._queue = []
        self._event = asyncio.Event()
        self._counter = 0  # For stable sorting
    
    async def put(self, event, priority=5):
        """Add event to queue with priority (lower number = higher priority)"""
        heapq.heappush(self._queue, (priority, self._counter, event))
        self._counter += 1
        self._event.set()
    
    async def get(self):
        """Get next event from queue"""
        while not self._queue:
            self._event.clear()
            await self._event.wait()
        
        # Reset event for next wait if queue is now empty
        if len(self._queue) == 1:
            self._event.clear()
        
        _, _, event = heapq.heappop(self._queue)
        return event
    
    def empty(self):
        """Check if queue is empty"""
        return len(self._queue) == 0
    
    def qsize(self):
        """Get queue size"""
        return len(self._queue)
