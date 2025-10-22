"""
central API 
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

        self._message_queue = Queue()
        self._response_queue = Queue()
        self._message_handler_active = False
        self._handler_thread = None

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

        if name == "_agent_field" and instance is not None:
            self._start_message_handler()
    
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

    def _start_message_handler(self):
        """Start the background message handling thread"""
        if not self._message_handler_active:
            self._message_handler_active = True
            self._handler_thread = threading.Thread(
                target=self._message_handler_worker,
                daemon=True,
                name="AgentMessageHandler"
            )
            self._handler_thread.start()
            print("Started AgentMessageHandler thread")

    def _message_handler_worker(self):
        while self._message_handler_active:
            try:
                try:
                    message_data = self._message_queue.get(timeout=1.0)
                except Empty:
                    continue

                response = self._process_agent_message(message_data)
                
                self._response_queue.put({
                    "id": message_data.get("id"),
                    "response": response,
                    "timestamp": datetime.now().isoformat()
                })    

                self._message_queue.task_done()

            except Exception as e:
                print(f"Error in message handler: {e}")
                if "logger" in self.apis:
                    logger = self.get_api("logger")
                    if logger and hasattr(logger, 'log'):
                        logger.log(f"Error in message handler: {e}", "ERROR", "APIRegistry", "_message_handler_worker")
                
                if "message_data" in locals():
                    self._response_queue.put({
                        "id": message_data.get("id", "unknown"),
                        "response": f"Error processing message: {e}",
                        "error": True,
                        "timestamp": datetime.now().isoformat()
                    })

    def _process_agent_message(self, message_data):
        """Process agent message using existing EventHandler"""
        try:
            message = message_data['message']
            source = message_data.get('source', 'api_registry')

            agent_core = self.get_api("agent")
            if not agent_core:
                return "Agent core not available"

            event_handler = getattr(agent_core, 'event_handler', None)
            if not event_handler:
                try:
                    event_handler = self.get_api("_agent_events")
                except Exception as e:
                    self.log(f"Event handler retrieval error: {e}", "ERROR", "_process_agent_message")
                    return f"Event handler not available | {e}"
            
            # Create event for user input
            from time import time
            event = {
                "type": "user_input",
                "data": message,
                "source": source,
                "timestamp": time()
            }
            
            try:
                if hasattr(event_handler, 'handle_event_sync'):
                    self.log("Using handle_event_sync for event processing", "DEBUG", "_process_agent_message")
                    result = event_handler.handle_event_sync(event)
                else:
                    self.log("handle_event_sync not found, using async with timeout", "WARNING", "_process_agent_message")
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            lambda: asyncio.run(event_handler.handle_event(event))
                        )
                        result = future.result(timeout=120.0)
                
                if result:
                    # Handle different response formats
                    if hasattr(result, 'content'):
                        return str(result.content)
                    elif hasattr(result, 'token'):
                        return str(result.token)
                    elif isinstance(result, str):
                        return result
                    else:
                        return str(result)
                else:
                    return "I processed your message but didn't generate a response."
                    


            except concurrent.futures.TimeoutError:
                return "Response generation timed out; response is likely still being generated."
            except Exception as injection_error:
                return f"Event handling error: {injection_error}"
                
        except Exception as e:
            return f"Message processing error: {e}"
   
    def handle_agent_message(self, message: str, source: str = None, 
                           tags: list = None, timeout: float = 30.0) -> str:
        """
        MAIN API: Handle agent message with response        
        """

        try:
            if not self._message_handler_active:
                return "Agent message handler not active"
            
            # Generate unique message ID
            message_id = f"msg_{int(datetime.now().timestamp() * 1000)}"

            # Prepare message data
            message_data = {
                'id': message_id,
                'message': message,
                'source': source,
                'tags': tags or ['gui_message'],
                'timestamp': datetime.now().timestamp()
            }
            
            # Queue the message
            self._message_queue.put(message_data)
            
            # Wait for response
            start_time = datetime.now().timestamp()
            while datetime.now().timestamp() - start_time < timeout:
                try:
                    response_data = self._response_queue.get(timeout=0.1)
                    if response_data['id'] == message_id:
                        return response_data['response']
                    else:
                        # Put back if not our response
                        self._response_queue.put(response_data)
                except Empty:
                    continue
            
            return "Timeout waiting for agent response"
            
        except Exception as e:
            return f"Error handling message: {e}"

    async def handle_shutdown(self):
        """
        Graceful cognitive shutdown sequence to prevent amnesia and preserve agent state
        """
        try:
            print("Initiating graceful cognitive shutdown...")
            
            # Stop message handler first
            self._message_handler_active = False
            if self._handler_thread and self._handler_thread.is_alive():
                print("Stopping message handler...")
                self._handler_thread.join(timeout=5.0)

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

            # Phase 4: Restart message handler if field is available
            if "_agent_field" in self.apis:
                self._start_message_handler()
            
            print("Cognitive state restoration completed successfully")
            
        except Exception as e:
            print(f"Error during state restoration: {e}")
            print("Starting with fresh cognitive state...")

    def get_agent_status(self) -> dict:
        """Get comprehensive agent system status"""
        try:
            status = {
                'message_handler_active': self._message_handler_active,
                'handler_thread_alive': self._handler_thread.is_alive() if self._handler_thread else False,
                'pending_messages': self._message_queue.qsize(),
                'pending_responses': self._response_queue.qsize(),
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

