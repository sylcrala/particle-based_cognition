"""
Centralized core module for agent initialization and management.
"""
import asyncio
import datetime
import heapq
import time
import concurrent
import traceback
from apis.api_registry import api
from apis.agent.event_handler import EventHandler

events = None

class AgentCore:
    def __init__(self):
        
        self.log("Starting Agent Core initialization...", context="AgentCore.__init__")
        self.config = api.get_api("config")
        self.mode = self.config.agent_mode
        self.log("Configuration loaded successfully", context="AgentCore.__init__")
        self.log(f"Agent mode: {self.mode}", context="AgentCore.__init__")

        self.log("Importing MemoryBank...", context="AgentCore.__init__")
        from apis.agent.memory.memory import MemoryBank  
        self.memory_bank = MemoryBank(field = None)
        self.log("MemoryBank imported successfully", context="AgentCore.__init__")

        #self.log("Importing EventHandler...", context="AgentCore.__init__")
        #self.event_handler = EventHandler(memory=self.memory_bank, field=None)
        #self.log("EventHandler imported successfully", context="AgentCore.__init__")

        self.log("Importing AdaptiveDistanceEngine...", context="AgentCore.__init__")
        from apis.agent.engine.adaptive_engine import AdaptiveDistanceEngine
        self.adaptive_engine = AdaptiveDistanceEngine()
        self.log("AdaptiveDistanceEngine imported successfully", context="AgentCore.__init__")

        self.log("Initializing AgentAnchor...", context="AgentCore.__init__")
        self.agent_anchor = AgentAnchor(memory=self.memory_bank, field=None)
        self.log("AgentAnchor initialized successfully", context="AgentCore.__init__")

        self.log("Importing ParticleField...", context="AgentCore.__init__")
        from apis.agent.engine.field import ParticleField  
        self.particle_field = ParticleField(
            adaptive_engine = self.adaptive_engine, 
            event_handler = self.agent_anchor, 
            memory_bank= self.memory_bank
        )
        self.log("ParticleField imported successfully", context="AgentCore.__init__")
        
        self.log("Importing LexiconStore...", context="AgentCore.__init__")
        from apis.agent.cognition.linguistics.lexicon_store import LexiconStore  
        self.lexicon_store = LexiconStore(
            adaptive_engine = self.adaptive_engine, 
            memory = self.memory_bank
        )
        self.log("LexiconStore imported successfully", context="AgentCore.__init__")

        self.log("Importing MetaVoice...", context="AgentCore.__init__")
        from apis.agent.cognition.linguistics.voice import MetaVoice  
        self.meta_voice = MetaVoice(
            field = self.particle_field, 
            memory = self.memory_bank, 
            lexicon = self.lexicon_store, 
            model_handler = None
        )
        self.log("MetaVoice imported successfully", context="AgentCore.__init__")

        if self.mode == "llm-extension":
            self.log("Importing ModelHandler...", context="AgentCore.__init__")
            from apis.model.model_handler import ModelHandler
            self.model_handler = ModelHandler(
                events = self.agent_anchor, 
                meta_voice = self.meta_voice
            )
            self.log("ModelHandler imported successfully", context="AgentCore.__init__")
        else:
            self.model_handler = None
            self.log("ModelHandler skipped due to cog-growth mode", context="AgentCore.__init__")

        self.log("Importing CognitionLoop...", context="AgentCore.__init__")
        from apis.agent.loop import CognitionLoop
        self.cognition_loop = CognitionLoop(
            events = self.agent_anchor, 
            field = self.particle_field, 
            adaptive_engine = self.adaptive_engine, 
            memory = self.memory_bank, 
            voice = self.meta_voice, 
            lexicon = self.lexicon_store
        )

        self.log("Importing InferenceEngine...", context="AgentCore.__init__")
        from apis.agent.cognition.reasoning.inference_engine import InferenceEngine
        self.inference_engine = InferenceEngine()
        self.log("InferenceEngine imported successfully", context="AgentCore.__init__")

        self.log("CognitionLoop imported successfully", context="AgentCore.__init__")

        self.log("Importing research modules...")
        self.log("Importing external resources...", context="AgentCore.__init__")
        # import and register additional resources
        from apis.research.external_resources import ExternalResources
        self.external_resources = ExternalResources()
        self.log("ExternalResources imported and registered successfully", context="AgentCore.__init__")

        self.log("Importing wikipedia research module...", context="AgentCore.__init__")
        from apis.research.wikipedia import WikipediaSearcher
        self.wikipedia_searcher = WikipediaSearcher()
        self.log("WikipediaSearcher imported successfully", context="AgentCore.__init__")


        self.log("Agent Core module imports complete, finalizing module threading", context="AgentCore.__init__")
        # anchor threading
        self.agent_anchor.field = self.particle_field
        # memory threading
        self.memory_bank.field = self.particle_field
        # field threading
        self.particle_field.voice = self.meta_voice
        # voice threading if in llm-extension mode
        if self.mode == "llm-extension":
            self.meta_voice.model_handler = self.model_handler
        self.log("Module threading complete", context="AgentCore.__init__")

        self._running = False
        self.log("Agent Core initialization complete", context="AgentCore.__init__")

        
    def log(self, message, level = None, context = None, source = None):
        if source != None:
            source = "AgentCore"
        else:
            source = source


        if context != None:
            context = context
        else:
            context = "no context"

        if level != None:
            level = level
        else:
            level = "INFO"

        api.call_api("logger", "log", (message, level, context, source))

    
    async def initialize(self):
        """Initial setup and registration of all components"""
        self.log("Initializing Agent Core components...", context="initialize")

        api.register_api("_agent_cognition_loop", self.cognition_loop)
        api.register_api("_agent_field", self.particle_field)
        api.register_api("_agent_memory", self.memory_bank)
        api.register_api("_agent_lexicon", self.lexicon_store)
        api.register_api("_agent_meta_voice", self.meta_voice)
        api.register_api("_agent_adaptive_engine", self.adaptive_engine)
        api.register_api("_agent_events", self.agent_anchor)
        api.register_api("_agent_anchor", self.agent_anchor)
        api.register_api("_agent_inference_engine", self.inference_engine)
        # research modules
        api.register_api("external_resources", self.external_resources)
        api.register_api("wikipedia_searcher", self.wikipedia_searcher)

        if self.mode == "llm-extension":
            api.register_api("_agent_model_handler", self.model_handler)

        # Initialize inference engine after APIs are registered
        self.inference_engine.initialize()

        await self.agent_anchor.initialize()

        print("systems check: ")
        print(f"Registered APIs: {api.list_apis()}")
        self.log(f"System Startup Final Report: \n\nRegistered APIs: \n{api.list_apis()}", "INFO", "initialize()")

        self._running = True


    async def run(self):
        """Main orchestration method - replaces run_agent()"""
        try:
            self.log("Agent Core initializing cognitive processes...", context="run")
            
            await self.initialize()
            #await self.memory_bank.verify_memory_system_health()               # **DEBUG** for testing DB health in the case of issues
            
            # Restoring previous state
            try:
                await api.handle_startup_restoration()
                self.log("Previous cognitive state restored", context="run")
            except Exception as e:
                self.log(f"State restoration error: {e}", level="ERROR", context="run")
                self.log(f"Full traceback:\n{traceback.format_exc()}")
                raise Exception(f"Error: {e} \n\n Traceback: {traceback.format_exc()}")

            # Ensure lexicon is loaded
            try:
                await self.lexicon_store.load_lexicon()
                self.log("Lexicon loaded successfully", context="run")
            except Exception as e:
                self.log(f"Lexicon loading error: {e}", level="ERROR", context="run")

            # Log module availability
            module_status = {
                "model_handler": "ONLINE" if self.model_handler else "OFFLINE",
                "cognition_loop": "ONLINE" if self.cognition_loop else "OFFLINE",
                "particle_field": "ONLINE" if self.particle_field else "OFFLINE",
                "memory_bank": "ONLINE" if self.memory_bank else "OFFLINE",
                "meta_voice": "ONLINE" if self.meta_voice else "OFFLINE",
                "lexicon": "ONLINE" if self.lexicon_store else "OFFLINE",
                "adaptive_engine": "ONLINE" if self.adaptive_engine else "OFFLINE",
            }    
            self.log(f"Module Status: {module_status}", context="initialize_cognitive_systems")
            
            # Start the cognitive loops
            self.log("Starting cognitive loops...", context="run")
            self.cognition_loop.conscious_active = True
            self.particle_field._update_active = True
            
            # Main loop that keeps restarting tasks if they complete
            while self._running:
                try:
                    self.log("Starting cognitive task group...", level="DEBUG", context="run")
                    
                    # Create tasks that run continuously
                    tasks = [
                        asyncio.create_task(self.particle_field.continuous_particle_updates(), name="particle_field"),
                        asyncio.create_task(self.cognition_loop.conscious_loop(), name="conscious_loop"),
                        asyncio.create_task(self.cognition_loop.subconscious_loop(), name="subconscious_loop")
                    ]
                    
                    # Wait for any task to complete (which shouldn't happen in normal operation)
                    done, pending = await asyncio.wait(
                        tasks, 
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # If we reach here, something completed unexpectedly
                    completed_tasks = []
                    for task in done:
                        task_name = task.get_name()
                        if task.exception():
                            self.log(f"Task '{task_name}' failed with exception: {task.exception()}", 
                                level="ERROR", context="run")
                            completed_tasks.append(f"{task_name}(ERROR)")
                        else:
                            result = task.result()
                            self.log(f"Task '{task_name}' completed unexpectedly with result: {result}", 
                                level="WARNING", context="run")
                            completed_tasks.append(f"{task_name}(COMPLETED)")
                    
                    # Cancel remaining tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    
                    # Log what happened and restart if still running
                    if self._running:
                        self.log(f"Cognitive tasks completed unexpectedly: {completed_tasks}. Restarting...", 
                            level="WARNING", context="run")
                        await asyncio.sleep(1)  # Brief pause before restart
                    
                except Exception as e:
                    if self._running:
                        self.log(f"Task group error: {e}, restarting...", level="ERROR", context="run")
                        await asyncio.sleep(5)  # Longer pause on error
                    else:
                        break

            # Cleanup when shutting down
            self.log("Shutting down cognitive loops...", context="run")
            self.cognition_loop.conscious_active = False
            self.particle_field._update_active = False
            
        except KeyboardInterrupt:
            self.log("Received shutdown signal", context="run")
            self._running = False
            self.cognition_loop.conscious_active = False
            await self.cognition_loop.shutdown_sequence()

        except Exception as e:
            self.log(f"Runtime error: {e}", level="ERROR", context="run")
            self.log(f"Full traceback: {traceback.format_exc()}", level="ERROR", context="run")
            self._running = False
            self.cognition_loop.conscious_active = False
            await self.cognition_loop.shutdown_sequence()


    def shutdown(self):
        """Externally-callable method to stop the agent"""
        self.log("External shutdown requested", context="shutdown")
        asyncio.run(api.handle_shutdown())
        self._running = False
        self.cognition_loop.conscious_active = False
        
        for task in asyncio.all_tasks():
            if not task.done():
                task.cancel()



class AgentAnchor:
    def __init__(self, memory = None, field = None):
        self.logger = api.get_api("logger")

        # Event handling properties
        self.event_queue = PriorityEventQueue()
        self.event_handlers = {}
        self.initialized = False
        self.running = False
        self.event_loop_task = None
        self.start_time = 0
        self._agent_loop = None

        self.event_failure_counts = {}
        self.max_failures = 5 # max retries before disabling an event type (event sys circuit breaker)

        # Core particle properties
        self.permanent_cores = []
        self.temporary_cores = []
        self.core_roles = {
            "memory_coordination": None,
            "decision_making": None,
            "social_interaction": None,
            "system_monitoring": None,
            "reflective_thoughts": None,
            "identity_anchor": None,
            "reasoning_coordinator": None,
            "knowledge_curator": None,
            "hypothesis_generator": None
        }

        # References
        self.field = field
        self.memory = memory

        # Initialize default event handlers
        self.register_default_handlers()

    def log(self, message, level="INFO", context=None):
        """Use shared logging system"""
        context = context or "no_context"
            
        if self.logger:
            self.logger.log(message, level, context, "AgentAnchor")
        else:
            print(f"[{level}] {message}")  # Fallback
    
    def get_core_by_role(self, role):
        """Retrieve core particle by its assigned role"""
        try:
            self.log(f"Retrieving core for role: {role}", "DEBUG", "get_core_by_role")
            particle_list = self.permanent_cores or self.field.particles 
            for core in particle_list:
                if core.role == role:
                    self.log(f"Core found for role {role}: ID {core.id}", "DEBUG", "get_core_by_role")
                    return core
        except Exception as e:
            self.log(f"Error retrieving core for role {role}: {e}", "ERROR", "get_core_by_role")
            self.log(f"Full traceback:\n{traceback.format_exc()}", "ERROR", "get_core_by_role")


    ### event handling methods ###

    def register_default_handlers(self):
        """Register default event handlers for common events"""
        self.event_handlers.update({
            "particle_created": self.handle_particle_created,
            "user_input": self.handle_user_input,
            "system_idle": self.handle_system_idle,
            "system_events": self.handle_system_events,
            "shutdown": self.handle_shutdown,
            "reflection_triggered": self.handle_reflection,
            "reflection_request": self.handle_reflection,
            "cognitive_event": self.handle_cognitive_event,
            "memory_event": self.handle_memory_event,
            "reasoning_cycle": self.handle_reasoning_cycle,
            "learning_moment_detected": self.handle_learning_moment
        })
    
    async def emit_event(self, event_type, data=None, source="unknown", priority=None):
        """Emit an event with optional priority"""
        try:
            event = {
                "type": event_type,
                "data": data,
                "source": source,
                "timestamp": datetime.datetime.now().timestamp()
            }
            
            if priority is None:
                priority = self.get_default_priority(event_type, source)

            # Direct event handling for specific tasks (those that require event returns)
            if event_type in ("user_input", "system_events"):
                handling_core = None
                if event_type == "user_input":
                    result = await self.handle_event(event)
                    return result
                elif event_type == "system_events":
                    if data == "system_metrics request":
                        result = await self.handle_event(event)
                        return result

            else:
                await self.event_queue.put(event, priority=priority)
                self.log(f"Event emitted: {event_type} from {source}", "DEBUG", context="emit_event")
                return True
            
        except Exception as e:
            self.log(f"Error emitting event {event_type} from {source}: {e}", "ERROR", context="emit_event")
            self.log(f"Full traceback:\n{traceback.format_exc()}", "ERROR", context="emit_event")
            return None
    
    async def handle_event_through_core(self, core, event):
        """Route event to specified core for handling"""
        event_type = event["type"]

        if not core:
            self.log("No core available for event handling", "WARNING", context="handle_event_through_core")
            return f"No core particle available for event of type: {event_type}"

        # Let core handle event autonomously if it can
        try:
            result = await core.handle_event(event)

            core._update_decision_history(event, result)

            if result is not None:
                return result
        
        except Exception as e:
            self.log(f"Error in core {core.id} handling event {event['type']}: {e}", "ERROR", context="handle_event_through_core")
            self.log(f"Full traceback:\n{traceback.format_exc()}", "ERROR", context="handle_event_through_core")
            return f"Error: {e}"

    def send_message(self, message: str, source: str = None) -> str:
        """
        MAIN API: Handle agent message with response        
        """

        return self.emit_event_sync(message, "user_input", source or "agentanchor")

    def emit_event_sync(self, data, event_type, source="unknown", timeout=300.0): ##
        """Syncronous routing to emit_event"""
        try:
            if not hasattr(self, '_agent_loop'):
                self.log("No event loop available for sync event emission", "ERROR", "emit_event_sync")
                return "Error: No event loop available"

            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.emit_event(event_type, data, source),
                    self._agent_loop
                )
                result = future.result(timeout=timeout)
                return result
            
            except concurrent.futures.TimeoutError:
                self.log("Timeout in sync event emission", "ERROR", "emit_event_sync")
                return "Error: Timeout"
        except Exception as e:
            self.log(f"Error in sync event emission: {e}", "ERROR", "emit_event_sync")
            self.log(f"Full traceback:\n{traceback.format_exc()}", "ERROR", "emit_event_sync")
            return f"Error: {e}"
    
    async def get_events_by_type(self, event_type):
        """Retrieve events of a specific type without removing them"""
        return [event for _, _, event in self.event_queue._queue if event["type"] == event_type]
    
    def register_listener(self, event_type, callback, once=False):
        """Register a callback for a specific event type"""
        async def listener(event):
            await callback(event)
            if once:
                self.event_handlers.pop(event_type, None)
        
        self.event_handlers[event_type] = listener
        self.log(f"Listener registered for event type: {event_type}", "DEBUG", context="register_listener")

    def get_default_priority(self, event_type, source):
        """Get default priority for different event types"""
        priority_map = {
            "user_input": 2,
            "shutdown": 0,
            "particle_created": 2,
            "system_idle": 7,
            "system_events": 1,
            "cognitive_event": 3,
            "reflection_triggered": 4, # internal trigger for reflection
            "reflection_request": 6, # external request for reflection
            "memory_event": 3,
            "reasoning_cycle": 2,
            "learning_moment_detected": 3
        }
        
        base_priority = priority_map.get(event_type, 5)
        
        # User events get higher priority
        if source == "user":
            base_priority = min(base_priority, 2)
            
        return base_priority
    
    async def handle_event(self, event, core_particle = None):
        """Central event dispatcher"""
        current_time = datetime.datetime.now().timestamp()
        if hasattr(self, "start_time") and current_time - self.start_time < 5.0:
            self.log("Skipping event handling during startup stabilization period", "DEBUG", context = "handle_event")
            return "Please retry after stabilization period."
        
        event_key = f"{event['type']}_{event.get('source', 'unknown')}"
        if event_key in self.event_failure_counts:
            if self.event_failure_counts[event_key] >= self.max_failures:
                self.log(f"Event type {event['type']} from {event.get('source', 'unknown')} disabled due to repeated failures", "ERROR", context="handle_event")
                return "[Error: Event cancelled due to cascading failures]"

        event_type = event["type"]

        self.log(f"Handling event: {event_type} from {event['source']}", context="handle_event")

        # Find and execute handler
        handler = self.event_handlers.get(event_type, self.handle_unknown_event)
        
        try:
            result = await handler(event, core_particle if core_particle else None)

            if event_key in self.event_failure_counts:
                del self.event_failure_counts[event_key] # reset failure count on success
            return result
        except Exception as e:
            self.event_failure_counts[event_key] = self.event_failure_counts.get(event_key, 0) + 1
            self.log(f"Event handling failure #{self.event_failure_counts[event_key]} for {event_key}: {e}")
            return f"Error handling event: {e}"
    
    def handle_event_sync(self, data, event_type, source="unknown"): ##
        """Syncronous event handler for thread-safe GUI""" 
        try:
            event = {
                "type": event_type,
                "data": data,
                "source": source,
                "timestamp": datetime.datetime.now().timestamp()
            }

            if not hasattr(self, '_agent_loop'):
                self.log("No event loop available for sync event handling", "ERROR", "handle_event_sync")
                return "Error: No event loop available"

            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.handle_event(event),
                    self._agent_loop
                )
                result = future.result(timeout=120.0)
                return result
            
            except concurrent.futures.TimeoutError:
                self.log("Timeout in sync event handling", "ERROR", "handle_event_sync")
                return "Error: Timeout"
            except Exception as e:
                self.log(f"Error in sync event handling: {e}", "ERROR", "handle_event_sync")
                self.log(f"Full traceback:\n{traceback.format_exc()}", "ERROR", "handle_event_sync")
                return f"Error: {e}"

        except Exception as e:
            self.log(f"Sync event handling error: {e}", "ERROR", "handle_event_sync")
            return f"Error: {e}"

    async def handle_memory_event(self, event, core_particle = None):
        """Handle memory-related events"""
        event_type = event["type"]
        event_data = event["data"]
        self.log(f"Memory event received: {event_type}", "DEBUG", context="handle_memory_event")

        # Could route to specialized memory system
        if event_data == "memory_consolidation":
            try:
                self.log("Starting memory consolidation process without core particles...", "DEBUG", "handle_memory_event")
                if not self.memory:
                    self.log("No memory system available for consolidation", "ERROR", "handle_memory_event")
                    return
                
                    
                # Get recent high-activation particles for memory consolidation  
                if self.field:
                    particles = self.field.get_all_particles()
                    high_activation = [p for p in particles if hasattr(p, 'activation') and p.activation > 0.65]
                    
                    for particle in high_activation[:15]:  # Consolidate top 15
                        if hasattr(particle, 'metadata') and particle.metadata:
                            await self.memory.consolidate_particle_memory(particle)

                self.log(f"Memory consolidation completed for {len(high_activation)} particles", "DEBUG", "handle_memory_event")
                return True

            except Exception as e:
                self.log(f"Memory consolidation error: {e}", "ERROR", "handle_memory_event")

        elif event_data == "memory_retrieval":
            # TODO
            pass

        elif event_data == "memory_store":
            # TODO
            pass

        elif event_data == "emergency_state_save":
            # TODO
            pass


        return False

    async def handle_particle_created(self, event, core_particle = None):
        """Handle particle creation events"""
        particle_data = event["data"]
        try:
            await self.field.spawn_particle(
                type=particle_data.get("type", "unknown"),
                metadata=particle_data.get("metadata", {}),
                energy=particle_data.get("energy", 0.5),
                activation=particle_data.get("activation", 0.5),
                source_particle_id=str(core_particle.id) if core_particle else None,
                emit_event=False  # Avoid recursive event emission
            )
            self.log(f"Particle created: {particle_data.get('particle_id', 'unknown')}", "DEBUG", context="handle_particle_created")
            return True
        except Exception as e:
            self.log(f"Error creating particle: {e}", "ERROR", context="handle_particle_created")
            self.log(f"Full traceback:\n{traceback.format_exc()}", "ERROR", context="handle_particle_created")
            return False

    
    async def handle_user_input(self, event, core_particle = None):
        """Handle user input events"""
        try:
            handling_core = self.get_core_by_role("social_interaction")
            if handling_core:
                result = await self.handle_event_through_core(handling_core, event)
                self.log(f"User input event handled by core: {handling_core.id}", "DEBUG", context="emit_event")
                return result
            
        except Exception as e:
            self.log(f"Error routing user input through core: {e} | handling input directly", "ERROR", context="handle_user_input")
            self.log(f"Full traceback:\n{traceback.format_exc()}", "ERROR", context="handle_user_input")

            user_data = event["data"]
            
            # Check for shutdown commands
            if isinstance(user_data, str) and user_data.lower() in ("exit", "quit", "shutdown"):
                await self.emit_event("shutdown", {"reason": "user_request"}, source="user")
                self.log("Shutdown command received from user input", "SYSTEM", "handle_user_input")
                return "Shutdown initiated"
            
            config = api.get_api("config")
            if config and config.user_name:
                user_name = config.user_name
            else:
                user_name = "Unknown User"

            # Route to particle field for processing as a backup generation method (direct field access rather than core particle routing)
            field = self.field
            if field:
                try:
                    input_for_agent = f"{user_name} said: <s>{user_data}</s>"
                    result = await field.inject_action(input_for_agent, source="user_input", source_particle_id=str(core_particle.id) if core_particle else None)
                    
                    if result:
                        self.log(f"User input processed, response generated: {result}", "DEBUG", context="handle_user_input")
                        return result
                    else:
                        self.log("No response generated for user input", "WARNING", context="handle_user_input")
                        return "[System] No response available."
                        
                except Exception as e:
                    self.log(f"Error processing user input: {e}", "ERROR", context="handle_user_input")
                    self.log(f"Full traceback:\n{traceback.format_exc()}")
                    return "[System] Error processing input."
            else:
                self.log("No particle field available for user input", "ERROR", context="handle_user_input")
                return "[System] Cognitive system unavailable."
            
    async def handle_system_events(self, event, core_particle = None):
        """Handle general system events"""
        event_type = event["type"]
        event_data = event["data"]
        self.log(f"System event received: {event_type}", "DEBUG", context="handle_system_events")
        if event_data == "system_metrics request":
            try:
                handling_core = self.get_core_by_role("system_monitoring")
                if handling_core:
                    result = await self.handle_event_through_core(handling_core, event)
                    self.log(f"System metrics event handled by core: {handling_core.id}", "DEBUG", context="emit_event")
                    return result
            except Exception as e:
                self.log(f"Error routing system event through core: {e} | handling directly", "ERROR", context="handle_system_events")
                self.log(f"Full traceback:\n{traceback.format_exc()}", "ERROR", context="handle_system_events")
                return False

    async def handle_system_idle(self, event, core_particle = None):
        """Handle system idle events for maintenance"""
        self.log("System idle - performing maintenance", "DEBUG", context="handle_system_idle")
        
        # Could trigger particle pruning, memory consolidation, etc.
        memory = self.memory
        if memory and hasattr(memory, 'perform_maintenance'):
            await memory.perform_maintenance()
        # TODO 
        return True

    async def handle_shutdown(self, event, core_particle = None):
        """Handle shutdown events"""
        reason = event["data"].get("reason", "unknown")
        self.log(f"Shutdown event received: {reason}", "SYSTEM", context="handle_shutdown")

        # Trigger graceful shutdown through API registry
        api.handle_shutdown()
        
        # Stop event loop
        self.running = False
        if self.event_loop_task:
            self.event_loop_task.cancel()
        
        return True

    async def handle_reflection(self, event, core_particle = None):
        """Handle reflection events"""
        reflection_type = event["type"]
        reflection_data = event["data"]
        self.log(f"Reflection triggered: {reflection_type}", "DEBUG", context="handle_reflection")
        try:
            handling_core = self.get_core_by_role("reflective_thoughts")
            if handling_core:
                result = await self.handle_event_through_core(handling_core, event)
                self.log(f"Reflection event handled by core: {handling_core.id}", "DEBUG", context="emit_event")
                return result
        except Exception as e:
            self.log(f"Error routing reflection through core: {e} | handling directly", "ERROR", context="handle_reflection")
            self.log(f"Full traceback:\n{traceback.format_exc()}", "ERROR", context="handle_reflection")
            return False
        
    async def handle_reasoning_cycle(self, event, core_particle=None):
        """Handle reasoning cycle events from CoT pipeline"""
        event_type = event["type"]
        event_data = event["data"]
        
        # Extract trigger information if available
        trigger_source = "unknown"
        if isinstance(event_data, dict):
            trigger_source = event_data.get("trigger", "unknown")
        
        self.log(f"Reasoning cycle triggered: {trigger_source}", "DEBUG", context="handle_reasoning_cycle")
        
        try:
            handling_core = self.get_core_by_role("reasoning_coordinator")
            if handling_core:
                result = await self.handle_event_through_core(handling_core, event)
                self.log(f"Reasoning cycle handled by core: {handling_core.id}", "DEBUG", context="handle_reasoning_cycle")
                return result
            else:
                self.log("No reasoning coordinator core found", "WARNING", context="handle_reasoning_cycle")
                return False
        except Exception as e:
            self.log(f"Error routing reasoning cycle through core: {e}", "ERROR", context="handle_reasoning_cycle")
            self.log(f"Full traceback:\n{traceback.format_exc()}", "ERROR", context="handle_reasoning_cycle")
            return False

    async def handle_learning_moment(self, event, core_particle=None):
        """Handle learning moment detection events"""
        learning_data = event.get("data", {})
        token = learning_data.get("token", "unknown")
        self.log(f"Learning moment detected: {token}", "DEBUG", context="handle_learning_moment")
        
        try:
            handling_core = self.get_core_by_role("knowledge_curator")
            if handling_core:
                result = await self.handle_event_through_core(handling_core, event)
                self.log(f"Learning moment handled by core: {handling_core.id}", "DEBUG", context="handle_learning_moment")
                return result
            else:
                self.log("No knowledge curator core found", "WARNING", context="handle_learning_moment")
                return False
        except Exception as e:
            self.log(f"Error routing learning moment through core: {e}", "ERROR", context="handle_learning_moment")
            self.log(f"Full traceback:\n{traceback.format_exc()}", "ERROR", context="handle_learning_moment")
            return False

    async def handle_cognitive_event(self, event, core_particle = None):
        """Handle general cognitive events"""
        self.log(f"Cognitive event: {event['data']}", "DEBUG", context="handle_cognitive_event")
        # TODO
        return True
    
    async def handle_unknown_event(self, event, core_particle = None):
        """Handle unknown event types"""
        self.log(f"Unknown event type: {event['type']}", "WARNING", context="handle_unknown_event")
        # TODO
        return None
    
    def register_handler(self, event_type, handler):
        """Register a custom event handler"""
        self.event_handlers[event_type] = handler
        self.log(f"Registered handler for event type: {event_type}", "DEBUG", context="register_handler")
    
    async def start_event_loop(self):
        """Start the main event processing loop"""
        self.running = True
        self.log("Event handler started", "SYSTEM", context="start_event_loop")
        
        try:
            while self.running:
                event = await self.event_queue.get()
                await self.handle_event(event)
        except asyncio.CancelledError:
            self.log("Event loop cancelled - shutting down gracefully", "SYSTEM", context="start_event_loop")
        except Exception as e:
            self.log(f"Event loop error: {e}", "ERROR", context="start_event_loop")
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
            self.log("Idle scheduler cancelled", "DEBUG", context="start_idle_scheduler")
    
    async def initialize(self):
        """Initialize the event system with background tasks"""
        self.initialized = False
        # Start event loop
        self.event_loop_task = asyncio.create_task(self.start_event_loop())
        
        # Start idle scheduler  
        self.idle_task = asyncio.create_task(self.start_idle_scheduler())

        self._agent_loop = asyncio.get_event_loop()

        await asyncio.sleep(2.0)  # Give tasks a moment to start
        self.start_time = datetime.datetime.now().timestamp()
        self.initialized = True
        self.log("Event handler initialized with background tasks", "SYSTEM", context="initialize")
    
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

        self.log("Event handler shutdown complete", "SYSTEM", context="shutdown")

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
    
    def pop(self, event):
        """Remove specific event from queue"""
        for i, (_, _, e) in enumerate(self._queue):
            if e == event:
                del self._queue[i]
                heapq.heapify(self._queue)
                return True
        return False

    
    def qsize(self):
        """Get queue size"""
        return len(self._queue)
