"""
Centralized core module for agent initialization and management.
"""
import asyncio
import datetime
from apis.api_registry import api

  



class AgentCore:
    def __init__(self):
        self.log("Starting Agent Core initialization...", context="AgentCore.__init__")
        self.config = api.get_api("config")
        self.log("Configuration loaded successfully", context="AgentCore.__init__")

        self.log("Importing MemoryBank...", context="AgentCore.__init__")
        from apis.agent.memory.memory import MemoryBank  
        self.memory_bank = MemoryBank(field = None)
        self.log("MemoryBank imported successfully", context="AgentCore.__init__")

        self.log("Importing EventHandler...", context="AgentCore.__init__")
        from apis.agent.event_handler import EventHandler
        self.event_handler = EventHandler(memory = self.memory_bank, field = None)
        self.log("EventHandler imported successfully", context="AgentCore.__init__")

        self.log("Importing AdaptiveDistanceEngine...", context="AgentCore.__init__")
        from apis.agent.engine.adaptive_engine import AdaptiveDistanceEngine
        self.adaptive_engine = AdaptiveDistanceEngine()
        self.log("AdaptiveDistanceEngine imported successfully", context="AgentCore.__init__")

        self.log("Importing ParticleField...", context="AgentCore.__init__")
        from apis.agent.engine.field import ParticleField  
        self.particle_field = ParticleField(adaptive_engine = self.adaptive_engine, event_handler = self.event_handler, memory_bank= self.memory_bank)
        self.log("ParticleField imported successfully", context="AgentCore.__init__")
        
        self.log("Importing LexiconStore...", context="AgentCore.__init__")
        from apis.agent.cognition.linguistics.lexicon_store import LexiconStore  
        self.lexicon_store = LexiconStore(adaptive_engine = self.adaptive_engine, memory = self.memory_bank)
        self.log("LexiconStore imported successfully", context="AgentCore.__init__")

        self.log("Importing MetaVoice...", context="AgentCore.__init__")
        from apis.agent.cognition.linguistics.voice import MetaVoice  
        self.meta_voice = MetaVoice(field = self.particle_field, memory = self.memory_bank, lexicon = self.lexicon_store, model_handler = None)
        self.log("MetaVoice imported successfully", context="AgentCore.__init__")

        self.log("Importing ModelHandler...", context="AgentCore.__init__")
        from apis.model.model_handler import ModelHandler
        self.model_handler = ModelHandler(events = self.event_handler, meta_voice = self.meta_voice)
        self.log("ModelHandler imported successfully", context="AgentCore.__init__")
        
        self.log("Importing CognitionLoop...", context="AgentCore.__init__")
        from apis.agent.loop import CognitionLoop
        self.cognition_loop = CognitionLoop(events = self.event_handler, field = self.particle_field, adaptive_engine = self.adaptive_engine, memory = self.memory_bank, voice = self.meta_voice, lexicon = self.lexicon_store)
        self.log("CognitionLoop imported successfully", context="AgentCore.__init__")

        self.log("Importing and registering external resources...", context="AgentCore.__init__")
        # import and register additional resources
        from apis.research.external_resources import ExternalResources
        self.log("ExternalResources imported and registered successfully", context="AgentCore.__init__")

        self.log("Agent Core module imports complete, finalizing module threading", context="AgentCore.__init__")
        # event threading
        self.event_handler.field = self.particle_field
        # memory threading
        self.memory_bank.field = self.particle_field
        # field threading
        self.particle_field.voice = self.meta_voice
        # voice threading
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
        api.register_api("_agent_events", self.event_handler)
        api.register_api("_agent_model_handler", self.model_handler)

        print("systems check: ")
        print(f"Registered APIs: {api.list_apis()}")
        self.log(f"Registered APIs: {api.list_apis()}", "INFO", "initialize()")

        self._running = True


    async def run(self):
        """Main orchestration method - replaces run_agent()"""
        try:
            self.log("Agent Core initializing cognitive processes...", context="run")
            
            await self.initialize()
            await self.event_handler.initialize()
            #await self.memory_bank.verify_memory_system_health()               # **DEBUG** for testing DB health in the case of issues
            
            # Restoring previous state
            try:
                await api.handle_startup_restoration()
                self.log("Previous cognitive state restored", context="run")
            except Exception as e:
                self.log(f"State restoration error: {e}", level="ERROR", context="run")
                import traceback
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
                "events_handler": "ONLINE" if self.event_handler else "OFFLINE",
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
            import traceback
            self.log(f"Full traceback: {traceback.format_exc()}", level="ERROR", context="run")
            self._running = False
            self.cognition_loop.conscious_active = False
            await self.cognition_loop.shutdown_sequence()


    def shutdown(self):
        """Externally-callable method to stop the agent"""
        self.log("External shutdown requested", context="shutdown")
        api.handle_shutdown()
        self._running = False
        self.cognition_loop.conscious_active = False
        
        for task in asyncio.all_tasks():
            if not task.done():
                task.cancel()

    def handle_agent_message(self, message: str, source: str = None, 
                           tags: list = None, timeout: float = 60.0) -> str:
        """
        MAIN API: Handle agent message with response        
        """
        """
        # testing async call from sync context for emit_event, if it doesnt work try below method
        try:
            future = asyncio.gather(
                self.event_handler.emit_event("user_input", message, source or "api_registry", 1),
                return_exceptions=True
            )
            return future
        except RuntimeError:
            return asyncio.run(self.event_handler.emit_event("user_input", message, source or "api_registry", 1))

        """
        try:
            # Ensure event handler has loop reference
            if not hasattr(self.event_handler, '_agent_loop') or self.event_handler._agent_loop is None:
                self.log("Event handler missing loop reference, attempting to set", "WARNING", "handle_agent_message")
                # Try to get the loop from a known async context
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        self.event_handler._agent_loop = loop
                except:
                    pass

            # Generate unique message ID
            message_id = f"msg_{int(datetime.datetime.now().timestamp() * 1000)}"

            # Prepare message data
            message_data = {
                'id': message_id,
                'message': message,
                'source': source,
                'tags': tags or ['gui_message'],
                'timestamp': datetime.datetime.now().timestamp()
            }

            message = message_data['message']
            source = message_data.get('source', 'api_registry')

            try:
                self.log("Using handle_event_sync for event processing", "DEBUG", "_process_agent_message")
                result = self.event_handler.handle_event_sync(message, "user_input", source)
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
                
            except Exception as injection_error:
                return f"Event handling error: {injection_error}"
                
        except Exception as e:
            return f"Message processing error: {e}"
            
