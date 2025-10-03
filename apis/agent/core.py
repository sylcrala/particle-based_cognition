"""
Centralized core module for agent initialization and management.
"""
import asyncio

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
                    
                    # Use gather to run all tasks concurrently and restart if any complete
                    result = [
                        asyncio.gather(self.particle_field.update_particles(), return_exceptions=True),
                        asyncio.gather(self.cognition_loop.conscious_loop(), return_exceptions=True),
                        asyncio.gather(self.cognition_loop.field_monitor_loop(), return_exceptions=True),
                        asyncio.gather(self.cognition_loop.subconscious_loop(), return_exceptions=True)
                        #asyncio.gather(self.particle_field.continuous_particle_updates(), return_exceptions=True)
                    ]
                                
                    # Log which tasks completed/failed
                    self.log(f"Task group completed with results: {result}", level="WARNING", context="run")
                    
                    # If we reach here, one or more tasks completed unexpectedly
                    if self._running:
                        self.log("Task group completed unexpectedly, restarting...", level="WARNING", context="run")
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