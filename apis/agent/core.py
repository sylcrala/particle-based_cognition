"""
Centralized core module for agent initialization and management.
"""
import asyncio

from apis.api_registry import api

  



class AgentCore:
    def __init__(self):
        self.config = api.get_api("config")

        from apis.agent.memory.memory import MemoryBank  
        self.memory_bank = MemoryBank(field = None)

        from apis.agent.event_handler import EventHandler
        self.event_handler = EventHandler(memory = self.memory_bank, field = None)

        from apis.agent.engine.adaptive_engine import AdaptiveDistanceEngine
        self.adaptive_engine = AdaptiveDistanceEngine()

        from apis.agent.engine.field import ParticleField  
        self.particle_field = ParticleField(adaptive_engine = self.adaptive_engine, event_handler = self.event_handler, memory_bank= self.memory_bank)
        
        from apis.agent.cognition.linguistics.lexicon_store import LexiconStore  
        self.lexicon_store = LexiconStore(adaptive_engine = self.adaptive_engine, memory = self.memory_bank)

        from apis.agent.cognition.linguistics.voice import MetaVoice  
        self.meta_voice = MetaVoice(field = self.particle_field, memory = self.memory_bank, lexicon = self.lexicon_store, model_handler = None)

        from apis.model.model_handler import ModelHandler
        self.model_handler = ModelHandler(events = self.event_handler, meta_voice = self.meta_voice)
        
        from apis.agent.loop import CognitionLoop
        self.cognition_loop = CognitionLoop(events = self.event_handler, field = self.particle_field, adaptive_engine = self.adaptive_engine, memory = self.memory_bank, voice = self.meta_voice, lexicon = self.lexicon_store)


        # import and register additional resources
        from apis.research.external_resources import ExternalResources

        # event threading
        self.event_handler.field = self.particle_field

        # memory threading
        self.memory_bank.field = self.particle_field

        # field threading
        self.particle_field.voice = self.meta_voice

        # voice threading
        self.meta_voice.model_handler = self.model_handler



        
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

        # Initialize event handler
        try:
            await self.event_handler.initialize()
        except Exception as e:
            self.log(f"Event handler initialization error: {e}", level="ERROR", context="run")


    async def run(self):
        """Main orchestration method - replaces run_agent()"""
        try:
            self.log("Agent Core initializing cognitive processes...", context="run")

            await self.initialize()

            # Restoring previous state
            try:
                await api.handle_startup_restoration()
                self.log("Previous cognitive state restored", context="run")
            except Exception as e:
                self.log(f"State restoration error: {e}", level="ERROR", context="run")

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

            
            # Start the three cognitive loops
            self.log("Starting cognitive loops...", context="run")
            self.cognition_loop.conscious_active = True
            self.particle_field._update_active = True
            await asyncio.gather(
                self.cognition_loop.conscious_loop(),
                self.cognition_loop.subconscious_loop(), 
                self.cognition_loop.field_monitor_loop(),
                self.particle_field.continuous_particle_updates(),
                return_exceptions=True
            )
        
        except KeyboardInterrupt:
            self.log("Received shutdown signal", context="run")
            self.cognition_loop.conscious_active = False
            await self.cognition_loop.shutdown_sequence()

        except Exception as e:
            self.log(f"Runtime error: {e}", level="ERROR", context="run")
            self.cognition_loop.conscious_active = False
            await self.cognition_loop.shutdown_sequence()
