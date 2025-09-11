"""
agent runtime loop (main async internal loop | essentially "run_agent()" from runtime.py in old format)
"""

import asyncio
from apis.api_registry import api

# Import cognitive engines to register their APIs
import apis.agent.engine.field  # Registers particle_field API
import apis.agent.engine.particle_engine  # Registers particle_engine API
import apis.agent.engine.adaptive_engine  # Registers adaptive_engine API
import apis.agent.memory.memory  # Registers memory_bank API
import apis.agent.cognition.linguistics.voice  # Registers meta_voice API
import apis.agent.cognition.linguistics.lexicon_store  # Registers lexicon_store API
import apis.agent.event_handler  # Registers event_handler API
import apis.model.model_handler  # Registers model_handler API


class CognitionLoop:
    def __init__(self):
        self.logger = api.get_api("logger")

        try:
            self.config = api.get_api("config")
            self.agent_config = self.config.get_agent_config()
            self.log("Config loaded", context="CognitionLoop.__init__")
        except Exception as e:
            self.log(f"Config load error: {e}", level="ERROR", context="CognitionLoop.__init__")
            raise e
        
        self.name = self.agent_config.get("name")

        self.conscious_active = False
        self.subconscious_cycle_count = 0

        self.log("CognitionLoop initialized", context="CognitionLoop.__init__")

    def log(self, message, level = None, context = None, source = None):
        if source != None:
            source = "CognitionLoop"
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


    async def get_status(self):
        status = {
            "is_online": self.conscious_active,
            "cycle_count": self.subconscious_cycle_count
        }
        return status

    async def initialize_cognitive_systems(self):
        """Initialize all cognitive systems and seed particles"""
        try:
            self.log("Initializing cognitive field...", context="initialize_cognitive_systems")
            
            # Get required APIs
            field_api = api.get_api("particle_field")
            memory_api = api.get_api("memory_bank")
            model_handler_api = api.get_api("model_handler")
            meta_voice_api = api.get_api("meta_voice")
            adaptive_engine_api = api.get_api("adaptive_engine")
            
            # Initialize model handler first (foundational LLM)
            if model_handler_api:
                self.log("Model handler available and ready", context="initialize_cognitive_systems")
            else:
                self.log("Model handler not available", level="WARNING", context="initialize_cognitive_systems")
            
            # Initialize particle field
            if field_api:
                await field_api.seed_particles()
                self.log("Particle field seeded successfully", context="initialize_cognitive_systems")
            else:
                self.log("Particle field not available", level="ERROR", context="initialize_cognitive_systems")
            
            # Log API availability
            api_status = {
                "particle_field": "ONLINE" if field_api else "OFFLINE",
                "memory_bank": "ONLINE" if memory_api else "OFFLINE", 
                "model_handler": "ONLINE" if model_handler_api else "OFFLINE",
                "meta_voice": "ONLINE" if meta_voice_api else "OFFLINE",
                "adaptive_engine": "ONLINE" if adaptive_engine_api else "OFFLINE"
            }
            
            self.log(f"API Status: {api_status}", context="initialize_cognitive_systems")
            self.log("Initialization complete, cognitive systems online", context="initialize_cognitive_systems")
            return True
            
        except Exception as e:
            self.log(f"Initialization failed: {e}", level="ERROR", context="initialize_cognitive_systems")
            return False

    async def perform_maintenance_cycle(self):
        """Memory consolidation, reflection, and particle pruning"""
        try:
            field_api = api.get_api("particle_field")
            meta_voice_api = api.get_api("meta_voice")
            memory_api = api.get_api("memory_bank")
            
            if not field_api:
                return
                
            # Get recent particles for reflection
            recent_particles = field_api.get_all_particles()[-5:]  # RECENT_MEM_LIMIT
            
            # Reflect on memory particles
            for particle in recent_particles:
                if particle.type == "memory" and meta_voice_api:
                    await meta_voice_api.reflect(particle)
            
            # Prune low-value particles
            await field_api.prune_low_value_particles()
            
            self.log("Maintenance cycle completed", context="perform_maintenance_cycle")
            
        except Exception as e:
            self.log(f"Maintenance cycle error: {e}", level="ERROR", context="perform_maintenance_cycle")

    async def shutdown_sequence(self):
        """Clean shutdown with data preservation"""
        try:
            self.log("SYS-shutdown: Initiating shutdown sequence...", context="shutdown_sequence")
            
            # Get APIs for cleanup
            memory_api = api.get_api("memory_bank")
            lexicon_api = api.get_api("lexicon_store")
            
            # Save critical data (if APIs support it)
            if hasattr(memory_api, 'save_to_file'):
                await memory_api.save_to_file()
            if hasattr(lexicon_api, 'save'):
                await lexicon_api.save()
                
            self.log("Shutdown complete. Session ended.", context="shutdown_sequence")
            
            # Cancel remaining tasks
            for task in asyncio.all_tasks():
                if not task.done():
                    task.cancel()
                    
        except Exception as e:
            self.log(f"Shutdown error: {e}", level="ERROR", context="shutdown_sequence")

    async def run(self):
        """Main orchestration method - replaces run_agent()"""
        try:
            # Initialize systems
            if not await self.initialize_cognitive_systems():
                self.log("Failed to initialize systems", level="ERROR", context="run")
                return False
            
            # Start the three cognitive loops
            self.log("Starting cognitive loops...", context="run")
            
            await asyncio.gather(
                self.conscious_loop(),
                self.subconscious_loop(), 
                self.field_monitor_loop(),
                return_exceptions=True
            )
            
        except KeyboardInterrupt:
            self.log("Received shutdown signal", context="run")
        except Exception as e:
            self.log(f"Runtime error: {e}", level="ERROR", context="run")
        finally:
            await self.shutdown_sequence()

    async def conscious_loop(self):
        """Handles active interactions and processing"""
        while True:
            try:
                # Mark conscious state as active
                self.conscious_active = True
                
                # Get event handler for user interactions
                events_api = api.get_api("event_handler")
                if not events_api:
                    await asyncio.sleep(1.0)
                    continue
                
                # Process high-priority events (user inputs, external triggers)
                if hasattr(events_api, 'run_event_loop'):
                    await events_api.run_event_loop()
                
                # Active memory retrieval and processing
                await self.process_active_cognition()
                
                # Brief rest between conscious cycles
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.log(f"Conscious loop error: {e}", level="ERROR", context="conscious_loop")
                await asyncio.sleep(1.0)
            finally:
                self.conscious_active = False

    async def process_active_cognition(self):
        """Handle active cognitive processing during conscious state"""
        try:
            field_api = api.get_api("particle_field")
            if not field_api:
                return
                
            # Get high-activation particles for conscious processing
            particles = field_api.get_all_particles()
            active_particles = [p for p in particles if hasattr(p, 'activation') and p.activation > 0.7]
            
            # Process active particles (conscious attention)
            for particle in active_particles[:5]:  # Focus on top 5
                if hasattr(particle, 'observe'):
                    # Conscious observation collapses quantum states
                    collapsed_state = particle.observe(context="conscious_attention")
                    
                    if collapsed_state == 'certain':
                        self.log(f"Conscious attention collapsed particle {particle.id} to certain state", 
                                level="DEBUG", context="process_active_cognition")
                        
        except Exception as e:
            self.log(f"Active cognition error: {e}", level="ERROR", context="process_active_cognition")


    async def subconscious_loop(self):
        """Handles background processing and cognitive tasks"""
        while True:
            try:
                # Increment cycle counter
                self.subconscious_cycle_count += 1
                
                # Perform maintenance every 100 cycles (adjustable)
                if self.subconscious_cycle_count % 100 == 0:
                    self.log(f"Performing subconscious maintenance cycle {self.subconscious_cycle_count}", 
                            context="subconscious_loop")
                    await self.perform_maintenance_cycle()
                
                # Process any pending reflections
                await self.process_reflection_queue()
                
                # Field-level quantum state monitoring
                await self.monitor_quantum_states()
                
                # Sleep to prevent excessive CPU usage
                await asyncio.sleep(1.0)  # Adjust timing as needed
                
            except Exception as e:
                self.log(f"Subconscious loop error: {e}", level="ERROR", context="subconscious_loop")
                await asyncio.sleep(5.0)  # Longer sleep on error


    async def process_reflection_queue(self):
        """Process reflection queue for particle learning"""
        try:
            field_api = api.get_api("particle_field")
            if not field_api:
                return
                
            # Get particles that need reflection processing
            particles = field_api.get_particles_by_type("lingual")
            reflection_candidates = [p for p in particles if p.metadata.get("needs_reflection", False)]
            
            for particle in reflection_candidates[:3]:  # Process a few at a time
                if hasattr(particle, 'learn_from_particle'):
                    await particle.learn_from_particle(particle)
                    particle.metadata["needs_reflection"] = False
                    
        except Exception as e:
            self.log(f"Reflection processing error: {e}", level="ERROR", context="process_reflection_queue")


    async def monitor_quantum_states(self):
        """Monitor particle superposition states and trigger collapses"""
        try:
            field_api = api.get_api("particle_field")
            if not field_api:
                return
                
            particles = field_api.get_all_particles()
            
            # Check for particles that need quantum state evaluation
            for particle in particles:
                if hasattr(particle, 'superposition') and hasattr(particle, 'observe'):
                    # Simple collapse trigger: if particle hasn't been observed recently
                    if not hasattr(particle, 'last_observed'):
                        particle.last_observed = 0
                        
                    # Trigger collapse for high-energy particles that haven't been observed
                    if particle.energy > 0.8 and (asyncio.get_event_loop().time() - particle.last_observed) > 10:
                        collapsed_state = particle.observe(context="background_monitoring")
                        particle.last_observed = asyncio.get_event_loop().time()
                        
                        self.log(f"Quantum collapse triggered for particle {particle.id}: {collapsed_state}", 
                                context="monitor_quantum_states")
                        
        except Exception as e:
            self.log(f"Quantum monitoring error: {e}", level="ERROR", context="monitor_quantum_states")


    async def field_monitor_loop(self):
        """Quantum field state monitoring and optimization"""
        while True:
            try:
                field_api = api.get_api("particle_field")
                system_metrics_api = api.get_api("system_metrics")
                
                if not field_api:
                    await asyncio.sleep(5.0)
                    continue
                
                # Get system state
                particles = field_api.get_all_particles()
                if system_metrics_api:
                    metrics = await system_metrics_api.get_system_metrics()
                else:
                    metrics = {}
                
                # Field optimization based on system load
                particle_count = len(particles)
                
                if particle_count > 150:  # MAX_PARTICLE_COUNT
                    self.log(f"High particle count detected: {particle_count}, triggering optimization", 
                            context="field_monitor_loop")
                    await field_api.prune_low_value_particles()
                
                # Monitor particle interactions and energy distribution
                total_energy = sum(p.energy for p in particles if hasattr(p, 'energy'))
                avg_energy = total_energy / max(particle_count, 1)
                
                if avg_energy < 0.1:
                    self.log("Low field energy detected, may need particle injection", 
                            level="DEBUG", context="field_monitor_loop")
                
                # Log field state periodically
                if self.subconscious_cycle_count % 500 == 0:
                    stats = field_api.get_particle_population_stats() if hasattr(field_api, 'get_particle_population_stats') else {}
                    self.log(f"Field state: {particle_count} particles, avg_energy: {avg_energy:.3f}, stats: {stats}", 
                            context="field_monitor_loop")
                
                await asyncio.sleep(2.0)  # Monitor every 2 seconds
                
            except Exception as e:
                self.log(f"Field monitoring error: {e}", level="ERROR", context="field_monitor_loop")
                await asyncio.sleep(10.0)

# Register the API
api.register_api("cognition_loop", CognitionLoop())

