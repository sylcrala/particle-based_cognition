"""
agent runtime loop (main async internal loop | essentially "run_agent()" from runtime.py in old format)
"""

import asyncio
from apis.api_registry import api



class CognitionLoop:
    
    def __init__(self, events = None, field = None, adaptive_engine = None, memory = None, voice = None, lexicon = None):

        self.logger = api.get_api("logger")
        self.events = events
        self.field = field
        self.adaptive_engine = adaptive_engine
        self.memory = memory
        self.meta_voice = voice
        self.lexicon_store = lexicon
        self.system_metrics_api = api.get_api("system_metrics")

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

    async def perform_maintenance_cycle(self):
        """Memory consolidation, reflection, and particle pruning"""
        try:

            if not self.field:
                return
                
            # Get recent particles for reflection
            particles = list(self.field.get_all_particles())  
                
            # Reflect on memory particles
            for particle in particles:
                if particle.type == "memory":
                    await self.meta_voice.reflect(particle)
                    await particle.reflect()

                if particle.type == "lingual":
                    await self.lexicon_store.learn_from_particle(particle)


            # Prune low-value particles
            await self.field.prune_low_value_particles()

            self.log("Maintenance cycle completed", context="perform_maintenance_cycle")
            
        except Exception as e:
            self.log(f"Maintenance cycle error: {e}", level="ERROR", context="perform_maintenance_cycle")

    async def shutdown_sequence(self):
        """Clean shutdown with data preservation"""
        try:
            self.log("SYS-shutdown: Initiating shutdown sequence...", context="shutdown_sequence")
            
            
            # Save critical data 
            if hasattr(self.memory, 'save_to_file'):
                await self.memory.save_to_file()
            if hasattr(self.lexicon_store, 'save'):
                await self.lexicon_store.save()

            self.field.stop_particle_updates()
            self.conscious_active = False

            self.log("Shutdown complete. Session ended.", context="shutdown_sequence")
            
            # Cancel remaining tasks
            for task in asyncio.all_tasks():
                if not task.done():
                    task.cancel()
                    
        except Exception as e:
            self.log(f"Shutdown error: {e}", level="ERROR", context="shutdown_sequence")



    async def conscious_loop(self):
        """Handles active interactions and processing"""
        while self.conscious_active:
            try:
                
                # Get event handler for user interactions
                if not self.events:
                    await asyncio.sleep(1.0)
                    continue
                
                # Active memory retrieval and processing
                await self.process_active_cognition()
                
                # Brief rest between conscious cycles
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.log(f"Conscious loop error: {e}", level="ERROR", context="conscious_loop")
                await asyncio.sleep(1.0)


    async def process_active_cognition(self):
        """Handle active cognitive processing during conscious state"""
        try:
                
            # Get high-activation particles for conscious processing
            particles = self.field.get_all_particles()
            active_particles = [p for p in particles if hasattr(p, 'activation') and p.activation > 0.7]
            
            # Process active particles (conscious attention)
            for particle in active_particles[:5]:  # Focus on top 5
                if hasattr(particle, 'observe'):
                    # Conscious observation collapses quantum states
                    collapsed_state = particle.observe(context="conscious_attention")
                    self.log(f"Conscious attention collapsed particle {particle.id} to {collapsed_state} state", 
                            level="DEBUG", context="process_active_cognition")

                    if collapsed_state == 'certain':
                        self.log(f"Conscious attention collapsed particle {particle.id} to certain state", 
                                level="DEBUG", context="process_active_cognition")
                        
                        
        except Exception as e:
            self.log(f"Active cognition error: {e}", level="ERROR", context="process_active_cognition")

    def get_chat_history(self):
        return self.meta_voice.chat_history


    async def subconscious_loop(self):
        """Handles background processing and cognitive tasks"""
        while self.conscious_active:
            try:
                # Increment cycle counter
                self.subconscious_cycle_count += 1

                # Perform maintenance every 10 cycles (adjustable)
                if self.subconscious_cycle_count % 10 == 0:
                    self.log(f"Performing subconscious maintenance cycle {self.subconscious_cycle_count}", 
                            context="subconscious_loop")
                    await self.perform_maintenance_cycle()
                
                # Process any pending reflections
                await self.process_reflection_queue()
                
                # Background memory consolidation
                if self.subconscious_cycle_count % 20 == 0:
                    self.consolidate_memories() 

                
                # Sleep to prevent excessive CPU usage
                await asyncio.sleep(2.0)  # Adjust timing as needed
                
            except Exception as e:
                self.log(f"Subconscious loop error: {e}", level="ERROR", context="subconscious_loop")
                await asyncio.sleep(5.0)  # Longer sleep on error


    async def process_reflection_queue(self):
        """Process reflection queue for particle learning"""
        try:


            # Get particles that need reflection processing
            particles = self.field.get_particles_by_type("lingual")
            reflection_candidates = [p for p in particles if p.metadata.get("needs_reflection", False)]
            
            for particle in reflection_candidates[:3]:  # Process a few at a time
                if hasattr(particle, 'learn_from_particle'):
                    await particle.learn_from_particle(particle)
                    particle.metadata["needs_reflection"] = False
                    
        except Exception as e:
            self.log(f"Particle reflection processing error: {e}", level="ERROR", context="process_reflection_queue")

        try:
            self.log("Processing generative reflection...", context="process_reflection_queue")
            await self.meta_voice.reflect()
            self.log("Generative reflection completed", context="process_reflection_queue")

        except Exception as e:
            self.log(f"Generative reflection error: {e}", level="ERROR", context="process_reflection_queue")



    async def monitor_quantum_states(self):
        """Monitor particle superposition states and trigger collapses"""
        try:
            particles = self.field.get_all_particles()
            self.log(f"Field returned type: {type(particles)}, value: {particles}", "DEBUG", context="monitor_quantum_states")
            
            for particle in particles:
                if not isinstance(particle, (list, tuple)) or isinstance(particle, str):
                    self.log(f"Invalid particle detected during quantum monitoring: {particle.id}", level="ERROR", context="monitor_quantum_states")
                    continue  # Stop processing if particles are not valid
                

            particle_count = len(particles)
            self.log(f"Monitoring {particle_count} particles for quantum state evaluation", level="DEBUG", context="monitor_quantum_states")
            if particle_count == 0:
                self.log("No particles to monitor", level="DEBUG", context="monitor_quantum_states")
                return

            # Check for particles that need quantum state evaluation
            for particle in particles:
                if hasattr(particle, 'superposition') and hasattr(particle, 'observe'):
                    # Simple collapse trigger: if particle hasn't been observed recently
                    if not hasattr(particle, 'last_observed'):
                        particle.last_observed = 0
                        particle.superposition = "collapsed"
                        
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
        while self.conscious_active:
            try:                
                # Field-level quantum state monitoring
                await self.monitor_quantum_states()

                # Get all particles
                particles = self.field.get_all_particles()
                particle_count = len(particles)
                
                # Monitor particle interactions and energy distribution
                total_energy = sum(p.energy for p in particles if hasattr(p, 'energy'))
                if particle_count == 0:
                    avg_energy = 0
                else:
                    avg_energy = total_energy / particle_count
                
                if avg_energy < 0.1:
                    self.log("Low field energy detected, may need particle injection", 
                            level="DEBUG", context="field_monitor_loop")
                
                # Log field state periodically
                if self.subconscious_cycle_count % 500 == 0:
                    stats = self.field.get_particle_population_stats() if hasattr(self.field, 'get_particle_population_stats') else {}
                    self.log(f"Field state: {particle_count} particles, avg_energy: {avg_energy:.3f}, stats: {stats}", 
                            context="field_monitor_loop")
                
                await asyncio.sleep(2.0)  # Monitor every 2 seconds
                
            except Exception as e:
                self.log(f"Field monitoring error: {e}", level="ERROR", context="field_monitor_loop")
                await asyncio.sleep(10.0)

    async def consolidate_memories(self):
        """Consolidate recent memories and update long-term storage"""
        try:
            if not self.memory:
                return
                
            # Get recent high-activation particles for memory consolidation  
            if self.field:
                particles = self.field.get_all_particles()
                high_activation = [p for p in particles if hasattr(p, 'activation') and p.activation > 0.8]
                
                for particle in high_activation[:3]:  # Consolidate top 3
                    if hasattr(particle, 'metadata') and particle.metadata:
                        await self.memory.consolidate_particle_memory(particle)
            
            self.log("Memory consolidation completed", "DEBUG", "consolidate_memories")
            
        except Exception as e:
            self.log(f"Memory consolidation error: {e}", "ERROR", "consolidate_memories")


"""
if __name__ == "__main__":
    import asyncio

    loop = CognitionLoop()
    try:
        asyncio.run(loop.run())
    except KeyboardInterrupt:
        print("Cognition loop shutdown requested.")
        asyncio.run(loop.shutdown_sequence())
    except Exception as e:
        print(f"Cognition loop error: {e}")
"""