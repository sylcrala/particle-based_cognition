"""
Particle-based Cognition Engine - runtime loops, defines conscious and subconscious loops
Copyright (C) 2025 sylcrala

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version, subject to the additional terms 
specified in TERMS.md.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License and TERMS.md for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Additional terms apply per TERMS.md. See also ETHICS.md.
"""

import asyncio
import random
from apis.api_registry import api



class CognitionLoop:
    
    def __init__(self, events = None, field = None, adaptive_engine = None, memory = None, voice = None, lexicon = None, agent_categorizer = None):

        self.logger = api.get_api("logger")
        self.events = events
        self.field = field
        self.adaptive_engine = adaptive_engine
        self.memory = memory
        self.agent_categorizer = agent_categorizer
        self.meta_voice = voice
        self.lexicon_store = lexicon

        try:
            self.config = api.get_api("config")
            self.agent_config = self.config.get_agent_config()
            self.log("Config loaded", context="CognitionLoop.__init__")
        except Exception as e:
            self.log(f"Config load error: {e}", level="ERROR", context="CognitionLoop.__init__")
            raise e

        self.name = self.agent_config.get("name")

        self.conscious_active = False
        self.cycle_count = 0
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
            
            await self.events.emit_event("system_events", "system_metrics request", source="perform_maintenance_cycle")

            # TODO: add linguistic development here, process definition parsing during downtime

            # consolidate memories
            await self.events.emit_event("memory_event", "memory_consolidation", source="perform_maintenance_cycle")

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

            api.handle_shutdown()

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
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.log(f"Conscious loop error: {e}", level="ERROR", context="conscious_loop")
                await asyncio.sleep(1.0)
                

    async def process_active_cognition(self):
        """Handle active cognitive processing during conscious state"""
        try:
                
            # Get high-activation particles for conscious processing
            particles = self.field.get_all_particles()
            active_particles = [p for p in particles if p.id in self.field.alive_particles and p.activation > 0.6]
            
            # Set attention region
            if active_particles:
                self.field.set_conscious_attention(particles=[active_particles[:10]])


            # Process active particles (conscious attention)
            for particle in active_particles[:10]:  # Focus on top 10
                if hasattr(particle, 'observe'):
                    # Conscious observation collapses quantum states
                    collapsed_state = particle.observe(context="conscious_attention")
                    self.log(f"Conscious attention collapsed particle {particle.id} to {collapsed_state} state", 
                            level="DEBUG", context="process_active_cognition")

                    if collapsed_state == 'certain':
                        self.log(f"Conscious attention collapsed particle {particle.id} to certain state", 
                                level="DEBUG", context="process_active_cognition")
                        
                if particle.type == "lingual" and hasattr(self.lexicon_store, 'learn_from_particle'):
                    await self.lexicon_store.learn_from_particle(particle)
                    
                    # Real-time semantic token observation for active content
                    if (hasattr(self, 'agent_categorizer') and 
                        self.agent_categorizer and 
                        hasattr(self.agent_categorizer, 'background_processor')):
                        
                        # Extract meaningful tokens from particle content for immediate analysis
                        particle_content = getattr(particle, 'metadata', {}).get('token', '')
                        if particle_content:
                            context = {
                                "source": "conscious_attention",
                                "particle_id": particle.id,
                                "activation": particle.activation
                            }
                            await self.agent_categorizer.background_processor.observe_compressed_token(
                                particle_content, context
                            )


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
                chance = random.random()
                
                # Log cycle progress every 5 cycles for monitoring
                if self.subconscious_cycle_count % 5 == 0:
                    self.log(f"Subconscious cycle {self.subconscious_cycle_count} completed", level="DEBUG", context="subconscious_loop")

                if self.field:
                    if self.field._update_active:
                        # update particles with batching to prevent blocking
                        await self.field.update_particles()
                        await asyncio.sleep(0.05)  

                    # Trigger quantum monitoring every 8 cycles 
                    if self.subconscious_cycle_count % 8 == 0:
                        await self.monitor_quantum_states(
                            use_spatial_selection=True,
                            max_particles=150  
                        )
                        await asyncio.sleep(0.02)  

                # Perform maintenance every 10 cycles 
                if self.subconscious_cycle_count % 10 == 0:
                    #self.log(f"Performing subconscious maintenance cycle {self.subconscious_cycle_count}", context="subconscious_loop")
                    await self.perform_maintenance_cycle()
                    await asyncio.sleep(0)

                # Process any pending reflections every 20 cycles or ~8% chance
                if self.subconscious_cycle_count % 20 == 0 or chance < 0.08:
                    #self.log(f"Processing subconscious reflections for cycle {self.subconscious_cycle_count}", context="subconscious_loop")
                    await self.events.emit_event("reflection_triggered", "Request to process particle reflections", source="subconscious_loop")
                    await asyncio.sleep(0)

                # Trigger autonomous reasoning every 15 cycles
                if self.subconscious_cycle_count % 15 == 0:
                    #self.log(f"Triggering autonomous reasoning cycle {self.subconscious_cycle_count}", context="subconscious_loop")
                    await self.events.emit_event("reasoning_cycle", "Autonomous reasoning and inference", source="subconscious_loop")
                    await asyncio.sleep(0)

                ## semantic gravity processing
                # maintenance every 40 cycles (reduced frequency)
                if self.subconscious_cycle_count % 40 == 0:
                    if (hasattr(self, 'agent_categorizer') and 
                        self.agent_categorizer and 
                        hasattr(self.agent_categorizer, 'background_processor')):
                        await self.agent_categorizer.background_processor._subconscious_maintenance_processing()
                        await asyncio.sleep(0.02)  # Yield after heavy operation

                # full semantic analysis queue every 50 cycles (reduced frequency)
                if self.subconscious_cycle_count % 50 == 0:
                    if (hasattr(self, 'agent_categorizer') and 
                        self.agent_categorizer and 
                        hasattr(self.agent_categorizer, 'background_processor')):
                        await self.agent_categorizer.background_processor._process_analysis_queue()
                        await asyncio.sleep(0.02)  # Yield after heavy operation

                ## lexicon batch processing
                # process pending lexicon batch every 50 cycles (reduced frequency)
                if self.subconscious_cycle_count % 50 == 0:
                    if hasattr(self.lexicon_store, '_flush_pending_batch'):
                        await self.lexicon_store._flush_pending_batch()
                        await asyncio.sleep(0.02)  # Yield after batch operation

                ## inference engine
                # causal inference processing every 60 cycles (reduced frequency)
                if self.subconscious_cycle_count % 60 == 0:
                    try:
                        inference_engine = api.get_api("_agent_inference_engine")
                        if inference_engine and hasattr(inference_engine, 'infer_causal_relationships'):
                            await inference_engine.infer_causal_relationships()
                            await asyncio.sleep(0.02)  # Yield after inference
                    except Exception as e:
                        self.log(f"Inference processing error: {e}", "ERROR", "subconscious_loop")

                # Log and save field state every 50 cycles
                if self.subconscious_cycle_count % 50 == 0:
                    self.log(f"Saving field state on cycle {self.subconscious_cycle_count}", 
                            context="subconscious_loop")
                    try:
                        self.memory.emergency_save() # FIXME: change to emit_event call for core particle
                        self.log("Field state save completed", context="subconscious_loop")
                    except Exception as e:
                        self.log(f"Field state save error: {e}", level="ERROR", context="subconscious_loop")
                        import traceback
                        self.log(f"Full traceback:\n{traceback.format_exc()}", level="ERROR", context="subconscious_loop")

                    stats = self.field.get_particle_population_stats() if hasattr(self.field, 'get_particle_population_stats') else {}
                    self.log(f"Field particle population stats: {stats}", 
                            context="field_monitor_loop")

                # Sleep to prevent excessive CPU usage
                await asyncio.sleep(0.25)  # Adjust timing as needed
                
            except Exception as e:
                self.log(f"Subconscious loop error: {e}", level="ERROR", context="subconscious_loop")
                await asyncio.sleep(1.0)  # Longer sleep on error


    async def process_reflection_queue(self):
        """Process reflection queue for particle learning - DEPRECATED"""
        if self.subconscious_cycle_count < 5:
            return # skip early cycles to allow time for system stabilization

        self.log("Processing reflection queue...", context="process_reflection_queue")

        try:
            chance = random.random()
            if chance < 0.315: # ~31.5% chance every cycle
                self.log("Processing lingual particle reflections...", context="process_reflection_queue")
                particles = self.field.get_particles_by_type("lingual")
                reflection_candidates = [p for p in particles if p.metadata.get("needs_reflection", False) and p.activation > 0.5]
                
                for particle in reflection_candidates[:3]:  # Process a few at a time
                    if hasattr(particle, 'learn_from_particle'):
                        await particle.learn_from_particle(particle)
                        await self.meta_voice.reflect(particle)
                        particle.metadata["needs_reflection"] = False
                        self.log(f"Processed reflection for particle {particle.id}", "DEBUG", "process_reflection_queue")       
        
            self.log("lingual particle reflection processing completed", context="process_reflection_queue")
        except Exception as e:
            self.log(f"lingual particle reflection processing error: {e}", level="ERROR", context="process_reflection_queue")

        try:
            chance = random.random()
            if chance < 0.315: # ~31.5% chance every cycle
                self.log("Processing memory particle reflections...", context="process_reflection_queue")
                particles = self.field.get_particles_by_type("memory")
                memory_candidates = [p for p in particles if p.activation > 0.5]
                for particle in memory_candidates[:3]:  # Process a few at a time
                    await self.meta_voice.reflect(particle)
                    self.log(f"Processed reflection for particle {particle.id}", "DEBUG", "process_reflection_queue")

                self.log("Memory particle reflection processing completed", context="process_reflection_queue")
        except Exception as e:
            self.log(f"Random memory consolidation error: {e}", level="ERROR", context="process_reflection_queue")

        try:
            chance = random.random()
            if chance < 0.1575: # ~15.75% chance every cycle
                try:
                    self.log("Processing generative reflection...", context="process_reflection_queue")
                    particle_list = self.field.get_all_particles()
                    alive_particles = [p for p in particle_list if p.id in self.field.alive_particles]
                    chosen_particle = random.choice(alive_particles)
                    reflection_particle = chosen_particle
                    await self.meta_voice.reflect(particle = reflection_particle)
                    self.log("Generative reflection completed", context="process_reflection_queue")
                except Exception as e:
                    self.log(f"Generative reflection error: {e}", level="ERROR", context="process_reflection_queue")
                    import traceback
                    self.log(f"Full traceback: {traceback.format_exc()}", level="ERROR", context="process_reflection_queue")

        except Exception as e:
            self.log(f"Generative reflection error: {e}", level="ERROR", context="process_reflection_queue")



    async def monitor_quantum_states(self, use_spatial_selection = True, max_particles = 250):
        """Monitor particle superposition states and trigger collapses"""
        try:
            if use_spatial_selection and self.field:
                conscious_region = self.field.get_conscious_attention_region()
                background_sectors = self.field.rotate_monitoring_sectors()

                particles = self.field.get_particles_in_region(
                    primary_region = conscious_region,
                    secondary_regions = background_sectors,
                    max_count = max_particles
                )

                self.log(f"Spatial quantum monitoring: {type(particles)}, value: {particles}", "DEBUG", context="monitor_quantum_states")

            else:
                particles = self.field.particles
                self.log(f"Full field quantum monitoring: {type(particles)}, value: {particles}", "DEBUG", context="monitor_quantum_states")
                
            for particle in particles:
                if not hasattr(particle, "id"):
                    continue  
                
                                
                # Check if particle has the necessary quantum properties
                if hasattr(particle, 'superposition') and hasattr(particle, 'observe'):
                    try:
                        # Safer observation pattern with type checking
                        if isinstance(particle.superposition, dict) and "certain" in particle.superposition:
                            # Dictionary with string keys
                            collapsed_state = particle.observe(context="background_monitoring")
                            particle.last_observed = asyncio.get_event_loop().time()
                            self.log(f"Quantum collapse triggered for particle {particle.id}: {collapsed_state}", 
                                    context="monitor_quantum_states")
                        elif isinstance(particle.superposition, dict) and 0 in particle.superposition:
                            # Dictionary with integer keys - adapt the observe method
                            self.log(f"Integer-keyed superposition detected for {particle.id}", 
                                "DEBUG", context="monitor_quantum_states")
                            # Handle integer-keyed superposition differently if needed
                        elif isinstance(particle.superposition, str):
                            # Handle string superposition
                            self.log(f"String superposition detected for {particle.id}: {particle.superposition}", 
                                "DEBUG", context="monitor_quantum_states")
                            # May need to convert or initialize properly
                        else:
                            # Unknown superposition format
                            self.log(f"Unknown superposition format for {particle.id}: {type(particle.superposition)}", 
                                    "WARNING", context="monitor_quantum_states")
                            
                    except Exception as e:
                        self.log(f"Observation error for particle {particle.id}: {e}", level="ERROR", context="monitor_quantum_states")
                        # Could initialize a proper superposition here if needed
                        

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
            import traceback
            self.log(f"Quantum monitoring error: {e}", level="ERROR", context="monitor_quantum_states")
            self.log(traceback.format_exc(), level="ERROR", context="monitor_quantum_states")


    async def field_monitor_loop(self):
        """Quantum field state monitoring and optimization - DEPCRECATED"""
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

                await asyncio.sleep(3.0)  # Monitor every 3 seconds

            except Exception as e:
                self.log(f"Field monitoring error: {e}", level="ERROR", context="field_monitor_loop")
                await asyncio.sleep(10.0)




    async def consolidate_memories(self):
        """Consolidate recent memories and update long-term storage - DEPRECATED"""
        try:
            if not self.memory:
                return
                
            # Get recent high-activation particles for memory consolidation  
            if self.field:
                particles = self.field.get_all_particles()
                high_activation = [p for p in particles if hasattr(p, 'activation') and p.activation > 0.65]
                
                for particle in high_activation[:15]:  # Consolidate top 15
                    if hasattr(particle, 'metadata') and particle.metadata:
                        await self.memory.consolidate_particle_memory(particle)

            self.log(f"Memory consolidation completed for {len(high_activation)} particles", "DEBUG", "consolidate_memories")

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