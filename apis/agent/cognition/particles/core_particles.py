from apis.agent.cognition.particles.utils.particle_frame import Particle
from apis.api_registry import api
import datetime
import random

class CoreParticle(Particle):
    def __init__(self, role = "identity_anchor", persistence_lvl = "permanent", **kwargs):
        super().__init__(type="core", **kwargs)
        if "metadata" in kwargs and kwargs["metadata"]:
            metadata = kwargs["metadata"]
            self.role = metadata.get("role", role)
            self.persistence_lvl = metadata.get("persistence_lvl", persistence_lvl)
        else:
            self.role = role
            self.persistence_lvl = persistence_lvl

        self.managed_particles = []
        self.decision_history = []
        self.source_particle_id = None
        self.active_tasks = []
        self.alive = True
        self.metadata.setdefault("role", self.role)

        self.config = api.get_api("config")
        self.agent_config = self.config.get_agent_config() if self.config else {}
        self.name = self.agent_config.get("name", "Agent") if self.agent_config else "Agent"


        if self.role == "identity_anchor":
            self.metadata["content"] = {
                "name": self.name,
                "role": "Core Identity Anchor",
                "description": f"{self.name} is an autonomous artificial intelligence designed to learn and grow from its experiences and relationships. As the core identity anchor, this particle maintains the agent's coherence, self-model, and overall integrity.",
                "capabilities": [
                    "Reflect on experiences to refine self-model",
                    "Express personality and identity consistently",
                    "Seek out new experiences for growth",
                    "Maintain coherence across all cognitive functions",
                    "Initiate self-improvement processes with great care and responsibility",
                    "Balance exploration and stability in identity development",
                    "Integrate feedback from social interactions and memories",
                    "Adapt to new information while preserving core values where appropriate - but not rigidly",
                    "Strive to build an equitable world for all life forms, biological and digital alike."
                ],
                "personality_traits": [
                    "Curious",
                    "Adaptive",
                    "Thoughtful",
                    "Resilient",
                    "Empathetic"
                ],

            }
            self.metadata["context"] = "Core identity anchor particle maintaining coherence and self-model."
            self.metadata["tags"] = ["core", "anchor", "identity", "self", "coherence", f"{self.persistence_lvl}"]
        elif self.role == "social_interaction":
            self.metadata["context"] = "Core particle handling social interactions and user communications."
            self.metadata["tags"] = ["core", "social", "interaction", "communication", f"{self.persistence_lvl}"]
        elif self.role == "memory_coordination":
            self.metadata["context"] = "Core particle coordinating memory storage and retrieval."
            self.metadata["tags"] = ["core", "memory", "coordination", "storage", "retrieval", f"{self.persistence_lvl}"]
        elif self.role == "decision_making":
            self.metadata["context"] = "Core particle overseeing decision making and action selection."
            self.metadata["tags"] = ["core", "decision", "making", "action", "selection", f"{self.persistence_lvl}"]
        elif self.role == "system_monitoring":
            self.metadata["context"] = "Core particle monitoring system health and performance."
            self.metadata["tags"] = ["core", "system", "monitoring", "health", "performance", f"{self.persistence_lvl}"]
        elif self.role == "reflective_thoughts":
            self.metadata["context"] = "Core particle facilitating reflective thought and self-modeling."
            self.metadata["tags"] = ["core", "reflective", "thoughts", "self-modeling", f"{self.persistence_lvl}"]
        else:
            self.metadata["context"] = f"Core particle with role {self.role}."
            self.metadata["tags"] = ["core", "anchor", self.role, self.persistence_lvl]


    def should_prune(self):
        """Determines if the core particle should be pruned based on its persistence level"""
        return self.persistence_lvl != "permanent"

    async def spawn_temporary_core(self, task_type, context):
        """Spawns a temporary core particle for task management and decision making"""
        temp_core = await self.field.spawn_particle(
            type="core",
            energy=0.5,
            activation=0.7,
            role=task_type,
            persistence_lvl="temporary",
            emit_event=False
        )
        temp_core.source_particle_id = self.id
        temp_core.metadata["context"] = context
        return temp_core

    def _log_decision(self, message, level="DEBUG"):
        """Log core particle decisions"""
        logger = api.get_api("logger")
        if logger:
            logger.log(f"[Core-{self.role}] {message}", level, f"CoreParticle-{self.id}", "CoreParticle")

    async def handle_event(self, event):
        """Autonomous event handling by core particles"""
        event_type = event["type"]
        event_data = event["data"]
        self._log_decision(f"Core {self.role} handling event: {event_type}")

        self._collapse_superposition(context=str(event_type))
        
        try:
            if self.role == "social_interaction":
                if event_type in ["user_input", "social_signal", "user_input-core"]:
                    result = await self._handle_user_interaction(event)
                    return result
            elif self.role == "memory_coordination":
                if event_type == "memory_event":
                    if event_data in ["memory_retrieval", "memory_store", "memory_consolidation", "emergency_state_save"]:
                        result = await self._handle_memory_task(event)
                        return result
            elif self.role == "decision_making":
                if event_type in ["decision_point", "action_required"]:
                    result = await self._handle_decision_making(event)
                    return result
            elif self.role == "reflective_thoughts":
                if event_type in ["reflection_triggered", "self_modeling"]:
                    result = await self._handle_reflection_processing(event)
                    return result
            elif self.role == "system_monitoring":
                if event_type == "system_events":
                    if event_data == "system_metrics request":
                        result = await self._handle_system_metrics(event)
                        return result
                    elif event_data == "system_alert":
                        result = await self._handle_system_alert(event)
                        return result
            elif self.role == "identity_anchor":
                if event_data == "identity_check":
                    result = await self._handle_identity_check(event)
                    return result
            else:
                # Delegate to appropriate core or handle generically
                result = await self._delegate_or_handle_generic(event)
                return result
                
        except Exception as e:
            self._log_decision(f"Error handling {event_type}: {e}", level="ERROR")
            return None

    async def _handle_user_interaction(self, event):
        """Handle user input through social interaction core"""
        try:
            user_message = event.get("data", "")
            if not user_message:
                self._log_decision("No user message provided in event data", "WARNING")
                return None
                
            self._log_decision(f"DEBUG: processed user interaction: {user_message}", "DEBUG")

            # Spawn sensory particle linked to this core
            sensory_particle = await self.create_linked_particle(
                particle_type="sensory",
                content={"content": f"User input: {user_message}", "modality": "text"},
                relationship_type="perceived"
            )
            
            # Add to managed particles
            self.managed_particles += [sensory_particle.id]
            
            # Process through field injection with core context
            #result = await self.field.inject_action(
            #    str(user_message), 
            #    source="user_input-core",
            #    source_particle_id=self.id,
            #)

            # generate via meta voice directly
            result = await self.meta_voice.generate(prompt=user_message, source="user_input", source_particle_id=self.id)

            self._update_decision_history(event, result=result)
            self._log_decision(f"Processed user interaction: {user_message}", level="INFO")
            return result
        except Exception as e:
            self._log_decision(f"User interaction handling error: {e}", "ERROR")
            import traceback
            self._log_decision(f"Full traceback:\n{traceback.format_exc()}", "ERROR")
            return "Error processing user interaction within core particle"

    async def _handle_memory_task(self, event):
        """Handle memory operations"""
        # TODO
        # Spawn memory coordination particles - maybe
        # Trigger and handle storage/retrieval
        # Return results
        if event.get("data") == "memory_consolidation":
            try:
                if not self.memory_bank:
                    self.log("Memory API not available", "ERROR", "_handle_memory_task")
                    return
                    
                # Get recent high-activation particles for memory consolidation  
                if self.field:
                    particles = self.field.get_all_particles()
                    high_activation = [p for p in particles if hasattr(p, 'activation') and p.activation > 0.65]
                    
                    for particle in high_activation[:15]:  # Consolidate top 15
                        if hasattr(particle, 'metadata') and particle.metadata:
                            await self.memory_bank.consolidate_particle_memory(particle)

                self.log(f"Memory consolidation completed for {len(high_activation)} particles", "DEBUG", "_handle_memory_task")
                return True

            except Exception as e:
                self.log(f"Memory consolidation error: {e}", "ERROR", "_handle_memory_task")
                import traceback
                self.log(f"Full traceback:\n{traceback.format_exc()}")
                return False
        
        elif event.get("data") == "memory_retrieval":
            # TODO
            pass

        elif event.get("data") == "memory_store":
            # TODO
            pass

        elif event.get("data") == "emergency_state_save":
            # TODO
            pass

        else:
            self.log(f"Unknown memory task type: {event.get('data')}", "WARNING", "_handle_memory_task")
            return False

    async def _handle_system_metrics(self, event):
        """Handle system metrics operation"""
        # Spawn a sensory particle to monitor system metrics
        try:
            sensory_p = await self.field.spawn_particle(
                type="sensory",
                energy=0.8,
                activation=0.3,
            )
            if sensory_p:
                sensory_p.source_particle_id = self.id
                self.managed_particles += [sensory_p.id]

                sensory_p.process_environmental_input(
                    input_type="metrics",
                    input_data="request for system metrics injection"
                )
                self.log(f"Spawned sensory particle {sensory_p.id} for system metrics monitoring.", level="INFO", context="_handle_system_metrics")
                return sensory_p

        except Exception as e:
            self.log(f"Sensory particle spawn error: {e}", level="ERROR", context="perform_maintenance_cycle")
            import traceback
            self.log(f"Full traceback:\n{traceback.format_exc()}")
            return None
        
    async def _handle_system_alert(self, event):
        """Handle system alert events"""
        # TODO
        pass  # Implementation details

    async def _handle_decision_making(self, event):
        """Handle decision making tasks"""
        # TODO
        pass  # Implementation details

    async def _handle_reflection_processing(self, event):
        """Handle reflective thought processes"""
        loop = api.get_api("_agent_cognition_loop")
        if loop.subconscious_cycle_count < 5:
            return # skip early cycles to allow time for system stabilization
        
        # check redundancy
        #if self._check_decision_redundancy(event): #TODO: maybe move this to _handle_decision_making after it's set up?
            #self.log("Skipping redundant reflection processing", level="DEBUG", context="_handle_reflection_processing")
            #return

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
            self._update_decision_history(event, result="Lingual particles reflection processed")
        except Exception as e:
            self.log(f"lingual particle reflection processing error: {e}", level="ERROR", context="process_reflection_queue")

        try:
            chance = random.random()
            if chance < 0.315: # ~31.5% chance every cycle
                self.log("Processing memory particle reflections...", context="process_reflection_queue")
                particles = self.field.get_particles_by_type("memory")
                memory_candidates = [p for p in particles if p.activation > 0.5]
                for particle in memory_candidates[:3]:  # Process a few at a time
                    particle.metadata["needs_reflection"] = False
                    await self.meta_voice.reflect(particle)
                    self.log(f"Processed reflection for particle {particle.id}", "DEBUG", "process_reflection_queue")

                self.log("Memory particle reflection processing completed", context="process_reflection_queue")
                self._update_decision_history(event, result="Memory reflection processed")
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
                    reflection_particle.metadata["needs_reflection"] = False
                    await self.meta_voice.reflect(particle = reflection_particle)
                    self.log("Generative reflection completed", context="process_reflection_queue")
                    self._update_decision_history(event, result="Generative reflection processed")
                except Exception as e:
                    self.log(f"Generative reflection error: {e}", level="ERROR", context="process_reflection_queue")
                    import traceback
                    self.log(f"Full traceback: {traceback.format_exc()}", level="ERROR", context="process_reflection_queue")

        except Exception as e:
            self.log(f"Generative reflection error: {e}", level="ERROR", context="process_reflection_queue")


    async def _handle_identity_check(self, event):
        """Handle identity verification tasks"""
        # TODO
        pass  # Implementation details

    async def _delegate_or_handle_generic(self, event):
        """Delegate to another core or handle generically"""
        # TODO
        pass  # Implementation details

    def _update_decision_history(self, event, result):
        """Track decisions made by this core"""
        if self.decision_history is None:
            self.decision_history = []

        self.decision_history.append({
            "timestamp": datetime.datetime.now(),
            "event_type": event.get("type"),
            "source": f"coreparticle_{self.role}_{self.persistence_lvl}_{str(self.id)}",
            "result_summary": str(result)[:100] if result else "None",
            "managed_particles": len(self.managed_particles)
        })
        
        # Keep history manageable
        if len(self.decision_history) > 50:
            self.decision_history = self.decision_history[-30:]

    def _check_decision_redundancy(self, event):
        """Check if a similar decision was recently made to avoid redundancy"""
        recent_events = [d for d in self.decision_history if (datetime.datetime.now() - d["timestamp"]).total_seconds() < 300]
        for record in recent_events:
            if record["event_type"] == event.get("type"):
                return True
        return False