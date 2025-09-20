"""
handles agent "voice" generation (text, other modes not yet implemented) - uses the llm and lingual particles for personalized generation (internally and externally)
"""
import uuid
import random
from time import time
import numpy as np
import json
from apis.api_registry import api
from datetime import datetime as dt

INTERNAL_SOURCE = "internal_dialogue"
EXTERNAL_SOURCE = "external_dialogue"


class MetaVoice:
    def __init__(self, field = None, memory = None, lexicon = None, model_handler = None):
        self.logger = api.get_api("logger")

        self.field = field
        self.memory_bank = memory
        self.lexicon_store = lexicon
        self.model_handler = model_handler

        self.chat_history = []
        self.thoughts = []

    def log(self, message):
        self.logger.log(message, "INFO", "MetaVoice", "MetaVoice")

    async def generate(self, prompt: str, source: str, max_tokens=1600, temperature=0.7, top_p=0.95, context_particles=None, context_id = None, tags = None) -> str:
        """Generate response using model handler and quantum-aware particle context"""

        if not self.model_handler:
            return "[Error: Model handler not available]"
            
        input_source = str(source)

        gentags = [tag for tag in (tags or []) if isinstance(tag, str)] + ["user_input" if source == "user_input" else "internal_thought"]

        # Create input particle and trigger field-level quantum effects
        if input_source == "user_input":
            input_particle = await self.process_input(prompt, input_source)
            
            # Trigger contextual collapse for user interactions
            if self.field and input_particle:
                collapse_log = await self.field.trigger_contextual_collapse(
                    input_particle, 
                    "user_interaction",
                    cascade_radius=0.7
                )
                self.log(f"User interaction triggered {len(collapse_log)} particle collapses")

        # Score context particles with quantum awareness
        quantum_context = self.score_quantum_context_particles(prompt, context_particles or [])

        # Generate response via model handler
        try:
            response = await self.model_handler.generate(
                str(prompt), 
                context_id=str(uuid.uuid4()) or context_id, 
                tags=gentags,
                max_new_tokens=max_tokens,
                source=input_source
            )
            
            # Create response particle linked to input
            if self.field and input_particle:
                response_particle = await input_particle.create_linked_particle(
                    particle_type="lingual",
                    content=response,
                    relationship_type="response_generation"
                )
                
                # Response should be relatively certain
                if response_particle and hasattr(response_particle, 'update_superposition'):
                    response_particle.update_superposition(certainty_delta=0.3)

            mem_entry = {
                "input": prompt,
                "response": response,
                "context": quantum_context
            }
            
            if source == "user_input":
                self.chat_history.append({"Misty": response, "timestamp": dt.now().timestamp()})
            else:
                self.thoughts.append({"thought": response, "timestamp": dt.now().timestamp()})
            
            await self.memory_bank.update("conversation", mem_entry, [input_particle.id, response_particle.id], "voice generation", tags=gentags)

            return response
        except Exception as e:
            self.log(f"Generation error: {e}")
            return "[Error: Generation failed]"

    def score_quantum_context_particles(self, input_text, context_particles, top_k=4):
        """Score context particles considering their quantum states"""
        if not context_particles:
            return []
            
        scored_particles = []
        
        for particle in context_particles:
            base_score = 1.0
            
            # Quantum state influence on scoring
            if hasattr(particle, 'superposition'):
                certainty = particle.superposition.get('certain', 0.5)
                # Certain particles get higher base score
                base_score *= (0.5 + certainty * 0.5)
                
            # Linkage influence
            if hasattr(particle, 'linked_particles'):
                linkage_count = len(particle.linked_particles.get('children', []))
                base_score *= (1.0 + linkage_count * 0.1)
                
            # Distance/relevance (existing logic)
            if hasattr(particle, 'get_embedding') and particle.get_embedding():
                input_embedding = api.call_api("model_handler", "embed_text", (input_text,))
                particle_embedding = particle.get_embedding()
                if input_embedding and particle_embedding:
                    distance = np.linalg.norm(np.array(input_embedding) - np.array(particle_embedding))
                    relevance_score = 1 / (1 + distance)  # Closer means more relevant
                    base_score *= relevance_score            
            
            scored_particles.append((base_score, particle))
            
        # Sort by score and return top_k
        scored_particles.sort(key=lambda x: x[0], reverse=True)
        return [particle for _, particle in scored_particles[:top_k]]

    async def process_input(self, text, source=None):
        """Process input text and create lingual particles"""
        try:

            # Create lingual particle for input processing
            await self.field.spawn_particle(
                type="lingual",
                metadata={
                    "content": text,
                    "source": source or "unknown",
                    "timestamp": time(),
                    "processing_type": "input"
                },
                energy=0.7,
                activation=0.8,
                emit_event=True
            )

            self.chat_history.append({"Tony": text, "timestamp": dt.now().timestamp()})
        except Exception as e:
            self.log(f"Input processing error: {e}")

    async def spawn_and_learn_token(self, token, source=None):
        """Create and learn from token via particle field"""
        try:

            if self.field:
                # Spawn lingual particle for token
                lp = await self.field.spawn_particle(
                    type="lingual",
                    metadata={
                        "token": token,
                        "source": source or "unknown",
                        "timestamp": time()
                    },
                    energy=0.5,
                    activation=0.6,
                    emit_event=True
                )
                
                # Learn token if lexicon available
                if self.lexicon_store and hasattr(lp, 'learn'):
                    await lp.learn(token=token)
                    
        except Exception as e:
            self.log(f"Token learning error: {e}")

    async def reflect(self, particle=None):
        """Create reflection particle"""
        try:
            context_id=str(uuid.uuid4())
            if not self.field or not self.model_handler:
                return
            
            if particle is None:
                chance = random.randint(0, 100) / 100

                if chance < 0.4:
                    return  # 40% chance to skip reflection

                elif chance >= 0.4 and chance < 0.5:
                    # Generate reflection
                    particle = await self.process_input("how i am.", source = "internal_reflection")
                    reflection_prompt = f"I'm thinking about {str(particle.get_content())}"
                    self.log(f"Reflecting on particle {particle.id}: {particle.get_content()}")


                elif chance >= 0.5 and chance < 0.6:
                    if self.lexicon_store:
                        words = self.lexicon_store.get_terms(top_n=3)
                        reflection_prompt = f"Considering some concepts: {', '.join(words)}"
                        self.log(f"Reflecting on lexicon concepts: {', '.join(words)}")

                elif chance >= 0.6 and chance < 0.8:
                    memory = self.memory_bank.get_random_memory()
                    if memory:
                        reflection_prompt = f"Recalling memory: {memory}"
                        self.log(f"Reflecting on memory: {memory}")

                elif chance >= 0.8:
                    particles = self.field.get_all_particles()
                    particle_positions = str([p.position for p in particles if hasattr(p, 'position')])
                    if particle_positions:
                        reflection_prompt = f"Thinking about my inner state and connections: {particle_positions}"
                        self.log(f"Reflected on field contents, all particle positions")
                    else:
                        reflection_prompt = "Reflecting on my current state."
                        self.log("No particle positions available for reflection OR unable to parse")
            
            else:
                reflection_prompt = f"[Reflection] Reflecting on: {particle.get_content()}"
                self.log(f"Reflecting on particle {particle.id}: {particle.get_content()}")

            reflection_response = await self.generate(
                    reflection_prompt,
                    context_id=context_id,
                    tags=["reflection", "internal"],
                    source="internal_reflection"
                )

            # Create reflection particle
            response_particle = await self.field.spawn_particle(
                type="lingual",
                metadata={
                    "content": reflection_response,
                    "source": "reflection",
                    "reflection_target": particle.id,
                    "timestamp": time()
                },
                energy=0.6,
                activation=0.5,
                source_particle_id=particle.id,
                emit_event=True
            )

            await self.memory_bank.update(
                    key=f"reflection:{uuid.uuid4()}",
                    value=reflection_response,
                    links=[(particle.id), (response_particle.id)] if particle else [(response_particle.id)],
                    tags=["reflection", "internal"],
                    source="internal_reflection",
                    source_particle_id=response_particle.id
                )
            
            
            
                
        except Exception as e:
            self.log(f"Reflection error: {e}")

