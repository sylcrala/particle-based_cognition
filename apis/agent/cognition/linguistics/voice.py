"""
handles agent "voice" generation (text, other modes not yet implemented) - uses the llm and lingual particles for personalized generation (internally and externally)
"""
import uuid
import random
from time import time
import numpy as np

from apis.api_registry import api

INTERNAL_SOURCE = "internal_dialogue"
EXTERNAL_SOURCE = "external_dialogue"


class MetaVoice:
    def __init__(self):
        self.logger = api.get_api("logger")
        self.model_handler = api.get_api("model_handler")
        self.field_api = api.get_api("particle_field")
        self.memory_bank = api.get_api("memory_bank")
        self.lexicon_store = api.get_api("lexicon_store")

    def log(self, message):
        self.logger.log(message, "INFO", "MetaVoice", "MetaVoice")

    async def generate(self, prompt: str, source: str, max_tokens=800, temperature=0.7, top_p=0.95, context_particles=None) -> str:
        """Generate response using model handler and quantum-aware particle context"""
        
        # Get model handler from API
        model_handler = self.model_handler
        field_api = self.field_api

        if not model_handler:
            return "[Error: Model handler not available]"
            
        input_source = str(source)

        # Create input particle and trigger field-level quantum effects
        if input_source == "user_input":
            input_particle = await self.process_input(prompt, input_source)
            
            # Trigger contextual collapse for user interactions
            if field_api and input_particle:
                collapse_log = await field_api.trigger_contextual_collapse(
                    input_particle, 
                    "user_interaction",
                    cascade_radius=0.7
                )
                self.log(f"User interaction triggered {len(collapse_log)} particle collapses")

        # Score context particles with quantum awareness
        quantum_context = self.score_quantum_context_particles(prompt, context_particles or [])

        # Generate response via model handler
        try:
            response = await model_handler.generate(
                str(prompt), 
                context_id=str(uuid.uuid4()), 
                tags=["user_input" if input_source == "user_input" else "internal_thought"],
                max_new_tokens=max_tokens,
                source=input_source
            )
            
            # Create response particle linked to input
            if field_api and input_particle:
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

            api.call_api("memory_bank", "update", ("conversation", mem_entry, [input_particle.id, response_particle.id], "voice generation"))

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
            field_api = api.get_api("particle_field")
            if not field_api:
                return
                
            # Create lingual particle for input processing
            await field_api.spawn_particle(
                id=None,
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
            
        except Exception as e:
            self.log(f"Input processing error: {e}")

    async def spawn_and_learn_token(self, token, source=None):
        """Create and learn from token via particle field"""
        try:
            field_api = api.get_api("particle_field")
            lexicon_api = api.get_api("lexicon_store")
            
            if field_api:
                # Spawn lingual particle for token
                lp = await field_api.spawn_particle(
                    id=None,
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
                if lexicon_api and hasattr(lp, 'learn'):
                    await lp.learn(token=token)
                    
        except Exception as e:
            self.log(f"Token learning error: {e}")

    async def reflect(self, particle):
        """Create reflection particle"""
        try:
            field_api = api.get_api("particle_field")
            model_handler = api.get_api("model_handler")
            
            if not field_api or not model_handler:
                return
                
            # Generate reflection
            reflection_prompt = f"[Reflection] Reflecting on: {particle.get_content()}"
            
            reflection_response = await model_handler.generate(
                reflection_prompt,
                context_id=str(uuid.uuid4()),
                tags=["reflection", "internal"],
                max_new_tokens=200,
                source="internal_reflection"
            )
            
            # Create reflection particle
            await field_api.spawn_particle(
                id=None,
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
            
        except Exception as e:
            self.log(f"Reflection error: {e}")

# Register the API
api.register_api("meta_voice", MetaVoice())
