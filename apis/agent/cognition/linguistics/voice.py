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
from apis.agent.utils.embedding import ParticleLikeEmbedding

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
        else:
            input_particle = await self.field.spawn_particle(
                type="lingual",
                metadata={
                    "content": prompt,
                    "source": input_source,
                    "timestamp": time(),
                    "processing_type": "internal"
                },
                energy=0.6,
                activation=0.5,
                emit_event=True
            )

        # Score context particles with quantum awareness
        quantum_context = self.score_quantum_context_particles(prompt, context_particles or [])

        # Generate response via model handler
        try:
            response = await self.model_handler.generate(
                str(prompt), 
                context_id=str(uuid.uuid4()) or str(context_id), 
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

            mem_key = f"voicegen_{str(uuid.uuid4())}"

            mem_entry = {
                "input": prompt,
                "response": response,
                "context": self.sanitize_context_particles(quantum_context)  # ✅ Sanitize context particles
            }
            
            if source == "user_input":
                self.chat_history.append({"Misty": response, "timestamp": dt.now().timestamp()})
            else:
                self.thoughts.append({"thought": response, "timestamp": dt.now().timestamp()})

            await self.memory_bank.update(key=mem_key, value=mem_entry, links=[str(input_particle.id), str(response_particle.id)], source_particle_id=str(input_particle.id), source="voice generation", tags=gentags, memory_type="conversation")

            return response
        except Exception as e:
            self.log(f"Generation error: {e}")
            return "[Error: Generation failed]"
        
    async def safe_get_particle_content(self, particle):
        """Safely extract content from particle for reflection"""
        try:
            if hasattr(particle, 'get_content'):
                content = await particle.get_content()
            elif hasattr(particle, 'token'):
                content = particle.token
            elif hasattr(particle, 'content'):
                content = particle.content
            elif hasattr(particle, 'metadata') and isinstance(particle.metadata, dict):
                content = particle.metadata.get('content', particle.metadata.get('token', ''))
            else:
                content = str(particle)
            
            # Ensure it's a string and not a complex object
            if isinstance(content, (dict, list, tuple)):
                return str(content)
            return str(content) if content is not None else "empty content"
            
        except Exception as e:
            self.log(f"Error extracting particle content: {e}")
            return f"content extraction error: {str(e)}"
    
    def sanitize_context_particles(self, context_particles):
        """Convert particle objects to serializable data"""
        if not context_particles:
            return []
        
        sanitized = []
        for particle in context_particles:
            try:
                if hasattr(particle, '__class__') and 'Particle' in str(type(particle)):
                    sanitized.append({
                        'id': str(getattr(particle, 'id', 'unknown')),
                        'content': str(getattr(particle, 'token', getattr(particle, 'content', ''))),
                        'type': particle.__class__.__name__,
                        'energy': float(getattr(particle, 'energy', 0.0)),
                        'activation': float(getattr(particle, 'activation', 0.0)),
                        'quantum_state': getattr(particle, 'quantum_state', 'uncertain')
                    })
                else:
                    sanitized.append(str(particle))
            except Exception as e:
                self.log(f"Error sanitizing particle: {e}")
                sanitized.append({'error': str(e), 'type': 'sanitization_failed'})
        
        return sanitized

    def score_quantum_context_particles(self, input_text, context_particles: list, top_k=4):
        """Score context particles considering their quantum states"""
        if not context_particles:
            return []
            
        if not isinstance(context_particles, (list, tuple)):
            context_particles = [context_particles]

        scored_particles = []
        
        try:
            input_embedding = ParticleLikeEmbedding().encode([input_text])  # Pass as list!
            if input_embedding:
                input_embedding = input_embedding[0]  # Get first embedding
        except Exception as e:
            self.log(f"Error creating input embedding: {e}")
            input_embedding = None
        
        for particle in context_particles:
            try:
                base_score = 1.0
                
                if hasattr(particle, 'superposition'):
                    certainty = particle.superposition.get('certain', 0.5)
                    # Certain particles get higher base score
                    base_score *= (0.5 + certainty * 0.5)
                    
                if hasattr(particle, 'linked_particles'):
                    linkage_count = len(particle.linked_particles.get('children', []))
                    base_score *= (1.0 + linkage_count * 0.1)
                    
                if (hasattr(particle, 'embedding') and 
                    particle.embedding is not None and 
                    input_embedding is not None):
                    
                    try:
                        particle_embedding = np.array(particle.embedding)
                        input_embedding_array = np.array(input_embedding)
                        
                        if particle_embedding.size > 0 and input_embedding_array.size > 0:
                            distance = np.linalg.norm(input_embedding_array - particle_embedding)
                            relevance_score = 1 / (1 + distance)  # Closer means more relevant
                            base_score *= relevance_score
                    except Exception as embedding_error:
                        self.log(f"Embedding comparison error: {embedding_error}")
                
                scored_particles.append((base_score, particle))
                
            except Exception as particle_error:
                self.log(f"Error scoring particle {getattr(particle, 'id', 'unknown')}: {particle_error}")
                continue
            
        # Sort by score and return top_k
        scored_particles.sort(key=lambda x: x[0], reverse=True)
        return [particle for _, particle in scored_particles[:top_k]]

    async def process_input(self, text, source=None):
        """Process input text and create lingual particles"""
        try:

            # Create lingual particle for input processing
            particle = await self.spawn_and_learn_token(text, source="user_input")

            self.chat_history.append({"Tony": text, "timestamp": dt.now().timestamp()})

            return particle
        except Exception as e:
            self.log(f"Input processing error: {e}")
            return None

    async def spawn_and_learn_token(self, tokens, source=None):
        """Create and learn from token via particle field - FIXED"""
        try:
            # Convert tuple to string if needed
            if isinstance(tokens, tuple):
                tokens = ' '.join(str(token) for token in tokens if token)
            
            # Ensure overall msg (tokens) is a string
            if not isinstance(tokens, str):
                tokens = str(tokens)

            # FIXED: Proper string method usage and logic
            # Check if tokens contain separators for splitting
            separators = [" ", ",", "_", "-", ".", "!"]
            should_split = any(sep in tokens for sep in separators)
            
            if should_split and len(tokens) > 10:  # Only split longer strings
                # Split tokens by common separators
                import re
                token_list = re.split(r'[\s,._!-]+', tokens)
                token_list = [t.strip() for t in token_list if t.strip()]  # Remove empty
                
                created_particles = []
                for token in token_list[:5]:  # Limit to first 5 tokens
                    if self.field and len(token) > 1:  # Skip single characters
                        lp = await self.field.spawn_particle(
                            type="lingual",
                            metadata={
                                "token": token,
                                "content": token,
                                "source": source or "unknown",
                                "processing_type": "input" if source == "user_input" else "learning_token",
                                "timestamp": time()
                            },
                            energy=0.3,
                            activation=0.4,
                            emit_event=False
                        )
                        
                        # Learn token 
                        if self.lexicon_store and lp and hasattr(lp, 'learn'):
                            try:
                                await lp.learn(token=token)
                            except Exception as learn_error:
                                self.log(f"Token learning error for '{token}': {learn_error}")
                        
                        if lp:
                            created_particles.append(lp)
                
                return created_particles[0] if created_particles else None
            
            else:
                # Single token processing
                if self.field:
                    lp = await self.field.spawn_particle(
                        type="lingual",
                        metadata={
                            "token": tokens,
                            "content": tokens,
                            "source": source or "unknown",
                            "processing_type": "input" if source == "user_input" else "learning_token",
                            "timestamp": time()
                        },
                        energy=0.3,
                        activation=0.4,
                        emit_event=False
                    )
                    
                    # Learn token 
                    if self.lexicon_store and lp and hasattr(lp, 'learn'):
                        try:
                            await lp.learn(token=tokens)
                        except Exception as learn_error:
                            self.log(f"Token learning error for '{tokens}': {learn_error}")

                    return lp
                        
            return None
                        
        except Exception as e:
            self.log(f"Token learning error: {e}")
            import traceback
            self.log(f"Full traceback:\n{traceback.format_exc()}")
            return None

    async def reflect(self, particle=None):
        """Create reflection particle"""
        try:
            if not self.field or not self.model_handler:
                return
            
            chance = random.randint(0, 100) / 100
            if particle is None:
                
                if chance < 0.4:
                    return  # 40% chance to skip reflection

                elif chance >= 0.4 and chance < 0.5:
                    # Generate reflection
                    particle = await self.process_input("how i am.", source = "internal_reflection")
                    safe_content = await self.safe_get_particle_content(particle)
                    reflection_prompt = f"I'm thinking about {safe_content}"
                    self.log(f"Reflecting on particle {particle.id}: {safe_content}")


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
                reflection_prompt = f"[Reflection] Reflecting on: {str(await particle.get_content())}"
                self.log(f"Reflecting on particle {particle.id}: {str(await particle.get_content())}")

            context_id=str(uuid.uuid4())

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
            if chance < 0.15:
                await response_particle.learn(reflection_response)
            
            try:
                await self.memory_bank.update(
                        key=f"reflection_{str(uuid.uuid4())}",
                        value=reflection_response,
                        links=[str(particle.id), str(response_particle.id)] if particle else [str(response_particle.id)],
                        tags=["reflection", "internal"],
                        source="internal_reflection",
                        source_particle_id=str(response_particle.id)
                    )
            except Exception as e:
                self.log(f"Memory update error: {e}")
        except Exception as e:
            self.log(f"Reflection error: {e}")

 
