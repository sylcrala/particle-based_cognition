"""
Particle-based Cognition Engine - the framework's "voice" and linguistic processing center, the generation method is determined by the system's set mode; featuring the ability for generation via integration LLM or pure experiential growth.
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

import uuid
import random
from time import time
import numpy as np
import json
import string
import traceback
from apis.api_registry import api
from datetime import datetime as dt
from apis.agent.utils.embedding import ParticleLikeEmbedding

INTERNAL_SOURCE = "internal_dialogue"
EXTERNAL_SOURCE = "external_dialogue"

ENGLISH_ALPHABET = {
    "letters": list(string.ascii_lowercase),
    "punctuation": ".,?!:;'-—()",
    "control": [" ", "\n", "<EOS>", "<PAUSE>", "<s>", "</s>"],
    
    # for future use with phonetic processing
    "phonetic_groups": {
        "vowels": ["a", "e", "i", "o", "u"],
        "consonants": ["b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "x", "y", "z"],
        "liquids": ["l", "r"],  # Semi-vowel sounds
        "nasals": ["m", "n"],   # Nasal sounds
        "fricatives": ["f", "v", "s", "z", "sh", "th"],  # expand later
        "stops": ["p", "b", "t", "d", "k", "g"]  # expand later
    }
}

class MetaVoice:
    def __init__(self, field = None, memory = None, lexicon = None, model_handler = None):
        self.logger = api.get_api("logger")

        self.field = field
        self.memory_bank = memory
        self.lexicon_store = lexicon
        self.model_handler = model_handler

        self.config = api.get_api("config")
        self.agent_config = self.config.get_agent_config() if self.config else {}
        self.agent_name = self.config.agent_name
        self.user_name = self.config.user_name
        
        self._recently_processed = set()
        self.chat_history = []
        self.thoughts = []

    def log(self, message, level = None, context = None):
        context = context or "no_context"
        level = level or "INFO"
        self.logger.log(message, level, "MetaVoice", context)

    async def generate(self, prompt: str, source: str, max_tokens=1600, temperature=0.7, top_p=0.95, context_particles=None, context_id = None, tags = None, source_particle_id = None) -> str:
        """Voice generation router - routes to appropriate generation method based on agent mode set in config"""  
        mode = self.config.agent_mode
        self.log(f"DEBUG: Detected mode = '{mode}'", "DEBUG", context="generate")

        if mode == "llm-extension":
            return await self.generate_with_model(prompt, source, max_tokens, temperature, top_p, context_particles, context_id, tags)
        elif mode == "cog-growth":
            return await self.generate_internal(prompt, source, context_particles, tags, source_particle_id)
        else:
            self.log(f"Unknown agent mode '{mode}' - unable to generate response", "WARNING", context="generate")
            return "Invalid agent mode - no response generated"
        
    async def generate_internal(self, prompt: str, source: str, context_particles = None, tags = None, source_particle_id = None) -> str:
        """Generate response using internal thought process and linguistic capabilities - depends on agent growth and knowledge gained over time - used exclusively for cog-growth mode"""
        input_source = str(source)
        if tags:
            gentags = [tag for tag in (tags or []) if isinstance(tag, str)] + ["user_input" if source == "user_input" else "internal_thought"]

        # Get lexicon terms and build vocabulary
        lexicon_terms = self.lexicon_store.get_terms()
        word_vocabulary = [term for term in lexicon_terms if isinstance(term, str) and len(term) > 1]   

        # Create input particle and trigger field-level quantum effects
        if input_source == "user_input":
            input_particle = await self.process_input(prompt, input_source, source_particle_id)
            
            # Trigger contextual collapse for user interactions
            if self.field and input_particle:
                collapse_log = await self.field.trigger_contextual_collapse(
                    input_particle, 
                    "user_interaction",
                    cascade_radius=0.7
                )
                self.field.set_conscious_attention(particles=[input_particle])
                input_particle.metadata["needs_attention"] = True
                self.log(f"User interaction triggered {len(collapse_log)} particle collapses and refocused conscious attention")
        else:
            input_particle = await self.process_thought(prompt, input_source, source_particle_id)


        # Trigger contextual collapse for internal processing
        if self.field and input_particle:
            collapse_log = await self.field.trigger_contextual_collapse(
                input_particle,
                "internal_processing",
                cascade_radius=0.7
            )
            self.log(f"Internal processing triggered {len(collapse_log)} particle collapses")

        # Trigger reasoning cycle for pseudo-Chain-of-Thought before generation
        anchor = api.get_api("_agent_anchor")
        if anchor:
            try:
                await anchor.emit_event(
                    "reasoning_cycle",
                    {
                        "trigger": "mid_generation_cot",
                        "input_prompt": prompt,
                        "input_particle_id": str(input_particle.id),
                        "source": source
                    },
                    source="voice_generation_cot"
                )
                self.log(f"Triggered mid-generation reasoning cycle for CoT pipeline", "DEBUG", context="generate_internal")
            except Exception as e:
                self.log(f"Failed to trigger reasoning cycle: {e}", "WARNING", context="generate_internal")

        #self.log(f"DEBUG: children type: {type(input_particle.linked_particles.get("children", []))}", "DEBUG")

        # build full context list and generate seed base
        if context_particles is not None:
            if isinstance(context_particles, list):
                full_context = [input_particle] + [p for p in context_particles]
                # Score context particles with quantum awareness
                quantum_context = self.score_quantum_context_particles(prompt, full_context or [])
                seed_base = (
                int(input_particle.activation * 1000) +
                int(input_particle.energy * 1000) +
                int(input_particle.position[6] * 1000) +  # frequency
                int(input_particle.position[7] * 1000) +  # memory phase
                int(input_particle.position[8] * 1000) +  # valence
                hash(str(input_particle.metadata)) % 1000 +
                sum(int(p.activation * 350) for p in context_particles) +
                sum(int(p.energy * 350) for p in context_particles) +
                sum(int(p.position[6] * 350) for p in context_particles) +  # frequency
                sum(int(p.position[7] * 350) for p in context_particles) +  # memory phase
                sum(int(p.position[8] * 350) for p in context_particles) +  # valence
                sum(hash(str(p.metadata)) % 350 for p in context_particles)
                )

                # create random seed based off seed base
                random.seed(seed_base)

            else:
                full_context = [context_particles] + [input_particle]
                # Score context particles with quantum awareness
                seed_base = (
                    int(input_particle.activation * 1000) +
                    int(input_particle.energy * 1000) +
                    int(input_particle.position[6] * 1000) +  # frequency
                    int(input_particle.position[7] * 1000) +  # memory phase
                    int(input_particle.position[8] * 1000) +  # valence
                    hash(str(input_particle.metadata)) % 1000 +
                    int(context_particles.activation * 350) +
                    int(context_particles.energy * 350) +
                    int(context_particles.position[6] * 350) +  # frequency
                    int(context_particles.position[7] * 350) +  # memory phase
                    int(context_particles.position[8] * 350) +  # valence
                    hash(str(context_particles.metadata)) % 350
                )

                # create random seed based off seed base
                random.seed(seed_base)
            

        # Semantic memory retrieval
        memories = []
        memory_phrases = []
        if self.memory_bank:
            memories = await self.memory_bank.quantum_memory_retrieval(prompt, collapse_threshold=0.5)
            for mem in memories:
                if isinstance(mem, dict):
                    content = mem.get("value", {})
                    if "response" in content:
                        text = content.get("response")
                        if isinstance(text, str) and len(text.split()) < 15:
                            memory_phrases.append(text)

        try:
            expression_parts = []
            expression_length = max(5, int(input_particle.activation * 20) + 3)

            avg_valence = sum(p.position[8] for p in self.field.particles if p.id in self.field.alive_particles) / len(self.field.alive_particles)

            for i in range(expression_length):
                if len(word_vocabulary) > 5 and random.random() < (0.7 + (avg_valence / 10)):  # More positive valence = more likely to pick known words
                    expression_parts.append(random.choice(word_vocabulary))
                elif len(memory_phrases) > 5 and random.random() < (0.2 + (avg_valence / 15)):  # Memory-influenced generation
                    expression_parts.append(random.choice(memory_phrases))
                elif random.random() < (0.1 + (avg_valence / 20)):  # Some chance to insert punctuation
                    expression_parts.append(random.choice(ENGLISH_ALPHABET["punctuation"]))
                #elif random.random() < (0.1 + (avg_valence / 30)):  # Some chance to insert control chars - temporarily disabled while we test overall functionality
                #    expression_parts.append(random.choice(ENGLISH_ALPHABET["control"]))
                else:
                    # Character sequence generation using expanded particle positions
                    # Calculate system-wide averages for influence
                    alive_particles = [p for p in self.field.particles if p.id in self.field.alive_particles]
                    if alive_particles:
                        avg_activation = sum(p.activation for p in alive_particles) / len(alive_particles)
                        avg_frequency = sum(p.position[6] for p in alive_particles) / len(alive_particles)
                        avg_intent = sum(p.position[10] for p in alive_particles) / len(alive_particles)
                    else:
                        avg_activation = 0.5
                        avg_frequency = 0.0
                        avg_intent = 0.5
                    
                    # Expanded position influence using multiple particle dimensions
                    pos_influence = (
                        abs(input_particle.position[i % len(input_particle.position)]) * 0.3 +  # Cyclic position influence
                        abs(input_particle.position[6]) * 0.2 +  # Frequency influence
                        abs(input_particle.position[8]) * 0.2 +  # Valence influence  
                        abs(input_particle.position[10]) * 0.15 + # Intent influence
                        abs(avg_activation) * 0.1 +               # System activation influence
                        abs(avg_frequency) * 0.05 +               # System frequency influence
                        abs(avg_intent) * 0.05                    # System intent influence
                    )
                    
                    # Determine character sequence length (1-5 chars based on influences)
                    char_length = max(1, int(pos_influence * 4) + 1)
                    char_length = min(char_length, 12)  # Cap at 12 characters
                    
                    # Generate character sequence
                    char_sequence = ""
                    for j in range(char_length):
                        # Use different position indices for variety
                        pos_index = (i + j) % len(input_particle.position)
                        char_seed = int(abs(input_particle.position[pos_index]) * len(ENGLISH_ALPHABET["phonetic_groups"]["consonants"])) % len(ENGLISH_ALPHABET["phonetic_groups"]["consonants"])
                        
                        # Bias toward vowels or consonants based on valence and intent
                        if input_particle.position[8] > 0.6 and random.random() < 0.4:  # High valence -> more vowels
                            char_choice = random.choice(ENGLISH_ALPHABET["phonetic_groups"]["vowels"])
                        elif input_particle.position[10] > 0.7 and random.random() < 0.3:  # High intent -> consonants
                            char_choice = ENGLISH_ALPHABET["phonetic_groups"]["consonants"][char_seed]
                        else:
                            # Default: use position-influenced character selection
                            char_choice = ENGLISH_ALPHABET["phonetic_groups"]["consonants"][char_seed]
                        
                        char_sequence += char_choice
                    
                    expression_parts.append(char_sequence)

            phrase = " ".join(expression_parts)
            output = str(phrase)
                
            # Create response particle linked to input
            if input_particle and hasattr(input_particle, 'create_linked_particle'):
                output_particle = await input_particle.create_linked_particle(
                    particle_type="lingual",
                    content=output,
                    relationship_type="response_generation"
                )
                await output_particle.learn_phrase(output, prompt)
            self.log(f"cog-growth generation: {output}", "DEBUG", context="generate_internal")
            
            # Observe agent communication for background semantic gravity analysis
            if self.memory_bank and self.memory_bank.agent_categorizer:
                try:
                    # FIXED: Separate user input context from Iris's response content  
                    context_data = {
                        "trigger_context": prompt,  # Original user input (human language)
                        "response_content": output,  # Iris's actual response (may contain compressed language)
                        "usage_context": "agent_response" if source == "user_input" else "internal_thought",
                        "response_type": "cog_growth_internal",
                        "content_type": "iris_response"  # Flag as Iris's generated content
                    }
                    await self.memory_bank.agent_categorizer.observe_agent_communication(output, context_data)
                    self.log(f"Agent communication observed for semantic analysis: {output[:50]}... (responding to: {prompt[:30]}...)", "DEBUG", context="generate_internal")
                except Exception as e:
                    self.log(f"Error observing agent communication: {e}", "WARNING", context="generate_internal")
            
            return output
        except Exception as e:
            self.log(f"Internal generation error: {e}", "ERROR", context="generate_internal")
            self.log(f"Full traceback:\n{traceback.format_exc()}", "ERROR", context="generate_internal")
            return "[Error: Internal generation failed]"

        

    async def generate_with_model(self, prompt: str, source: str, max_tokens=1600, temperature=0.7, top_p=0.95, context_particles=None, context_id = None, tags = None) -> str:
        """Generate response using model handler and quantum-aware particle context - used exclusively for llm-extension mode"""

        if not self.model_handler:
            return "[Error: Model handler not available - this method requires llm-extension mode to be enabled]"
            
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
                emit_event=False
            )

        if context_particles is not None:
            full_context = context_particles + [input_particle] if input_particle else context_particles or []
        else:
            full_context = [input_particle] if input_particle else []
            
        # Score context particles with quantum awareness
        quantum_context = self.score_quantum_context_particles(prompt, full_context or [])

        # Semantic memory retrieval
        memories = []
        if self.memory_bank:
            memories = await self.memory_bank.quantum_memory_retrieval(prompt)
            self.log(f"Retrieved {len(memories)} relevant memories for context", context="generate_with_model")

        # Build contextual prompt with memory integration
        contextual_prompt = self._build_contextual_prompt(prompt, memories, quantum_context)

        # Generate response via model handler
        try:
            response = await self.model_handler.generate(
                contextual_prompt, 
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
                "context": self.sanitize_context_particles(quantum_context) 
            }
            
            if source == "user_input":
                self.chat_history.append({f"{self.agent_name}": response, "timestamp": dt.now().timestamp()})
            else:
                self.thoughts.append({"thought": response, "timestamp": dt.now().timestamp()})

            await self.memory_bank.update(
                key=mem_key, 
                value=mem_entry, 
                links=[str(input_particle.id), str(response_particle.id)], 
                source_particle_id=input_particle.id if input_particle else None, 
                source="response generation", 
                tags=gentags, 
                memory_type="memories"
            )

            # Observe agent communication for background semantic gravity analysis
            if self.memory_bank and hasattr(self.memory_bank, 'agent_categorizer') and self.memory_bank.agent_categorizer:
                try:
                    # FIXED: Separate user input context from model response content
                    context_data = {
                        "trigger_context": prompt,  # Original user input (human language)  
                        "response_content": response,  # Model's response content
                        "usage_context": "agent_response",
                        "response_type": "llm_extension_model",
                        "content_type": "model_generated"  # Flag as model-generated content
                    }
                    await self.memory_bank.agent_categorizer.observe_agent_communication(response, context_data)
                    self.log(f"Agent communication observed for semantic analysis: {response[:50]}... (responding to: {prompt[:30]}...)", "DEBUG", context="generate_with_model")
                except Exception as e:
                    self.log(f"Error observing agent communication: {e}", "WARNING", context="generate_with_model")

            return response
        except Exception as e:
            self.log(f"Model generation error: {e}", level="ERROR", context="generate_with_model")
            return "[Error: Model generation failed]"
        
    async def safe_get_particle_content(self, particle):
        """Safely extract content from particle for reflection with structure awareness"""
        try:
            # First, try to get basic content
            content = None
            if hasattr(particle, 'get_content'):
                content = await particle.get_content()
            elif hasattr(particle, 'token'):
                content = particle.token
            elif hasattr(particle, 'content'):
                content = particle.content
            elif hasattr(particle, 'metadata') and isinstance(particle.metadata, dict):
                content = particle.metadata.get('content') or particle.metadata.get('token') or particle.metadata.get('text')
            
            # For self-awareness, include SAFE particle structure summary
            structure_summary = self._create_safe_particle_summary(particle)
            
            # Combine content with structure summary
            if content:
                if isinstance(content, (dict, list, tuple)):
                    content_str = str(content)[:200]  # Limit content size
                else:
                    content_str = str(content)[:200] if content else "empty"
                
                return f"Content: {content_str} | Structure: {structure_summary}"
            else:
                return f"Structure: {structure_summary}"
            
        except Exception as e:
            self.log(f"Error extracting particle content: {e}")
            return f"content extraction error: {str(e)}"
    
    def _create_safe_particle_summary(self, particle):
        """Create a safe, condensed summary of particle structure for self-awareness"""
        try:
            summary = {
                "id": str(getattr(particle, 'id', 'unknown'))[:8],  # First 8 chars of ID
                "type": str(getattr(particle, 'type', 'unknown')),
                "energy": round(float(getattr(particle, 'energy', 0.0)), 3),
                "activation": round(float(getattr(particle, 'activation', 0.0)), 3),
                "alive": bool(getattr(particle, 'alive', False))
            }
            
            # Add safe position data (first 3 dimensions only to prevent overflow)
            if hasattr(particle, 'position') and particle.position is not None:
                try:
                    pos = particle.position
                    if hasattr(pos, '__len__') and len(pos) >= 3:
                        summary["pos_xyz"] = [
                            round(float(pos[0]), 3),
                            round(float(pos[1]), 3), 
                            round(float(pos[2]), 3)
                        ]
                        # Add key semantic dimensions
                        if len(pos) > 6:
                            summary["frequency"] = round(float(pos[6]), 3)
                        if len(pos) > 8:
                            summary["valence"] = round(float(pos[8]), 3)
                        if len(pos) > 10:
                            summary["intent"] = round(float(pos[10]), 3)
                except (IndexError, TypeError, ValueError):
                    summary["pos_xyz"] = "parsing_error"
            
            # Add safe metadata summary
            if hasattr(particle, 'metadata') and isinstance(particle.metadata, dict):
                meta_keys = list(particle.metadata.keys())[:5]  # Only first 5 keys
                summary["meta_keys"] = meta_keys
                
                # Include specific important metadata
                if 'source' in particle.metadata:
                    summary["source"] = str(particle.metadata['source'])[:20]
                if 'timestamp' in particle.metadata:
                    summary["timestamp"] = str(particle.metadata['timestamp'])[:20]
            
            # Add quantum state if available
            if hasattr(particle, 'superposition'):
                try:
                    summary["quantum_state"] = {
                        "certain": round(float(particle.superposition.get('certain', 0.5)), 3),
                        "uncertain": round(float(particle.superposition.get('uncertain', 0.5)), 3)
                    }
                except (TypeError, ValueError):
                    summary["quantum_state"] = "unknown"
            
            # Convert to string and limit size to prevent segfaults
            summary_str = str(summary)
            if len(summary_str) > 500:  # Limit to 500 chars
                summary_str = summary_str[:497] + "..."
            
            return summary_str
            
        except Exception as e:
            return f"structure_summary_error: {str(e)[:50]}"
    
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
                
                # Score based on quantum certainty
                if hasattr(particle, 'superposition') and isinstance(particle.superposition, dict):
                    certainty = particle.superposition.get('certain', 0.5)
                    # Certain particles get higher base score
                    base_score *= (0.5 + certainty * 0.5)
                    
                # Score based on linkage (connected particles are more relevant)
                if hasattr(particle, 'linked_particles') and isinstance(particle.linked_particles, dict):
                    linkage_count = len(particle.linked_particles.get('children', []))
                    base_score *= (1.0 + linkage_count * 0.1)
                    
                # Handle semantic embedding comparison (384D text embeddings)
                if (hasattr(particle, 'embedding') and 
                    particle.embedding is not None and 
                    input_embedding is not None):
                    
                    try:
                        particle_embedding = np.array(particle.embedding)
                        input_embedding_array = np.array(input_embedding)
                        
                        # Check if embeddings are compatible dimensions
                        if (particle_embedding.size > 0 and input_embedding_array.size > 0 and 
                            particle_embedding.shape == input_embedding_array.shape):
                            
                            distance = np.linalg.norm(input_embedding_array - particle_embedding)
                            relevance_score = 1 / (1 + distance)  # Closer means more relevant
                            base_score *= relevance_score
                            
                        elif particle_embedding.size == 12 and input_embedding_array.size == 384:
                            # Particle has 12D position embedding, input has 384D text embedding
                            # Skip direct comparison but use content-based scoring instead
                            self.log(f"Dimension mismatch: particle={particle_embedding.size}D, input={input_embedding_array.size}D - using content fallback")
                            
                            # Try content-based semantic comparison instead
                            if hasattr(particle, 'token') or hasattr(particle, 'content'):
                                particle_text = getattr(particle, 'token', getattr(particle, 'content', ''))
                                if particle_text and isinstance(particle_text, str) and len(particle_text) > 0:
                                    try:
                                        # Generate embedding for particle content
                                        particle_content_embedding = ParticleLikeEmbedding().encode([str(particle_text)])
                                        if particle_content_embedding and len(particle_content_embedding) > 0:
                                            particle_content_embedding = particle_content_embedding[0]
                                            distance = np.linalg.norm(input_embedding_array - np.array(particle_content_embedding))
                                            relevance_score = 1 / (1 + distance)
                                            base_score *= relevance_score
                                    except Exception as content_embedding_error:
                                        self.log(f"Content embedding fallback error: {content_embedding_error}")
                        else:
                            self.log(f"Incompatible embedding dimensions: particle={particle_embedding.shape}, input={input_embedding_array.shape}")
                            
                    except Exception as embedding_error:
                        self.log(f"Embedding comparison error: {embedding_error}")
                
                # Additional scoring factors for particles without compatible embeddings
                if hasattr(particle, 'activation') and particle.activation > 0.7:
                    base_score *= 1.2  # Boost highly activated particles
                    
                if hasattr(particle, 'energy') and particle.energy > 0.6:
                    base_score *= 1.1  # Boost high-energy particles
                    
                # Recent particles get slight boost
                if hasattr(particle, 'metadata') and isinstance(particle.metadata, dict):
                    timestamp = particle.metadata.get('timestamp', 0)
                    if timestamp > 0:
                        age = time() - timestamp
                        if age < 60:  # Less than 1 minute old
                            base_score *= 1.05
                
                scored_particles.append((base_score, particle))
                
            except Exception as particle_error:
                self.log(f"Error scoring particle {getattr(particle, 'id', 'unknown')}: {particle_error}")
                continue
            
        # Sort by score and return top_k
        scored_particles.sort(key=lambda x: x[0], reverse=True)
        return [particle for _, particle in scored_particles[:top_k]]

    async def process_input(self, text, source=None, source_particle_id = None):
        """Process input text and create lingual particles"""
        try:

            # Create lingual particle for input processing
            particle = await self.spawn_input_particle(text, source="user_input", source_particle_id=source_particle_id or None)
            self.chat_history.append({f"{self.user_name}": text, "timestamp": dt.now().timestamp()})
            self.log(f"Processed user input into lingual particle ID: {str(particle.id) if particle else 'N/A'}", "DEBUG", context="process_input")

            return particle
        except Exception as e:
            self.log(f"Input processing error: {e}")
            return None
        
    async def process_thought(self, text, source=None, source_particle_id = None):
        """Process internal thought text and create lingual particles"""
        try:
            # Create lingual particle for internal processing
            particle = await self.spawn_input_particle(text, source=source, source_particle_id=source_particle_id or None)
            
            if source == "internal_reflection":
                self.thoughts.append({"reflection": text, "timestamp": dt.now().timestamp()})
            else:
                self.thoughts.append({"thought": text, "timestamp": dt.now().timestamp()})

            return particle
        except Exception as e:
            self.log(f"Thought processing error: {e}")
            return None
        
    async def spawn_input_particle(self, text, source=None, source_particle_id = None):
        """Spawn a lingual particle for input text"""
        try:
            if self.field:
                if source_particle_id:
                    source_particle = self.field.get_particle_by_id(source_particle_id)
                    if source == "user_input":
                        particle = await source_particle.create_linked_particle(
                            particle_type="lingual",
                            content = text,
                            relationship_type="input_processing",
                            context = source or "unknown"
                        )
                    else:
                        particle = await source_particle.create_linked_particle(
                            particle_type="lingual",
                            content = text,
                            relationship_type="internal_processing",
                            context = source or "unknown"
                        )
                    
                    text_hash = hash(text.lower().strip()) # hashing processed text to the recently processed list for caching
                    if text_hash not in self._recently_processed:
                        await self.lexicon_store.add_from_particle(particle)
                        self._recently_processed.add(text_hash)

                        if len(self._recently_processed) > 100:
                            self._recently_processed = set(list(self._recently_processed)[-50:]) 

                    return particle
                
                else:
                    particle = await self.field.spawn_particle(
                        type="lingual",
                        metadata={
                            "content": text,
                            "source": source or "unknown",
                            "processing_type": "input" if source == "user_input" else "internal",
                            "timestamp": time()
                        },
                        energy=0.5,
                        activation=0.7,
                        source_particle_id=source_particle_id if source_particle_id else None,
                        emit_event=False
                    )

                    text_hash = hash(text.lower().strip())
                    if text_hash not in self._recently_processed:
                        await self.lexicon_store.add_from_particle(particle)
                        self._recently_processed.add(text_hash)

                        if len(self._recently_processed) > 100:
                            self._recently_processed = set(list(self._recently_processed)[-50:]) 

                    return particle
            return None
        except Exception as e:
            self.log(f"Error spawning input particle: {e}", "ERROR", context="spawn_input_particle")
            return None

    async def spawn_and_learn_token(self, tokens, source=None): 
        """Create and learn from token via particle field"""
        try:
            # Ensure overall msg (tokens) is a string
            if not isinstance(tokens, str):
                tokens = str(tokens)

            
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
                                "context": tokens,
                                "source": source or "unknown",
                                "processing_type": "learning_token",
                            },
                            energy=0.3,
                            activation=0.4,
                            emit_event=False
                        )
                        
                        # Learn token with proper origin context
                        if self.lexicon_store and lp and hasattr(lp, 'learn'):
                            try:
                                await lp.learn(token=token, origin=tokens, context=source)
                            except Exception as learn_error:
                                self.log(f"Token learning error for '{token}': {learn_error}")
                                self.log(f"Full traceback:\n{traceback.format_exc()}")
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
                    
                    # Learn token with proper origin context
                    if self.lexicon_store and lp and hasattr(lp, 'learn'):
                        try:
                            await lp.learn(token=tokens, origin=f"input_processing:{source}" if source else "input_processing", context=source)
                        except Exception as learn_error:
                            self.log(f"Token learning error for '{tokens}': {learn_error}")
                            self.log(f"Full traceback:\n{traceback.format_exc()}")
                    return lp
                        
            return None
                        
        except Exception as e:
            self.log(f"Token learning error: {e}")
            self.log(f"Full traceback:\n{traceback.format_exc()}")
            return None

    async def reflect(self, particle=None):
        """Create reflection particle"""
        try:
            if not self.field:
                return
            
            chance = random.randint(0, 100) / 100
            if particle is None:
                
                if chance < 0.05:
                    # General reflection
                    reflection_prompt = " "
                    self.log("Generating open-ended reflection")

                elif chance >= 0.05 and chance < 0.2:
                    # Reflect on recent thought
                    if self.thoughts:
                        recent_thought = self.thoughts[-1]
                        reflection_prompt = f"I'm reflecting on my recent thought: {recent_thought.get('thought', '')}"
                        self.log(f"Reflecting on recent thought: {recent_thought.get('thought', '')}")
                    else:
                        reflection_prompt = "I'm drawing a blank."
                        self.log("No recent thoughts available for reflection")

                elif chance >= 0.2 and chance < 0.4:
                    # Reflect on recent chat
                    if self.chat_history:
                        recent_chat = self.chat_history[-1]
                        reflection_prompt = f"I'm reflecting on my recent conversation: {recent_chat.get('Tony', '')}"
                        self.log("Reflecting on recent chats")
                    else:
                        reflection_prompt = "I have no recent conversations to reflect on."
                        self.log("No recent chat history available for reflection")

                elif chance >= 0.4 and chance < 0.5:
                    # Generate reflection
                    particle = await self.process_input("what I'm experiencing", source="internal_reflection")
                    safe_content = await self.safe_get_particle_content(particle)
                    reflection_prompt = f"I'm thinking about my neuron {particle.id} and it's content: {safe_content}"
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
                    # Route particle structure reflection through core.py event system for safety
                    anchor = api.get_api("_agent_anchor")
                    if anchor:
                        try:
                            # Emit event for safe particle structure introspection
                            await anchor.emit_event(
                                "particle_structure_introspection",
                                {
                                    "trigger": "self_awareness_reflection",
                                    "request_type": "safe_structure_summary",
                                    "max_particles": 5  # Limit to prevent segfaults
                                },
                                source="voice_self_awareness"
                            )
                            reflection_prompt = "Thinking about my internal neural architecture and particle interactions."
                            self.log("Reflected on internal structure via core event system")
                        except Exception as e:
                            self.log(f"Core event routing failed: {e}, falling back to safe reflection")
                            reflection_prompt = "Reflecting on my current cognitive state."
                    else:
                        reflection_prompt = "Reflecting on my current state."
                        self.log("No core event system available for structure reflection")
            
            else:
                safe_content = await self.safe_get_particle_content(particle)
                reflection_prompt = f"I'm thinking about my neuron {particle.id} and it's content: {safe_content}"
                self.log(f"Reflecting on particle {particle.id}: {safe_content}")

            # Trigger reasoning cycle before reflection generation for CoT processing
            anchor = api.get_api("_agent_anchor")
            if anchor:
                try:
                    await anchor.emit_event(
                        "reasoning_cycle",
                        {
                            "trigger": "reflection_cot",
                            "reflection_prompt": reflection_prompt,
                            "target_particle_id": str(particle.id) if particle else None,
                            "reflection_type": "generative" if particle is None else "targeted"
                        },
                        source="voice_reflection_cot"
                    )
                    self.log(f"Triggered reasoning cycle for reflection CoT pipeline", "DEBUG", context="reflect")
                except Exception as e:
                    self.log(f"Failed to trigger reasoning cycle during reflection: {e}", "WARNING", context="reflect")

            context_id=str(uuid.uuid4())

            reflection_response = await self.generate(
                    reflection_prompt,
                    context_id=context_id,
                    tags=["reflection", "internal"],
                    source="internal_reflection"
                )

            self.log(f"Reflection response: {reflection_response}", "DEBUG", context="reflect")

            # FIXED: Trigger translation analysis with proper content separation
            try:
                if self.memory_bank and hasattr(self.memory_bank, 'agent_categorizer') and self.memory_bank.agent_categorizer:
                    # Separate trigger context from actual response content
                    context_data = {
                        "trigger_context": reflection_prompt,  # Human-readable trigger/prompt
                        "response_content": str(reflection_response),  # Iris's actual compressed response
                        "usage_context": "reflection_response", 
                        "response_type": "internal_reflection",
                        "reflection_target": particle.id if particle else "general",
                        "trigger_source": "post_reflection_processing",
                        "content_type": "iris_compressed_language"  # Flag this as Iris's language, not human
                    }
                    # Analyze the RESPONSE content (Iris's compressed tokens) with trigger as context
                    await self.memory_bank.agent_categorizer.observe_agent_communication(str(reflection_response), context_data)
                    self.log(f"Triggered translation analysis for compressed response: {str(reflection_response)[:50]}... (triggered by: {reflection_prompt[:30]}...)", "DEBUG", context="reflect")
                else:
                    self.log("Translation framework not available for reflection analysis", "DEBUG", context="reflect")
            except Exception as e:
                self.log(f"Error triggering translation analysis for reflection: {e}", "WARNING", context="reflect")

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
                emit_event=False
            )
            if chance < 0.15:
                if isinstance(reflection_response, str):
                    await response_particle.learn(reflection_response, origin="internal_reflection", context="reflection")
                elif isinstance(reflection_response, list):
                    for token in reflection_response:
                        if isinstance(token, str):
                            await response_particle.learn(token, origin="internal_reflection", context="reflection")
                else:
                    self.log(f"Unexpected reflection_response type: {type(reflection_response)}", "WARNING")
                    
                # NEW: Additional translation observation after particle learning
                try:
                    if self.memory_bank and hasattr(self.memory_bank, 'agent_categorizer') and self.memory_bank.agent_categorizer:
                        learning_context = {
                            "human_phrase": reflection_prompt,
                            "usage_context": "particle_learning_reflection",
                            "response_type": "post_learning_reflection", 
                            "particle_id": response_particle.id if response_particle else None,
                            "learning_probability": chance,
                            "trigger_source": "post_learning_processing"
                        }
                        # Observe the learned content for additional compressed tokens
                        if isinstance(reflection_response, str):
                            await self.memory_bank.agent_categorizer.observe_agent_communication(reflection_response, learning_context)
                        elif isinstance(reflection_response, list):
                            for token in reflection_response:
                                if isinstance(token, str):
                                    await self.memory_bank.agent_categorizer.observe_agent_communication(token, learning_context)
                        self.log(f"Triggered translation analysis after particle learning", "DEBUG", context="reflect")
                except Exception as e:
                    self.log(f"Error in post-learning translation analysis: {e}", "WARNING", context="reflect")
            
            try:
                await self.memory_bank.update(
                        key=f"reflection_{str(uuid.uuid4())}",
                        value=reflection_response,
                        links=[str(particle.id), str(response_particle.id)] if particle else [str(response_particle.id)],
                        tags=["reflection", "internal"],
                        source="internal_reflection",
                        source_particle_id=response_particle.id
                    )
            except Exception as e:
                self.log(f"Memory update error: {e}")
                self.log(f"Full traceback:\n{traceback.format_exc()}")
        except Exception as e:
            self.log(f"Reflection error: {e}")
            self.log(f"Full traceback:\n{traceback.format_exc()}")

    def _build_contextual_prompt(self, prompt: str, memories: list, quantum_context: list) -> str:
        """Build enhanced prompt with memory and particle context integration"""
        try: 
            context_parts = []

            # Add context from core identity anchor
            if self.field and hasattr(self.field, 'particles'):
                for particle in self.field.particles:
                    if particle.id in self.field.alive_particles:#TODO FIXME check if this is fixed
                        if particle.type == "core" and particle.role == "identity_anchor":
                            if hasattr(particle, 'metadata') and isinstance(particle.metadata, dict):
                                content = particle.metadata.get('content', '')
                                if content and isinstance(content, dict):
                                    for i in content:
                                        context_parts.append(f"Core Identity Anchor Context [{i}]: {content[i]}")
                    
            # Add context from quantum particles
            if quantum_context:
                context = []
                for particle in quantum_context[:2]:  # Top 2 most relevant
                    if hasattr(particle, 'metadata') and isinstance(particle.metadata, dict):
                        content = particle.metadata.get('content', '')
                        if content and len(content) > 10:
                            context.append(f"- {content[:100]}...")
                
                if context:
                    context_parts.append("Recent context:\n" + "\n".join(context))
            
            # Add relevant memories for continuity
            if memories:
                memory_context = []
                for memory in memories[:3]:  # Top 3 most relevant memories
                    if isinstance(memory, dict):
                        memory_value = memory.get('value', memory.get('payload', {}))
                        if isinstance(memory_value, dict):
                            # Extract meaningful content
                            if 'input' in memory_value and 'response' in memory_value:
                                memory_context.append(f"Previous: {memory_value['input'][:80]}... -> {memory_value['response'][:80]}...")
                            elif 'content' in memory_value:
                                memory_context.append(f"Memory: {memory_value['content'][:100]}...")
                
                if memory_context:
                    context_parts.append("Relevant memories:\n" + "\n".join(memory_context))
            
            # Construct final prompt
            if context_parts:
                context_section = "\n\n".join(context_parts)
                return f"{context_section}\n\nCurrent input: {prompt}"
            else:
                return prompt
                
        except Exception as e:
            self.log(f"Error building contextual prompt: {e}", level="WARNING")
            return prompt  # Fallback to original prompt

 
