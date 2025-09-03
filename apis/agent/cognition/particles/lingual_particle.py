"""
lingual particle handles natural language and internal dialogue processing
"""


import random
import string
import time
import uuid
from datetime import datetime
from apis.agent.cognition.particles.utils.particle_frame import Particle

from apis.api_registry import api

INTERNAL_SOURCE = "internal_dialogue"
EXTERNAL_SOURCE = "external_dialogue"


class LingualParticle(Particle):
    def __init__(self, source = None, token = None, **kwargs):
        super().__init__(type="lingual", **kwargs)
        self.token = token or self.metadata.get("token")                                         # stored message
        self.embedding = self._message_to_vector(self.token)   # embedded vector for message

        self.lexicon_store = api.get_api("lexicon_store")
        self.lexicon_id = api.call_api("lexicon_store", "get_term_id", (self.token))

        self.particle_source = source                                                   # particle injection source

        self.definition = api.call_api("lexicon_store", "get_term_def", (self.token))                   # token's stored definition if available
        self.usage_count = 0                                                            # how often the token is seen or used
        self.category = api.call_api("lexicon_store", "get_term_type", (self.token))              # token's stored semantic class
        #self.enrichment_sources = self.lexicon_store.get_term_sources(self.token)      # overall list of sources used

        self.tokenizer = api.get_api("model_handler").tokenizer


    async def create_linked_particle(self, particle_type, content, relationship_type="triggered"):
        """Create a new particle linked to this lingual particle"""
        field_api = api.get_api("particle_field")
        if not field_api:
            return None
            
        metadata = {
            "content": content,
            "triggered_by": self.id,
            "relationship": relationship_type,
            "source": "lingual_particle_creation"
        }
        
        # Inherit some quantum state from parent
        if hasattr(self, 'superposition'):
            # New particle starts with some uncertainty from parent
            energy = 0.5 + (self.superposition['certain'] * 0.3)
            activation = 0.4 + (self.superposition['certain'] * 0.2)
        else:
            energy = 0.5
            activation = 0.4
            
        return await field_api.spawn_particle(
            type=particle_type,
            metadata=metadata,
            energy=energy,
            activation=activation,
            source_particle_id=self.id,  # Creates genealogy linkage
            emit_event=True
        )

    async def trigger_memory_formation(self, content, importance=0.5):
        """Trigger creation of a memory particle from this lingual particle"""
        memory_particle = await self.create_linked_particle(
            particle_type="memory",
            content=content,
            relationship_type="memory_formation"
        )
        
        if memory_particle:
            memory_particle.metadata["importance"] = importance
            memory_particle.metadata["formed_from_lingual"] = True
            
            # Create quantum entanglement between lingual and memory
            if hasattr(self, 'entangle_with') and hasattr(memory_particle, 'entangle_with'):
                self.entangle_with(memory_particle, strength=0.3)
                
        return memory_particle

    def should_update_policy(self):
        return not self.metadata.get("locked_policy", False)


    def choose_policy_from_mood(self):
        if self.should_update_policy():
            return self.infer_policy()
    

    async def generate_expression_from_particle(self, tokens, context_particles=None, source = None):
        if source != "user_input" or EXTERNAL_SOURCE:
            self.log(f"triggering generation for internal particle: {self.particle.id}", source = "LingualParticle", context = "generate_expression_from_particle()")
            #return None

        self.log(f"triggering generation for external particle: {self.particle.id}", source = "LingualParticle", context = "generate_expression_from_particle()")

        if context_particles:
            for p in context_particles:
                if hasattr(p, "importance") and p.importance > 0.8:
                    tokens.append(f"({p.metadata.get('summary', '')})")


        if self.tokenizer:
            output = self.tokenizer.convert_tokens_to_string(tokens)
        else:
            output = " ".join(tokens)

        
        # self.learn(output, tokens, context_particles)  --- deprecated call; moving to dedicated reflection queue for downtime
        api.call_api("lexicon_store", "add_term", (output, tokens, context_particles))
        return output
    

    def modulate_external_particles(self, internal_particles, external_particles):
        for ip in internal_particles:
            for ep in external_particles:
                ep.energy += ip.energy * 0.1
                ep.activation += ip.activation * 0.05
                ep.metadata["mood_influence"] = ip.metadata.get("mood", "neutral")
                ep.metadata.setdefault("influences", []).append({
                    "from": ip.id,
                    "bias": ip.metadata.get("bias", "neutral"),
                    "energy": ip.energy
                })


    def initialize_context_profile(self, token=None, context=None):
        token = token or self.token
        context = context or []

        if not token:
            return False  
        try:
            self.lexicon_id = api.call_api("lexicon_store", "get_term_id", (token))
            self.definition = api.call_api("lexicon_store", "get_term_def", (token))
            self.category = api.call_api("lexicon_store", "get_term_type", (token))
        except Exception as e:
            self.log(f"Unable to initialize context profile; skipping for {str(self.id)} | {e}", source="LingualParticle", context="initialize_context_profile()", level="ERROR")
            return

        # Gating logic: skip if not from external-user
        if self.particle_source in {"external", "external_dialogue", "user_input", EXTERNAL_SOURCE}:
            self.metadata["learning_locked"] = False
            return True

        self.log(f"[InitContext] Delaying non-user particle: {self.particle_source}", source="LingualParticle", context="initialize_context_profile()")
        self.metadata["learning_locked"] = False
        return False


    async def learn(self, token=None, context=None):
        """
        Learns about the current token by defining, classifying, and storing it.
        Stores information in the lexicon and logs memory events.
        """
        token = token or self.token
        context = context or [self.expression] if self.expression else []

        if not token or not isinstance(token, str) or token.strip() == "":
            self.log(f"[Learn] Skipping invalid token: {token}", source="LingualParticle", context="learn()")
            return

        if not self.initialize_context_profile(token, context):
            return

        # Skip if already well-defined
        if api.call_api("lexicon_store", "has_deep_entry", (token)):
            self.log(f"[Learn] Token already deeply stored: {token}", source="LingualParticle", context="learn()")
            return

        # Classify and define
        classified = api.call_api("external_resources", "classify_term", (token, context))
        definition, sources = await self.define_term(token, phrase=context)

        stored = self._store_in_memory(token, definition, classified, sources)

        if not stored:
            await api.call_api(
                "lexicon_store", 
                "add_term", 
                (
                token,
                context,
                definition or "No definition.",
                context,
                sources or "unknown",
                classified.get("type", "unknown"),
                classified.get("tags", []),
                classified.get("intent", "neutral"),
                str(self.id),
                self.embedding
            ))



        source_str = ", ".join(sources.keys()) if isinstance(sources, dict) else str(sources)
        await api.call_api("memory_bank", "link_token", (token, definition, source_str))

        api.call_api("memory_bank", "update", (f"learn-{token}-{int(time.time())}", f"[Lexical Learn] Learned: '{token}' with classification {classified}"))

        self.log(f"[Learn] Lexical acquisition complete: {token} | type: {classified.get('type')}", source="LingualParticle", context="learn()")


    async def learn_from_particle(self, particle):
        text = self.expression
        tone = particle.traits.get("tone")
        origin = particle.traits.get("origin")

        context = {text, tone, origin}


        tokens = text.split()
        

        for token in tokens: 
            if token not in self.lexicon_store.lexicon or api.call_api("lexicon_store", "get_term_def", (token)) == Exception:
                classified = api.call_api("external_resources", "classify_term", (token, context))
                definition, sources = await self.define_term(token, tokens)

                if not self._store_in_memory(token, definition, classified, sources):
                    api.call_api(
                        "lexicon_store", 
                        "add_term",
                        (
                            token,
                            tokens,
                            definition,
                            context or "learn",
                            sources,
                            classified["type"],
                            classified["tags"],
                            classified["intent"],
                            str(self.id),
                            self.embedding
                        )
                    )

                source=", ".join(sources.keys()) if isinstance(sources, dict) else str(sources)
                api.call_api("memory_bank", "link_token", (token, definition, source))

            


        await self.memory_bank.update(
            key=f"learn-{int(time.time())}",
            value=f"I learned {len(tokens)} new terms.",
            persistent=True
        )

        self.log(f"[Learn] learned about {particle.id} | {particle.type}.", source="LingualParticle", context="learn_from_particle()")


    async def define_term(self, term, phrase):
        defs = api.call_api("external_resources", "get_external_definitions", (term, phrase))
        final_def, sources_used = self.compare_and_merge_definitions(defs)

        #await self.reflect_on_def(term, sources_used)
        return final_def, sources_used or "Definition unavailable."


    async def reflect_on_def(self, term, sources):
        summary = f"I encountered the term '{term}'. SpaCy defines it as: {sources.get('spacy')}"
        api.call_api("memory_bank", "update", (f"reflect-def-{term}", summary))
        self.log(f"[Reflect] {summary}", source = "LingualParticle", context = "reflect_on_def()")


    def compare_and_merge_definitions(self, def_dict):
        ranked_defs = sorted(def_dict.items(), key=lambda item: len(item[1] or ""), reverse=True)
        sources_used = {k: v for k, v in ranked_defs if v}
        if not sources_used:
            return None, {}
        best_source, best_def = next(iter(sources_used.items()))
        return best_def, sources_used

    """ - temporarily disabled until particle field and engine update is complete for proper integration
    def _store_in_memory(self, token, definition, classification, sources):
        memory_particles = self.particle_engine.particles
        context = classification.get("context", "learn")
        for mem in memory_particles:
            if isinstance(mem, MemoryParticle) and mem.metadata.get("key") == f"lex-{token}":
                mem.metadata.update({
                    "definitions": definition,
                    "context": context,
                    "source": sources,
                    "term_type": classification["type"],
                    "tags": classification["tags"],
                    "intent": classification["intent"],
                    "updated": time.time()
                })
                return True
        return False
    """

    def decay(self):
        decay_factor = 0.95 if self.particle_source == INTERNAL_SOURCE else 0.98
        self.energy *= decay_factor
        self.activation *= decay_factor


    @staticmethod
    async def spawn_from_lexicon(engine, key, token, definition=None, persistent=False, source=None):
        now = datetime.now().timestamp()
        return await engine.spawn_particle(
            id=uuid.uuid4(),
            type="lingual",
            metadata={
                "key": key,
                "token": token,
                "definitions": definition or "No definition available.",
                "created_at": now,
                "source": "LexiconStore",
                "confidence": 1.0,
                "importance": 1.0,
                "layer": "core" if persistent else "ephemeral"
            },
            energy=0.12,
            activation=0.15,
            AE_policy="reflective",
            particle_source = source or EXTERNAL_SOURCE
        )
    

    @classmethod
    def from_reflection(cls, reflection_result, tokenizer=None):
        particle = cls(
            type="lingual",
            metadata={
                "expression": reflection_result["output"],
                "tone": reflection_result["tone"],
                "origin": reflection_result["triggered_by"],
                "referenced_particles": [p.name for p in reflection_result["referenced_particles"]],
                "source": "reflection_cycle",
                "created_at": reflection_result["timestamp"]
            },
            tokenizer=tokenizer,
            energy=0.1,
            activation=0.1
        )

        particle.particle = particle
        return particle
    
    
    @property
    def traits(self):
        return self.metadata

