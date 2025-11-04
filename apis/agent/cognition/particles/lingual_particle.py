"""
Particle-based Cognition Engine - lingual particles for natural language expression and processing
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

import random
import string
import time
import uuid
import asyncio
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
        self.ext_res = api.get_api("external_resources")

        self.metadata.setdefault("token", self.token or "")
        self.metadata.setdefault("content", self.token or "")
        self.metadata.setdefault("source", source)

        self.lexicon_id = None

        self.particle_source = None                                                     # particle injection source

        self.definition = None                                                          # token's stored definition if available
        self.usage_count = 0                                                            # how often the token is seen or used
        self.category = None                                                            # token's stored semantic class
        #self.enrichment_sources = self.lexicon_store.get_term_sources(self.token)      # overall list of sources used

        if self.config.agent_mode == "llm-extension":
            self.tokenizer = api.get_api("_agent_model_handler").tokenizer
        if self.config.agent_mode == "cog-growth":
            self.tokenizer = self.lexicon_store.custom_tokenizer

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
                
    async def sync_data(self):
        try:
            self.definition = await self.lexicon_store.get_term_def(self.token)
            self.category = self.lexicon_store.get_term_type(self.token)
            self.lexicon_id = await self.lexicon_store.get_term_id(self.token)
            return True
        except Exception as e:
            self.log(f"Error syncing data for token {self.token}: {e}", source="LingualParticle", context="sync_data()", level="ERROR")
            return False

    async def initialize_context_profile(self, token=None, context=None):
        token = token or self.token
        context = context or []

        if not token:
            return False  
        try:
            self.lexicon_id = await self.lexicon_store.get_term_id(token)
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

    async def learn_phrase(self, phrase=None, context=None):
        """Takes a full phrase, tokenizes it via custom tokenizer, and learns each token using learn()."""
        if not phrase:
            self.log(f"No phrase provided to learn from.", source="LingualParticle", context="learn_phrase()")
            return
        
        tokens = self.lexicon_store.custom_tokenizer(phrase)
        if not tokens:
            self.log(f"No tokens extracted from phrase: {phrase}", source="LingualParticle", context="learn_phrase()")
            return
        
        for token in tokens:
            await self.learn(token, origin=phrase, context=context if context else None)


    async def learn(self, token=None, context=None, origin=None):
        """
        Learns about the current token by defining, classifying, and storing it.
        Stores information in the lexicon and logs memory events.
        """
        token = str(token) or str(self.token)
        if len(token.split()) > 1:
            self.log(f"Multi-word token detected, manually splitting: {token}", source="LingualParticle", context="learn()")
            token = self.lexicon_store.custom_tokenizer(token)
        
        context = context or [self.expression] if self.expression else self.metadata.get("context", [])
        now = datetime.now().timestamp()

        if not token:
            self.log(f"[Learn] Skipping invalid token: {token}", source="LingualParticle", context="learn()")
            return
        
        if isinstance(token, list): # processes list of tokens for custom_tokenizer usage, otherwise processes normally (str)
            for word in token:
                if not await self.sync_data():
                    self.log(f"[Learn] Sync failed; skipping learning for token: {word}", source="LingualParticle", context="learn()")
                    return
                
                # Skip if already well-defined
                if self.lexicon_store.has_deep_entry(word):
                    self.log(f"[Learn] Token already deeply stored: {word}", source="LingualParticle", context="learn()")
                    return

                # Classify and define
                classified = self.ext_res.classify_term(word)
                definitions = await self.define_term(token)
                sources = definitions.keys() if isinstance(definitions, dict) else ["unknown"]   

                stored = await self._store_in_memory(word, definitions, classified, sources)

                if not stored:
                    await self.lexicon_store.add_term(
                        word,
                        origin or None,
                        definitions or "No definition.",
                        context,
                        sources or "unknown",
                        classified["type"] or "unknown",
                        classified["tags"] or [],
                        classified["intent"] or "neutral",
                        self.id,
                        self.embedding
                    )



                source_str = ", ".join(sources.keys()) if isinstance(sources, dict) else str(sources)
                await self.memory_bank.link_token(word, definitions, source_str)

                await self.memory_bank.update(
                    key = f"learn-{word}-{int(now)}", 
                    value = f"[Lexical Learn] Learned: '{word}' with classification {classified}",
                    source = "lp_learn",
                    source_particle_id=self.id,
                    memory_type="memories",
                )

            self.log(f"[Learn] Lexical acquisition complete: {token} | type: {classified["type"]}", source="LingualParticle", context="learn()")
                
        else:
            if not await self.sync_data():
                self.log(f"[Learn] Sync failed; skipping learning for token: {token}", source="LingualParticle", context="learn()")
                return

            # Classify and define
            classified = self.ext_res.classify_term(token)
            definitions = await self.define_term(token)
            sources = definitions.keys() if isinstance(definitions, dict) else ["unknown"]   

            stored = await self._store_in_memory(token, definitions, classified, sources)

            if not stored:
                await self.lexicon_store.add_term(
                    token,
                    context,
                    definitions or "No definition.",
                    context,
                    sources or "unknown",
                    classified["type"] or "unknown",
                    classified["tags"] or [],
                    classified["intent"] or "neutral",
                    self.id,
                    self.embedding
                )



            source_str = ", ".join(sources.keys()) if isinstance(sources, dict) else str(sources)
            await self.memory_bank.link_token(token, definitions, source_str)

            await self.memory_bank.update(
                key = f"learn-{token}-{int(now)}", 
                value = f"[Lexical Learn] Learned: '{token}' with classification {classified}",
                source = "lp_learn",
                source_particle_id=self.id,
                memory_type="memories",
            )

            self.log(f"[Learn] Lexical acquisition complete: {token} | type: {classified["type"]}", source="LingualParticle", context="learn()")
            
            # Trigger learning moment notification for knowledge curator (cog-growth mode only)
            if self.config.agent_mode == "cog-growth" and definitions:
                # Only trigger for significant learning (not punctuation, common words, etc.)
                if classified["type"] not in ["punctuation", "common", "stop_word"]:
                    await self._trigger_learning_moment_event({
                        "token": token,
                        "definition": definitions,
                        "classification": classified,
                        "sources": sources,
                        "particle_id": self.id,
                        "context": context,
                        "origin": origin
                    })


    async def _trigger_learning_moment_event(self, learning_data):
        """Notify knowledge curator core of significant learning event"""
        try:
            anchor = api.get_api("_agent_anchor")
            if anchor:
                await anchor.emit_event(
                    "learning_moment_detected",
                    learning_data,
                    source="lingual_particle_learning"
                )
                self.log(
                    f"Triggered learning moment for: {learning_data.get('token')}", 
                    "DEBUG", 
                    "_trigger_learning_moment_event"
                )
        except Exception as e:
            # Fail gracefully - learning still happened locally
            self.log(
                f"Failed to trigger learning moment event: {e}", 
                "WARNING",
                "_trigger_learning_moment_event"
            )

    async def learn_from_particle(self, particle=None):

        if not particle or not hasattr(particle, 'type'):
            self.log(f"Invalid particle: {particle}, learning from self instead", "WARNING", "learn_from_particle")
            particle = self
        elif particle.alive is False:
            self.log(f"Particle {particle.id} is not alive, learning from self instead.", "WARNING", "learn_from_particle")
            particle = self
        else:
            particle = particle

        text = particle.expression
        tone = particle.traits.get("tone")
        origin = particle.traits.get("origin")
        now = datetime.now().timestamp()

        context = {text, tone, origin}

        text = str(text)

        if len(text.split()) > 1:
            self.log(f"Multi-word text detected, manually tokenizing: {text}", source="LingualParticle", context="learn_from_particle()")
            tokens = self.lexicon_store.custom_tokenizer(text)
        else:
            tokens = [text]

        if not tokens: # NULL CHECK
            self.log(f"No tokens extracted from text: {text}", source="LingualParticle", context="learn_from_particle()")
            return

        for token in tokens: 
            if token not in self.lexicon_store.lexicon or await self.lexicon_store.get_term_def(token) == Exception:
                classified = self.ext_res.classify_term(token)
                definitions = await self.define_term(token, tokens)
                sources = definitions.keys() if isinstance(definitions, dict) else ["unknown"]                    

                child = await self._store_in_memory(token=token, origin_phrase=tokens, definitions=definitions, classification=classified, sources=sources, context = "learn")
                particle.linked_particles["children"].append(child.id)

                for key, value in definitions.items():
                    await self.memory_bank.link_token(token, value, key, particle)

        await self.memory_bank.update(
            key=f"learn-{int(now)}",
            value=f"I learned {len(tokens)} new terms.",
            memory_type="memories",
            source_particle_id=self.id,
            source = "learn_from_particle",
        )

        # Trigger learning moment for knowledge curator (batch learning from particle)
        if self.config.agent_mode == "cog-growth" and len(tokens) > 0:
            await self._trigger_learning_moment_event({
                "tokens": tokens[:5],  # Send first 5 tokens
                "source_particle": particle.id,
                "particle_type": particle.type,
                "learning_type": "particle_observation",
                "particle_id": self.id,
                "context": context
            })

        self.log(f"[Learn] learned about {particle.id} | {particle.type}.", source="LingualParticle", context="learn_from_particle()")

    async def define_term(self, token):
        """Use available resources to define a given term - use in idle time for comprehensive definitions; be caution of blocking"""
        try:
            if token is None:
                return
            
            resources = api.get_api("external_resources")
            if token in self.lexicon_store.lexicon:
                await self.lexicon_store.encounter_existing_term(token)
            new_definition = await resources.get_external_definitions(term = token)
            if new_definition:
                definitions = []
                for source, definition in new_definition.items():
                    definitions.append({
                        source: definition
                    })
                
                self.log(f"Defined term {token} with definitions from sources: {list(new_definition.keys())}", "INFO", "define_term")
                return definitions
            else:
                self.log(f"No external definitions found for {token}", "INFO", "define_term")
                return "No definitions found"

        except Exception as e:
            self.log(f"Error defining term {token}: {e}", "ERROR", "define_term")
            return None

    async def _define_term(self, term, phrase=None):
        """Defines a term using external resources and merges definitions. - DEPRECATED"""
        defs = await self.ext_res.get_external_definitions(term)
        final_def, sources_used = self.compare_and_merge_definitions(defs)

        #await self.reflect_on_def(term, sources_used)
        #await self.meta_voice.reflect(self)
        return final_def, sources_used or "Definition unavailable."


    def compare_and_merge_definitions(self, def_dict):
        """Compares definitions from various sources and selects the best one. - DEPRECATED"""
        #ranked_defs = sorted(def_dict, key=lambda item: len(item[1] or ""), reverse=True)
        sources_used = {k: v for k, v in def_dict.items() if v}
        if not sources_used:
            return None, {}
        best_source, best_def = next(iter(sources_used.items()))
        return best_def, sources_used


    async def _store_in_memory(self, token, origin_phrase=None, definitions=None, classification=None, sources=None):
        try:
            particle = await self.memory_bank.update(
                key = f"lexicon-{token}",
                value = definitions or "No definition available.",
                source = "lingual_particle",
                tags = ["lexicon", "definition", "linguistics"],
                source_particle_id=self.id,
                memory_type="lexicon",
                metadata = {
                    "token": token,
                    "definition": definitions,
                    "classification": classification,
                    "sources": sources,
                    "context": origin_phrase or "unknown",
                    "learnt_from_particle": self.id
                }
            )
            # Ensure classification is a proper dict
            if not classification or not isinstance(classification, dict):
                classified = self.ext_res.classify_term(token)
            else:
                classified = classification
            
            await self.lexicon_store.add_term(
                token=token,
                full_phrase=origin_phrase or "Unknown origin",
                definitions=definitions,
                #context or "unknown",
                sources="lingual_particle",
                term_type=classified.get("type", "unknown"),
                tags=classified.get("tags", []),
                intent=classified.get("intent", "neutral"),
                particle_id=self.id,
                particle_embedding=self.embedding
                )

            return particle
        except Exception as e:
            self.log(f"Error storing in memory: {e}", source="LingualParticle", context="_store_in_memory()", level="ERROR")
            return False


    def decay(self):
        decay_factor = 0.95 if self.particle_source == INTERNAL_SOURCE else 0.98
        self.energy *= decay_factor
        self.activation *= decay_factor


    @staticmethod
    async def spawn_from_lexicon(self, engine, key, token, definition=None, persistent=False, source=None):
        now = datetime.now().timestamp()
        await self.field.spawn_particle(
            id=None,
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
            particle_source = source or EXTERNAL_SOURCE,
            source_particle_id = self.id,
            emit_event = False
        )
    

    @classmethod
    def from_reflection(self, cls, reflection_result):
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
            tokenizer=self.tokenizer,
            energy=0.1,
            activation=0.1,
            source_particle_id = self.id
        )

        particle.particle = particle
        return particle
    
    
    @property
    def traits(self):
        return self.metadata

