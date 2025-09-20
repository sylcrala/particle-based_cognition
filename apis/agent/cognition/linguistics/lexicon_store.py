"""
handles linguistic processing - processing inputs and internal thoughts and parses token(s) and phrases for definition 
"""
import uuid
from datetime import datetime
import asyncio

from apis.api_registry import api
from apis.research.external_resources import ExternalResources


class LexiconStore:
    def __init__(self, memory = None, adaptive_engine = None):
        self.logger = api.get_api("logger")
        self.lexicon = {}
        self.collection_name = "lexicon"

        self.memory = memory
        self.adaptive_engine = adaptive_engine


    async def _insert_entry_to_chroma(self, token, entry):
        entry_data = {
            "token": token,
            "id": entry.get("id", "lex-" + str(uuid.uuid4()) + "-entry"),
            "definitions": entry.get("definitions", []),
            "contexts": entry.get("contexts", []),
            "sources": entry.get("sources", []),
            "times_encountered": entry.get("times_encountered", 1),
            "term_origin": entry.get("term_origin"),
            "last_seen": entry.get("last_seen", datetime.now().isoformat()),
            "intent": entry.get("intent"),
            "type": entry.get("type"),
            "tags": entry.get("tags", []),
            "scope": entry.get("scope", "external"),
            "source_particle_id": entry.get("source_particle_id"),
            "strength": entry.get("strength", 1.0),
            "decay": entry.get("decay", 0.97),
            "context_summary": entry.get("context_summary", ""),
        }

        await self.memory.update(
            key=token,
            value=entry_data,
            source="lexicon",
            tags= entry.get("tags", []),
            collection=self.collection_name
        )

    async def add_term(self, token, full_phrase=None, definitions=None, context=None,
                       source="unknown", intent=None, term_type=None, tags=None,
                       scope="external", particle_id=None, particle_embedding=None):

        full_phrase = full_phrase or token
        entry_id = "lex-" + str(uuid.uuid4()) + "-entry"

        new_entry = {
            "token": token,
            "id": entry_id,
            "definitions": [],
            "contexts": [context] if context else [],
            "sources": [source] if source else [],
            "times_encountered": 1,
            "term_origin": full_phrase,
            "last_seen": datetime.now().isoformat(),
            "intent": intent,
            "type": term_type,
            "tags": tags or [],
            "scope": scope,
            "source_particle_id": particle_id,
            "strength": 1.0,
            "decay": 0.97,
            "context_summary": "",
        }

        if definitions:
            if not isinstance(definitions, list):
                definitions = [definitions]
            for definition in definitions:
                new_entry["definitions"].append({
                    "text": definition,
                    "source": source,
                    "timestamp": datetime.now().isoformat()
                })

        # Try to fetch and merge with an existing lexicon entry if exists
        if self.memory:
            existing = await self.memory.query(
                collection=self.collection_name,
                key={"token": token}
            )

        if existing:
            existing = existing[0]  # first matched doc
            for key in ["contexts", "sources", "tags"]:
                new_entry[key] = list(set(existing.get(key, []) + new_entry.get(key, [])))
            new_entry["definitions"] = existing.get("definitions", []) + new_entry.get("definitions", [])
            new_entry["times_encountered"] = existing.get("times_encountered", 0) + 1
            new_entry["last_seen"] = datetime.now().isoformat()
            new_entry["id"] = existing.get("id", entry_id)

        updated_tags = tags.copy() if tags else []
        updated_tags.extend(["lingual", f"origin: {token}"])

        if self.memory:
            await self.memory.update(
                key=entry_id,
                value=new_entry,
                source=source,
                tags=updated_tags,
                memory_type="lexicon"
            )
            
        # Use adaptive engine for embeddings
        if particle_embedding:
            self.adaptive_engine.set_embedding(entry_id, particle_embedding)
        


    async def get_term(self, token):
        if self.memory:
            result = await self.memory.query(
                collection=self.collection_name,
                key={"token": token}
            )
            return result[0] if result else None
        else:
            self.logger.log("Memory bank not available", level="ERROR", context="get_term()")
            return None

    def get_term_id(self, term):
        # returning the unique ID for the given term, if it's available; else returning an exception
        if term not in self.lexicon:
            self.logger.log(f"Term not found in lexicon: {term} | skipping term", level="WARNING", context="get_term_id()")
            return None
        else:
            return self.lexicon[term]["id"]
        
    def get_terms(self, top_n = None):
        try:
            if not self.memory:
                self.logger.log("Memory bank not available", level="ERROR", context="get_terms()")
                return []
            
            all_terms = [term for term in self.lexicon]

            if top_n and len(all_terms) > top_n:
                all_terms = all_terms[:top_n]
            return all_terms
    
        except Exception as e:
            self.logger.log(f"Error retrieving terms: {e}", level="ERROR", context="get_terms()")
            return []
    
    def get_term_def(self, term):
        if term not in self.lexicon:
            return "Term not found in lexicon: {term} | skipping term"
        else:
            return self.lexicon[term]["definitions"]

    async def load_lexicon(self):
        if self.memory:
            try:
                memories = await self.memory.get_memories_by_type("lexicon")
                for memory in memories:
                    token = memory.get("content")
                    if token:
                        self.lexicon[token] = memory.get("metadata", {})
                self.logger.log(f"Loaded lexicon with {len(self.lexicon)} entries", context="load_lexicon()")
            except Exception as e:
                self.logger.log(f"Error loading lexicon from memory bank: {e}", level="ERROR", context="load_lexicon()")
                print(f"Error loading lexicon from memory bank: {e}")
        else:
            self.logger.log("Memory bank not available", level="ERROR", context="load_lexicon()")

    def get_term_type(self, term):
        """Get the type classification of a term"""
        try:
            if term in self.lexicon:
                return self.lexicon[term].get('type', 'unknown')
            return None
        except Exception as e:
            self.logger.log(f"Error getting term type: {e}", "ERROR", "get_term_type")
            return None

    def get_content(self):
        """Get lexicon content summary"""
        try:
            return {
                'term_count': len(self.lexicon),
                'sample_terms': list(self.lexicon.keys())[:5],
                'status': 'active'
            }
        except Exception as e:
            self.logger.log(f"Error getting content: {e}", "ERROR", "get_content")
            return {'term_count': 0, 'status': 'error'}
        
    async def learn_from_particle(self, particle):
        """Wrapper to trigger learning from a particle via its own learn_from_particle method"""
        try:
            if not particle or not hasattr(particle, 'type'):
                return False
                
            # Only process lingual particles (they have the learning logic)
            if particle.type == "lingual" and hasattr(particle, 'learn_from_particle'):
                # Let the lingual particle handle its own learning
                await particle.learn_from_particle(particle)
                self.logger.log(f"Triggered learning from lingual particle {particle.id}", "INFO", "learn_from_particle")
                return True
                
            # For non-lingual particles, extract basic info
            elif particle.type in ["memory", "cognitive"] and hasattr(particle, 'metadata'):
                content = particle.metadata.get('content', '')
                if content and isinstance(content, str):
                    tokens = content.split()
                    for token in tokens[:5]:  # Process first 5 tokens
                        if token and len(token) > 2:  # Skip short tokens
                            await self.add_term(
                                token=token,
                                context_particles=[particle.id],
                                source=f"particle_{particle.type}",
                                particle_energy=getattr(particle, 'energy', 0),
                                particle_activation=getattr(particle, 'activation', 0)
                            )
                            
                    self.logger.log(f"Learned {len(tokens[:5])} tokens from {particle.type} particle {particle.id}", 
                            "INFO", "learn_from_particle")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.log(f"Error learning from particle: {e}", "ERROR", "learn_from_particle")
            return False

