"""
handles linguistic processing - processing inputs and internal thoughts and parses token(s) and phrases for definition 
"""
import uuid
import string
from datetime import datetime
import asyncio
import json

from apis.api_registry import api
from apis.research.external_resources import ExternalResources


class LexiconStore:
    def __init__(self, memory = None, adaptive_engine = None):
        self.logger = api.get_api("logger")
        self.lexicon = {}
        self.collection_name = "lexicon"

        self.memory = memory
        self.adaptive_engine = adaptive_engine

    def custom_tokenizer(self, text):
        """Custom tokenizer to split text into words and punctuation"""
        tokens = []
        word = ''
        for char in text:
            if char in string.whitespace:
                if word:
                    tokens.append(word)
                    word = ''
            elif char in string.punctuation:
                if word:
                    tokens.append(word)
                    word = ''
                tokens.append(char)
            else:
                word += char
        if word:
            tokens.append(word)
        return tokens

    async def add_term(self, token, full_phrase=None, definitions=None, context=None,
                      source="unknown", intent=None, term_type=None, tags=None,
                      scope="external", particle_id=None, particle_embedding=None, **kwargs):
        """Enhanced add_term with Qdrant memory system - maintains all functionality"""
        
        try:
            full_phrase = full_phrase or token
            entry_id = f"lex_{uuid.uuid4().hex}"  # Updated ID format for Qdrant
            
            # Create comprehensive lexicon entry - now with full Qdrant flexibility
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
                "source_particle_id": str(particle_id) if particle_id else None,
                "strength": 1.0,
                "decay": 0.97,
                "context_summary": "",
                
                # Enhanced fields for consciousness tracking
                "consciousness_level": kwargs.get("consciousness_level", 0.5),
                "particle_energy": kwargs.get("particle_energy", 0.0),
                "particle_activation": kwargs.get("particle_activation", 0.0),
                "learning_context": kwargs.get("learning_context", {}),
                "semantic_connections": kwargs.get("semantic_connections", []),
                **kwargs  # Store any additional data without restrictions!
            }

            # Process definitions with enhanced structure
            if definitions:
                if not isinstance(definitions, list):
                    definitions = [definitions]
                for definition in definitions:
                    if isinstance(definition, str):
                        new_entry["definitions"].append({
                            "text": definition,
                            "source": source,
                            "timestamp": datetime.now().isoformat(),
                            "confidence": 1.0
                        })
                    else:
                        # Allow complex definition objects
                        new_entry["definitions"].append(definition)

            # Check for existing term using Qdrant memory system
            existing = None
            if self.memory:
                try:
                    # Use the memory system's lexicon search capability
                    existing_memories = await self.memory.get_memories_by_type("lexicon")
                    for memory in existing_memories:
                        if isinstance(memory, dict) and memory.get("token") == token:
                            existing = memory
                            break
                except Exception as query_error:
                    self.logger.log(f"Error querying existing lexicon entry: {query_error}", "WARNING", "add_term")

            # Merge with existing entry if found
            if existing:
                # Intelligent merging preserving all data
                for key in ["contexts", "sources", "tags"]:
                    existing_list = existing.get(key, [])
                    new_list = new_entry.get(key, [])
                    new_entry[key] = list(set(existing_list + new_list))
                
                # Merge definitions
                existing_defs = existing.get("definitions", [])
                new_entry["definitions"] = existing_defs + new_entry.get("definitions", [])
                
                # Update counters and metadata
                new_entry["times_encountered"] = existing.get("times_encountered", 0) + 1
                new_entry["last_seen"] = datetime.now().isoformat()
                new_entry["id"] = existing.get("id", entry_id)
                
                # Update strength and consciousness based on usage
                new_entry["strength"] = min(1.0, existing.get("strength", 1.0) + 0.1)
                new_entry["consciousness_level"] = min(1.0, existing.get("consciousness_level", 0.5) + 0.05)

            # Prepare tags for memory system
            updated_tags = (tags.copy() if tags else [])
            updated_tags.extend(["lingual", f"origin:{token}", "lexicon_entry"])
            
            # Store in Qdrant memory system with full flexibility
            if self.memory:
                memory_result = await self.memory.update(
                    key=f"lexicon_{token}",  # Unique key for lexicon entries
                    value=new_entry,  # Store the complete entry structure
                    source=source,
                    tags=updated_tags,
                    memory_type="lexicon",  # This routes to lexicon collection
                    source_particle_id=str(particle_id),
                    consciousness_level=new_entry.get("consciousness_level", 0.5),
                    **kwargs  # Pass through any additional metadata
                )
                
                if memory_result:
                    self.logger.log(f"Term '{token}' stored in Qdrant lexicon", "INFO", "add_term")
                else:
                    self.logger.log(f"Failed to store term '{token}' in memory", "ERROR", "add_term")
                    return None
            
            # Update local lexicon cache for fast access
            self.lexicon[token] = new_entry
            
            # Set embedding in adaptive engine if provided
            if particle_embedding and self.adaptive_engine:
                self.adaptive_engine.set_embedding(entry_id, particle_embedding)
            
            return new_entry
            
        except Exception as e:
            self.logger.log(f"Error adding term '{token}': {e}", "ERROR", "add_term")
            import traceback
            self.logger.log(f"Add term traceback: {traceback.format_exc()}", "ERROR", "add_term")
            return None

    def has_deep_entry(self, token: str):
        """Check if lexicon has a deep entry for the token"""
        try:
            if token in self.lexicon:
                entry = self.lexicon[token]
                if isinstance(entry, dict):
                    return len(entry.get("definitions", [])) > 0 or len(entry.get("contexts", [])) > 0
            return False
        except Exception as e:
            self.logger.log(f"Error checking deep entry for {token}: {e}", "ERROR", "has_deep_entry")
            return False

    async def get_term(self, token):
        """Enhanced get_term using Qdrant memory system with local cache"""
        try:
            # First check local cache for performance
            if token in self.lexicon:
                return self.lexicon[token]
            
            if not self.memory:
                self.logger.log("Memory bank not available", "ERROR", "get_term")
                return None
            
            # Query Qdrant memory system
            lexicon_memories = await self.memory.get_memories_by_type("lexicon")
            
            for memory in lexicon_memories:
                if isinstance(memory, dict) and memory.get("token") == token:
                    # Update local cache
                    self.lexicon[token] = memory
                    return memory
            
            # Try semantic search as fallback if exact match not found
            if hasattr(self.memory, 'search_memories'):
                semantic_results = await self.memory.search_memories(f"lexicon {token}", limit=3)
                for result in semantic_results:
                    if (result.get('memory_type') == 'lexicon' and 
                        result.get('value', {}).get('token') == token):
                        entry = result['value']
                        self.lexicon[token] = entry
                        return entry
            
            return None
            
        except Exception as e:
            self.logger.log(f"Error getting term {token}: {e}", "ERROR", "get_term")
            return None

    async def get_term_id(self, term):
        """Get unique ID for term - enhanced for Qdrant system"""
        try:
            # Check local cache first
            if term in self.lexicon:
                return self.lexicon[term]["id"]
            
            # Query from memory system
            term_data = await self.get_term(term)
            if term_data:
                return term_data.get("id")
            
            # If not found, attempt to add it
            self.logger.log(f"Term {term} not found in lexicon, attempting to add it.", "INFO", "get_term_id")
            added_entry = await self.add_term(term)
            if added_entry:
                return added_entry["id"]
            
            return None
            
        except Exception as e:
            self.logger.log(f"Error getting term ID for {term}: {e}", "ERROR", "get_term_id")
            return None
        
    def get_terms(self, top_n=None):
        """Get all terms from lexicon - enhanced with Qdrant performance"""
        try:
            if not self.memory:
                self.logger.log("Memory bank not available", "ERROR", "get_terms")
                return []
            
            # Return from local cache if available
            all_terms = list(self.lexicon.keys())
            
            # If cache is empty or insufficient, this will be populated by load_lexicon
            if len(all_terms) == 0:
                self.logger.log("Local lexicon cache empty, consider calling load_lexicon()", "WARNING", "get_terms")
            
            if top_n and len(all_terms) > top_n:
                all_terms = all_terms[:top_n]
                
            return all_terms
    
        except Exception as e:
            self.logger.log(f"Error retrieving terms: {e}", "ERROR", "get_terms")
            return []
    
    async def get_term_def(self, term):
        """Get term definitions - enhanced for Qdrant system"""
        try:
            term_data = await self.get_term(term)
            if term_data:
                return term_data.get("definitions", [])
            
            # If not found, attempt to add it
            self.logger.log(f"Term {term} not found in lexicon, attempting to add it.", "INFO", "get_term_def")
            added_entry = await self.add_term(term)
            if added_entry:
                return added_entry.get("definitions", [])
            
            return f"Term not found in lexicon: {term} | skipping term"
            
        except Exception as e:
            self.logger.log(f"Error getting definition for {term}: {e}", "ERROR", "get_term_def")
            return f"Error retrieving definition for: {term}"

    async def load_lexicon(self):
        """Enhanced lexicon loading from Qdrant memory system"""
        try:
            if not self.memory:
                self.logger.log("Memory bank not available", "ERROR", "load_lexicon")
                return
            
            self.logger.log("Loading lexicon from Qdrant memory system...", "INFO", "load_lexicon")
            
            # Load with extended timeout for large lexicons
            memories = await asyncio.wait_for(
                self.memory.get_memories_by_type("lexicon", limit=5000),  # Increased limit
                timeout=60.0  # Extended timeout
            )
            
            if memories:
                loaded_count = 0
                consciousness_terms = 0
                
                for memory in memories:
                    if isinstance(memory, dict):
                        token = memory.get("token")
                        if token:
                            # Store complete entry data in local cache
                            self.lexicon[token] = memory
                            loaded_count += 1
                            
                            # Track consciousness-aware terms
                            if memory.get("consciousness_level", 0) > 0.7:
                                consciousness_terms += 1
                
                self.logger.log(
                    f"Loaded {loaded_count} lexicon entries from Qdrant ({consciousness_terms} high-consciousness)", 
                    "INFO", "load_lexicon"
                )
                
            else:
                self.logger.log("No lexicon memories found in Qdrant system", "INFO", "load_lexicon")
                
        except asyncio.TimeoutError:
            self.logger.log("Timeout while loading lexicon from memory system", "ERROR", "load_lexicon")
        except Exception as e:
            self.logger.log(f"Error loading lexicon from memory system: {e}", "ERROR", "load_lexicon")
            import traceback
            self.logger.log(f"Lexicon loading traceback:\n{traceback.format_exc()}", "ERROR", "load_lexicon")

    def get_term_type(self, term):
        """Get term type classification - unchanged functionality"""
        try:
            if term in self.lexicon:
                return self.lexicon[term].get('type', 'unknown')
            else:
                self.logger.log(f"Term {term} not found in lexicon when getting type.", "WARNING", "get_term_type")
                return 'unknown'
        except Exception as e:
            self.logger.log(f"Error getting term type: {e}", "ERROR", "get_term_type")
            return None

    def get_content(self):
        """Enhanced content summary with consciousness insights"""
        try:
            # Basic statistics
            total_terms = len(self.lexicon)
            sample_terms = list(self.lexicon.keys())[:5]
            
            # Enhanced statistics for consciousness tracking
            consciousness_terms = sum(1 for term_data in self.lexicon.values() 
                                    if isinstance(term_data, dict) and 
                                    term_data.get("consciousness_level", 0) > 0.7)
            
            high_usage_terms = [
                (token, data.get("times_encountered", 1)) 
                for token, data in self.lexicon.items() 
                if isinstance(data, dict)
            ]
            high_usage_terms.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'term_count': total_terms,
                'sample_terms': sample_terms,
                'consciousness_terms': consciousness_terms,
                'consciousness_ratio': consciousness_terms / max(total_terms, 1),
                'most_used_terms': high_usage_terms[:5],
                'status': 'active'
            }
            
        except Exception as e:
            self.logger.log(f"Error getting content summary: {e}", "ERROR", "get_content")
            return {'term_count': 0, 'status': 'error'}
        
    async def learn_from_particle(self, particle):
        """Enhanced particle learning with Qdrant consciousness tracking"""
        try:
            if not particle or not hasattr(particle, 'type'):
                self.logger.log(f"Invalid particle: {particle}", "WARNING", "learn_from_particle")
                return False
            if particle.alive is False:
                self.logger.log(f"Particle {particle.id} is not alive, skipping learning.", "WARNING", "learn_from_particle")
                return False
                
            # Process lingual particles (enhanced consciousness tracking)
            if particle.type == "lingual" and hasattr(particle, 'learn_from_particle'):
                await particle.learn_from_particle(particle)
                self.logger.log(f"Triggered learning from lingual particle {particle.id}", "INFO", "learn_from_particle")
                return True
                
            # Enhanced processing for other particle types
            elif particle.type in ["memory", "cognitive"] and hasattr(particle, 'metadata'):
                content = particle.metadata.get('content', '')
                if content and isinstance(content, str):
                    tokens = self.custom_tokenizer(content)
                    learned_count = 0

                    for token in tokens:  # Process all tokens
                        if token and len(token) > 2:  # Skip short tokens
                            # Enhanced add_term call with consciousness data
                            result = await self.add_term(
                                token=token,
                                context=f"learned_from_particle_{particle.id}",
                                source=f"particle_{particle.type}",
                                term_type="particle_derived",
                                tags=["particle_learning", particle.type],
                                particle_id=particle.id,
                                consciousness_level=min(1.0, (getattr(particle, 'energy', 0) + 
                                                            getattr(particle, 'activation', 0)) / 2.0),
                                particle_energy=getattr(particle, 'energy', 0),
                                particle_activation=getattr(particle, 'activation', 0),
                                learning_method="particle_observation"
                            )
                            if result:
                                learned_count += 1
                            
                    self.logger.log(f"Learned {learned_count} tokens from {particle.type} particle {particle.id}", 
                                  "INFO", "learn_from_particle")
                    return learned_count > 0
            
            return False
            
        except Exception as e:
            self.logger.log(f"Error learning from particle: {e}", "ERROR", "learn_from_particle")
            import traceback
            self.logger.log(f"Full traceback:\n{traceback.format_exc()}", "ERROR", "learn_from_particle")
            return False

    # Additional enhanced methods for Qdrant system
    
    async def search_terms(self, query: str, limit: int = 10):
        """New method: Semantic search for terms using Qdrant capabilities"""
        try:
            if not self.memory or not hasattr(self.memory, 'search_memories'):
                return []
            
            # Use memory system's semantic search
            results = await self.memory.search_memories(f"lexicon {query}", limit=limit)
            
            matching_terms = []
            for result in results:
                if (result.get('memory_type') == 'lexicon' and 
                    isinstance(result.get('value'), dict)):
                    term_data = result['value']
                    matching_terms.append({
                        'token': term_data.get('token'),
                        'similarity_score': result.get('similarity_score', 0),
                        'consciousness_level': term_data.get('consciousness_level', 0),
                        'times_encountered': term_data.get('times_encountered', 1),
                        'definitions': term_data.get('definitions', [])
                    })
            
            return matching_terms
            
        except Exception as e:
            self.logger.log(f"Error in semantic term search: {e}", "ERROR", "search_terms")
            return []

    async def get_consciousness_insights(self):
        """New method: Analyze consciousness patterns in lexicon"""
        try:
            if not self.lexicon:
                await self.load_lexicon()
            
            total_terms = len(self.lexicon)
            if total_terms == 0:
                return {"total_terms": 0, "status": "empty"}
            
            # Analyze consciousness patterns
            consciousness_levels = [
                data.get("consciousness_level", 0) 
                for data in self.lexicon.values() 
                if isinstance(data, dict)
            ]
            
            high_consciousness = [
                (token, data) for token, data in self.lexicon.items()
                if isinstance(data, dict) and data.get("consciousness_level", 0) > 0.7
            ]
            
            most_encountered = sorted([
                (token, data.get("times_encountered", 1)) 
                for token, data in self.lexicon.items() 
                if isinstance(data, dict)
            ], key=lambda x: x[1], reverse=True)
            
            return {
                "total_terms": total_terms,
                "avg_consciousness_level": sum(consciousness_levels) / len(consciousness_levels),
                "high_consciousness_terms": len(high_consciousness),
                "consciousness_ratio": len(high_consciousness) / total_terms,
                "top_consciousness_terms": [
                    {"token": token, "level": data.get("consciousness_level")}
                    for token, data in high_consciousness[:10]
                ],
                "most_encountered": [
                    {"token": token, "encounters": count}
                    for token, count in most_encountered[:10]
                ]
            }
            
        except Exception as e:
            self.logger.log(f"Error generating consciousness insights: {e}", "ERROR", "get_consciousness_insights")
            return {"error": str(e)}

