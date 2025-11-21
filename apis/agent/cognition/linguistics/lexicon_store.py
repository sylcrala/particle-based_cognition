"""
Particle-based Cognition Engine - the lexicon development and storage system, for managing learned terms and their definitions
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
import string
from datetime import datetime
import asyncio
import random

from apis.api_registry import api


class LexiconStore:
    def __init__(self, memory = None, adaptive_engine = None):
        self.logger = api.get_api("logger")
        self.lexicon = {}
        self.collection_name = "lexicon"

        self.memory = memory
        self.adaptive_engine = adaptive_engine
        self.memory_coordinator = None  # Reference to memory coordination particle
        
        # Agent categorization access (will be linked from memory bank)
        self.agent_categorizer = None
        
        # Batch storage system for efficiency
        self.pending_batch = []
        self.batch_size = 10  # Store in batches of 10
        self.batch_timeout = 30  # Store after 30 seconds regardless
        self.last_batch_time = datetime.now()
        
        # Deduplication cache to prevent repeat learning
        self.recent_tokens = set()  # Cache for recently processed tokens
        self.cache_max_size = 1000
        self.cache_cleanup_interval = 100  # Clean cache every N operations
        

    def custom_tokenizer(self, text):
        """Custom tokenizer to split text into words and punctuation - enhanced with validation"""
        if not text or not isinstance(text, str):
            return []
            
        tokens = []
        word = ''
        
        for char in text:
            # REMOVED: Single character tokenization from lexicon lookup
            # This was causing excessive single-character learning noise
            # if char in self.lexicon:
            #     tokens.append(char)
            
            if char in string.whitespace:
                if word.strip():  # Only add non-empty words
                    tokens.append(word.strip())
                    word = ''
            elif char in string.punctuation:
                if word.strip():  # Only add non-empty words
                    tokens.append(word.strip())
                    word = ''
                tokens.append(char)
            else:
                word += char
                
        if word.strip():  # Only add non-empty final word
            tokens.append(word.strip())
            
        # Filter out single characters and empty strings to reduce cognitive noise
        # Allow only meaningful words (length >= 2) or common single-char elements (punctuation)
        filtered_tokens = []
        for token in tokens:
            if token and token.strip():
                if len(token) >= 2 or token in string.punctuation:
                    filtered_tokens.append(token)
        
        return filtered_tokens
    
    def _setup_memory_coordination(self):
        """Set up connection to memory coordination particle for performance optimization"""
        try:
            if self.memory and hasattr(self.memory, 'field') and self.memory.field:
                # Find memory coordination particle
                particles = self.memory.field.get_all_particles()
                for particle in particles:
                    if (hasattr(particle, 'role') and 
                        particle.role == "memory_coordination" and
                        hasattr(particle, 'memory_cache')):
                        self.memory_coordinator = particle
                        self.logger.log("LexiconStore connected to memory coordination particle", "INFO", "_setup_memory_coordination")
                        # Link lexicon store to coordinator
                        particle.lexicon_store = self
                        return
                        
                self.logger.log("Memory coordination particle not found for LexiconStore", "WARNING", "_setup_memory_coordination")
            else:
                self.logger.log("No memory or field available for LexiconStore coordination", "WARNING", "_setup_memory_coordination")
                
        except Exception as e:
            self.logger.log(f"LexiconStore memory coordination setup failed: {e}", "WARNING", "_setup_memory_coordination")
            self.memory_coordinator = None

    def _check_token_deduplication(self, token: str) -> bool:
        """Check if token was recently processed to prevent duplicate learning"""
        try:
            # Clean cache if too large
            if len(self.recent_tokens) > self.cache_max_size:
                # Keep only recent 500 tokens
                token_list = list(self.recent_tokens)
                self.recent_tokens = set(token_list[-500:])
                self.logger.log(f"Cleaned token deduplication cache, kept {len(self.recent_tokens)} tokens", "DEBUG", "_check_token_deduplication")
            
            # Check for recent processing
            if token.lower() in self.recent_tokens:
                #self.logger.log(f"Token '{token}' already recently processed, skipping to prevent duplication", "DEBUG", "_check_token_deduplication")
                return False  # Skip - already processed
            
            # Add to cache
            self.recent_tokens.add(token.lower())
            return True  # Proceed with processing
            
        except Exception as e:
            self.logger.log(f"Error in token deduplication check: {e}", "WARNING", "_check_token_deduplication")
            return True  # Allow processing on error

    async def _flush_pending_batch(self):
        """Flush pending batch to memory storage"""
        try:
            if not self.pending_batch:
                return
                
            batch_size = len(self.pending_batch)
            self.logger.log(f"Flushing batch of {batch_size} lexicon entries to memory", "DEBUG", "_flush_pending_batch")
            
            # Route through coordinator if available for batch optimization
            coordinator_result = await self._route_through_coordinator("batch_store_lexicon", {
                "entries": self.pending_batch,
                "batch_size": batch_size
            })
            
            if coordinator_result is not None:
                self.logger.log(f"Batch storage coordinated successfully: {batch_size} entries", "INFO", "_flush_pending_batch")
            else:
                # Fallback to individual storage
                for entry in self.pending_batch:
                    await self.memory.update(
                        key=entry["key"],
                        value=entry["value"],
                        source=entry.get("source", "lexicon_batch"),
                        tags=entry.get("tags", ["lexicon", "batch"]),
                        memory_type="lexicon",
                        source_particle_id=entry.get("source_particle_id"),
                        **entry.get("kwargs", {})
                    )
                
                self.logger.log(f"Batch fallback storage completed: {batch_size} entries", "INFO", "_flush_pending_batch")
            
            # Clear batch
            self.pending_batch.clear()
            self.last_batch_time = datetime.now()
            
        except Exception as e:
            self.logger.log(f"Error flushing lexicon batch: {e}", "ERROR", "_flush_pending_batch")
            # Clear batch anyway to prevent endless retries
            self.pending_batch.clear()

    async def _add_to_batch(self, key: str, value: dict, source: str = None, tags: list = None, 
                           source_particle_id = None, **kwargs):
        """Add entry to pending batch for efficient storage"""
        try:
            batch_entry = {
                "key": key,
                "value": value,
                "source": source,
                "tags": tags,
                "source_particle_id": source_particle_id,
                "kwargs": kwargs
            }
            
            self.pending_batch.append(batch_entry)
            
            # Check if batch should be flushed
            should_flush = (
                len(self.pending_batch) >= self.batch_size or
                (datetime.now() - self.last_batch_time).total_seconds() > self.batch_timeout
            )
            
            if should_flush:
                await self._flush_pending_batch()
                
        except Exception as e:
            self.logger.log(f"Error adding to batch: {e}", "ERROR", "_add_to_batch")

    async def force_flush_batch(self):
        """Force flush any pending batch entries - call during shutdown"""
        try:
            if self.pending_batch:
                self.logger.log(f"Force flushing {len(self.pending_batch)} pending lexicon entries", "INFO", "force_flush_batch")
                await self._flush_pending_batch()
        except Exception as e:
            self.logger.log(f"Error in force flush: {e}", "ERROR", "force_flush_batch")

    async def _route_through_coordinator(self, operation_type, params):
        """Route operation through memory coordinator if available"""
        if self.memory_coordinator:
            try:
                event = {
                    "type": "memory_task",
                    "data": operation_type,
                    "params": params,
                    "source": "LexiconStore"
                }
                result = await self.memory_coordinator._handle_memory_task(event)
                return result
            except Exception as e:
                self.logger.log(f"LexiconStore coordinator routing failed: {e} - falling back to direct", "WARNING", "_route_through_coordinator")
                return None
        return None
    
    async def add_from_particle(self, particle):
        """Add terms from particle content with enhanced deduplication"""
        try:
            content = str(particle.metadata)
            if not content:
                self.logger.log(f"No content in particle {particle.id} to add to lexicon", "WARNING", "add_from_particle")
                return False
            
            tokens = self.custom_tokenizer(content)
            total_count = len(tokens)
            added_count = 0
            skipped_duplicate_count = 0

            for token in tokens:
                # Apply deduplication check
                if not self._check_token_deduplication(token):
                    skipped_duplicate_count += 1
                    continue
                    
                if random.random() < 0.5:  # 50% chance to skip - helps prevent only initial terms being added
                    continue

                if added_count == total_count or added_count >= 20: # limit to 20 terms per particle to prevent overloading
                    break
                
                await self.add_term(
                    token=token,
                    context=f"from_particle_{particle.id}",
                    source=f"particle_{particle.type}",
                    term_type="particle_metadata",
                    tags=["particle_learning", f"{particle.type}", "parsed_metadata"],
                    particle_id=particle.id
                )
                added_count += 1
                
            if skipped_duplicate_count > 0:
                self.logger.log(f"Skipped {skipped_duplicate_count} duplicate tokens from particle {particle.id}", "DEBUG", "add_from_particle")
                
            return True

        except Exception as e:
            self.logger.log(f"Error adding from particle: {e}", "ERROR", "add_from_particle")
            return False


        except Exception as e:
            self.logger.log(f"Error adding from particle: {e}", "ERROR", "add_from_particle")
            return False

    async def add_term(self, token, full_phrase=None, definitions=None, context=None,
                      source="unknown", intent=None, term_type=None, tags=None,
                      scope="external", particle_id=None, particle_embedding=None, field_position=None, **kwargs):
        """Enhanced add_term with Qdrant memory system, deduplication, batching, and field position storage"""
        
        # Early deduplication check
        if not self._check_token_deduplication(token):
            self.logger.log(f"Token '{token}' already deeply stored, skipping duplicate storage", "DEBUG", "add_term")
            return self.lexicon.get(token, None)
        
        if token in self.lexicon:
            return self.lexicon[token]

        if token is None or token.strip() == "":
            self.logger.log("Cannot add empty token to lexicon", "WARNING", "add_term")
            return None
            
        # Cognitive noise filter - skip single characters except meaningful ones
        if len(token) == 1 and token not in string.punctuation:
            self.logger.log(f"Skipping single character token '{token}' as cognitive noise", "DEBUG", "add_term")
            return None

        try:
            LEX_KEY = f"lexicon_{token.lower()}"

            full_phrase = full_phrase or token
            entry_id = f"lex_{uuid.uuid4().hex}"  # Updated ID format for Qdrant
            particle_id_for_entry = str(particle_id.item()) if particle_id is not None and hasattr(particle_id, 'size') and particle_id.size > 0 and not isinstance(particle_id, uuid.UUID) else str(particle_id) if particle_id is not None else None

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
                "source_particle_id": particle_id_for_entry,
                "strength": 1.0,
                "decay": 0.97,
                "context_summary": "",
                
                # Enhanced fields for consciousness tracking
                "consciousness_level": kwargs.get("consciousness_level", 0.5),
                "particle_energy": kwargs.get("particle_energy", 0.0),
                "particle_activation": kwargs.get("particle_activation", 0.0),
                "learning_context": kwargs.get("learning_context", {}),
                "semantic_connections": kwargs.get("semantic_connections", []),
                
                # Field position data for spatial semantic analysis
                "field_position": self._process_field_position(field_position, particle_id),
                "spatial_semantic_data": self._init_spatial_semantic_data(field_position),
                
                **kwargs  # Store any additional data without restrictions!
            }

            # Process definitions with enhanced structure
            if definitions:
                if isinstance(definitions, list):
                    for def_item in definitions:
                        if isinstance(def_item, dict):
                            new_entry["definitions"].append(def_item)
                        elif isinstance(def_item, str):
                            new_entry["definitions"].append({
                                "text": def_item,
                                "source": "provided",
                                "timestamp": datetime.now().isoformat(),
                                "confidence": 0.9
                            })
                elif isinstance(definitions, str):
                    # Handle string definitions directly
                    new_entry["definitions"].append({
                        "text": definitions,
                        "source": "provided_string",
                        "timestamp": datetime.now().isoformat(),
                        "confidence": 0.8
                    })
                elif isinstance(definitions, dict):
                    # Handle dictionary definitions
                    new_entry["definitions"].append(definitions)

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
                existing_defs = existing.get("definitions")
                if existing_defs != []:
                    new_entry["definitions"] = existing_defs + new_entry.get("definitions")
                
                # Update counters and metadata
                new_entry["times_encountered"] = existing.get("times_encountered", 0) + 1
                new_entry["last_seen"] = datetime.now().isoformat()
                new_entry["id"] = existing.get("id", entry_id)
                
                # Update strength and consciousness based on usage
                new_entry["strength"] = min(1.0, existing.get("strength", 1.0) + 0.1)
                new_entry["consciousness_level"] = min(1.0, existing.get("consciousness_level", 0.5) + 0.025)

            # Prepare tags for memory system
            updated_tags = (tags.copy() if tags else [])
            updated_tags.extend(["lingual", f"origin:{token}", "lexicon_entry"])

            # final definition check - if no definition, attempt quick define
            if len(new_entry["definitions"]) == 0:
                pending = await self.quick_define(token)
                for definition in pending:
                    new_entry["definitions"].append(definition)

            # Agent categorization for lexicon entries
            if self.agent_categorizer:
                try:
                    # Check if token appears to be compressed language
                    is_compressed = (len(token) <= 6 and 
                                   sum(1 for c in token if c in 'aeiou') < len(token) / 2)
                    
                    if is_compressed or context or definitions:
                        # Request categorization from agent
                        category_suggestion = kwargs.get('agent_category')
                        category_result = await self.agent_categorizer.request_categorization(
                            new_entry, category_suggestion
                        )
                        
                        if category_result and category_result != "pending_agent_categorization":
                            new_entry["agent_category"] = category_result
                            new_entry["agent_categorized"] = True
                            updated_tags.append(f"category:{category_result}")
                            self.logger.log(f"Categorized lexicon entry '{token}' as '{category_result}'", "DEBUG", "add_term")
                            
                except Exception as e:
                    self.logger.log(f"Categorization failed for '{token}': {e}", "WARNING", "add_term")

            
            # Store in Qdrant memory system using batch system for efficiency
            if self.memory:
                # Use batch storage instead of individual memory.update calls
                await self._add_to_batch(
                    key=LEX_KEY,  # Unique key for lexicon entries
                    value=new_entry,  # Store the complete entry structure
                    source=source,
                    tags=updated_tags,
                    source_particle_id=particle_id,
                    consciousness_level=new_entry.get("consciousness_level", 0.5),
                    memory_type="lexicon",  # This routes to lexicon collection
                    **kwargs  # Pass through any additional metadata
                )
                
                self.logger.log(f"Added '{token}' to lexicon batch (batch size: {len(self.pending_batch)})", "DEBUG", "add_term")
                
            

            # Update local lexicon cache for fast access
            self.lexicon[token] = new_entry
            
            # Set embedding in adaptive engine if provided
            if particle_embedding is not None and self.adaptive_engine:
                self.adaptive_engine.set_embedding(entry_id, particle_embedding)
            
            return new_entry
            
        except Exception as e:
            self.logger.log(f"Error adding term '{token}': {e}", "ERROR", "add_term")
            import traceback
            self.logger.log(f"Add term traceback: {traceback.format_exc()}", "ERROR", "add_term")
            return None
        
    async def quick_define(self, token: str = None):
        """Quick definition fetcher via direct external resource spacy query - non-blocking"""
        try:
            if token is None:
                return "No token provided for quick definition"
            
            resources = api.get_api("external_resources")
            if token in self.lexicon:
                await self.encounter_existing_term(token)
            new_definition = await resources.wn_quick_def(token)

            if new_definition:
                self.logger.log(f"Quick definition found for {token}: {new_definition}", "INFO", "quick_define")
                definition_entry = {
                    "text": new_definition,
                    "source": "quick_spacy",
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.6
                }
                return definition_entry
            else:
                self.logger.log(f"No quick definition found for {token}", "INFO", "quick_define")
                return "No quick definition found"

        except Exception as e:
            self.logger.log(f"Error in quick defining term {token}: {e}", "ERROR", "quick_define")
            return "Error retrieving quick definition"

    def has_deep_entry(self, token: str):
        """Check if lexicon has a deep entry for the token"""
        try:
            if token in self.lexicon:
                entry = self.lexicon[token]
                asyncio.create_task(self.encounter_existing_term(token))
                if isinstance(entry, dict):
                    return len(entry.get("definitions", [])) > 0 or len(entry.get("contexts", [])) > 0
            return False
        except Exception as e:
            self.logger.log(f"Error checking deep entry for {token}: {e}", "ERROR", "has_deep_entry")
            return False
        
    async def encounter_existing_term(self, token, context=None, source=None):
        """Bump consciousness when encountering existing terms"""
        if token in self.lexicon:
            current_entry = self.lexicon[token]
            current_entry["consciousness_level"] = min(1.0, 
                current_entry.get("consciousness_level", 0.5) + 0.005)
            
            await self.memory.update(
                key=f"lexicon_{token}",
                value=current_entry,
                memory_type="lexicon",
                consciousness_level=current_entry["consciousness_level"]
            )

    async def get_term(self, token):
        """Enhanced get_term using memory coordination for caching and performance"""
        # Try to route through memory coordinator for caching and performance optimization
        coordinator_result = await self._route_through_coordinator("term_lookup", {
            "token": token,
            "operation": "get_term",
            "consciousness_level": getattr(self, '_current_consciousness_level', 0.5)
        })
        
        if coordinator_result is not None:
            return coordinator_result
        
        # Fallback to direct access if coordinator unavailable
        return await self._direct_get_term(token)
    
    async def _direct_get_term(self, token):
        """Direct term lookup (fallback method)"""
        try:
            # First check local cache for performance
            if token in self.lexicon:
                await self.encounter_existing_term(token)
                return self.lexicon[token]
            
            if not self.memory:
                self.logger.log("Memory bank not available", "ERROR", "_direct_get_term")
                return None  # Return None instead of token string
            
            # Query Qdrant memory system
            lexicon_memories = await self.memory.get_memories_by_type("lexicon")
            
            for memory in lexicon_memories:
                if isinstance(memory, dict) and memory.get("token") == token:
                    # Update local cache
                    self.lexicon[token] = memory
                    return memory
            
            return None  # Return None instead of token string when not found
            
        except Exception as e:
            self.logger.log(f"Error getting term {token}: {e}", "ERROR", "_direct_get_term")
            return None  # Return None instead of token string on error

    async def get_term_id(self, term):
        """Get unique ID for term - enhanced for Qdrant system with validation"""
        # Validate term input
        if not term or not isinstance(term, str) or term.strip() == "" or term.lower() == "none":
            self.logger.log(f"Invalid term for get_term_id: {term} - skipping", "WARNING", "get_term_id")
            return None
            
        try:
            # Check local cache first
            if term in self.lexicon:
                await self.encounter_existing_term(term)
                return self.lexicon[term]["id"]
            
            # Query from memory system
            term_data = await self.get_term(term)
            if term_data and isinstance(term_data, dict):
                return term_data.get("id")

            return "Term not found"
            
        except Exception as e:
            self.logger.log(f"Error getting term ID for {term}: {e}", "ERROR", "get_term_id")
            return "Error retrieving term ID"
        
    def get_terms(self, top_n=None):
        """Get all terms from lexicon - enhanced with Qdrant performance"""
        try:
            if not self.memory:
                self.logger.log("Memory bank not available", "ERROR", "get_terms")
                return []
                      
            all_terms = []
            for key, data in self.lexicon.items():
                if isinstance(data, dict) and "token" in data:
                    all_terms.append(data["token"])
                else:
                    self.logger.log(f"Unexpected lexicon entry format for key {key}", "WARNING", "get_terms")
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
        """Get term definitions - enhanced for Qdrant system with validation"""
        # Validate term input
        if not term or not isinstance(term, str) or term.strip() == "" or term.lower() == "none":
            self.logger.log(f"Invalid term for get_term_def: {term} - skipping", "WARNING", "get_term_def")
            return None
            
        try:
            term_data = await self.get_term(term)
            if term_data and isinstance(term_data, dict):
                await self.encounter_existing_term(term)
                definitions = term_data.get("definitions")
                if definitions:
                    return definitions
                else:
                    definition = await self.quick_define(term)
                    self.lexicon[term]["definitions"].append(definition)
                    return definition
            
            if term not in self.lexicon:
                self.logger.log(f"Term {term} not found in lexicon", "INFO", "get_term_def")
                return f"Term {term} not found in lexicon"

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
            memories =  await self.memory.get_memories_by_type("lexicon", limit=10000)
            
            if memories:
                loaded_count = 0
                consciousness_terms = 0
                
                for memory in memories:
                    try:
                        # Handle both dictionary and string memory formats
                        if isinstance(memory, dict):
                            token = memory.get("token")
                            if token:
                                # Store complete entry data in local cache
                                self.lexicon[token] = memory
                                loaded_count += 1
                                
                                # Track consciousness-aware terms
                                if memory.get("consciousness_level", 0) > 0.5:
                                    consciousness_terms += 1
                        elif isinstance(memory, str):
                            # Legacy string format - create basic dictionary entry
                            self.logger.log(f"Converting legacy string memory entry: {memory[:50]}...", "DEBUG", "load_lexicon")
                            # Skip string entries for now - they need proper parsing
                            continue
                        else:
                            self.logger.log(f"Unexpected memory type: {type(memory)}", "WARNING", "load_lexicon")
                    except Exception as entry_error:
                        self.logger.log(f"Error loading lexicon entry: {entry_error}", "WARNING", "load_lexicon")
                
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
        """Get term type classification with coordination optimization"""
        try:
            # For non-async operations, we can still benefit from cache lookup patterns
            if self.memory_coordinator and hasattr(self.memory_coordinator, 'term_existence_cache'):
                # Quick existence check to avoid expensive lookups
                term_exists = self.memory_coordinator.term_existence_cache.is_term_known(term)
                if term_exists is False:
                    return "unknown"
            
            if term in self.lexicon:
                asyncio.create_task(self.encounter_existing_term(term))
                return self.lexicon[term].get('type', 'unknown')
            else:
                self.logger.log(f"Term {term} not found in lexicon when getting type.", "WARNING", "get_term_type")
                return "unknown"
        except Exception as e:
            self.logger.log(f"Error getting term type: {e}", "ERROR", "get_term_type")
            return "unknown"

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
    
    def _process_field_position(self, field_position, particle_id):
        """Process and validate field position data for storage"""
        try:
            # If explicit position provided, use it
            if field_position and isinstance(field_position, (list, tuple)) and len(field_position) >= 3:
                return {
                    "x": float(field_position[0]),
                    "y": float(field_position[1]), 
                    "z": float(field_position[2]),
                    "source": "explicit",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Try to get position from particle if available
            if particle_id and self.memory and hasattr(self.memory, 'field') and self.memory.field:
                try:
                    particles = self.memory.field.get_all_particles()
                    for particle in particles:
                        if str(particle.id) == str(particle_id):
                            if hasattr(particle, 'position'):
                                particle_pos = particle.position
                                
                                # Handle both object format (.x, .y, .z) and numpy array format ([x, y, z])
                                if hasattr(particle_pos, 'x'):
                                    # Object format with .x, .y, .z attributes
                                    x, y, z = float(particle_pos.x), float(particle_pos.y), float(particle_pos.z)
                                elif hasattr(particle_pos, '__len__') and len(particle_pos) >= 3:
                                    # Numpy array or list format [x, y, z]
                                    x, y, z = float(particle_pos[0]), float(particle_pos[1]), float(particle_pos[2])
                                else:
                                    # Unknown position format, skip this particle
                                    continue
                                
                                return {
                                    "x": x,
                                    "y": y, 
                                    "z": z,
                                    "source": "particle",
                                    "timestamp": datetime.now().isoformat()
                                }
                            break
                except Exception as e:
                    self.logger.log(f"Error extracting position from particle {particle_id}: {e}", "DEBUG", "_process_field_position")
            
            # Return None if no position available
            return None
            
        except Exception as e:
            self.logger.log(f"Error processing field position: {e}", "WARNING", "_process_field_position")
            return None
    
    def _init_spatial_semantic_data(self, field_position):
        """Initialize spatial semantic analysis data structure"""
        return {
            "has_position": field_position is not None,
            "spatial_clusters": [],
            "nearest_neighbors": [],
            "semantic_distance_cache": {},
            "last_spatial_analysis": None,
            "spatial_significance": 0.0
        }
    
    def get_tokens_with_positions(self, limit=None):
        """Get lexicon tokens that have field positions for spatial analysis"""
        try:
            positioned_tokens = []
            
            for token, data in self.lexicon.items():
                if isinstance(data, dict):
                    field_pos = data.get("field_position")
                    if field_pos and isinstance(field_pos, dict) and all(k in field_pos for k in ['x', 'y', 'z']):
                        positioned_tokens.append({
                            "token": token,
                            "position": field_pos,
                            "consciousness_level": data.get("consciousness_level", 0.5),
                            "times_encountered": data.get("times_encountered", 1),
                            "type": data.get("type", "unknown")
                        })
            
            # Sort by consciousness level (most conscious first)
            positioned_tokens.sort(key=lambda x: x["consciousness_level"], reverse=True)
            
            if limit:
                positioned_tokens = positioned_tokens[:limit]
                
            return positioned_tokens
            
        except Exception as e:
            self.logger.log(f"Error getting positioned tokens: {e}", "ERROR", "get_tokens_with_positions")
            return []
    
    def calculate_spatial_semantic_distance(self, token1, token2):
        """Calculate semantic distance between two tokens using their field positions"""
        try:
            if token1 not in self.lexicon or token2 not in self.lexicon:
                return None
                
            pos1 = self.lexicon[token1].get("field_position")
            pos2 = self.lexicon[token2].get("field_position")
            
            if not pos1 or not pos2:
                return None
                
            # Calculate Euclidean distance
            dx = pos1["x"] - pos2["x"]
            dy = pos1["y"] - pos2["y"] 
            dz = pos1["z"] - pos2["z"]
            
            distance = (dx**2 + dy**2 + dz**2)**0.5
            
            return {
                "distance": distance,
                "tokens": [token1, token2],
                "positions": [pos1, pos2],
                "calculated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log(f"Error calculating spatial distance between {token1} and {token2}: {e}", "ERROR", "calculate_spatial_semantic_distance")
            return None

