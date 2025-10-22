"""
combination of memory_backend, memory_io - central memory framework (via memory particles and vector DB (TBD))
"""

import json
import asyncio
import os
import uuid
from typing import Dict, List, Any, Optional, Union
import random
from datetime import datetime
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, Range, ScrollRequest, SearchRequest, OptimizersConfig
)

from apis.api_registry import api
from apis.agent.utils.embedding import ParticleLikeEmbedding

class MemoryBank:
    def __init__(self, field = None):
        self.logger = api.get_api("logger")
        self.config = api.get_api("config")
        self.agent_config = self.config.get_agent_config() if self.config else {}

        self.base_mem_path = self.agent_config.get("memory_dir")
        self.mempath = f"{self.base_mem_path}/memory_matrix/"

        try:
            self.field = field
            os.makedirs(os.path.dirname(self.mempath), exist_ok=True)
            self.client = QdrantClient(path=self.mempath)
            self.embeddings = ParticleLikeEmbedding(field=self.field, dimension=384)


            self.memories = "memories"
            self.system = "system_states"
            self.lexicon = "lexicon"
            self.consolidated = "consolidated_memories"

            self._init_collections()
            
            mem_count = self._get_collection_count(self.memories)
            sys_count = self._get_collection_count(self.system)
            lex_count = self._get_collection_count(self.lexicon)
            con_count = self._get_collection_count(self.consolidated)
            self.log(f"MemoryBank initialized. Memories: {mem_count}, System States: {sys_count}, Lexicon: {lex_count}, Consolidated Memories: {con_count}", "INFO", "MemoryBank.__init__")
        
        except Exception as e:
            self.log(f"CRITICAL: MemoryBank initialization failed: {e}", "ERROR", "MemoryBank.__init__")
            raise e

    def log(self, message, level = None, context = None):
        source = "MemoryBank"

        if context != None:
            context = context
        else:
            context = "no context"

        if level != None:
            level = level
        else:
            level = "INFO"

        self.logger.log(message, level, context, source)

    def _init_collections(self):
        """Initialize collections in the Qdrant client"""
        try:
            collections_response = self.client.get_collections()
            existing = {col.name for col in collections_response.collections}

            collection_configs = {
                self.memories: {
                    "description": "Main memory storage",
                    "size": 384,
                    "distance": Distance.COSINE
                },
                self.system: {
                    "description": "System state snapshots",
                    "size": 384,
                    "distance": Distance.COSINE
                },
                self.lexicon: {
                    "description": "Lexicon and linguistic entries",
                    "size": 384,
                    "distance": Distance.COSINE
                },
                self.consolidated: {
                    "description": "Consolidated high-activation memories",
                    "size": 384,
                    "distance": Distance.COSINE
                }
            }

            for collection_name, config in collection_configs.items():
                if collection_name not in existing:
                    self.client.create_collection(
                        collection_name = collection_name,
                        vectors_config = VectorParams(
                            size=config["size"],
                            distance=config["distance"]
                        ),
                        optimizers_config = OptimizersConfig(
                            default_segment_number = 2,
                            max_segment_size = 20000,
                            memmap_threshold = 10000,
                            indexing_threshold = 20000,
                            flush_interval_sec = 5,
                            max_optimization_threads = 2,
                            deleted_threshold = 0.2,
                            vacuum_min_vector_number = 1000
                        )
                    )
                    self.log(f"Created collection: {collection_name}", "INFO", "_init_collections")
                else:
                    self.log(f"Using existing collection: {collection_name}", "INFO", "_init_collections")   
        
        except Exception as e:
            self.log(f"Error initializing collections: {e}", "ERROR", "_init_collections")
            raise e

    def _get_collection_count(self, collection_name: str) -> int:
        """Get count of points in a collection"""
        try:
            # Handle both string names and collection objects
            if isinstance(collection_name, str):
                name = collection_name
            else:
                # Handle collection objects from Qdrant
                name = getattr(collection_name, 'name', str(collection_name))
            
            collection_info = self.client.get_collection(name)
            return collection_info.points_count  # Use points_count attribute
            
        except Exception as e:
            self.log(f"Error getting count for collection {collection_name}: {e}", "ERROR", "_get_collection_count")
            return 0


    async def search_memories_by_embedding(self, embedding, limit: int = 10) -> List[Dict]:
        try:

            results = self.client.search(
                collection_name=self.memories,
                query_vector=embedding,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            memories = []
            for result in results:
                memories.append({
                    'id': result.id,
                    'payload': result.payload,
                    'similarity_score': result.score,
                    'key': result.payload.get('key'),
                    'value': result.payload.get('value'),
                    'memory_type': result.payload.get('memory_type'),
                    'consciousness_level': result.payload.get('consciousness_level', 0.5)
                })
            
            return memories
            
        except Exception as e:
            self.log(f"Error searching for memories by embedding: {e}")
            return []

    async def search_memories(self, query_text: str, limit: int = 10) -> List[Dict]:
        """Semantic search across all memories"""
        try:
            embedding = self.embeddings([query_text])[0]
            
            results = self.client.search(
                collection_name=self.memories,
                query_vector=embedding,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            memories = []
            for result in results:
                memories.append({
                    'id': result.id,
                    'payload': result.payload,
                    'similarity_score': result.score,
                    'key': result.payload.get('key'),
                    'value': result.payload.get('value'),
                    'memory_type': result.payload.get('memory_type'),
                    'consciousness_level': result.payload.get('consciousness_level', 0.5)
                })
            
            return memories
            
        except Exception as e:
            self.log(f"Error in semantic memory search: {e}", "ERROR", "search_memories")
            return []

    async def get_memories_by_type(self, memory_type: str, limit: int = 100) -> List[Dict]:
        """Get memories by type with full payload"""
        try:
            self.log(f"Retrieving memories of type: {memory_type}", context="get_memories_by_type")
            if memory_type == "memory" or memory_type == "memories":
                collection = self.memories
            elif memory_type == "system" or memory_type == "system_states":
                collection = self.system
            elif memory_type == "lexicon":
                collection = self.lexicon


            # Use scroll for efficient pagination
            results, _ = self.client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="memory_type",
                            match=MatchValue(value=memory_type)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True
            )
            
            memories = []
            for point in results:
                # Return the original value stored in payload
                memories.append(point.payload.get("value"))
            
            self.log(f"Retrieved {len(memories)} memories of type {memory_type}", 
                    "INFO", "get_memories_by_type")
            return memories
            
        except Exception as e:
            self.log(f"Error retrieving memories by type: {e}", "ERROR", "get_memories_by_type")
            return []

    async def get_memory_by_particle_id(self, particle_id: str) -> Optional[Dict]:
        """Get memory by source particle ID"""
        try:
            results, _ = self.client.scroll(
                collection_name=self.memories,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_particle_id",
                            match=MatchValue(value=str(particle_id))
                        )
                    ]
                ),
                limit=1,
                with_payload=True
            )
            
            if results:
                point = results[0]
                return {
                    'id': point.id,
                    'value': point.payload.get('value'),
                    'payload': point.payload
                }
            
            return None
            
        except Exception as e:
            self.log(f"Error retrieving memory for particle {particle_id}: {e}", 
                    "ERROR", "get_memory_by_particle_id")
            return None

    async def quantum_memory_retrieval(self, query: str, collapse_threshold: float = 0.7):
        """Enhanced quantum-aware memory retrieval with particle integration"""
        self.log(f"Quantum memory retrieval for: {query}", context="quantum_memory_retrieval")
        
        if not self.field:
            return await self.search_memories(query)
        
        # Find memory particles in field
        memory_particles = self.field.get_particles_by_type("memory")
        
        # Create temporary query particle
        query_particle = await self.field.spawn_particle(
            id=None,
            type="lingual",
            metadata={"content": query, "source": "memory_query", "temporary": True},
            energy=0.8,
            activation=0.9,
            emit_event=False
        )
        
        relevant_memories = []
        
        # Process memory particles with quantum collapse
        for memory in memory_particles:
            distance = query_particle.distance_to(memory)
            relevance = max(0, 1.0 - distance)
            
            if relevance > 0.3:  # Relevance threshold
                if hasattr(memory, 'observe'):
                    collapsed_state = memory.observe(context="memory_retrieval")
                    
                    if collapsed_state == 'certain' or relevance > collapse_threshold:
                        # Get corresponding Qdrant memory
                        qdrant_memory = await self.get_memory_by_particle_id(memory.id)
                        if qdrant_memory:
                            qdrant_memory['quantum_state'] = collapsed_state
                            qdrant_memory['relevance'] = relevance
                            relevant_memories.append(qdrant_memory)
                        
                        # Create interaction linkage
                        if hasattr(self.field, 'create_interaction_linkage'):
                            await self.field.create_interaction_linkage(
                                query_particle.id, memory.id, "memory_retrieval"
                            )
        
        # Clean up temporary particle
        if hasattr(query_particle, 'alive'):
            query_particle.alive = False
        
        return relevant_memories

    def _sanitize_payload(self, payload):
        """Sanitize payload to prevent particle object serialization"""
        try:
            clean_payload = {}
            
            for key, value in payload.items():
                # Handle particle objects
                if hasattr(value, '__class__') and 'Particle' in str(type(value)):
                    # Extract particle content instead of storing object
                    clean_payload[f"{key}_content"] = getattr(value, 'token', getattr(value, 'content', str(value)))
                    clean_payload[f"{key}_id"] = str(getattr(value, 'id', 'unknown'))
                    clean_payload[f"{key}_type"] = value.__class__.__name__
                    
                # Handle lists/tuples that might contain particles
                elif isinstance(value, (list, tuple)):
                    clean_list = []
                    for item in value:
                        if hasattr(item, '__class__') and 'Particle' in str(type(item)):
                            clean_list.append(str(getattr(item, 'id', item)))
                        else:
                            clean_list.append(item)
                    clean_payload[key] = clean_list
                    
                # Handle JSON-safe types
                elif isinstance(value, (str, int, float, bool, type(None))):
                    clean_payload[key] = value
                    
                # Convert everything else to string
                else:
                    clean_payload[key] = str(value)
            
            return clean_payload
            
        except Exception as e:
            self.log(f"Payload sanitization error: {e}", "ERROR", "_sanitize_payload")
            return {"error": str(e), "raw_payload_keys": list(payload.keys()) if isinstance(payload, dict) else "invalid"}

    async def update(self, key: str, value: Any, source: str = None, tags: List[str] = None, 
                    memory_type: str = None, source_particle_id: str = None, 
                    links: List[str] = None, **kwargs) -> Optional[Any]:
        """Store any data structure with complete flexibility - no restrictions!"""
        try:
            self.log(f"Storing memory: {key}", context="update")
            
            tags = tags or []
            now = datetime.now().timestamp()
            
            # Create memory particle first (if field available)
            memory_particle = None
            if self.field:
                memory_metadata = {
                    "key": key,
                    "content": value,
                    "source": source or "unknown",
                    "tags": tags,
                    "memory_type": memory_type or "general",
                    "links": links or [],
                    "created_at": now
                }
                
                memory_particle = await self.field.spawn_particle(
                    id=None,
                    type="memory",
                    metadata=memory_metadata,
                    source_particle_id=source_particle_id,
                    emit_event=False
                )
                
                if not memory_particle:
                    self.log("Failed to spawn memory particle", "WARNING", "update")
            
            self.log("[Memory Update] checkpoint 1", "DEBUG", "update")
            
            # Generate unique point ID
            point_id = int(uuid.uuid4())
            used_collection = self.memories if memory_type == "memory" or memory_type == "memories" else self.lexicon if memory_type == "lexicon" or memory_type == "linguistic" else self.system if memory_type == "system" or memory_type == "system_states" else self.memories
            raw_payload = {
                "key": key,
                "value": value,  
                "source": source or "unknown",
                "tags": tags,
                "memory_type": memory_type or "general",
                "links": links or [],
                "timestamp": now,
                "created_at": datetime.now().isoformat(),
                
                # Particle integration
                "source_particle_id": source_particle_id,
                "memory_particle_id": str(memory_particle.id) if memory_particle else None,
                
                # Consciousness metadata
                "consciousness_level": kwargs.get("consciousness_level", 0.5),
                "quantum_state": kwargs.get("quantum_state", "uncertain"),
                "energy_level": kwargs.get("energy_level", 0.0),
                "activation_level": kwargs.get("activation_level", 0.0),
                
                # Store all additional kwargs
                **kwargs
            }

            payload = self._sanitize_payload(raw_payload)
            
            self.log("[Memory Update] checkpoint 2", "DEBUG", "update")
            
            # Generate embedding for semantic search
            if isinstance(value, dict):
                embed_text = json.dumps(value, default=str)[:1000]  # Limit for embedding
            elif isinstance(value, (list, tuple)):
                embed_text = str(value)[:1000]
            else:
                embed_text = str(value)[:1000]
            
            # Add key and type to embedding context
            embedding_context = f"{key} {memory_type or 'general'} {embed_text}"
            embedding = self.embeddings([embedding_context])[0]
            
            # Create Qdrant point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            
            try:
                # Store in Qdrant with wait for confirmation
                operation_result = self.client.upsert(
                    collection_name=used_collection,
                    wait=True,
                    points=[point]
                )
                
                self.log(f"Memory '{key}' stored successfully in Qdrant (ID: {point_id})", 
                        "INFO", "update")
                
                self.log("[Memory Update] checkpoint 3 - Success", "DEBUG", "update")
                
                # Update particle with Qdrant point ID
                if memory_particle and hasattr(memory_particle, 'metadata'):
                    memory_particle.metadata['qdrant_point_id'] = point_id
                    memory_particle.metadata['qdrant_stored'] = True
                
                return memory_particle or point_id
                
            except Exception as qdrant_error:
                self.log(f"Qdrant storage error: {qdrant_error}", "ERROR", "update")
                import traceback
                self.log(f"Full traceback:\n{traceback.format_exc()}", "ERROR", "update")
                
                # Fallback: store in particle metadata if Qdrant fails
                if memory_particle and hasattr(memory_particle, 'metadata'):
                    memory_particle.metadata['qdrant_failed'] = True
                    memory_particle.metadata['fallback_storage'] = {
                        'payload': payload,
                        'embedding': embedding[:50],  # Store truncated embedding
                        'error': str(qdrant_error)
                    }
                    
                return memory_particle
            
        except Exception as e:
            self.log(f"CRITICAL MEMORY FAILURE: {e}", "ERROR", "update")
            import traceback
            self.log(f"Memory error traceback: {traceback.format_exc()}", "ERROR", "update")
            return None
        
    async def query(self, key: str, collection: str = None) -> Any:
        """Query memory with fallback to semantic search"""
        self.log(f"Querying memory for key: {key}", context="query")
        
        used_collection = collection or self.memories
        
        # First try particle field if available
        if self.field:
            memory_particles = self.field.get_particles_by_type("memory")
            matches = [p for p in memory_particles if p.metadata.get("key") == key]
            if matches:
                target = matches[-1]
                # Boost particle on access
                target.energy = min(1.0, target.energy + 0.1)
                target.activation = min(1.0, target.activation + 0.1)
                return target.metadata.get("content")
        
        try:
            # Exact key match
            exact_results, _ = self.client.scroll(
                collection_name=used_collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="key",
                            match=MatchValue(value=key)
                        )
                    ]
                ),
                limit=1,
                with_payload=True
            )
            
            if exact_results:
                point = exact_results[0]
                self.log(f"Found exact match for key: {key}", "DEBUG", "query")
                return point.payload.get("value")
            
            # Semantic search fallback
            embedding = self.embeddings([key])[0]
            semantic_results = self.client.search(
                collection_name=used_collection,
                query_vector=embedding,
                limit=1,
                score_threshold=0.7,  # High similarity threshold
                with_payload=True
            )
            
            if semantic_results:
                point = semantic_results[0]
                self.log(f"Found semantic match for key: {key} (score: {point.score})", "DEBUG", "query")
                return point.payload.get("value")
            
            return None
            
        except Exception as e:
            self.log(f"Error querying Qdrant: {e}", "ERROR", "query")
            return None
    
      
    def get_random_memory(self) -> Optional[Any]:
        """Get a random memory from the collection"""
        try:
            # Get total count first
            collection_info = self.client.get_collection(self.memories)
            total_count = collection_info.points_count
            
            if total_count == 0:
                self.log("No memories found in collection", "WARNING", "get_random_memory")
                return None
            
            # Get a random offset
            offset = random.randint(0, max(0, total_count - 1))
            
            results, _ = self.client.scroll(
                collection_name=self.memories,
                limit=1,
                offset=offset,
                with_payload=True
            )
            
            if results:
                point = results[0]
                self.log(f"Random memory retrieved: {point.payload.get('key', 'unknown key')}", 
                        context="get_random_memory")
                return point.payload.get("value")
            
            return None
            
        except Exception as e:
            self.log(f"Error getting random memory: {e}", "ERROR", "get_random_memory")
            return None

    async def emergency_memory_diagnostic(self):
        """Comprehensive Qdrant memory system diagnostic"""
        try:
            self.log("Starting emergency Qdrant memory diagnostic", "WARNING", "emergency_memory_diagnostic")
            
            # Check collections
            collections = self.client.get_collections()
            self.log(f"Found collections: {[c.name for c in collections.collections]}", "INFO")
            
            # Check collection health and counts
            for collection in [self.memories, self.system, self.consolidated]:
                try:
                    count = self._get_collection_count(collection)
                    self.log(f"Collection {collection}: {count} points", "INFO")
                    
                    # Sample a few points
                    if count > 0:
                        sample, _ = self.client.scroll(collection_name=collection, limit=3, with_payload=True)
                        self.log(f"Sample from {collection}: {len(sample)} points retrieved", "INFO")
                        
                except Exception as col_error:
                    self.log(f"Error checking collection {collection}: {col_error}", "ERROR")
            
            # Test embedding system
            test_embed = self.embeddings(["diagnostic test document"])
            self.log(f"Embedding test: {len(test_embed[0])} dimensions", "INFO")
            
            # Check particle field
            if self.field:
                particles = self.field.get_all_particles()
                self.log(f"Active particles: {len(particles)}", "INFO")
                
                # Emergency particle->memory sync if needed
                memory_particles = [p for p in particles if p.type == "memory"]
                if memory_particles:
                    self.log(f"Found {len(memory_particles)} memory particles for potential recovery", "INFO")
            
            self.log("Emergency diagnostic completed", "INFO", "emergency_memory_diagnostic")
            
        except Exception as e:
            self.log(f"Emergency diagnostic failed: {e}", "ERROR", "emergency_memory_diagnostic")

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return ParticleLikeEmbedding(documents)
    
    def emergency_save(self):
        """Emergency memory preservation during shutdown"""
        try:
            if self.field:
                field_state = self.field.get_field_state_for_database()
                if field_state:
                    self.save_field_state_to_db(field_state)
            
            adaptive_eng = api.get_api("_agent_adaptive_engine")
            if adaptive_eng:
                adaptive_eng.save_learning_state()
            
            now = datetime.now().isoformat()

            # Get collections properly
            collections_response = self.client.get_collections()
            total_points = 0
            collection_names = []
            
            # Process each collection correctly
            for collection_info in collections_response.collections:
                try:
                    # Get collection name properly
                    collection_name = collection_info.name
                    
                    # Get count using the collection name (string)
                    count = self._get_collection_count(collection_name)
                    total_points += count
                    collection_names.append(collection_name)
                    
                    self.log(f"Memory collection '{collection_name}' preserved: {count} entries", 
                            "INFO", "emergency_save")
                            
                except Exception as collection_error:
                    self.log(f"Error processing collection {collection_info}: {collection_error}", 
                            "ERROR", "emergency_save")
                    continue
            
            # Save additional state information
            base_path = self.agent_config.get("memory_dir")
            state_file = f"{base_path}/memory_shutdown_state.json"
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            shutdown_state = {
                "timestamp": now,
                "collections": collection_names,
                "total_points": total_points,
                "client_status": "connected",
                "field_state_saved": field_state is not None if self.field else False
            }
            
            with open(state_file, 'w') as f:
                json.dump(shutdown_state, f, indent=2)

            self.log(f"Emergency memory save completed: {shutdown_state['total_points']} points preserved", 
                    "INFO", "emergency_save")
            
        except Exception as e:
            self.log(f"Emergency save error: {e}", "ERROR", "emergency_save")
            import traceback
            self.log(f"Full traceback:\n{traceback.format_exc()}", "ERROR", "emergency_save")
        
    def force_save(self):
        """Force save - Qdrant handles persistence automatically"""
        try:
            collections_response = self.client.get_collections()
            total = 0
            
            for collection_info in collections_response.collections:
                try:
                    count = self._get_collection_count(collection_info.name)
                    total += count
                except Exception as e:
                    self.log(f"Error getting count for {collection_info.name}: {e}", "ERROR", "force_save")
                    continue
                    
            self.log(f"Force save check: {total} points in {len(collections_response.collections)} collections", 
                    "INFO", "force_save")
                    
        except Exception as e:
            self.log(f"Force save check failed: {e}", "ERROR", "force_save")
    
    def save_field_state_to_db(self, field_state):
        """Save particle field state to Qdrant for centralized persistence"""
        try:
            # Create field state document for embedding
            field_state_doc = {
                "type": "particle_field_state",
                "timestamp": datetime.now().isoformat(),
                "data": field_state
            }
            
            # Generate embedding from field state data
            embedding_text = f"particle field state {field_state_doc['timestamp']} {len(field_state.get('particles_summary', []))} particles"
            embedding = self.embeddings([embedding_text])[0]
            
            # Create Qdrant point with timestamp-based ID for historical tracking
            import hashlib
            timestamp_str = field_state_doc['timestamp']
            # Generate deterministic ID from timestamp for consistency
            uuid_seed = hashlib.md5(timestamp_str.encode()).hexdigest()
            timestamp_id = int(uuid_seed[:15], 16)  # Deterministic but unique per timestamp
            
            point = PointStruct(
                id=timestamp_id,  # Timestamp-based ID preserves multiple states
                vector=embedding,
                payload=field_state_doc
            )
            
            # Use Qdrant upsert (not ChromaDB syntax!)
            self.client.upsert(
                collection_name=self.system,  # Use string name
                wait=True,
                points=[point]
            )
            
            particle_count = len(field_state.get('particles_summary', []))
            self.log(f"Field state saved to Qdrant: {particle_count} particles (ID: {timestamp_id})", 
                    "INFO", "save_field_state_to_db")
            
        except Exception as e:
            self.log(f"Error saving field state to Qdrant: {e}", "ERROR", "save_field_state_to_db")
            import traceback
            self.log(f"Save error traceback:\n{traceback.format_exc()}", "ERROR", "save_field_state_to_db")
    

    def restore_field_state(self):
        """Restore particle field state from Qdrant"""
        try:
            # Get ALL field states and find the most recent
            results, _ = self.client.scroll(
                collection_name=self.system,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="type",
                            match=MatchValue(value="particle_field_state")
                        )
                    ]
                ),
                # No limit - ensure we get ALL states to find true latest
                with_payload=True
            )
            
            if results and len(results) > 0:
                # Log available states for debugging
                self.log(f"Found {len(results)} field states in Qdrant", "DEBUG", "restore_field_state")
                
                # Find most recent by timestamp
                point = max(results, key=lambda x: x.payload.get('timestamp', ''))
                field_data = point.payload.get('data')
                timestamp = point.payload.get('timestamp')
                particle_count = len(field_data.get('particles_summary', [])) if field_data else 0
                
                # Log restoration details
                self.log(f"Restoring field state: {timestamp} ({particle_count} particles)", 
                        "INFO", "restore_field_state")
                
                self.log(f"Field state restored from Qdrant: {timestamp}", 
                        "INFO", "restore_field_state")
                
                return field_data
            else:
                self.log("No field state found in Qdrant", "INFO", "restore_field_state")
                return None
                
        except Exception as e:
            self.log(f"Error restoring field state from Qdrant: {e}", "ERROR", "restore_field_state")
            import traceback
            self.log(f"Restore error traceback:\n{traceback.format_exc()}", "ERROR", "restore_field_state")
            return None

    async def link_token(self, token, definition, source, particle = None):
        """
        Link a token to its definition in memory
        """
        #entry_id = str(uuid.uuid4())
        key = token.lower()
        metadata = {
            "definition": definition,
            "source": source
        }

        result = await self.update(
            key = key,
            value = metadata,
            source = "linguistic_parsing",
            tags = ["linguistic", "definition"],
            memory_type = "lexicon",
            source_particle_id = particle.id if particle else None
        )
        self.log(f"Linked token '{token}' to definition in memory", "INFO", "link_token")
        #return entry_id
    
    async def consolidate_particle_memory(self, particle) -> bool:
        """Consolidate high-activation particle into Qdrant long-term storage"""
        try:
            if not particle or not hasattr(particle, 'id'):
                self.log("Invalid particle for consolidation", "WARNING", "consolidate_particle_memory")
                return False
            
            # Create comprehensive consolidation data
            consolidation_data = {
                "particle_id": str(particle.id),
                "original_type": getattr(particle, 'type', 'unknown'),
                "final_energy": getattr(particle, 'energy', 0),
                "final_activation": getattr(particle, 'activation', 0),
                "position": particle.position.tolist() if hasattr(particle, 'position') else None,
                "metadata": getattr(particle, 'metadata', {}),
                "quantum_state": getattr(particle, 'quantum_state', 'uncertain'),
                "linked_particles": [str(pid) for pid in getattr(particle, 'linked_particles', {}).keys()],
                "consolidation_timestamp": datetime.now().timestamp(),
                "lifespan": getattr(particle, 'age', 0),
                "total_interactions": getattr(particle, 'interaction_count', 0)
            }
            
            # Store in consolidated collection
            point_id = f"{str(particle.id)}"
            
            # Generate embedding from particle data
            embedding_text = f"consolidated {particle.type} particle {particle.id} {particle.metadata}"
            embedding = self.embeddings([str(embedding_text)])[0]
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "consolidation_data": consolidation_data,
                    "memory_type": "consolidated",
                    "consciousness_level": min(particle.energy + particle.activation, 1.0),
                    "consolidation_reason": "high_activation",
                    "searchable_content": f"Consolidated {particle.type} particle with high activation",
                    **consolidation_data  # Flatten for easy searching
                }
            )
            
            # Store in Qdrant
            self.client.upsert(
                collection_name=self.consolidated,
                wait=True,
                points=[point]
            )
            
            # Mark particle as consolidated
            if hasattr(particle, 'metadata'):
                particle.metadata['consolidated'] = True
                particle.metadata['qdrant_consolidated_id'] = point_id
                particle.metadata['consolidation_timestamp'] = datetime.now().timestamp()
            
            # Boost particle energy (survival bonus)
            if hasattr(particle, 'energy'):
                particle.energy = min(1.0, particle.energy * 1.2)
            
            self.log(f"Consolidated particle {particle.id} into Qdrant (ID: {point_id})", 
                    "INFO", "consolidate_particle_memory")
            return True
            
        except Exception as e:
            self.log(f"Error consolidating particle memory: {e}", "ERROR", "consolidate_particle_memory")
            return False

    async def consolidate_memories(self):
        """Consolidate recent memories and update long-term storage"""
        try:
                
            # Get recent high-activation particles for memory consolidation  
            if self.field:
                particles = self.field.get_all_particles()
                high_activation = [p for p in particles if hasattr(p, 'activation') and p.activation > 0.65]
                
                for particle in high_activation[:15]:  # Consolidate top 15
                    if hasattr(particle, 'metadata') and particle.metadata:
                        await self.consolidate_particle_memory(particle)

            self.log(f"Memory consolidation completed for {len(high_activation)} particles", "DEBUG", "consolidate_memories")

        except Exception as e:
            self.log(f"Memory consolidation error: {e}", "ERROR", "consolidate_memories")

    async def get_consolidated_memories(self, particle_type: str = None, limit: int = 10) -> List[Dict]:
        """Retrieve consolidated memories with filtering"""
        try:
            scroll_filter = None
            if particle_type:
                scroll_filter = Filter(
                    must=[
                        FieldCondition(
                            key="original_type",
                            match=MatchValue(value=particle_type)
                        )
                    ]
                )
            
            results, _ = self.client.scroll(
                collection_name=self.consolidated,
                scroll_filter=scroll_filter,
                limit=limit,
                with_payload=True,
                order_by="consolidation_timestamp"  # Most recent first
            )
            
            memories = []
            for point in results:
                memories.append({
                    'id': point.id,
                    'consolidation_data': point.payload.get('consolidation_data'),
                    'consciousness_level': point.payload.get('consciousness_level'),
                    'payload': point.payload
                })
            
            return memories
            
        except Exception as e:
            self.log(f"Error retrieving consolidated memories: {e}", "ERROR", "get_consolidated_memories")
            return []

