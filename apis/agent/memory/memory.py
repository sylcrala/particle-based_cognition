"""
combination of memory_backend, memory_io - central memory framework (via memory particles and vector DB (TBD))
"""

import json
import os
import uuid
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction
import random
from datetime import datetime

from apis.api_registry import api
from apis.agent.utils.embedding import ParticleLikeEmbedding

class MemoryBank:
    def __init__(self, field = None):
        self.logger = api.get_api("logger")
        self.client = PersistentClient(path="./data/agent/memory_matrix.chromadb")
        self.embeddings = ParticleLikeEmbedding()

        self.field = field

        self.memories = self.client.get_or_create_collection("memories")
        self.system_collection = self.client.get_or_create_collection("system_states")

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

    async def get_memories_by_type(self, memory_type):
        """Retrieve memories of a specific type"""
        self.log(f"Retrieving memories of type: {memory_type}", context="get_memories_by_type")
        try:
            results = self.memories.get(
                where={"type": memory_type},
                include=["documents", "metadatas"]
            )

            memories = []
            if results and results.get("documents"):
                for i, doc in enumerate(results["documents"]):
                    metadata = results.get('metadatas', [{}])[i] if results.get('metadatas') else {}
                    memories.append({
                        "content": doc,
                        "metadata": metadata,
                        "type": memory_type
                    })

            return memories
        
        except Exception as e:
            self.log(f"Error retrieving memories of type {memory_type}: {e}", level="ERROR", context="get_memories_by_type")
            return []

        

    async def quantum_memory_retrieval(self, query, collapse_threshold=0.7):
        """Retrieve memories with quantum collapse awareness"""
        self.log(f"Quantum memory retrieval for: {query}", context="quantum_memory_retrieval")
        
        
        if not self.field:
            return await self.query(query)  # Fallback to regular query
        
        # Find memory particles
        memory_particles = self.field.get_particles_by_type("memory")
        
        # Create query particle for comparison
        query_metadata = {
            "content": query,
            "source": "memory_query",
            "temporary": True
        }
        
        query_particle = await self.field.spawn_particle(
            id=None,  # Let field generate ID
            type="lingual",
            metadata=query_metadata,
            energy=0.8,
            activation=0.9,
            emit_event=False  # Don't emit for temporary particles
        )
        
        # Find relevant memories and trigger collapses
        relevant_memories = []
        
        for memory in memory_particles:
            # Calculate relevance (you can enhance this with your distance functions)
            distance = query_particle.distance_to(memory)
            relevance = max(0, 1.0 - distance)
            
            if relevance > 0.3:  # Relevance threshold
                # Trigger memory recall
                if hasattr(memory, 'observe'):
                    collapsed_state = memory.observe(context="memory_retrieval")
                    
                    # Only include certain memories in results
                    if collapsed_state == 'certain' or relevance > collapse_threshold:
                        recall_particle = await memory.trigger_recall_response(query)
                        if recall_particle:
                            relevant_memories.append(memory)
                            
                        # Create interaction linkage
                        await self.field.create_interaction_linkage(
                            query_particle.id, memory.id, "memory_retrieval"
                        )
        
        # Clean up temporary query particle
        query_particle.alive = False
        
        return relevant_memories

    async def update(self, key, value, source=None, tags=None, memory_type=None, source_particle_id=None):
        """Fixed memory storage with proper ChromaDB integration"""
        try:
            # Create memory particle first
            memory_metadata = {
                "key": key,
                "content": value,
                "source": source or "unknown",
                "tags": tags or [],
                "memory_type": memory_type or "general",
                "created_at": datetime.now().timestamp()
            }
            
            # Spawn particle with proper ID
            memory_particle = await self.field.spawn_particle(
                id=f"mem_{key}_{int(datetime.now().timestamp())}",
                type="memory",
                metadata=memory_metadata,
                source_particle_id=source_particle_id,
                emit_event=True
            )
            
            # Fix embedding generation
            if isinstance(value, dict):
                doc_text = json.dumps(value)
            else:
                doc_text = str(value)
                
            # CRITICAL FIX: Proper embedding call
            embeddings = self.embeddings.encode([doc_text])  # Fix constructor call
            
            # Store in ChromaDB with CORRECT structure
            self.memories.add(
                documents=[doc_text],
                embeddings=[embeddings[0].tolist()],  # Ensure list format
                ids=[memory_particle.id],  # Use particle ID as document ID
                metadatas=[{
                    "key": key,
                    "type": memory_type or "general",
                    "tags": json.dumps(tags or []),  # Serialize tags
                    "source": source or "unknown",
                    "particle_id": memory_particle.id,
                    "timestamp": datetime.now().timestamp()
                }]
            )
            
            self.log(f"✅ Memory successfully stored: {key}", "INFO", "update")
            return memory_particle
            
        except Exception as e:
            self.log(f"❌ CRITICAL MEMORY FAILURE: {e}", "ERROR", "update")
            import traceback
            self.log(f"Memory error traceback: {traceback.format_exc()}", "ERROR", "update")
            return None

    async def query(self, key):
        self.log(f"Querying memory for key: {key}", context="query()")
        
        # Use particle field to find memory particles
        memory_particles = self.field.get_particles_by_type("memory")
        
        matches = [p for p in memory_particles if p.metadata.get("key") == key]
        if matches:
            target = matches[-1]
            # Increase activation and energy on access
            target.energy = min(1.0, target.energy + 0.1)
            target.activation = min(1.0, target.activation + 0.1)
            return target.metadata.get("content")
        
        # If not found in active particles, check ChromaDB
        try:
            results = self.memories.query(
                query_texts=[key],
                n_results=1
            )
            if results['documents'] and len(results['documents'][0]) > 0:
                return json.loads(results['documents'][0][0])
        except Exception as e:
            self.log(f"Error querying ChromaDB: {e}", level="ERROR", context="query()")
        
        return None
      
    def get_random_memory(self):
        results = self.memories.get(
            include=["documents", "metadatas"]
        )
        if results and results.get("documents"):
            # Return a random document from the results
            if len(results["documents"]) > 0:
                idx = random.randint(0, len(results["documents"]) - 1)
                doc = results["documents"][idx]
                metadata = results.get('metadatas', [{}])[idx] if results.get('metadatas') else {}

                self.log(f"Random memory retrieved: {metadata.get('key', 'unknown key')}", context="get_random_memory()")

                return doc
        else:
            self.log("No memories found in database", level="ERROR", context="get_random_memory()")
            return None

    async def emergency_memory_diagnostic(self):
        """Comprehensive memory system health check"""
        try:
            # Check collections
            collections = self.client.list_collections()
            self.log(f"🔍 Found collections: {[c.name for c in collections]}", "INFO")
            
            # Check memory collection health
            mem_count = self.memories.count()
            self.log(f"📊 Memory entries: {mem_count}", "INFO")
            
            # Test embedding system
            test_embed = self.embeddings.encode(["test document"])
            self.log(f"🧠 Embedding test: {len(test_embed)} dimensions", "INFO")
            
            # Test particle field connection
            particles = self.field.get_all_particles() if self.field else []
            self.log(f"⚛️ Active particles: {len(particles)}", "INFO")
            
            # Attempt recovery save
            if mem_count == 0 and particles:
                self.log("🚨 ATTEMPTING EMERGENCY PARTICLE->MEMORY RECOVERY", "WARNING")
                for particle in particles[:5]:  # Save first 5 particles
                    await self.update(
                        key=f"recovery_{particle.id}",
                        value={"particle_state": str(particle.metadata)},
                        source="emergency_recovery",
                        memory_type="recovery"
                    )
                    
        except Exception as e:
            self.log(f"💥 Memory diagnostic failed: {e}", "ERROR")

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return ParticleLikeEmbedding(documents)
    
    def emergency_save(self):
        """
        Emergency memory preservation during shutdown
        """
        try:
            # Save particle field state to database (centralized approach)
            if self.field:
                field_state = self.field.get_field_state_for_database()
                if field_state:
                    self.save_field_state_to_db(field_state)
            
            # Force flush any pending chromadb operations
            collections = self.client.list_collections()
            for collection in collections:
                # ChromaDB automatically persists, but we can log the state
                count = collection.count()
                self.log(f"Memory collection '{collection.name}' preserved: {count} entries", 
                               "INFO", "emergency_save")
            
            # Save additional state information
            state_file = "./data/agent/memory_shutdown_state.json"
            shutdown_state = {
                "timestamp": datetime.now().isoformat(),
                "collections": [col.name for col in collections],
                "total_memories": sum(col.count() for col in collections),
                "field_state_saved": field_state is not None if self.field else False
            }
            
            with open(state_file, 'w') as f:
                json.dump(shutdown_state, f, indent=2)
                
            self.log(f"Emergency memory save completed: {shutdown_state['total_memories']} memories preserved", 
                           "INFO", "emergency_save")
            
        except Exception as e:
            self.log(f"Emergency save error: {e}", level="ERROR", context="emergency_save")
    
    def force_save(self):
        """
        Critical fallback save when emergency_save fails
        """
        try:
            # Minimal save - just ensure chromadb persistence
            collections = self.client.list_collections()
            total = sum(col.count() for col in collections)
            print(f"🚨 Force save: {total} memories in {len(collections)} collections")
        except Exception as e:
            print(f"❌ Force save failed: {e}")
    
    def save_field_state_to_db(self, field_state):
        """
        Save particle field state to ChromaDB for centralized persistence
        """
        try:
            # Get or create system state collection
            system_collection = self.system_collection
            # Save field state with timestamp
            field_state_doc = {
                "type": "particle_field_state",
                "timestamp": datetime.now().isoformat(),
                "data": field_state
            }
            
            # Use upsert to replace previous field state
            system_collection.upsert(
                ids=["particle_field_state"],
                documents=[json.dumps(field_state_doc)],
                metadatas=[{"type": "field_state", "timestamp": field_state_doc["timestamp"]}]
            )
            
            self.log(f"Field state saved to database: {len(field_state.get('particles_summary', []))} particles", 
                    "INFO", "save_field_state_to_db")
            
        except Exception as e:
            self.log(f"Error saving field state to database: {e}", level="ERROR", context="save_field_state_to_db")
    
    def restore_field_state(self):
        """
        Restore particle field state from ChromaDB
        """
        try:
            # Get system state collection
            system_collection = self.client.get_collection("system_states")
            
            # Query for latest field state
            results = system_collection.get(
                ids=["particle_field_state"]
            )
            
            if results['documents'] and len(results['documents']) > 0:
                field_state_doc = json.loads(results['documents'][0])
                field_data = field_state_doc.get('data')
                
                self.log(f"Field state restored from database: {field_state_doc.get('timestamp')}", 
                        "INFO", "restore_field_state")
                
                return field_data
            else:
                self.log("No field state found in database", "INFO", "restore_field_state")
                return None
                
        except Exception as e:
            self.log(f"Error restoring field state: {e}", level="ERROR", context="restore_field_state")
            return None

    async def link_token(self, token, definition, source, particle = None):
        """
        Link a token to its definition in memory
        """
        entry_id = str(uuid.uuid4())
        key = token.lower()
        metadata = {
            "definition": definition,
            "source": source
        }

        self.update(
            key = key,
            value = metadata,
            source = "linguistic_parsing",
            tags = ["linguistic", "definition"],
            memory_type = "linguistic",
            source_particle_id = particle.id if particle else None
        )
        self.log(f"Linked token '{token}' to definition in memory", "INFO", "link_token")
        return entry_id
    
    async def consolidate_particle_memory(self, particle):
        """Consolidate high-activation particle into long-term memory storage"""
        try:
            if not particle or not hasattr(particle, 'id'):
                self.log("Invalid particle for consolidation", "WARNING", "consolidate_particle_memory")
                return False
                
            # Extract particle data for consolidation
            consolidation_data = {
                "particle_id": particle.id,
                "type": getattr(particle, 'type', 'unknown'),
                "energy": getattr(particle, 'energy', 0),
                "activation": getattr(particle, 'activation', 0),
                "position": particle.position.tolist() if hasattr(particle, 'position') else None,
                "metadata": getattr(particle, 'metadata', {}),
                "quantum_state": getattr(particle, 'quantum_state', 'uncertain'),
                "linked_particles": getattr(particle, 'linked_particles', {}),
                "consolidation_timestamp": datetime.now().timestamp()
            }
            
            # Create consolidated memory entry
            memory_key = f"consolidated_particle_{particle.id}_{int(datetime.now().timestamp())}"
            memory_content = f"Consolidated memory from {particle.type} particle {particle.id}"
            
            # Add consolidation metadata
            tags = ["consolidated", "high_activation", particle.type]
            if hasattr(particle, 'metadata') and particle.metadata:
                if particle.metadata.get('tags'):
                    tags.extend(particle.metadata['tags'])
            
            # Store in memory database
            await self.update(
                key=memory_key,
                value=memory_content,
                links=[particle.id],
                tags=tags,
                memory_type="consolidated",
                source="particle_consolidation",
                source_particle_id=particle.id,
                metadata=consolidation_data
            )
            
            # Mark particle as consolidated
            if hasattr(particle, 'metadata'):
                particle.metadata['consolidated'] = True
                particle.metadata['consolidation_timestamp'] = datetime.now().timestamp()
            
            # Boost particle energy for being consolidated (survival bonus)
            if hasattr(particle, 'energy'):
                particle.energy = min(1.0, particle.energy * 1.2)
                
            self.log(f"Consolidated particle {particle.id} into memory {memory_key}", 
                    "INFO", "consolidate_particle_memory")
            return True
            
        except Exception as e:
            self.log(f"Error consolidating particle memory: {e}", "ERROR", "consolidate_particle_memory")
            return False

    async def get_consolidated_memories(self, particle_type=None, limit=10):
        """Retrieve consolidated particle memories"""
        try:
            # Query consolidated memories
            memories = await self.get_memories_by_type("consolidated")
            
            if particle_type:
                # Filter by original particle type
                filtered_memories = []
                for memory in memories:
                    if memory.get('metadata', {}).get('type') == particle_type:
                        filtered_memories.append(memory)
                memories = filtered_memories
            
            # Sort by consolidation timestamp (most recent first)
            memories.sort(key=lambda m: m.get('metadata', {}).get('consolidation_timestamp', 0), reverse=True)
            
            return memories[:limit] if limit else memories
            
        except Exception as e:
            self.log(f"Error retrieving consolidated memories: {e}", "ERROR", "get_consolidated_memories")
            return []

