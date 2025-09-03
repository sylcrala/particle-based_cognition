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
    def __init__(self):
        self.logger = api.get_api("logger")
        self.client = PersistentClient(path="./data/agent/memory_matrix.chromadb")
        self.embeddings = ParticleLikeEmbedding()

        self.memories = self.client.get_or_create_collection("memories")

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

    async def quantum_memory_retrieval(self, query, collapse_threshold=0.7):
        """Retrieve memories with quantum collapse awareness"""
        self.log(f"Quantum memory retrieval for: {query}", context="quantum_memory_retrieval")
        
        field_api = api.get_api("particle_field")
        if not field_api:
            return await self.query(query)  # Fallback to regular query
        
        # Find memory particles
        memory_particles = field_api.get_particles_by_type("memory")
        
        # Create query particle for comparison
        query_metadata = {
            "content": query,
            "source": "memory_query",
            "temporary": True
        }
        
        query_particle = await field_api.spawn_particle(
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
                        await field_api.create_interaction_linkage(
                            query_particle.id, memory.id, "memory_retrieval"
                        )
        
        # Clean up temporary query particle
        query_particle.alive = False
        
        return relevant_memories

    async def update(self, key, value, links = None, source = None, tags=None, memory_type=None, source_particle_id=None):
        # Create memory particle via ParticleField for proper linkage tracking
        field_api = api.get_api("particle_field")
        
        memory_metadata = {
            "key": key,
            "content": value,
            "source": source,
            "tags": tags or [],
            "memory_type": memory_type or "general",
            "created_at": datetime.now().timestamp()
        }
        
        # Spawn particle with linkage tracking
        memory_particle = await field_api.spawn_particle(
            id=None,  # Let field generate ID
            type="memory",
            metadata=memory_metadata,
            source_particle_id=source_particle_id,  # Track what triggered this memory
            emit_event=True
        )
        
        # Store in ChromaDB
        doc = json.dumps(value)
        embedding = self.embed_documents([doc])[0]

        self.log(f"[MemoryBank] Created memory particle for key '{key}'", context="add()")

        links.append(memory_particle.id)
        if source_particle_id:
            links.append(source_particle_id)

        self.memories.add(
            documents=[doc],
            embeddings=[embedding],
            ids=[str(uuid.uuid4())],
            metadatas=[{
                "key": key,
                "tags": tags or [],
                "type": memory_type,
                "linked_particles": links,
                "source": source,
                "timestamp": datetime.now().timestamp()
            }]
        )

        return memory_particle


    async def query(self, key):
        self.log(f"Querying memory for key: {key}", context="query()")
        
        # Use particle field to find memory particles
        field_api = api.get_api("particle_field")
        memory_particles = field_api.get_particles_by_type("memory")
        
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
      

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return ParticleLikeEmbedding(documents)
    
    def emergency_save(self):
        """
        Emergency memory preservation during shutdown
        """
        try:
            # Save particle field state to database (centralized approach)
            field_api = api.get_api("particle_field")
            if field_api:
                field_state = field_api.get_field_state_for_database()
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
                "field_state_saved": field_state is not None if field_api else False
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
            try:
                system_collection = self.client.get_collection("system_states")
            except:
                system_collection = self.client.create_collection("system_states")
            
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

# Register the API
api.register_api("memory_bank", MemoryBank())
