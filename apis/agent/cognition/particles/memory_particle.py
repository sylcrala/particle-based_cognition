"""
Particle-based Cognition Engine - memory particles, for managing memory storage and retrieval
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

from datetime import datetime
import uuid
import json

from apis.agent.cognition.particles.utils.particle_frame import Particle
from apis.api_registry import api


class MemoryParticle(Particle):
    def __init__(self, **kwargs):
        super().__init__(type="memory", **kwargs)
        self.token = self.metadata.get("content", "")
        self.embedding = None
        self.last_accessed = datetime.now().timestamp()
        self.retrieval_count = self.metadata.setdefault("retrieval_count", 0)
        self.importance = self.metadata.setdefault("importance", 0.5)

        self.memory_bank = api.get_api("_agent_memory")

        self.qdrant_point_id = self.metadata.get("qdrant_point_id")
        self.memory_type = self.metadata.get("memory_type", "general")
        self.consciousness_level = self.metadata.get("consciousness_level", 0.5)

    def append_content(self, value):
        self.token += ", " + value

    async def update_self(self, new_value):
        """Enhanced update with Qdrant integration"""
        try:
            # Update content
            if isinstance(new_value, str):
                self.append_content(new_value)
            else:
                self.token = str(new_value)
                self.metadata["content"] = self.token
            
            # Update embedding
            self.embedding = self._message_to_vector(self.token)
            
            # Update in Qdrant with consciousness tracking
            if self.memory_bank:
                result = await self.memory_bank.update(
                    key=self.metadata.get("key", f"memory_{self.id}"),
                    value=self.token,
                    source="memory_particle_update",
                    tags=self.metadata.get("tags", []) + ["updated"],
                    memory_type=self.memory_type,
                    source_particle_id=self.id,
                    consciousness_level=min(1.0, self.consciousness_level + 0.1),  # Boost on update
                    energy_level=self.energy,
                    activation_level=self.activation,
                    quantum_state=getattr(self, 'quantum_state', 'uncertain'),
                    update_timestamp=datetime.now().isoformat()
                )
                
                # Update Qdrant point ID if returned
                if result and hasattr(result, 'id'):
                    self.qdrant_point_id = result.id
                    self.metadata["qdrant_point_id"] = result.id
                
                self.log(f"Memory particle {self.id} updated in Qdrant", "INFO", "update_self")
            
            # Boost energy on successful update
            self.energy = min(1.0, self.energy + 0.05)
            
        except Exception as e:
            self.log(f"Error updating memory particle: {e}", "ERROR", "update_self")


    
    async def retrieve_related(self, k=5, similarity_threshold=0.7):
        """Enhanced retrieval with Qdrant semantic search"""
        try:
            if not self.memory_bank:
                self.log("No memory bank available for retrieval", "WARNING", "retrieve_related")
                return []
            
            # Use Qdrant semantic search
            results = await self.memory_bank.search_memories_by_embedding(
                self.embedding, 
                limit=k
            )
            
            related_particles = []
            for memory_data in results:
                # Skip self
                if memory_data.get('id') == str(self.id):
                    continue
                
                # Only include high similarity memories
                if memory_data.get('similarity_score', 0) < similarity_threshold:
                    continue
                
                # Create related memory particle
                related_content = {
                    "key": memory_data.get('key', f"related_{uuid.uuid4()}"),
                    "content": memory_data.get('value', ''),
                    "source": memory_data.get('payload', {}).get('source', 'unknown'),
                    "tags": memory_data.get('payload', {}).get('tags', []),
                    "similarity_score": memory_data.get('similarity_score'),
                    "original_memory_id": memory_data.get('id')
                }
                
                related_particle = await self.create_linked_particle(
                    "memory", 
                    related_content, 
                    relationship_type="semantic_similarity"
                )
                
                if related_particle:
                    related_particles.append(related_particle)
            
            self.log(f"Retrieved {len(related_particles)} related memories", "INFO", "retrieve_related")
            return related_particles
            
        except Exception as e:
            self.log(f"Error retrieving related memories: {e}", "ERROR", "retrieve_related")
            return []


    async def reflect(self):
        """Memory reflection with deep analysis"""
        try:
            related = await self.retrieve_related(k=3)
            
            # Create reflection content
            if related:
                reflection_content = f"Memory '{self.token[:50]}...' connects to {len(related)} related memories: "
                connection_themes = []
                
                for rel_particle in related:
                    similarity = rel_particle.metadata.get('similarity_score', 0)
                    connection_themes.append(f"[{similarity:.2f}] {rel_particle.token[:30]}...")
                
                reflection_content += "; ".join(connection_themes)
            else:
                reflection_content = f"Memory '{self.token[:50]}...' appears to be unique with no strong connections found."
            
            # Store reflection in memory system
            if self.memory_bank:
                reflection_result = await self.memory_bank.update(
                    key=f"reflection_{str(uuid.uuid4())}",
                    value=reflection_content,
                    source="memory_particle_reflection",
                    tags=["reflection", "memory_analysis", self.memory_type],
                    memory_type="reflection",
                    source_particle_id=self.id,
                    consciousness_level=min(1.0, self.consciousness_level + 0.2),
                    reflection_depth=len(related),
                    parent_memory_id=str(self.id)
                )
                
                self.log(f"Memory reflection created: {len(related)} connections analyzed", 
                        "INFO", "reflect")
                return reflection_result
            
        except Exception as e:
            self.log(f"Error during memory reflection: {e}", "ERROR", "reflect")
            return None


    async def create_linked_particle(self, particle_type, content, relationship_type="triggered"):
        """Enhanced particle creation with consciousness inheritance"""
        try:
            # Better metadata structure
            metadata = {
                "content": content if isinstance(content, str) else content.get("content", ""),
                "triggered_by": str(self.id),
                "relationship": relationship_type,
                "source": "memory_particle_creation",
                "parent_memory_type": self.memory_type,
                "inheritance_level": self.consciousness_level * 0.8,  # Inherit reduced consciousness
                **({} if isinstance(content, str) else content)  # Merge additional metadata
            }
            
            # Smarter energy calculation
            base_energy = 0.6
            importance_bonus = self.importance * 0.3
            consciousness_bonus = self.consciousness_level * 0.2
            energy = min(1.0, base_energy + importance_bonus + consciousness_bonus)
            
            activation = min(1.0, 0.5 + (self.importance * 0.2) + (self.retrieval_count * 0.05))
            
            new_particle = await self.field.spawn_particle(
                type=particle_type,
                metadata=metadata,
                energy=energy,
                activation=activation,
                source_particle_id=self.id,
                emit_event=True
            )
            
            if new_particle:
                # Establish bidirectional linking
                if not hasattr(self, 'linked_particles'):
                    self.linked_particles = {}
                self.linked_particles[str(new_particle.id)] = {
                    'type': particle_type,
                    'relationship': relationship_type,
                    'created_at': datetime.now().timestamp()
                }
                
                self.log(f"Created linked {particle_type} particle {new_particle.id}", 
                        "INFO", "create_linked_particle")
            
            return new_particle
            
        except Exception as e:
            self.log(f"Error creating linked particle: {e}", "ERROR", "create_linked_particle")
            return None

    async def trigger_recall_response(self, query_context):
        """Enhanced memory recall with quantum state awareness"""
        try:
            # Quantum observation for recall
            if hasattr(self, 'observe'):
                collapsed_state = self.observe(context="memory_recall")
                self.log(f"Memory quantum state collapsed to: {collapsed_state}", "DEBUG", "trigger_recall_response")
            
            # Create recall particle with enhanced metadata
            recall_content = {
                "content": self.token,
                "recall_context": query_context,
                "memory_source_id": str(self.id),
                "recall_confidence": min(1.0, self.importance + (self.retrieval_count * 0.1)),
                "quantum_state_at_recall": getattr(self, 'quantum_state', 'uncertain'),
                "memory_age": datetime.now().timestamp() - self.metadata.get('created_at', datetime.now().timestamp()),
                "retrieval_history": self.retrieval_count
            }
            
            recall_particle = await self.create_linked_particle(
                particle_type="lingual",
                content=recall_content,
                relationship_type="memory_recall"
            )
            
            if recall_particle:
                # Set high certainty for successful recall
                if hasattr(recall_particle, 'update_superposition'):
                    certainty_boost = min(0.5, 0.2 + (self.importance * 0.3))
                    recall_particle.update_superposition(certainty_delta=certainty_boost)
                
                # Update recall statistics
                self.mark_accessed()
                
                self.log(f"Memory recall triggered for query: {query_context[:30]}...", 
                        "INFO", "trigger_recall_response")
            
            return recall_particle
            
        except Exception as e:
            self.log(f"Error triggering memory recall: {e}", "ERROR", "trigger_recall_response")
            return None

    def mark_accessed(self):
        """Enhanced access tracking with consciousness boost"""
        self.last_accessed = datetime.now().timestamp()
        self.retrieval_count += 1
        
        # Boost importance and consciousness on access
        access_boost = min(0.1, 0.01 + (1.0 / (self.retrieval_count + 1)))
        self.importance = min(1.0, self.importance + access_boost)
        self.consciousness_level = min(1.0, self.consciousness_level + (access_boost * 0.5))
        
        # Update metadata
        self.metadata["last_accessed"] = self.last_accessed
        self.metadata["retrieval_count"] = self.retrieval_count
        self.metadata["importance"] = self.importance
        self.metadata["consciousness_level"] = self.consciousness_level
        
        self.log(f"Memory accessed - count: {self.retrieval_count}, importance: {self.importance:.3f}", 
                "DEBUG", "mark_accessed")

    def compare_and_merge(self, other):
        """Enhanced merging with consciousness consideration"""
        try:
            if not self.embedding or not other.embedding:
                return False
            
            similarity = self._cosine_similarity(self.embedding, other.embedding)
            
            # Dynamic similarity threshold based on consciousness
            consciousness_factor = (self.consciousness_level + other.consciousness_level) / 2
            threshold = 0.85 - (consciousness_factor * 0.1)  # Higher consciousness = easier merging
            
            if similarity > threshold:
                # Merge content
                self.append_content(other.token)
                
                # Merge importance and consciousness
                self.importance = max(self.importance, other.importance)
                self.consciousness_level = min(1.0, (self.consciousness_level + other.consciousness_level) / 2 + 0.1)
                
                # Merge retrieval counts
                self.retrieval_count += other.retrieval_count
                
                # Merge tags and metadata
                self_tags = set(self.metadata.get("tags", []))
                other_tags = set(other.metadata.get("tags", []))
                self.metadata["tags"] = list(self_tags.union(other_tags))
                
                # Track the merge
                self.metadata.setdefault("merged_particles", []).append({
                    "particle_id": str(other.id),
                    "merge_time": datetime.now().timestamp(),
                    "similarity_score": similarity
                })
                
                self.log(f"Successfully merged with particle {other.id} (similarity: {similarity:.3f})", 
                        "INFO", "compare_and_merge")
                return True
            
            return False
            
        except Exception as e:
            self.log(f"Error comparing and merging particles: {e}", "ERROR", "compare_and_merge")
            return False
        
    async def consolidate_to_long_term(self):
        """Consolidate this memory particle to long-term Qdrant storage"""
        try:
            if self.memory_bank and hasattr(self.memory_bank, 'consolidate_particle_memory'):
                success = await self.memory_bank.consolidate_particle_memory(self)
                if success:
                    self.metadata['consolidated'] = True
                    self.log(f"Memory particle {self.id} consolidated to long-term storage", 
                            "INFO", "consolidate_to_long_term")
                return success
            return False
            
        except Exception as e:
            self.log(f"Error consolidating memory: {e}", "ERROR", "consolidate_to_long_term")
            return False
        
    def get_memory_summary(self):
        """Get comprehensive summary of memory particle state"""
        try:
            return {
                "id": str(self.id),
                "content_preview": self.token[:100] + "..." if len(self.token) > 100 else self.token,
                "memory_type": self.memory_type,
                "importance": self.importance,
                "consciousness_level": self.consciousness_level,
                "retrieval_count": self.retrieval_count,
                "energy": self.energy,
                "activation": self.activation,
                "last_accessed": self.last_accessed,
                "quantum_state": getattr(self, 'quantum_state', 'uncertain'),
                "qdrant_stored": self.qdrant_point_id is not None,
                "linked_particles": len(getattr(self, 'linked_particles', {})),
                "tags": self.metadata.get("tags", []),
                "age": datetime.now().timestamp() - self.metadata.get('created_at', datetime.now().timestamp())
            }
            
        except Exception as e:
            self.log(f"Error generating memory summary: {e}", "ERROR", "get_memory_summary")
            return {"id": str(self.id), "error": str(e)}