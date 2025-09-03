"""
memory particle handles the storage, retrieval, and management of information within the agent
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
        self.embedding = self._message_to_vector(self.token)
        self.last_accessed = datetime.now().timestamp()
        self.retrieval_count = self.metadata.setdefault("retrieval_count", 0)
        self.importance = self.metadata.setdefault("importance", 0.5)



    async def update_self(self, memory_bank, new_value):
        self.append_content(new_value)
        self.embedding = self._message_to_vector(self.token)
        await memory_bank.update(
            key=self.metadata["key"],
            value=self.token,
            source="reflection",
            tags=self.metadata.get("tags", [])
        )

    
    async def retrieve_related(self, memory_bank, k=5):
        results = memory_bank.query_memory_by_vector(self.embedding, k)
        related_particles = []
        for doc, meta in results:
            p = await MemoryParticle.spawn_from_bank(
                engine=memory_bank.particle_engine,
                key=meta["key"],
                value=json.loads(doc),
                persistent=(meta.get("layer") == "core")
            )
            related_particles.append(p)
        return related_particles


    async def reflect(self, memory_bank):
        related = await self.retrieve_related(memory_bank)
        reflection = f"'{self.token}' relates to {len(related)} memory fragments."
        await memory_bank.update(
            key=f"reflection:{uuid.uuid4()}",
            value=reflection,
            source="reflection"
        )


    async def create_linked_particle(self, particle_type, content, relationship_type="triggered"):
        """Create a new particle linked to this memory particle"""
        field_api = api.get_api("particle_field")
        if not field_api:
            return None
            
        metadata = {
            "content": content,
            "triggered_by": self.id,
            "relationship": relationship_type,
            "source": "memory_particle_creation"
        }
        
        # Memory particles create confident offspring
        energy = 0.6 + (self.importance * 0.3)
        activation = 0.5 + (self.importance * 0.2)
            
        return await field_api.spawn_particle(
            type=particle_type,
            metadata=metadata,
            energy=energy,
            activation=activation,
            source_particle_id=self.id,
            emit_event=True
        )

    async def trigger_recall_response(self, query_context):
        """Create lingual particle for memory recall response"""
        # Memory recall should be relatively certain
        if hasattr(self, 'observe'):
            collapsed_state = self.observe(context="memory_recall")
            
        recall_particle = await self.create_linked_particle(
            particle_type="lingual",
            content=self.token,
            relationship_type="memory_recall"
        )
        
        if recall_particle:
            recall_particle.metadata["recall_context"] = query_context
            recall_particle.metadata["memory_source"] = self.id
            
            # High certainty for memory recall
            if hasattr(recall_particle, 'update_superposition'):
                recall_particle.update_superposition(certainty_delta=0.4)
                
        return recall_particle

    def mark_accessed(self):
        self.last_accessed = datetime.now().timestamp()
        self.retrieval_count += 1


    def compare_and_merge(self, other):
        similarity = self._cosine_similarity(self.embedding, other.embedding)
        if similarity > 0.85:
            self.append_content(other.token)
            self.importance = max(self.importance, other.importance)
            return True
        return False


