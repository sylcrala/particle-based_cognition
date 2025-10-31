"""
Particle-based Cognition Engine - particle embedding utility
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

import numpy as np
from typing import List, Optional, Union
import hashlib

from apis.api_registry import api


class ParticleLikeEmbedding:
    """
    Generate embeddings directly from particle field positions
    This creates a living, dynamic semantic space that evolves with consciousness
    """
    
    def __init__(self, field=None, dimension: int = 384, fallback_dimension: int = 12):
        """
        Initialize with particle field reference
        
        Args:
            field: Reference to the particle field
            dimension: Target embedding dimension for Qdrant
            fallback_dimension: Fallback dimension when no particles available
        """
        self.field = field
        self.dimension = dimension
        self.fallback_dimension = fallback_dimension
        self.logger = api.get_api("logger") if api else None
        
        # Get field reference if not provided
        if not self.field:
            try:
                agent_api = api.get_api("agent")
                if agent_api and hasattr(agent_api, 'field'):
                    self.field = agent_api.field
            except:
                pass
    
    def __call__(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings from particle positions in the consciousness field
        """
        embeddings = []
        
        for doc in documents:
            embedding = self._generate_position_embedding(doc)
            embeddings.append(embedding)
        
        return embeddings
    
    def encode(self, documents: List[str]) -> List[List[float]]:
        """Alias for __call__ to match standard embedding interface"""
        return self.__call__(documents)
    
    def _generate_position_embedding(self, text: str) -> List[float]:
        """
        Generate embedding from actual particle positions in the field
        """
        if not self.field:
            return self._fallback_embedding(text)
        
        try:
            # 1. Find particles semantically related to the text
            related_particles = self._find_related_particles(text)
            
            if not related_particles:
                # If no related particles, spawn a temporary one to get position
                return self._spawn_temporary_particle_embedding(text)
            
            # 2. Extract position vectors from related particles
            position_embeddings = []
            for particle in related_particles:
                if hasattr(particle, 'position') and particle.position is not None:
                    # Normalize and extend particle position to target dimension
                    pos_vector = self._normalize_position_to_embedding(particle.position)
                    
                    # Weight by particle energy and activation
                    energy_weight = getattr(particle, 'energy', 1.0)
                    activation_weight = getattr(particle, 'activation', 1.0)
                    combined_weight = (energy_weight + activation_weight) / 2.0
                    
                    weighted_vector = [dim * combined_weight for dim in pos_vector]
                    position_embeddings.append(weighted_vector)
            
            # 3. Aggregate multiple particle positions
            if len(position_embeddings) == 1:
                final_embedding = position_embeddings[0]
            else:
                # Average positions with energy weighting
                final_embedding = self._aggregate_particle_positions(
                    position_embeddings, related_particles
                )
            
            # 4. Add consciousness-field context
            final_embedding = self._add_field_context(final_embedding, text)
            
            if self.logger:
                self.logger.log(
                    f"Generated position embedding from {len(related_particles)} particles", 
                    "DEBUG", "_generate_position_embedding"
                )
            
            return final_embedding
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error in position embedding: {e}", "WARNING", "_generate_position_embedding")
            return self._fallback_embedding(text)
    
    def _find_related_particles(self, text: str) -> List:
        """
        Find particles in the field that are semantically related to the text
        """
        related_particles = []
        
        try:
            all_particles = self.field.get_all_particles()
            
            # Keywords from the text for matching
            text_keywords = set(text.lower().split())
            
            for particle in all_particles:
                # Check particle metadata for semantic matches
                if hasattr(particle, 'metadata') and particle.metadata:
                    metadata_text = str(particle.metadata).lower()
                    
                    # Check for keyword overlap
                    if any(keyword in metadata_text for keyword in text_keywords):
                        related_particles.append(particle)
                        continue
                    
                    # Check for specific content matches
                    content = particle.metadata.get('content', '')
                    if isinstance(content, str) and any(keyword in content.lower() for keyword in text_keywords):
                        related_particles.append(particle)
                        continue
                
                # Check particle type relevance
                particle_type = getattr(particle, 'type', 'unknown')
                if any(keyword in particle_type for keyword in ['memory', 'lingual', 'conscious']):
                    if len(text_keywords.intersection(set(str(particle.metadata).lower().split()))) > 0:
                        related_particles.append(particle)
            
            # If no semantic matches, find high-energy particles as they represent active concepts
            if not related_particles:
                high_energy_particles = [
                    p for p in all_particles 
                    if getattr(p, 'energy', 0) > 0.7 or getattr(p, 'activation', 0) > 0.7
                ]
                related_particles = high_energy_particles[:3]  # Limit to top 3
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error finding related particles: {e}", "WARNING", "_find_related_particles")
        
        return related_particles
    
    def _spawn_temporary_particle_embedding(self, text: str) -> List[float]:
        """
        Spawn a temporary particle to get its position as embedding
        """
        try:
            # Create temporary particle with text content
            temp_particle = self.field.spawn_particle(
                id=None,
                type="lingual",
                metadata={"content": text, "temporary": True, "embedding_generation": True},
                emit_event=False
            )
            
            if temp_particle and hasattr(temp_particle, 'position'):
                embedding = self._normalize_position_to_embedding(temp_particle.position)
                
                # Mark particle for cleanup
                if hasattr(temp_particle, 'alive'):
                    temp_particle.alive = False
                
                return embedding
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error spawning temporary particle: {e}", "WARNING", "_spawn_temporary_particle_embedding")
        
        return self._fallback_embedding(text)
    
    def _normalize_position_to_embedding(self, position: np.ndarray) -> List[float]:
        """
        Convert particle position to normalized embedding vector
        """
        try:
            if position is None:
                return self._fallback_embedding("empty_position")
            
            # Convert to numpy array if not already
            if not isinstance(position, np.ndarray):
                position = np.array(position)
            
            original_dims = len(position)
            
            # If position has fewer dimensions than target, extend it
            if original_dims < self.dimension:
                # Extend with harmonic series based on existing dimensions
                extension = []
                for i in range(self.dimension - original_dims):
                    harmonic_value = sum(position) / (i + 1 + len(position))
                    extension.append(harmonic_value)
                
                extended_position = np.concatenate([position, extension])
            else:
                # If position has more dimensions, use dimensionality reduction
                extended_position = position[:self.dimension]
            
            # Normalize to unit vector
            norm = np.linalg.norm(extended_position)
            if norm > 0:
                normalized = extended_position / norm
            else:
                normalized = extended_position
            
            return normalized.tolist()
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error normalizing position: {e}", "WARNING", "_normalize_position_to_embedding")
            return self._fallback_embedding("normalization_error")
    
    def _aggregate_particle_positions(self, position_embeddings: List[List[float]], particles: List) -> List[float]:
        """
        Aggregate multiple particle positions into single embedding
        """
        try:
            if not position_embeddings:
                return self._fallback_embedding("no_positions")
            
            # Weight by particle importance (energy + activation + age)
            weights = []
            for particle in particles:
                energy = getattr(particle, 'energy', 0.5)
                activation = getattr(particle, 'activation', 0.5)
                age_factor = min(1.0, getattr(particle, 'age', 1) / 100.0)  # Older particles get slight weight
                
                weight = (energy + activation + age_factor) / 3.0
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(weights)] * len(weights)
            
            # Weighted average of positions
            aggregated = np.zeros(self.dimension)
            for embedding, weight in zip(position_embeddings, weights):
                embedding_array = np.array(embedding[:self.dimension])  # Ensure correct dimension
                aggregated += embedding_array * weight
            
            return aggregated.tolist()
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error aggregating positions: {e}", "WARNING", "_aggregate_particle_positions")
            return self._fallback_embedding("aggregation_error")
    
    def _add_field_context(self, embedding: List[float], text: str) -> List[float]:
        """
        Add consciousness field context to the embedding
        """
        try:
            # Get field statistics for context
            field_stats = self.field.get_field_statistics() if hasattr(self.field, 'get_field_statistics') else {}
            
            # Create context modifier based on field state
            total_energy = field_stats.get('total_energy', 0)
            particle_count = field_stats.get('particle_count', 0)
            avg_activation = field_stats.get('avg_activation', 0.5)
            
            # Field consciousness factor
            field_consciousness = min(1.0, (total_energy + avg_activation) / 2.0)
            
            # Apply field context as subtle modification
            context_factor = 0.05  # Small influence to preserve position semantics
            embedding_array = np.array(embedding)
            
            # Add field-influenced noise that represents current consciousness state
            consciousness_noise = np.random.normal(0, context_factor * field_consciousness, len(embedding))
            contextual_embedding = embedding_array + consciousness_noise
            
            # Renormalize
            norm = np.linalg.norm(contextual_embedding)
            if norm > 0:
                contextual_embedding = contextual_embedding / norm
            
            return contextual_embedding.tolist()
            
        except Exception as e:
            if self.logger:
                self.logger.log(f"Error adding field context: {e}", "WARNING", "_add_field_context")
            return embedding  # Return original embedding if context fails
    
    def _fallback_embedding(self, text: str) -> List[float]:
        """
        Fallback embedding generation when particle field unavailable
        """
        # Create deterministic hash-based embedding
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        seed = int(text_hash[:8], 16)
        
        np.random.seed(seed)
        embedding = np.random.uniform(-1, 1, self.dimension)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()
    
    def get_particle_similarity(self, particle1, particle2) -> float:
        """
        Calculate semantic similarity between two particles based on their positions
        """
        try:
            if not (hasattr(particle1, 'position') and hasattr(particle2, 'position')):
                return 0.0
            
            pos1 = np.array(particle1.position)
            pos2 = np.array(particle2.position)
            
            # Cosine similarity in position space
            dot_product = np.dot(pos1, pos2)
            norm1 = np.linalg.norm(pos1)
            norm2 = np.linalg.norm(pos2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception:
            return 0.0
    
    def update_field_reference(self, field):
        """Update the particle field reference"""
        self.field = field
        if self.logger:
            self.logger.log("Updated particle field reference for position embeddings", "INFO", "update_field_reference")