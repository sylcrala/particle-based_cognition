"""
Particle-based Cognition Engine - adaptive inference engine for reasoning, analysis, and knowledge integration
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
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
import asyncio
from apis.api_registry import api


class InferenceChain:
    """Represents a multi-hop inference with confidence tracking"""
    
    def __init__(self, particles: List, reasoning_type: str = "deductive"):
        self.particles = particles
        self.reasoning_type = reasoning_type  # deductive, inductive, abductive
        self.confidence = 0.0
        self.timestamp = datetime.now()
        self.validated = False
        
    def calculate_confidence(self):
        """Calculate confidence based on particle properties and chain coherence"""
        if not self.particles:
            return 0.0
        
        # Average activation across chain
        avg_activation = sum(p.activation for p in self.particles) / len(self.particles)
        
        # Average energy (higher energy = stronger belief)
        avg_energy = sum(p.energy for p in self.particles) / len(self.particles)
        
        # Chain coherence (position distance variance)
        coherence = self._calculate_coherence()
        
        # Temporal coherence (particles created closer together = stronger connection)
        temporal_coherence = self._calculate_temporal_coherence()
        
        # Weighted combination
        self.confidence = (
            avg_activation * 0.35 +
            avg_energy * 0.25 +
            coherence * 0.25 +
            temporal_coherence * 0.15
        )
        
        return self.confidence
    
    def _calculate_coherence(self):
        """Measure spatial coherence of particle positions"""
        if len(self.particles) < 2:
            return 1.0
        
        positions = [p.position[:3] for p in self.particles]
        distances = []
        for i in range(len(positions) - 1):
            dist = np.linalg.norm(
                np.array(positions[i]) - np.array(positions[i+1])
            )
            distances.append(dist)
        
        avg_distance = sum(distances) / len(distances)
        # Closer particles = higher coherence
        return 1.0 / (1.0 + avg_distance)
    
    def _calculate_temporal_coherence(self):
        """Measure temporal coherence (creation time proximity)"""
        if len(self.particles) < 2:
            return 1.0
        
        creation_times = [p.position[3] for p in self.particles]  # w dimension
        time_diffs = []
        for i in range(len(creation_times) - 1):
            diff = abs(creation_times[i] - creation_times[i+1])
            time_diffs.append(diff)
        
        avg_diff = sum(time_diffs) / len(time_diffs)
        # Smaller time gaps = higher temporal coherence
        return 1.0 / (1.0 + avg_diff * 0.001)  # Scale factor for timestamps
    
    def to_dict(self):
        """Export chain as dictionary"""
        return {
            "particle_ids": [str(p.id) for p in self.particles],
            "reasoning_type": self.reasoning_type,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "validated": self.validated,
            "content_chain": " → ".join([
                str(p.metadata.get("content", ""))[:50] 
                for p in self.particles
            ])
        }


class InferenceEngine:
    """
    Advanced reasoning engine for experiential learning and inference
    """
    
    def __init__(self):
        self.logger = api.get_api("logger")
        self.field = None  # Set on initialization
        self.memory_bank = None
        self.lexicon_store = None
        self.adaptive_engine = None
        
        self.inference_history = []
        self.contradiction_log = []
        self.concept_hierarchy = {}  # Abstract concepts → concrete particles
        
    def initialize(self):
        """Initialize engine with API references"""
        self.field = api.get_api("_agent_field")
        self.memory_bank = api.get_api("_agent_memory")
        self.lexicon_store = api.get_api("_agent_lexicon")
        self.adaptive_engine = api.get_api("_agent_adaptive_engine")
        
        if not all([self.field, self.memory_bank, self.lexicon_store]):
            self.log("Warning: Some APIs not available during initialization", "WARNING")
    
    def log(self, message, level="INFO", context="InferenceEngine"):
        """Logging wrapper"""
        if self.logger:
            self.logger.log(message, level, context, "InferenceEngine")
    
    # ==================== MULTI-HOP INFERENCE ====================
    
    async def perform_multi_hop_inference(
        self, 
        start_particles: List = None, 
        max_depth: int = 3,
        min_confidence: float = 0.5
    ) -> List[InferenceChain]:
        """
        Perform multi-hop reasoning by traversing particle linkages
        
        Args:
            start_particles: Starting particles (or auto-select high-activation ones)
            max_depth: Maximum hops in inference chain
            min_confidence: Minimum confidence threshold for returned chains
        
        Returns:
            List of validated InferenceChain objects
        """
        if not self.field:
            self.log("Field not initialized", "ERROR")
            return []
        
        # Auto-select starting particles if not provided
        if not start_particles:
            all_particles = self.field.get_all_particles()
            start_particles = [
                p for p in all_particles 
                if hasattr(p, 'activation') and p.activation > 0.6
            ][:10]  # Top 10 high-activation particles
        
        inference_chains = []
        
        for start_particle in start_particles:
            # Traverse in multiple directions
            chains = await self._traverse_inference_paths(
                start_particle, 
                max_depth=max_depth,
                visited=set()
            )
            
            for chain in chains:
                inference = InferenceChain(chain, reasoning_type="deductive")
                confidence = inference.calculate_confidence()
                
                if confidence >= min_confidence:
                    inference_chains.append(inference)
                    self.log(
                        f"Generated inference chain: {len(chain)} hops, "
                        f"confidence={confidence:.3f}",
                        "DEBUG"
                    )
        
        # Store in history
        self.inference_history.extend(inference_chains)
        
        # Keep history manageable
        if len(self.inference_history) > 100:
            self.inference_history = self.inference_history[-50:]
        
        return inference_chains
    
    async def _traverse_inference_paths(
        self, 
        particle, 
        max_depth: int,
        visited: Set,
        current_path: List = None
    ) -> List[List]:
        """Recursively traverse particle linkages to build inference paths"""
        if current_path is None:
            current_path = []
        
        if max_depth <= 0 or particle.id in visited:
            return [current_path] if len(current_path) >= 2 else []
        
        visited.add(particle.id)
        current_path = current_path + [particle]
        
        paths = []
        
        # Follow children linkages
        if hasattr(particle, 'linked_particles') and 'children' in particle.linked_particles:
            children_ids = particle.linked_particles['children']
            for child_id in children_ids[:3]:  # Limit branching factor
                child = self.field.get_particle_by_id(child_id)
                if child and child.alive:
                    sub_paths = await self._traverse_inference_paths(
                        child,
                        max_depth - 1,
                        visited.copy(),
                        current_path.copy()
                    )
                    paths.extend(sub_paths)
        
        # If no children, return current path if valid
        if not paths and len(current_path) >= 2:
            paths.append(current_path)
        
        return paths
    
    # ==================== CONTRADICTION DETECTION ====================
    
    async def detect_contradictions(self) -> List[Dict]:
        """
        Find contradicting particles in the field based on:
        - Opposing valence (position[8]) with similar content
        - Similar positions but conflicting metadata
        - Semantic contradictions in linked particles
        """
        contradictions = []
        
        if not self.field:
            return contradictions
        
        particles = self.field.get_alive_particles()
        
        # Compare particles pairwise for contradictions
        for i, p1 in enumerate(particles[:50]):  # Limit for performance
            for p2 in particles[i+1:min(i+20, len(particles))]:
                if self._are_contradictory(p1, p2):
                    contradiction = {
                        "particle_1": p1.id,
                        "particle_2": p2.id,
                        "type": self._classify_contradiction(p1, p2),
                        "severity": self._calculate_contradiction_severity(p1, p2),
                        "timestamp": datetime.now().isoformat()
                    }
                    contradictions.append(contradiction)
                    self.contradiction_log.append(contradiction)
        
        self.log(f"Detected {len(contradictions)} contradictions", "DEBUG")
        return contradictions
    
    def _are_contradictory(self, p1, p2) -> bool:
        """Check if two particles contradict each other"""
        # Valence opposition check (opposite emotional valence)
        valence_diff = abs(p1.position[8] - p2.position[8])
        if valence_diff > 1.5:  # Strong opposition
            # Check if they're semantically similar (close in x,y,z)
            spatial_dist = np.linalg.norm(p1.position[:3] - p2.position[:3])
            if spatial_dist < 0.3:  # Close in space but opposite valence
                return True
        
        # TODO: Add more sophisticated contradiction detection
        # - Semantic contradiction via content analysis
        # - Temporal contradiction (same event, different outcomes)
        # - Logical contradiction (A and not-A)
        
        return False
    
    def _classify_contradiction(self, p1, p2) -> str:
        """Classify type of contradiction"""
        valence_diff = abs(p1.position[8] - p2.position[8])
        if valence_diff > 1.5:
            return "valence_opposition"
        return "unknown"
    
    def _calculate_contradiction_severity(self, p1, p2) -> float:
        """Calculate how severe the contradiction is (0.0 to 1.0)"""
        # Based on particle energy and activation
        avg_importance = (p1.energy + p2.energy + p1.activation + p2.activation) / 4
        return min(avg_importance, 1.0)
    
    async def resolve_contradiction(self, contradiction: Dict) -> Optional[str]:
        """
        Resolve contradiction via quantum collapse
        
        Returns resolution strategy used
        """
        p1 = self.field.get_particle_by_id(contradiction["particle_1"])
        p2 = self.field.get_particle_by_id(contradiction["particle_2"])
        
        if not (p1 and p2):
            return None
        
        # Strategy 1: Collapse lower-energy particle
        if abs(p1.energy - p2.energy) > 0.2:
            weaker = p1 if p1.energy < p2.energy else p2
            if hasattr(weaker, 'observe'):
                weaker.observe(context="contradiction_resolution")
            return "collapse_weaker"
        
        # Strategy 2: Create superposition particle that holds both states
        else:
            superposition_particle = await self.field.spawn_particle(
                type="lingual",
                metadata={
                    "content": "Ambiguous state: multiple perspectives exist",
                    "source": "contradiction_resolution",
                    "linked_contradictions": [str(p1.id), str(p2.id)]
                },
                energy=0.6,
                activation=0.5,
                emit_event=False
            )
            return "superposition_created"
    
    # ==================== CONCEPT ABSTRACTION ====================
    
    async def build_concept_hierarchy(self) -> Dict:
        """
        Build hierarchical concept map by clustering similar particles
        
        Returns hierarchy: {abstract_concept: [concrete_particle_ids]}
        """
        if not self.field:
            return {}
        
        particles = self.field.get_alive_particles()
        
        # Cluster particles by position similarity (simple k-means-like approach)
        clusters = await self._cluster_particles(particles, n_clusters=10)
        
        # Create abstract concept particle for each cluster
        hierarchy = {}
        
        for i, cluster in enumerate(clusters):
            if len(cluster) < 2:
                continue  # Skip singleton clusters
            
            # Create abstract particle at cluster centroid
            centroid = self._calculate_centroid([p.position for p in cluster])
            
            # Generate abstract concept name from cluster content
            concept_name = await self._generate_concept_name(cluster)
            
            # Spawn abstract particle
            abstract_particle = await self.field.spawn_particle(
                type="lingual",
                metadata={
                    "content": concept_name,
                    "source": "concept_abstraction",
                    "abstract_level": 1,
                    "cluster_size": len(cluster),
                    "concrete_particles": [str(p.id) for p in cluster]
                },
                energy=0.7,
                activation=0.6,
                emit_event=False
            )
            
            if abstract_particle:
                # Override position to centroid
                abstract_particle.position[:3] = centroid
                hierarchy[concept_name] = [p.id for p in cluster]
        
        self.concept_hierarchy = hierarchy
        self.log(f"Built concept hierarchy with {len(hierarchy)} abstract concepts", "INFO")
        
        return hierarchy
    
    async def _cluster_particles(self, particles: List, n_clusters: int = 10) -> List[List]:
        """Simple clustering by position similarity"""
        if len(particles) < n_clusters:
            return [[p] for p in particles]
        
        # Initialize centroids randomly
        import random
        centroids = [p.position[:3] for p in random.sample(particles, n_clusters)]
        
        # Simple k-means (1 iteration for efficiency)
        clusters = [[] for _ in range(n_clusters)]
        
        for particle in particles:
            # Find nearest centroid
            distances = [
                np.linalg.norm(particle.position[:3] - centroid)
                for centroid in centroids
            ]
            nearest_idx = distances.index(min(distances))
            clusters[nearest_idx].append(particle)
        
        return [c for c in clusters if c]  # Filter empty clusters
    
    def _calculate_centroid(self, positions: List) -> np.ndarray:
        """Calculate centroid of positions"""
        positions = [np.array(pos[:3]) for pos in positions]
        return np.mean(positions, axis=0)
    
    async def _generate_concept_name(self, cluster: List) -> str:
        """Generate abstract concept name from cluster content"""
        # Collect content from cluster particles
        contents = []
        for p in cluster[:5]:  # Sample first 5
            content = p.metadata.get("content", "")
            if isinstance(content, str) and content:
                contents.append(content)
        
        if not contents:
            return f"Abstract Concept {len(cluster)} particles"
        
        # Simple approach: use most common words
        words = " ".join(contents).split()[:10]
        return f"Concept: {' '.join(words[:3])}..." if words else "Unknown Concept"
    
    # ==================== CAUSAL REASONING ====================
    
    async def infer_causal_relationships(self) -> List[Dict]:
        """
        Infer causal relationships using temporal ordering (position[3] - creation time)
        
        Returns list of causal relationships: {cause_id, effect_id, strength}
        """
        causal_links = []
        
        if not self.field:
            return causal_links
        
        particles = self.field.get_alive_particles()
        
        # Sort by creation time
        sorted_particles = sorted(particles, key=lambda p: p.position[3])
        
        # Look for patterns: earlier particles linked to later ones
        for i, earlier in enumerate(sorted_particles[:-1]):
            for later in sorted_particles[i+1:i+10]:  # Check next 10 particles
                # Check if they're linked
                if (hasattr(earlier, 'linked_particles') and 
                    'children' in earlier.linked_particles and
                    later.id in earlier.linked_particles['children']):
                    
                    # Calculate causal strength based on temporal proximity and energy
                    time_diff = later.position[3] - earlier.position[3]
                    temporal_strength = 1.0 / (1.0 + time_diff * 0.001)
                    
                    energy_transfer = min(earlier.energy, later.energy) / max(earlier.energy, later.energy, 0.1)
                    
                    causal_strength = (temporal_strength + energy_transfer) / 2
                    
                    causal_links.append({
                        "cause_id": str(earlier.id),
                        "effect_id": str(later.id),
                        "strength": causal_strength,
                        "time_diff": time_diff,
                        "timestamp": datetime.now().isoformat()
                    })
        
        self.log(f"Inferred {len(causal_links)} causal relationships", "DEBUG")
        return causal_links
    
    # ==================== UTILITY METHODS ====================
    
    def get_inference_statistics(self) -> Dict:
        """Get statistics about inference engine performance"""
        return {
            "total_inferences": len(self.inference_history),
            "avg_confidence": (
                sum(chain.confidence for chain in self.inference_history) / 
                len(self.inference_history)
                if self.inference_history else 0.0
            ),
            "contradictions_detected": len(self.contradiction_log),
            "concept_hierarchy_size": len(self.concept_hierarchy),
            "timestamp": datetime.now().isoformat()
        }
