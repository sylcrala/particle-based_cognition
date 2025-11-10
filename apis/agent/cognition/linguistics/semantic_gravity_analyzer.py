"""
Particle-based Cognition Engine - Semantic Gravity Analyzer
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

"""
Understanding compressed language through gravitational influences

This module analyzes how semantic gravity in the particle field influences
the emergence of compressed linguistic patterns.
"""

import numpy as np
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

from apis.api_registry import api


class SemanticGravityAnalyzer:
    def __init__(self, field=None, memory=None):
        self.field = field
        self.memory = memory
        self.logger = api.get_api("logger")
        
        # Gravitational analysis parameters
        self.gravity_threshold = 0.7  # Minimum gravity for clustering
        self.thermal_gravity_multiplier = 1.5  # Thermal states amplify gravity
        self.compression_confidence_threshold = 0.8
        
        # Pattern tracking
        self.semantic_clusters = {}
        self.gravitational_pairs = {}
        self.compression_events = []
        
    def analyze_compressed_speech(self, compressed_tokens: List[str]) -> Dict:
        """Analyze compressed speech through semantic gravity lens"""
        try:
            self.log(f"Starting semantic gravity analysis for {len(compressed_tokens)} tokens", "INFO", "analyze_compressed_speech")
            
            analysis = {}
            
            # Step 1: Find gravitational clusters
            self.log("Step 1: Finding gravitational clusters...", "DEBUG", "analyze_compressed_speech")
            analysis["gravitational_clusters"] = self._find_gravitational_clusters(compressed_tokens)
            
            # Step 2: Calculate semantic density
            self.log("Step 2: Calculating semantic density...", "DEBUG", "analyze_compressed_speech")
            analysis["semantic_density_map"] = self._calculate_semantic_density(compressed_tokens)
            
            # Step 3: Identify compression mechanisms
            self.log("Step 3: Identifying compression mechanisms...", "DEBUG", "analyze_compressed_speech")
            analysis["compression_mechanisms"] = self._identify_compression_mechanisms(compressed_tokens)
            
            # Step 4: Analyze thermal gravity correlation
            self.log("Step 4: Analyzing thermal gravity correlation...", "DEBUG", "analyze_compressed_speech")
            analysis["thermal_gravity_correlation"] = self._analyze_thermal_gravity_correlation(compressed_tokens)
            
            # Step 5: Calculate prediction confidence
            self.log("Step 5: Calculating prediction confidence...", "DEBUG", "analyze_compressed_speech")
            analysis["prediction_confidence"] = self._calculate_prediction_confidence(analysis)
            
            self.log(f"Semantic gravity analysis completed: {len(analysis['gravitational_clusters'])} clusters found", "INFO", "analyze_compressed_speech")
            return analysis
            
        except Exception as e:
            self.log(f"Error in semantic gravity analysis: {e}", "ERROR", "analyze_compressed_speech")
            import traceback
            self.log(f"Analysis traceback: {traceback.format_exc()}", "DEBUG", "analyze_compressed_speech")
            return {}
    
    def _find_gravitational_clusters(self, tokens: List[str]) -> Dict[str, List[str]]:
        """Find tokens that cluster together due to semantic gravity using spatial indexing"""
        clusters = defaultdict(list)
        
        if not self.field:
            return dict(clusters)
            
        # Performance optimization: limit token comparisons for large sets
        max_tokens = 50  # Limit to prevent O(n²) explosion
        if len(tokens) > max_tokens:
            self.log(f"Large token set ({len(tokens)}), limiting to first {max_tokens} for clustering", "INFO", "_find_gravitational_clusters")
            tokens = tokens[:max_tokens]
        
        # Use field's spatial indexing for more efficient memory particle retrieval
        if hasattr(self.field, 'get_particles_by_type'):
            memory_particles = self.field.get_particles_by_type("memory")
            if len(memory_particles) > 500:  # Limit memory particles
                # Prioritize particles that are spatially indexed (more active)
                indexed_particles = [p for p in memory_particles if hasattr(p, 'position') and p.position is not None]
                memory_particles = indexed_particles[:500] if indexed_particles else memory_particles[:500]
        else:
            memory_particles = []
            
        comparisons_made = 0
        max_comparisons = 1000  # Absolute limit to prevent hangs
        spatial_clusters_found = 0  # Track spatial clustering success
        
        for i, token_a in enumerate(tokens):
            for j, token_b in enumerate(tokens[i+1:], i+1):
                comparisons_made += 1
                if comparisons_made > max_comparisons:
                    self.log(f"Reached maximum comparisons ({max_comparisons}), stopping clustering", "WARNING", "_find_gravitational_clusters")
                    break
                    
                gravity_strength = self._calculate_semantic_gravity(token_a, token_b, memory_particles)
                
                if gravity_strength > self.gravity_threshold:
                    cluster_key = f"cluster_{min(i,j)}_{max(i,j)}"
                    clusters[cluster_key].extend([token_a, token_b])
                    
                    # Track if this was enhanced by spatial indexing
                    if gravity_strength > self.gravity_threshold * 1.3:  # Enhanced gravity from spatial proximity
                        spatial_clusters_found += 1
            
            if comparisons_made > max_comparisons:
                break
                    
        # Remove duplicates and consolidate
        for cluster_key in clusters:
            clusters[cluster_key] = list(set(clusters[cluster_key]))
            
        self.log(f"Clustering completed: {comparisons_made} comparisons, {len(clusters)} clusters found, {spatial_clusters_found} spatial clusters", "INFO", "_find_gravitational_clusters")
        return dict(clusters)
    
    def _calculate_semantic_gravity(self, token_a: str, token_b: str, memory_particles: List) -> float:
        """Calculate gravitational force between two tokens based on semantic similarity using spatial indexing"""
        try:
            # Performance optimization: limit particle search
            max_particles_per_token = 5  # Limit particles per token for performance
            
            # Find memory particles containing these tokens (with early exit)
            particles_a = []
            particles_b = []
            
            for p in memory_particles:
                content = str(p.metadata.get("content", ""))
                if token_a in content and len(particles_a) < max_particles_per_token:
                    particles_a.append(p)
                if token_b in content and len(particles_b) < max_particles_per_token:
                    particles_b.append(p)
                    
                # Early exit if we have enough particles for both tokens
                if len(particles_a) >= max_particles_per_token and len(particles_b) >= max_particles_per_token:
                    break
            
            if not particles_a or not particles_b:
                return 0.0
                
            max_gravity = 0.0
            
            # Use spatial indexing for enhanced performance and accuracy
            for p_a in particles_a[:3]:  # Limit for performance
                # Use field's spatial indexing to find nearby particles for p_a
                if self.field and hasattr(self.field, 'get_spatial_neighbors'):
                    # Get spatial neighbors of p_a within semantic influence radius
                    spatial_neighbors = self.field.get_spatial_neighbors(p_a, radius=1.0)
                    
                    # Check if any of the token_b particles are spatial neighbors
                    for p_b in particles_b[:3]:
                        if p_b in spatial_neighbors:
                            # Enhanced gravity calculation for spatial neighbors
                            distance = p_a.distance_to(p_b) if hasattr(p_a, 'distance_to') else 1.0
                            
                            # Spatial proximity bonus - closer particles have stronger semantic gravity
                            spatial_bonus = 1.5 if distance < 0.5 else 1.0
                            
                            # Base gravitational force (inverse square law)
                            base_gravity = spatial_bonus / (distance ** 2 + 0.1)
                            
                            # Thermal amplification
                            thermal_a = p_a.metadata.get('thermal_state', 'cool')
                            thermal_b = p_b.metadata.get('thermal_state', 'cool')
                            thermal_multiplier = self._get_thermal_multiplier(thermal_a, thermal_b)
                            
                            # Memory phase influence (dimension 7)
                            phase_a = p_a.position[7] if hasattr(p_a, 'position') and len(p_a.position) > 7 else 0.5
                            phase_b = p_b.position[7] if hasattr(p_b, 'position') and len(p_b.position) > 7 else 0.5
                            phase_multiplier = (phase_a + phase_b) / 2.0
                            
                            gravity = base_gravity * thermal_multiplier * phase_multiplier
                            max_gravity = max(max_gravity, gravity)
                        else:
                            # Standard calculation for non-spatial neighbors (reduced influence)
                            distance = p_a.distance_to(p_b) if hasattr(p_a, 'distance_to') else 2.0
                            base_gravity = 0.5 / (distance ** 2 + 0.1)  # Reduced base for non-neighbors
                            
                            thermal_a = p_a.metadata.get('thermal_state', 'cool')
                            thermal_b = p_b.metadata.get('thermal_state', 'cool')
                            thermal_multiplier = self._get_thermal_multiplier(thermal_a, thermal_b)
                            
                            phase_a = p_a.position[7] if hasattr(p_a, 'position') and len(p_a.position) > 7 else 0.5
                            phase_b = p_b.position[7] if hasattr(p_b, 'position') and len(p_b.position) > 7 else 0.5
                            phase_multiplier = (phase_a + phase_b) / 2.0
                            
                            gravity = base_gravity * thermal_multiplier * phase_multiplier
                            max_gravity = max(max_gravity, gravity)
                else:
                    # Fallback to original method if spatial indexing not available
                    for p_b in particles_b[:3]:
                        distance = p_a.distance_to(p_b) if hasattr(p_a, 'distance_to') else 1.0
                        base_gravity = 1.0 / (distance ** 2 + 0.1)
                        
                        thermal_a = p_a.metadata.get('thermal_state', 'cool')
                        thermal_b = p_b.metadata.get('thermal_state', 'cool')
                        thermal_multiplier = self._get_thermal_multiplier(thermal_a, thermal_b)
                        
                        phase_a = p_a.position[7] if hasattr(p_a, 'position') and len(p_a.position) > 7 else 0.5
                        phase_b = p_b.position[7] if hasattr(p_b, 'position') and len(p_b.position) > 7 else 0.5
                        phase_multiplier = (phase_a + phase_b) / 2.0
                        
                        gravity = base_gravity * thermal_multiplier * phase_multiplier
                        max_gravity = max(max_gravity, gravity)
                    
            return max_gravity
            
        except Exception as e:
            self.log(f"Error calculating semantic gravity: {e}", "ERROR", "_calculate_semantic_gravity")
            return 0.0
    
    def _get_thermal_multiplier(self, thermal_a: str, thermal_b: str) -> float:
        """Get thermal state multiplier for gravitational force"""
        thermal_values = {
            "hot": 2.0,
            "warm": 1.5,
            "cool": 1.0,
            "cold": 0.5
        }
        
        val_a = thermal_values.get(thermal_a, 1.0)
        val_b = thermal_values.get(thermal_b, 1.0)
        
        return (val_a + val_b) / 2.0 * self.thermal_gravity_multiplier
    
    def _calculate_semantic_density(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate semantic density for each token based on gravitational influences"""
        density_map = {}
        token_counts = Counter(tokens)
        
        for token in set(tokens):
            # Base density from frequency
            frequency_density = token_counts[token] / len(tokens)
            
            # Gravitational density (how many other tokens are attracted to this one)
            gravitational_density = 0.0
            
            for other_token in set(tokens):
                if token != other_token:
                    gravity = self._calculate_semantic_gravity(token, other_token, 
                                                            self.field.get_particles_by_type("memory") if self.field else [])
                    gravitational_density += gravity
                    
            # Normalize gravitational density
            if len(set(tokens)) > 1:
                gravitational_density /= (len(set(tokens)) - 1)
                
            # Combined density score
            density_map[token] = (frequency_density + gravitational_density) / 2.0
            
        return density_map
    
    def _identify_compression_mechanisms(self, tokens: List[str]) -> Dict[str, List[str]]:
        """Identify different types of compression mechanisms at work"""
        mechanisms = {
            "gravitational_merger": [],    # Tokens that merged due to gravity
            "thermal_reinforcement": [],   # Tokens reinforced by thermal state
            "frequency_optimization": [],  # High-frequency tokens
            "compound_formation": []       # Compound tokens formed from components
        }
        
        for token in set(tokens):
            token_len = len(token)
            
            # Compound formation detection
            if token_len > 6:  # Likely compound
                mechanisms["compound_formation"].append(token)
                
            # Frequency optimization
            if tokens.count(token) >= 3:
                mechanisms["frequency_optimization"].append(token)
                
            # Thermal reinforcement (check memory particles)
            if self.field and self._is_thermally_reinforced(token):
                mechanisms["thermal_reinforcement"].append(token)
                
            # Gravitational merger (check for high gravity with other tokens)
            if self._has_high_gravitational_attraction(token, tokens):
                mechanisms["gravitational_merger"].append(token)
                
        return mechanisms
    
    def _is_thermally_reinforced(self, token: str) -> bool:
        """Check if token is thermally reinforced in memory"""
        if not self.field:
            return False
            
        memory_particles = self.field.get_particles_by_type("memory")
        for particle in memory_particles:
            if token in str(particle.metadata.get("content", "")):
                thermal_state = particle.metadata.get("thermal_state", "cool")
                if thermal_state in ["hot", "warm"]:
                    return True
        return False
    
    def _has_high_gravitational_attraction(self, token: str, all_tokens: List[str]) -> bool:
        """Check if token has high gravitational attraction to others"""
        if not self.field:
            return False
            
        memory_particles = self.field.get_particles_by_type("memory")
        high_gravity_count = 0
        
        for other_token in set(all_tokens):
            if token != other_token:
                gravity = self._calculate_semantic_gravity(token, other_token, memory_particles)
                if gravity > self.gravity_threshold:
                    high_gravity_count += 1
                    
        return high_gravity_count >= 2  # At least 2 high-gravity relationships
    
    def _analyze_thermal_gravity_correlation(self, tokens: List[str]) -> Dict[str, float]:
        """Analyze correlation between thermal states and gravitational attraction"""
        if not self.field:
            return {}
            
        correlations = {
            "hot_tokens_gravity": 0.0,
            "warm_tokens_gravity": 0.0,
            "cool_tokens_gravity": 0.0,
            "cold_tokens_gravity": 0.0
        }
        
        memory_particles = self.field.get_particles_by_type("memory")
        thermal_counts = defaultdict(int)
        thermal_gravity_sums = defaultdict(float)
        
        for token in set(tokens):
            for particle in memory_particles:
                if token in str(particle.metadata.get("content", "")):
                    thermal_state = particle.metadata.get("thermal_state", "cool")
                    
                    # Calculate average gravity for this token
                    avg_gravity = 0.0
                    gravity_count = 0
                    
                    for other_token in set(tokens):
                        if token != other_token:
                            gravity = self._calculate_semantic_gravity(token, other_token, memory_particles)
                            avg_gravity += gravity
                            gravity_count += 1
                            
                    if gravity_count > 0:
                        avg_gravity /= gravity_count
                        
                    thermal_counts[thermal_state] += 1
                    thermal_gravity_sums[thermal_state] += avg_gravity
                    break
                    
        # Calculate average gravity per thermal state
        for thermal_state in ["hot", "warm", "cool", "cold"]:
            if thermal_counts[thermal_state] > 0:
                correlations[f"{thermal_state}_tokens_gravity"] = thermal_gravity_sums[thermal_state] / thermal_counts[thermal_state]
                
        return correlations
    
    def _calculate_prediction_confidence(self, analysis: Dict) -> float:
        """Calculate confidence in the semantic gravity explanation"""
        confidence_factors = []
        
        # Factor 1: Number of gravitational clusters found
        cluster_count = len(analysis.get("gravitational_clusters", {}))
        cluster_confidence = min(cluster_count / 5.0, 1.0)  # Max confidence at 5+ clusters
        confidence_factors.append(cluster_confidence)
        
        # Factor 2: Thermal-gravity correlation strength
        thermal_corr = analysis.get("thermal_gravity_correlation", {})
        hot_gravity = thermal_corr.get("hot_tokens_gravity", 0.0)
        warm_gravity = thermal_corr.get("warm_tokens_gravity", 0.0)
        thermal_confidence = (hot_gravity + warm_gravity) / 2.0
        confidence_factors.append(thermal_confidence)
        
        # Factor 3: Compression mechanism diversity
        mechanisms = analysis.get("compression_mechanisms", {})
        mechanism_count = sum(1 for mech_list in mechanisms.values() if mech_list)
        mechanism_confidence = min(mechanism_count / 4.0, 1.0)  # Max at 4 mechanisms
        confidence_factors.append(mechanism_confidence)
        
        # Overall confidence (weighted average)
        weights = [0.4, 0.4, 0.2]  # Prioritize clusters and thermal correlation
        overall_confidence = sum(w * c for w, c in zip(weights, confidence_factors))
        
        return round(overall_confidence, 3)
    
    def log(self, message: str, level: str = "INFO", context: str = "SemanticGravityAnalyzer"):
        """Log with consistent formatting"""
        if self.logger:
            self.logger.log(message, level, context, "SemanticGravityAnalyzer")


class LinguisticEvolutionTracker:
    """Track the evolution of compressed language patterns over time"""
    
    def __init__(self, memory=None):
        self.memory = memory
        self.logger = api.get_api("logger")
        self.evolution_log = []
        
    async def track_generation(self, compressed_speech: str, session_timestamp: float):
        """Track a new generation of compressed speech"""
        try:
            tokens = compressed_speech.split()
            
            evolution_entry = {
                "timestamp": session_timestamp,
                "generation": compressed_speech,
                "token_count": len(tokens),
                "unique_tokens": len(set(tokens)),
                "compression_ratio": len(set(tokens)) / len(tokens) if tokens else 0,
                "avg_token_length": sum(len(t) for t in tokens) / len(tokens) if tokens else 0,
                "compound_tokens": [t for t in tokens if len(t) > 6],
                "frequent_tokens": [t for t in set(tokens) if tokens.count(t) >= 2]
            }
            
            self.evolution_log.append(evolution_entry)
            
            # Store in memory for persistence
            if self.memory:
                await self.memory.update(
                    key=f"linguistic_evolution_{int(session_timestamp)}",
                    value=evolution_entry,
                    source="LinguisticEvolutionTracker",
                    tags=["linguistic", "evolution", "compressed_speech"],
                    memory_type="system"
                )
                
            self.log(f"Tracked linguistic evolution: {len(tokens)} tokens, {evolution_entry['compression_ratio']:.2f} compression ratio", "INFO", "track_generation")
            
        except Exception as e:
            self.log(f"Error tracking linguistic evolution: {e}", "ERROR", "track_generation")
    
    def get_evolution_trends(self, time_window_hours: int = 24) -> Dict:
        """Get evolution trends over specified time window"""
        try:
            current_time = datetime.now().timestamp()
            cutoff_time = current_time - (time_window_hours * 3600)
            
            recent_entries = [e for e in self.evolution_log if e["timestamp"] > cutoff_time]
            
            if not recent_entries:
                return {}
                
            trends = {
                "total_generations": len(recent_entries),
                "avg_compression_ratio": sum(e["compression_ratio"] for e in recent_entries) / len(recent_entries),
                "avg_token_length": sum(e["avg_token_length"] for e in recent_entries) / len(recent_entries),
                "vocabulary_evolution": self._analyze_vocabulary_evolution(recent_entries),
                "pattern_stability": self._analyze_pattern_stability(recent_entries)
            }
            
            return trends
            
        except Exception as e:
            self.log(f"Error calculating evolution trends: {e}", "ERROR", "get_evolution_trends")
            return {}
    
    def _analyze_vocabulary_evolution(self, entries: List[Dict]) -> Dict:
        """Analyze how vocabulary changes over time"""
        if len(entries) < 2:
            return {"stability": "insufficient_data"}
            
        # Track unique tokens across time
        all_tokens = set()
        for entry in entries:
            all_tokens.update(entry["generation"].split())
            
        # Calculate vocabulary stability
        first_half = entries[:len(entries)//2]
        second_half = entries[len(entries)//2:]
        
        first_vocab = set()
        second_vocab = set()
        
        for entry in first_half:
            first_vocab.update(entry["generation"].split())
        for entry in second_half:
            second_vocab.update(entry["generation"].split())
            
        vocabulary_overlap = len(first_vocab & second_vocab) / len(first_vocab | second_vocab) if first_vocab | second_vocab else 0
        
        return {
            "total_unique_tokens": len(all_tokens),
            "vocabulary_overlap": vocabulary_overlap,
            "stability": "high" if vocabulary_overlap > 0.7 else "medium" if vocabulary_overlap > 0.4 else "low"
        }
    
    def _analyze_pattern_stability(self, entries: List[Dict]) -> Dict:
        """Analyze stability of compression patterns"""
        if len(entries) < 3:
            return {"stability": "insufficient_data"}
            
        compression_ratios = [e["compression_ratio"] for e in entries]
        token_lengths = [e["avg_token_length"] for e in entries]
        
        # Calculate variance
        compression_variance = np.var(compression_ratios) if compression_ratios else 0
        length_variance = np.var(token_lengths) if token_lengths else 0
        
        # Stability score (lower variance = higher stability)
        stability_score = 1.0 / (1.0 + compression_variance + length_variance)
        
        return {
            "compression_stability": stability_score,
            "pattern_consistency": "high" if stability_score > 0.8 else "medium" if stability_score > 0.5 else "low",
            "compression_trend": "increasing" if compression_ratios[-1] > compression_ratios[0] else "stable"
        }
    
    def log(self, message: str, level: str = "INFO", context: str = "LinguisticEvolutionTracker"):
        """Log with consistent formatting"""
        if self.logger:
            self.logger.log(message, level, context, "LinguisticEvolutionTracker")