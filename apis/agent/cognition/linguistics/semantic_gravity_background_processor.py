"""
Particle-based Cognition Engine - Autonomous Semantic Gravity Background Processor
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
Continuous analysis of compressed language evolution

This module runs continuous background analysis of compressed tokens, automatically generating
categories and translation mappings through semantic gravity clustering.
"""

import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np

from apis.api_registry import api


class SemanticGravityBackgroundProcessor:
    """Autonomous background processor for continuous semantic gravity analysis"""
    
    def __init__(self, gravity_analyzer=None, agent_categorizer=None, memory=None, field=None, auto_start=True):
        self.gravity_analyzer = gravity_analyzer
        self.agent_categorizer = agent_categorizer
        self.memory = memory
        self.field = field
        self.auto_start = auto_start
        self.logger = api.get_api("logger")
        
        # Background processing parameters
        self.processing_interval = 120  # Start with 2 minutes, will adapt dynamically
        self.min_processing_interval = 60  # Minimum 1 minute
        self.max_processing_interval = 600  # Maximum 10 minutes
        self.min_token_frequency = 2  # Minimum frequency to analyze
        self.analysis_window_minutes = 10  # Look at last 10 minutes of activity
        self.chance_threshold = 0.3  # 30% chance per cycle when prerequisites met
        
        # Token tracking
        self.compressed_token_cache = defaultdict(list)  # token -> [timestamps]
        self.last_analysis_time = 0
        self.analysis_queue = set()
        self.background_task = None
        self.is_processing = False
        
        # Translation mapping system
        self.translation_candidates = {}  # compressed -> [human_candidates]
        self.confidence_scores = {}  # (compressed, human) -> confidence
        self.context_correlations = {}  # compressed -> context_patterns
        
        # Hybrid processing system
        self.subconscious_processing_enabled = True
        self.maintenance_batch_size = 5  # Process 5 tokens per maintenance cycle
        self.last_maintenance_time = 0
        
    async def start_background_processing(self):
        """Start the background semantic gravity processing loop"""
        try:
            if not self.auto_start:
                self.log("Auto-start disabled, background processing will not start", "INFO", "start_background_processing")
                return
            
            if self.background_task:
                return  # Already running
                
            # Initialize from existing lexicon before starting
            await self.initialize_from_lexicon()
            
            # Load existing translation mappings from memory - TEMPORARILY DISABLED to prevent recursion
            # await self._load_translation_mappings_from_memory()
            #self.log("Translation mapping loading temporarily disabled to prevent recursion loops", "INFO", "start_background_processing")
                
            #self.background_task = asyncio.create_task(self._background_loop())
            #self.log("Background semantic gravity processor started", "INFO", "start_background_processing")
        except Exception as e:
            self.log(f"Error starting background processing: {e}", "ERROR", "start_background_processing")

    async def stop_background_processing(self):
        """Stop the background processing loop"""
        if self.background_task:
            self.background_task.cancel()
            self.background_task = None
            self.log("Background semantic gravity processor stopped", "INFO", "stop_background_processing")
    

    async def initialize_from_lexicon(self):
        """Initialize processor with existing lexicon tokens for analysis"""
        try:
            self.log("Starting lexicon initialization process", "INFO", "initialize_from_lexicon")
            
            # Get lexicon store from agent categorizer memory system
            if not self.agent_categorizer or not hasattr(self.agent_categorizer, 'memory'):
                self.log("No memory system available for lexicon initialization", "WARNING", "initialize_from_lexicon")
                return
            
            memory = self.memory
            self.log(f"Memory system found: {type(memory).__name__}", "DEBUG", "initialize_from_lexicon")
            
            # Test basic memory connectivity
            try:
                # First check if we can access the direct method 
                self.log("Testing direct memory access...", "DEBUG", "initialize_from_lexicon")
                direct_result = await memory._direct_get_memories_by_type("lexicon", 10)
                self.log(f"Direct memory access succeeded: {len(direct_result)} entries", "DEBUG", "initialize_from_lexicon")
            except Exception as direct_error:
                self.log(f"Direct memory access failed: {direct_error}", "ERROR", "initialize_from_lexicon")
            
            # Load existing lexicon entries
            self.log("Attempting to load lexicon entries from memory", "DEBUG", "initialize_from_lexicon")
            try:
                lexicon_memories = await memory.get_memories_by_type("lexicon")  # Load all entries for proper classification
                self.log(f"get_memories_by_type succeeded, returned type: {type(lexicon_memories)}", "DEBUG", "initialize_from_lexicon")
            except Exception as memory_error:
                self.log(f"get_memories_by_type failed with error: {memory_error}", "ERROR", "initialize_from_lexicon")
                return
            
            self.log(f"Retrieved {len(lexicon_memories) if lexicon_memories else 0} lexicon entries from memory", "INFO", "initialize_from_lexicon")
            
            if not lexicon_memories:
                self.log("No lexicon entries found for initialization", "INFO", "initialize_from_lexicon")
                return
            
            compressed_tokens_found = 0
            tokens_with_definitions = 0
            tokens_with_positions = 0
            total_tokens_processed = 0
            
            for memory_entry in lexicon_memories:
                total_tokens_processed += 1
                if not isinstance(memory_entry, dict):
                    self.log(f"Skipping non-dict entry: {type(memory_entry)}", "DEBUG", "initialize_from_lexicon")
                    continue
                
                # Parse lexicon entry based on known storage formats only
                token_data = None
                
                # Check if it's a direct token field (newer format)
                if "token" in memory_entry:
                    token_data = memory_entry
                    
                # Check if it's stored in 'value' field as parseable dict
                elif "value" in memory_entry:
                    try:
                        import json
                        value = memory_entry["value"]
                        
                        # Only try simple JSON parsing - no complex fallbacks
                        if isinstance(value, str) and value.startswith("{"):
                            try:
                                token_data = json.loads(value)
                            except json.JSONDecodeError:
                                # Skip unparseable entries - no complex extraction
                                continue
                        elif isinstance(value, dict):
                            # Value is already a dict
                            token_data = value
                    except Exception:
                        # Skip problematic entries
                        continue
                
                # Check for key-based format (lexicon_tokenname)
                elif "key" in memory_entry:
                    key = memory_entry["key"]
                    if key.startswith("lexicon_"):
                        token = key.replace("lexicon_", "")
                        if token and len(token) > 1 and len(token) < 20:  # Reasonable token length
                            token_data = {
                                "token": token, 
                                "type": "key_derived",
                                "content": memory_entry.get("value", "")
                            }
                
                # Skip if we couldn't extract token data
                if not token_data or "token" not in token_data:
                    # Only log occasionally to reduce noise
                    if total_tokens_processed % 100 == 0:
                        self.log(f"Skipping entry without parseable token data (entry {total_tokens_processed})", "DEBUG", "initialize_from_lexicon")
                    continue
                    
                token = token_data.get("token")
                if not token or len(token.strip()) == 0:
                    # Only log occasionally to reduce noise  
                    if total_tokens_processed % 100 == 0:
                        self.log(f"Skipping entry with empty token (entry {total_tokens_processed})", "DEBUG", "initialize_from_lexicon")
                    continue
                
                # Only log token processing for compressed tokens or occasionally for others
                is_compressed = self._is_compressed_token(token)
                if is_compressed or total_tokens_processed % 200 == 0:
                    self.log(f"Processing token: '{token}' (compressed: {is_compressed}, type: {token_data.get('type', 'unknown')})", "DEBUG", "initialize_from_lexicon")
                
                # Check if this is a compressed token
                if is_compressed:
                    compressed_tokens_found += 1
                    
                    # Simulate historical observations for this token
                    base_timestamp = datetime.now().timestamp() - 3600  # Start 1 hour ago
                    times_encountered = token_data.get("times_encountered", 1)
                    
                    # Create synthetic observation timestamps
                    synthetic_timestamps = []
                    for i in range(min(times_encountered, 10)):  # Limit to avoid overload
                        # Spread observations over the last hour
                        timestamp = base_timestamp + (i * 360)  # Every 6 minutes
                        synthetic_timestamps.append(timestamp)
                    
                    self.compressed_token_cache[token] = synthetic_timestamps
                    
                    # Extract context from lexicon entry - try both storage formats
                    context_data = {
                        "human_phrase": token_data.get("contexts", [""])[0] if token_data.get("contexts") else "",
                        "usage_context": f"lexicon_entry_{token_data.get('type', 'unknown')}",
                        "definitions": token_data.get("definitions", [])
                    }
                    
                    # Also check memory_entry for additional context (from outer storage)
                    if "contexts" in memory_entry:
                        contexts = memory_entry.get("contexts", [])
                        if contexts and not context_data["human_phrase"]:
                            context_data["human_phrase"] = contexts[0]
                    
                    # Check for field position data
                    field_position = token_data.get("field_position") or memory_entry.get("field_position")
                    if field_position and isinstance(field_position, dict):
                        tokens_with_positions += 1
                        context_data["field_position"] = field_position
                    
                    # Try to extract human language correlates from definitions
                    definitions = token_data.get("definitions", [])
                    if definitions:
                        tokens_with_definitions += 1
                        # Extract human language from definitions
                        for definition in definitions:
                            if isinstance(definition, dict):
                                def_text = definition.get("text", "") or definition.get("wikipedia", "")
                                if def_text:
                                    context_data["human_phrase"] = def_text[:200]  # Limit length
                                    break
                    
                    # Update context correlations
                    self._update_context_correlation(token, context_data)
                    
                    # Add to analysis queue if it has enough data
                    if times_encountered >= self.min_token_frequency:
                        self.analysis_queue.add(token)
            
            self.log(f"Initialized from lexicon: {compressed_tokens_found} compressed tokens found, {tokens_with_definitions} with definitions, {tokens_with_positions} with field positions, {len(self.analysis_queue)} queued for analysis", "INFO", "initialize_from_lexicon")
            
            # Perform spatial clustering analysis if we have positioned tokens
            if tokens_with_positions > 0:
                await self._perform_spatial_clustering_analysis()
            
            # Backwards compatibility: update existing lexicon entries without positions
            unpositioned_tokens = compressed_tokens_found - tokens_with_positions
            if unpositioned_tokens > 0:
                self.log(f"Found {unpositioned_tokens} compressed tokens without positions, starting backwards compatibility update", "INFO", "initialize_from_lexicon")
                await self._update_legacy_lexicon_positions()
            
            # If we have enough tokens, trigger an immediate analysis
            if len(self.analysis_queue) >= 5:
                self.log("Triggering immediate analysis of lexicon tokens", "INFO", "initialize_from_lexicon")
                asyncio.create_task(self._process_analysis_queue())
                
        except Exception as e:
            self.log(f"Error initializing from lexicon: {e}", "ERROR", "initialize_from_lexicon")
    
    async def _perform_spatial_clustering_analysis(self):
        """Perform spatial clustering using field's efficient region-based access"""
        try:
            # Use field's efficient region access instead of full-field scan
            if not self.field or not hasattr(self.field, 'get_particles_in_region'):
                self.log("Field region access not available, skipping spatial clustering", "WARNING", "_perform_spatial_clustering_analysis")
                return
                
            # Get active grid sectors from field's spatial index
            if hasattr(self.field, 'get_all_grid_sectors'):
                active_sectors = self.field.get_all_grid_sectors()
                self.log(f"Found {len(active_sectors)} active grid sectors for spatial analysis", "DEBUG", "_perform_spatial_clustering_analysis")
            else:
                self.log("Grid sectors not available, using alternative approach", "DEBUG", "_perform_spatial_clustering_analysis")
                return
            
            positioned_tokens = []
            
            # Process each active sector (respects field's energy constraints)
            for sector_key in active_sectors[:10]:  # Limit to 10 most active sectors
                try:
                    # Get particles in this region using field's efficient access
                    sector_particles = self.field.get_particles_in_region(primary_region=sector_key, max_count=50)
                    
                    for particle in sector_particles:
                        if not hasattr(particle, 'metadata') or not particle.metadata:
                            continue
                        
                        metadata = particle.metadata
                        token = metadata.get("token")
                        
                        if token and self._is_compressed_token(token) and hasattr(particle, 'position'):
                            positioned_tokens.append({
                                "token": token,
                                "position": {
                                    "x": particle.position[0],
                                    "y": particle.position[1],
                                    "z": particle.position[2]
                                },
                                "particle": particle,
                                "metadata": metadata,
                                "sector": sector_key
                            })
                    
                except Exception as e:
                    self.log(f"Error processing sector {sector_key}: {e}", "DEBUG", "_perform_spatial_clustering_analysis")
                    continue
            
            if len(positioned_tokens) < 2:
                self.log(f"Insufficient positioned tokens for spatial clustering ({len(positioned_tokens)})", "INFO", "_perform_spatial_clustering_analysis") 
                return
            
            # Perform efficient clustering within sectors (avoids cross-field calculations)
            sector_clusters = self._cluster_by_sectors(positioned_tokens)
            
            # Generate translation mappings from sector-based clusters
            for sector_key, cluster_tokens in sector_clusters.items():
                if len(cluster_tokens) >= 2:
                    await self._analyze_spatial_cluster(cluster_tokens)
                    
            self.log(f"Spatial clustering completed: {len(sector_clusters)} sector clusters, {len(positioned_tokens)} total positioned tokens", "INFO", "_perform_spatial_clustering_analysis")
            
        except Exception as e:
            self.log(f"Error in efficient spatial clustering analysis: {e}", "ERROR", "_perform_spatial_clustering_analysis")
    
    def _cluster_by_sectors(self, positioned_tokens):
        """Cluster tokens by field sectors, respecting spatial locality"""
        sector_clusters = {}
        
        for token_data in positioned_tokens:
            sector = token_data.get("sector", "unknown")
            
            if sector not in sector_clusters:
                sector_clusters[sector] = []
            
            sector_clusters[sector].append(token_data)
        
        # Further subdivide large sectors using local proximity
        refined_clusters = {}
        cluster_id = 0
        
        for sector, tokens in sector_clusters.items():
            if len(tokens) <= 5:
                # Small sector, keep as single cluster
                refined_clusters[f"{sector}_cluster_{cluster_id}"] = tokens
                cluster_id += 1
            else:
                # Large sector, subdivide by local proximity
                subclusters = self._subdivide_sector_by_proximity(tokens, max_cluster_size=5)
                for subcluster in subclusters:
                    refined_clusters[f"{sector}_cluster_{cluster_id}"] = subcluster
                    cluster_id += 1
        
        return refined_clusters
    
    def _subdivide_sector_by_proximity(self, tokens, max_cluster_size=5):
        """Subdivide sector tokens by local proximity without expensive calculations"""
        if len(tokens) <= max_cluster_size:
            return [tokens]
        
        subclusters = []
        remaining_tokens = tokens.copy()
        
        while remaining_tokens:
            # Start new cluster with first remaining token
            cluster_seed = remaining_tokens.pop(0)
            current_cluster = [cluster_seed]
            
            # Add nearby tokens to cluster (simple distance check)
            seed_pos = cluster_seed["position"]
            tokens_to_remove = []
            
            for token_data in remaining_tokens:
                if len(current_cluster) >= max_cluster_size:
                    break
                    
                token_pos = token_data["position"]
                # Simple 3D distance (only using x,y,z for efficiency)
                distance = ((seed_pos["x"] - token_pos["x"])**2 + 
                           (seed_pos["y"] - token_pos["y"])**2 + 
                           (seed_pos["z"] - token_pos["z"])**2)**0.5
                
                if distance <= 2.0:  # Local proximity threshold
                    current_cluster.append(token_data)
                    tokens_to_remove.append(token_data)
            
            # Remove clustered tokens from remaining
            for token_data in tokens_to_remove:
                remaining_tokens.remove(token_data)
            
            subclusters.append(current_cluster)
        
        return subclusters
    
    def _spatial_cluster_tokens(self, positioned_tokens, distance_threshold=5.0):
        """Cluster tokens based on their spatial proximity"""
        clusters = {}
        cluster_id = 0
        
        for token_data in positioned_tokens:
            token = token_data["token"]
            pos = token_data["position"]
            
            # Find closest existing cluster
            best_cluster = None
            min_distance = float('inf')
            
            for cid, cluster_tokens in clusters.items():
                # Calculate distance to cluster center
                cluster_center = self._calculate_cluster_center([t["position"] for t in cluster_tokens])
                distance = self._adaptive_distance(pos, cluster_center)
                
                if distance < min_distance and distance <= distance_threshold:
                    min_distance = distance
                    best_cluster = cid
            
            # Add to existing cluster or create new one
            if best_cluster is not None:
                clusters[best_cluster].append(token_data)
            else:
                clusters[cluster_id] = [token_data]
                cluster_id += 1
                
        return clusters
    
    def _calculate_cluster_center(self, positions):
        """Calculate the center point of a cluster of positions"""
        if not positions:
            return {"x": 0, "y": 0, "z": 0}
            
        avg_x = sum(p["x"] for p in positions) / len(positions)
        avg_y = sum(p["y"] for p in positions) / len(positions)
        avg_z = sum(p["z"] for p in positions) / len(positions)
        
        return {"x": avg_x, "y": avg_y, "z": avg_z}
    
    def _adaptive_distance(self, pos1, pos2):
        """Calculate distance using adaptive engine methods instead of hardcoded Euclidean"""
        try:
            # Try to use adaptive engine for distance calculation
            if self.agent_categorizer and hasattr(self.agent_categorizer, 'memory'):
                memory = self.agent_categorizer.memory
                if hasattr(memory, 'field') and memory.field and hasattr(memory.field, 'adaptive_engine'):
                    adaptive_engine = memory.field.adaptive_engine
                    
                    # Convert position dictionaries to format expected by adaptive engine
                    if hasattr(adaptive_engine, 'calculate_distance'):
                        # Create temporary position objects for distance calculation
                        pos1_vec = [pos1["x"], pos1["y"], pos1["z"]]
                        pos2_vec = [pos2["x"], pos2["y"], pos2["z"]]
                        
                        distance = adaptive_engine.calculate_distance(pos1_vec, pos2_vec)
                        return distance
            
            # Fallback to Euclidean if adaptive engine not available
            dx = pos1["x"] - pos2["x"]
            dy = pos1["y"] - pos2["y"]
            dz = pos1["z"] - pos2["z"]
            return (dx**2 + dy**2 + dz**2)**0.5
            
        except Exception as e:
            self.log(f"Error in adaptive distance calculation, using Euclidean fallback: {e}", "DEBUG", "_adaptive_distance")
            # Safe fallback to Euclidean
            dx = pos1["x"] - pos2["x"]
            dy = pos1["y"] - pos2["y"]
            dz = pos1["z"] - pos2["z"]
            return (dx**2 + dy**2 + dz**2)**0.5
    
    async def _analyze_spatial_cluster(self, cluster_tokens):
        """Analyze a spatial cluster to generate translation mappings"""
        try:
            # Extract compressed tokens from cluster
            compressed_tokens = [t["token"] for t in cluster_tokens if self._is_compressed_token(t["token"])]
            
            if not compressed_tokens:
                return
                
            # Look for human language correlates in the cluster
            for token_data in cluster_tokens:
                token = token_data["token"]
                if token in self.context_correlations:
                    context_data = self.context_correlations[token]
                    
                    # Extract human candidates from cluster context
                    human_candidates = self._extract_human_candidates(token, context_data)
                    
                    # Create cross-cluster translation mappings
                    for compressed_token in compressed_tokens:
                        for human_candidate in human_candidates:
                            # Calculate confidence based on spatial proximity
                            spatial_confidence = await self._calculate_spatial_cluster_confidence(
                                compressed_token, human_candidate, cluster_tokens
                            )
                            
                            if spatial_confidence > 0.4:  # Lower threshold for spatial analysis
                                # Store translation mapping
                                if compressed_token not in self.translation_candidates:
                                    self.translation_candidates[compressed_token] = []
                                if human_candidate not in self.translation_candidates[compressed_token]:
                                    self.translation_candidates[compressed_token].append(human_candidate)
                                    
                                self.confidence_scores[(compressed_token, human_candidate)] = spatial_confidence
                                
                                self.log(f"Spatial cluster mapping: '{compressed_token}' -> '{human_candidate}' (confidence: {spatial_confidence:.3f})", "DEBUG", "_analyze_spatial_cluster")
                                
        except Exception as e:
            self.log(f"Error analyzing spatial cluster: {e}", "ERROR", "_analyze_spatial_cluster")
    
    async def _calculate_spatial_cluster_confidence(self, compressed_token: str, human_candidate: str, cluster_tokens: List) -> float:
        """Calculate confidence for translation mapping based on spatial cluster analysis"""
        try:
            # Factor 1: Cluster coherence (how tightly clustered are the tokens)
            positions = [t["position"] for t in cluster_tokens]
            cluster_center = self._calculate_cluster_center(positions)
            
            # Calculate average distance from center
            avg_distance = sum(self._adaptive_distance(pos, cluster_center) for pos in positions) / len(positions)
            coherence_factor = max(0.0, 1.0 - (avg_distance / 10.0))  # Normalize to 0-1
            
            # Factor 2: Consciousness levels in cluster
            consciousness_levels = [t.get("consciousness_level", 0.5) for t in cluster_tokens]
            avg_consciousness = sum(consciousness_levels) / len(consciousness_levels)
            
            # Factor 3: Token frequency in cluster
            times_encountered = [t.get("times_encountered", 1) for t in cluster_tokens]
            avg_frequency = sum(times_encountered) / len(times_encountered)
            frequency_factor = min(avg_frequency / 5.0, 1.0)  # Normalize to 0-1
            
            # Weighted combination
            spatial_confidence = (
                coherence_factor * 0.4 +
                avg_consciousness * 0.4 +
                frequency_factor * 0.2
            )
            
            return spatial_confidence
            
        except Exception as e:
            self.log(f"Error calculating spatial cluster confidence: {e}", "ERROR", "_calculate_spatial_cluster_confidence")
            return 0.0
    
    async def _update_legacy_lexicon_positions(self):
        """Backwards compatibility: spawn temporary particles for lexicon entries without positions"""
        try:
            if not self.agent_categorizer or not hasattr(self.agent_categorizer, 'memory'):
                return
                
            memory = self.agent_categorizer.memory
            lexicon_store = getattr(memory, 'lexicon_store', None) if hasattr(memory, 'lexicon_store') else None
            
            if not lexicon_store or not self.field:
                self.log("Cannot update legacy positions: missing lexicon store or field", "WARNING", "_update_legacy_lexicon_positions")
                return
                
            # Get all lexicon entries without positions
            lexicon_memories = await memory.get_memories_by_type("lexicon", limit=500)  # Process in batches
            
            unpositioned_tokens = []
            positioned_count = 0
            
            for memory_entry in lexicon_memories:
                if not isinstance(memory_entry, dict):
                    continue
                    
                token = memory_entry.get("token")
                if not token:
                    continue
                    
                # Check if token already has position
                field_position = memory_entry.get("field_position")
                if field_position and isinstance(field_position, dict):
                    continue  # Already has position
                    
                # Only process compressed tokens without positions
                if self._is_compressed_token(token):
                    unpositioned_tokens.append({
                        "token": token,
                        "entry": memory_entry,
                        "definitions": memory_entry.get("definitions", []),
                        "consciousness_level": memory_entry.get("consciousness_level", 0.5)
                    })
            
            if not unpositioned_tokens:
                self.log("No unpositioned compressed tokens found for legacy update", "INFO", "_update_legacy_lexicon_positions")
                return
                
            self.log(f"Processing {len(unpositioned_tokens)} unpositioned tokens for backwards compatibility", "INFO", "_update_legacy_lexicon_positions")
            
            # Process tokens in small batches to avoid overwhelming the system
            batch_size = 10
            for i in range(0, len(unpositioned_tokens), batch_size):
                batch = unpositioned_tokens[i:i+batch_size]
                
                for token_data in batch:
                    try:
                        position = await self._spawn_temporary_particle_for_position(token_data)
                        if position:
                            # Update the lexicon entry with the new position
                            await self._update_lexicon_entry_position(token_data["token"], position)
                            positioned_count += 1
                            
                    except Exception as e:
                        self.log(f"Error processing token {token_data['token']}: {e}", "WARNING", "_update_legacy_lexicon_positions")
                        continue
                
                # Small delay between batches to prevent system overload
                await asyncio.sleep(0.1)
            
            self.log(f"Legacy position update completed: {positioned_count} tokens updated with positions", "INFO", "_update_legacy_lexicon_positions")
            
        except Exception as e:
            self.log(f"Error in legacy lexicon position update: {e}", "ERROR", "_update_legacy_lexicon_positions")
    
    async def _spawn_temporary_particle_for_position(self, token_data):
        """Spawn a temporary particle based on semantic content to determine position"""
        try:
            token = token_data["token"]
            definitions = token_data["definitions"]
            consciousness_level = token_data["consciousness_level"]
            
            # Create semantic content for particle positioning
            semantic_content = {
                "token": token,
                "compressed_type": "legacy_positioning",
                "semantic_gravity": True,
                "definitions_summary": self._extract_semantic_summary(definitions),
                "consciousness_level": consciousness_level
            }
            
            # Spawn temporary lingual particle using temp=True
            temp_particle_id = f"temp_position_{token}_{datetime.now().timestamp()}"
            
            # Use field to spawn particle with semantic content
            if hasattr(self.field, 'spawn_particle'):
                particle = await self.field.spawn_particle(
                    id=temp_particle_id,
                    type="lingual",
                    temp=True,  # Use temp=True instead of manual temp particle creation
                    temp_purpose="position_generation",  # Purpose-specific behavior
                    metadata=semantic_content,
                    energy=0.05,  # Low energy for temporary particle
                    activation=consciousness_level,
                    particle_source="legacy_positioning",
                    source_particle_id=None,
                    emit_event=False
                )
                
                if particle and hasattr(particle, 'position'):
                    # Extract position - handle both numpy array and object formats
                    particle_pos = particle.position
                    
                    if hasattr(particle_pos, 'x'):
                        # Position object with x, y, z attributes
                        position = {
                            "x": float(particle_pos.x),
                            "y": float(particle_pos.y),
                            "z": float(particle_pos.z),
                            "source": "legacy_temporary_particle",
                            "timestamp": datetime.now().isoformat()
                        }
                    elif hasattr(particle_pos, '__len__') and len(particle_pos) >= 3:
                        # Numpy array or list format
                        position = {
                            "x": float(particle_pos[0]),
                            "y": float(particle_pos[1]),
                            "z": float(particle_pos[2]),
                            "source": "legacy_temporary_particle",
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        # Fallback for unexpected format
                        self.log(f"Unexpected position format for particle {temp_particle_id}: {type(particle_pos)}", "WARNING", "_spawn_temporary_particle_for_position")
                        return None
                    
                    # Remove temporary particle
                    if hasattr(self.field, 'remove_particle_with_id'):
                        await self.field.remove_particle_with_id(temp_particle_id)
                    
                    self.log(f"Generated position for legacy token '{token}': ({position['x']:.2f}, {position['y']:.2f}, {position['z']:.2f})", "DEBUG", "_spawn_temporary_particle_for_position")
                    return position
            
            return None
            
        except Exception as e:
            self.log(f"Error spawning temporary particle for {token_data['token']}: {e}", "ERROR", "_spawn_temporary_particle_for_position")
            return None
    
    def _extract_semantic_summary(self, definitions):
        """Extract semantic summary from definitions for particle positioning"""
        if not definitions:
            return ""
            
        summary_parts = []
        for definition in definitions[:3]:  # Use first 3 definitions
            if isinstance(definition, dict):
                text = definition.get("text", "") or definition.get("wikipedia", "")
                if text:
                    # Extract first 50 characters for semantic positioning
                    summary_parts.append(text[:50])
            elif isinstance(definition, str):
                summary_parts.append(definition[:50])
                
        return " | ".join(summary_parts)
    
    async def _update_lexicon_entry_position(self, token, position):
        """Update lexicon entry with new position data"""
        try:
            if not self.agent_categorizer or not hasattr(self.agent_categorizer, 'memory'):
                return False
                
            memory = self.agent_categorizer.memory
            
            # Get existing lexicon entry
            lexicon_memories = await memory.get_memories_by_type("lexicon")
            
            for memory_entry in lexicon_memories:
                if isinstance(memory_entry, dict) and memory_entry.get("token") == token:
                    # Update with position data
                    memory_entry["field_position"] = position
                    memory_entry["spatial_semantic_data"] = {
                        "has_position": True,
                        "spatial_clusters": [],
                        "nearest_neighbors": [],
                        "semantic_distance_cache": {},
                        "last_spatial_analysis": datetime.now().isoformat(),
                        "spatial_significance": 0.5,  # Default significance
                        "position_source": "legacy_update"
                    }
                    
                    # Update in memory
                    await memory.update(
                        key=f"lexicon_{token.lower()}",
                        value=memory_entry,
                        memory_type="lexicon",
                        source="legacy_position_update"
                    )
                    
                    return True
            
            return False
            
        except Exception as e:
            self.log(f"Error updating lexicon entry position for {token}: {e}", "ERROR", "_update_lexicon_entry_position")
            return False
            return 0.0
    
    async def observe_compressed_token(self, token: str, context: Dict = None):
        """Observe a compressed token for background analysis with enhanced filtering"""
        if not token or len(token) > 20:  # Skip very long tokens
            return
        
        # Enhanced pre-filtering to avoid processing noise
        if not self._is_worth_processing(token):
            return
            
        # Check for recent processing to prevent spam
        if self._is_recently_processed(token):
            self.log(f"Token '{token}' already recently processed, skipping to prevent duplication", "DEBUG", "observe_compressed_token")
            return
        
        # Check if token appears compressed (more consonants than vowels, short)
        if self._is_compressed_token(token):
            timestamp = datetime.now().timestamp()
            self.compressed_token_cache[token].append(timestamp)
            
            # Mark as recently processed
            self._mark_recently_processed(token, timestamp)
            
            # Keep only recent observations (within analysis window)
            cutoff = timestamp - (self.analysis_window_minutes * 60)
            self.compressed_token_cache[token] = [
                t for t in self.compressed_token_cache[token] if t > cutoff
            ]
            
            # Add to analysis queue if frequent enough
            if len(self.compressed_token_cache[token]) >= self.min_token_frequency:
                self.analysis_queue.add(token)
                
            # Store context for translation mapping
            if context:
                self._update_context_correlation(token, context)
    
    def _is_worth_processing(self, token: str) -> bool:
        """Enhanced filtering to determine if token is worth processing"""
        token_lower = token.lower()
        
        # Skip JSON structure tokens completely
        json_structure = {'{', '}', '[', ']', ':', ',', '"', "'", '\\', 'true', 'false', 'null'}
        if token in json_structure:
            return False
        
        # Skip numbers and common separators
        if token.isdigit() or token in {'.', '..', '...', '-', '_', '|', '/', '\\'}:
            return False
        
        # Only skip truly basic structure words - much more permissive
        basic_structure_words = {
            'the', 'and', 'or', 'but', 'a', 'an', 'is', 'are', 'was', 'were',
            'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'from'
        }
        if token_lower in basic_structure_words:
            return False
        
        # Only skip obvious system metadata - allow cognitive concepts
        metadata_fields = {
            'spawn', 'update', 'remove', 'type', 'source', 'timestamp', 'id'
        }
        if token_lower in metadata_fields:
            return False
        
        # Must contain at least one letter
        if not any(c.isalpha() for c in token):
            return False
            
        return True
    
    def _is_recently_processed(self, token: str) -> bool:
        """Check if token was recently processed to avoid spam"""
        if not hasattr(self, '_recent_processing_cache'):
            self._recent_processing_cache = {}
        
        current_time = datetime.now().timestamp()
        last_processed = self._recent_processing_cache.get(token, 0)
        
        # Consider "recent" as within last 30 seconds
        return (current_time - last_processed) < 30
    
    def _mark_recently_processed(self, token: str, timestamp: float):
        """Mark token as recently processed"""
        if not hasattr(self, '_recent_processing_cache'):
            self._recent_processing_cache = {}
        
        self._recent_processing_cache[token] = timestamp
        
        # Clean old entries periodically
        if len(self._recent_processing_cache) > 1000:
            cutoff = timestamp - 60  # Keep last minute only
            self._recent_processing_cache = {
                k: v for k, v in self._recent_processing_cache.items() 
                if v > cutoff
            }
    
    def _is_compressed_token(self, token: str) -> bool:
        """Determine if a token appears to be compressed language"""
        if len(token) < 2 or len(token) > 15:
            return False
        
        # Common English words are NOT compressed tokens
        common_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall',
            'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'what', 'where', 'when', 'why', 'how', 'who', 'which', 'whose', 'whom',
            'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
            'one', 'two', 'first', 'last', 'good', 'new', 'old', 'right', 'way', 'even',
            'back', 'any', 'may', 'say', 'get', 'go', 'know', 'take', 'see', 'come', 'think',
            'look', 'want', 'give', 'use', 'find', 'tell', 'ask', 'work', 'seem', 'feel',
            'try', 'leave', 'call', 'good', 'new', 'first', 'last', 'long', 'great', 'little',
            'own', 'other', 'old', 'right', 'big', 'high', 'different', 'small', 'large', 'next',
            'early', 'young', 'important', 'few', 'public', 'bad', 'same', 'able', 'to',
            'time', 'person', 'year', 'way', 'day', 'thing', 'man', 'world', 'life', 'hand',
            'part', 'child', 'eye', 'woman', 'place', 'work', 'week', 'case', 'point', 'government',
            'company', 'number', 'group', 'problem', 'fact', 'money', 'story', 'example', 'lot',
            'water', 'history', 'today', 'school', 'country', 'american', 'information', 'nothing',
            'right', 'against', 'far', 'fun', 'house', 'let', 'put', 'end', 'why', 'turn',
            'american', 'place', 'course', 'business', 'made', 'area', 'available', 'community',
            'home', 'room', 'program', 'policy', 'book', 'lot', 'study', 'game', 'member',
            'power', 'hour', 'lot', 'business', 'eye', 'system', 'program', 'question', 'run',
            'believe', 'hold', 'bring', 'happen', 'write', 'provide', 'sit', 'stand', 'lose',
            'pay', 'meet', 'include', 'continue', 'set', 'learn', 'change', 'lead', 'understand',
            'watch', 'follow', 'stop', 'create', 'speak', 'read', 'allow', 'add', 'spend', 'grow',
            'open', 'walk', 'win', 'offer', 'remember', 'love', 'consider', 'appear', 'buy', 'wait',
            'serve', 'die', 'send', 'expect', 'build', 'stay', 'fall', 'cut', 'reach', 'kill',
            'remain', 'suggest', 'raise', 'pass', 'sell', 'require', 'report', 'decide', 'pull'
        }
        
        if token.lower() in common_words:
            return False
        
        # Geographic/proper names are generally NOT compressed tokens  
        if token.istitle():  # Capitalized words like "German", "Myanmar", "Commons"
            # Check if it looks like a reasonable proper name
            if len(token) >= 3 and token.isalpha():
                return False
        
        # Check for obvious abbreviations or codes that ARE compressed
        if len(token) <= 4 and token.isalpha():
            # Very short alphabetic tokens might be compressed
            vowels = sum(1 for c in token.lower() if c in 'aeiou')
            consonants = len(token) - vowels
            # Only consider it compressed if it's very consonant-heavy
            return consonants >= len(token) * 0.8  # 80% consonants
        
        # Check for hex-like tokens or identifiers
        if any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
            # Mixed alphanumeric (like "a488f11a") likely compressed
            return True
        
        # For longer tokens, use a more sophisticated approach
        if len(token) > 4:
            # Count unusual letter patterns
            unusual_patterns = 0
            
            # Check for consecutive consonants
            consonant_runs = 0
            current_run = 0
            for c in token.lower():
                if c not in 'aeiou':
                    current_run += 1
                else:
                    if current_run >= 3:  # 3+ consecutive consonants
                        consonant_runs += 1
                    current_run = 0
            if current_run >= 3:
                consonant_runs += 1
                
            if consonant_runs > 0:
                unusual_patterns += consonant_runs
            
            # Check for unusual letter combinations
            unusual_combos = ['qb', 'qw', 'qx', 'qz', 'xz', 'bq', 'cq', 'dq', 'fq', 'gq', 'hq', 'jq', 'kq', 'lq', 'mq', 'nq', 'pq', 'rq', 'sq', 'tq', 'vq', 'wq', 'yq', 'zq']
            for combo in unusual_combos:
                if combo in token.lower():
                    unusual_patterns += 1
            
            # Tokens with unusual patterns are likely compressed
            return unusual_patterns >= 2
        
        return False
    
    async def _background_loop(self):
        """Main background processing loop with adaptive intervals - DEPRECATED"""
        try:
            while True:
                # Adaptive interval based on system load and token quality
                current_interval = self._calculate_adaptive_interval()
                await asyncio.sleep(current_interval)
                
                if self._should_process():
                    await self._process_analysis_queue()
                else:
                    # Hybrid approach: run maintenance processing during quiet periods
                    await self._subconscious_maintenance_processing()
                    
        except asyncio.CancelledError:
            self.log("Background processing loop cancelled", "INFO", "_background_loop")
        except Exception as e:
            self.log(f"Error in background processing loop: {e}", "ERROR", "_background_loop")
    
    async def _subconscious_maintenance_processing(self):
        """Process queued translation analysis during subconscious periods"""
        if not self.subconscious_processing_enabled:
            return
            
        current_time = datetime.now().timestamp()
        
        # Only do maintenance every 5 minutes to avoid interference
        if current_time - self.last_maintenance_time < 300:
            return
            
        try:
            # Find lexicon entries that need translation analysis
            pending_tokens = await self._get_tokens_needing_translation_analysis()
            
            if pending_tokens:
                self.log(f"Subconscious maintenance: processing {len(pending_tokens)} pending translation analyses", "INFO", "_subconscious_maintenance_processing")
                
                # Process in small batches to avoid blocking
                batch = pending_tokens[:self.maintenance_batch_size]
                for token_data in batch:
                    await self._process_maintenance_token(token_data)
                
                self.last_maintenance_time = current_time
                
        except Exception as e:
            self.log(f"Error in subconscious maintenance processing: {e}", "ERROR", "_subconscious_maintenance_processing")
    
    async def _get_tokens_needing_translation_analysis(self) -> List[Dict]:
        """Get lexicon entries that need translation analysis"""
        try:
            if not self.agent_categorizer or not hasattr(self.agent_categorizer, 'memory'):
                return []
                
            memory = self.agent_categorizer.memory
            lexicon_memories = await memory.get_memories_by_type("lexicon", limit=100)
            
            pending_tokens = []
            
            for entry in lexicon_memories:
                if not isinstance(entry, dict):
                    continue
                    
                token = entry.get("token")
                if not token or not self._is_compressed_token(token):
                    continue
                
                # Check if needs translation analysis
                needs_analysis = (
                    entry.get("needs_translation_analysis", True) or  # Default to True for backward compatibility
                    entry.get("need_learning", False) or  # Existing flag
                    not entry.get("translation_mappings", [])  # No existing mappings
                )
                
                if needs_analysis:
                    pending_tokens.append({
                        "token": token,
                        "entry": entry,
                        "priority": self._calculate_translation_priority(entry)
                    })
            
            # Sort by priority (highest first)
            pending_tokens.sort(key=lambda x: x["priority"], reverse=True)
            
            return pending_tokens
            
        except Exception as e:
            self.log(f"Error getting tokens needing translation analysis: {e}", "ERROR", "_get_tokens_needing_translation_analysis")
            return []
    
    def _calculate_translation_priority(self, entry: Dict) -> float:
        """Calculate priority for translation analysis"""
        priority = 0.0
        
        # Higher priority for frequently encountered tokens
        times_encountered = entry.get("times_encountered", 1)
        priority += min(times_encountered / 10.0, 1.0) * 0.4
        
        # Higher priority for tokens with rich context
        definitions = entry.get("definitions", [])
        priority += min(len(definitions) / 3.0, 1.0) * 0.3
        
        # Higher priority for recently active tokens
        last_seen = entry.get("last_seen", 0)
        time_since_seen = datetime.now().timestamp() - last_seen
        recency_factor = max(0.0, 1.0 - (time_since_seen / 86400))  # Decay over 24 hours
        priority += recency_factor * 0.3
        
        return priority
    
    async def _process_maintenance_token(self, token_data: Dict):
        """Process a single token during maintenance"""
        try:
            token = token_data["token"]
            entry = token_data["entry"]
            
            # Simulate compressed token observation to trigger analysis
            context = {
                "human_phrase": entry.get("definitions", [{}])[0].get("text", "") if entry.get("definitions") else "",
                "usage_context": "maintenance_processing",
                "maintenance_mode": True
            }
            
            # Add to cache and analysis queue
            self.compressed_token_cache[token].append(datetime.now().timestamp())
            self.analysis_queue.add(token)
            
            # Update context correlations
            self._update_context_correlation(token, context)
            
            # If we have enough tokens for a mini-analysis, process them
            if len(self.analysis_queue) >= 3:
                await self._process_mini_analysis_batch()
                
        except Exception as e:
            self.log(f"Error processing maintenance token {token_data['token']}: {e}", "ERROR", "_process_maintenance_token")
    
    async def _process_mini_analysis_batch(self):
        """Process a small batch during maintenance without full analysis overhead"""
        try:
            if self.is_processing:
                return
                
            self.is_processing = True
            
            # Take small batch for maintenance processing
            tokens_to_analyze = list(self.analysis_queue)[:3]  # Only 3 tokens
            for token in tokens_to_analyze:
                self.analysis_queue.discard(token)
            
            if tokens_to_analyze:
                # Lightweight analysis - just update translation mappings
                await self._update_translation_mappings_only(tokens_to_analyze)
                
                # Update lexicon entries with new translation data
                await self._embed_translations_in_lexicon(tokens_to_analyze)
                
                self.log(f"Maintenance mini-batch completed: {len(tokens_to_analyze)} tokens processed", "DEBUG", "_process_mini_analysis_batch")
                
        except Exception as e:
            self.log(f"Error in mini analysis batch: {e}", "ERROR", "_process_mini_analysis_batch")
        finally:
            self.is_processing = False
    
    async def _update_translation_mappings_only(self, tokens: List[str]):
        """Lightweight translation mapping update without full semantic gravity analysis"""
        try:
            for token in tokens:
                context_data = self.context_correlations.get(token, {})
                human_candidates = self._extract_human_candidates(token, context_data)
                
                for human_candidate in human_candidates:
                    # Simple confidence calculation for maintenance mode
                    confidence = self._calculate_simple_confidence(token, human_candidate, context_data)
                    
                    if confidence > 0.4:  # Lower threshold for maintenance processing
                        if token not in self.translation_candidates:
                            self.translation_candidates[token] = []
                        if human_candidate not in self.translation_candidates[token]:
                            self.translation_candidates[token].append(human_candidate)
                            
                        self.confidence_scores[(token, human_candidate)] = confidence
                        
        except Exception as e:
            self.log(f"Error updating translation mappings: {e}", "ERROR", "_update_translation_mappings_only")
    
    def _calculate_simple_confidence(self, token: str, human_candidate: str, context_data: Dict) -> float:
        """Simplified confidence calculation for maintenance processing"""
        confidence = 0.3  # Base confidence
        
        # Boost for context co-occurrence
        human_phrases = context_data.get("nearby_human_phrases", [])
        if any(human_candidate.lower() in phrase.lower() for phrase in human_phrases):
            confidence += 0.3
            
        # Boost for definitions
        definitions = context_data.get("definitions", [])
        if definitions:
            confidence += 0.2
            
        # Boost for strong compression pattern
        if self._has_strong_compression_pattern(token):
            confidence += 0.2
            
        return min(confidence, 1.0)
    
    async def _embed_translations_in_lexicon(self, tokens: List[str]):
        """Embed translation mappings directly into lexicon entries"""
        try:
            if not self.agent_categorizer or not hasattr(self.agent_categorizer, 'memory'):
                return
                
            memory = self.agent_categorizer.memory
            
            for token in tokens:
                # Get current lexicon entry
                lexicon_key = f"lexicon_{token.lower()}"
                existing_entry = await memory.query(lexicon_key)
                
                if existing_entry and isinstance(existing_entry, dict):
                    # Add translation mappings to the entry
                    translation_mappings = []
                    candidates = self.translation_candidates.get(token, [])
                    
                    for candidate in candidates:
                        confidence = self.confidence_scores.get((token, candidate), 0.0)
                        translation_mappings.append({
                            "human_form": candidate,
                            "confidence": confidence,
                            "confidence_category": self._get_confidence_category(confidence),
                            "last_updated": datetime.now().isoformat(),
                            "source": "background_processor"
                        })
                    
                    # Update the entry
                    existing_entry.update({
                        "translation_mappings": translation_mappings,
                        "needs_translation_analysis": len(translation_mappings) == 0,  # False if we have mappings
                        "last_translation_update": datetime.now().isoformat(),
                        "translation_status": "completed" if translation_mappings else "pending"
                    })
                    
                    # Save back to memory
                    await memory.update(
                        key=lexicon_key,
                        value=existing_entry,
                        memory_type="lexicon",
                        source="translation_embedding"
                    )
                    
                    self.log(f"Embedded {len(translation_mappings)} translations for token '{token}' in lexicon", "DEBUG", "_embed_translations_in_lexicon")
                    
        except Exception as e:
            self.log(f"Error embedding translations in lexicon: {e}", "ERROR", "_embed_translations_in_lexicon")
    
    def _calculate_adaptive_interval(self) -> float:
        """Calculate adaptive processing interval based on current conditions"""
        # Base interval
        interval = self.processing_interval
        
        # Adjust based on queue size
        queue_size = len(self.analysis_queue)
        if queue_size == 0:
            interval = self.max_processing_interval  # Longest wait when nothing to do
        elif queue_size > 10:
            interval = self.min_processing_interval  # Shortest wait when busy
        elif queue_size > 5:
            interval = self.processing_interval * 0.7  # Slightly faster
        else:
            interval = self.processing_interval * 1.3  # Slightly slower
        
        # Adjust based on recent processing success
        if hasattr(self, '_recent_processing_success'):
            if self._recent_processing_success:
                interval *= 0.9  # Process faster when successful
            else:
                interval *= 1.5  # Process slower when failing
        
        # Clamp to min/max bounds
        interval = max(self.min_processing_interval, min(self.max_processing_interval, interval))
        
        # Log adaptation for debugging
        if interval != self.processing_interval:
            self.log(f"Adaptive interval: {interval}s (queue: {queue_size}, base: {self.processing_interval}s)", "DEBUG", "_calculate_adaptive_interval")
        
        return interval
    
    def _should_process(self) -> bool:
        """Determine if processing should occur this cycle - on-demand approach"""
        # Priority 1: Process immediately if we have high-value tokens
        high_value_tokens = self._get_high_value_tokens()
        if high_value_tokens:
            self.log(f"Triggering immediate processing for {len(high_value_tokens)} high-value tokens", "INFO", "_should_process")
            return True
        
        # Priority 2: Process if queue has diverse, meaningful tokens
        if len(self.analysis_queue) >= 8 and self._has_token_diversity():
            return True
        
        # Priority 3: Time-based processing with much longer intervals to reduce load
        time_since_last = datetime.now().timestamp() - self.last_analysis_time
        if time_since_last > 900 and len(self.analysis_queue) > 2:  # 15 minutes instead of 5
            self.log(f"Time-based processing: {time_since_last/60:.1f} minutes since last analysis", "DEBUG", "_should_process")
            return True
        
        # Priority 4: Emergency processing to prevent queue overflow
        if len(self.analysis_queue) >= 20:
            self.log(f"Emergency processing: queue size {len(self.analysis_queue)}", "WARNING", "_should_process")
            return True
            
        return False
    
    def _get_high_value_tokens(self) -> List[str]:
        """Identify tokens that should trigger immediate processing"""
        high_value = []
        
        for token in self.analysis_queue:
            # High-value criteria
            observations = self.compressed_token_cache.get(token, [])
            
            # Tokens with burst activity
            if len(observations) >= 5:
                recent_observations = [t for t in observations if t > (datetime.now().timestamp() - 120)]  # Last 2 minutes
                if len(recent_observations) >= 3:  # Burst of activity
                    high_value.append(token)
                    continue
            
            # Tokens with strong compression characteristics
            if self._has_strong_compression_pattern(token):
                high_value.append(token)
                continue
            
            # Tokens with rich context data
            context_data = self.context_correlations.get(token, {})
            if (len(context_data.get("definitions", [])) > 0 or 
                len(context_data.get("nearby_human_phrases", [])) > 2):
                high_value.append(token)
        
        return high_value
    
    def _has_strong_compression_pattern(self, token: str) -> bool:
        """Check if token shows strong compression characteristics"""
        # Very short with many consonants
        if len(token) <= 4:
            vowels = sum(1 for c in token.lower() if c in 'aeiou')
            consonants = len(token) - vowels
            if consonants >= vowels * 2:  # 2:1 consonant ratio
                return True
        
        # Contains uncommon letter combinations
        rare_combinations = ['gh', 'dh', 'bh', 'kh', 'ng', 'nk', 'ps', 'ts', 'ks']
        if any(combo in token.lower() for combo in rare_combinations):
            return True
        
        return False
    
    def _has_token_diversity(self) -> bool:
        """Check if queue has diverse types of tokens worth processing together"""
        if len(self.analysis_queue) < 3:
            return False
        
        # Check for diversity in token lengths
        lengths = [len(token) for token in self.analysis_queue]
        length_diversity = len(set(lengths)) >= 3
        
        # Check for diversity in compression patterns
        compression_types = set()
        for token in self.analysis_queue:
            if self._has_strong_compression_pattern(token):
                compression_types.add("strong_compression")
            elif len(token) <= 6:
                compression_types.add("short_form")
            else:
                compression_types.add("longer_form")
        
        pattern_diversity = len(compression_types) >= 2
        
        return length_diversity and pattern_diversity
    
    def _is_meaningful_regex_token(self, token: str) -> bool:
        """Check if a regex-extracted token is meaningful (kept for compatibility but unused in simplified parsing)"""
        if not token or len(token) < 2 or len(token) > 20:
            return False
        
        # Skip structure characters and common words
        structure_chars = {'{', '}', '[', ']', ':', ',', '"', "'", '\\', '_', '-', '.', '/', '(', ')', '%', '?', '!'}
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'token', 'key', 'value'}
        
        return (token not in structure_chars 
                and token.lower() not in common_words
                and token.isalpha()  # Only alphabetic
                and any(c in token.lower() for c in 'aeiou'))  # Must have vowels

    async def _process_analysis_queue(self):
        """Process the current analysis queue"""
        if self.is_processing or not self.analysis_queue:
            return
            
        self.is_processing = True
        try:
            # Get tokens to analyze
            tokens_to_analyze = list(self.analysis_queue)
            self.analysis_queue.clear()
            
            self.log(f"Processing {len(tokens_to_analyze)} compressed tokens for semantic analysis", "INFO", "_process_analysis_queue")
            
            # Perform semantic gravity analysis
            if self.gravity_analyzer:
                analysis_results = await self._analyze_token_gravity(tokens_to_analyze)
                
                # Generate categories from results
                if self.agent_categorizer:
                    await self._generate_categories_from_analysis(analysis_results)
                
                # Update translation mappings
                await self._update_translation_mappings(tokens_to_analyze, analysis_results)
                
                # Save translation mappings to persistent memory
                await self._save_translation_mappings_to_memory()
                
            self.last_analysis_time = datetime.now().timestamp()
            
            # Track processing success for adaptive intervals
            self._recent_processing_success = len(tokens_to_analyze) > 0 and analysis_results
            
        except Exception as e:
            self.log(f"Error processing analysis queue: {e}", "ERROR", "_process_analysis_queue")
            self._recent_processing_success = False
        finally:
            self.is_processing = False
    
    async def _analyze_token_gravity(self, tokens: List[str]) -> Dict:
        """Analyze semantic gravity for a set of tokens"""
        try:
            self.log(f"Starting semantic gravity analysis for {len(tokens)} tokens", "DEBUG", "_analyze_token_gravity")
            
            # Add timeout protection to prevent hangs
            import asyncio
            
            async def run_analysis():
                # Use the semantic gravity analyzer
                self.log("Calling gravity_analyzer.analyze_compressed_speech...", "DEBUG", "_analyze_token_gravity")
                analysis = self.gravity_analyzer.analyze_compressed_speech(tokens)
                self.log("Gravity analyzer completed successfully", "DEBUG", "_analyze_token_gravity")
                
                # Enhance with frequency and temporal data
                self.log("Enhancing analysis with frequency and temporal data...", "DEBUG", "_analyze_token_gravity")
                enhanced_analysis = {
                    **analysis,
                    "frequency_analysis": self._calculate_token_frequencies(tokens),
                    "temporal_patterns": self._analyze_temporal_patterns(tokens),
                    "context_correlations": {token: self.context_correlations.get(token, {}) for token in tokens}
                }
                self.log("Analysis enhancement completed", "DEBUG", "_analyze_token_gravity")
                
                return enhanced_analysis
            
            # Run with timeout to prevent infinite hangs
            try:
                analysis_result = await asyncio.wait_for(run_analysis(), timeout=30.0)  # 30 second timeout
                self.log("Semantic gravity analysis completed successfully", "DEBUG", "_analyze_token_gravity")
                return analysis_result
            except asyncio.TimeoutError:
                self.log("Semantic gravity analysis timed out after 30 seconds - returning empty analysis", "WARNING", "_analyze_token_gravity")
                return {
                    "gravitational_clusters": {},
                    "semantic_density_map": {},
                    "compression_mechanisms": {},
                    "thermal_gravity_correlation": {},
                    "frequency_analysis": self._calculate_token_frequencies(tokens),
                    "temporal_patterns": {},
                    "context_correlations": {},
                    "prediction_confidence": 0.0,
                    "timeout_occurred": True
                }
            
        except Exception as e:
            self.log(f"Error in token gravity analysis: {e}", "ERROR", "_analyze_token_gravity")
            import traceback
            self.log(f"Analysis error traceback: {traceback.format_exc()}", "DEBUG", "_analyze_token_gravity")
            return {}
    
    def _calculate_token_frequencies(self, tokens: List[str]) -> Dict[str, float]:
        """Calculate relative frequencies and growth rates for tokens"""
        frequencies = {}
        
        for token in tokens:
            observations = self.compressed_token_cache.get(token, [])
            if observations:
                # Recent frequency (last 5 minutes)
                recent_cutoff = datetime.now().timestamp() - 300
                recent_count = sum(1 for t in observations if t > recent_cutoff)
                
                # Total frequency
                total_count = len(observations)
                
                # Growth rate (recent vs total average)
                window_minutes = self.analysis_window_minutes
                avg_rate = total_count / window_minutes if window_minutes > 0 else 0
                recent_rate = recent_count / 5.0  # per 5 minutes
                
                frequencies[token] = {
                    "total_count": total_count,
                    "recent_count": recent_count,
                    "avg_rate": avg_rate,
                    "recent_rate": recent_rate,
                    "growth_factor": recent_rate / avg_rate if avg_rate > 0 else 1.0
                }
                
        return frequencies
    
    def _analyze_temporal_patterns(self, tokens: List[str]) -> Dict[str, Dict]:
        """Analyze temporal usage patterns for tokens"""
        patterns = {}
        
        for token in tokens:
            observations = self.compressed_token_cache.get(token, [])
            if len(observations) < 2:
                continue
                
            # Calculate intervals between observations
            intervals = [observations[i] - observations[i-1] for i in range(1, len(observations))]
            
            if intervals:
                patterns[token] = {
                    "avg_interval": np.mean(intervals),
                    "interval_variance": np.var(intervals),
                    "usage_regularity": 1.0 / (1.0 + np.var(intervals)),  # Higher = more regular
                    "burst_patterns": self._detect_burst_patterns(observations)
                }
                
        return patterns
    
    def _detect_burst_patterns(self, timestamps: List[float]) -> Dict:
        """Detect burst patterns in token usage"""
        if len(timestamps) < 3:
            return {"has_bursts": False}
            
        intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        avg_interval = np.mean(intervals)
        
        # Detect bursts (intervals much shorter than average)
        burst_threshold = avg_interval * 0.3
        burst_count = sum(1 for interval in intervals if interval < burst_threshold)
        
        return {
            "has_bursts": burst_count > 0,
            "burst_count": burst_count,
            "burst_ratio": burst_count / len(intervals) if intervals else 0
        }
    
    async def _generate_categories_from_analysis(self, analysis_results: Dict):
        """Generate categories from gravitational analysis results"""
        try:
            gravitational_clusters = analysis_results.get("gravitational_clusters", {})
            
            for cluster_name, cluster_tokens in gravitational_clusters.items():
                if len(cluster_tokens) >= 2:  # Only create categories for meaningful clusters
                    # Generate category using gravitational categorization system
                    if hasattr(self.agent_categorizer, 'request_categorization'):
                        category_content = {
                            "cluster_tokens": cluster_tokens,
                            "analysis_timestamp": datetime.now().timestamp(),
                            "generation_method": "background_gravity_analysis",
                            "frequency_data": analysis_results.get("frequency_analysis", {}),
                            "temporal_patterns": analysis_results.get("temporal_patterns", {})
                        }
                        
                        # Create compressed category name from cluster
                        compressed_category = self._create_compressed_category_name(cluster_tokens)
                        
                        await self.agent_categorizer.request_categorization(
                            category_content, compressed_category
                        )
                        
                        self.log(f"Generated category '{compressed_category}' from gravity cluster {cluster_name}", "INFO", "_generate_categories_from_analysis")
                        
        except Exception as e:
            self.log(f"Error generating categories from analysis: {e}", "ERROR", "_generate_categories_from_analysis")
    
    def _create_compressed_category_name(self, cluster_tokens: List[str]) -> str:
        """Create a compressed category name from cluster tokens"""
        # Take consonant patterns from most frequent tokens
        consonant_patterns = []
        
        for token in cluster_tokens[:3]:  # Use top 3 tokens
            consonants = ''.join(c for c in token if c not in 'aeiou')
            if consonants:
                consonant_patterns.append(consonants[:2])
                
        if consonant_patterns:
            base_name = ''.join(consonant_patterns)
            return f"{base_name[:8]}gr"  # 'gr' for gravity-generated
        else:
            return f"gr{len(cluster_tokens)}"
    
    async def _update_translation_mappings(self, tokens: List[str], analysis_results: Dict):
        """Update translation mappings between compressed and human language"""
        try:
            # Get context correlations for each token
            for token in tokens:
                context_data = self.context_correlations.get(token, {})
                
                # Extract potential human language correlates from context
                human_candidates = self._extract_human_candidates(token, context_data)
                
                # Calculate confidence scores using semantic gravity
                for human_candidate in human_candidates:
                    confidence = await self._calculate_translation_confidence(
                        token, human_candidate, analysis_results, context_data
                    )
                    
                    # Add small random variance to prevent identical mappings
                    confidence_variance = random.uniform(-0.05, 0.05)
                    confidence = max(0.0, min(1.0, confidence + confidence_variance))
                    
                    self.confidence_scores[(token, human_candidate)] = confidence
                    
                    # Store high-confidence mappings with more selective threshold
                    confidence_threshold = 0.5 if len(human_candidates) > 6 else 0.6
                    if confidence > confidence_threshold:
                        if token not in self.translation_candidates:
                            self.translation_candidates[token] = []
                        if human_candidate not in self.translation_candidates[token]:
                            self.translation_candidates[token].append(human_candidate)
                            
                        self.log(f"Translation mapping: '{token}' -> '{human_candidate}' (confidence: {confidence:.3f})", "DEBUG", "_update_translation_mappings")
                        
        except Exception as e:
            self.log(f"Error updating translation mappings: {e}", "ERROR", "_update_translation_mappings")
    
    def _extract_human_candidates(self, compressed_token: str, context_data: Dict) -> List[str]:
        """Extract potential human language candidates from context"""
        candidates = []
        
        # Look for patterns in context where human language appears near compressed tokens
        human_phrases = context_data.get("nearby_human_phrases", [])
        questions = context_data.get("source_questions", [])
        definitions = context_data.get("definitions", [])
        
        self.log(f"Extracting candidates for '{compressed_token}': {len(definitions)} definitions, {len(human_phrases)} phrases", "DEBUG", "_extract_human_candidates")
        
        # Extract from definitions first (most reliable source)
        if definitions:
            for definition in definitions:
                if isinstance(definition, dict):
                    # Handle different definition formats
                    def_text = ""
                    if "text" in definition:
                        def_text = definition["text"]
                    elif "wikipedia" in definition:
                        def_text = definition["wikipedia"]
                    elif isinstance(definition, str):
                        def_text = definition
                    
                    if def_text:
                        # Extract meaningful words from definition
                        words = self._extract_meaningful_words(def_text)
                        candidates.extend(words[:5])  # Take top 5 from each definition (increased from 3)
                elif isinstance(definition, str):
                    words = self._extract_meaningful_words(definition)
                    candidates.extend(words[:5])
        
        # Extract keywords from human context
        for phrase in human_phrases + questions:
            words = self._extract_meaningful_words(phrase)
            candidates.extend(words[:3])  # Keep 3 from general context
            
        # Add some conceptual diversity by including related semantic concepts
        semantic_expansions = self._expand_semantic_concepts(compressed_token, candidates)
        candidates.extend(semantic_expansions)
            
        # Remove duplicates, filter by relevance, and return top candidates
        unique_candidates = list(set(candidates))
        # Filter out very short or common words
        filtered_candidates = [c for c in unique_candidates 
                             if len(c) > 2 and c.lower() not in self._get_stop_words()]
        
        # Sort by length and complexity (prefer more descriptive words)
        filtered_candidates.sort(key=lambda x: (len(x), x.count('ion') + x.count('ing') + x.count('ment')), reverse=True)
        
        final_candidates = filtered_candidates[:12]  # Return top 12 candidates (increased from 8)
        
        self.log(f"Generated {len(final_candidates)} candidates for '{compressed_token}': {final_candidates}", "DEBUG", "_extract_human_candidates")
        
        return final_candidates
    
    def _expand_semantic_concepts(self, compressed_token: str, existing_candidates: List[str]) -> List[str]:
        """Expand semantic concepts to add diversity to candidate pool"""
        expansions = []
        
        # Create conceptual expansions based on token characteristics
        if len(compressed_token) == 2:
            # Two-letter tokens might be abbreviations
            conceptual_expansions = [
                "concept", "process", "action", "state", "method", "system",
                "function", "operation", "element", "component", "structure"
            ]
        else:
            # Longer tokens might be compound concepts
            conceptual_expansions = [
                "analysis", "synthesis", "evaluation", "processing", "mechanism",
                "framework", "pattern", "relationship", "interaction", "phenomenon"
            ]
        
        # Add expansions that aren't already in candidates
        for expansion in conceptual_expansions:
            if expansion not in existing_candidates:
                expansions.append(expansion)
                if len(expansions) >= 3:  # Limit semantic expansions
                    break
        
        return expansions
    
    def _extract_meaningful_words(self, text: str) -> List[str]:
        """Extract meaningful words from text, filtering out common words"""
        if not text:
            return []
            
        # Clean and split text
        words = text.lower().replace('[', '').replace(']', '').replace('(', '').replace(')', '').split()
        
        # Filter for meaningful words
        stop_words = self._get_stop_words()
        meaningful_words = []
        
        for w in words:
            if (len(w) > 2 
                and w not in stop_words
                and w.isalpha()  # Only alphabetic words
                and not w.startswith('http')):  # Exclude URLs
                meaningful_words.append(w)
        
        # Log extracted words for debugging
        if meaningful_words:
            self.log(f"Extracted meaningful words from '{text[:50]}...': {meaningful_words[:5]}", "DEBUG", "_extract_meaningful_words")
        
        return meaningful_words
    
    def _get_stop_words(self) -> set:
        """Get set of common stop words to filter out"""
        return {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 
            'did', 'man', 'use', 'way', 'when', 'were', 'will', 'with', 'that',
            'this', 'they', 'from', 'have', 'been', 'each', 'which', 'their',
            'said', 'she', 'than', 'what', 'make', 'time', 'very', 'about'
        }
    
    async def _calculate_translation_confidence(self, compressed_token: str, human_candidate: str, analysis_results: Dict, context_data: Dict) -> float:
        """Calculate confidence score for a translation mapping"""
        confidence_factors = []
        
        # Factor 1: Frequency correlation
        token_freq = analysis_results.get("frequency_analysis", {}).get(compressed_token, {})
        if token_freq:
            # Higher frequency compressed tokens are more likely to map to common concepts
            freq_factor = min(token_freq.get("total_count", 0) / 10.0, 1.0)
            confidence_factors.append(freq_factor)
            
        # Factor 2: Context co-occurrence
        context_factor = self._calculate_context_cooccurrence(compressed_token, human_candidate, context_data)
        confidence_factors.append(context_factor)
        
        # Factor 3: Semantic similarity (if field available)
        if self.field:
            semantic_factor = await self._calculate_semantic_similarity(compressed_token, human_candidate)
            confidence_factors.append(semantic_factor)
            
        # Factor 4: Gravitational clustering strength
        gravity_factor = self._calculate_gravity_factor(compressed_token, analysis_results)
        confidence_factors.append(gravity_factor)
        
        # Weighted average
        if confidence_factors:
            weights = [0.3, 0.4, 0.2, 0.1][:len(confidence_factors)]
            weighted_sum = sum(w * f for w, f in zip(weights, confidence_factors))
            total_weight = sum(weights)
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return 0.0
    
    def _calculate_context_cooccurrence(self, compressed_token: str, human_candidate: str, context_data: Dict) -> float:
        """Calculate how often compressed token and human candidate appear in similar contexts"""
        # Simple implementation - check if they appear in same conversations
        human_phrases = context_data.get("nearby_human_phrases", [])
        
        cooccurrence_count = 0
        for phrase in human_phrases:
            if human_candidate.lower() in phrase.lower():
                cooccurrence_count += 1
                
        return min(cooccurrence_count / 3.0, 1.0)  # Normalize to 0-1
    
    async def _calculate_semantic_similarity(self, compressed_token: str, human_candidate: str) -> float:
        """Calculate semantic similarity using particle field memory and spatial positions"""
        try:
            if not self.field:
                return await self._calculate_spatial_semantic_similarity(compressed_token, human_candidate)
                
            memory_particles = self.field.get_particles_by_type("memory")
            
            # Find particles containing these tokens
            compressed_particles = [p for p in memory_particles if compressed_token in str(p.metadata.get("content", ""))]
            human_particles = [p for p in memory_particles if human_candidate in str(p.metadata.get("content", ""))]
            
            if not compressed_particles or not human_particles:
                # Fallback to spatial analysis if no particles found
                return await self._calculate_spatial_semantic_similarity(compressed_token, human_candidate)
                
            # Calculate average distance between particle sets
            total_similarity = 0.0
            comparison_count = 0
            
            for cp in compressed_particles[:3]:  # Limit for performance
                for hp in human_particles[:3]:
                    # Use semantic gravity calculation
                    distance = cp.distance_to(hp) if hasattr(cp, 'distance_to') else 1.0
                    similarity = 1.0 / (distance + 0.1)  # Convert distance to similarity
                    total_similarity += similarity
                    comparison_count += 1
                    
            return total_similarity / comparison_count if comparison_count > 0 else 0.0
            
        except Exception as e:
            self.log(f"Error calculating semantic similarity: {e}", "ERROR", "_calculate_semantic_similarity")
            return 0.0
    
    async def _calculate_spatial_semantic_similarity(self, compressed_token: str, human_candidate: str) -> float:
        """Calculate semantic similarity using stored field positions when particles aren't available"""
        try:
            # Get lexicon store from agent categorizer
            if not self.agent_categorizer or not hasattr(self.agent_categorizer, 'memory'):
                return 0.0
                
            memory = self.agent_categorizer.memory
            lexicon_store = getattr(memory, 'lexicon_store', None) if hasattr(memory, 'lexicon_store') else None
            
            if not lexicon_store:
                return 0.0
                
            # Calculate spatial distance if both tokens have positions
            spatial_distance = lexicon_store.calculate_spatial_semantic_distance(compressed_token, human_candidate)
            
            if spatial_distance:
                distance = spatial_distance["distance"]
                # Convert distance to similarity (closer = more similar)
                max_meaningful_distance = 10.0  # Adjust based on field scale
                similarity = max(0.0, 1.0 - (distance / max_meaningful_distance))
                
                self.log(f"Spatial semantic similarity: {compressed_token} ↔ {human_candidate} = {similarity:.3f} (distance: {distance:.3f})", "DEBUG", "_calculate_spatial_semantic_similarity")
                return similarity
            
            return 0.0
            
        except Exception as e:
            self.log(f"Error calculating spatial semantic similarity: {e}", "ERROR", "_calculate_spatial_semantic_similarity")
            return 0.0
    
    def _calculate_gravity_factor(self, compressed_token: str, analysis_results: Dict) -> float:
        """Calculate factor based on gravitational clustering strength"""
        clusters = analysis_results.get("gravitational_clusters", {})
        
        for cluster_tokens in clusters.values():
            if compressed_token in cluster_tokens:
                # Tokens in larger clusters are more gravitationally significant
                return min(len(cluster_tokens) / 5.0, 1.0)
                
        return 0.1  # Default for unclustered tokens
    
    async def _analyze_spatial_neighborhood(self, token: str, field_position: Dict, context: Dict):
        """Energy-aware spatial neighborhood analysis using field's built-in constraints"""
        try:
            if not self.field:
                self.log(f"No field available for spatial analysis of token '{token}'", "DEBUG", "_analyze_spatial_neighborhood")
                return
            
            # Check if we have sufficient energy/activation for spatial analysis
            if not self._can_afford_spatial_analysis(context):
                self.log(f"Insufficient energy for spatial analysis of '{token}', deferring", "DEBUG", "_analyze_spatial_neighborhood")
                return
            
            # Get nearby particles using energy-constrained field indexing
            nearby_particles = await self._get_nearby_particles(field_position, radius=2.0)  # Reduced radius for efficiency
            
            if len(nearby_particles) < 2:
                self.log(f"Insufficient spatial neighbors ({len(nearby_particles)}) for analysis of '{token}'", "DEBUG", "_analyze_spatial_neighborhood") 
                return
            
            # Extract human language content with energy considerations
            spatial_human_content = []
            energy_budget = 10  # Limit content extraction operations
            
            for particle_data in nearby_particles:
                if energy_budget <= 0:
                    break
                    
                # Only process high-accessibility particles (low energy cost)
                energy_cost = particle_data.get("energy_cost", 1.0)
                if energy_cost > 0.7:  # Skip expensive particles
                    continue
                    
                human_content = await self._extract_human_content_from_particle(particle_data)
                if human_content:
                    spatial_human_content.extend(human_content[:3])  # Limit per particle
                    energy_budget -= 1
            
            if not spatial_human_content:
                self.log(f"No accessible human content found for '{token}' spatial analysis", "DEBUG", "_analyze_spatial_neighborhood")
                return
            
            # Store spatial correlations with energy metrics
            if token in self.context_correlations:
                correlation_data = self.context_correlations[token]
                correlation_data["spatial_correlations"] = {
                    "nearby_particle_count": len(nearby_particles),
                    "spatial_human_content": spatial_human_content[:8],  # Reduced from 10
                    "neighborhood_semantic_density": len(spatial_human_content),
                    "energy_constrained": True,  # Flag that energy constraints were applied
                    "accessible_particle_count": len([p for p in nearby_particles if p.get("energy_cost", 1.0) <= 0.7]),
                    "field_position": field_position,
                    "analysis_timestamp": datetime.now().isoformat()
                }
                
                # Generate hybrid translation candidates (energy-aware)
                await self._generate_hybrid_translation_candidates(token, correlation_data)
                
                self.log(f"Energy-aware spatial analysis for '{token}': {len(nearby_particles)} nearby, {len(spatial_human_content)} content extracts", "DEBUG", "_analyze_spatial_neighborhood")
            
        except Exception as e:
            self.log(f"Error in energy-aware spatial analysis for '{token}': {e}", "ERROR", "_analyze_spatial_neighborhood")
    
    def _can_afford_spatial_analysis(self, context: Dict) -> bool:
        """Check if we can afford energy cost for spatial analysis"""
        # Simple heuristic: limit spatial analysis frequency
        current_time = datetime.now().timestamp()
        
        if not hasattr(self, '_last_spatial_analysis_time'):
            self._last_spatial_analysis_time = 0
        
        if not hasattr(self, '_spatial_analysis_count'):
            self._spatial_analysis_count = 0
        
        # Cooldown period to prevent excessive spatial queries
        time_since_last = current_time - self._last_spatial_analysis_time
        if time_since_last < 30:  # 30 second cooldown
            return False
        
        # Rate limiting: max 3 spatial analyses per minute
        if self._spatial_analysis_count > 3:
            if time_since_last < 60:  # Reset counter every minute
                return False
            else:
                self._spatial_analysis_count = 0
        
        # Update counters
        self._last_spatial_analysis_time = current_time
        self._spatial_analysis_count += 1
        
        return True
    
    async def _get_nearby_particles(self, field_position: Dict, radius: float = 3.0) -> List[Dict]:
        """Get particles using field's efficient spatial indexing system"""
        try:
            if not self.field or not hasattr(self.field, 'get_spatial_neighbors'):
                self.log("No field spatial indexing available, skipping spatial analysis", "DEBUG", "_get_nearby_particles")
                return []
            
            # Create temp particle using field's spawn_particle method with temp=True
            # Convert field_position dict to proper position array for the particle
            position_array = np.array([
                field_position.get('x', 0.5),           # x (spatial)
                field_position.get('y', 0.5),           # y (spatial)  
                field_position.get('z', 0.5),           # z (spatial)
                field_position.get('w', 0.0),           # w (time of creation)
                field_position.get('t', 0.0),           # t (localized time)
                field_position.get('a', 0.0),           # a (age)
                field_position.get('frequency', 0.0),   # f (frequency)
                field_position.get('memory_phase', 0.5), # m (memory phase)
                field_position.get('v', 0.0),           # v (valence)
                field_position.get('i', 0.5),           # i (identity code)
                field_position.get('n', 0.5),           # n (intent)
            ], dtype=np.float32)
            
            # Add phase vector (circadian phase) to make it 12D
            phase_angle = field_position.get('phase', 0.0)
            phase_vector = np.array([np.cos(phase_angle), np.sin(phase_angle)], dtype=np.float32)
            full_position = np.concatenate((position_array, phase_vector))
            
            temp_particle = await self.field.spawn_particle(
                type="lingual",
                temp=True,
                temp_purpose="spatial_search",  # Purpose-specific behavior
                energy=0.01,
                activation=0.01,
                position=full_position,
                emit_event=False
            )
            
            if not temp_particle:
                self.log("Failed to create temp particle for spatial search", "ERROR", "_get_nearby_particles")
                return []
            
            # Use field's efficient spatial neighbor search with energy constraints
            nearby_particles = self.field.get_spatial_neighbors(temp_particle, radius=min(radius, 1.5))
            
            # Clean up temp particle immediately after use
            temp_particle.alive = False
            self.field.alive_particles.discard(temp_particle.id)
            self.field.particles = [p for p in self.field.particles if p.id != temp_particle.id]
            
            if not nearby_particles:
                self.log(f"No spatial neighbors found within radius {radius} using field indexing", "DEBUG", "_get_nearby_particles")
                return []

            # Convert to our data format with distance calculation
            neighbor_data = []
            
            for particle in nearby_particles[:15]:  # Limit to 15 to prevent expensive operations
                if not hasattr(particle, 'position') or not particle.position:
                    continue

                try:
                    # Use field's energy-constrained distance calculation if available
                    if hasattr(self.field, 'calculate_interaction_energy_cost'):
                        # Energy cost serves as proxy for distance difficulty
                        energy_cost = self.field.calculate_interaction_energy_cost(temp_particle, particle, radius)
                        distance = energy_cost * 100  # Convert to approximate distance
                    else:
                        # Fallback to simple 3D distance using only spatial dimensions
                        temp_pos = temp_particle.position[:3]
                        particle_pos = particle.position[:3] if len(particle.position) >= 3 else particle.position
                        distance = np.linalg.norm(temp_pos - particle_pos)
                    
                    neighbor_data.append({
                        "particle": particle,
                        "distance": distance,
                        "position": {
                            "x": particle.position[0] if len(particle.position) > 0 else 0.5,
                            "y": particle.position[1] if len(particle.position) > 1 else 0.5, 
                            "z": particle.position[2] if len(particle.position) > 2 else 0.5
                        },
                        "metadata": getattr(particle, 'metadata', {}),
                        "type": getattr(particle, 'type', 'unknown'),
                        "energy_cost": getattr(particle, 'energy', 0.5)  # Include energy as accessibility factor
                    })
                    
                except Exception as e:
                    self.log(f"Error processing spatial neighbor: {e}", "DEBUG", "_get_nearby_particles")
                    continue
            
            # Sort by energy accessibility (lower cost = more accessible)
            neighbor_data.sort(key=lambda x: x["distance"])
            
            self.log(f"Found {len(neighbor_data)} spatial neighbors using field indexing (energy-constrained)", "DEBUG", "_get_nearby_particles")
            return neighbor_data
            
        except Exception as e:
            self.log(f"Error in efficient spatial neighbor search: {e}", "ERROR", "_get_nearby_particles")
            return []
    
    async def _extract_human_content_from_particle(self, particle_data: Dict) -> List[str]:
        """Extract human language content from a particle for spatial correlation"""
        human_content = []
        
        try:
            particle = particle_data.get("particle")
            metadata = particle_data.get("metadata", {})
            
            # Extract from particle metadata
            if metadata:
                # Content field
                if "content" in metadata:
                    content = metadata["content"]
                    if isinstance(content, str) and self._is_human_language(content):
                        human_content.extend(self._extract_meaningful_words(content))
                
                # Definitions  
                if "definitions" in metadata:
                    definitions = metadata["definitions"]
                    if isinstance(definitions, list):
                        for definition in definitions:
                            if isinstance(definition, dict):
                                def_text = definition.get("text", "") or definition.get("wikipedia", "")
                                if def_text and self._is_human_language(def_text):
                                    human_content.extend(self._extract_meaningful_words(def_text))
                            elif isinstance(definition, str) and self._is_human_language(definition):
                                human_content.extend(self._extract_meaningful_words(definition))
                
                # Context fields
                for field in ["context", "origin", "source_phrase", "trigger"]:
                    if field in metadata:
                        field_content = metadata[field]
                        if isinstance(field_content, str) and self._is_human_language(field_content):
                            human_content.extend(self._extract_meaningful_words(field_content))
            
            # Remove duplicates and filter
            unique_content = list(set(human_content))
            return [c for c in unique_content if len(c) > 2][:5]  # Top 5 per particle
            
        except Exception as e:
            self.log(f"Error extracting human content from particle: {e}", "DEBUG", "_extract_human_content_from_particle")
            return []
    
    def _is_human_language(self, text: str) -> bool:
        """Determine if text appears to be human language (vs compressed tokens)"""
        if not text or len(text) < 3:
            return False
        
        words = text.split()
        if len(words) == 0:
            return False
        
        # Check for human language characteristics
        avg_word_length = sum(len(w) for w in words) / len(words)
        has_common_words = any(w.lower() in {'the', 'and', 'for', 'are', 'but', 'not', 'with', 'have', 'this', 'that'} for w in words)
        has_proper_structure = len(words) > 1 or avg_word_length > 4
        
        # Exclude obviously compressed tokens
        is_likely_compressed = (len(words) == 1 and 
                               len(text) <= 6 and 
                               sum(1 for c in text.lower() if c in 'aeiou') < len(text) / 3)
        
        return has_proper_structure and (has_common_words or avg_word_length > 5) and not is_likely_compressed
    
    async def _generate_hybrid_translation_candidates(self, token: str, correlation_data: Dict):
        """Generate translation candidates using hybrid field-position + content approach"""
        try:
            hybrid_candidates = {}
            
            # Spatial-based candidates (60% weight)
            spatial_correlations = correlation_data.get("spatial_correlations", {})
            spatial_content = spatial_correlations.get("spatial_human_content", [])
            
            for candidate in spatial_content:
                confidence = 0.6 * self._calculate_spatial_confidence(token, candidate, spatial_correlations)
                if confidence > 0.1:  # Minimum threshold
                    hybrid_candidates[candidate] = hybrid_candidates.get(candidate, 0.0) + confidence
            
            # Content-based candidates (40% weight)
            trigger_contexts = correlation_data.get("trigger_contexts", [])
            
            # Extract human words from trigger contexts
            for trigger_ctx in trigger_contexts:
                trigger_words = self._extract_meaningful_words(trigger_ctx)
                for candidate in trigger_words:
                    confidence = 0.4 * self._calculate_contextual_confidence(token, candidate, trigger_ctx)
                    if confidence > 0.1:
                        hybrid_candidates[candidate] = hybrid_candidates.get(candidate, 0.0) + confidence
            
            # Store hybrid confidence scores
            if "hybrid_confidence_scores" not in correlation_data:
                correlation_data["hybrid_confidence_scores"] = {}
            correlation_data["hybrid_confidence_scores"].update(hybrid_candidates)
            
            # Update translation candidates
            if token not in self.translation_candidates:
                self.translation_candidates[token] = []
            
            # Add new candidates with sufficient confidence
            for candidate, confidence in hybrid_candidates.items():
                if confidence > 0.3 and candidate not in self.translation_candidates[token]:
                    self.translation_candidates[token].append(candidate)
                    self.confidence_scores[(token, candidate)] = confidence
            
            self.log(f"Generated {len(hybrid_candidates)} hybrid translation candidates for '{token}': {list(hybrid_candidates.keys())[:5]}", "DEBUG", "_generate_hybrid_translation_candidates")
            
        except Exception as e:
            self.log(f"Error generating hybrid translation candidates for '{token}': {e}", "ERROR", "_generate_hybrid_translation_candidates")
    
    def _calculate_spatial_confidence(self, token: str, candidate: str, spatial_data: Dict) -> float:
        """Calculate confidence for spatial-based translation"""
        base_confidence = 0.5
        
        # Factor 1: Semantic density (more nearby content = higher confidence)
        density = spatial_data.get("neighborhood_semantic_density", 0)
        density_factor = min(density / 10.0, 1.0) * 0.4
        
        # Factor 2: Candidate relevance (longer words often more meaningful)
        relevance_factor = min(len(candidate) / 10.0, 1.0) * 0.3
        
        # Factor 3: Nearby particle count
        nearby_count = spatial_data.get("nearby_particle_count", 0)
        count_factor = min(nearby_count / 5.0, 1.0) * 0.3
        
        return base_confidence + density_factor + relevance_factor + count_factor
    
    def _calculate_contextual_confidence(self, token: str, candidate: str, context: str) -> float:
        """Calculate confidence for context-based translation"""
        base_confidence = 0.3
        
        # Factor 1: Context relevance (how meaningful is the context)
        context_words = len(context.split())
        context_factor = min(context_words / 15.0, 1.0) * 0.4
        
        # Factor 2: Candidate position in context (earlier = more relevant)
        words = context.lower().split()
        if candidate.lower() in words:
            position = words.index(candidate.lower())
            position_factor = max(0.0, 1.0 - (position / len(words))) * 0.3
        else:
            position_factor = 0.0
        
        # Factor 3: Token-candidate affinity (similar length patterns)
        length_ratio = min(len(token), len(candidate)) / max(len(token), len(candidate))
        affinity_factor = length_ratio * 0.3
        
        return base_confidence + context_factor + position_factor + affinity_factor

    def _update_context_correlation(self, token: str, context: Dict):
        """Enhanced context correlation with hybrid mapping support"""
        if token not in self.context_correlations:
            self.context_correlations[token] = {
                "definitions": [],
                "human_phrases": [],
                "nearby_human_phrases": [],
                "questions": [],
                "trigger_contexts": [],  # New: separate trigger contexts from responses
                "response_instances": [],  # New: track Iris's response instances
                "spatial_correlations": {},  # New: field-position based correlations
                "hybrid_confidence_scores": {}  # New: hybrid translation confidence
            }
        
        correlation_data = self.context_correlations[token]
        
        # Enhanced context processing for hybrid mapping
        if context:
            # Handle new separated context format
            if "trigger_context" in context and "response_content" in context:
                # Store trigger context separately from response content
                trigger_ctx = context["trigger_context"]
                if trigger_ctx and trigger_ctx not in correlation_data["trigger_contexts"]:
                    correlation_data["trigger_contexts"].append(trigger_ctx)
                
                # Track response instances (Iris's actual language)
                response_content = context["response_content"]
                if response_content and response_content not in correlation_data["response_instances"]:
                    correlation_data["response_instances"].append(response_content)
                
                self.log(f"Updated enhanced context for '{token}': trigger='{trigger_ctx[:30]}...', response='{response_content[:20]}...'", "DEBUG", "_update_context_correlation")
            
            # Legacy format support
            elif "human_phrase" in context:
                human_phrase = context["human_phrase"]
                if human_phrase and human_phrase not in correlation_data["human_phrases"]:
                    correlation_data["human_phrases"].append(human_phrase)
            
            # Definitions and other context data
            if "definitions" in context and context["definitions"]:
                for definition in context["definitions"]:
                    if definition and definition not in correlation_data["definitions"]:
                        correlation_data["definitions"].append(definition)
            
            # Field position data for spatial analysis
            if "field_position" in context:
                field_pos = context["field_position"]
                if isinstance(field_pos, dict):
                    # Trigger spatial field analysis for hybrid mapping
                    asyncio.create_task(self._analyze_spatial_neighborhood(token, field_pos, context))
        
        # Keep only recent context (last 20 entries) for each category
        for key in ["trigger_contexts", "response_instances", "human_phrases", "definitions", "questions"]:
            if key in correlation_data and len(correlation_data[key]) > 20:
                correlation_data[key] = correlation_data[key][-20:]

    def _original_update_context_correlation(self, token: str, context: Dict):
        """Legacy context correlation method for backward compatibility"""
        if token not in self.context_correlations:
            self.context_correlations[token] = {
                "nearby_human_phrases": [],
                "source_questions": [],
                "usage_contexts": []
            }
            
        # Store relevant context
        if "human_phrase" in context:
            self.context_correlations[token]["nearby_human_phrases"].append(context["human_phrase"])
            
        if "source_question" in context:
            self.context_correlations[token]["source_questions"].append(context["source_question"])
            
        if "usage_context" in context:
            self.context_correlations[token]["usage_contexts"].append(context["usage_context"])
            
        # Keep only recent context (last 20 entries)
        for key in self.context_correlations[token]:
            if len(self.context_correlations[token][key]) > 20:
                self.context_correlations[token][key] = self.context_correlations[token][key][-20:]
    
    def get_translation_suggestions(self, compressed_token: str) -> List[Tuple[str, float]]:
        """Get translation suggestions for a compressed token"""
        suggestions = []
        
        candidates = self.translation_candidates.get(compressed_token, [])
        for candidate in candidates:
            confidence = self.confidence_scores.get((compressed_token, candidate), 0.0)
            suggestions.append((candidate, confidence))
            
        # Sort by confidence
        return sorted(suggestions, key=lambda x: x[1], reverse=True)
    
    def get_all_translation_mappings(self) -> List[Dict]:
        """Get all translation mappings with metadata for dashboard display"""
        mappings = []
        
        for compressed_token, candidates in self.translation_candidates.items():
            for candidate in candidates:
                confidence = self.confidence_scores.get((compressed_token, candidate), 0.0)
                
                # Get additional metadata
                frequency_data = {}
                observations = self.compressed_token_cache.get(compressed_token, [])
                if observations:
                    frequency_data = {
                        "observation_count": len(observations),
                        "recent_activity": len([t for t in observations if t > (datetime.now().timestamp() - 300)])  # Last 5 minutes
                    }
                
                mappings.append({
                    "compressed_token": compressed_token,
                    "human_candidate": candidate,
                    "confidence": confidence,
                    "confidence_category": self._get_confidence_category(confidence),
                    "frequency_data": frequency_data,
                    "last_seen": max(observations) if observations else 0
                })
        
        # Sort by confidence (highest first)
        return sorted(mappings, key=lambda x: x["confidence"], reverse=True)
    
    def _get_confidence_category(self, confidence: float) -> str:
        """Categorize confidence score for display"""
        if confidence >= 0.8:
            return "High"
        elif confidence >= 0.6:
            return "Medium"
        elif confidence >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def get_processor_stats(self) -> Dict:
        """Get statistics about the background processor"""
        # Calculate additional stats
        total_observations = sum(len(timestamps) for timestamps in self.compressed_token_cache.values())
        high_confidence_mappings = sum(1 for confidence in self.confidence_scores.values() if confidence > 0.7)
        
        return {
            "tokens_tracked": len(self.compressed_token_cache),
            "total_observations": total_observations,
            "queue_size": len(self.analysis_queue),
            "translation_mappings": len(self.translation_candidates),
            "high_confidence_mappings": high_confidence_mappings,
            "context_correlations": len(self.context_correlations),
            "last_analysis": self.last_analysis_time,
            "is_processing": self.is_processing,
            "background_running": self.background_task is not None,
            "avg_confidence": (sum(self.confidence_scores.values()) / len(self.confidence_scores)) if self.confidence_scores else 0.0
        }
    
    def log(self, message: str, level: str = "INFO", context: str = "SemanticGravityBackgroundProcessor"):
        """Log with consistent formatting"""
        if self.logger:
            self.logger.log(message, level, context, "SemanticGravityBackgroundProcessor")
    
    async def _save_translation_mappings_to_memory(self):
        """Save translation mappings to persistent memory"""
        try:
            if not self.agent_categorizer or not hasattr(self.agent_categorizer, 'memory'):
                self.log("Cannot save translation mappings: no memory system available", "WARNING", "_save_translation_mappings_to_memory")
                return
                
            memory = self.agent_categorizer.memory
            
            # Prepare mapping data for storage
            mapping_data = {
                "translation_candidates": dict(self.translation_candidates),
                "confidence_scores": {f"{k[0]}::{k[1]}": v for k, v in self.confidence_scores.items()},
                "context_correlations": dict(self.context_correlations),
                "last_updated": datetime.now().isoformat(),
                "version": "1.1",  # Increment version for enhanced loading
                "stats": {
                    "candidate_sets": len(self.translation_candidates),
                    "confidence_entries": len(self.confidence_scores),
                    "context_entries": len(self.context_correlations),
                    "total_candidates": sum(len(candidates) for candidates in self.translation_candidates.values())
                }
            }
            
            # Save to memory with specific key using the correct update method
            await memory.update(
                key="semantic_gravity_translation_mappings",
                value=mapping_data,
                memory_type="system",
                source="semantic_gravity_background_processor"
            )
            
            stats = mapping_data["stats"]
            self.log(f"Saved translation mappings to memory: {stats['candidate_sets']} token sets, {stats['total_candidates']} total candidates, {stats['confidence_entries']} confidence scores", "INFO", "_save_translation_mappings_to_memory")
            
        except Exception as e:
            self.log(f"Error saving translation mappings to memory: {e}", "ERROR", "_save_translation_mappings_to_memory")
    
    async def _load_translation_mappings_from_memory(self):
        """Load translation mappings from persistent memory - using direct access to avoid recursion"""
        try:
            if not self.agent_categorizer or not hasattr(self.agent_categorizer, 'memory'):
                self.log("Cannot load translation mappings: no memory system available", "WARNING", "_load_translation_mappings_from_memory")
                return
                
            memory = self.agent_categorizer.memory
            
            # Use direct memory access to avoid coordinator recursion loop
            stored_data = None
            try:
                # Try direct Qdrant access first to avoid coordinator loops
                if hasattr(memory, '_direct_query'):
                    stored_data = await memory._direct_query("semantic_gravity_translation_mappings")
                elif hasattr(memory, 'memories') and hasattr(memory.memories, 'get'):
                    # Fallback to direct collection access
                    stored_data = await memory.memories.get("semantic_gravity_translation_mappings")
                else:
                    # Last resort - skip loading to avoid recursion
                    self.log("Skipping translation mapping load to prevent recursion loop", "WARNING", "_load_translation_mappings_from_memory")
                    return
            except Exception as direct_error:
                self.log(f"Direct memory access failed: {direct_error} - skipping translation loading", "WARNING", "_load_translation_mappings_from_memory")
                return
            
            if stored_data and isinstance(stored_data, dict):
                initial_candidates_count = len(self.translation_candidates)
                initial_confidence_count = len(self.confidence_scores)
                initial_context_count = len(self.context_correlations)
                
                # Restore translation candidates
                if "translation_candidates" in stored_data and isinstance(stored_data["translation_candidates"], dict):
                    loaded_candidates = stored_data["translation_candidates"]
                    self.translation_candidates.update(loaded_candidates)
                    self.log(f"Loaded {len(loaded_candidates)} translation candidate sets", "DEBUG", "_load_translation_mappings_from_memory")
                
                # Restore confidence scores (convert back from string keys)
                if "confidence_scores" in stored_data and isinstance(stored_data["confidence_scores"], dict):
                    loaded_scores = 0
                    for key_str, confidence in stored_data["confidence_scores"].items():
                        if "::" in key_str and isinstance(confidence, (int, float)):
                            try:
                                compressed_token, human_candidate = key_str.split("::", 1)
                                self.confidence_scores[(compressed_token, human_candidate)] = float(confidence)
                                loaded_scores += 1
                            except (ValueError, TypeError) as e:
                                self.log(f"Skipping invalid confidence score entry {key_str}: {e}", "WARNING", "_load_translation_mappings_from_memory")
                    self.log(f"Loaded {loaded_scores} confidence scores", "DEBUG", "_load_translation_mappings_from_memory")
                
                # Restore context correlations
                if "context_correlations" in stored_data and isinstance(stored_data["context_correlations"], dict):
                    loaded_correlations = stored_data["context_correlations"]
                    # Validate structure before loading
                    valid_correlations = {}
                    for token, correlation_data in loaded_correlations.items():
                        if isinstance(correlation_data, dict):
                            # Ensure required keys exist with proper types
                            valid_correlation = {
                                "nearby_human_phrases": correlation_data.get("nearby_human_phrases", []),
                                "source_questions": correlation_data.get("source_questions", []),
                                "usage_contexts": correlation_data.get("usage_contexts", [])
                            }
                            # Ensure all values are lists
                            for key, value in valid_correlation.items():
                                if not isinstance(value, list):
                                    valid_correlation[key] = []
                            valid_correlations[token] = valid_correlation
                    
                    self.context_correlations.update(valid_correlations)
                    self.log(f"Loaded {len(valid_correlations)} context correlations", "DEBUG", "_load_translation_mappings_from_memory")
                
                # Log summary
                candidates_loaded = len(self.translation_candidates) - initial_candidates_count
                confidence_loaded = len(self.confidence_scores) - initial_confidence_count
                context_loaded = len(self.context_correlations) - initial_context_count
                
                last_updated = stored_data.get("last_updated", "unknown")
                version = stored_data.get("version", "unknown")
                
                self.log(f"Successfully loaded translation mappings from memory: {candidates_loaded} candidates, {confidence_loaded} confidence scores, {context_loaded} context correlations (version: {version}, last updated: {last_updated})", "INFO", "_load_translation_mappings_from_memory")
                
                # Trigger immediate processing if we have loaded data
                if candidates_loaded > 0:
                    self.log("Loaded mappings detected, scheduling immediate analysis", "INFO", "_load_translation_mappings_from_memory")
                    # Add loaded tokens to analysis queue
                    for token in self.translation_candidates.keys():
                        if self._is_compressed_token(token):
                            self.analysis_queue.add(token)
                
            else:
                self.log("No existing translation mappings found in memory - starting with fresh state", "INFO", "_load_translation_mappings_from_memory")
                
        except Exception as e:
            self.log(f"Error loading translation mappings from memory: {e}", "ERROR", "_load_translation_mappings_from_memory")
            # Don't fail completely - just continue with empty state
            self.log("Continuing with empty translation mapping state due to loading error", "WARNING", "_load_translation_mappings_from_memory")
    
    async def save_current_state(self):
        """Manually save current translation state (for use with core.py manual save)"""
        await self._save_translation_mappings_to_memory()
        
        stats = self.get_processor_stats()
        self.log(f"Manual save completed - {stats['translation_mappings']} mappings, {stats['tokens_tracked']} tokens tracked", "INFO", "save_current_state")
        
        return {
            "mappings_saved": stats['translation_mappings'],
            "tokens_tracked": stats['tokens_tracked'],
            "confidence_scores_saved": len(self.confidence_scores),
            "context_correlations_saved": len(self.context_correlations)
        }
    
    def get_persistence_status(self):
        """Get status of translation mapping persistence"""
        return {
            "loaded_candidates": len(self.translation_candidates),
            "loaded_confidence_scores": len(self.confidence_scores),
            "loaded_context_correlations": len(self.context_correlations),
            "total_candidate_mappings": sum(len(candidates) for candidates in self.translation_candidates.values()),
            "has_persistent_data": len(self.translation_candidates) > 0 or len(self.confidence_scores) > 0,
            "memory_available": self.agent_categorizer is not None and hasattr(self.agent_categorizer, 'memory'),
            "last_analysis_time": self.last_analysis_time,
            "processing_active": self.background_task is not None and not self.background_task.done()
        }