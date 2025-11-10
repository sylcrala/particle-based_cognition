"""
Particle-based Cognition Engine - Gravitational Categorization Integration
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
Autonomous category generation via semantic gravity

This module bridges the semantic gravity analyzer with the agent categorization system,
enabling autonomous category generation based on gravitational clustering patterns.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from apis.api_registry import api


class GravitationalCategoryGenerator:
    """Generate categories autonomously based on semantic gravity clustering"""
    
    def __init__(self, gravity_analyzer=None, agent_categorizer=None, memory=None):
        self.gravity_analyzer = gravity_analyzer
        self.agent_categorizer = agent_categorizer
        self.memory = memory
        self.logger = api.get_api("logger")
        
        # Generation parameters
        self.min_cluster_size = 2
        self.gravity_threshold = 0.6  # Lower than analyzer default for more sensitivity
        self.thermal_category_boost = 1.3
        self.auto_generation_enabled = True
        
        # Tracking
        self.generated_categories = {}
        self.gravity_category_mappings = {}
        self.generation_history = []
        
    async def analyze_and_generate_categories(self, compressed_speech: str, session_context: Dict = None) -> Dict:
        """Analyze compressed speech and automatically generate categories"""
        try:
            if not self.gravity_analyzer or not self.agent_categorizer:
                self.log("Missing required components for gravitational categorization", "WARNING", "analyze_and_generate_categories")
                return {}
                
            # Analyze compressed speech through gravity lens
            tokens = compressed_speech.split()
            gravity_analysis = self.gravity_analyzer.analyze_compressed_speech(tokens)
            
            generation_results = {
                "gravity_analysis": gravity_analysis,
                "generated_categories": [],
                "category_mappings": {},
                "thermal_correlations": {},
                "autonomous_actions": []
            }
            
            # Extract gravitational clusters for category generation
            clusters = gravity_analysis.get("gravitational_clusters", {})
            
            for cluster_name, cluster_tokens in clusters.items():
                if len(cluster_tokens) >= self.min_cluster_size:
                    # Generate category from gravitational cluster
                    category_result = await self._generate_category_from_cluster(
                        cluster_name, cluster_tokens, gravity_analysis
                    )
                    
                    if category_result:
                        generation_results["generated_categories"].append(category_result)
                        generation_results["category_mappings"][cluster_name] = category_result["category_id"]
                        
            # Analyze thermal-category correlations
            thermal_correlations = await self._analyze_thermal_category_correlations(
                tokens, gravity_analysis
            )
            generation_results["thermal_correlations"] = thermal_correlations
            
            # Log autonomous actions
            generation_results["autonomous_actions"] = await self._perform_autonomous_optimizations(
                generation_results
            )
            
            self.log(f"Gravitational category generation completed: {len(generation_results['generated_categories'])} categories created", "INFO", "analyze_and_generate_categories")
            return generation_results
            
        except Exception as e:
            self.log(f"Error in gravitational category generation: {e}", "ERROR", "analyze_and_generate_categories")
            return {}
    
    async def _generate_category_from_cluster(self, cluster_name: str, cluster_tokens: List[str], gravity_analysis: Dict) -> Optional[Dict]:
        """Generate a category from a gravitational cluster"""
        try:
            # Create compressed category name from cluster
            compressed_category = self._create_compressed_category_name(cluster_tokens)
            
            # Calculate category strength based on gravitational properties
            density_map = gravity_analysis.get("semantic_density_map", {})
            cluster_strength = sum(density_map.get(token, 0.0) for token in cluster_tokens) / len(cluster_tokens)
            
            # Determine thermal state for category
            thermal_correlation = gravity_analysis.get("thermal_gravity_correlation", {})
            category_thermal_state = self._determine_category_thermal_state(cluster_tokens, thermal_correlation)
            
            # Request category creation from agent
            category_content = {
                "cluster_name": cluster_name,
                "cluster_tokens": cluster_tokens,
                "gravitational_strength": cluster_strength,
                "thermal_state": category_thermal_state,
                "generation_method": "gravitational_clustering",
                "timestamp": datetime.now().timestamp()
            }
            
            # Let the agent categorizer handle the actual category creation
            category_result = await self.agent_categorizer.request_categorization(
                category_content, compressed_category
            )
            
            if category_result and category_result != "pending_agent_categorization":
                # Track the generated category
                generated_category = {
                    "category_id": category_result,
                    "compressed_name": compressed_category,
                    "cluster_tokens": cluster_tokens,
                    "gravitational_strength": cluster_strength,
                    "thermal_state": category_thermal_state,
                    "creation_timestamp": datetime.now().timestamp()
                }
                
                self.generated_categories[category_result] = generated_category
                self.gravity_category_mappings[cluster_name] = category_result
                
                # Log to generation history
                self.generation_history.append({
                    "action": "generate_category",
                    "cluster": cluster_name,
                    "category": category_result,
                    "strength": cluster_strength,
                    "timestamp": datetime.now().timestamp()
                })
                
                self.log(f"Generated category '{compressed_category}' from cluster '{cluster_name}' with strength {cluster_strength:.3f}", "INFO", "_generate_category_from_cluster")
                return generated_category
                
        except Exception as e:
            self.log(f"Error generating category from cluster '{cluster_name}': {e}", "ERROR", "_generate_category_from_cluster")
            
        return None
    
    def _create_compressed_category_name(self, cluster_tokens: List[str]) -> str:
        """Create a compressed category name from cluster tokens"""
        # Extract consonant patterns from tokens for compression
        compressed_parts = []
        
        for token in cluster_tokens[:3]:  # Use up to 3 tokens
            # Extract consonant backbone
            consonants = ''.join(c for c in token if c not in 'aeiou')
            if consonants:
                compressed_parts.append(consonants[:2])  # First 2 consonants
                
        # Combine and create category name
        if compressed_parts:
            base_name = ''.join(compressed_parts)
            # Limit length and add category marker
            return f"{base_name[:6]}cat"
        else:
            # Fallback naming
            return f"grav{len(cluster_tokens)}"
    
    def _determine_category_thermal_state(self, cluster_tokens: List[str], thermal_correlation: Dict) -> str:
        """Determine thermal state for category based on token analysis"""
        # Check thermal correlations
        hot_gravity = thermal_correlation.get("hot_tokens_gravity", 0.0)
        warm_gravity = thermal_correlation.get("warm_tokens_gravity", 0.0)
        
        # Classify based on gravitational strength in thermal states
        if hot_gravity > 0.7:
            return "hot"
        elif warm_gravity > 0.5 or hot_gravity > 0.3:
            return "warm"
        elif len(cluster_tokens) >= 3:  # Large clusters start warm
            return "warm"
        else:
            return "cool"
    
    async def _analyze_thermal_category_correlations(self, tokens: List[str], gravity_analysis: Dict) -> Dict:
        """Analyze correlations between thermal states and category potential"""
        try:
            thermal_correlation = gravity_analysis.get("thermal_gravity_correlation", {})
            compression_mechanisms = gravity_analysis.get("compression_mechanisms", {})
            
            correlations = {
                "thermal_reinforcement_categories": len(compression_mechanisms.get("thermal_reinforcement", [])),
                "gravitational_merger_categories": len(compression_mechanisms.get("gravitational_merger", [])),
                "hot_token_category_potential": thermal_correlation.get("hot_tokens_gravity", 0.0),
                "warm_token_category_potential": thermal_correlation.get("warm_tokens_gravity", 0.0),
                "optimization_suggestions": []
            }
            
            # Generate optimization suggestions
            if correlations["hot_token_category_potential"] > 0.8:
                correlations["optimization_suggestions"].append("increase_hot_token_categorization")
            
            if correlations["gravitational_merger_categories"] > 3:
                correlations["optimization_suggestions"].append("create_merger_super_categories")
                
            return correlations
            
        except Exception as e:
            self.log(f"Error analyzing thermal-category correlations: {e}", "ERROR", "_analyze_thermal_category_correlations")
            return {}
    
    async def _perform_autonomous_optimizations(self, generation_results: Dict) -> List[str]:
        """Perform autonomous optimizations based on analysis results"""
        actions = []
        
        try:
            if not self.auto_generation_enabled:
                return actions
                
            thermal_correlations = generation_results.get("thermal_correlations", {})
            suggestions = thermal_correlations.get("optimization_suggestions", [])
            
            for suggestion in suggestions:
                if suggestion == "increase_hot_token_categorization":
                    # Lower gravity threshold for hot tokens
                    self.gravity_threshold *= 0.9
                    actions.append(f"lowered_gravity_threshold_to_{self.gravity_threshold:.3f}")
                    
                elif suggestion == "create_merger_super_categories":
                    # Attempt to create super-categories from related categories
                    await self._create_super_categories(generation_results)
                    actions.append("created_super_categories")
                    
            # Store optimizations
            if actions and self.memory:
                await self.memory.update(
                    key=f"gravitational_optimizations_{int(datetime.now().timestamp())}",
                    value={
                        "actions": actions,
                        "generation_results": generation_results,
                        "timestamp": datetime.now().timestamp()
                    },
                    source="GravitationalCategoryGenerator",
                    tags=["gravitational", "optimization", "autonomous"],
                    memory_type="system"
                )
                
        except Exception as e:
            self.log(f"Error in autonomous optimizations: {e}", "ERROR", "_perform_autonomous_optimizations")
            
        return actions
    
    async def _create_super_categories(self, generation_results: Dict):
        """Create super-categories from related generated categories"""
        try:
            generated_categories = generation_results.get("generated_categories", [])
            
            if len(generated_categories) >= 3:
                # Group categories by thermal state
                thermal_groups = defaultdict(list)
                for category in generated_categories:
                    thermal_state = category.get("thermal_state", "cool")
                    thermal_groups[thermal_state].append(category)
                
                # Create super-categories for groups with multiple categories
                for thermal_state, categories in thermal_groups.items():
                    if len(categories) >= 2:
                        super_category_name = f"{thermal_state}grp"
                        
                        super_category_content = {
                            "type": "super_category",
                            "thermal_state": thermal_state,
                            "child_categories": [cat["category_id"] for cat in categories],
                            "creation_method": "gravitational_super_grouping"
                        }
                        
                        await self.agent_categorizer.request_categorization(
                            super_category_content, super_category_name
                        )
                        
                        self.log(f"Created super-category '{super_category_name}' with {len(categories)} child categories", "INFO", "_create_super_categories")
                        
        except Exception as e:
            self.log(f"Error creating super-categories: {e}", "ERROR", "_create_super_categories")
    
    def get_generation_stats(self) -> Dict:
        """Get statistics about gravitational category generation"""
        try:
            stats = {
                "total_generated_categories": len(self.generated_categories),
                "gravity_threshold": self.gravity_threshold,
                "generation_history_count": len(self.generation_history),
                "thermal_distribution": defaultdict(int),
                "avg_gravitational_strength": 0.0
            }
            
            # Calculate thermal distribution and average strength
            total_strength = 0.0
            for category in self.generated_categories.values():
                thermal_state = category.get("thermal_state", "cool")
                stats["thermal_distribution"][thermal_state] += 1
                total_strength += category.get("gravitational_strength", 0.0)
                
            if self.generated_categories:
                stats["avg_gravitational_strength"] = total_strength / len(self.generated_categories)
                
            stats["thermal_distribution"] = dict(stats["thermal_distribution"])
            return stats
            
        except Exception as e:
            self.log(f"Error getting generation stats: {e}", "ERROR", "get_generation_stats")
            return {}
    
    def log(self, message: str, level: str = "INFO", context: str = "GravitationalCategoryGenerator"):
        """Log with consistent formatting"""
        if self.logger:
            self.logger.log(message, level, context, "GravitationalCategoryGenerator")


class ThermalCategoryOptimizer:
    """Optimize category organization based on thermal state changes"""
    
    def __init__(self, agent_categorizer=None, memory=None, field=None):
        self.agent_categorizer = agent_categorizer
        self.memory = memory
        self.field = field
        self.logger = api.get_api("logger")
        
        # Optimization parameters
        self.thermal_reorganization_threshold = 0.3  # When to trigger reorganization
        self.optimization_interval = 3600  # 1 hour in seconds
        self.last_optimization = 0
        
    async def optimize_thermal_categories(self) -> Dict:
        """Optimize category organization based on thermal state changes"""
        try:
            current_time = datetime.now().timestamp()
            
            if current_time - self.last_optimization < self.optimization_interval:
                return {"status": "optimization_cooldown"}
                
            if not self.agent_categorizer:
                return {"status": "no_categorizer"}
                
            # Get current category thermal states
            category_stats = self.agent_categorizer.get_category_stats()
            thermal_distribution = category_stats.get("thermal_distribution", {})
            
            optimization_results = {
                "thermal_migrations": [],
                "category_merges": [],
                "thermal_promotions": [],
                "optimization_timestamp": current_time
            }
            
            # Analyze for thermal-based optimizations
            hot_categories = thermal_distribution.get("hot", 0)
            warm_categories = thermal_distribution.get("warm", 0)
            cool_categories = thermal_distribution.get("cool", 0)
            
            # Promote highly active cool categories to warm
            if cool_categories > warm_categories * 2:
                promoted = await self._promote_active_categories()
                optimization_results["thermal_promotions"] = promoted
                
            # Merge related hot categories for efficiency
            if hot_categories > 5:
                merged = await self._merge_related_hot_categories()
                optimization_results["category_merges"] = merged
                
            self.last_optimization = current_time
            
            self.log(f"Thermal category optimization completed: {len(optimization_results['thermal_promotions'])} promotions, {len(optimization_results['category_merges'])} merges", "INFO", "optimize_thermal_categories")
            return optimization_results
            
        except Exception as e:
            self.log(f"Error in thermal category optimization: {e}", "ERROR", "optimize_thermal_categories")
            return {"status": "error", "error": str(e)}
    
    async def _promote_active_categories(self) -> List[str]:
        """Promote active cool categories to warm"""
        promoted = []
        
        try:
            # Get categories with high usage but cool thermal state
            # This would require access to category usage statistics
            # Implementation depends on agent_categorizer interface
            
            # Placeholder for actual promotion logic
            self.log("Thermal promotion analysis completed", "DEBUG", "_promote_active_categories")
            
        except Exception as e:
            self.log(f"Error promoting active categories: {e}", "ERROR", "_promote_active_categories")
            
        return promoted
    
    async def _merge_related_hot_categories(self) -> List[str]:
        """Merge related hot categories to prevent thermal fragmentation"""
        merged = []
        
        try:
            # Analyze hot categories for merge opportunities
            # This would use semantic similarity between categories
            
            # Placeholder for actual merge logic
            self.log("Hot category merge analysis completed", "DEBUG", "_merge_related_hot_categories")
            
        except Exception as e:
            self.log(f"Error merging hot categories: {e}", "ERROR", "_merge_related_hot_categories")
            
        return merged
    
    def log(self, message: str, level: str = "INFO", context: str = "ThermalCategoryOptimizer"):
        """Log with consistent formatting"""
        if self.logger:
            self.logger.log(message, level, context, "ThermalCategoryOptimizer")