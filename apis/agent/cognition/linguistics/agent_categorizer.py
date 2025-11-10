"""
Particle-based Cognition Engine - Self-Determinable Categorization System
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
Allow the agent to create and manage their own categories

This module enables the agent to create, modify, and organize categories using their compressed language,
creating a bridge between their native communication and human-readable organization.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Tuple
from collections import defaultdict, Counter

from apis.api_registry import api


class AgentCategorizer:
    """Agent's self-determinable categorization system"""
    
    def __init__(self, memory=None, field=None, gravity_analyzer=None):
        self.memory = memory
        self.field = field
        self.gravity_analyzer = gravity_analyzer
        self.logger = api.get_api("logger")
        
        # Category management
        self.agent_categories = {}  # Agent's native categories in compressed language
        self.category_translations = {}  # Compressed → Human readable mappings
        self.category_hierarchy = {}  # Parent-child relationships
        self.category_usage_stats = defaultdict(int)
        
        # Dynamic categorization
        self.auto_categorization_rules = {}
        self.category_evolution_log = []
        
        # Background semantic gravity processor
        self.background_processor = None
        
    async def initialize_background_processor(self):
        """Initialize and start the background semantic gravity processor"""
        try:
            self.log("Attempting to initialize background processor...", "DEBUG", "initialize_background_processor")
            from .semantic_gravity_background_processor import SemanticGravityBackgroundProcessor
            
            self.background_processor = SemanticGravityBackgroundProcessor(
                gravity_analyzer=self.gravity_analyzer,
                agent_categorizer=self,
                memory=self.memory,
                field=self.field
            )
            
            self.log("Background processor object created, starting processing...", "DEBUG", "initialize_background_processor")
            await self.background_processor.start_background_processing()
            self.log("Background semantic gravity processor initialized and started", "INFO", "initialize_background_processor")
            
        except Exception as e:
            import traceback
            self.log(f"Error initializing background processor: {e}", "ERROR", "initialize_background_processor")
            self.log(f"Background processor traceback: {traceback.format_exc()}", "DEBUG", "initialize_background_processor")
            
    async def observe_agent_communication(self, agent_response: str, context: Dict = None):
        """Enhanced observe agent communication with hybrid field-position + content analysis"""
        if self.background_processor:
            # Enhanced context processing to separate trigger from response content
            enhanced_context = self._enhance_context_data(context)
            
            # Apply intelligent pre-filtering to avoid JSON noise
            meaningful_tokens = self._extract_meaningful_tokens(agent_response, enhanced_context)
            
            for token in meaningful_tokens:
                await self.background_processor.observe_compressed_token(token, enhanced_context)
    
    def _enhance_context_data(self, context: Dict = None) -> Dict:
        """Enhance context data to support hybrid translation mapping"""
        if not context:
            context = {}
        
        enhanced_context = context.copy()
        
        # Separate content types for better analysis
        if "trigger_context" in context and "response_content" in context:
            # New format: separated trigger and response
            enhanced_context["human_language_context"] = context["trigger_context"]
            enhanced_context["iris_language_content"] = context["response_content"]
            enhanced_context["content_separation"] = True
            
            self.log(f"Enhanced context: trigger='{context['trigger_context'][:30]}...', response='{context['response_content'][:30]}...'", "DEBUG", "_enhance_context_data")
        
        elif "human_phrase" in context:
            # Legacy format: assume human_phrase is the trigger context
            enhanced_context["human_language_context"] = context["human_phrase"]
            enhanced_context["content_separation"] = False
            
            self.log(f"Legacy context format detected: '{context['human_phrase'][:30]}...'", "DEBUG", "_enhance_context_data")
        
        return enhanced_context
    
    def _extract_meaningful_tokens(self, agent_response: str, context: Dict = None) -> List[str]:
        """Extract only meaningful tokens from agent response, filtering out JSON structure noise"""
        # Skip processing if this looks like JSON data output
        if self._is_json_structure(agent_response):
            return self._extract_particle_metadata_tokens(agent_response, context)
        
        # For regular text, use standard meaningful token extraction
        return self._extract_semantic_tokens(agent_response)
    
    def _is_json_structure(self, text: str) -> bool:
        """Detect if text is primarily JSON structure that should be filtered differently"""
        # Quick heuristics for JSON content
        json_indicators = text.count('{') + text.count('[') + text.count('"')
        punctuation_ratio = (text.count(',') + text.count(':') + text.count('{') + text.count('}')) / max(len(text), 1)
        
        return json_indicators > 5 or punctuation_ratio > 0.15
    
    def _extract_particle_metadata_tokens(self, json_text: str, context: Dict = None) -> List[str]:
        """Extract only meaningful tokens from particle metadata, not JSON structure"""
        meaningful_tokens = []
        
        try:
            import re
            
            # Extract particle type names
            particle_types = re.findall(r'"type":\s*"([^"]+)"', json_text)
            meaningful_tokens.extend([t for t in particle_types if self._is_meaningful_particle_token(t)])
            
            # Extract meaningful metadata values (not keys or structure)
            # Look for content fields in particle metadata
            content_patterns = [
                r'"content":\s*"([^"]{3,20})"',  # Content field
                r'"compressed":\s*"([^"]{2,15})"',  # Compressed language
                r'"definition":\s*"([^"]{3,30})"',  # Definitions
                r'"token":\s*"([^"]{2,15})"',  # Tokens
                r'"semantic":\s*"([^"]{3,20})"'  # Semantic content
            ]
            
            for pattern in content_patterns:
                matches = re.findall(pattern, json_text)
                for match in matches:
                    if self._is_meaningful_particle_token(match):
                        meaningful_tokens.append(match)
            
            # Log filtering results to track effectiveness
            if meaningful_tokens:
                self.log(f"Extracted {len(meaningful_tokens)} meaningful tokens from particle metadata: {meaningful_tokens[:5]}", "DEBUG", "_extract_particle_metadata_tokens")
            
        except Exception as e:
            self.log(f"Error extracting particle metadata tokens: {e}", "DEBUG", "_extract_particle_metadata_tokens")
        
        return meaningful_tokens[:10]  # Limit to prevent overload
    
    def _extract_semantic_tokens(self, text: str) -> List[str]:
        """Extract semantically meaningful tokens from regular text"""
        # Split and filter tokens
        words = text.split()
        meaningful_tokens = []
        
        # Only skip truly meaningless structure words - be much more permissive
        skip_tokens = {
            'the', 'and', 'or', 'but', 'a', 'an', 'is', 'are', 'was', 'were',
            'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with', 'from'
        }
        
        for word in words:
            cleaned_word = word.strip('.,!?;:"()[]{}').lower()
            
            if (len(cleaned_word) > 1 
                and len(cleaned_word) <= 20
                and cleaned_word not in skip_tokens
                and cleaned_word.isalpha()  # Only alphabetic characters
                and not cleaned_word.startswith('http')):  # Skip URLs
                
                meaningful_tokens.append(cleaned_word)
        
        return meaningful_tokens[:20]  # Increased limit for more comprehensive analysis
    
    def _is_meaningful_particle_token(self, token: str) -> bool:
        """Check if a token from particle metadata is worth processing"""
        if not token or len(token) < 2 or len(token) > 20:
            return False
        
        # Only skip obvious system noise - allow cognitive concepts through
        particle_system_words = {
            'spawn', 'update', 'remove', 'id', 'type', 'source', 'timestamp'
        }
        
        # Skip JSON structure tokens
        json_structure = {'{', '}', '[', ']', ':', ',', '"', "'", '\\', 'true', 'false', 'null'}
        
        token_lower = token.lower()
        
        return (token_lower not in particle_system_words 
                and token not in json_structure
                and token.isalpha()  # Only alphabetic characters
                and len([c for c in token_lower if c in 'aeiou']) > 0)  # Must have at least one vowel
                    
    def get_translation_suggestions(self, compressed_token: str) -> List[Tuple[str, float]]:
        """Get translation suggestions for a compressed token"""
        if self.background_processor:
            return self.background_processor.get_translation_suggestions(compressed_token)
        return []
        
    def get_background_processor_stats(self) -> Dict:
        """Get background processor statistics"""
        if self.background_processor:
            return self.background_processor.get_processor_stats()
        return {"status": "not_initialized"}
    
    def get_all_translation_mappings(self) -> List[Dict]:
        """Get all translation mappings with metadata for dashboard display"""
        if self.background_processor:
            return self.background_processor.get_all_translation_mappings()
        return []
    
    def get_persistence_status(self) -> Dict:
        """Get persistence status of translation mappings"""
        if self.background_processor:
            return self.background_processor.get_persistence_status()
        return {"status": "background_processor_not_initialized", "has_persistent_data": False}
        
    async def request_categorization(self, content: Any, agent_suggestion: str = None) -> str:
        """Request agent to categorize content, optionally with their suggestion"""
        try:
            # If agent provided a suggestion in compressed language, use it
            if agent_suggestion:
                category = await self._process_agent_category_suggestion(agent_suggestion, content)
                if category:
                    return category
            
            # Generate categorization request for the agent
            categorization_request = await self._generate_categorization_request(content)
            
            # Store request for agent to process
            if self.memory:
                await self.memory.update(
                    key=f"categorization_request_{uuid.uuid4().hex[:8]}",
                    value={
                        "content": content,
                        "request": categorization_request,
                        "timestamp": datetime.now().timestamp(),
                        "status": "pending_agent_response",
                        "compressed_suggestion": agent_suggestion
                    },
                    source="AgentCategorizer",
                    tags=["categorization", "agent_request", "pending"],
                    memory_type="system"
                )
            
            self.log(f"Categorization request generated for agent: {categorization_request}", "INFO", "request_categorization")
            return "pending_agent_categorization"
            
        except Exception as e:
            self.log(f"Error in categorization request: {e}", "ERROR", "request_categorization")
            return "categorization_error"
    
    async def _process_agent_category_suggestion(self, agent_category: str, content: Any) -> Optional[str]:
        """Process agent's category suggestion in compressed language"""
        try:
            # Check if this is a new category or existing one
            if agent_category in self.agent_categories:
                # Existing category - add content to it
                await self._add_to_existing_category(agent_category, content)
                self.category_usage_stats[agent_category] += 1
                return agent_category
            else:
                # New category - create it
                await self._create_new_agent_category(agent_category, content)
                return agent_category
                
        except Exception as e:
            self.log(f"Error processing agent category suggestion: {e}", "ERROR", "_process_agent_category_suggestion")
            return None
    
    async def _create_new_agent_category(self, agent_category: str, initial_content: Any):
        """Create a new category in agent's compressed language"""
        try:
            category_id = f"agent_cat_{uuid.uuid4().hex[:8]}"
            
            new_category = {
                "agent_name": agent_category,
                "category_id": category_id,
                "created_timestamp": datetime.now().timestamp(),
                "contents": [initial_content],
                "semantic_markers": self._extract_semantic_markers(initial_content),
                "usage_count": 1,
                "evolution_history": [],
                "parent_category": None,
                "child_categories": [],
                "thermal_state": "warm"  # New categories start warm
            }
            
            self.agent_categories[agent_category] = new_category
            
            # Attempt to generate human-readable translation
            human_translation = await self._generate_human_translation(agent_category, initial_content)
            if human_translation:
                self.category_translations[agent_category] = human_translation
            
            # Store in memory
            if self.memory:
                await self.memory.update(
                    key=f"agent_category_{agent_category}",
                    value=new_category,
                    source="AgentCategorizer",
                    tags=["agent_category", "compressed_language", "categorization"],
                    memory_type="system",
                    thermal_state="warm"
                )
            
            # Log evolution
            self.category_evolution_log.append({
                "action": "create",
                "category": agent_category,
                "timestamp": datetime.now().timestamp(),
                "content": str(initial_content)[:100]  # Truncated for logging
            })
            
            self.log(f"New agent category created: '{agent_category}' (translated: '{human_translation}')", "INFO", "_create_new_agent_category")
            
        except Exception as e:
            self.log(f"Error creating new agent category: {e}", "ERROR", "_create_new_agent_category")
    
    async def _add_to_existing_category(self, agent_category: str, content: Any):
        """Add content to existing agent category"""
        try:
            if agent_category not in self.agent_categories:
                return
                
            category = self.agent_categories[agent_category]
            category["contents"].append(content)
            category["usage_count"] += 1
            category["thermal_state"] = self._update_thermal_state(category["thermal_state"], "access")
            
            # Update semantic markers
            new_markers = self._extract_semantic_markers(content)
            category["semantic_markers"].extend(new_markers)
            category["semantic_markers"] = list(set(category["semantic_markers"]))  # Remove duplicates
            
            # Update in memory
            if self.memory:
                await self.memory.update(
                    key=f"agent_category_{agent_category}",
                    value=category,
                    source="AgentCategorizer",
                    tags=["agent_category", "compressed_language", "categorization"],
                    memory_type="system",
                    thermal_state=category["thermal_state"]
                )
            
            self.log(f"Content added to agent category '{agent_category}', usage: {category['usage_count']}", "DEBUG", "_add_to_existing_category")
            
        except Exception as e:
            self.log(f"Error adding to existing category: {e}", "ERROR", "_add_to_existing_category")
    
    def _extract_semantic_markers(self, content: Any) -> List[str]:
        """Extract semantic markers from content for categorization"""
        markers = []
        
        try:
            content_str = str(content).lower()
            
            # Extract compressed tokens if present
            compressed_tokens = []
            words = content_str.split()
            for word in words:
                # Look for compressed language patterns (short, consonant-heavy)
                if len(word) >= 2 and len(word) <= 6:
                    vowel_count = sum(1 for c in word if c in 'aeiou')
                    consonant_count = len(word) - vowel_count
                    if consonant_count > vowel_count:
                        compressed_tokens.append(word)
            
            markers.extend(compressed_tokens)
            
            # Extract other semantic indicators
            if "definition" in content_str:
                markers.append("definitional")
            if "memory" in content_str:
                markers.append("memorial")
            if "particle" in content_str:
                markers.append("particle_related")
                
        except Exception as e:
            self.log(f"Error extracting semantic markers: {e}", "WARNING", "_extract_semantic_markers")
            
        return markers[:10]  # Limit to prevent bloat
    
    async def _generate_human_translation(self, agent_category: str, sample_content: Any) -> Optional[str]:
        """Generate human-readable translation of agent's category"""
        try:
            # Analyze the compressed category name and content
            analysis = {
                "category_length": len(agent_category),
                "vowel_ratio": sum(1 for c in agent_category if c in 'aeiou') / len(agent_category) if agent_category else 0,
                "sample_content": str(sample_content)[:200]
            }
            
            # Pattern-based translation attempts
            translations = []
            
            # Short categories might be core concepts
            if len(agent_category) <= 3:
                translations.append("core_concept")
            
            # Medium length might be actions or states
            elif 3 < len(agent_category) <= 6:
                translations.append("action_or_state")
                
            # Long categories might be complex concepts
            else:
                translations.append("complex_concept")
            
            # Content-based hints
            content_str = str(sample_content).lower()
            if "definition" in content_str:
                translations.append("definitional")
            if "memory" in content_str:
                translations.append("memory_related")
            if "learn" in content_str:
                translations.append("learning_related")
                
            # Return best guess or combination
            if translations:
                return "_".join(translations[:2])  # Combine top 2 hints
            else:
                return f"agent_concept_{len(agent_category)}ch"
                
        except Exception as e:
            self.log(f"Error generating human translation: {e}", "WARNING", "_generate_human_translation")
            return f"unknown_category_{agent_category[:3]}"
    
    def _update_thermal_state(self, current_state: str, access_type: str) -> str:
        """Update thermal state based on access patterns"""
        thermal_hierarchy = ["cold", "cool", "warm", "hot"]
        
        try:
            current_index = thermal_hierarchy.index(current_state)
        except ValueError:
            current_index = 1  # Default to cool
            
        if access_type == "access":
            # Boost thermal state on access
            new_index = min(len(thermal_hierarchy) - 1, current_index + 1)
        elif access_type == "decay":
            # Reduce thermal state over time
            new_index = max(0, current_index - 1)
        else:
            new_index = current_index
            
        return thermal_hierarchy[new_index]
    
    async def _generate_categorization_request(self, content: Any) -> str:
        """Generate a request for agent to categorize content"""
        # Create a simple request that agent can understand
        content_preview = str(content)[:50].replace(" ", "")  # Compress for agent
        
        # Use pattern they're familiar with
        request = f"categorize {content_preview} need category"
        
        return request
    
    async def agent_recategorize(self, old_category: str, new_category: str, reason: str = ""):
        """Allow agent to recategorize content"""
        try:
            if old_category not in self.agent_categories:
                self.log(f"Cannot recategorize - old category '{old_category}' not found", "WARNING", "agent_recategorize")
                return False
                
            old_cat_data = self.agent_categories[old_category]
            
            # Create new category if it doesn't exist
            if new_category not in self.agent_categories:
                await self._create_new_agent_category(new_category, old_cat_data["contents"][0])
                # Remove the first content since it was used to create the category
                content_to_move = old_cat_data["contents"][1:] if len(old_cat_data["contents"]) > 1 else []
            else:
                content_to_move = old_cat_data["contents"]
            
            # Move all content to new category
            for content in content_to_move:
                await self._add_to_existing_category(new_category, content)
            
            # Log the evolution
            self.category_evolution_log.append({
                "action": "recategorize",
                "old_category": old_category,
                "new_category": new_category,
                "reason": reason,
                "timestamp": datetime.now().timestamp(),
                "content_count": len(old_cat_data["contents"])
            })
            
            # Archive old category
            old_cat_data["archived"] = True
            old_cat_data["archived_timestamp"] = datetime.now().timestamp()
            old_cat_data["archived_reason"] = f"recategorized_to_{new_category}"
            
            self.log(f"Agent recategorized '{old_category}' → '{new_category}' (reason: {reason})", "INFO", "agent_recategorize")
            return True
            
        except Exception as e:
            self.log(f"Error in agent recategorization: {e}", "ERROR", "agent_recategorize")
            return False
    
    def get_category_stats(self) -> Dict:
        """Get statistics about agent's categorization system"""
        try:
            stats = {
                "total_categories": len(self.agent_categories),
                "active_categories": len([c for c in self.agent_categories.values() if not c.get("archived", False)]),
                "total_categorized_items": sum(len(cat["contents"]) for cat in self.agent_categories.values()),
                "most_used_categories": [],
                "thermal_distribution": defaultdict(int),
                "evolution_events": len(self.category_evolution_log)
            }
            
            # Most used categories
            category_usage = [(cat["agent_name"], cat["usage_count"]) for cat in self.agent_categories.values()]
            stats["most_used_categories"] = sorted(category_usage, key=lambda x: x[1], reverse=True)[:5]
            
            # Thermal distribution
            for cat in self.agent_categories.values():
                if not cat.get("archived", False):
                    stats["thermal_distribution"][cat["thermal_state"]] += 1
                    
            return dict(stats)
            
        except Exception as e:
            self.log(f"Error getting category stats: {e}", "ERROR", "get_category_stats")
            return {}
    
    async def suggest_category_bridge(self, agent_category: str) -> Dict[str, Any]:
        """Suggest ways to bridge agent's category to human understanding"""
        try:
            if agent_category not in self.agent_categories:
                return {"error": "Category not found"}
                
            category = self.agent_categories[agent_category]
            
            bridge_suggestions = {
                "agent_category": agent_category,
                "human_translation": self.category_translations.get(agent_category, "unknown"),
                "content_analysis": {},
                "semantic_patterns": [],
                "suggested_human_categories": [],
                "confidence_score": 0.0
            }
            
            # Analyze content patterns
            all_content = [str(item) for item in category["contents"]]
            
            # Find common terms
            all_words = []
            for content in all_content:
                all_words.extend(content.lower().split())
            
            word_freq = Counter(all_words)
            bridge_suggestions["content_analysis"] = {
                "common_terms": word_freq.most_common(5),
                "content_count": len(category["contents"]),
                "avg_content_length": sum(len(str(item)) for item in category["contents"]) / len(category["contents"])
            }
            
            # Semantic patterns from markers
            bridge_suggestions["semantic_patterns"] = category["semantic_markers"]
            
            # Suggest human categories based on content
            human_suggestions = []
            for term, freq in word_freq.most_common(3):
                if len(term) > 3:  # Skip very short terms
                    human_suggestions.append(f"{term}_related")
                    
            bridge_suggestions["suggested_human_categories"] = human_suggestions
            
            # Calculate confidence based on content consistency
            confidence = min(1.0, len(category["semantic_markers"]) / 5.0 + category["usage_count"] / 10.0)
            bridge_suggestions["confidence_score"] = round(confidence, 2)
            
            return bridge_suggestions
            
        except Exception as e:
            self.log(f"Error suggesting category bridge: {e}", "ERROR", "suggest_category_bridge")
            return {"error": str(e)}
    
    def log(self, message: str, level: str = "INFO", context: str = "AgentCategorizer"):
        """Log with consistent formatting"""
        if self.logger:
            self.logger.log(message, level, context, "AgentCategorizer")


class CategoryBridge:
    """Bridge between agent's compressed categories and human-readable organization"""
    
    def __init__(self, agent_categorizer: 'AgentCategorizer'):
        self.agent_categorizer = agent_categorizer
        self.logger = api.get_api("logger")
        
        # Translation mappings
        self.translation_patterns = {}
        self.reverse_translations = {}
        self.confidence_scores = {}
        
    async def create_translation_mapping(self, agent_category: str, human_category: str, confidence: float = 0.8):
        """Create a translation mapping between agent and human categories"""
        try:
            self.translation_patterns[agent_category] = human_category
            self.reverse_translations[human_category] = agent_category
            self.confidence_scores[agent_category] = confidence
            
            # Store in memory
            if self.agent_categorizer.memory:
                await self.agent_categorizer.memory.update(
                    key=f"category_translation_{agent_category}",
                    value={
                        "agent_category": agent_category,
                        "human_category": human_category,
                        "confidence": confidence,
                        "created_timestamp": datetime.now().timestamp()
                    },
                    source="CategoryBridge",
                    tags=["translation", "category_bridge"],
                    memory_type="system"
                )
                
            self.log(f"Translation mapping created: '{agent_category}' ↔ '{human_category}' (confidence: {confidence})", "INFO", "create_translation_mapping")
            
        except Exception as e:
            self.log(f"Error creating translation mapping: {e}", "ERROR", "create_translation_mapping")
    
    def translate_to_human(self, agent_category: str) -> str:
        """Translate agent category to human-readable form"""
        return self.translation_patterns.get(agent_category, f"compressed_{agent_category}")
    
    def translate_to_agent(self, human_category: str) -> str:
        """Translate human category to agent's compressed form"""
        return self.reverse_translations.get(human_category, human_category)
    
    def get_translation_confidence(self, agent_category: str) -> float:
        """Get confidence score for translation"""
        return self.confidence_scores.get(agent_category, 0.0)
    
    def log(self, message: str, level: str = "INFO", context: str = "CategoryBridge"):
        """Log with consistent formatting"""
        if self.logger:
            self.logger.log(message, level, context, "CategoryBridge")