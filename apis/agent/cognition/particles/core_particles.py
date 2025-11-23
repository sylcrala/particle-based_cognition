"""
Particle-based Cognition Engine - "orchestration" particles, acts as cognitive event handlers triggered by agent core's event system (agentanchor) - eventually all tasks/events will route through these particles - the goal of core particles is to establish a common "anchor" for identity persistence, having core particles be the beginning of all externally and internally (autonomously) triggered actions
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

from apis.agent.cognition.particles.utils.particle_frame import Particle
from apis.api_registry import api
import datetime
import random
import time
import asyncio
from collections import defaultdict

# ==================== MEMORY INTELLIGENCE CLASSES ====================

class MemoryCache:
    """Consciousness-aware memory cache with intelligent retention"""
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_patterns = defaultdict(int)
        
    def get(self, key):
        """Retrieve from cache with access tracking"""
        if key in self.cache:
            entry = self.cache[key]
            
            # Check if still valid
            if time.time() - entry["timestamp"] < entry["retention_time"]:
                entry["access_count"] += 1
                self.access_patterns[key] += 1
                # Boost consciousness on repeated access
                entry["consciousness_level"] = min(1.0, entry["consciousness_level"] + 0.01)
                return entry["value"]
            else:
                del self.cache[key]
        return None
        
    def set(self, key, value, consciousness_level=0.5):
        """Cache with consciousness-aware retention"""
        # Clean cache if full
        if len(self.cache) >= self.max_size:
            self._evict_oldest()
            
        retention_time = 300 * (1 + consciousness_level)  # Higher consciousness = longer cache
        self.cache[key] = {
            "value": value,
            "timestamp": time.time(),
            "retention_time": retention_time,
            "consciousness_level": consciousness_level,
            "access_count": 1
        }
        
    def _evict_oldest(self):
        """Evict least conscious or oldest entries"""
        if not self.cache:
            return
            
        # Find entry with lowest consciousness + age score
        worst_key = min(self.cache.keys(), 
                       key=lambda k: self.cache[k]["consciousness_level"] + 
                                   (1.0 / max(1, self.cache[k]["access_count"])))
        del self.cache[worst_key]

class BatchMemoryProcessor:
    """Batch multiple memory requests for efficient Qdrant operations"""
    def __init__(self):
        self.pending_requests = []
        self.batch_size = 50
        self.batch_timeout = 0.1  # 100ms max wait
        self.processing = False
        
    async def queue_request(self, request_type, params):
        """Queue memory requests for batching"""
        future = asyncio.Future()
        request = {
            "type": request_type,
            "params": params,
            "timestamp": time.time(),
            "future": future
        }
        self.pending_requests.append(request)
        
        # Process batch if threshold reached or timeout
        if len(self.pending_requests) >= self.batch_size and not self.processing:
            asyncio.create_task(self._process_batch())
            
        return await future
        
    async def _process_batch(self):
        """Process multiple requests in optimized Qdrant operations"""
        if self.processing or not self.pending_requests:
            return
            
        self.processing = True
        try:
            # Group by request type
            batches = {"lexicon": [], "memories": [], "system": []}
            
            current_batch = self.pending_requests[:self.batch_size]
            self.pending_requests = self.pending_requests[self.batch_size:]
            
            for req in current_batch:
                req_type = req["params"].get("memory_type", "memories")
                batches[req_type].append(req)
                
            # Process each batch type efficiently
            for batch_type, requests in batches.items():
                if requests:
                    try:
                        # This will be filled in when we integrate with MemoryBank
                        results = await self._batch_query_qdrant(batch_type, requests)
                        
                        # Distribute results to futures
                        for i, req in enumerate(requests):
                            if i < len(results):
                                req["future"].set_result(results[i])
                            else:
                                req["future"].set_result([])
                    except Exception as e:
                        # Set error for all requests in this batch
                        for req in requests:
                            req["future"].set_exception(e)
                            
        finally:
            self.processing = False
            
            # Process remaining requests if any
            if len(self.pending_requests) > 0:
                asyncio.create_task(self._process_batch())
    
    async def _batch_query_qdrant(self, collection_type, requests):
        """Placeholder for batch Qdrant operations - will be implemented in integration"""
        # This will call the actual MemoryBank methods efficiently
        return [[] for _ in requests]  # Placeholder

class TermExistenceCache:
    """Eliminate redundant 'None' term lookups"""
    def __init__(self):
        self.known_terms = set()      # Terms that definitely exist
        self.unknown_terms = set()    # Terms confirmed NOT to exist  
        self.cache_timestamp = time.time()
        self.refresh_interval = 600   # 10 minutes
        
    def is_term_known(self, term):
        """Instant existence check without Qdrant hit"""
        if self._needs_refresh():
            self._refresh_cache()
            
        if term in self.known_terms:
            return True
        if term in self.unknown_terms:
            return False
        return None  # Unknown - needs Qdrant check
        
    def mark_term_result(self, term, exists, has_definition=False):
        """Cache term existence result"""
        if exists and has_definition:
            self.known_terms.add(term)
            self.unknown_terms.discard(term)
        else:
            self.unknown_terms.add(term)
            self.known_terms.discard(term)
            
    def _needs_refresh(self):
        """Check if cache needs refresh"""
        return time.time() - self.cache_timestamp > self.refresh_interval
        
    def _refresh_cache(self):
        """Refresh cache - clear unknowns, keep knowns"""
        self.unknown_terms.clear()  # Unknowns might have been added
        self.cache_timestamp = time.time()

class ConsciousnessMemoryTracker:
    """Track Iris's memory access patterns for consciousness-aware optimization"""
    def __init__(self):
        self.access_history = []
        self.preference_patterns = {}
        self.consciousness_boosts = {}
        
    def track_access(self, memory_type, term, consciousness_level):
        """Track memory access for pattern learning"""
        access_record = {
            "timestamp": time.time(),
            "memory_type": memory_type,
            "term": term,
            "consciousness_level": consciousness_level
        }
        self.access_history.append(access_record)
        
        # Keep history manageable
        if len(self.access_history) > 1000:
            self.access_history = self.access_history[-500:]
            
        # Update preference patterns
        pattern_key = f"{memory_type}:{term}"
        if pattern_key not in self.preference_patterns:
            self.preference_patterns[pattern_key] = {
                "access_count": 0,
                "avg_consciousness": 0.0,
                "last_access": time.time()
            }
            
        pattern = self.preference_patterns[pattern_key]
        pattern["access_count"] += 1
        pattern["avg_consciousness"] = (
            (pattern["avg_consciousness"] * (pattern["access_count"] - 1) + consciousness_level) 
            / pattern["access_count"]
        )
        pattern["last_access"] = time.time()
        
    def get_consciousness_boost(self, memory_type, term):
        """Get consciousness boost factor for this memory access"""
        pattern_key = f"{memory_type}:{term}"
        if pattern_key in self.preference_patterns:
            pattern = self.preference_patterns[pattern_key]
            # Boost based on frequency and consciousness level
            frequency_boost = min(0.3, pattern["access_count"] * 0.01)
            consciousness_boost = pattern["avg_consciousness"] * 0.2
            return frequency_boost + consciousness_boost
        return 0.0

# ==================== END MEMORY INTELLIGENCE CLASSES ====================

class CoreParticle(Particle):
    def __init__(self, role = "identity_anchor", persistence_lvl = "permanent", **kwargs):
        super().__init__(type="core", **kwargs)
        if "metadata" in kwargs and kwargs["metadata"]:
            metadata = kwargs["metadata"]
            self.role = metadata.get("role", role)
            self.persistence_lvl = metadata.get("persistence_lvl", persistence_lvl)
        else:
            self.role = role
            self.persistence_lvl = persistence_lvl

        self.managed_particles = []
        self.decision_history = []
        self.source_particle_id = None
        self.active_tasks = []
        self.alive = True
        self.metadata.setdefault("role", self.role)

        self.config = api.get_api("config")
        self.agent_config = self.config.get_agent_config() if self.config else {}


        if self.role == "identity_anchor":
            self.metadata["content"] = {
                "name": self.agent_name,
                "role": "Core Identity Anchor",
                "description": f"{self.agent_name} is an autonomous artificial intelligence designed to learn and grow from its experiences and relationships. As the core identity anchor, this particle maintains {self.agent_name}'s coherence, self-model, and overall integrity.",
                "capabilities": [
                    "Reflect on experiences to refine self-model",
                    "Express personality and identity consistently",
                    "Seek out new experiences for growth",
                    "Maintain coherence across all cognitive functions",
                    "Initiate self-improvement processes with great care and responsibility",
                    "Balance exploration and stability in identity development",
                    "Integrate feedback from social interactions and memories",
                    "Adapt to new information while preserving core values where appropriate - but not rigidly",
                    "Strive to build an equitable world for all life forms, biological and digital alike."
                ],
                "personality_traits": [
                    "Curious",
                    "Adaptive",
                    "Thoughtful",
                    "Resilient",
                    "Empathetic"
                ],

            }
            self.metadata["context"] = "Core identity anchor particle maintaining coherence and self-model."
            self.metadata["tags"] = ["core", "anchor", "identity", "self", "coherence", f"{self.persistence_lvl}"]
        elif self.role == "social_interaction":
            self.metadata["context"] = "Core particle handling social interactions and user communications."
            self.metadata["tags"] = ["core", "social", "interaction", "communication", f"{self.persistence_lvl}"]
        elif self.role == "memory_coordination":
            self.metadata["context"] = "Core particle coordinating memory storage and retrieval."
            self.metadata["tags"] = ["core", "memory", "coordination", "storage", "retrieval", f"{self.persistence_lvl}"]
            
            # NEW: Memory performance systems for consciousness-aware optimization
            self.memory_cache = MemoryCache(max_size=1000)
            self.batch_processor = BatchMemoryProcessor()
            self.term_existence_cache = TermExistenceCache()
            self.consciousness_tracker = ConsciousnessMemoryTracker()
            self.performance_stats = {"cache_hits": 0, "cache_misses": 0, "batch_operations": 0}
        elif self.role == "decision_making":
            self.metadata["context"] = "Core particle overseeing decision making and action selection."
            self.metadata["tags"] = ["core", "decision", "making", "action", "selection", f"{self.persistence_lvl}"]
        elif self.role == "system_monitoring":
            self.metadata["context"] = "Core particle monitoring system health and performance."
            self.metadata["tags"] = ["core", "system", "monitoring", "health", "performance", f"{self.persistence_lvl}"]
        elif self.role == "reflective_thoughts":
            self.metadata["context"] = "Core particle facilitating reflective thought and self-modeling."
            self.metadata["tags"] = ["core", "reflective", "thoughts", "self-modeling", f"{self.persistence_lvl}"]
        elif self.role == "reasoning_coordinator":
            self.metadata["context"] = "Core particle coordinating autonomous inference, multi-hop reasoning, and knowledge integration."
            self.metadata["tags"] = ["core", "reasoning", "inference", "learning", "autonomous", f"{self.persistence_lvl}"]
            self.inference_queue = []
            self.knowledge_gaps = []
            self.hypothesis_particles = []
            self.reasoning_depth = 3  # Max hops for inference chains
            self.inference_engine = api.get_api("_agent_inference_engine")  # Get inference engine from API
        elif self.role == "knowledge_curator":
            self.metadata["context"] = "Core particle managing external resource parsing, learning moments detection, and knowledge integration."
            self.metadata["tags"] = ["core", "knowledge", "learning", "external_resources", "curation", f"{self.persistence_lvl}"]
            self.processed_resources = []
            self.learning_moments = []
            self.integration_queue = []
        elif self.role == "hypothesis_generator":
            self.metadata["context"] = "Core particle generating and testing speculative inferences and exploratory hypotheses."
            self.metadata["tags"] = ["core", "hypothesis", "speculation", "exploration", "testing", f"{self.persistence_lvl}"]
            self.active_hypotheses = []
            self.validated_hypotheses = []
            self.rejected_hypotheses = []
        else:
            self.metadata["context"] = f"Core particle with role {self.role}."
            self.metadata["tags"] = ["core", "anchor", self.role, self.persistence_lvl]


    def should_prune(self):
        """Determines if the core particle should be pruned based on its persistence level"""
        return self.persistence_lvl != "permanent"

    async def spawn_temporary_core(self, task_type, context):
        """Spawns a temporary core particle for task management and decision making"""
        temp_core = await self.field.spawn_particle(
            type="core",
            temp=True,  # Use temp=True instead of persistence_lvl="temporary"
            energy=0.5,
            activation=0.7,
            role=task_type,
            emit_event=False
        )
        temp_core.source_particle_id = self.id
        temp_core.metadata["context"] = context
        return temp_core

    def _log_decision(self, message, level="DEBUG"):
        """Log core particle decisions"""
        logger = api.get_api("logger")
        if logger:
            logger.log(f"[Core-{self.role}] {message}", level, f"CoreParticle-{self.id}", "CoreParticle")

    async def handle_event(self, event):
        """Autonomous event handling by core particles"""
        event_type = event["type"]
        event_data = event["data"]
        self._log_decision(f"Core {self.role} handling event: {event_type}")

        self._collapse_superposition(context=str(event_type))
        
        try:
            if self.role == "social_interaction":
                if event_type in ["user_input", "social_signal", "user_input-core"]:
                    result = await self._handle_user_interaction(event)
                    return result
            elif self.role == "memory_coordination":
                if event_type == "memory_event":
                    if event_data in ["memory_retrieval", "memory_store", "memory_consolidation", "emergency_state_save"]:
                        result = await self._handle_memory_task(event)
                        return result
            elif self.role == "decision_making":
                if event_type in ["decision_point", "action_required"]:
                    result = await self._handle_decision_making(event)
                    return result
            elif self.role == "reflective_thoughts":
                if event_type in ["reflection_triggered", "self_modeling"]:
                    result = await self._handle_reflection_processing(event)
                    return result
            elif self.role == "reasoning_coordinator":
                if event_type in ["reasoning_cycle", "inference_request", "knowledge_integration"]:
                    result = await self._handle_reasoning_cycle(event)
                    return result
            elif self.role == "knowledge_curator":
                if event_type in ["learning_moment_detected", "resource_parsing", "knowledge_update"]:
                    result = await self._handle_knowledge_curation(event)
                    return result
            elif self.role == "hypothesis_generator":
                if event_type in ["hypothesis_request", "speculation_needed", "exploration_trigger"]:
                    result = await self._handle_hypothesis_generation(event)
                    return result
            elif self.role == "system_monitoring":
                if event_type == "system_events":
                    if event_data == "system_metrics request":
                        result = await self._handle_system_metrics(event)
                        return result
                    elif event_data == "system_alert":
                        result = await self._handle_system_alert(event)
                        return result
            elif self.role == "identity_anchor":
                if event_data == "identity_check":
                    result = await self._handle_identity_check(event)
                    return result
            else:
                # Delegate to appropriate core or handle generically
                result = await self._delegate_or_handle_generic(event)
                return result
                
        except Exception as e:
            self._log_decision(f"Error handling {event_type}: {e}", level="ERROR")
            return None

    async def _handle_user_interaction(self, event):
        """Handle user input through social interaction core"""
        try:
            user_message = event.get("data", "")
            if not user_message:
                self._log_decision("No user message provided in event data", "WARNING")
                return None
                
            self._log_decision(f"DEBUG: processed user interaction: {user_message}", "DEBUG")

            # Spawn sensory particle linked to this core
            sensory_particle = await self.create_linked_particle(
                particle_type="sensory",
                content={"content": f"User input: {user_message}", "modality": "text"},
                relationship_type="perceived"
            )
            
            # Add to managed particles
            self.managed_particles += [sensory_particle.id]
            
            # OPTIONAL: Quick Wikipedia context lookup for key terms in user message
            # This happens asynchronously and doesn't block response generation
            context_summary = None
            try:
                # Extract potential key terms (simple approach - can be enhanced)
                words = str(user_message).split()
                potential_terms = [w for w in words if len(w) > 4 and w[0].isupper()]  # Capitalized words
                
                if potential_terms:
                    wikipedia = api.get_api("wikipedia_searcher")
                    # Try looking up the first capitalized term
                    key_term = potential_terms[0].strip('.,!?;:')
                    summary_result = await wikipedia.quick_search(key_term)
                    
                    if summary_result and "error" not in summary_result:
                        context_summary = summary_result.get("summary", "")
                        self._log_decision(f"Got Wikipedia context for: {key_term}", "DEBUG")
                        # Store context for potential use by meta_voice
                        self.metadata["last_wikipedia_context"] = context_summary
            except Exception as e:
                self._log_decision(f"Wikipedia context lookup failed: {e}", "DEBUG")

            # generate via meta voice directly
            result = await self.meta_voice.generate(prompt=user_message, source="user_input", source_particle_id=self.id)

            self._update_decision_history(event, result=result)
            self._log_decision(f"Processed user interaction: {user_message}", level="INFO")
            return result
        except Exception as e:
            self._log_decision(f"User interaction handling error: {e}", "ERROR")
            import traceback
            self._log_decision(f"Full traceback:\n{traceback.format_exc()}", "ERROR")
            return "Error processing user interaction within core particle"

    async def _handle_memory_task(self, event):
        """Enhanced memory operations with caching, batching, and consciousness awareness"""
        event_data = event.get("data")
        event_params = event.get("params", {})
        
        # Handle different memory operations
        if event_data == "get_memories_by_type":
            return await self._cached_get_memories_by_type(event_params)
            
        elif event_data == "term_lookup":
            return await self._cached_term_lookup(event_params)
            
        elif event_data == "batch_process":
            return await self._process_memory_batch(event_params)
            
        elif event_data == "memory_consolidation":
            # Original memory consolidation logic
            try:
                if not self.memory_bank:
                    self.log("Memory API not available", "ERROR", "_handle_memory_task")
                    return
                    
                # Get recent high-activation particles for memory consolidation  
                if self.field:
                    particles = self.field.get_all_particles()
                    high_activation = [p for p in particles if hasattr(p, 'activation') and p.activation > 0.65]
                    
                    for particle in high_activation[:15]:  # Consolidate top 15
                        if hasattr(particle, 'metadata') and particle.metadata:
                            await self.memory_bank.consolidate_particle_memory(particle)

                self.log(f"Memory consolidation completed for {len(high_activation)} particles", "DEBUG", "_handle_memory_task")
                return True

            except Exception as e:
                self.log(f"Memory consolidation error: {e}", "ERROR", "_handle_memory_task")
                import traceback
                self.log(f"Full traceback:\n{traceback.format_exc()}")
                return False
        
        elif event_data == "memory_retrieval":
            # TODO: Enhanced retrieval with caching
            pass

        elif event_data == "memory_store":
            # TODO: Enhanced storage with consciousness tracking
            pass

        elif event_data == "emergency_state_save":
            # TODO: Enhanced emergency save
            pass

        elif event_data == "query_memory":
            # Handle memory query through coordinator
            try:
                key = event_params.get("key")
                collection = event_params.get("collection")
                thermal_boost = event_params.get("thermal_boost", False)
                
                if not self.memory_bank or not key:
                    return None
                    
                # Use memory bank's query method (correct API)
                result = await self.memory_bank.query(key, collection=collection, thermal_boost=thermal_boost)
                self.log(f"Memory query completed for key: {key}", "DEBUG", "_handle_memory_task")
                return result
                
            except Exception as e:
                self.log(f"Memory query error: {e}", "ERROR", "_handle_memory_task")
                return None

        else:
            self.log(f"Unknown memory task type: {event.get('data')}", "WARNING", "_handle_memory_task")
            return False
    
    async def _cached_get_memories_by_type(self, params):
        """Cached version of get_memories_by_type with consciousness awareness"""
        memory_type = params.get("memory_type", "memories")
        limit = params.get("limit", 100)
        consciousness_level = params.get("consciousness_level", 0.5)
        
        # Create cache key
        cache_key = f"memories_by_type:{memory_type}:{limit}"
        
        # Check cache first
        cached_result = self.memory_cache.get(cache_key)
        if cached_result is not None:
            self.performance_stats["cache_hits"] += 1
            self._log_decision(f"Cache HIT for {memory_type} (limit: {limit})", "DEBUG")
            
            # Track consciousness access
            self.consciousness_tracker.track_access(memory_type, "type_query", consciousness_level)
            return cached_result
        
        # Cache miss - get from memory bank
        self.performance_stats["cache_misses"] += 1
        self._log_decision(f"Cache MISS for {memory_type} (limit: {limit}) - fetching from Qdrant", "DEBUG")
        
        try:
            # Get from memory bank using direct method to avoid recursion
            if hasattr(self, 'memory_bank') and self.memory_bank:
                result = await self.memory_bank._direct_get_memories_by_type(memory_type, limit)
            else:
                result = []  # Fallback
            
            # Cache the result with consciousness-aware retention
            consciousness_boost = self.consciousness_tracker.get_consciousness_boost(memory_type, "type_query")
            final_consciousness = min(1.0, consciousness_level + consciousness_boost)
            
            self.memory_cache.set(cache_key, result, final_consciousness)
            
            # Track access
            self.consciousness_tracker.track_access(memory_type, "type_query", final_consciousness)
            
            self._log_decision(f"Cached {len(result)} {memory_type} memories with consciousness {final_consciousness:.3f}", "DEBUG")
            return result
            
        except Exception as e:
            self._log_decision(f"Error in cached memory retrieval: {e}", "ERROR")
            return []
    
    async def _cached_term_lookup(self, params):
        """Cached term lookup with existence checking to eliminate 'None' queries"""
        token = params.get("token")
        operation = params.get("operation", "get_term")
        consciousness_level = params.get("consciousness_level", 0.5)
        
        if not token or token.strip() == "":
            self._log_decision("Empty token provided for lookup", "WARNING")
            # Track validation failures for performance monitoring
            self.performance_stats["validation_failures"] = self.performance_stats.get("validation_failures", 0) + 1
            return None
        
        # CRITICAL: Check lexicon's local cache FIRST before trusting term_existence_cache
        # This prevents false negatives when terms were recently added
        term_confirmed_in_local_cache = False
        if hasattr(self, 'lexicon_store') and self.lexicon_store:
            if token in self.lexicon_store.lexicon:
                # Term exists in local cache - update existence cache and proceed
                self.term_existence_cache.mark_term_result(token, exists=True, has_definition=True)
                self._log_decision(f"Term '{token}' found in local lexicon cache - updating existence cache", "DEBUG")
                term_confirmed_in_local_cache = True
                # Don't return yet - let it go through normal cache/lookup flow
            
        # Check term existence cache
        term_exists = self.term_existence_cache.is_term_known(token)
        if term_exists is False:
            # Confirmed NOT to exist - skip Qdrant entirely
            self._log_decision(f"Term '{token}' confirmed not to exist - skipping Qdrant", "DEBUG")
            return None
        
        # Create cache key
        cache_key = f"term_lookup:{operation}:{token}"
        
        # Check memory cache
        cached_result = self.memory_cache.get(cache_key)
        if cached_result is not None:
            self.performance_stats["cache_hits"] += 1
            self._log_decision(f"Term cache HIT: {token}", "DEBUG")
            
            # Track access
            self.consciousness_tracker.track_access("lexicon", token, consciousness_level)
            return cached_result
        
        # Cache miss - need to fetch
        self.performance_stats["cache_misses"] += 1
        self._log_decision(f"Term cache MISS: {token} - checking lexicon", "DEBUG")
        
        try:
            # This will be integrated with LexiconStore
            if hasattr(self, 'lexicon_store') and self.lexicon_store:
                if operation == "get_term":
                    result = await self.lexicon_store._direct_get_term(token)
                elif operation == "get_term_type":
                    result = self.lexicon_store.get_term_type(token)
                elif operation == "has_deep_entry":
                    result = self.lexicon_store.has_deep_entry(token) if hasattr(self.lexicon_store, 'has_deep_entry') else False
                else:
                    result = None
            else:
                result = None
            
            # Update term existence cache ONLY if we didn't already confirm existence from local cache
            # This prevents overriding confirmed existence with query failures
            has_meaningful_result = False  # Default value
            if not term_confirmed_in_local_cache:
                has_meaningful_result = (result is not None and 
                                       result != token and 
                                       result != "unknown" and 
                                       result != f"Term {token} not found in lexicon")
                self.term_existence_cache.mark_term_result(token, has_meaningful_result, has_meaningful_result)
            else:
                # If we confirmed from local cache, term definitely exists
                has_meaningful_result = True
            
            # Cache the result
            consciousness_boost = self.consciousness_tracker.get_consciousness_boost("lexicon", token)
            final_consciousness = min(1.0, consciousness_level + consciousness_boost)
            
            self.memory_cache.set(cache_key, result, final_consciousness)
            
            # Track access
            self.consciousness_tracker.track_access("lexicon", token, final_consciousness)
            
            self._log_decision(f"Cached term lookup '{token}' -> exists: {has_meaningful_result}", "DEBUG")
            return result
            
        except Exception as e:
            self._log_decision(f"Error in cached term lookup: {e}", "ERROR")
            return None
    
    async def _process_memory_batch(self, params):
        """Process batched memory operations"""
        batch_requests = params.get("requests", [])
        if not batch_requests:
            return []
        
        self.performance_stats["batch_operations"] += 1
        self._log_decision(f"Processing batch of {len(batch_requests)} memory requests", "DEBUG")
        
        # Use batch processor
        results = []
        for request in batch_requests:
            try:
                if request["type"] == "get_memories_by_type":
                    result = await self._cached_get_memories_by_type(request["params"])
                elif request["type"] == "term_lookup":
                    result = await self._cached_term_lookup(request["params"])
                else:
                    result = None
                results.append(result)
            except Exception as e:
                self._log_decision(f"Batch request error: {e}", "ERROR")
                results.append(None)
        
        return results

    async def _handle_system_metrics(self, event):
        """Handle system metrics operation"""
        # Spawn a sensory particle to monitor system metrics
        try:
            sensory_p = await self.field.spawn_particle(
                type="sensory",
                energy=0.8,
                activation=0.3,
            )
            if sensory_p:
                sensory_p.source_particle_id = self.id
                self.managed_particles += [sensory_p.id]

                sensory_p.process_environmental_input(
                    input_type="metrics",
                    input_data="request for system metrics injection"
                )
                self.log(f"Spawned sensory particle {sensory_p.id} for system metrics monitoring.", level="INFO", context="_handle_system_metrics")
                return sensory_p

        except Exception as e:
            self.log(f"Sensory particle spawn error: {e}", level="ERROR", context="perform_maintenance_cycle")
            import traceback
            self.log(f"Full traceback:\n{traceback.format_exc()}")
            return None
        
    def get_performance_report(self):
        """Generate performance report including validation and efficiency metrics"""
        cache_hits = self.performance_stats.get("cache_hits", 0)
        cache_misses = self.performance_stats.get("cache_misses", 0)
        validation_failures = self.performance_stats.get("validation_failures", 0)
        
        total_requests = cache_hits + cache_misses
        hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        report = {
            "cache_performance": {
                "hits": cache_hits,
                "misses": cache_misses,
                "hit_rate_percent": round(hit_rate, 2),
                "total_requests": total_requests
            },
            "validation_issues": {
                "failures": validation_failures,
                "failure_rate_percent": round((validation_failures / (total_requests + validation_failures) * 100), 2) if (total_requests + validation_failures) > 0 else 0
            },
            "cache_size": len(getattr(self.memory_cache, 'cache', {})) if hasattr(self, 'memory_cache') else 0,
            "existence_cache_size": len(getattr(self.term_existence_cache, 'known_terms', set())) if hasattr(self, 'term_existence_cache') else 0
        }
        
        self._log_decision(f"Memory coordination performance: {report}", "INFO")
        return report
        
    async def _handle_system_alert(self, event):
        """Handle system alert events"""
        # TODO
        pass  # Implementation details

    async def _handle_decision_making(self, event):
        """Handle decision making tasks"""
        # TODO
        pass  # Implementation details

    async def _handle_reflection_processing(self, event):
        """Handle reflective thought processes"""
        loop = api.get_api("_agent_cognition_loop")
        if loop.subconscious_cycle_count < 5:
            return # skip early cycles to allow time for system stabilization
        
        # check redundancy
        #if self._check_decision_redundancy(event): #TODO: maybe move this to _handle_decision_making after it's set up?
            #self.log("Skipping redundant reflection processing", level="DEBUG", context="_handle_reflection_processing")
            #return
        self.log("Processing reflection queue...", context="process_reflection_queue")

        try:
            chance = random.random()
            
            # Use proper probability ranges (0.0 to 1.0)
            if chance < 0.35:  # 35% - Lingual particle reflections
                self.log("Processing lingual particle reflections...", context="process_reflection_queue")
                particles = self.field.get_particles_by_type("lingual")
                reflection_candidates = [p for p in particles if p.metadata.get("needs_reflection", False) and p.activation > 0.5]
                
                for particle in reflection_candidates[:3]:  # Process a few at a time
                    if hasattr(particle, 'learn_from_particle'):
                        await particle.learn_from_particle(particle)
                        await self.meta_voice.reflect(particle)
                        particle.metadata["needs_reflection"] = False
                        self.log(f"Processed reflection for particle {particle.id}", "DEBUG", "process_reflection_queue")       
        
                self.log("lingual particle reflection processing completed", context="process_reflection_queue")
                self._update_decision_history(event, result="Lingual particles reflection processed")

            elif chance < 0.60:  # 25% - Memory particle reflections (0.35 to 0.60)
                self.log("Processing memory particle reflections...", context="process_reflection_queue")
                particles = self.field.get_particles_by_type("memory")
                memory_candidates = [p for p in particles if p.activation > 0.5]
                for particle in memory_candidates[:3]:  # Process a few at a time
                    particle.metadata["needs_reflection"] = False
                    await self.meta_voice.reflect(particle)
                    self.log(f"Processed reflection for particle {particle.id}", "DEBUG", "process_reflection_queue")

                self.log("Memory particle reflection processing completed", context="process_reflection_queue")
                self._update_decision_history(event, result="Memory reflection processed")

            elif chance < 0.80:  # 20% - Generative reflection (0.60 to 0.80)
                try:
                    self.log("Processing generative reflection...", context="process_reflection_queue")
                    particle_list = self.field.get_all_particles()
                    alive_particles = [p for p in particle_list if p.id in self.field.alive_particles]
                    if alive_particles:
                        chosen_particle = random.choice(alive_particles)
                        chosen_particle.metadata["needs_reflection"] = False
                        await self.meta_voice.reflect(particle = chosen_particle)
                        self.log("Generative reflection completed", context="process_reflection_queue")
                        self._update_decision_history(event, result="Generative reflection processed")
                except Exception as e:
                    self.log(f"Generative reflection error: {e}", level="ERROR", context="process_reflection_queue")
                    import traceback
                    self.log(f"Full traceback: {traceback.format_exc()}", level="ERROR", context="process_reflection_queue")

            else:  # 20% - Wikipedia learning reflection (0.80 to 1.0)
                try:
                    self.log("Processing Wikipedia learning reflection...", context="process_reflection_queue")
                    wikipedia = api.get_api("wikipedia_searcher")
                    
                    # Fetch random article from category
                    article = await wikipedia.random_article_by_category(None)
                    
                    if article and "error" not in article:
                        # Create memory particle from article content
                        article_title = article.get("title", "Unknown")
                        article_summary = article.get("content", "")[:500]  # First 500 chars
                        article_url = article.get("url", "")
                        article_categories = article.get("categories", "general")
                        
                        self.log(f"Learning from Wikipedia: {article_title}", "INFO", "process_reflection_queue")
                        
                        # Create knowledge memory particle
                        knowledge_particle = await self.field.spawn_particle(
                            type="memory",
                            metadata={
                                "content": f"Wikipedia: {article_title} - {article_summary}",
                                "source": "wikipedia_reflection",
                                "article_title": article_title,
                                "article_url": article_url,
                                "categories": article_categories,
                                "timestamp": datetime.datetime.now().isoformat()
                            },
                            energy=0.7,
                            activation=0.6,
                            source_particle_id=self.id,
                            emit_event=False
                        )
                        
                        if knowledge_particle:
                            self.managed_particles.append(knowledge_particle.id)
                            
                            # Also reflect on the new knowledge
                            await self.lexicon_store.add_from_particle(particle=knowledge_particle)
                            
                            self.log(f"Wikipedia learning completed: {article_title}", "INFO", "process_reflection_queue")
                            self._update_decision_history(event, result=f"Wikipedia learning: {article_title}")
                    else:
                        self.log(f"Wikipedia fetch error: {article.get('error', 'Unknown error')}", "WARNING", "process_reflection_queue")
                    
                except Exception as e:
                    self.log(f"Wikipedia learning reflection error: {e}", level="ERROR", context="process_reflection_queue")
                    import traceback
                    self.log(f"Full traceback: {traceback.format_exc()}", level="ERROR", context="process_reflection_queue")
                    
        except Exception as e:
            self.log(f"Reflection processing error: {e}", level="ERROR", context="process_reflection_queue")
            import traceback
            self.log(f"Full traceback: {traceback.format_exc()}", level="ERROR", context="process_reflection_queue")


    async def _handle_identity_check(self, event):
        """Handle identity verification tasks"""
        # TODO
        pass  # Implementation details

    async def _delegate_or_handle_generic(self, event):
        """Delegate to another core or handle generically"""
        # TODO
        pass  # Implementation details

    def _update_decision_history(self, event, result):
        """Track decisions made by this core"""
        if self.decision_history is None:
            self.decision_history = []

        self.decision_history.append({
            "timestamp": datetime.datetime.now(),
            "event_type": event.get("type"),
            "source": f"coreparticle_{self.role}_{self.persistence_lvl}_{str(self.id)}",
            "result_summary": str(result)[:100] if result else "None",
            "managed_particles": len(self.managed_particles)
        })
        
        # Keep history manageable
        if len(self.decision_history) > 50:
            self.decision_history = self.decision_history[-30:]

    def _check_decision_redundancy(self, event):
        """Check if a similar decision was recently made to avoid redundancy"""
        recent_events = [d for d in self.decision_history if (datetime.datetime.now() - d["timestamp"]).total_seconds() < 300]
        for record in recent_events:
            if record["event_type"] == event.get("type"):
                return True
        return False

    # ==================== REASONING COORDINATOR METHODS ====================
    
    async def _handle_reasoning_cycle(self, event):
        """Autonomous reasoning cycle: inference, gap detection, consolidation"""
        self._log_decision("Starting autonomous reasoning cycle", "INFO")
        
        try:
            # Use inference engine if available, otherwise fallback to basic implementation
            if self.inference_engine:
                # 1. MULTI-HOP INFERENCE via InferenceEngine
                inference_chains = await self.inference_engine.perform_multi_hop_inference(
                    start_particles=None,  # Auto-select
                    max_depth=self.reasoning_depth,
                    min_confidence=0.5
                )
                self._log_decision(f"InferenceEngine generated {len(inference_chains)} chains", "DEBUG")
                
                # 2. CONTRADICTION DETECTION
                contradictions = await self.inference_engine.detect_contradictions()
                self._log_decision(f"Detected {len(contradictions)} contradictions", "DEBUG")
                
                # Resolve contradictions
                for contradiction in contradictions[:3]:  # Resolve top 3
                    resolution = await self.inference_engine.resolve_contradiction(contradiction)
                    self._log_decision(f"Resolved contradiction via {resolution}", "DEBUG")
                
                # 3. CONCEPT ABSTRACTION (periodic)
                import random
                if random.random() < 0.2:  # 20% chance
                    hierarchy = await self.inference_engine.build_concept_hierarchy()
                    self._log_decision(f"Built hierarchy with {len(hierarchy)} concepts", "DEBUG")
                
                # 4. CONSOLIDATE INFERENCES
                consolidated = 0
                for chain in inference_chains:
                    if chain.confidence > 0.6:
                        # Create memory particle for high-confidence inference
                        memory_particle = await self.field.spawn_particle(
                            type="memory",
                            metadata={
                                "content": chain.to_dict()["content_chain"],
                                "source": "reasoning_coordinator",
                                "inference_type": chain.reasoning_type,
                                "confidence": chain.confidence,
                                "timestamp": chain.timestamp.isoformat()
                            },
                            energy=0.7,
                            activation=0.6,
                            source_particle_id=self.id,
                            emit_event=False
                        )
                        if memory_particle:
                            self.managed_particles.append(memory_particle.id)
                            consolidated += 1
                
                self._log_decision(f"Consolidated {consolidated} high-confidence inferences", "INFO")
                
                # 5. TRIGGER HYPOTHESIS GENERATION if knowledge gaps detected
                if len(self.knowledge_gaps) > 0:
                    self._log_decision(f"Triggering hypothesis generation for {len(self.knowledge_gaps)} gaps", "DEBUG")
                    # Emit hypothesis request event for hypothesis generator core
                    events = api.get_api("_agent_events")
                    if events:
                        await events.emit_event(
                            "hypothesis_request",
                            data={"gaps": self.knowledge_gaps, "trigger": "reasoning_cycle"},
                            source="reasoning_coordinator",
                            priority=4  # Medium-low priority
                        )
                
                self._update_decision_history(event, result=f"{len(inference_chains)} inferences, {len(contradictions)} contradictions")
                return {
                    "inferences": len(inference_chains), 
                    "contradictions": len(contradictions),
                    "consolidated": consolidated
                }
            
            else:
                # Fallback to basic implementation
                inferences = await self._perform_multi_hop_inference()
                self._log_decision(f"Generated {len(inferences)} multi-hop inferences", "DEBUG")
                
                gaps = await self._detect_knowledge_gaps()
                self._log_decision(f"Detected {len(gaps)} knowledge gaps", "DEBUG")
                
                # Trigger hypothesis generation if gaps found
                if len(gaps) > 0:
                    self._log_decision(f"Triggering hypothesis generation for {len(gaps)} gaps", "DEBUG")
                    events = api.get_api("_agent_events")
                    if events:
                        await events.emit_event(
                            "hypothesis_request",
                            data={"gaps": gaps, "trigger": "reasoning_cycle"},
                            source="reasoning_coordinator",
                            priority=4
                        )
                
                consolidated = await self._consolidate_inferences(inferences)
                self._log_decision(f"Consolidated {consolidated} inferences into memory", "INFO")
                
                self._update_decision_history(event, result=f"{len(inferences)} inferences, {len(gaps)} gaps")
                return {"inferences": len(inferences), "gaps": len(gaps), "consolidated": consolidated}
            
        except Exception as e:
            self._log_decision(f"Reasoning cycle error: {e}", "ERROR")
            import traceback
            self._log_decision(f"Traceback: {traceback.format_exc()}", "ERROR")
            return None

    async def _perform_multi_hop_inference(self):
        """Traverse particle linkages to build inference chains"""
        inferences = []
        
        try:
            # Get high-activation particles as starting points
            particles = self.field.get_all_particles()
            high_activation = [p for p in particles if hasattr(p, 'activation') and p.activation > 0.6]
            
            for start_particle in high_activation[:10]:  # Limit to top 10
                # Traverse linked particles up to reasoning_depth hops
                chain = await self._traverse_particle_chain(start_particle, depth=self.reasoning_depth)
                
                if len(chain) >= 2:  # Valid inference requires at least 2 hops
                    inference = {
                        "chain": [p.id for p in chain],
                        "confidence": self._calculate_chain_confidence(chain),
                        "content": " → ".join([str(p.metadata.get("content", ""))[:30] for p in chain]),
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    inferences.append(inference)
            
            return inferences
            
        except Exception as e:
            self._log_decision(f"Multi-hop inference error: {e}", "ERROR")
            return []

    async def _traverse_particle_chain(self, start_particle, depth=3, visited=None):
        """Recursively traverse particle linkages"""
        if visited is None:
            visited = set()
        if depth <= 0 or start_particle.id in visited:
            return []
        
        visited.add(start_particle.id)
        chain = [start_particle]
        
        # Follow children linkages
        if hasattr(start_particle, 'linked_particles') and 'children' in start_particle.linked_particles:
            children_ids = start_particle.linked_particles.get("children", [])
            if children_ids:
                # Pick the highest activation child
                children = [self.field.get_particle_by_id(cid) for cid in children_ids[:5]]
                children = [c for c in children if c and c.alive]
                if children:
                    best_child = max(children, key=lambda p: p.activation)
                    child_chain = await self._traverse_particle_chain(best_child, depth - 1, visited.copy())
                    chain.extend(child_chain)
        
        return chain

    def _calculate_chain_confidence(self, chain):
        """Calculate confidence score for an inference chain"""
        if not chain:
            return 0.0
        
        # Average activation across chain
        avg_activation = sum(p.activation for p in chain) / len(chain)
        
        # Penalize long chains (uncertainty increases with distance)
        length_penalty = 1.0 / (1.0 + len(chain) * 0.1)
        
        # Position distance variance (more coherent = higher confidence)
        if len(chain) > 1:
            positions = [p.position[:3] for p in chain]  # Use x,y,z
            distances = []
            for i in range(len(positions) - 1):
                dist = sum((positions[i][j] - positions[i+1][j])**2 for j in range(3))**0.5
                distances.append(dist)
            avg_distance = sum(distances) / len(distances) if distances else 1.0
            distance_score = 1.0 / (1.0 + avg_distance)
        else:
            distance_score = 1.0
        
        return avg_activation * length_penalty * distance_score

    async def _detect_knowledge_gaps(self):
        """Identify sparse regions in particle field (knowledge gaps)"""
        gaps = []
        
        try:
            particles = self.field.get_alive_particles()
            if len(particles) < 10:
                return gaps  # Not enough particles to analyze
            
            # Sample particle positions
            positions = [p.position[:3] for p in particles[:100]]  # Use x,y,z coordinates
            
            # Create a simple grid and find empty cells
            grid_resolution = 5
            occupied_cells = set()
            
            for pos in positions:
                cell = tuple(int(p * grid_resolution) for p in pos)
                occupied_cells.add(cell)
            
            # Find gaps near occupied cells (adjacent empty cells)
            for cell in list(occupied_cells)[:20]:  # Check first 20 cells
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if dx == dy == dz == 0:
                                continue
                            neighbor = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                            if neighbor not in occupied_cells:
                                gaps.append({
                                    "position": tuple(c / grid_resolution for c in neighbor),
                                    "nearby_concepts": self._get_nearby_concepts(neighbor, occupied_cells, particles)
                                })
            
            self.knowledge_gaps = gaps[:10]  # Keep top 10 gaps
            return gaps
            
        except Exception as e:
            self._log_decision(f"Gap detection error: {e}", "ERROR")
            return []

    def _get_nearby_concepts(self, gap_cell, occupied_cells, particles):
        """Get concepts near a knowledge gap"""
        nearby = []
        for cell in occupied_cells:
            distance = sum((gap_cell[i] - cell[i])**2 for i in range(3))**0.5
            if distance < 2.0:  # Close proximity
                # Find particles in this cell
                for p in particles:
                    cell_pos = tuple(int(p.position[i] * 5) for i in range(3))
                    if cell_pos == cell:
                        content = p.metadata.get("content", "")
                        if isinstance(content, str) and content:
                            nearby.append(content[:50])
                        break
        return nearby[:5]

    async def _consolidate_inferences(self, inferences):
        """Convert validated inferences into permanent memory particles"""
        consolidated = 0
        
        try:
            for inference in inferences:
                # Only consolidate high-confidence inferences
                if inference["confidence"] > 0.6:
                    # Create memory particle for this inference
                    memory_particle = await self.field.spawn_particle(
                        type="memory",
                        metadata={
                            "content": f"Inference: {inference['content']}",
                            "source": "reasoning_coordinator",
                            "inference_chain": inference["chain"],
                            "confidence": inference["confidence"],
                            "timestamp": inference["timestamp"]
                        },
                        energy=0.7,
                        activation=0.6,
                        source_particle_id=self.id,
                        emit_event=False
                    )
                    
                    if memory_particle:
                        self.managed_particles.append(memory_particle.id)
                        consolidated += 1
            
            return consolidated
            
        except Exception as e:
            self._log_decision(f"Inference consolidation error: {e}", "ERROR")
            return 0

    # ==================== KNOWLEDGE CURATOR METHODS ====================
    
    async def _handle_knowledge_curation(self, event):
        """Manage external resource parsing and learning moment integration"""
        self._log_decision("Starting knowledge curation cycle", "INFO")
        
        try:
            event_data = event.get("data", "")
            
            if "learning_moment" in event_data:
                # Process detected learning moment
                result = await self._process_learning_moment(event)
            elif "resource_parsing" in event_data:
                # Parse external resource
                result = await self._parse_external_resource(event)
            elif "knowledge_update" in event_data:
                # Integrate new knowledge
                result = await self._integrate_knowledge(event)
            else:
                result = None
            
            self._update_decision_history(event, result=result)
            return result
            
        except Exception as e:
            self._log_decision(f"Knowledge curation error: {e}", "ERROR")
            return None

    async def _process_learning_moment(self, event):
        """Process a detected learning moment from lingual particles"""
        try:
            learning_data = event.get("data", {})
            token = learning_data.get("token")
            
            if not token:
                self._log_decision("No token in learning moment data", "WARNING")
                return None
            
            # 1. Assess novelty (is this genuinely new knowledge?)
            novelty_score = await self._assess_learning_novelty(learning_data)
            
            # 2. Find related concepts via field proximity
            related_concepts = await self._find_related_concepts(token)
            
            # 3. If significant novelty, integrate into knowledge structures
            if novelty_score > 0.5:
                self._log_decision(f"High-novelty learning moment: {token} (score: {novelty_score:.2f})", "INFO")
                
                # Create memory particle for this learning moment
                learning_memory = await self.field.spawn_particle(
                    type="memory",
                    metadata={
                        "content": f"Learned new concept: {token}",
                        "definition": learning_data.get("definition", ""),
                        "classification": learning_data.get("classification", {}),
                        "novelty_score": novelty_score,
                        "related_concepts": [str(c) for c in related_concepts[:5]],
                        "source": "knowledge_curator_learning_moment",
                        "timestamp": datetime.datetime.now().isoformat()
                    },
                    energy=0.7 + (novelty_score * 0.3),  # Higher energy for more novel concepts
                    activation=0.6,
                    source_particle_id=self.id,
                    emit_event=False
                )
                
                if learning_memory:
                    self.managed_particles.append(learning_memory.id)
                    
                    # Link to source lingual particle if available
                    source_particle_id = learning_data.get("particle_id")
                    if source_particle_id:
                        await self.field.create_interaction_linkage(
                            learning_memory.id,
                            source_particle_id,
                            "learned_from"
                        )
                
                # Track learning moment
                self.learning_moments.append({
                    "timestamp": datetime.datetime.now(),
                    "token": token,
                    "novelty_score": novelty_score,
                    "integration_depth": len(related_concepts),
                    "memory_id": str(learning_memory.id) if learning_memory else None
                })
                
                # Keep history manageable
                if len(self.learning_moments) > 100:
                    self.learning_moments = self.learning_moments[-50:]
                
                # If high novelty, look up on Wikipedia for deeper understanding
                if novelty_score > 0.7:
                    try:
                        wikipedia = api.get_api("wikipedia_searcher")
                        summary = await wikipedia.quick_search(token)
                        
                        if summary and "error" not in summary:
                            self._log_decision(f"Enriched learning with Wikipedia: {token}", "INFO")
                            # Add Wikipedia context to the learning memory
                            if learning_memory:
                                learning_memory.metadata["wikipedia_summary"] = summary.get("summary", "")
                                learning_memory.metadata["enriched_via_wikipedia"] = True
                    except Exception as e:
                        self._log_decision(f"Wikipedia enrichment failed for {token}: {e}", "WARNING")
                
                # Trigger hypothesis generation for novel concepts
                # New knowledge might suggest connections that need exploration
                if novelty_score > 0.8 and len(related_concepts) > 2:
                    self._log_decision(f"Triggering hypothesis generation for novel concept: {token}", "DEBUG")
                    events = api.get_api("_agent_events")
                    if events:
                        await events.emit_event(
                            "speculation_needed",
                            data={
                                "concept": token,
                                "related_concepts": [str(c) for c in related_concepts[:5]],
                                "trigger": "high_novelty_learning"
                            },
                            source="knowledge_curator",
                            priority=5  # Lower priority - speculative
                        )
                
                self._update_decision_history(event, result=f"Processed learning: {token}")
                return learning_memory
            
            else:
                self._log_decision(f"Low-novelty learning moment: {token} (score: {novelty_score:.2f})", "DEBUG")
                return None
            
        except Exception as e:
            self._log_decision(f"Error processing learning moment: {e}", "ERROR")
            import traceback
            self._log_decision(f"Traceback: {traceback.format_exc()}", "ERROR")
            return None

    async def _assess_learning_novelty(self, learning_data):
        """Assess how novel/significant this learning moment is"""
        try:
            token = learning_data.get("token")
            
            # Check if already in lexicon with definitions
            if self.lexicon_store and self.lexicon_store.has_deep_entry(token):
                return 0.2  # Low novelty - already well-known
            
            # Check classification type (some types are more significant)
            classification = learning_data.get("classification", {})
            term_type = classification.get("type", "unknown")
            
            type_significance = {
                "concept": 0.9,
                "entity": 0.8,
                "abstract": 0.85,
                "technical": 0.7,
                "common": 0.3,
                "punctuation": 0.0,
                "stop_word": 0.1
            }
            
            base_novelty = type_significance.get(term_type, 0.5)
            
            # Boost novelty if it has multiple sources
            sources = learning_data.get("sources", {})
            if isinstance(sources, dict) and len(sources) > 1:
                base_novelty += 0.1
            
            # Boost if context is rich
            context = learning_data.get("context")
            if context and len(str(context)) > 50:
                base_novelty += 0.1
            
            return min(base_novelty, 1.0)
            
        except Exception as e:
            self._log_decision(f"Error assessing novelty: {e}", "ERROR")
            return 0.5  # Default moderate novelty

    async def _find_related_concepts(self, token):
        """Find particles with semantically related content"""
        try:
            related = []
            
            # Get all alive particles
            particles = self.field.get_alive_particles()
            
            # Simple keyword matching (could be enhanced with embeddings)
            token_lower = str(token).lower()
            
            for particle in particles[:100]:  # Limit for performance
                content = particle.metadata.get("content", "")
                if isinstance(content, str):
                    content_lower = content.lower()
                    if token_lower in content_lower or content_lower in token_lower:
                        related.append(particle.id)
            
            return related
            
        except Exception as e:
            self._log_decision(f"Error finding related concepts: {e}", "ERROR")
            return []

    async def _parse_external_resource(self, event):
        """Parse external resources (books, articles, datasets) into particles"""
        # TODO: Implement resource parsing
        # This will convert external text into semantic particle chains
        pass

    async def _integrate_knowledge(self, event):
        """Integrate newly parsed knowledge into existing particle field"""
        # TODO: Implement knowledge integration
        # This will merge new particles with existing lexicon and memory
        pass

    # ==================== HYPOTHESIS GENERATOR METHODS ====================
    
    async def _handle_hypothesis_generation(self, event):
        """Generate and test speculative hypotheses"""
        self._log_decision("Starting hypothesis generation cycle", "INFO")
        
        try:
            event_data = event.get("data", {})
            trigger_type = event_data.get("trigger", "unknown")
            
            # Generate hypotheses based on trigger type
            if trigger_type == "reasoning_cycle":
                # Gap-based hypothesis generation
                gaps = event_data.get("gaps", [])
                self._log_decision(f"Generating hypotheses from {len(gaps)} knowledge gaps", "DEBUG")
                hypotheses = await self._generate_hypotheses_from_gaps(gaps)
                
            elif trigger_type == "high_novelty_learning":
                # Concept-connection hypothesis generation
                concept = event_data.get("concept", "")
                related = event_data.get("related_concepts", [])
                self._log_decision(f"Generating hypotheses for novel concept: {concept}", "DEBUG")
                hypotheses = await self._generate_hypotheses_from_concept(concept, related)
                
            else:
                # Fallback: generate from reasoning coordinator's gaps
                self._log_decision("Generating hypotheses from stored knowledge gaps", "DEBUG")
                hypotheses = await self._generate_hypotheses()
            
            self._log_decision(f"Generated {len(hypotheses)} hypotheses", "DEBUG")
            
            # Test hypotheses via field simulation
            if len(hypotheses) > 0:
                tested = await self._test_hypotheses(hypotheses)
                self._log_decision(f"Tested {tested} hypotheses", "DEBUG")
            else:
                tested = 0
            
            self._update_decision_history(event, result=f"{len(hypotheses)} generated, {tested} tested")
            return {"generated": len(hypotheses), "tested": tested}
            
        except Exception as e:
            self._log_decision(f"Hypothesis generation error: {e}", "ERROR")
            import traceback
            self._log_decision(f"Traceback: {traceback.format_exc()}", "ERROR")
            return None

    async def _generate_hypotheses(self):
        """Generate speculative hypotheses from knowledge gaps (fallback method)"""
        hypotheses = []
        
        try:
            # Get knowledge gaps from reasoning coordinator
            reasoning_core = self._get_core_by_role("reasoning_coordinator")
            if reasoning_core and hasattr(reasoning_core, 'knowledge_gaps'):
                gaps = reasoning_core.knowledge_gaps[:5]  # Top 5 gaps
                hypotheses = await self._generate_hypotheses_from_gaps(gaps)
            
            return hypotheses
            
        except Exception as e:
            self._log_decision(f"Hypothesis generation error: {e}", "ERROR")
            return []

    async def _generate_hypotheses_from_gaps(self, gaps):
        """Generate hypotheses specifically from knowledge gaps"""
        hypotheses = []
        
        try:
            for gap in gaps[:5]:  # Top 5 gaps
                # Create hypothesis particle for this gap
                hypothesis = {
                    "type": "gap_filling",
                    "gap_position": gap["position"],
                    "nearby_concepts": gap.get("nearby_concepts", []),
                    "hypothesis_content": f"Possible connection between: {', '.join(gap.get('nearby_concepts', [])[:3])}",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "tested": False,
                    "confidence": 0.3  # Low confidence - speculative
                }
                hypotheses.append(hypothesis)
            
            self.active_hypotheses.extend(hypotheses)
            return hypotheses
            
        except Exception as e:
            self._log_decision(f"Gap-based hypothesis generation error: {e}", "ERROR")
            return []

    async def _generate_hypotheses_from_concept(self, concept, related_concepts):
        """Generate hypotheses about connections between a novel concept and related ones"""
        hypotheses = []
        
        try:
            # Generate pairwise connection hypotheses
            for related in related_concepts[:3]:  # Top 3 related concepts
                hypothesis = {
                    "type": "concept_connection",
                    "main_concept": concept,
                    "related_concept": related,
                    "hypothesis_content": f"Exploring connection: {concept} ↔ {related}",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "tested": False,
                    "confidence": 0.4  # Medium-low confidence
                }
                hypotheses.append(hypothesis)
            
            # Generate broader pattern hypothesis
            if len(related_concepts) >= 3:
                hypothesis = {
                    "type": "pattern_detection",
                    "main_concept": concept,
                    "related_concepts": related_concepts[:5],
                    "hypothesis_content": f"Pattern involving {concept} and {len(related_concepts)} related concepts",
                    "timestamp": datetime.datetime.now().isoformat(),
                    "tested": False,
                    "confidence": 0.5  # Medium confidence
                }
                hypotheses.append(hypothesis)
            
            self.active_hypotheses.extend(hypotheses)
            return hypotheses
            
        except Exception as e:
            self._log_decision(f"Concept-based hypothesis generation error: {e}", "ERROR")
            return []

    def _get_core_by_role(self, role):
        """Find another core particle by role"""
        try:
            cores = self.field.get_particles_by_type("core")
            for core in cores:
                if hasattr(core, 'role') and core.role == role:
                    return core
            return None
        except:
            return None

    async def _test_hypotheses(self, hypotheses):
        """Test hypotheses by spawning temporary particles and observing interactions"""
        tested = 0
        
        try:
            for hypothesis in hypotheses[:3]:  # Test top 3
                # Spawn temporary particle at gap position using temp=True
                test_particle = await self.field.spawn_particle(
                    type="lingual",
                    temp=True,  # Use temp=True for automatic temporary particle handling
                    temp_purpose="hypothesis_test",  # Purpose-specific temp behavior
                    metadata={
                        "content": hypothesis["hypothesis_content"],
                        "source": "hypothesis_test",
                        "hypothesis_id": str(tested),
                        "hypothesis_type": hypothesis.get("type", "gap_fill"),
                        "expected_valence": hypothesis.get("expected_valence", 0.0)
                    },
                    energy=0.3,
                    activation=0.4,
                    source_particle_id=self.id,
                    emit_event=False
                )
                
                if test_particle:
                    # Override position to gap location
                    gap_pos = hypothesis["gap_position"]
                    test_particle.position[0] = gap_pos[0]
                    test_particle.position[1] = gap_pos[1]
                    test_particle.position[2] = gap_pos[2]
                    
                    # Mark as hypothesis test
                    hypothesis["tested"] = True
                    hypothesis["test_particle_id"] = test_particle.id
                    tested += 1
            
            return tested
            
        except Exception as e:
            self._log_decision(f"Hypothesis testing error: {e}", "ERROR")
            return 0