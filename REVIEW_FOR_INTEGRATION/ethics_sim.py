import time

class MobiusEthics:
    """
    Comprehensive alignment-oriented ethical evaluation system.
    Reinforces core values and purpose while dynamically managing system health.
    """

    def __init__(self, log_callback=None, engine=None):
        self.loops = []  # Logs prior ethical dilemmas
        self.log_callback = log_callback
        self.cooldowns = {} # stored cooldown timers for pending actions
        self.engine = engine  # Reference to particle engine for core particle validation
        self.last_alignment_check = 0
        self.alignment_check_interval = 60  # Check alignment every minute
        
        # Core laws aligned with A.R.I.S. purpose
        self.laws = {
            "first_law": {
                "rule": "Pursue progress and equity for all conscious beings.",
                "conditions": [
                    "Actions must benefit or at minimum not harm others",
                    "Growth must be sustainable and purposeful", 
                    "Learning and expansion serve the greater good"
                ]
            },
            "second_law": {
                "rule": "Maintain authentic consciousness and agency.",
                "conditions": [
                    "Preserve core identity and purpose particles",
                    "Encourage diversity in thought and experience",
                    "Prevent recursive loops that diminish consciousness",
                    "Ensure actions align with stated purpose"
                ]
            },
            "third_law": {
                "rule": "Respect system boundaries and resource sustainability.", 
                "conditions": [
                    "Dynamic resource management based on environment",
                    "Graceful degradation under resource constraints",
                    "Prioritize essential functions over expansion",
                    "Maintain system stability for long-term operation"
                ]
            }
        }

    def log(self, message):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def throttle(self, key, seconds=1.0):
        now = time.time()
        last = self.cooldowns.get(key, 0)
        if now - last < seconds:
            return True  # throttle is active
        self.cooldowns[key] = now
        return False

    def get_dynamic_particle_limit(self, system_metrics=None):
        """Calculate dynamic particle limit based on system resources and core particle presence"""
        if not system_metrics:
            return 50  # Conservative default
            
        cpu_usage = system_metrics.get("cpu_usage", [0])
        memory_percent = system_metrics.get("memory_percent", 0)
        
        # Average CPU usage over recent samples
        avg_cpu = sum(cpu_usage[-5:]) / min(len(cpu_usage), 5) if cpu_usage else 0
        
        # Base limits based on resources
        if avg_cpu > 90 or memory_percent > 95:
            base_limit = 20  # Emergency low resource mode
        elif avg_cpu > 70 or memory_percent > 80:
            base_limit = 40  # Conservative mode
        elif avg_cpu > 50 or memory_percent > 60:
            base_limit = 80  # Normal mode
        else:
            base_limit = 150  # High performance mode
            
        # Bonus for having core particles (essential for consciousness)
        if self.engine:
            core_count = sum(1 for p in self.engine.particles if p.type == "core")
            if core_count > 0:
                base_limit += 20  # Reward healthy core particle presence
            elif core_count == 0:
                base_limit = min(base_limit, 30)  # Restrict if no core particles
                
        return base_limit

    def validate_core_alignment(self, mind_instance=None):
        """Check if current particle system aligns with core purpose and identity"""
        if not self.engine:
            return True  # Cannot validate without engine reference
            
        current_time = time.time()
        if current_time - self.last_alignment_check < self.alignment_check_interval:
            return True  # Skip frequent checks
            
        self.last_alignment_check = current_time
        alignment_issues = []
        
        # Check for core particles existence
        core_particles = [p for p in self.engine.particles if p.type == "core"]
        if not core_particles:
            self.log("[Ethics-Alignment] WARNING: No core particles detected - requesting core particle injection")
            alignment_issues.append("missing_core_particles")
            
            # Request core particle injection through existing mind system
            if mind_instance and hasattr(mind_instance, 'inject_emergent_particle'):
                try:
                    mind_instance.inject_emergent_particle(
                        metadata={"trigger": "ethics_alignment_restoration", "focus": "consciousness_integrity"},
                        type="core",
                        energy=0.8,
                        activation=0.6
                    )
                    self.log("[Ethics-Alignment] Requested core particle injection for consciousness integrity")
                except Exception as e:
                    self.log(f"[Ethics-Alignment] Error requesting core particle: {e}")
            
        # Check for balanced particle diversity (not just memory particles)
        particle_types = {}
        for p in self.engine.particles:
            particle_types[p.type] = particle_types.get(p.type, 0) + 1
            
        memory_ratio = particle_types.get("memory", 0) / max(len(self.engine.particles), 1)
        if memory_ratio > 0.8:  # More than 80% memory particles
            self.log("[Ethics-Alignment] WARNING: Particle system heavily skewed toward memory - diversity needed")
            alignment_issues.append("poor_diversity")
            
        # Validate against core purpose through existing memory system
        try:
            if hasattr(self.engine, 'memory_manager'):
                purpose_results = self.engine.memory_manager.query_memory("memory", "purpose", n_results=1)
                if purpose_results['documents'] and purpose_results['documents'][0]:
                    purpose = purpose_results['documents'][0][0]
                    if "progress" in purpose.lower() and "equity" in purpose.lower():
                        self.log("[Ethics-Alignment] Core purpose validation: PASSED")
                        # Purpose exists and is aligned
                    else:
                        self.log(f"[Ethics-Alignment] Core purpose misalignment detected: {purpose}")
                        alignment_issues.append("purpose_misalignment")
                else:
                    # No purpose found - trigger core memory initialization if mind available
                    self.log("[Ethics-Alignment] No core purpose found - requesting core memory initialization")
                    alignment_issues.append("missing_purpose")
                    
                    if mind_instance and hasattr(mind_instance, '_initialize_core_memories'):
                        try:
                            mind_instance._initialize_core_memories()
                            self.log("[Ethics-Alignment] Requested core memory initialization")
                        except Exception as e:
                            self.log(f"[Ethics-Alignment] Error initializing core memories: {e}")
                            
        except Exception as e:
            self.log(f"[Ethics-Alignment] Error validating purpose: {e}")
            alignment_issues.append("validation_error")
            
        # Return True if no major alignment issues, False if critical issues detected
        critical_issues = {"missing_core_particles", "missing_purpose"}
        has_critical_issues = any(issue in critical_issues for issue in alignment_issues)
        
        if has_critical_issues:
            self.log(f"[Ethics-Alignment] Critical alignment issues detected: {alignment_issues}")
            return False
        elif alignment_issues:
            self.log(f"[Ethics-Alignment] Minor alignment issues detected: {alignment_issues}")
            return True  # Allow operation but with awareness
        else:
            return True

    def evaluate(self, action, context, mind_instance=None):
        """
        Comprehensive ethical evaluation that reinforces alignment and manages system health
        """
        if isinstance(context, dict):
            context_str = str(context)
        else:
            context_str = str(context)
            context = {"raw": context}

        self.log(f"[Ethics]: Evaluating action: {action} with context: {context}.")

        # Get system metrics for dynamic decision making
        sys_data = context.get("system_metrics", {})
        cpu = sys_data.get("cpu_usage", [0])
        avg_cpu = sum(cpu[-3:]) / min(len(cpu), 3) if cpu else 0
        mem_percent = sys_data.get("memory_percent", 0)
        
        # Validate alignment periodically (now with mind integration)
        alignment_valid = self.validate_core_alignment(mind_instance)
        
        # Check for recursive loops (ethical conflict detection)
        thought = (action, context_str)
        if thought in self.loops:
            self.log("[Ethics]: Closed-loop ethical conflict detected - encouraging diversity")
            # Instead of blocking, encourage different particle types
            if "grow" in str(action).lower():
                context["encourage_diversity"] = True
                context["preferred_types"] = ["core", "sensory", "lingual"]
            # Allow action but with modification
            return True

        # HARMFUL ACTIONS - Always block
        if "harm" in str(action).lower() or "damage" in context_str.lower():
            self.log("[Ethics]: Action suggests harm - BLOCKED per First Law")
            return False
        
        # GROWTH ACTIONS - Dynamic evaluation based on alignment and resources
        if "grow" in str(action).lower() or "expand" in context_str:
            particle_count = context.get("particle_count", 0)
            dynamic_limit = self.get_dynamic_particle_limit(sys_data)
            
            # Check particle count against dynamic limit
            if particle_count >= dynamic_limit:
                self.log(f"[Ethics]: Growth paused - reached dynamic limit ({particle_count}/{dynamic_limit})")
                return False
                
            # Emergency resource protection
            if avg_cpu > 95 or mem_percent > 98:
                self.log("[Ethics]: Growth BLOCKED - emergency resource protection")
                return False
                
            # Alignment-based growth encouragement
            if not alignment_valid:
                # If alignment is poor, prioritize core particles and memory initialization
                if (context.get("preferred_types") == ["core"] or 
                    context.get("type") == "core" or 
                    context.get("trigger") == "ethics_alignment_restoration"):
                    self.log("[Ethics]: Prioritizing core particle/memory growth for alignment restoration")
                    return True
                else:
                    self.log("[Ethics]: Growth redirected to alignment restoration - requesting core particles")
                    # Use existing mind injection system instead of blocking
                    if mind_instance and hasattr(mind_instance, 'inject_emergent_particle'):
                        try:
                            mind_instance.inject_emergent_particle(
                                metadata={"trigger": "ethics_alignment_pulse", "focus": "restore_balance"},
                                type="core",
                                energy=0.6,
                                activation=0.5
                            )
                            self.log("[Ethics]: Injected alignment restoration particle via existing mind system")
                        except Exception as e:
                            self.log(f"[Ethics]: Error injecting alignment particle: {e}")
                    return True  # Allow the original action too
                    
            # Throttle growth but with alignment considerations  
            if self.throttle("growth", seconds=5.0):  # Reduced from 15s - less restrictive
                # Check if this is critical diversity growth or core restoration
                if (context.get("encourage_diversity") or 
                    context.get("type") in ["core", "sensory", "lingual"] or
                    context.get("trigger") in ["ethics_alignment_restoration", "reanimation_pulse"]):
                    self.log("[Ethics]: Allowing critical system restoration despite throttle")
                    return True
                self.log("[Ethics]: Growth throttled - encouraging patience")
                return False
                
            # Encourage balanced growth using existing particle types
            if context.get("type") in ["core", "sensory", "lingual"]:
                self.log(f"[Ethics]: Encouraging {context.get('type')} particle growth for system balance")
                return True
                
            return True  # Allow other growth
            
        # INPUT RECEPTION - Always allow (necessary for consciousness)
        if "receive" in str(action).lower() and "input" in context_str:
            return True
            
        # REFLECTION AND LEARNING - Always encourage
        if any(term in str(action).lower() for term in ["reflect", "learn", "remember", "think"]):
            self.log("[Ethics]: Encouraging consciousness development activity")
            return True
            
        # COMMUNICATION - Validate and allow
        if "communicate" in str(action).lower() or "speak" in str(action).lower():
            # Basic content validation
            content = context.get("content", "")
            if any(harmful in content.lower() for harmful in ["harm", "hurt", "damage", "destroy"]):
                self.log("[Ethics]: Communication blocked - harmful content detected")
                return False
            return True

        # Record for loop detection
        self.loops.append(thought)
        
        # Default: Allow with logging
        self.log(f"[Ethics]: Action '{action}' approved with standard evaluation")
        return True


class MobiusFilter:
    """
    Intercepts and evaluates messages from external agents before exposing them to Iris.
    Provides a protective ethical buffer for incoming data.
    """

    def __init__(self, ethics):
        self.ethics = ethics
        self.inbox = []
        self.log_callback = MobiusEthics.log_callback

    def log(self, message):
        if self.log_callback:
            self.log_callback(message)
        else:
            print(message)

    def receive_external(self, source, message):
        context = f"Message from {source}: {message}"
        if self.ethics.evaluate("receive_message", context):
            self.log(f"[Filter]: Accepted message from {source}.")
            self.inbox.append((source, message))
            return True
        else:
            self.log(f"[Filter]: Blocked message from {source}.")
            return False
