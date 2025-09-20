"""
central API 
"""
import asyncio


class APIRegistry:
    def __init__(self):
        self.apis = {}

    
    def register_api(self, name, api, permissions = None, user_only = False):
        self.apis[name] = {
            "api": api,
            "permissions": set(permissions or []),
            "user_only": user_only
        }
    

    def get_api(self, name):
        return self.apis.get(name, {}).get("api")
    

    def is_user_only(self, name):
        return self.apis.get(name, {}).get("user_only", False)
    

    def list_apis(self):
        return list(self.apis.keys())
    

    def call_api(self, name, method, *args, user_initiated = False, **kwargs):

        if self.is_user_only(name) and not user_initiated:
            raise PermissionError(f"API '{name}' requires user initiation.")
        
        api = self.get_api(name)

        if not api or not hasattr(api, method):
            raise AttributeError(f"API '{name}' does not have method '{method}'.")
        
        method_obj = getattr(api, method)
        
        if asyncio.iscoroutine(method_obj):
            return method_obj(*args, **kwargs)
        else:
            return method_obj(*args, **kwargs)

    async def handle_shutdown(self):
        """
        Graceful cognitive shutdown sequence to prevent amnesia and preserve agent state
        """
        try:
            print("Initiating graceful cognitive shutdown...")

            # Phase 1: Save critical memory state
            if "memory_bank" in self.apis:
                memory = self.get_api("memory_bank")
                if memory and hasattr(memory, 'emergency_save'):
                    print("Performing emergency memory save...")
                    memory.emergency_save()

            # Phase 2: Preserve particle field state
            if "particle_field" in self.apis:
                field = self.get_api("particle_field")
                if field and hasattr(field, 'save_field_state'):
                    print("Saving particle field state...")
                    field.save_field_state()

            # Phase 3: Save adaptive learning progress
            if "adaptive_engine" in self.apis:
                adaptive = self.get_api("adaptive_engine")
                if adaptive and hasattr(adaptive, 'save_learning_state'):
                    print("Preserving learning adaptations...")
                    adaptive.save_learning_state()
            
            # Phase 4: System state snapshot
            if "logger" in self.apis:
                logger = self.get_api("logger")
                if logger:
                    logger.log("System graceful shutdown completed", "INFO", "APIRegistry", "shutdown")

            # Phase 5: Final shutdown
            if "cognition_loop" in self.apis:
                cognition = self.get_api("cognition_loop")
                cognition.conscious_active = False

            print("Cognitive shutdown sequence completed successfully")
            
        except Exception as e:
            print(f"Error during graceful shutdown: {e}")
            # Emergency fallback - still try to save what we can
            try:
                if "memory_bank" in self.apis:
                    memory = self.get_api("memory_bank")
                    if memory and hasattr(memory, 'force_save'):
                        memory.force_save()
                        print("Emergency memory save completed")
            except:
                print("Critical: Could not perform emergency save")
    """    
    def register_shutdown_signal_handlers(self):
        # **DISABLED** - conflicts with TUI signal handling (causes dirty shutdowns - hanging textual mouse tracking - etc)
        # Register system signal handlers for graceful shutdown on Ctrl+C, etc.
        
        import signal
        import sys
        
        def signal_handler(signum, frame):
            print(f"\nReceived shutdown signal ({signum})")
            self.handle_shutdown()
            sys.exit(0)
        
        # Register handlers for common shutdown signals
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)  # Termination request
        
        print("Shutdown signal handlers registered")
    """
    async def handle_startup_restoration(self):
        """
        Restore cognitive state from previous shutdown to maintain continuity
        """
        try:
            print("Initiating cognitive state restoration...")
            
            # Phase 1: Restore memory state (field state from database)
            if "memory_bank" in self.apis:
                memory = self.get_api("memory_bank")
                if memory and hasattr(memory, 'restore_field_state'):
                    print("Restoring field state from memory database...")
                    field_data = memory.restore_field_state()
                    
                    # Phase 2: Apply restored field state to particle field
                    if field_data and "particle_field" in self.apis:
                        field = self.get_api("particle_field")
                        if field and hasattr(field, 'restore_from_state'):
                            print("Reconstructing particle field...")
                            await field.restore_from_state(field_data)
            
            # Phase 3: Restore adaptive learning state
            if "adaptive_engine" in self.apis:
                adaptive = self.get_api("adaptive_engine")
                if adaptive and hasattr(adaptive, 'restore_learning_state'):
                    print("Restoring learning adaptations...")
                    adaptive.restore_learning_state()
            
            # Phase 4: Notify consciousness system of restoration
            if "consciousness_loop" in self.apis:
                consciousness = self.get_api("consciousness_loop")
                if consciousness and hasattr(consciousness, 'on_state_restored'):
                    print("Notifying consciousness of restoration...")
                    consciousness.on_state_restored()
            
            print("Cognitive state restoration completed successfully")
            
        except Exception as e:
            print(f"Error during state restoration: {e}")
            print("Starting with fresh cognitive state...")

api = APIRegistry()

