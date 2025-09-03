#!/usr/bin/env python3
"""
Main entry point for Quantum Cognitive System with integrated TUI console
"""

import asyncio
import sys
import threading
from queue import Queue
from apis.api_registry import api

# Import shared services to register APIs
import shared_services.logging
import shared_services.config
import shared_services.system_metrics
import shared_services.system_events
import apis.personal_tools.todo_list.todolist_api

# Global log queue for TUI communication
logger = api.get_api("logger")

def log_to_console(message, level="INFO", context = None):
    """Send log messages to both system logger and TUI console"""
    import time

    if context != None:
        context = context
    else:
        context = "no_context"

    if logger:
        logger.log(message, level, context=context, source="MainApplication")

async def initialize_cognitive_systems():
    """Initialize all cognitive systems and log to TUI"""
    log_to_console("Quantum Cognitive System Starting...", "SYSTEM")
    log_to_console("=" * 50, "SYSTEM")
    from apis.agent import loop
    from apis.agent import event_handler
    loop.initialize_cognitive_systems()

    # Initialize system metrics first (foundational monitoring)
    system_metrics = api.get_api("system_metrics")
    if system_metrics:
        log_to_console("System metrics monitoring initialized", "SYSTEM")
    else:
        log_to_console("System metrics not available", "WARNING")
    
    # Initialize configuration
    config = api.get_api("config")
    if config:
        agent_config = config.get_agent_config()
        log_to_console(f"Configuration loaded for agent: {agent_config.get('name', 'Unknown')}", "SYSTEM")
    else:
        log_to_console("Configuration not available", "ERROR")
    
    # Register graceful shutdown handlers
    api.register_shutdown_signal_handlers()
    log_to_console("Shutdown signal handlers registered", "SYSTEM")
    

    # Restore previous cognitive state if available
    log_to_console("Checking for previous cognitive state...", "SYSTEM")
    await api.handle_startup_restoration()
    
    # Initialize all cognitive systems
    log_to_console("Initializing cognitive systems...", "SYSTEM")
    
    # Get core APIs
    memory_api = api.get_api("memory_bank")
    field_api = api.get_api("particle_field") 
    adaptive_api = api.get_api("adaptive_engine")
    model_handler_api = api.get_api("model_handler")
    
    log_to_console("Core systems status:", "SYSTEM")
    log_to_console(f"   Memory Bank: {'ONLINE' if memory_api else 'OFFLINE'}", "SYSTEM")
    log_to_console(f"   Particle Field: {'ONLINE' if field_api else 'OFFLINE'}", "SYSTEM")
    log_to_console(f"   Adaptive Engine: {'ONLINE' if adaptive_api else 'OFFLINE'}", "SYSTEM")
    log_to_console(f"   Model Handler: {'ONLINE' if model_handler_api else 'OFFLINE'}", "SYSTEM")
    log_to_console(f"   Event Handler: {'ONLINE' if event_handler else 'OFFLINE'}", "SYSTEM")
    log_to_console(f"   System Metrics: {'ONLINE' if system_metrics else 'OFFLINE'}", "SYSTEM")
    
    # Show current state
    if field_api:
        particles = field_api.get_all_particles()
        log_to_console(f"   Active particles: {len(particles)}", "SYSTEM")
    
    log_to_console("System Status: READY", "SUCCESS")
    log_to_console("   Use navigation menu or Ctrl+C for graceful shutdown", "INFO")
    
    return {"memory": memory_api, "field": field_api, "adaptive": adaptive_api, "events": event_handler}

def run_tui():
    """Run the TUI in a separate thread"""
    try:
        from tui.app import MainApp

        # Create TUI app with logger access
        app = MainApp()
        app.run()
        
        
    except Exception as e:
        print(f"TUI Error: {e}")


async def main():
    """Main orchestrator for cognitive system + TUI"""
    try:
        # Initialize system event handler (this will auto-register agent handler)
        system_event_handler = api.get_api("system_event_handler")
        if system_event_handler:
            await system_event_handler.initialize()
            log_to_console("System event handling initialized", "SYSTEM")
        
        # Start Cognitive Loop in separate thread
        #cog_thread = threading.Thread(target=await initialize_cognitive_systems, daemon=True)
        #cog_thread.start()

        # Initialize cognitive systems
        #cognition = await initialize_cognitive_systems()

        # Start TUI in separate thread
        tui_thread = threading.Thread(target=run_tui, daemon=True)
        tui_thread.start()

        # Keep main thread alive and handle events
        if system_event_handler:
            # Wait for the master event loop to finish (shutdown will cancel it)
            await system_event_handler.master_task

    except KeyboardInterrupt:
        log_to_console("Graceful shutdown initiated...", "SYSTEM")
        # Use system event handler for coordinated shutdown
        system_event_handler = api.get_api("system_event_handler")
        if system_event_handler:
            await system_event_handler.shutdown()
        api.handle_shutdown()
    except Exception as e:
        log_to_console(f"Unexpected error: {e}", "ERROR")
        log_to_console("Attempting graceful shutdown...", "SYSTEM")
        api.handle_shutdown()

if __name__ == "__main__":
    try:
        # Run the main async function
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\nQuantum Cognitive System shutdown complete")
    except Exception as e:
        print(f"\nCritical error: {e}")
        sys.exit(1)
