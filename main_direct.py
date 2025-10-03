#!/usr/bin/env python3
"""
Main entry point for Cognition Framework (no GUI or TUI)
"""

import asyncio
import os
import sys
import time
import subprocess
import threading
from pathlib import Path
from queue import Queue
from apis.api_registry import api
from shared_services import config, logging




# Global log queue for TUI communication
logger = api.get_api("logger")

def log_to_console(message, level="INFO", context = None):
    """Send log messages to both system logger and TUI console"""

    if context != None:
        context = context
    else:
        context = "no_context"

    if logger:
        logger.log(message, level, context=context, source="MainApplication_NOINTERFACE")


async def main():
    """Main orchestrator for cognitive system + TUI"""

    print("Starting Persistence Toolkit, all systems initializing...")
    print("=" * 60)
    print("This terminal is dedicated for the cognition framework and system logs.")
    print("Please do not close this terminal to ensure proper operation.")
    try:
            log_to_console("Launching Agent Core", "WARNING", "main()")

            from apis.agent.core import AgentCore
            agent_core = AgentCore()
            
            # Register agent in shared API registry
            api.register_api("agent", agent_core)

            # Start agent in background task
            await agent_core.run()

            log_to_console("Misty's consciousness initialized successfully", "INFO", "main()")


    except Exception as e:
        log_to_console(f"Error launching Agent Core: {e}", "ERROR", "main()")
        print(f"Error launching Agent Core: {e}")
    return






if __name__ == "__main__":
    try:        
        asyncio.run(main())
        
    except KeyboardInterrupt:
        print("\nShutdown signal received, initiating graceful shutdown...")
        log_to_console("Shutdown signal received, initiating graceful shutdown...", "SYSTEM", "main()")
        asyncio.run(api.handle_shutdown())
        log_to_console("System shutdown complete. Goodbye!", "SUCCESS", "main()")
    except Exception as e:
        print(f"\nCritical error: {e}")
        log_to_console(f"Critical error: {e}", "ERROR", "main()")
        sys.exit(1)
