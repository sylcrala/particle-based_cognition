#!/usr/bin/env python3
"""
Main entry point for Quantum Cognitive System with integrated TUI console
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
from shared_services import config
from shared_services import logging



# Global log queue for TUI communication
logger = api.get_api("logger")

def log_to_console(message, level="INFO", context = None):
    """Send log messages to both system logger and TUI console"""

    if context != None:
        context = context
    else:
        context = "no_context"

    if logger:
        logger.log(message, level, context=context, source="MainApplication")



def launch_tui_terminal():
    current_dir = Path(__file__).parent
    tui_script = "./tui_launcher.py"

    try:
        if sys.platform == "win32":
            subprocess.Popen([
                "cmd", "/c", "start", "cmd", "/k", f"cd /d {current_dir} && python {tui_script}"
            ])
            log_to_console("Launched TUI in new terminal window (Windows)", "INFO")

        elif sys.platform == "darwin":
            subprocess.Popen([
                "osascript", "-e",
                f'tell application "Terminal" to do script "cd {current_dir} && python3 {tui_script}"'
            ])
            log_to_console("Launched TUI in new terminal window (macOS)", "INFO")
        
        elif sys.platform == "linux":
            try:
                subprocess.Popen([
                    "gnome-terminal", "--title=Persistence Toolkit",
                    "--", "python", str(tui_script)
                ])
                log_to_console("Launched TUI in new terminal window using gnome-terminal (Linux)", "INFO")

            except FileNotFoundError:
                subprocess.Popen([
                    "xterm", "-title", "Persistence Toolkit", "-e", f"python {tui_script}"
                ])
                log_to_console("Launched TUI in new terminal window using xterm (Linux)", "INFO")

        return True
    
    except Exception as e:
        log_to_console(f"Failed to launch TUI terminal: {e}", "ERROR")
        return False

async def main():
    """Main orchestrator for cognitive system + TUI"""

    print("Starting Persistence Toolkit, all systems initializing...")
    print("=" * 60)
    print("This terminal is dedicated for the cognition framework and system logs.")
    print("Please do not close this terminal to ensure proper operation.")


    print("=" * 60)
    log_to_console(f"TUI startup initializing...", "INFO", "main()")
    print("Launching TUI console in separate terminal...")
    try:
        # Launch TUI in separate terminal window
        #time.sleep(30)
        if launch_tui_terminal():
            log_to_console(f"TUI startup complete, initializing cognitive systems...", "INFO", "main()")
            print("TUI launched successfully in new terminal window.")
            #time.sleep(2)  # Give TUI time to initialize
        else:
            log_to_console("TUI launch failed, launching TUI in main terminal (likely blocking the cognition framework)", "WARNING", "main()")
            from tui.app import MainApp
            app = MainApp()
            app.run()
            return
    except Exception as e:
        log_to_console(f"Error launching TUI: {e}", "ERROR", "main()")
        print(f"Error launching TUI: {e}")
        return



    # Start cognitive systems initialization asynchronously
    try:
        print("=" * 60)
        print("Beginning cognitive systems initialization...")
        log_to_console("Beginning cognitive systems initialization...", "INFO", "main()")
        
        try:
            print("Importing and registering agent core...")
            from apis.agent.core import AgentCore

            agent_core = AgentCore()

            api.register_api("agent", agent_core)

            print("Agent core initialized successfully.")
            log_to_console("Agent core initialized successfully.", "INFO", "main()")
        except Exception as e:
            log_to_console(f"Error initializing cognition framework APIs: {e}", "ERROR", "main()")
            print(f"Error initializing cognition framework APIs: {e}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")
            return

        print("=" * 60)

        try:    
            status = await agent_core.cognition_loop.get_status()

            print("Final systems check: ")
            print(f"Registered APIs: {api.list_apis()}")
            log_to_console(f"Registered APIs: {api.list_apis()}", "INFO", "main()")
            print(f"Agent Status: {status}")

        except Exception as e:
            log_to_console(f"Error during final agent initialization: {e}", "ERROR", "main()")
            print(f"Error during final agent initialization: {e}")
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")

        log_to_console("Cognitive systems initialization complete", "SUCCESS", "main()")
        print("Main terminal ready for monitoring. Press Ctrl+C to exit gracefully.")

        try:
            await agent_core.run() # starting cognition loop lastly to ensure console doesn't close
        except Exception as e:
            log_to_console(f"Error during final agent initialization: {e}", "ERROR", "main()")
            print(f"Error during final agent initialization: {e}")            
            import traceback
            print(f"Full traceback:\n{traceback.format_exc()}")

    except Exception as e:
        log_to_console(f"Error during cognitive systems initialization: {e}", "ERROR", "main()")
        print(f"Error during cognitive systems initialization: {e}")            
        import traceback
        print(f"Full traceback:\n{traceback.format_exc()}")






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
