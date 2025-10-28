from PyQt6.QtWidgets import QApplication
from gui.gui import MainWindow
import sys
import threading
import asyncio
from apis.api_registry import api
from shared_services import config
from shared_services import logging
from shared_services import system_metrics
from apis.personal_tools.todo_list import todolist_api

from apis.agent import core

def setup_x11_environment():
    """Force XWayland mode for better NVIDIA OpenGL support"""
    import os
    
    # Force Qt6 to use X11 backend (runs in XWayland under Wayland)
    os.environ["QT_QPA_PLATFORM"] = "xcb"
    
    # Force X11 for other toolkits too
    os.environ["GDK_BACKEND"] = "x11"
    os.environ["SDL_VIDEODRIVER"] = "x11"
    
    # NVIDIA-specific optimizations for XWayland
    os.environ["__GL_VENDOR_LIBRARY_NAME"] = "nvidia"
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    os.environ["__GL_SYNC_TO_VBLANK"] = "1"
    
    # Enable direct rendering through XWayland
    os.environ["LIBGL_ALWAYS_INDIRECT"] = "0"
    
    # Force OpenGL instead of GLES
    os.environ["QT_OPENGL"] = "desktop"


def initialize_agent_safe():
    """Thread-safe agent initialization"""
    try:
        # Import and initialize agent
        print("Initializing agent systems...")
        agent_core = api.get_api("agent")  # Already registered in core.startup()
        
        if agent_core:
            print("Starting agent async loop...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            loop.run_until_complete(agent_core.run())
            print("Agent async loop started.")
        else:
            print("ERROR: Agent core not found in API registry")
            
    except Exception as e:
        print(f"Error in agent initialization: {str(e)}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    config = api.get_api("config")
    agent_core = api.get_api("agent")
    
    if config.wayland_active:
        print("Wayland detected, applying compatibility fixes")
        # Setup X11 environment for better OpenGL support
        setup_x11_environment()

    try:
        # Create Qt application
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()

        app.processEvents() # ensure gui shows prior to cog framework startup

        # Start agent in separate thread after GUI is ready
        agent_thread = threading.Thread(
            target=initialize_agent_safe, 
            daemon=True
        )
        agent_thread.start()

        # Run Qt event loop
        sys.exit(app.exec())
    
    except KeyboardInterrupt as e:
        print("Shutting down application...")
        api.handle_shutdown()
        sys.exit(0)
    
    except Exception as e:
        print(f"Fatal error in main application: {str(e)}")
        import traceback
        print(f"Full traceback:\n{traceback.print_exc()}")
        sys.exit(1)