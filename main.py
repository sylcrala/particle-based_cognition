"""
Particle-based Cognition Engine - Main entry point with toggleable 3D visualization
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

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from gui.gui import MainWindow
import sys
import threading
import asyncio
from apis.api_registry import api
from shared_services import config
from shared_services import logging
from shared_services import system_metrics
from apis.general_automation import todolist_api

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


def run_gui():
    """Run the GUI - returns the QApplication instance"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app, window

def cognition(agent_core = None):
    """Run the cognition engine in the main thread"""
    if agent_core:
        print("Starting agent in unified event loop...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(agent_core.run())
    else:
        print("ERROR: Agent core not found in API registry")

def main():
    config = api.get_api("config")
    agent_core = api.get_api("agent")
    
    if config.wayland_active:
        print("Wayland detected, applying compatibility fixes")
        setup_x11_environment()

    try:
        # Initialize GUI
        print("Starting GUI...")
        app, window = run_gui()
        app.processEvents()  
        
        print("Starting cognitive engine...")
        cognitive_thread = threading.Thread(
            target=cognition, 
            args=(agent_core,),  
            daemon=True, 
            name="CognitiveThread"
        )
        cognitive_thread.start()
    
        print("Running GUI event loop...")
        sys.exit(app.exec())
        
    except KeyboardInterrupt:
        print("Shutting down application...")
        try:
            asyncio.run(api.handle_shutdown())
        except:
            print("Error during shutdown handling")
            pass
        sys.exit(0)
        
    except Exception as e:
        print(f"Fatal error in main application: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()