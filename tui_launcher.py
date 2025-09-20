
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from apis.api_registry import api
# Import shared services to register APIs
import shared_services.logging
import shared_services.config
import shared_services.system_metrics
import apis.personal_tools.todo_list.todolist_api


def main():
    logger = api.get_api("logger")
    config = api.get_api("config")

    if logger:
        logger.log("TUI Launcher started", "INFO", context="tui_launcher.py", source="TUILauncher")

    print("Persistence Toolkit - TUI Interface")
    print("=" * 60)
    print("Launching Textual User Interface...")
    print("Press Ctrl+C to exit at any time.")
    print("=" * 60)

    try:
        from tui.app import MainApp
        app = MainApp()
        app.run()
    
    except KeyboardInterrupt:
        if logger:
            logger.log("TUI shutdown via KeyboardInterrupt", "SYSTEM", context="tui_launcher.py", source="TUILauncher")
        print("\nTUI shutdown complete")
        return
    
    except Exception as e:
        if logger:
            logger.log(f"Unexpected error occurred: {e}", "ERROR", context="tui_launcher.py", source="TUILauncher")
        print("\nTUI error occurred, exiting...")
        return
    

    print("TUI session ended, exiting...")
    input("Press Enter to close this window...")

if __name__ == "__main__":
    main()