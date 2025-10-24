from PyQt6.QtWidgets import QApplication
from gui.gui import MainWindow
import sys
from apis.api_registry import api
from shared_services import config
from shared_services import logging
from shared_services import system_metrics
from apis.personal_tools.todo_list import todolist_api


if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = MainWindow()
    window.run()
    sys.exit(QApplication.instance().exec())