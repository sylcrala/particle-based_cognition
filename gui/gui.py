"""
Particle-based Cognition Engine - GUI application module
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

import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWidgets import (
    QPushButton, 
    QMainWindow, 
    QStackedLayout, 
    QStackedWidget, 
    QHBoxLayout, 
    QVBoxLayout, 
    QGridLayout, 
    QWidget, 
    QGroupBox,
    QLabel,
)
from PyQt6.QtCore import Qt, QMimeData
from PyQt6.QtGui import QDrag, QPalette

#rom apis.api_registry import api
#from shared_services import logging

# TODO:
## 1. build out GUI frame/layout
#### - rebuild main window
#### - rebuild visualizer
#### - rebuild chat window
###### - add option to export chat logs
###### - add option to view previous chat sessions (no "loading" - all sessions are contextually persistent in memory)
#### - clean up design through rebuild

## 2. build diagnostics APIs (modularized, able to be called independently from anywhere if needed) 
#### - log viewer (with detailed filtering options)
#### - system health monitor (CPU, memory, disk, network, etc)
#### - various tests able to be ran if cognition sys is online
#### - etc

## 3. add functionality to interact with APIs

## 4. add functionality to visualize and analyze data (graphs, charts, etc) (diagnostics)

## 5. add functionality to export data (CSV, JSON, etc) (diagnostics)

class MainWindow(QMainWindow):
    """Main window class"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Particle-Based Cognition Engine - Main Window")
        self.resize(1280, 720)

        # set up layout and widget foundation
        self.central_widget = QWidget() # foundational widget
        self.setCentralWidget(self.central_widget)
        self.base_layout = QVBoxLayout() # main vertical layout to place tab bar above main stacked layout
        self.central_widget.setLayout(self.base_layout)
        self.palette = QPalette()
        self.palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.lightGray)
        self.central_widget.setPalette(self.palette)
        self.central_widget.setAutoFillBackground(True)

        self.tabbar = QWidget() # tab bar container
        self.tabbar_layout = QHBoxLayout() # small tab bar at the top for nav
        self.tabbar.setLayout(self.tabbar_layout)
        self.main_widget = QStackedWidget() # main widget container

        self.base_layout.addWidget(self.tabbar) # adding to base layout
        self.base_layout.addWidget(self.main_widget) # adding to base layout

        # set up tabs for stacked layout - tab index listed next to each
        from gui.tabs.visualizer.visualizer import VisualizerTab
        from gui.tabs.chat.chat import ChatTab
        from gui.tabs.config.config import ConfigTab
        self.diagnostics_tab = QWidget() #0
        self.main_widget.addWidget(self.diagnostics_tab)
        self.analytics_tab = QWidget() #1
        self.main_widget.addWidget(self.analytics_tab)
        self.memory_tab = QWidget() #2
        self.main_widget.addWidget(self.memory_tab)
        self.visualizer_tab = VisualizerTab() #3
        self.main_widget.addWidget(self.visualizer_tab)
        self.chat_tab = ChatTab() #4
        self.main_widget.addWidget(self.chat_tab)
        self.logs_tab = QWidget() #5
        self.main_widget.addWidget(self.logs_tab)
        self.config_tab = ConfigTab() #6
        self.main_widget.addWidget(self.config_tab)

        self.main_widget.setCurrentIndex(3) # default to visualization tab at launch

        # set up tab buttons
        self.diagnostics_tab_btn = QPushButton("Diagnostics")
        self.diagnostics_tab_btn.pressed.connect(lambda: self.switch_tab(0))
        self.analytics_tab_btn = QPushButton("Analytics")
        self.analytics_tab_btn.pressed.connect(lambda: self.switch_tab(1))
        self.memory_tab_btn = QPushButton("Memories")
        self.memory_tab_btn.pressed.connect(lambda: self.switch_tab(2))
        self.visualizer_tab_btn = QPushButton("Visualizer")
        self.visualizer_tab_btn.pressed.connect(lambda: self.switch_tab(3))
        self.chat_tab_btn = QPushButton("Chat")
        self.chat_tab_btn.pressed.connect(lambda: self.switch_tab(4))
        self.logs_tab_btn = QPushButton("Log Viewer")
        self.logs_tab_btn.pressed.connect(lambda: self.switch_tab(5))
        self.config_tab_btn = QPushButton("Configuration")
        self.config_tab_btn.pressed.connect(lambda: self.switch_tab(6))

        self.tabbar_layout.addWidget(self.diagnostics_tab_btn)
        self.tabbar_layout.addWidget(self.analytics_tab_btn)
        self.tabbar_layout.addWidget(self.memory_tab_btn)
        self.tabbar_layout.addWidget(self.visualizer_tab_btn)
        self.tabbar_layout.addWidget(self.chat_tab_btn)
        self.tabbar_layout.addWidget(self.logs_tab_btn)
        self.tabbar_layout.addWidget(self.config_tab_btn)

    def run(self):
        """Starts the main window"""
        self.show()

    def switch_tab(self, index: int):
        """Switches the main content area (stacked widget) to the given tab index"""
        try:
            self.main_widget.setCurrentIndex(index)
        except Exception as e:
            print(f"Error switching tab: {e}")
    


class DragButton(QPushButton):
    """A class dedicated to drag-and-dropping objects"""
    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.MouseButton.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            drag.setMimeData(mime)
            drag.exec(Qt.DropAction.MoveAction)


