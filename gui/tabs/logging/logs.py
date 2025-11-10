"""
Particle-based Cognition Engine - GUI log tab utilities
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

from apis.api_registry import api
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QStackedLayout
)
from PyQt6.QtGui import QPalette
from PyQt6.QtCore import QTimer

class LogsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # foundation
        self.base_layout = QVBoxLayout()
        self.setLayout(self.base_layout)
        self.bar_layout = QHBoxLayout()
        self.base_layout.addLayout(self.bar_layout, stretch=1)
        self.content_layout = QHBoxLayout()
        self.base_layout.addLayout(self.content_layout, stretch=10)
        
        # set up content area 
        from gui.tabs.logging.utils.file_explorer import FileExplorer
        from gui.tabs.logging.utils.file_viewer import FileViewer
        from gui.tabs.logging.utils.logstream import LogStream
        self.log_dir_viewer_layout = QVBoxLayout()
        self.log_dir_viewer = FileExplorer(parent=self)
        self.log_dir_viewer_layout.addWidget(self.log_dir_viewer)
        self.file_viewer_layout = QStackedLayout()
        self.file_viewer = FileViewer(parent=self)
        self.logstream = LogStream(parent=self)
        self.file_viewer_layout.addWidget(self.logstream)
        self.file_viewer_layout.addWidget(self.file_viewer)
        self.file_viewer_layout.setCurrentWidget(self.file_viewer)
        self.content_layout.addLayout(self.log_dir_viewer_layout, stretch=3)
        self.content_layout.addLayout(self.file_viewer_layout, stretch=7)

        # set up bar
        self.clear_button = QPushButton("Clear Viewer")
        self.clear_button.clicked.connect(self.clear_viewer)
        self.bar_layout.addWidget(self.clear_button)
        
        self.refresh_button = QPushButton("Toggle Live Stream")
        self.refresh_button.clicked.connect(self.toggle_live_logs)
        self.bar_layout.addWidget(self.refresh_button)


    def toggle_live_logs(self):
        """Switches to the live log stream view."""
        if self.file_viewer_layout.currentWidget() == self.file_viewer:
            self.file_viewer_layout.setCurrentWidget(self.logstream)
        else:
            self.file_viewer_layout.setCurrentWidget(self.file_viewer)

    def clear_viewer(self):
        """Clears the log file viewer area."""
        self.file_viewer.text_area.clear()
        self.file_viewer_layout.setCurrentWidget(self.file_viewer)

    def load_log_file(self, file_path):
        """Loads the selected log file into the viewer area."""
        try:
            self.file_viewer.load_file(file_path)
        except Exception as e:
            print(f"Error loading log file: {e}")

