"""
Particle-based Cognition Engine - GUI log tab utilities - file explorer widget
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
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTreeView, QFileDialog
from PyQt6.QtGui import QPalette, QFileSystemModel
from PyQt6.QtCore import QModelIndex, QDir


class FileExplorer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # foundation
        self.base_layout = QVBoxLayout()
        self.setLayout(self.base_layout)
        self.parent_class = parent
        
        # pull config info
        self.config = api.get_api("config")
        self.logging_config = self.config.get_logging_config()
        self.logs_dir = self.logging_config.get("session_log_dir")

        # set up file explorer view
        self.file_model = QFileSystemModel()
        self.file_model.setRootPath(self.logs_dir)

        self.tree_view = QTreeView()
        self.tree_view.setModel(self.file_model)
        self.tree_view.setRootIndex(self.file_model.index(self.logs_dir))
        self.tree_view.clicked.connect(self.on_file_clicked)
        self.base_layout.addWidget(self.tree_view)

    def on_file_clicked(self, index: QModelIndex):
        """Handle file click events to load the selected file in the viewer."""
        file_path = self.file_model.filePath(index)
        self.parent_class.load_log_file(file_path)




