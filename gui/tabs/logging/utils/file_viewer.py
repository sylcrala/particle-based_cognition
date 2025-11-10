"""
Particle-based Cognition Engine - GUI log tab utilities - file viewer widget
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
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTextEdit
from PyQt6.QtGui import QPalette

class FileViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # foundation
        self.base_layout = QVBoxLayout()
        self.setLayout(self.base_layout)
        
        # text area to display file content
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.base_layout.addWidget(self.text_area)
        
    def load_file(self, file_path):
        """Load and display the content of the specified file."""
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                self.text_area.setPlainText(content)
        except Exception as e:
            self.text_area.setPlainText(f"Error loading file: {e}")