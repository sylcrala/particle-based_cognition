"""
Particle-based Cognition Engine - GUI chat tab utilities
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

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QTextEdit, QPushButton, QLabel
from PyQt6.QtCore import Qt
from apis.api_registry import api
import asyncio
from pathlib import Path

class ChatTab(QWidget):
    """Class dedicated to the chat interface and related utilities"""
    def __init__(self):
        super().__init__()

        self.config = api.get_api("config")
    
        # set up layouts
        self.base_layout = QHBoxLayout()
        self.setLayout(self.base_layout)
        self.chat_layout = QVBoxLayout()
        self.chat_input_layout = QHBoxLayout()
        self.base_layout.addLayout(self.chat_layout, stretch=5)
        self.utility_layout = QVBoxLayout()
        self.base_layout.addLayout(self.utility_layout, stretch=1)

        # set up chat components
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_input = QLineEdit()
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)

        self.chat_layout.addWidget(self.chat_display)
        self.chat_layout.addLayout(self.chat_input_layout)
        self.chat_input_layout.addWidget(self.chat_input)
        self.chat_input_layout.addWidget(self.send_button)

        # set up utility components
        self.utility_layout.addWidget(QLabel("<b>Utilities</b>"))
        self.export_btn = QPushButton("Export Chat History")
        self.export_btn.clicked.connect(self.export_chat_history)
        self.utility_layout.addWidget(self.export_btn)

        self.clear_btn = QPushButton("Clear Chat")
        self.clear_btn.clicked.connect(self.clear_chat)
        self.utility_layout.addWidget(self.clear_btn)
        


    def clear_chat(self):
        """Clears the chat display"""
        self.chat_display.clear()

    def export_chat_history(self):
        """Exports the current chat history to a text file"""
        try:
            chat_history = self.chat_display.toPlainText()
            session = self.config.session_id
            filename = f"chat_history_{session}.txt"
            filepath = Path(self.config.export_path / filename)
            with open(filepath, "w") as f:
                f.write(chat_history)
            self.chat_display.append(f"<b>Chat history exported to {filepath} </b>")
        except Exception as e:
            self.chat_display.append(f"<b>Error exporting chat history:</b> {str(e)}")

    def send_message(self):
        """Handles sending a message from the input box to the agent and displaying the response"""
        events = api.get_api("_agent_anchor")
        
        user_message = self.chat_input.text()
        if not user_message.strip():
            return  # ignore empty messages
        self.chat_display.append(f"<b>You:</b> {user_message}")

        try:
            # Send message to agent and get response 
            response = events.send_message(user_message, source="gui_chat")
            self.chat_display.append(f"<b>Agent:</b> {response}")
        except Exception as e:
            self.chat_display.append(f"<b>Error:</b> {str(e)}")
        