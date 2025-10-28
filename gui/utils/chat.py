from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLineEdit, QTextEdit, QPushButton, QLabel
from PyQt6.QtCore import Qt
from apis.api_registry import api
import asyncio

class ChatTab(QWidget):
    """Class dedicated to the chat interface and related utilities"""
    def __init__(self):
        super().__init__()
    
        # set up layouts
        self.base_layout = QHBoxLayout()
        self.setLayout(self.base_layout)
        self.chat_layout = QVBoxLayout()
        self.chat_input_layout = QHBoxLayout()
        self.base_layout.addLayout(self.chat_layout, stretch=4)
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
        # TODO
        self.utility_layout.addWidget(QLabel("TODO: finish this"))
        self.utility_layout.addWidget(QLabel("maybe ability to view chat history?"))
        self.utility_layout.addWidget(QLabel("or configure chat settings?"))
        self.utility_layout.addWidget(QLabel("also don't forget to finish the chat tab, send_message isn't tied together"))


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
        