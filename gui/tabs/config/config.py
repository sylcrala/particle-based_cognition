"""
Particle-based Cognition Engine - GUI config tab utilities
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

from PyQt6.QtWidgets import QWidget, QStackedLayout, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QLineEdit, QMessageBox
import os
from apis.api_registry import api
config_api = api.get_api("config")

class ConfigTab(QWidget):
    def __init__(self):
        super().__init__()

        self.config = config_api

        # set up layouts
        self.base_layout = QHBoxLayout()
        self.setLayout(self.base_layout)

        self.content_layout = QStackedLayout()
        self.nav_layout = QVBoxLayout()
        self.base_layout.addLayout(self.content_layout, stretch=4)
        self.base_layout.addLayout(self.nav_layout, stretch=1)

        # set up content area
        self.content_layout.addWidget(self.system_settings_widget())
        self.content_layout.addWidget(self.agent_settings_widget())
        self.content_layout.addWidget(self.llm_settings_widget())
        self.content_layout.addWidget(self.adaptive_engine_settings_widget())

        # set up navigation buttons
        self.nav_buttons = []
        self.add_nav_button("System Settings", 0)
        self.add_nav_button("Agent Settings", 1)
        self.add_nav_button("LLM Settings", 2)
        self.add_nav_button("Adaptive Engine Settings", 3)

    def add_nav_button(self, label, index):
        button = QPushButton(label)
        button.clicked.connect(lambda: self.content_layout.setCurrentIndex(index))
        self.nav_layout.addWidget(button)
        self.nav_buttons.append(button)

    def system_settings_widget(self):
        """Creates widget for system settings"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # System Language
        lang_label = QLabel("System Language:")
        lang_input = QLineEdit(self.config.system_language)
        layout.addWidget(lang_label)
        layout.addWidget(lang_input)

        # os-specific settings
        os_label = QLabel("Operating System:")
        os_name = QLineEdit(self.config.os_name)
        os_version = QLineEdit(self.config.os_version)
        wayland_label = QLabel("Wayland Active:")
        wayland_input = QLineEdit(str(self.config.wayland_active))
        layout.addWidget(os_label)
        layout.addWidget(os_name)
        layout.addWidget(os_version)
        layout.addWidget(wayland_label)
        layout.addWidget(wayland_input)

        # user name
        user_label = QLabel("User Name:")
        user_input = QLineEdit(self.config.user_name)
        layout.addWidget(user_label)
        layout.addWidget(user_input)

        # agent mode
        mode_label = QLabel("Agent Mode:")
        mode_input = QLineEdit(self.config.agent_mode)
        mode_options = QLabel("Options: llm-extension, cog-growth")
        layout.addWidget(mode_label)
        layout.addWidget(mode_input)
        layout.addWidget(mode_options)

        # agent name
        name_label = QLabel("Agent Name:")
        name_input = QLineEdit(self.config.agent_name)
        name_input.setReadOnly(True)
        name_info = QLabel("Note: Agent name is auto-set based on agent mode.")
        layout.addWidget(name_label)
        layout.addWidget(name_input)
        layout.addWidget(name_info)

        return widget
    
    def agent_settings_widget(self):
        """Creates widget for agent settings"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # display agent config
        agent_config = self.config.get_agent_config()
        for key, value in agent_config.items():
            label = QLabel(f"{key}: ")
            value = QLineEdit(str(value))
            layout.addWidget(label)
            layout.addWidget(value)

        return widget
    
    def llm_settings_widget(self):
        """Creates widget for llm settings"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # display llm config
        llm_config = self.config.get_llm_config()
        for key, value in llm_config.items():
            label = QLabel(f"{key}: ")
            value = QLineEdit(str(value))
            layout.addWidget(label)
            layout.addWidget(value)

        return widget

    def adaptive_engine_settings_widget(self):
        """Creates widget for adaptive engine settings"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # display agent config
        ae_config = self.config.get_adaptive_engine_config()
        for key, value in ae_config.items():
            label = QLabel(f"{key}: ")
            value = QLineEdit(str(value))
            layout.addWidget(label)
            layout.addWidget(value)

        return widget
    
    def save_settings(self):
        """Saves the current settings - TODO"""
        pass
