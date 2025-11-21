"""
Particle-based Cognition Engine - GUI utilities - Universal Particle Detail Popup
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

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
    QTableWidget, QTableWidgetItem, QGroupBox, QScrollArea,
    QGridLayout, QPushButton, QTabWidget, QSplitter, QHeaderView, QWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from apis.api_registry import api
import json
from datetime import datetime


class UniversalParticleDetailPopup(QDialog):
    """
    Universal particle detail popup that can be used throughout the entire GUI system.
    
    This provides a standardized way to view detailed particle information from any
    module or tab in the system. Simply import and call with a particle ID.
    
    Usage:
        from gui.utils.particle_popup import UniversalParticleDetailPopup
        popup = UniversalParticleDetailPopup(particle_id, parent)
        popup.show()
    """
    
    def __init__(self, particle_id, parent=None, agent=None):
        super().__init__(parent)
        self.particle_id = particle_id
        self.agent = agent or api.get_api("agent")
        self.particle_data = None
        
        self.setWindowTitle(f"Particle Details: {particle_id[:12]}...")
        self.setModal(False)  # Allow non-modal operation
        self.resize(800, 600)
        
        self.init_ui()
        self.load_particle_data()
        
    def init_ui(self):
        """Initialize the popup UI"""
        layout = QVBoxLayout(self)
        
        # Header with particle ID and basic info
        header_group = QGroupBox("Particle Information")
        header_layout = QGridLayout(header_group)
        
        self.particle_id_label = QLabel(f"ID: {self.particle_id}")
        self.particle_id_label.setFont(QFont("Courier", 10))
        header_layout.addWidget(QLabel("Particle ID:"), 0, 0)
        header_layout.addWidget(self.particle_id_label, 0, 1)
        
        self.particle_type_label = QLabel("Type: Loading...")
        header_layout.addWidget(QLabel("Type:"), 1, 0)
        header_layout.addWidget(self.particle_type_label, 1, 1)
        
        self.particle_energy_label = QLabel("Energy: Loading...")
        header_layout.addWidget(QLabel("Energy:"), 2, 0)
        header_layout.addWidget(self.particle_energy_label, 2, 1)
        
        self.particle_activation_label = QLabel("Activation: Loading...")
        header_layout.addWidget(QLabel("Activation:"), 3, 0)
        header_layout.addWidget(self.particle_activation_label, 3, 1)
        
        layout.addWidget(header_group)
        
        # Tabbed content for different aspects of particle data
        self.content_tabs = QTabWidget()
        
        # Tab 1: Position and Field Data
        self.position_tab = self._create_position_tab()
        self.content_tabs.addTab(self.position_tab, "Position & Field")
        
        # Tab 2: Metadata and Content
        self.metadata_tab = self._create_metadata_tab()
        self.content_tabs.addTab(self.metadata_tab, "Metadata & Content")
        
        # Tab 3: Interactions and Relations
        self.interactions_tab = self._create_interactions_tab()
        self.content_tabs.addTab(self.interactions_tab, "Interactions")
        
        # Tab 4: History and Evolution
        self.history_tab = self._create_history_tab()
        self.content_tabs.addTab(self.history_tab, "History")
        
        layout.addWidget(self.content_tabs)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("Refresh Data")
        self.refresh_button.clicked.connect(self.load_particle_data)
        button_layout.addWidget(self.refresh_button)
        
        button_layout.addStretch()
        
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
    def _create_position_tab(self):
        """Create position and field data tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 12D Position data
        position_group = QGroupBox("12-Dimensional Position")
        position_layout = QVBoxLayout(position_group)
        
        self.position_table = QTableWidget(12, 2)
        self.position_table.setHorizontalHeaderLabels(["Dimension", "Value"])
        self.position_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # Set dimension labels
        for i in range(12):
            self.position_table.setItem(i, 0, QTableWidgetItem(f"Dimension {i+1}"))
            self.position_table.setItem(i, 1, QTableWidgetItem("0.0"))
            
        position_layout.addWidget(self.position_table)
        layout.addWidget(position_group)
        
        # Field location data
        field_group = QGroupBox("Field Location")
        field_layout = QGridLayout(field_group)
        
        self.grid_location_label = QLabel("Grid Key: Unknown")
        self.spatial_index_label = QLabel("Spatial Index: Unknown")
        self.neighbors_label = QLabel("Nearby Particles: Unknown")
        
        field_layout.addWidget(QLabel("Grid Location:"), 0, 0)
        field_layout.addWidget(self.grid_location_label, 0, 1)
        field_layout.addWidget(QLabel("Spatial Index:"), 1, 0)
        field_layout.addWidget(self.spatial_index_label, 1, 1)
        field_layout.addWidget(QLabel("Neighbors:"), 2, 0)
        field_layout.addWidget(self.neighbors_label, 2, 1)
        
        layout.addWidget(field_group)
        
        return tab
        
    def _create_metadata_tab(self):
        """Create metadata and content tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Metadata table
        metadata_group = QGroupBox("Metadata")
        metadata_layout = QVBoxLayout(metadata_group)
        
        self.metadata_table = QTableWidget(0, 2)
        self.metadata_table.setHorizontalHeaderLabels(["Key", "Value"])
        self.metadata_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        metadata_layout.addWidget(self.metadata_table)
        layout.addWidget(metadata_group)
        
        # Content analysis
        content_group = QGroupBox("Content Analysis")
        content_layout = QVBoxLayout(content_group)
        
        self.content_analysis = QTextEdit()
        self.content_analysis.setMaximumHeight(200)
        content_layout.addWidget(self.content_analysis)
        
        layout.addWidget(content_group)
        
        return tab
        
    def _create_interactions_tab(self):
        """Create interactions and relations tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Connected particles
        connections_group = QGroupBox("Connected Particles")
        connections_layout = QVBoxLayout(connections_group)
        
        self.connections_table = QTableWidget(0, 4)
        self.connections_table.setHorizontalHeaderLabels([
            "Connected Particle", "Relation Type", "Strength", "Direction"
        ])
        self.connections_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.connections_table.cellDoubleClicked.connect(self.on_connection_double_click)
        
        connections_layout.addWidget(self.connections_table)
        layout.addWidget(connections_group)
        
        # Quantum entanglements
        entanglements_group = QGroupBox("Quantum Entanglements")
        entanglements_layout = QVBoxLayout(entanglements_group)
        
        self.entanglements_table = QTableWidget(0, 3)
        self.entanglements_table.setHorizontalHeaderLabels([
            "Entangled Particle", "Entanglement Type", "Strength"
        ])
        self.entanglements_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        entanglements_layout.addWidget(self.entanglements_table)
        layout.addWidget(entanglements_group)
        
        return tab
        
    def _create_history_tab(self):
        """Create history and evolution tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Creation info
        creation_group = QGroupBox("Creation Information")
        creation_layout = QGridLayout(creation_group)
        
        self.creation_time_label = QLabel("Created: Unknown")
        self.creator_label = QLabel("Creator: Unknown")
        self.creation_context_label = QLabel("Context: Unknown")
        
        creation_layout.addWidget(QLabel("Creation Time:"), 0, 0)
        creation_layout.addWidget(self.creation_time_label, 0, 1)
        creation_layout.addWidget(QLabel("Creator:"), 1, 0)
        creation_layout.addWidget(self.creator_label, 1, 1)
        creation_layout.addWidget(QLabel("Context:"), 2, 0)
        creation_layout.addWidget(self.creation_context_label, 2, 1)
        
        layout.addWidget(creation_group)
        
        # Evolution history
        evolution_group = QGroupBox("Evolution History")
        evolution_layout = QVBoxLayout(evolution_group)
        
        self.evolution_table = QTableWidget(0, 4)
        self.evolution_table.setHorizontalHeaderLabels([
            "Timestamp", "Event Type", "Description", "Impact"
        ])
        self.evolution_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        evolution_layout.addWidget(self.evolution_table)
        layout.addWidget(evolution_group)
        
        return tab
        
    def load_particle_data(self):
        """Load and display particle data"""
        try:
            if not self.agent:
                self._show_error("Agent not available")
                return
                
            # Try to get particle from field
            particle = None
            if (hasattr(self.agent, 'particle_field') and self.agent.particle_field):
                particle = self.agent.particle_field.get_particle_by_id(self.particle_id)
                
            if particle:
                self.particle_data = particle
                self._populate_particle_data()
            else:
                self._show_error("Particle not found in field")
                
        except Exception as e:
            self._show_error(f"Error loading particle data: {str(e)}")
            
    def _populate_particle_data(self):
        """Populate the UI with particle data"""
        if not self.particle_data:
            return
            
        try:
            # Update header info
            particle_type = getattr(self.particle_data, 'type', 'Unknown')
            particle_energy = getattr(self.particle_data, 'energy', 0.0)
            particle_activation = getattr(self.particle_data, 'activation', 0.0)
            
            self.particle_type_label.setText(f"Type: {particle_type}")
            self.particle_energy_label.setText(f"Energy: {particle_energy:.6f}")
            self.particle_activation_label.setText(f"Activation: {particle_activation:.6f}")
            
            # Populate position data
            self._populate_position_data()
            
            # Populate metadata
            self._populate_metadata()
            
            # Populate interactions (placeholder)
            self._populate_interactions()
            
            # Populate history (placeholder)
            self._populate_history()
            
        except Exception as e:
            self._show_error(f"Error populating data: {str(e)}")
            
    def _populate_position_data(self):
        """Populate position and field data"""
        try:
            position = getattr(self.particle_data, 'position', None)
            if position is not None:
                # Update 12D position table
                for i in range(min(12, len(position))):
                    value = f"{float(position[i]):.6f}" if i < len(position) else "0.0"
                    self.position_table.setItem(i, 1, QTableWidgetItem(value))
                    
            # Field location data (if available)
            # This would need integration with spatial indexing system
            self.grid_location_label.setText("Grid Key: Requires spatial grid integration")
            self.spatial_index_label.setText("Spatial Index: Requires field indexing")
            self.neighbors_label.setText("Neighbors: Requires proximity analysis")
            
        except Exception as e:
            self._show_error(f"Error populating position: {str(e)}")
            
    def _populate_metadata(self):
        """Populate metadata table"""
        try:
            metadata = getattr(self.particle_data, 'metadata', {}) or {}
            
            # Clear existing data
            self.metadata_table.setRowCount(0)
            
            # Populate metadata table
            for row, (key, value) in enumerate(metadata.items()):
                self.metadata_table.insertRow(row)
                self.metadata_table.setItem(row, 0, QTableWidgetItem(str(key)))
                self.metadata_table.setItem(row, 1, QTableWidgetItem(str(value)[:100]))  # Limit length
                
            # Content analysis
            content = metadata.get('content') or metadata.get('token') or 'No content'
            analysis = f"Content: {content}\\n\\n"
            analysis += f"Metadata keys: {len(metadata)}\\n"
            analysis += f"Content length: {len(str(content))}\\n"
            analysis += f"Has semantic data: {'Yes' if any('semantic' in k.lower() for k in metadata.keys()) else 'No'}\\n"
            
            self.content_analysis.setText(analysis)
            
        except Exception as e:
            self._show_error(f"Error populating metadata: {str(e)}")
            
    def _populate_interactions(self):
        """Populate interactions data (placeholder)"""
        # Clear tables
        self.connections_table.setRowCount(0)
        self.entanglements_table.setRowCount(0)
        
        # Add placeholder data
        self.connections_table.insertRow(0)
        self.connections_table.setItem(0, 0, QTableWidgetItem("Interaction analysis"))
        self.connections_table.setItem(0, 1, QTableWidgetItem("requires enhanced"))
        self.connections_table.setItem(0, 2, QTableWidgetItem("particle field"))
        self.connections_table.setItem(0, 3, QTableWidgetItem("integration"))
        
    def _populate_history(self):
        """Populate history data (placeholder)"""
        # Clear table
        self.evolution_table.setRowCount(0)
        
        # Creation info
        self.creation_time_label.setText("Created: Requires temporal tracking")
        self.creator_label.setText("Creator: Requires provenance system")
        self.creation_context_label.setText("Context: Requires context tracking")
        
        # Add placeholder evolution entry
        self.evolution_table.insertRow(0)
        self.evolution_table.setItem(0, 0, QTableWidgetItem(datetime.now().strftime('%H:%M:%S')))
        self.evolution_table.setItem(0, 1, QTableWidgetItem("Viewing"))
        self.evolution_table.setItem(0, 2, QTableWidgetItem("Particle details viewed in popup"))
        self.evolution_table.setItem(0, 3, QTableWidgetItem("Informational"))
        
    def on_connection_double_click(self, row, column):
        """Handle double-click on connection to open another particle popup"""
        try:
            connected_particle_item = self.connections_table.item(row, 0)
            if connected_particle_item:
                connected_id = connected_particle_item.text()
                if connected_id and connected_id != "Interaction analysis":
                    # Open another popup for the connected particle
                    popup = UniversalParticleDetailPopup(connected_id, self.parent(), self.agent)
                    popup.show()
        except Exception as e:
            pass  # Ignore errors for placeholder data
            
    def _show_error(self, message):
        """Show error message in all tabs"""
        error_text = f"Error: {message}"
        
        # Show in position tab
        for i in range(12):
            self.position_table.setItem(i, 1, QTableWidgetItem("Error"))
            
        # Show in metadata
        self.content_analysis.setText(error_text)
        
        # Update labels
        self.particle_type_label.setText("Type: Error")
        self.particle_energy_label.setText("Energy: Error")
        self.particle_activation_label.setText("Activation: Error")


# Convenience function for easy use across the GUI
def show_particle_details(particle_id, parent=None, agent=None):
    """
    Convenience function to show particle details from anywhere in the GUI.
    
    Usage:
        from gui.utils.particle_popup import show_particle_details
        show_particle_details(particle_id, self)
    """
    if agent is None:
        agent = api.get_api("agent")

    popup = UniversalParticleDetailPopup(particle_id, parent, agent)
    popup.show()
    return popup