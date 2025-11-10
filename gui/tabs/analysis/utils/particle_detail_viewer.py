"""
Particle-based Cognition Engine - GUI analysis utilities - particle detail viewer
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
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, 
    QTableWidget, QTableWidgetItem, QGroupBox, QScrollArea,
    QGridLayout, QPushButton, QDialog, QTabWidget, QFrame,
    QSplitter, QHeaderView
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette
from apis.api_registry import api
import json
from datetime import datetime


class ParticleDetailDialog(QDialog):
    """Dialog for displaying comprehensive particle details"""
    
    def __init__(self, particle_id, parent=None):
        super().__init__(parent)
        self.particle_id = particle_id
        self.particle_data = None
        self.agent = api.get_api("agent")
        
        self.setWindowTitle(f"Particle Details - {particle_id[:12]}...")
        self.setModal(False)  # Allow multiple detail windows
        self.resize(800, 600)
        
        self.init_ui()
        self.load_particle_data()
        
    def init_ui(self):
        """Initialize the particle detail UI"""
        layout = QVBoxLayout(self)
        
        # Header with particle ID and basic info
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 5px;")
        header_layout = QHBoxLayout(header_frame)
        
        self.id_label = QLabel(f"Particle ID: {self.particle_id}")
        self.id_label.setFont(QFont("Monaco", 10, QFont.Weight.Bold))
        header_layout.addWidget(self.id_label)
        
        header_layout.addStretch()
        
        self.status_label = QLabel("Loading...")
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        header_layout.addWidget(self.status_label)
        
        layout.addWidget(header_frame)
        
        # Create tabbed interface for different data views
        self.tab_widget = QTabWidget()
        
        # Tab 1: Basic Properties
        self.basic_tab = self._create_basic_properties_tab()
        self.tab_widget.addTab(self.basic_tab, "Basic Properties")
        
        # Tab 2: Dimensional Positions
        self.position_tab = self._create_position_analysis_tab()
        self.tab_widget.addTab(self.position_tab, "Dimensional Analysis")
        
        # Tab 3: Metadata
        self.metadata_tab = self._create_metadata_tab()
        self.tab_widget.addTab(self.metadata_tab, "Metadata & Content")
        
        # Tab 4: Relationships
        self.relationships_tab = self._create_relationships_tab()
        self.tab_widget.addTab(self.relationships_tab, "Particle Relationships")
        
        # Tab 5: Quantum State
        self.quantum_tab = self._create_quantum_state_tab()
        self.tab_widget.addTab(self.quantum_tab, "Quantum State")
        
        layout.addWidget(self.tab_widget)
        
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
        
    def _create_basic_properties_tab(self):
        """Create basic properties display tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Core properties group
        core_group = QGroupBox("Core Properties")
        core_layout = QGridLayout(core_group)
        
        # Labels for core properties (will be populated with data)
        self.type_label = QLabel("Type: Unknown")
        self.energy_label = QLabel("Energy: 0.0")
        self.activation_label = QLabel("Activation: 0.0")
        self.alive_label = QLabel("Status: Unknown")
        self.age_label = QLabel("Age: Unknown")
        
        core_layout.addWidget(self.type_label, 0, 0)
        core_layout.addWidget(self.energy_label, 0, 1)
        core_layout.addWidget(self.activation_label, 1, 0)
        core_layout.addWidget(self.alive_label, 1, 1)
        core_layout.addWidget(self.age_label, 2, 0, 1, 2)
        
        layout.addWidget(core_group)
        
        # Performance metrics group
        performance_group = QGroupBox("Performance Metrics")
        performance_layout = QGridLayout(performance_group)
        
        self.vitality_label = QLabel("Vitality: Unknown")
        self.creation_index_label = QLabel("Creation Index: Unknown")
        self.last_updated_label = QLabel("Last Updated: Unknown")
        
        performance_layout.addWidget(self.vitality_label, 0, 0)
        performance_layout.addWidget(self.creation_index_label, 0, 1)
        performance_layout.addWidget(self.last_updated_label, 1, 0, 1, 2)
        
        layout.addWidget(performance_group)
        
        layout.addStretch()
        return widget
        
    def _create_position_analysis_tab(self):
        """Create dimensional position analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Position table
        position_group = QGroupBox("12-Dimensional Position Vector")
        position_layout = QVBoxLayout(position_group)
        
        self.position_table = QTableWidget(12, 3)
        self.position_table.setHorizontalHeaderLabels(["Dimension", "Value", "Interpretation"])
        self.position_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        # Set up dimension labels
        dimension_labels = [
            "X (Spatial)", "Y (Spatial)", "Z (Spatial)", 
            "Creation Index", "Field Stability", "Agent Interaction",
            "Frequency", "Memory Phase", "Valence", "Compression",
            "Intent", "Phase Vector"
        ]
        
        for i, label in enumerate(dimension_labels):
            self.position_table.setItem(i, 0, QTableWidgetItem(label))
            self.position_table.setItem(i, 1, QTableWidgetItem("0.000"))
            self.position_table.setItem(i, 2, QTableWidgetItem("Neutral"))
        
        position_layout.addWidget(self.position_table)
        layout.addWidget(position_group)
        
        # Semantic position analysis
        semantic_group = QGroupBox("Semantic Position Analysis")
        semantic_layout = QVBoxLayout(semantic_group)
        
        self.semantic_analysis = QTextEdit()
        self.semantic_analysis.setMaximumHeight(150)
        self.semantic_analysis.setPlainText("Semantic analysis will appear here...")
        semantic_layout.addWidget(self.semantic_analysis)
        
        layout.addWidget(semantic_group)
        
        return widget
        
    def _create_metadata_tab(self):
        """Create metadata and content display tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Content group
        content_group = QGroupBox("Particle Content")
        content_layout = QVBoxLayout(content_group)
        
        self.content_display = QTextEdit()
        self.content_display.setMaximumHeight(200)
        content_layout.addWidget(self.content_display)
        
        layout.addWidget(content_group)
        
        # Metadata table
        metadata_group = QGroupBox("Metadata")
        metadata_layout = QVBoxLayout(metadata_group)
        
        self.metadata_table = QTableWidget(0, 2)
        self.metadata_table.setHorizontalHeaderLabels(["Key", "Value"])
        self.metadata_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.metadata_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        
        metadata_layout.addWidget(self.metadata_table)
        layout.addWidget(metadata_group)
        
        return widget
        
    def _create_relationships_tab(self):
        """Create particle relationships display tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Linked particles
        linked_group = QGroupBox("Linked Particles")
        linked_layout = QVBoxLayout(linked_group)
        
        self.relationships_table = QTableWidget(0, 3)
        self.relationships_table.setHorizontalHeaderLabels(["Relationship Type", "Target Particle", "Status"])
        self.relationships_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        linked_layout.addWidget(self.relationships_table)
        layout.addWidget(linked_group)
        
        # Source particle info
        source_group = QGroupBox("Source Particle")
        source_layout = QVBoxLayout(source_group)
        
        self.source_particle_label = QLabel("No source particle")
        source_layout.addWidget(self.source_particle_label)
        
        layout.addWidget(source_group)
        
        layout.addStretch()
        return widget
        
    def _create_quantum_state_tab(self):
        """Create quantum state analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Superposition state
        superposition_group = QGroupBox("Superposition State")
        superposition_layout = QGridLayout(superposition_group)
        
        self.certain_label = QLabel("Certain: 0.5")
        self.uncertain_label = QLabel("Uncertain: 0.5")
        self.collapsed_state_label = QLabel("Collapsed State: None")
        self.observation_context_label = QLabel("Observation Context: None")
        
        superposition_layout.addWidget(self.certain_label, 0, 0)
        superposition_layout.addWidget(self.uncertain_label, 0, 1)
        superposition_layout.addWidget(self.collapsed_state_label, 1, 0)
        superposition_layout.addWidget(self.observation_context_label, 1, 1)
        
        layout.addWidget(superposition_group)
        
        # Quantum history
        history_group = QGroupBox("Observation History")
        history_layout = QVBoxLayout(history_group)
        
        self.observation_history = QTextEdit()
        self.observation_history.setMaximumHeight(200)
        self.observation_history.setPlainText("No observation history available...")
        
        history_layout.addWidget(self.observation_history)
        layout.addWidget(history_group)
        
        layout.addStretch()
        return widget
        
    def load_particle_data(self):
        """Load comprehensive particle data from the agent"""
        try:
            if not self.agent:
                self.status_label.setText("Agent not available")
                return
                
            # Get particle field
            particle_field = getattr(self.agent, 'particle_field', None)
            if not particle_field:
                self.status_label.setText("Particle field not available")
                return
                
            # Find the particle
            particle = particle_field.get_particle_by_id(self.particle_id)
            if not particle:
                self.status_label.setText("Particle not found")
                return
                
            self.particle_data = particle
            self.update_displays()
            self.status_label.setText("Data loaded successfully")
            
        except Exception as e:
            self.status_label.setText(f"Error loading data: {str(e)}")
            
    def update_displays(self):
        """Update all display elements with current particle data"""
        if not self.particle_data:
            return
            
        try:
            particle = self.particle_data
            
            # Update basic properties
            self._update_basic_properties(particle)
            
            # Update position analysis
            self._update_position_analysis(particle)
            
            # Update metadata
            self._update_metadata(particle)
            
            # Update relationships
            self._update_relationships(particle)
            
            # Update quantum state
            self._update_quantum_state(particle)
            
        except Exception as e:
            self.status_label.setText(f"Error updating displays: {str(e)}")
            
    def _update_basic_properties(self, particle):
        """Update basic properties display"""
        try:
            self.type_label.setText(f"Type: {getattr(particle, 'type', 'Unknown')}")
            self.energy_label.setText(f"Energy: {getattr(particle, 'energy', 0.0):.4f}")
            self.activation_label.setText(f"Activation: {getattr(particle, 'activation', 0.0):.4f}")
            
            alive = getattr(particle, 'alive', False)
            self.alive_label.setText(f"Status: {'Alive' if alive else 'Dead'}")
            
            # Calculate age if possible
            if hasattr(particle, 'w'):
                age = datetime.now().timestamp() - particle.w
                self.age_label.setText(f"Age: {age:.2f} seconds")
            
            # Update performance metrics
            self.vitality_label.setText(f"Vitality: {getattr(particle, 'vitality', 0.0):.4f}")
            self.creation_index_label.setText(f"Creation Index: {getattr(particle, 'creation_index', 'Unknown')}")
            
            if hasattr(particle, 'last_updated'):
                last_updated = datetime.fromtimestamp(particle.last_updated).strftime("%H:%M:%S")
                self.last_updated_label.setText(f"Last Updated: {last_updated}")
                
        except Exception as e:
            self.status_label.setText(f"Error updating basic properties: {str(e)}")
            
    def _update_position_analysis(self, particle):
        """Update position analysis display"""
        try:
            if hasattr(particle, 'position') and particle.position is not None:
                position = particle.position
                
                # Interpretation mapping
                interpretations = [
                    lambda x: f"Spatial coordinate ({x:.3f})",
                    lambda x: f"Spatial coordinate ({x:.3f})",
                    lambda x: f"Spatial coordinate ({x:.3f})",
                    lambda x: f"Creation order: {x:.0f}",
                    lambda x: "Stable" if x > 0.5 else "Unstable",
                    lambda x: "High interaction" if x > 0.5 else "Low interaction",
                    lambda x: f"Frequency: {x:.3f} Hz",
                    lambda x: f"Memory phase: {x:.3f}",
                    lambda x: "Positive" if x > 0 else "Negative" if x < 0 else "Neutral",
                    lambda x: "Compressed" if x > 0.5 else "Expanded",
                    lambda x: "Intentional" if x > 0.5 else "Reactive",
                    lambda x: f"Phase: {x:.3f}"
                ]
                
                for i in range(min(len(position), 12)):
                    value = position[i]
                    self.position_table.setItem(i, 1, QTableWidgetItem(f"{value:.6f}"))
                    
                    if i < len(interpretations):
                        interpretation = interpretations[i](value)
                        self.position_table.setItem(i, 2, QTableWidgetItem(interpretation))
                        
                # Update semantic analysis
                semantic_content = getattr(particle, 'metadata', {}).get('semantic_content', '')
                if semantic_content:
                    analysis_text = f"Semantic Content: {semantic_content}\n\n"
                    analysis_text += f"Spatial Position: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})\n"
                    analysis_text += f"Valence: {position[8]:.3f} ({'Positive' if position[8] > 0 else 'Negative'})\n"
                    analysis_text += f"Frequency: {position[6]:.3f} Hz\n"
                    analysis_text += f"Intent: {position[10]:.3f}"
                    self.semantic_analysis.setPlainText(analysis_text)
                    
        except Exception as e:
            self.status_label.setText(f"Error updating position analysis: {str(e)}")
            
    def _update_metadata(self, particle):
        """Update metadata display"""
        try:
            # Update content display
            content = ""
            if hasattr(particle, 'token'):
                content = f"Token: {particle.token}\n"
            if hasattr(particle, 'content'):
                content += f"Content: {particle.content}\n"
                
            metadata = getattr(particle, 'metadata', {})
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    if key in ['content', 'token']:
                        content += f"{key}: {value}\n"
                        
            self.content_display.setPlainText(content)
            
            # Update metadata table
            if isinstance(metadata, dict):
                self.metadata_table.setRowCount(len(metadata))
                for row, (key, value) in enumerate(metadata.items()):
                    self.metadata_table.setItem(row, 0, QTableWidgetItem(str(key)))
                    # Truncate long values
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:97] + "..."
                    self.metadata_table.setItem(row, 1, QTableWidgetItem(value_str))
                    
        except Exception as e:
            self.status_label.setText(f"Error updating metadata: {str(e)}")
            
    def _update_relationships(self, particle):
        """Update relationships display"""
        try:
            relationships = []
            
            # Check linked particles
            if hasattr(particle, 'linked_particles') and isinstance(particle.linked_particles, dict):
                linked = particle.linked_particles
                
                # Source particle
                if 'source' in linked and linked['source']:
                    self.source_particle_label.setText(f"Source: {linked['source']}")
                    relationships.append(("Parent", str(linked['source']), "Active"))
                    
                # Children particles
                children = linked.get('children', [])
                for child_id in children:
                    relationships.append(("Child", str(child_id), "Active"))
                    
                # Ghost particles
                ghosts = linked.get('ghost', [])
                for ghost_id in ghosts:
                    relationships.append(("Ghost", str(ghost_id), "Inactive"))
                    
            # Source particle ID
            if hasattr(particle, 'source_particle_id') and particle.source_particle_id:
                if not relationships or relationships[0][0] != "Parent":
                    relationships.insert(0, ("Source", str(particle.source_particle_id), "Active"))
                    
            # Update table
            self.relationships_table.setRowCount(len(relationships))
            for row, (rel_type, target, status) in enumerate(relationships):
                self.relationships_table.setItem(row, 0, QTableWidgetItem(rel_type))
                self.relationships_table.setItem(row, 1, QTableWidgetItem(target))
                self.relationships_table.setItem(row, 2, QTableWidgetItem(status))
                
        except Exception as e:
            self.status_label.setText(f"Error updating relationships: {str(e)}")
            
    def _update_quantum_state(self, particle):
        """Update quantum state display"""
        try:
            # Superposition state
            if hasattr(particle, 'superposition') and isinstance(particle.superposition, dict):
                superposition = particle.superposition
                certain = superposition.get('certain', 0.5)
                uncertain = superposition.get('uncertain', 0.5)
                
                self.certain_label.setText(f"Certain: {certain:.4f}")
                self.uncertain_label.setText(f"Uncertain: {uncertain:.4f}")
                
            # Collapsed state
            if hasattr(particle, 'collapsed_state'):
                collapsed = particle.collapsed_state or "None"
                self.collapsed_state_label.setText(f"Collapsed State: {collapsed}")
                
            # Observation context
            if hasattr(particle, 'observation_context'):
                context = particle.observation_context or "None"
                self.observation_context_label.setText(f"Observation Context: {context}")
                
            # Build observation history
            history_text = "Quantum State Information:\n\n"
            history_text += f"Current superposition probability distribution:\n"
            history_text += f"  - Certain state: {getattr(particle, 'superposition', {}).get('certain', 0.5):.4f}\n"
            history_text += f"  - Uncertain state: {getattr(particle, 'superposition', {}).get('uncertain', 0.5):.4f}\n\n"
            
            if hasattr(particle, 'collapsed_state') and particle.collapsed_state:
                history_text += f"Last collapse: {particle.collapsed_state}\n"
                
            if hasattr(particle, 'observation_context') and particle.observation_context:
                history_text += f"Observation context: {particle.observation_context}\n"
                
            self.observation_history.setPlainText(history_text)
            
        except Exception as e:
            self.status_label.setText(f"Error updating quantum state: {str(e)}")


class ParticleDetailViewer(QWidget):
    """Widget for displaying clickable particle references with detail view capability"""
    
    particle_clicked = pyqtSignal(str)  # Signal emitted when particle is clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        """Initialize the particle viewer UI"""
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel("Click on any particle ID below to view detailed information:")
        instructions.setStyleSheet("color: #666; font-style: italic; margin-bottom: 10px;")
        layout.addWidget(instructions)
        
        # Particle list table
        self.particle_table = QTableWidget(0, 4)
        self.particle_table.setHorizontalHeaderLabels(["Particle ID", "Type", "Energy", "Activation"])
        self.particle_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.particle_table.cellClicked.connect(self.on_particle_clicked)
        
        layout.addWidget(self.particle_table)
        
        # Status
        self.status_label = QLabel("Ready to display particle details")
        self.status_label.setStyleSheet("color: green; font-style: italic;")
        layout.addWidget(self.status_label)
        
    def add_particle_reference(self, particle_id, particle_type="unknown", energy=0.0, activation=0.0):
        """Add a particle reference to the clickable list"""
        row = self.particle_table.rowCount()
        self.particle_table.insertRow(row)
        
        # Make particle ID clickable
        id_item = QTableWidgetItem(particle_id[:12] + "...")
        id_item.setData(Qt.ItemDataRole.UserRole, particle_id)  # Store full ID
        id_item.setToolTip(f"Click to view details for {particle_id}")
        
        self.particle_table.setItem(row, 0, id_item)
        self.particle_table.setItem(row, 1, QTableWidgetItem(str(particle_type)))
        self.particle_table.setItem(row, 2, QTableWidgetItem(f"{energy:.4f}"))
        self.particle_table.setItem(row, 3, QTableWidgetItem(f"{activation:.4f}"))
        
    def on_particle_clicked(self, row, column):
        """Handle particle click to open detail view"""
        try:
            id_item = self.particle_table.item(row, 0)
            if id_item:
                particle_id = id_item.data(Qt.ItemDataRole.UserRole)
                if particle_id:
                    # Open particle detail dialog
                    detail_dialog = ParticleDetailDialog(particle_id, self)
                    detail_dialog.show()
                    
                    # Emit signal for other components
                    self.particle_clicked.emit(particle_id)
                    
        except Exception as e:
            self.status_label.setText(f"Error opening particle details: {str(e)}")
            
    def clear_particles(self):
        """Clear all particle references"""
        self.particle_table.setRowCount(0)
        
    def update_status(self, message):
        """Update the status message"""
        self.status_label.setText(message)