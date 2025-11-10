"""
Particle-based Cognition Engine - GUI visualizer tab utilities - field statistics
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

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import QTimer
from apis.api_registry import api

class FieldStats(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.field = None
        self.foundation_layout = QVBoxLayout()
        self.setLayout(self.foundation_layout)
        self.base_layout = QHBoxLayout()
        self.foundation_layout.addLayout(self.base_layout, stretch=5)
        self.legend_layout = QHBoxLayout()
        self.foundation_layout.addLayout(self.legend_layout, stretch=1)

        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()
        self.base_layout.addLayout(self.left_layout)
        self.base_layout.addLayout(self.right_layout)


        self.alive_particle_layout = QHBoxLayout()
        self.alive_particle_label = QLabel("Alive:")
        self.alive_particle_count = QLabel("N/A")
        self.alive_particle_layout.addWidget(self.alive_particle_label)
        self.alive_particle_layout.addWidget(self.alive_particle_count)
        self.left_layout.addLayout(self.alive_particle_layout)

        self.total_particle_layout = QHBoxLayout()
        self.total_particle_label = QLabel("Total:")
        self.total_particle_count = QLabel("N/A")
        self.total_particle_layout.addWidget(self.total_particle_label)
        self.total_particle_layout.addWidget(self.total_particle_count)
        self.left_layout.addLayout(self.total_particle_layout)

        self.avg_energy_layout = QHBoxLayout()
        self.avg_energy_label = QLabel("Avg Energy:")
        self.avg_energy_value = QLabel("N/A")
        self.avg_energy_layout.addWidget(self.avg_energy_label)
        self.avg_energy_layout.addWidget(self.avg_energy_value)
        self.left_layout.addLayout(self.avg_energy_layout)

        self.avg_activation_layout = QHBoxLayout()
        self.avg_activation_label = QLabel("Avg Activation:")
        self.avg_activation_value = QLabel("N/A")
        self.avg_activation_layout.addWidget(self.avg_activation_label)
        self.avg_activation_layout.addWidget(self.avg_activation_value)
        self.left_layout.addLayout(self.avg_activation_layout)

        self.avg_valence_layout = QHBoxLayout()
        self.avg_valence_label = QLabel("Avg Valence:")
        self.avg_valence_value = QLabel("N/A")
        self.avg_valence_layout.addWidget(self.avg_valence_label)
        self.avg_valence_layout.addWidget(self.avg_valence_value)
        self.right_layout.addLayout(self.avg_valence_layout)
    
        self.avg_frequency_layout = QHBoxLayout()
        self.avg_frequency_label = QLabel("Avg Frequency:")
        self.avg_frequency_value = QLabel("N/A")
        self.avg_frequency_layout.addWidget(self.avg_frequency_label)
        self.avg_frequency_layout.addWidget(self.avg_frequency_value)
        self.right_layout.addLayout(self.avg_frequency_layout)

        self.avg_memory_phase = QHBoxLayout()
        self.avg_memory_phase_label = QLabel("Avg Memory Phase:")
        self.avg_memory_phase_value = QLabel("N/A")
        self.avg_memory_phase.addWidget(self.avg_memory_phase_label)
        self.avg_memory_phase.addWidget(self.avg_memory_phase_value)
        self.right_layout.addLayout(self.avg_memory_phase)

        self.legend_layout.addWidget(QLabel("Legend TBD"))

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_stats)
        self.update_timer.start(1000)  # Update every second

    def update_stats(self):
        """Fetch current field stats and update the display"""
        try:
            self.field = api.get_api("_agent_field")
            if not self.field:
                # Field not available
                self.alive_particle_count.setText("N/A")
                self.total_particle_count.setText("N/A")
                self.avg_energy_value.setText("N/A")
                self.avg_activation_value.setText("N/A")
                self.avg_valence_value.setText("N/A")
                self.avg_frequency_value.setText("N/A")
                self.avg_memory_phase_value.setText("N/A")
                return
            
            # Update particle counts
            alive_count = getattr(self.field, 'alive_particle_count', 0)
            total_count = getattr(self.field, 'total_particle_count', 0)
            
            self.alive_particle_count.setText(str(alive_count))
            self.total_particle_count.setText(str(total_count))
            
            # Update averages with safe attribute access
            avg_energy = getattr(self.field, 'avg_energy', 0.0)
            avg_activation = getattr(self.field, 'avg_activation', 0.0)
            avg_valence = getattr(self.field, 'avg_valence', 0.0)
            avg_frequency = getattr(self.field, 'avg_frequency', 0.0)
            avg_mem_phase = getattr(self.field, 'avg_mem_phase', 0.0)
            
            self.avg_energy_value.setText(f"{avg_energy:.3f}")
            self.avg_activation_value.setText(f"{avg_activation:.3f}")
            self.avg_valence_value.setText(f"{avg_valence:.3f}")
            self.avg_frequency_value.setText(f"{avg_frequency:.3f}")
            self.avg_memory_phase_value.setText(f"{avg_mem_phase:.3f}")
            
        except Exception as e:
            # Show error in the UI
            error_msg = f"Error: {str(e)[:20]}"
            self.alive_particle_count.setText(error_msg)
            self.total_particle_count.setText(error_msg)
            self.avg_energy_value.setText(error_msg)
            self.avg_activation_value.setText(error_msg)
            self.avg_valence_value.setText(error_msg)
            self.avg_frequency_value.setText(error_msg)
            self.avg_memory_phase_value.setText(error_msg)
            print(f"Error updating field stats: {str(e)}")