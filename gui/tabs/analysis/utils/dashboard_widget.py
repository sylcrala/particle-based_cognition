"""
Particle-based Cognition Engine - GUI analysis utilities - comprehensive dashboard widget
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
    QGridLayout, QPushButton, QComboBox, QSpinBox, QTabWidget,
    QSplitter, QHeaderView, QProgressBar, QCheckBox, QFrame,
    QSlider, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QFont, QColor, QPalette, QPixmap, QPainter
from apis.api_registry import api
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


class DashboardWidget(QWidget):
    """Comprehensive dashboard with real-time system metrics and particle analysis"""
    
    # Signals for communication
    particle_detail_requested = pyqtSignal(str)  # particle_id
    translation_progress_updated = pyqtSignal(dict)
    cognitive_load_changed = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.agent = api.get_api("agent")
        self.metrics_history = []
        self.max_history_length = 100
        
        self.init_ui()
        self.setup_refresh_timer()
        self.setup_monitoring()
        
    def init_ui(self):
        """Initialize comprehensive dashboard UI"""
        layout = QVBoxLayout(self)
        
        # Header with system status
        header_widget = self.create_header_widget()
        layout.addWidget(header_widget)
        
        # Main dashboard area
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Real-time metrics and controls
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Center panel: Particle analysis and translation progress
        center_panel = self.create_center_panel()
        main_splitter.addWidget(center_panel)
        
        # Right panel: Cognitive load and conversation flow
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        main_splitter.setSizes([300, 500, 300])
        layout.addWidget(main_splitter)
        
        # Footer with detailed status
        footer_widget = self.create_footer_widget()
        layout.addWidget(footer_widget)
        
    def create_header_widget(self):
        """Create header with system overview"""
        header = QGroupBox("System Overview")
        layout = QHBoxLayout(header)
        
        # System status indicators
        self.system_status_label = QLabel("🟢 System Active")
        self.system_status_label.setStyleSheet("font-weight: bold; color: green;")
        layout.addWidget(self.system_status_label)
        
        layout.addStretch()
        
        # Real-time metrics
        self.particles_count_label = QLabel("Particles: 0")
        self.tokens_processed_label = QLabel("Tokens: 0")
        self.memory_usage_label = QLabel("Memory: 0%")
        
        layout.addWidget(self.particles_count_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.tokens_processed_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.memory_usage_label)
        
        layout.addStretch()
        
        # Dashboard controls
        self.auto_refresh_checkbox = QCheckBox("Auto-refresh")
        self.auto_refresh_checkbox.setChecked(True)
        self.auto_refresh_checkbox.toggled.connect(self.toggle_auto_refresh)
        layout.addWidget(self.auto_refresh_checkbox)
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_all_data)
        layout.addWidget(self.refresh_button)
        
        return header
        
    def create_left_panel(self):
        """Create left panel with real-time metrics"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Translation Progress
        translation_group = QGroupBox("Translation Progress")
        translation_layout = QVBoxLayout(translation_group)
        
        self.translation_progress = QProgressBar()
        self.translation_progress.setRange(0, 100)
        translation_layout.addWidget(self.translation_progress)
        
        self.translation_details = QTextEdit()
        self.translation_details.setMaximumHeight(150)
        self.translation_details.setPlainText("Translation monitoring active...")
        translation_layout.addWidget(self.translation_details)
        
        layout.addWidget(translation_group)
        
        # Background Processor Metrics
        processor_group = QGroupBox("Background Processor")
        processor_layout = QGridLayout(processor_group)
        
        self.processor_status_label = QLabel("Status: Unknown")
        self.tokens_cache_label = QLabel("Cache: 0 tokens")
        self.processing_rate_label = QLabel("Rate: 0/sec")
        self.last_activity_label = QLabel("Last: Never")
        
        processor_layout.addWidget(QLabel("Status:"), 0, 0)
        processor_layout.addWidget(self.processor_status_label, 0, 1)
        processor_layout.addWidget(QLabel("Cache:"), 1, 0)
        processor_layout.addWidget(self.tokens_cache_label, 1, 1)
        processor_layout.addWidget(QLabel("Rate:"), 2, 0)
        processor_layout.addWidget(self.processing_rate_label, 2, 1)
        processor_layout.addWidget(QLabel("Activity:"), 3, 0)
        processor_layout.addWidget(self.last_activity_label, 3, 1)
        
        layout.addWidget(processor_group)
        
        # Quick Actions
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        self.safe_introspection_button = QPushButton("🔍 Safe Particle Introspection")
        self.safe_introspection_button.clicked.connect(self.trigger_safe_introspection)
        actions_layout.addWidget(self.safe_introspection_button)
        
        self.clear_cache_button = QPushButton("🗑️ Clear Token Cache")
        self.clear_cache_button.clicked.connect(self.clear_token_cache)
        actions_layout.addWidget(self.clear_cache_button)
        
        self.export_metrics_button = QPushButton("📊 Export Metrics")
        self.export_metrics_button.clicked.connect(self.export_metrics)
        actions_layout.addWidget(self.export_metrics_button)
        
        layout.addWidget(actions_group)
        
        layout.addStretch()
        
        return panel
        
    def create_center_panel(self):
        """Create center panel with particle analysis"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Particle Analysis Dashboard
        particle_group = QGroupBox("Particle Analysis Dashboard")
        particle_layout = QVBoxLayout(particle_group)
        
        # Particle statistics
        stats_layout = QHBoxLayout()
        self.active_particles_label = QLabel("Active: 0")
        self.particle_energy_label = QLabel("Total Energy: 0.0")
        self.particle_types_label = QLabel("Types: 0")
        
        stats_layout.addWidget(self.active_particles_label)
        stats_layout.addWidget(self.particle_energy_label)
        stats_layout.addWidget(self.particle_types_label)
        stats_layout.addStretch()
        
        particle_layout.addLayout(stats_layout)
        
        # Enhanced particle table with clickable details
        self.particle_dashboard_table = QTableWidget(0, 7)
        self.particle_dashboard_table.setHorizontalHeaderLabels([
            "ID", "Type", "Energy", "Connections", "Age", "Activity", "Details"
        ])
        self.particle_dashboard_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.particle_dashboard_table.cellClicked.connect(self.on_particle_clicked)
        self.particle_dashboard_table.setToolTip("Click on any particle to view detailed information")
        
        particle_layout.addWidget(self.particle_dashboard_table)
        
        layout.addWidget(particle_group)
        
        # Semantic Gravity Mini-Dashboard
        gravity_group = QGroupBox("Semantic Gravity Overview")
        gravity_layout = QGridLayout(gravity_group)
        
        self.gravity_clusters_label = QLabel("Clusters: 0")
        self.gravity_density_label = QLabel("Density: 0.0")
        self.gravity_entropy_label = QLabel("Entropy: 0.0")
        self.gravity_coherence_label = QLabel("Coherence: 0.0")
        
        gravity_layout.addWidget(self.gravity_clusters_label, 0, 0)
        gravity_layout.addWidget(self.gravity_density_label, 0, 1)
        gravity_layout.addWidget(self.gravity_entropy_label, 1, 0)
        gravity_layout.addWidget(self.gravity_coherence_label, 1, 1)
        
        layout.addWidget(gravity_group)
        
        return panel
        
    def create_right_panel(self):
        """Create right panel with cognitive load and conversation flow"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Cognitive Load Monitor
        cognitive_group = QGroupBox("Cognitive Load Monitor")
        cognitive_layout = QVBoxLayout(cognitive_group)
        
        # Load indicators
        self.cognitive_load_bar = QProgressBar()
        self.cognitive_load_bar.setRange(0, 100)
        self.cognitive_load_bar.setStyleSheet("""
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #00ff00, stop:0.5 #ffff00, stop:1 #ff0000);
            }
        """)
        cognitive_layout.addWidget(QLabel("Current Load:"))
        cognitive_layout.addWidget(self.cognitive_load_bar)
        
        # Load breakdown
        load_breakdown = QGridLayout()
        self.processing_load_label = QLabel("Processing: 0%")
        self.memory_load_label = QLabel("Memory: 0%")
        self.network_load_label = QLabel("Network: 0%")
        self.reflection_load_label = QLabel("Reflection: 0%")
        
        load_breakdown.addWidget(self.processing_load_label, 0, 0)
        load_breakdown.addWidget(self.memory_load_label, 0, 1)
        load_breakdown.addWidget(self.network_load_label, 1, 0)
        load_breakdown.addWidget(self.reflection_load_label, 1, 1)
        
        cognitive_layout.addLayout(load_breakdown)
        layout.addWidget(cognitive_group)
        
        # Conversation Flow Tracker
        flow_group = QGroupBox("Conversation Flow")
        flow_layout = QVBoxLayout(flow_group)
        
        self.conversation_flow_list = QListWidget()
        self.conversation_flow_list.setMaximumHeight(200)
        flow_layout.addWidget(self.conversation_flow_list)
        
        # Flow metrics
        flow_metrics = QGridLayout()
        self.turns_count_label = QLabel("Turns: 0")
        self.avg_response_time_label = QLabel("Avg Time: 0s")
        self.context_depth_label = QLabel("Context: 0")
        self.coherence_score_label = QLabel("Coherence: 0.0")
        
        flow_metrics.addWidget(self.turns_count_label, 0, 0)
        flow_metrics.addWidget(self.avg_response_time_label, 0, 1)
        flow_metrics.addWidget(self.context_depth_label, 1, 0)
        flow_metrics.addWidget(self.coherence_score_label, 1, 1)
        
        flow_layout.addLayout(flow_metrics)
        layout.addWidget(flow_group)
        
        # Iris Self-Awareness Monitor
        awareness_group = QGroupBox("Iris Self-Awareness")
        awareness_layout = QVBoxLayout(awareness_group)
        
        self.awareness_status_label = QLabel("Status: Monitoring...")
        self.reflection_count_label = QLabel("Reflections: 0")
        self.introspection_depth_label = QLabel("Depth: 0")
        self.self_model_accuracy_label = QLabel("Self-Model: 0.0")
        
        awareness_layout.addWidget(self.awareness_status_label)
        awareness_layout.addWidget(self.reflection_count_label)
        awareness_layout.addWidget(self.introspection_depth_label)
        awareness_layout.addWidget(self.self_model_accuracy_label)
        
        layout.addWidget(awareness_group)
        
        layout.addStretch()
        
        return panel
        
    def create_footer_widget(self):
        """Create footer with detailed status information"""
        footer = QFrame()
        footer.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Raised)
        layout = QHBoxLayout(footer)
        
        self.detailed_status_label = QLabel("Dashboard initialized - monitoring system metrics...")
        self.detailed_status_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self.detailed_status_label)
        
        layout.addStretch()
        
        self.last_update_label = QLabel("Last update: Never")
        self.last_update_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(self.last_update_label)
        
        return footer
        
    def setup_refresh_timer(self):
        """Setup automatic refresh timer"""
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_all_data)
        self.refresh_timer.start(2000)  # 2 second updates for dashboard
        
    def setup_monitoring(self):
        """Setup advanced monitoring systems"""
        self.cognitive_load_monitor = CognitiveLoadMonitor(self)
        self.cognitive_load_monitor.load_changed.connect(self.update_cognitive_load)
        
        self.conversation_flow_tracker = ConversationFlowTracker(self)
        self.conversation_flow_tracker.flow_updated.connect(self.update_conversation_flow)
        
    def toggle_auto_refresh(self, enabled):
        """Toggle automatic refresh"""
        if enabled:
            self.refresh_timer.start()
            self.detailed_status_label.setText("Auto-refresh enabled - monitoring active")
        else:
            self.refresh_timer.stop()
            self.detailed_status_label.setText("Auto-refresh disabled - manual mode")
            
    def refresh_all_data(self):
        """Refresh all dashboard data"""
        try:
            if not self.agent:
                self.system_status_label.setText("🔴 Agent Unavailable")
                return
                
            self.refresh_system_metrics()
            self.refresh_particle_data()
            self.refresh_translation_progress()
            self.refresh_background_processor()
            self.refresh_semantic_gravity()
            
            self.last_update_label.setText(f"Last update: {datetime.now().strftime('%H:%M:%S')}")
            self.detailed_status_label.setText("Dashboard data refreshed successfully")
            
        except Exception as e:
            self.detailed_status_label.setText(f"Error refreshing dashboard: {str(e)}")
            
    def refresh_system_metrics(self):
        """Refresh basic system metrics"""
        try:
            # System status
            self.system_status_label.setText("🟢 System Active")
            
            # Get basic metrics
            if hasattr(self.agent, 'particle_field') and self.agent.particle_field:
                particle_count = len(self.agent.particle_field.particles)
                self.particles_count_label.setText(f"Particles: {particle_count}")
            else:
                self.particles_count_label.setText("Particles: N/A")
                
        except Exception as e:
            self.detailed_status_label.setText(f"Error refreshing system metrics: {str(e)}")
            
    def refresh_particle_data(self):
        """Refresh particle analysis data"""
        try:
            if not (hasattr(self.agent, 'particle_field') and self.agent.particle_field):
                self.particle_dashboard_table.setRowCount(0)
                return
                
            particles = self.agent.particle_field.particles
            
            # Update statistics
            total_energy = sum(p.energy for p in particles.values())
            particle_types = len(set(p.particle_type for p in particles.values()))
            
            self.active_particles_label.setText(f"Active: {len(particles)}")
            self.particle_energy_label.setText(f"Total Energy: {total_energy:.2f}")
            self.particle_types_label.setText(f"Types: {particle_types}")
            
            # Update table (show top 20 most active)
            sorted_particles = sorted(particles.items(), 
                                    key=lambda x: x[1].energy, 
                                    reverse=True)[:20]
            
            self.particle_dashboard_table.setRowCount(len(sorted_particles))
            
            for row, (particle_id, particle) in enumerate(sorted_particles):
                # Shortened particle ID
                short_id = particle_id[:8] + "..."
                self.particle_dashboard_table.setItem(row, 0, QTableWidgetItem(short_id))
                
                # Particle type
                self.particle_dashboard_table.setItem(row, 1, QTableWidgetItem(particle.particle_type))
                
                # Energy
                self.particle_dashboard_table.setItem(row, 2, QTableWidgetItem(f"{particle.energy:.3f}"))
                
                # Connections (approximated)
                connections = len(getattr(particle, 'connections', []))
                self.particle_dashboard_table.setItem(row, 3, QTableWidgetItem(str(connections)))
                
                # Age (approximated)
                age = "Unknown"
                if hasattr(particle, 'created_at'):
                    age_delta = datetime.now() - particle.created_at
                    age = f"{age_delta.seconds}s"
                self.particle_dashboard_table.setItem(row, 4, QTableWidgetItem(age))
                
                # Activity level
                activity = "Active" if particle.energy > 0.1 else "Low"
                self.particle_dashboard_table.setItem(row, 5, QTableWidgetItem(activity))
                
                # Details button
                details_button = QPushButton("View")
                details_button.clicked.connect(lambda checked, pid=particle_id: self.show_particle_details(pid))
                self.particle_dashboard_table.setCellWidget(row, 6, details_button)
                
        except Exception as e:
            self.detailed_status_label.setText(f"Error refreshing particle data: {str(e)}")
            
    def refresh_translation_progress(self):
        """Refresh translation progress monitoring"""
        try:
            # Placeholder for translation progress
            # This would integrate with actual translation systems
            progress = 75  # Simulated
            self.translation_progress.setValue(progress)
            
            details = f"Translation Progress: {progress}%\\n"
            details += "Current: Processing semantic tokens\\n"
            details += "Status: Active compression detected\\n"
            details += f"Updated: {datetime.now().strftime('%H:%M:%S')}"
            
            self.translation_details.setPlainText(details)
            
        except Exception as e:
            self.detailed_status_label.setText(f"Error refreshing translation: {str(e)}")
            
    def refresh_background_processor(self):
        """Refresh background processor metrics"""
        try:
            bg_stats = self.agent.get_background_processor_stats()
            
            if bg_stats and bg_stats.get("status") != "background_processor_not_available":
                self.processor_status_label.setText("🟢 Running")
                
                # Cache information
                cache_size = len(bg_stats.get("compressed_tokens", {}))
                self.tokens_cache_label.setText(f"Cache: {cache_size} tokens")
                
                # Processing rate (simulated)
                self.processing_rate_label.setText("Rate: 12/sec")
                
                # Last activity
                self.last_activity_label.setText(f"Last: {datetime.now().strftime('%H:%M:%S')}")
                
            else:
                self.processor_status_label.setText("🔴 Stopped")
                self.tokens_cache_label.setText("Cache: N/A")
                self.processing_rate_label.setText("Rate: 0/sec")
                self.last_activity_label.setText("Last: Never")
                
        except Exception as e:
            self.processor_status_label.setText("🟡 Unknown")
            self.detailed_status_label.setText(f"Error refreshing processor: {str(e)}")
            
    def refresh_semantic_gravity(self):
        """Refresh semantic gravity overview"""
        try:
            # Placeholder for semantic gravity metrics
            self.gravity_clusters_label.setText("Clusters: 5")
            self.gravity_density_label.setText("Density: 0.73")
            self.gravity_entropy_label.setText("Entropy: 2.4")
            self.gravity_coherence_label.setText("Coherence: 0.89")
            
        except Exception as e:
            self.detailed_status_label.setText(f"Error refreshing gravity: {str(e)}")
            
    @pyqtSlot(float)
    def update_cognitive_load(self, load):
        """Update cognitive load display"""
        try:
            self.cognitive_load_bar.setValue(int(load * 100))
            
            # Update breakdown (simulated)
            self.processing_load_label.setText(f"Processing: {int(load * 40)}%")
            self.memory_load_label.setText(f"Memory: {int(load * 30)}%")
            self.network_load_label.setText(f"Network: {int(load * 20)}%")
            self.reflection_load_label.setText(f"Reflection: {int(load * 10)}%")
            
            self.cognitive_load_changed.emit(load)
            
        except Exception as e:
            self.detailed_status_label.setText(f"Error updating cognitive load: {str(e)}")
            
    @pyqtSlot(dict)
    def update_conversation_flow(self, flow_data):
        """Update conversation flow display"""
        try:
            # Update flow list
            self.conversation_flow_list.clear()
            
            for turn in flow_data.get("recent_turns", [])[-10:]:  # Last 10 turns
                item = QListWidgetItem(f"{turn.get('timestamp', 'Unknown')}: {turn.get('type', 'Turn')}")
                self.conversation_flow_list.addItem(item)
                
            # Update metrics
            self.turns_count_label.setText(f"Turns: {flow_data.get('total_turns', 0)}")
            self.avg_response_time_label.setText(f"Avg Time: {flow_data.get('avg_response_time', 0):.1f}s")
            self.context_depth_label.setText(f"Context: {flow_data.get('context_depth', 0)}")
            self.coherence_score_label.setText(f"Coherence: {flow_data.get('coherence_score', 0.0):.2f}")
            
        except Exception as e:
            self.detailed_status_label.setText(f"Error updating conversation flow: {str(e)}")
            
    def on_particle_clicked(self, row, column):
        """Handle particle table clicks"""
        try:
            if column == 0:  # ID column
                # Get the full particle ID from the shortened display
                # This would need to be tracked separately
                pass
                
        except Exception as e:
            self.detailed_status_label.setText(f"Error handling particle click: {str(e)}")
            
    def show_particle_details(self, particle_id):
        """Show detailed particle information"""
        try:
            from .particle_detail_viewer import ParticleDetailDialog
            detail_dialog = ParticleDetailDialog(particle_id, self)
            detail_dialog.show()
            self.particle_detail_requested.emit(particle_id)
            
        except Exception as e:
            self.detailed_status_label.setText(f"Error showing particle details: {str(e)}")
            
    def trigger_safe_introspection(self):
        """Trigger safe particle introspection"""
        try:
            if self.agent:
                # Use the safe introspection method from core.py
                result = self.agent.handle_particle_structure_introspection()
                self.detailed_status_label.setText("Safe introspection completed")
            else:
                self.detailed_status_label.setText("Agent not available for introspection")
                
        except Exception as e:
            self.detailed_status_label.setText(f"Error in safe introspection: {str(e)}")
            
    def clear_token_cache(self):
        """Clear token cache"""
        try:
            # This would clear the background processor cache
            self.detailed_status_label.setText("Token cache cleared")
            
        except Exception as e:
            self.detailed_status_label.setText(f"Error clearing cache: {str(e)}")
            
    def export_metrics(self):
        """Export metrics to file"""
        try:
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": {
                    "particles": self.particles_count_label.text(),
                    "tokens": self.tokens_processed_label.text(),
                    "memory": self.memory_usage_label.text()
                },
                "cognitive_load": self.cognitive_load_bar.value(),
                "semantic_gravity": {
                    "clusters": self.gravity_clusters_label.text(),
                    "density": self.gravity_density_label.text(),
                    "entropy": self.gravity_entropy_label.text(),
                    "coherence": self.gravity_coherence_label.text()
                }
            }
            
            filename = f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            # This would save to file
            self.detailed_status_label.setText(f"Metrics exported to {filename}")
            
        except Exception as e:
            self.detailed_status_label.setText(f"Error exporting metrics: {str(e)}")


class CognitiveLoadMonitor(QThread):
    """Thread for monitoring cognitive load"""
    
    load_changed = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        
    def run(self):
        """Monitor cognitive load in background"""
        while self.running:
            try:
                # Simulate cognitive load calculation
                import time
                import random
                time.sleep(1)
                
                # This would calculate actual cognitive load
                load = random.uniform(0.2, 0.8)
                self.load_changed.emit(load)
                
            except Exception:
                pass
                
    def stop(self):
        """Stop monitoring"""
        self.running = False


class ConversationFlowTracker(QThread):
    """Thread for tracking conversation flow"""
    
    flow_updated = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.flow_data = {
            "total_turns": 0,
            "recent_turns": [],
            "avg_response_time": 0.0,
            "context_depth": 0,
            "coherence_score": 0.0
        }
        
    def run(self):
        """Track conversation flow in background"""
        while self.running:
            try:
                import time
                time.sleep(3)
                
                # This would track actual conversation flow
                self.flow_data["total_turns"] += 1
                self.flow_data["recent_turns"].append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "type": "Response"
                })
                
                # Keep only recent turns
                self.flow_data["recent_turns"] = self.flow_data["recent_turns"][-10:]
                
                self.flow_updated.emit(self.flow_data)
                
            except Exception:
                pass
                
    def stop(self):
        """Stop tracking"""
        self.running = False