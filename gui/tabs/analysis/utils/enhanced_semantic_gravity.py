"""
Particle-based Cognition Engine - GUI analysis utilities - enhanced semantic gravity analysis
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
    QSplitter, QHeaderView, QProgressBar, QCheckBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette
from apis.api_registry import api
import json
from datetime import datetime
from .particle_detail_viewer import ParticleDetailViewer, ParticleDetailDialog
from gui.utils.particle_popup import show_particle_details


class EnhancedSemanticGravityAnalyzer(QWidget):
    """Enhanced semantic gravity analysis with particle detail viewing"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.agent = api.get_api("agent")
        self.gravity_data = {}
        self.observation_history = []
        
        self.init_ui()
        self.setup_refresh_timer()
        
    def init_ui(self):
        """Initialize enhanced semantic gravity UI"""
        layout = QVBoxLayout(self)
        
        # Control panel
        controls_frame = QGroupBox("Analysis Controls")
        controls_layout = QHBoxLayout(controls_frame)
        
        # Analysis type buttons
        controls_layout.addWidget(QLabel("Analysis:"))
        
        self.cluster_button = QPushButton("Gravitational Clusters")
        self.cluster_button.setCheckable(True)
        self.cluster_button.setChecked(True)  # Default active
        self.cluster_button.clicked.connect(lambda: self.set_analysis_type("Gravitational Clustering"))
        controls_layout.addWidget(self.cluster_button)
        
        self.frequency_button = QPushButton("Token Frequency")
        self.frequency_button.setCheckable(True)
        self.frequency_button.clicked.connect(lambda: self.set_analysis_type("Token Frequency Analysis"))
        controls_layout.addWidget(self.frequency_button)
        
        self.spatial_button = QPushButton("Spatial Analysis")
        self.spatial_button.setCheckable(True)
        self.spatial_button.clicked.connect(lambda: self.set_analysis_type("Spatial Distribution"))
        controls_layout.addWidget(self.spatial_button)
        
        self.interactions_button = QPushButton("Particle Interactions")
        self.interactions_button.setCheckable(True)
        self.interactions_button.clicked.connect(lambda: self.set_analysis_type("Particle Interactions"))
        controls_layout.addWidget(self.interactions_button)
        
        self.compression_button = QPushButton("Compression Patterns")
        self.compression_button.setCheckable(True)
        self.compression_button.clicked.connect(lambda: self.set_analysis_type("Compression Patterns"))
        controls_layout.addWidget(self.compression_button)
        
        # Store current analysis type
        self.current_analysis = "Gravitational Clustering"
        
        controls_layout.addStretch()
        
        # Auto-refresh controls
        self.auto_refresh_checkbox = QCheckBox("Auto-refresh")
        self.auto_refresh_checkbox.setChecked(True)
        self.auto_refresh_checkbox.toggled.connect(self.toggle_auto_refresh)
        controls_layout.addWidget(self.auto_refresh_checkbox)
        
        self.refresh_interval_spinner = QSpinBox()
        self.refresh_interval_spinner.setRange(1, 60)
        self.refresh_interval_spinner.setValue(5)
        self.refresh_interval_spinner.setSuffix(" sec")
        self.refresh_interval_spinner.valueChanged.connect(self.update_refresh_interval)
        controls_layout.addWidget(self.refresh_interval_spinner)
        
        self.manual_refresh_button = QPushButton("Refresh Now")
        self.manual_refresh_button.clicked.connect(self.refresh_data)
        controls_layout.addWidget(self.manual_refresh_button)
        
        layout.addWidget(controls_frame)
        
        # Main content area that changes based on analysis type
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        
        # Create different content views for each analysis type
        self._create_content_views()
        
        # Set default view
        self.set_analysis_type("Gravitational Clustering")
        
        layout.addWidget(self.content_area)
        
        # Status bar
        self.status_label = QLabel("Enhanced semantic gravity analysis ready")
        self.status_label.setStyleSheet("color: green; font-style: italic; padding: 5px;")
        layout.addWidget(self.status_label)
        
    def _create_content_views(self):
        """Create different content views for each analysis type"""
        # Gravitational Clustering View
        self.clustering_view = self._create_clustering_view()
        
        # Token Frequency View  
        self.frequency_view = self._create_frequency_view()
        
        # Spatial Analysis View
        self.spatial_view = self._create_spatial_view()
        
        # Particle Interactions View
        self.interactions_view = self._create_interactions_view()
        
        # Compression Patterns View
        self.compression_view = self._create_compression_view()
        
        # Store all views
        self.content_views = {
            "Gravitational Clustering": self.clustering_view,
            "Token Frequency Analysis": self.frequency_view,
            "Spatial Distribution": self.spatial_view,
            "Particle Interactions": self.interactions_view,
            "Compression Patterns": self.compression_view
        }
        
    def _create_clustering_view(self):
        """Create gravitational clustering analysis view"""
        view = QWidget()
        layout = QVBoxLayout(view)
        
        # Statistics panel
        stats_group = QGroupBox("Clustering Statistics")
        stats_layout = QGridLayout(stats_group)
        
        self.cluster_total_label = QLabel("Total Clusters: 0")
        self.cluster_avg_size_label = QLabel("Average Size: 0.0")
        self.cluster_largest_label = QLabel("Largest Cluster: 0")
        self.cluster_density_label = QLabel("Average Density: 0.0")
        
        stats_layout.addWidget(self.cluster_total_label, 0, 0)
        stats_layout.addWidget(self.cluster_avg_size_label, 0, 1)
        stats_layout.addWidget(self.cluster_largest_label, 1, 0)
        stats_layout.addWidget(self.cluster_density_label, 1, 1)
        
        layout.addWidget(stats_group)
        
        # Clusters table
        clusters_group = QGroupBox("Gravitational Clusters")
        clusters_layout = QVBoxLayout(clusters_group)
        
        self.clusters_table = QTableWidget(0, 6)
        self.clusters_table.setHorizontalHeaderLabels([
            "Cluster ID", "Center Concept", "Token Count", "Density", "Representative Tokens", "Particles"
        ])
        self.clusters_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.clusters_table.cellDoubleClicked.connect(self.on_cluster_double_clicked)
        self.clusters_table.setToolTip("Double-click to view cluster details")
        
        clusters_layout.addWidget(self.clusters_table)
        layout.addWidget(clusters_group)
        
        return view
        
    def _create_frequency_view(self):
        """Create token frequency analysis view"""
        view = QWidget()
        layout = QVBoxLayout(view)
        
        # Frequency statistics
        freq_stats_group = QGroupBox("Frequency Statistics")
        freq_stats_layout = QGridLayout(freq_stats_group)
        
        self.freq_total_observations_label = QLabel("Total Observations: 0")
        self.freq_unique_tokens_label = QLabel("Unique Tokens: 0")
        self.freq_avg_frequency_label = QLabel("Average Frequency: 0.0")
        self.freq_most_active_label = QLabel("Most Active: None")
        
        freq_stats_layout.addWidget(self.freq_total_observations_label, 0, 0)
        freq_stats_layout.addWidget(self.freq_unique_tokens_label, 0, 1)
        freq_stats_layout.addWidget(self.freq_avg_frequency_label, 1, 0)
        freq_stats_layout.addWidget(self.freq_most_active_label, 1, 1)
        
        layout.addWidget(freq_stats_group)
        
        # Token frequency table
        freq_group = QGroupBox("Token Frequency Analysis")
        freq_layout = QVBoxLayout(freq_group)
        
        self.frequency_table = QTableWidget(0, 7)
        self.frequency_table.setHorizontalHeaderLabels([
            "Token", "Total Count", "Recent Activity", "Growth Rate", "Confidence", "Last Seen", "Particles"
        ])
        self.frequency_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.frequency_table.cellDoubleClicked.connect(self.on_frequency_double_clicked)
        self.frequency_table.setToolTip("Double-click to view token details and particles")
        
        freq_layout.addWidget(self.frequency_table)
        layout.addWidget(freq_group)
        
        return view
        
    def _create_spatial_view(self):
        """Create spatial analysis view"""
        view = QWidget()
        layout = QVBoxLayout(view)
        
        # Spatial statistics
        spatial_stats_group = QGroupBox("Spatial Distribution Statistics")
        spatial_stats_layout = QGridLayout(spatial_stats_group)
        
        self.spatial_positioned_label = QLabel("Positioned Particles: 0")
        self.spatial_spread_label = QLabel("Spatial Spread: 0.0")
        self.spatial_avg_distance_label = QLabel("Avg Distance: 0.0")
        self.spatial_hotspots_label = QLabel("Dense Regions: 0")
        
        spatial_stats_layout.addWidget(self.spatial_positioned_label, 0, 0)
        spatial_stats_layout.addWidget(self.spatial_spread_label, 0, 1)
        spatial_stats_layout.addWidget(self.spatial_avg_distance_label, 1, 0)
        spatial_stats_layout.addWidget(self.spatial_hotspots_label, 1, 1)
        
        layout.addWidget(spatial_stats_group)
        
        # Particle spatial data table
        spatial_group = QGroupBox("Particle Spatial Data")
        spatial_layout = QVBoxLayout(spatial_group)
        
        self.spatial_table = QTableWidget(0, 8)
        self.spatial_table.setHorizontalHeaderLabels([
            "Particle ID", "Type", "Token/Content", "X Position", "Y Position", "Z Position", "Energy", "Grid Location"
        ])
        self.spatial_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.spatial_table.cellDoubleClicked.connect(self.on_spatial_double_clicked)
        self.spatial_table.setToolTip("Double-click any particle to view detailed information")
        
        spatial_layout.addWidget(self.spatial_table)
        layout.addWidget(spatial_group)
        
        return view
        
    def _create_interactions_view(self):
        """Create particle interactions view"""
        view = QWidget()
        layout = QVBoxLayout(view)
        
        # Interactions statistics
        interactions_stats_group = QGroupBox("Interaction Statistics")
        interactions_stats_layout = QGridLayout(interactions_stats_group)
        
        self.interactions_total_label = QLabel("Total Interactions: 0")
        self.interactions_active_label = QLabel("Active Connections: 0")
        self.interactions_avg_strength_label = QLabel("Avg Strength: 0.0")
        self.interactions_clusters_label = QLabel("Interaction Clusters: 0")
        
        interactions_stats_layout.addWidget(self.interactions_total_label, 0, 0)
        interactions_stats_layout.addWidget(self.interactions_active_label, 0, 1)
        interactions_stats_layout.addWidget(self.interactions_avg_strength_label, 1, 0)
        interactions_stats_layout.addWidget(self.interactions_clusters_label, 1, 1)
        
        layout.addWidget(interactions_stats_group)
        
        # Interactions table
        interactions_group = QGroupBox("Particle Interactions")
        interactions_layout = QVBoxLayout(interactions_group)
        
        self.interactions_table = QTableWidget(0, 7)
        self.interactions_table.setHorizontalHeaderLabels([
            "Source Particle", "Target Particle", "Interaction Type", "Strength", "Duration", "Energy Transfer", "Status"
        ])
        self.interactions_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.interactions_table.cellDoubleClicked.connect(self.on_interaction_double_clicked)
        self.interactions_table.setToolTip("Double-click to view detailed interaction information")
        
        interactions_layout.addWidget(self.interactions_table)
        layout.addWidget(interactions_group)
        
        return view
        
    def _create_compression_view(self):
        """Create compression patterns view"""
        view = QWidget()
        layout = QVBoxLayout(view)
        
        # Compression statistics
        compression_stats_group = QGroupBox("Compression Statistics")
        compression_stats_layout = QGridLayout(compression_stats_group)
        
        self.compression_ratio_label = QLabel("Compression Ratio: 0%")
        self.compression_efficiency_label = QLabel("Efficiency: 0.0")
        self.compression_patterns_label = QLabel("Detected Patterns: 0")
        self.compression_entropy_label = QLabel("Entropy: 0.0")
        
        compression_stats_layout.addWidget(self.compression_ratio_label, 0, 0)
        compression_stats_layout.addWidget(self.compression_efficiency_label, 0, 1)
        compression_stats_layout.addWidget(self.compression_patterns_label, 1, 0)
        compression_stats_layout.addWidget(self.compression_entropy_label, 1, 1)
        
        layout.addWidget(compression_stats_group)
        
        # Compression patterns table
        compression_group = QGroupBox("Language Compression Patterns")
        compression_layout = QVBoxLayout(compression_group)
        
        self.compression_table = QTableWidget(0, 7)
        self.compression_table.setHorizontalHeaderLabels([
            "Compressed Token", "Original Length", "Compressed Length", "Efficiency", "Frequency", "Confidence", "Pattern Type"
        ])
        self.compression_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.compression_table.cellDoubleClicked.connect(self.on_compression_double_clicked)
        self.compression_table.setToolTip("Double-click to analyze compression pattern")
        
        compression_layout.addWidget(self.compression_table)
        layout.addWidget(compression_group)
        
        return view
        
    def setup_refresh_timer(self):
        """Setup automatic refresh timer"""
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(5000)  # 5 seconds default
        
    def toggle_auto_refresh(self, enabled):
        """Toggle automatic refresh"""
        if enabled:
            self.refresh_timer.start()
            self.status_label.setText("Auto-refresh enabled")
        else:
            self.refresh_timer.stop()
            self.status_label.setText("Auto-refresh disabled")
            
    def update_refresh_interval(self, seconds):
        """Update refresh interval"""
        self.refresh_timer.setInterval(seconds * 1000)
        self.status_label.setText(f"Refresh interval: {seconds} seconds")
        
    def refresh_data(self):
        """Refresh all semantic gravity data"""
        try:
            if not self.agent:
                self.status_label.setText("Agent not available")
                return
                
            # Get background processor stats
            bg_stats = self.agent.get_background_processor_stats()
            if bg_stats and bg_stats.get("status") != "background_processor_not_available":
                self.gravity_data = bg_stats
                self.update_observations_display()
                self.update_clusters_display()
                self.update_analysis_results()
                self.status_label.setText(f"Data refreshed at {datetime.now().strftime('%H:%M:%S')}")
            else:
                self.status_label.setText("Background processor not available")
                
        except Exception as e:
            self.status_label.setText(f"Error refreshing data: {str(e)}")
            
    def update_observations_display(self):
        """Update the observations table with current data"""
        try:
            observations = self.gravity_data.get("observations", {})
            
            if not observations:
                self.observations_table.setRowCount(0)
                return
                
            # Update statistics
            total_obs = len(observations)
            unique_tokens = len(set(obs.get("token", "") for obs in observations.values()))
            avg_confidence = sum(obs.get("confidence", 0) for obs in observations.values()) / total_obs if total_obs > 0 else 0
            
            self.total_observations_label.setText(f"Total: {total_obs}")
            self.unique_tokens_label.setText(f"Unique Tokens: {unique_tokens}")
            self.avg_confidence_label.setText(f"Avg Confidence: {avg_confidence:.3f}")
            
            # Update table
            self.observations_table.setRowCount(len(observations))
            
            for row, (obs_id, obs_data) in enumerate(observations.items()):
                token = obs_data.get("token", "Unknown")
                confidence = obs_data.get("confidence", 0.0)
                particle_id = obs_data.get("particle_id", "Unknown")
                particle_type = obs_data.get("particle_type", "Unknown")
                energy = obs_data.get("energy", 0.0)
                
                self.observations_table.setItem(row, 0, QTableWidgetItem(token))
                self.observations_table.setItem(row, 1, QTableWidgetItem(f"{confidence:.4f}"))
                
                # Make particle ID clickable
                particle_item = QTableWidgetItem(particle_id[:12] + "...")
                particle_item.setData(Qt.ItemDataRole.UserRole, particle_id)
                particle_item.setToolTip(f"Click to view details for {particle_id}")
                self.observations_table.setItem(row, 2, particle_item)
                
                self.observations_table.setItem(row, 3, QTableWidgetItem(particle_type))
                self.observations_table.setItem(row, 4, QTableWidgetItem(f"{energy:.4f}"))
                
                # Details button
                details_button = QPushButton("View")
                details_button.clicked.connect(lambda checked, pid=particle_id: self.show_particle_details(pid))
                self.observations_table.setCellWidget(row, 5, details_button)
                
                # Add to particle viewer
                self.particle_viewer.add_particle_reference(particle_id, particle_type, energy, 0.0)
                
        except Exception as e:
            self.status_label.setText(f"Error updating observations: {str(e)}")
            
    def update_clusters_display(self):
        """Update the clusters table"""
        try:
            clusters = self.gravity_data.get("clusters", {})
            
            self.clusters_table.setRowCount(len(clusters))
            
            for row, (cluster_id, cluster_data) in enumerate(clusters.items()):
                token_count = len(cluster_data.get("tokens", []))
                avg_distance = cluster_data.get("average_distance", 0.0)
                particles = cluster_data.get("particles", [])
                
                self.clusters_table.setItem(row, 0, QTableWidgetItem(cluster_id))
                self.clusters_table.setItem(row, 1, QTableWidgetItem(str(token_count)))
                self.clusters_table.setItem(row, 2, QTableWidgetItem(f"{avg_distance:.4f}"))
                self.clusters_table.setItem(row, 3, QTableWidgetItem(f"{len(particles)} particles"))
                
        except Exception as e:
            self.status_label.setText(f"Error updating clusters: {str(e)}")
            
    def set_analysis_type(self, analysis_type):
        """Set the active analysis type and update button states"""
        self.current_analysis = analysis_type
        
        # Update button states
        self.cluster_button.setChecked(analysis_type == "Gravitational Clustering")
        self.frequency_button.setChecked(analysis_type == "Token Frequency Analysis")
        self.spatial_button.setChecked(analysis_type == "Spatial Distribution")
        self.interactions_button.setChecked(analysis_type == "Particle Interactions")
        self.compression_button.setChecked(analysis_type == "Compression Patterns")
        
        # Clear current content
        for i in reversed(range(self.content_layout.count())):
            child = self.content_layout.takeAt(i)
            if child.widget():
                child.widget().setParent(None)
        
        # Add the appropriate view
        if analysis_type in self.content_views:
            self.content_layout.addWidget(self.content_views[analysis_type])
            
        # Refresh data for the new view
        self.update_analysis_results()
        
    def update_analysis_view(self):
        """Update analysis view when type changes"""
        self.update_analysis_results()
        
    def update_analysis_results(self):
        """Update the analysis results based on current view"""
        try:
            analysis_type = self.current_analysis
            
            if analysis_type == "Gravitational Clustering":
                self.populate_clustering_view()
            elif analysis_type == "Token Frequency Analysis":
                self.populate_frequency_view()
            elif analysis_type == "Spatial Distribution":
                self.populate_spatial_view()
            elif analysis_type == "Particle Interactions":
                self.populate_interactions_view()
            elif analysis_type == "Compression Patterns":
                self.populate_compression_view()
                
        except Exception as e:
            self.status_label.setText(f"Error updating analysis: {str(e)}")
            
    def populate_clustering_view(self):
        """Populate the gravitational clustering view"""
        try:
            # Get clustering data
            clusters = self._get_clustering_data()
            
            # Update statistics
            cluster_count = len(clusters)
            if cluster_count > 0:
                sizes = [cluster.get('size', 0) for cluster in clusters.values()]
                avg_size = sum(sizes) / len(sizes)
                largest = max(sizes)
                avg_density = sum(cluster.get('density', 0) for cluster in clusters.values()) / cluster_count
                
                self.cluster_total_label.setText(f"Total Clusters: {cluster_count}")
                self.cluster_avg_size_label.setText(f"Average Size: {avg_size:.1f}")
                self.cluster_largest_label.setText(f"Largest Cluster: {largest}")
                self.cluster_density_label.setText(f"Average Density: {avg_density:.3f}")
            else:
                self.cluster_total_label.setText("Total Clusters: 0")
                self.cluster_avg_size_label.setText("Average Size: 0.0")
                self.cluster_largest_label.setText("Largest Cluster: 0")
                self.cluster_density_label.setText("Average Density: 0.0")
            
            # Populate clusters table
            self.clusters_table.setRowCount(len(clusters))
            for row, (cluster_id, cluster_data) in enumerate(clusters.items()):
                self.clusters_table.setItem(row, 0, QTableWidgetItem(cluster_id))
                self.clusters_table.setItem(row, 1, QTableWidgetItem(cluster_data.get('center_concept', 'Unknown')))
                self.clusters_table.setItem(row, 2, QTableWidgetItem(str(cluster_data.get('size', 0))))
                self.clusters_table.setItem(row, 3, QTableWidgetItem(f"{cluster_data.get('density', 0):.3f}"))
                
                # Representative tokens
                tokens = cluster_data.get('representative_tokens', [])
                token_text = ', '.join(tokens[:3]) + ('...' if len(tokens) > 3 else '')
                self.clusters_table.setItem(row, 4, QTableWidgetItem(token_text))
                
                # Particles count
                particles = cluster_data.get('particles', [])
                self.clusters_table.setItem(row, 5, QTableWidgetItem(f"{len(particles)} particles"))
                
        except Exception as e:
            self.status_label.setText(f"Error populating clustering view: {str(e)}")
            
    def populate_frequency_view(self):
        """Populate the token frequency view"""
        try:
            # Get frequency data
            frequency_data = self._get_frequency_data()
            
            if frequency_data:
                # Update statistics
                total_obs = sum(data['count'] for data in frequency_data.values())
                unique_tokens = len(frequency_data)
                avg_freq = total_obs / unique_tokens if unique_tokens > 0 else 0
                
                # Find most active token
                most_active = max(frequency_data.items(), key=lambda x: x[1]['count']) if frequency_data else ("None", {"count": 0})
                
                self.freq_total_observations_label.setText(f"Total Observations: {total_obs}")
                self.freq_unique_tokens_label.setText(f"Unique Tokens: {unique_tokens}")
                self.freq_avg_frequency_label.setText(f"Average Frequency: {avg_freq:.1f}")
                self.freq_most_active_label.setText(f"Most Active: {most_active[0]} ({most_active[1]['count']})")
                
                # Populate frequency table
                sorted_tokens = sorted(frequency_data.items(), key=lambda x: x[1]['count'], reverse=True)
                self.frequency_table.setRowCount(len(sorted_tokens))
                
                for row, (token, data) in enumerate(sorted_tokens):
                    self.frequency_table.setItem(row, 0, QTableWidgetItem(token))
                    self.frequency_table.setItem(row, 1, QTableWidgetItem(str(data['count'])))
                    self.frequency_table.setItem(row, 2, QTableWidgetItem(str(data.get('recent_activity', 0))))
                    
                    # Growth rate with color coding
                    growth = data.get('growth_rate', 0.0)
                    growth_item = QTableWidgetItem(f"{growth:+.3f}")
                    if growth > 0:
                        growth_item.setBackground(QColor(144, 238, 144))
                    elif growth < 0:
                        growth_item.setBackground(QColor(255, 182, 193))
                    self.frequency_table.setItem(row, 3, growth_item)
                    
                    self.frequency_table.setItem(row, 4, QTableWidgetItem(f"{data.get('confidence', 0):.3f}"))
                    self.frequency_table.setItem(row, 5, QTableWidgetItem(data.get('last_seen', 'Never')))
                    self.frequency_table.setItem(row, 6, QTableWidgetItem(f"{len(data.get('particles', []))} particles"))
            
        except Exception as e:
            self.status_label.setText(f"Error populating frequency view: {str(e)}")
            
    def populate_spatial_view(self):
        """Populate the spatial analysis view"""
        try:
            # Get spatial data
            spatial_data = self._get_particle_spatial_data()
            
            # Update statistics
            positioned_count = len(spatial_data)
            self.spatial_positioned_label.setText(f"Positioned Particles: {positioned_count}")
            
            if positioned_count > 1:
                # Calculate spatial metrics
                positions = [data['position'] for data in spatial_data.values()]
                
                # Calculate spatial spread
                x_coords = [pos[0] for pos in positions]
                y_coords = [pos[1] for pos in positions]
                z_coords = [pos[2] for pos in positions]
                
                spread = max(max(x_coords) - min(x_coords), 
                           max(y_coords) - min(y_coords),
                           max(z_coords) - min(z_coords)) if positions else 0
                
                # Calculate average distance
                total_distance = 0
                count = 0
                for i, pos1 in enumerate(positions):
                    for pos2 in positions[i+1:]:
                        distance = ((pos1[0] - pos2[0])**2 + 
                                  (pos1[1] - pos2[1])**2 + 
                                  (pos1[2] - pos2[2])**2)**0.5
                        total_distance += distance
                        count += 1
                
                avg_distance = total_distance / count if count > 0 else 0
                
                self.spatial_spread_label.setText(f"Spatial Spread: {spread:.2f}")
                self.spatial_avg_distance_label.setText(f"Avg Distance: {avg_distance:.2f}")
                
                # Simple hotspot detection
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                center_z = sum(z_coords) / len(z_coords)
                
                hotspots = len([pos for pos in positions 
                               if ((pos[0] - center_x)**2 + (pos[1] - center_y)**2 + (pos[2] - center_z)**2)**0.5 < avg_distance * 0.5])
                self.spatial_hotspots_label.setText(f"Dense Regions: {hotspots}")
            
            # Populate spatial table
            self.spatial_table.setRowCount(len(spatial_data))
            for row, (particle_id, data) in enumerate(spatial_data.items()):
                # Store particle ID for double-click handling
                pid_item = QTableWidgetItem(particle_id[:12] + "...")
                pid_item.setData(Qt.ItemDataRole.UserRole, particle_id)
                pid_item.setToolTip(f"Double-click to view details for {particle_id}")
                self.spatial_table.setItem(row, 0, pid_item)
                
                self.spatial_table.setItem(row, 1, QTableWidgetItem(data.get('type', 'unknown')))
                self.spatial_table.setItem(row, 2, QTableWidgetItem(str(data.get('token', 'N/A'))[:15]))
                
                pos = data.get('position', [0, 0, 0])
                self.spatial_table.setItem(row, 3, QTableWidgetItem(f"{pos[0]:.3f}"))
                self.spatial_table.setItem(row, 4, QTableWidgetItem(f"{pos[1]:.3f}"))
                self.spatial_table.setItem(row, 5, QTableWidgetItem(f"{pos[2]:.3f}"))
                self.spatial_table.setItem(row, 6, QTableWidgetItem(f"{data.get('energy', 0):.4f}"))
                self.spatial_table.setItem(row, 7, QTableWidgetItem(data.get('grid_key', 'N/A')))
                
        except Exception as e:
            self.status_label.setText(f"Error populating spatial view: {str(e)}")
            
    def populate_interactions_view(self):
        """Populate the particle interactions view"""
        try:
            # Placeholder for interactions data
            interactions_data = self._get_interactions_data()
            
            self.interactions_total_label.setText("Total Interactions: 0")
            self.interactions_active_label.setText("Active Connections: 0")
            self.interactions_avg_strength_label.setText("Avg Strength: 0.0")
            self.interactions_clusters_label.setText("Interaction Clusters: 0")
            
            # Clear table for now
            self.interactions_table.setRowCount(1)
            self.interactions_table.setItem(0, 0, QTableWidgetItem("Interaction analysis"))
            self.interactions_table.setItem(0, 1, QTableWidgetItem("requires enhanced"))
            self.interactions_table.setItem(0, 2, QTableWidgetItem("particle field"))
            self.interactions_table.setItem(0, 3, QTableWidgetItem("integration"))
            self.interactions_table.setItem(0, 4, QTableWidgetItem("N/A"))
            self.interactions_table.setItem(0, 5, QTableWidgetItem("N/A"))
            self.interactions_table.setItem(0, 6, QTableWidgetItem("Coming soon"))
            
        except Exception as e:
            self.status_label.setText(f"Error populating interactions view: {str(e)}")
            
    def populate_compression_view(self):
        """Populate the compression patterns view"""
        try:
            # Get compression data
            compression_data = self._get_compression_data()
            
            if compression_data:
                # Update statistics
                total_tokens = len(compression_data)
                compressed_count = len([d for d in compression_data.values() if d.get('is_compressed', False)])
                ratio = (compressed_count / total_tokens * 100) if total_tokens > 0 else 0
                avg_efficiency = sum(d.get('efficiency', 0) for d in compression_data.values()) / total_tokens if total_tokens > 0 else 0
                
                self.compression_ratio_label.setText(f"Compression Ratio: {ratio:.1f}%")
                self.compression_efficiency_label.setText(f"Efficiency: {avg_efficiency:.3f}")
                self.compression_patterns_label.setText(f"Detected Patterns: {compressed_count}")
                
                # Calculate entropy (simplified)
                frequencies = [d.get('frequency', 1) for d in compression_data.values()]
                total_freq = sum(frequencies)
                entropy = -sum((f/total_freq) * (f/total_freq).bit_length() for f in frequencies if f > 0) if total_freq > 0 else 0
                self.compression_entropy_label.setText(f"Entropy: {entropy:.3f}")
                
                # Populate compression table
                sorted_compression = sorted(compression_data.items(), 
                                           key=lambda x: x[1].get('efficiency', 0), 
                                           reverse=True)
                
                self.compression_table.setRowCount(len(sorted_compression))
                for row, (token, data) in enumerate(sorted_compression):
                    self.compression_table.setItem(row, 0, QTableWidgetItem(token))
                    self.compression_table.setItem(row, 1, QTableWidgetItem(str(data.get('original_length', 0))))
                    self.compression_table.setItem(row, 2, QTableWidgetItem(str(len(token))))
                    
                    # Efficiency with color coding
                    efficiency = data.get('efficiency', 0)
                    eff_item = QTableWidgetItem(f"{efficiency:.3f}")
                    if efficiency > 0.7:
                        eff_item.setBackground(QColor(144, 238, 144))  # Green
                    elif efficiency > 0.4:
                        eff_item.setBackground(QColor(255, 255, 144))  # Yellow
                    self.compression_table.setItem(row, 3, eff_item)
                    
                    self.compression_table.setItem(row, 4, QTableWidgetItem(str(data.get('frequency', 0))))
                    self.compression_table.setItem(row, 5, QTableWidgetItem(f"{data.get('confidence', 0):.3f}"))
                    self.compression_table.setItem(row, 6, QTableWidgetItem(data.get('pattern_type', 'Unknown')))
            
        except Exception as e:
            self.status_label.setText(f"Error populating compression view: {str(e)}")
            
    # Data retrieval methods
        """Analyze gravitational clustering patterns"""
        try:
            observations = self.gravity_data.get("observations", {})
            clusters = self.gravity_data.get("clusters", {})
            
            analysis = "Gravitational Clustering Analysis\\n"
            analysis += "=" * 40 + "\\n\\n"
            
            analysis += f"Total Observations: {len(observations)}\\n"
            analysis += f"Active Clusters: {len(clusters)}\\n\\n"
            
            if clusters:
                # Analyze cluster characteristics
                cluster_sizes = [len(cluster.get("tokens", [])) for cluster in clusters.values()]
                avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes)
                largest_cluster = max(cluster_sizes)
                
                analysis += f"Average Cluster Size: {avg_cluster_size:.2f} tokens\\n"
                analysis += f"Largest Cluster: {largest_cluster} tokens\\n\\n"
                
                # Find most active clusters
                active_clusters = sorted(clusters.items(), 
                                       key=lambda x: len(x[1].get("tokens", [])), 
                                       reverse=True)[:3]
                
                analysis += "Top 3 Most Active Clusters:\\n"
                for i, (cluster_id, cluster_data) in enumerate(active_clusters, 1):
                    tokens = cluster_data.get("tokens", [])
                    analysis += f"{i}. Cluster {cluster_id}: {len(tokens)} tokens\\n"
                    if tokens:
                        analysis += f"   Sample tokens: {', '.join(tokens[:5])}\\n"
                        
            self.analysis_results.setPlainText(analysis)
            
        except Exception as e:
            self.analysis_results.setPlainText(f"Error in clustering analysis: {str(e)}")
            
    def analyze_token_frequency(self):
        """Analyze token frequency patterns"""
        try:
            observations = self.gravity_data.get("observations", {})
            
            # Count token frequencies
            token_counts = {}
            confidence_totals = {}
            
            for obs_data in observations.values():
                token = obs_data.get("token", "")
                confidence = obs_data.get("confidence", 0.0)
                
                token_counts[token] = token_counts.get(token, 0) + 1
                confidence_totals[token] = confidence_totals.get(token, 0) + confidence
                
            analysis = "Token Frequency Analysis\\n"
            analysis += "=" * 30 + "\\n\\n"
            
            # Sort by frequency
            sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
            
            analysis += f"Total Unique Tokens: {len(sorted_tokens)}\\n"
            analysis += f"Total Observations: {sum(token_counts.values())}\\n\\n"
            
            analysis += "Top 10 Most Frequent Tokens:\\n"
            for i, (token, count) in enumerate(sorted_tokens[:10], 1):
                avg_confidence = confidence_totals.get(token, 0) / count
                analysis += f"{i:2d}. {token:<15} ({count:3d} times, confidence: {avg_confidence:.3f})\\n"
                
            # Analyze confidence patterns
            high_confidence_tokens = [(token, confidence_totals[token]/token_counts[token]) 
                                    for token in token_counts 
                                    if confidence_totals[token]/token_counts[token] > 0.8]
            
            if high_confidence_tokens:
                analysis += "\\nHigh Confidence Tokens (>0.8):\\n"
                for token, avg_conf in sorted(high_confidence_tokens, key=lambda x: x[1], reverse=True)[:5]:
                    analysis += f"  {token}: {avg_conf:.4f}\\n"
                    
            self.analysis_results.setPlainText(analysis)
            
        except Exception as e:
            self.analysis_results.setPlainText(f"Error in frequency analysis: {str(e)}")
            
    def analyze_spatial_distribution(self):
        """Analyze spatial distribution of particles with detailed particle viewing"""
        try:
            # Get actual spatial data from the particle field
            spatial_data = self._get_particle_spatial_data()
            
            analysis = "Spatial Distribution Analysis\n"
            analysis += "=" * 35 + "\n\n"
            
            if spatial_data:
                analysis += f"Positioned Particles: {len(spatial_data)}\n"
                
                # Calculate spatial metrics
                positions = [data['position'] for data in spatial_data.values()]
                if len(positions) > 1:
                    # Calculate spatial spread
                    x_coords = [pos[0] for pos in positions]
                    y_coords = [pos[1] for pos in positions]
                    z_coords = [pos[2] for pos in positions]
                    
                    x_spread = max(x_coords) - min(x_coords) if x_coords else 0
                    y_spread = max(y_coords) - min(y_coords) if y_coords else 0
                    z_spread = max(z_coords) - min(z_coords) if z_coords else 0
                    
                    analysis += f"Spatial Spread: X={x_spread:.2f}, Y={y_spread:.2f}, Z={z_spread:.2f}\n"
                    
                    # Calculate average inter-particle distance
                    total_distance = 0
                    count = 0
                    for i, pos1 in enumerate(positions):
                        for pos2 in positions[i+1:]:
                            distance = ((pos1[0] - pos2[0])**2 + 
                                      (pos1[1] - pos2[1])**2 + 
                                      (pos1[2] - pos2[2])**2)**0.5
                            total_distance += distance
                            count += 1
                    
                    avg_distance = total_distance / count if count > 0 else 0
                    analysis += f"Average Inter-particle Distance: {avg_distance:.4f}\n\n"
                
                # Show top particles by energy
                sorted_particles = sorted(spatial_data.items(), 
                                        key=lambda x: x[1].get('energy', 0), 
                                        reverse=True)
                
                analysis += "Top Particles by Energy:\n"
                analysis += f"{'ID':<12} {'Type':<10} {'Energy':<8} {'Position (x,y,z)':<20} {'Token':<15}\n"
                analysis += "-" * 80 + "\n"
                
                for particle_id, data in sorted_particles[:10]:
                    pid = particle_id[:10] + ".."
                    ptype = data.get('type', 'unknown')[:8]
                    energy = data.get('energy', 0)
                    pos = data.get('position', [0,0,0])
                    token = data.get('token', 'N/A')[:12]
                    
                    analysis += f"{pid:<12} {ptype:<10} {energy:<8.4f} "
                    analysis += f"({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})".ljust(20)
                    analysis += f"{token:<15}\n"
                
                analysis += "\nDouble-click any particle ID in the observations table to view detailed information including:\n"
                analysis += "• Full position in 12D space\n"
                analysis += "• Energy and activation levels\n"
                analysis += "• Complete metadata\n"
                analysis += "• Field position and interactions\n"
                
            else:
                analysis += "No spatial data available.\n"
                analysis += "This may indicate:\n"
                analysis += "• Particle field not fully initialized\n"
                analysis += "• Spatial indexing not enabled\n"
                analysis += "• No particles with position data\n\n"
                analysis += "Check the system overview for particle field status."
                
            self.analysis_results.setPlainText(analysis)
            
        except Exception as e:
            self.analysis_results.setPlainText(f"Error in spatial analysis: {str(e)}")
            
    def _get_particle_spatial_data(self):
        """Get spatial data for particles from the field"""
        spatial_data = {}
        
        try:
            if not self.agent:
                return spatial_data
                
            # Try to access particle field directly
            if (hasattr(self.agent, 'particle_field') and self.agent.particle_field and 
                hasattr(self.agent.particle_field, 'spatial_grid')):
                
                field = self.agent.particle_field
                processed_count = 0
                max_particles = 100  # Limit for performance
                
                # Get particles from spatial grid
                for grid_key, particle_ids in field.spatial_grid.items():
                    if processed_count >= max_particles:
                        break
                        
                    for particle_id in list(particle_ids)[:5]:  # Limit per grid cell
                        particle = field.get_particle_by_id(particle_id)
                        if particle and hasattr(particle, 'position') and particle.position is not None:
                            try:
                                # Extract spatial data
                                pos = particle.position
                                metadata = getattr(particle, 'metadata', {}) or {}
                                
                                spatial_data[particle_id] = {
                                    'position': [float(pos[0]) if len(pos) > 0 else 0.0,
                                               float(pos[1]) if len(pos) > 1 else 0.0,
                                               float(pos[2]) if len(pos) > 2 else 0.0],
                                    'type': getattr(particle, 'type', 'unknown'),
                                    'energy': getattr(particle, 'energy', 0.0),
                                    'token': metadata.get('token') or metadata.get('content', 'N/A'),
                                    'grid_key': str(grid_key)
                                }
                                processed_count += 1
                                
                            except (IndexError, TypeError, ValueError):
                                continue
                                
        except Exception as e:
            # Return empty dict on error - will show "no data available" message
            pass
            
        return spatial_data
        
    def analyze_particle_interactions(self):
        """Analyze particle interactions and relationships"""
        analysis = "Particle Interactions Analysis\\n"
        analysis += "=" * 35 + "\\n\\n"
        analysis += "Interaction analysis will show:\\n\\n"
        analysis += "- Particle creation relationships\\n"
        analysis += "- Linkage patterns between particles\\n"
        analysis += "- Energy transfer patterns\\n"
        analysis += "- Quantum state correlations\\n\\n"
        analysis += "This requires enhanced particle field integration."
        
        self.analysis_results.setPlainText(analysis)
        
    def analyze_compression_patterns(self):
        """Analyze language compression patterns"""
        try:
            observations = self.gravity_data.get("observations", {})
            
            # Analyze token characteristics
            compressed_tokens = []
            human_tokens = []
            
            for obs_data in observations.values():
                token = obs_data.get("token", "")
                confidence = obs_data.get("confidence", 0.0)
                
                # Simple heuristic for compression detection
                if len(token) <= 4 and any(c.isalpha() for c in token):
                    compressed_tokens.append((token, confidence))
                else:
                    human_tokens.append((token, confidence))
                    
            analysis = "Language Compression Patterns\\n"
            analysis += "=" * 35 + "\\n\\n"
            
            analysis += f"Compressed Tokens: {len(compressed_tokens)}\\n"
            analysis += f"Human-like Tokens: {len(human_tokens)}\\n"
            analysis += f"Compression Ratio: {len(compressed_tokens)/(len(compressed_tokens)+len(human_tokens))*100:.1f}%\\n\\n"
            
            if compressed_tokens:
                analysis += "Sample Compressed Tokens:\\n"
                for token, conf in sorted(compressed_tokens, key=lambda x: x[1], reverse=True)[:10]:
                    analysis += f"  {token} (confidence: {conf:.3f})\\n"
                    
            self.analysis_results.setPlainText(analysis)
            
        except Exception as e:
            self.analysis_results.setPlainText(f"Error in compression analysis: {str(e)}")
            
    def _get_clustering_data(self):
        """Get clustering data from background processor"""
        clusters = {}
        try:
            if self.gravity_data:
                # Analyze translation mappings to form clusters
                translation_mappings = self.gravity_data.get("translation_mappings", {})
                
                # Group by human translations to form clusters
                translation_groups = {}
                for token, candidates in translation_mappings.items():
                    for candidate in candidates:
                        if candidate not in translation_groups:
                            translation_groups[candidate] = []
                        translation_groups[candidate].append(token)
                
                # Convert to cluster format
                for i, (concept, tokens) in enumerate(translation_groups.items()):
                    if len(tokens) > 0:  # Include single-token "clusters" too
                        cluster_id = f"cluster_{i}"
                        clusters[cluster_id] = {
                            'center_concept': concept,
                            'size': len(tokens),
                            'density': self._calculate_cluster_density(tokens),
                            'representative_tokens': tokens[:5],
                            'particles': self._get_particles_for_tokens(tokens)
                        }
        except Exception:
            pass
        return clusters
        
    def _get_frequency_data(self):
        """Get token frequency data"""
        frequency_data = {}
        try:
            if self.gravity_data:
                token_cache = self.gravity_data.get("compressed_token_cache", {})
                confidence_scores = self.gravity_data.get("confidence_scores", {})
                
                for token, timestamps in token_cache.items():
                    count = len(timestamps)
                    recent_count = len([t for t in timestamps if t > (datetime.now().timestamp() - 300)])
                    
                    # Calculate growth rate
                    if len(timestamps) > 1:
                        recent_timestamps = [t for t in timestamps if t > (datetime.now().timestamp() - 900)]
                        old_timestamps = [t for t in timestamps if t <= (datetime.now().timestamp() - 900)]
                        recent_rate = len(recent_timestamps) / 15.0 if recent_timestamps else 0
                        old_rate = len(old_timestamps) / max((timestamps[-1] - timestamps[0]) / 60, 1) if old_timestamps else 0
                        growth_rate = recent_rate - old_rate
                    else:
                        growth_rate = 0.0
                    
                    # Get average confidence
                    token_confidences = [conf for (t, candidate), conf in confidence_scores.items() if t == token]
                    avg_confidence = sum(token_confidences) / len(token_confidences) if token_confidences else 0.0
                    
                    # Last seen
                    last_seen = datetime.fromtimestamp(timestamps[-1]).strftime('%H:%M:%S') if timestamps else "Never"
                    
                    frequency_data[token] = {
                        'count': count,
                        'recent_activity': recent_count,
                        'growth_rate': growth_rate,
                        'confidence': avg_confidence,
                        'last_seen': last_seen,
                        'particles': self._get_particles_for_tokens([token])
                    }
        except Exception:
            pass
        return frequency_data
        
    def _get_compression_data(self):
        """Get compression analysis data"""
        compression_data = {}
        try:
            if self.gravity_data:
                token_cache = self.gravity_data.get("compressed_token_cache", {})
                translation_mappings = self.gravity_data.get("translation_mappings", {})
                
                for token, timestamps in token_cache.items():
                    # Determine if it's compressed (heuristic)
                    is_compressed = len(token) <= 5 and any(c.isalpha() for c in token)
                    
                    # Get translation candidates
                    candidates = translation_mappings.get(token, [])
                    original_length = max(len(c) for c in candidates) if candidates else len(token)
                    
                    # Calculate efficiency
                    efficiency = 1 - (len(token) / original_length) if original_length > 0 else 0
                    
                    # Determine pattern type
                    pattern_type = "Unknown"
                    if is_compressed:
                        if len(token) <= 3:
                            pattern_type = "Ultra-compressed"
                        elif any(c in token for c in 'qxz'):
                            pattern_type = "Rare-char compression"
                        else:
                            pattern_type = "Standard compression"
                    else:
                        pattern_type = "Uncompressed"
                    
                    compression_data[token] = {
                        'is_compressed': is_compressed,
                        'original_length': original_length,
                        'efficiency': efficiency,
                        'frequency': len(timestamps),
                        'confidence': 0.5,  # Placeholder
                        'pattern_type': pattern_type
                    }
        except Exception:
            pass
        return compression_data
        
    def _get_interactions_data(self):
        """Get particle interaction data (placeholder)"""
        return {}
        
    def _calculate_cluster_density(self, tokens):
        """Calculate density metric for a cluster"""
        if not tokens:
            return 0.0
        # Simple heuristic based on token similarity and frequency
        return min(1.0, len(tokens) * 0.1)
        
    def _get_particles_for_tokens(self, tokens):
        """Get particle references for given tokens"""
        particles = []
        try:
            # This would need to query the particle field for particles containing these tokens
            # For now, return placeholder
            particles = [f"particle_{i}" for i in range(min(3, len(tokens)))]
        except Exception:
            pass
        return particles
        
    # Double-click handlers for particle detail popup
    def on_cluster_double_clicked(self, row, column):
        """Handle cluster table double-click"""
        try:
            cluster_id_item = self.clusters_table.item(row, 0)
            if cluster_id_item:
                cluster_id = cluster_id_item.text()
                self.show_cluster_details(cluster_id)
        except Exception as e:
            self.status_label.setText(f"Error handling cluster click: {str(e)}")
            
    def on_frequency_double_clicked(self, row, column):
        """Handle frequency table double-click"""
        try:
            token_item = self.frequency_table.item(row, 0)
            if token_item:
                token = token_item.text()
                self.show_token_details(token)
        except Exception as e:
            self.status_label.setText(f"Error handling frequency click: {str(e)}")
            
    def on_spatial_double_clicked(self, row, column):
        """Handle spatial table double-click"""
        try:
            particle_item = self.spatial_table.item(row, 0)
            if particle_item:
                particle_id = particle_item.data(Qt.ItemDataRole.UserRole)
                if particle_id:
                    self.show_particle_details(particle_id)
        except Exception as e:
            self.status_label.setText(f"Error handling spatial click: {str(e)}")
            
    def on_interaction_double_clicked(self, row, column):
        """Handle interaction table double-click"""
        try:
            # Show interaction details
            self.status_label.setText("Interaction details coming soon...")
        except Exception as e:
            self.status_label.setText(f"Error handling interaction click: {str(e)}")
            
    def on_compression_double_clicked(self, row, column):
        """Handle compression table double-click"""
        try:
            token_item = self.compression_table.item(row, 0)
            if token_item:
                token = token_item.text()
                self.show_compression_analysis(token)
        except Exception as e:
            self.status_label.setText(f"Error handling compression click: {str(e)}")
            
    # Detail popup methods
    def show_particle_details(self, particle_id):
        """Show detailed particle information in universal popup"""
        try:
            # Use the universal particle popup
            show_particle_details(particle_id, self, self.agent)
            self.status_label.setText(f"Viewing details for particle: {particle_id[:12]}...")
        except Exception as e:
            self.status_label.setText(f"Error showing particle details: {str(e)}")
            
    def show_cluster_details(self, cluster_id):
        """Show cluster analysis details"""
        try:
            # Create a simple dialog for cluster details
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit
            
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Cluster Details: {cluster_id}")
            dialog.setModal(False)
            dialog.resize(500, 400)
            
            layout = QVBoxLayout(dialog)
            layout.addWidget(QLabel(f"Detailed analysis for {cluster_id}"))
            
            details = QTextEdit()
            details.setText(f"Cluster: {cluster_id}\\n\\nDetailed analysis would show:\\n• Token relationships\\n• Gravitational strength\\n• Temporal evolution\\n• Associated particles")
            layout.addWidget(details)
            
            dialog.show()
        except Exception as e:
            self.status_label.setText(f"Error showing cluster details: {str(e)}")
            
    def show_token_details(self, token):
        """Show token analysis details"""
        try:
            # Create a simple dialog for token details
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit
            
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Token Analysis: {token}")
            dialog.setModal(False)
            dialog.resize(500, 400)
            
            layout = QVBoxLayout(dialog)
            layout.addWidget(QLabel(f"Analysis for token: {token}"))
            
            details = QTextEdit()
            details.setText(f"Token: {token}\\n\\nAnalysis includes:\\n• Frequency patterns\\n• Context associations\\n• Translation candidates\\n• Temporal distribution")
            layout.addWidget(details)
            
            dialog.show()
        except Exception as e:
            self.status_label.setText(f"Error showing token details: {str(e)}")
            
    def show_compression_analysis(self, token):
        """Show compression pattern analysis"""
        try:
            # Create a simple dialog for compression analysis
            from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit
            
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Compression Analysis: {token}")
            dialog.setModal(False)
            dialog.resize(500, 400)
            
            layout = QVBoxLayout(dialog)
            layout.addWidget(QLabel(f"Compression pattern for: {token}"))
            
            details = QTextEdit()
            details.setText(f"Token: {token}\\n\\nCompression analysis:\\n• Pattern type\\n• Efficiency metrics\\n• Original candidates\\n• Usage frequency")
            layout.addWidget(details)
            
            dialog.show()
        except Exception as e:
            self.status_label.setText(f"Error showing compression analysis: {str(e)}")
            
    # Legacy methods for compatibility
    def on_observation_clicked(self, row, column):
        """Handle observation table clicks (legacy - now unused)"""
        pass
        
    def on_particle_detail_requested(self, particle_id):
        """Handle particle detail requests (legacy)"""
        self.show_particle_details(particle_id)
        
    def clear_data(self):
        """Clear all displayed data"""
        # Clear all tables in all views
        if hasattr(self, 'clusters_table'):
            self.clusters_table.setRowCount(0)
        if hasattr(self, 'frequency_table'):
            self.frequency_table.setRowCount(0)
        if hasattr(self, 'spatial_table'):
            self.spatial_table.setRowCount(0)
        if hasattr(self, 'interactions_table'):
            self.interactions_table.setRowCount(0)
        if hasattr(self, 'compression_table'):
            self.compression_table.setRowCount(0)
            
        self.gravity_data = {}
        self.status_label.setText("Data cleared")