"""
Particle-based Cognition Engine - GUI analysis tab with comprehensive system analytics
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
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, 
    QPushButton, QTextEdit, QTableWidget, QTableWidgetItem,
    QGroupBox, QScrollArea, QGridLayout, QProgressBar,
    QComboBox, QSpinBox, QCheckBox, QFrame, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette
from apis.api_registry import api
import json
from datetime import datetime


class AnalysisTab(QWidget):
    """Comprehensive analysis tab with multiple subtabs for different system metrics"""
    
    def __init__(self):
        super().__init__()
        self.agent = None
        self.auto_refresh = True
        self.refresh_interval = 5000  # 5 seconds
        
        self.init_ui()
        self.setup_refresh_timer()
        
    def init_ui(self):
        """Initialize the analysis tab UI with subtabs"""
        # Main layout
        layout = QVBoxLayout(self)
        
        # Header with controls
        header_layout = QHBoxLayout()
        
        # Title
        title_label = QLabel("System Analysis Dashboard")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        
        # Auto-refresh controls
        self.auto_refresh_checkbox = QCheckBox("Auto-refresh")
        self.auto_refresh_checkbox.setChecked(self.auto_refresh)
        self.auto_refresh_checkbox.stateChanged.connect(self.toggle_auto_refresh)
        header_layout.addWidget(QLabel("Refresh:"))
        header_layout.addWidget(self.auto_refresh_checkbox)
        
        # Refresh interval
        self.refresh_interval_spinbox = QSpinBox()
        self.refresh_interval_spinbox.setRange(1, 60)
        self.refresh_interval_spinbox.setValue(self.refresh_interval // 1000)
        self.refresh_interval_spinbox.setSuffix(" sec")
        self.refresh_interval_spinbox.valueChanged.connect(self.change_refresh_interval)
        header_layout.addWidget(self.refresh_interval_spinbox)
        
        # Manual refresh button
        self.refresh_button = QPushButton("Refresh Now")
        self.refresh_button.clicked.connect(self.refresh_data)
        header_layout.addWidget(self.refresh_button)
        
        layout.addLayout(header_layout)
        
        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)
        
        # Create tab widget for different analysis views
        self.tab_widget = QTabWidget()
        
        # Create individual analysis tabs
        self.translation_tab = TranslationDashboardTab()
        self.categorization_tab = CategorizationAnalysisTab()
        self.semantic_gravity_tab = SemanticGravityTab()
        self.memory_thermal_tab = MemoryThermalTab()
        self.system_overview_tab = SystemOverviewTab()
        
        # Add tabs
        self.tab_widget.addTab(self.system_overview_tab, "System Overview")
        self.tab_widget.addTab(self.translation_tab, "Translation Mapping")
        self.tab_widget.addTab(self.categorization_tab, "Categorization Analysis")
        self.tab_widget.addTab(self.semantic_gravity_tab, "Semantic Gravity")
        self.tab_widget.addTab(self.memory_thermal_tab, "Memory Thermal States")
        
        layout.addWidget(self.tab_widget)
        
        # Status bar
        self.status_label = QLabel("Initializing analysis systems...")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.status_label)
    
    def setup_refresh_timer(self):
        """Set up the auto-refresh timer"""
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)
        if self.auto_refresh:
            self.refresh_timer.start(self.refresh_interval)
    
    def toggle_auto_refresh(self, state):
        """Toggle auto-refresh on/off"""
        self.auto_refresh = state == Qt.CheckState.Checked
        if self.auto_refresh:
            self.refresh_timer.start(self.refresh_interval)
        else:
            self.refresh_timer.stop()
    
    def change_refresh_interval(self, value):
        """Change the refresh interval"""
        self.refresh_interval = value * 1000
        if self.auto_refresh:
            self.refresh_timer.stop()
            self.refresh_timer.start(self.refresh_interval)
    
    def refresh_data(self):
        """Refresh all analysis data"""
        try:
            # Get agent reference if not already available
            if not self.agent:
                self.agent = api.get_api("agent")
            
            if self.agent:
                # Refresh each subtab
                self.system_overview_tab.refresh_data(self.agent)
                self.translation_tab.refresh_data(self.agent)
                self.categorization_tab.refresh_data(self.agent)
                self.semantic_gravity_tab.refresh_data(self.agent)
                self.memory_thermal_tab.refresh_data(self.agent)
                
                self.status_label.setText(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
                self.status_label.setStyleSheet("color: green;")
            else:
                self.status_label.setText("Agent not available - waiting for system initialization")
                self.status_label.setStyleSheet("color: orange;")
                
        except Exception as e:
            self.status_label.setText(f"Error refreshing data: {str(e)}")
            self.status_label.setStyleSheet("color: red;")


class SystemOverviewTab(QWidget):
    """System overview with key metrics"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the system overview UI"""
        layout = QVBoxLayout(self)
        
        # Create grid for overview metrics
        grid_layout = QGridLayout()
        
        # Background Processor Status
        bg_group = QGroupBox("Semantic Gravity Background Processor")
        bg_layout = QVBoxLayout(bg_group)
        
        self.bg_status_label = QLabel("Status: Unknown")
        self.bg_tokens_tracked_label = QLabel("Tokens Tracked: 0")
        self.bg_queue_size_label = QLabel("Analysis Queue: 0")
        self.bg_mappings_label = QLabel("Translation Mappings: 0")
        
        bg_layout.addWidget(self.bg_status_label)
        bg_layout.addWidget(self.bg_tokens_tracked_label)
        bg_layout.addWidget(self.bg_queue_size_label)
        bg_layout.addWidget(self.bg_mappings_label)
        
        grid_layout.addWidget(bg_group, 0, 0)
        
        # Categorization System Status
        cat_group = QGroupBox("Agent Categorization System")
        cat_layout = QVBoxLayout(cat_group)
        
        self.cat_status_label = QLabel("Status: Unknown")
        self.cat_categories_label = QLabel("Agent Categories: 0")
        self.cat_translations_label = QLabel("Category Translations: 0")
        self.cat_rules_label = QLabel("Auto Rules: 0")
        
        cat_layout.addWidget(self.cat_status_label)
        cat_layout.addWidget(self.cat_categories_label)
        cat_layout.addWidget(self.cat_translations_label)
        cat_layout.addWidget(self.cat_rules_label)
        
        grid_layout.addWidget(cat_group, 0, 1)
        
        # Memory System Status
        mem_group = QGroupBox("Memory System")
        mem_layout = QVBoxLayout(mem_group)
        
        self.mem_status_label = QLabel("Status: Unknown")
        self.mem_thermal_label = QLabel("Thermal States: Unknown")
        self.mem_particles_label = QLabel("Memory Particles: Unknown")
        
        mem_layout.addWidget(self.mem_status_label)
        mem_layout.addWidget(self.mem_thermal_label)
        mem_layout.addWidget(self.mem_particles_label)
        
        grid_layout.addWidget(mem_group, 1, 0)
        
        # Agent Communication Status
        comm_group = QGroupBox("Agent Communication")
        comm_layout = QVBoxLayout(comm_group)
        
        self.comm_status_label = QLabel("Status: Unknown")
        self.comm_compressed_label = QLabel("Compressed Tokens: 0")
        self.comm_frequency_label = QLabel("Communication Frequency: Unknown")
        
        comm_layout.addWidget(self.comm_status_label)
        comm_layout.addWidget(self.comm_compressed_label)
        comm_layout.addWidget(self.comm_frequency_label)
        
        grid_layout.addWidget(comm_group, 1, 1)
        
        layout.addLayout(grid_layout)
        
        # Recent Activity Log
        activity_group = QGroupBox("Recent System Activity")
        activity_layout = QVBoxLayout(activity_group)
        
        self.activity_log = QTextEdit()
        self.activity_log.setMaximumHeight(150)
        self.activity_log.setReadOnly(True)
        
        activity_layout.addWidget(self.activity_log)
        layout.addWidget(activity_group)
        
    def refresh_data(self, agent):
        """Refresh system overview data"""
        try:
            # Background processor stats
            bg_stats = agent.get_background_processor_stats()
            if bg_stats.get("status") == "not_initialized":
                self.bg_status_label.setText("Status: Not Initialized")
                self.bg_status_label.setStyleSheet("color: orange;")
            elif bg_stats.get("background_running"):
                self.bg_status_label.setText("Status: Running")
                self.bg_status_label.setStyleSheet("color: green;")
                self.bg_tokens_tracked_label.setText(f"Tokens Tracked: {bg_stats.get('tokens_tracked', 0)}")
                self.bg_queue_size_label.setText(f"Analysis Queue: {bg_stats.get('queue_size', 0)}")
                self.bg_mappings_label.setText(f"Translation Mappings: {bg_stats.get('translation_mappings', 0)}")
            else:
                self.bg_status_label.setText("Status: Stopped")
                self.bg_status_label.setStyleSheet("color: red;")
            
            # Categorization stats
            cat_stats = agent.get_categorization_stats()
            if cat_stats.get("status") == "active":
                self.cat_status_label.setText("Status: Active")
                self.cat_status_label.setStyleSheet("color: green;")
                self.cat_categories_label.setText(f"Agent Categories: {cat_stats.get('agent_categories', 0)}")
                self.cat_translations_label.setText(f"Category Translations: {cat_stats.get('category_translations', 0)}")
                self.cat_rules_label.setText(f"Auto Rules: {cat_stats.get('auto_categorization_rules', 0)}")
            else:
                self.cat_status_label.setText("Status: Not Available")
                self.cat_status_label.setStyleSheet("color: orange;")
            
            # Add to activity log
            current_time = datetime.now().strftime('%H:%M:%S')
            activity_text = f"[{current_time}] System status updated - "
            activity_text += f"BG Processor: {bg_stats.get('tokens_tracked', 0)} tokens, "
            activity_text += f"Categories: {cat_stats.get('agent_categories', 0)}"
            
            self.activity_log.append(activity_text)
            
            # Keep activity log reasonable size
            if self.activity_log.document().blockCount() > 50:
                cursor = self.activity_log.textCursor()
                cursor.movePosition(cursor.MoveOperation.Start)
                cursor.select(cursor.SelectionType.BlockUnderCursor)
                cursor.removeSelectedText()
            
        except Exception as e:
            error_time = datetime.now().strftime('%H:%M:%S')
            self.activity_log.append(f"[{error_time}] Error updating overview: {str(e)}")


class TranslationDashboardTab(QWidget):
    """Translation mapping dashboard showing compressed → human language mappings"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the translation dashboard UI"""
        layout = QVBoxLayout(self)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Token Filter:"))
        self.token_filter = QComboBox()
        self.token_filter.addItem("All Tokens")
        self.token_filter.currentTextChanged.connect(self.filter_translations)
        controls_layout.addWidget(self.token_filter)
        
        controls_layout.addStretch()
        
        self.clear_button = QPushButton("Clear All Mappings")
        self.clear_button.clicked.connect(self.clear_mappings)
        controls_layout.addWidget(self.clear_button)
        
        layout.addLayout(controls_layout)
        
        # Translation table
        self.translation_table = QTableWidget()
        self.translation_table.setColumnCount(4)
        self.translation_table.setHorizontalHeaderLabels([
            "Compressed Token", "Human Translation", "Confidence", "Frequency"
        ])
        
        # Make table sortable and resizable
        self.translation_table.setSortingEnabled(True)
        self.translation_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(self.translation_table)
        
        # Token details section
        details_group = QGroupBox("Token Analysis Details")
        details_layout = QVBoxLayout(details_group)
        
        self.token_details = QTextEdit()
        self.token_details.setMaximumHeight(100)
        self.token_details.setReadOnly(True)
        
        details_layout.addWidget(self.token_details)
        layout.addWidget(details_group)
        
        # Connect table selection to show details
        self.translation_table.itemSelectionChanged.connect(self.show_token_details)
        
    def refresh_data(self, agent):
        """Refresh translation mapping data"""
        try:
            # Get comprehensive translation mappings
            mappings = agent.get_all_translation_mappings()
            
            if isinstance(mappings, list) and mappings:
                # Clear existing data
                self.translation_table.setRowCount(0)
                self.token_filter.clear()
                self.token_filter.addItem("All Tokens")
                
                # Populate table with enhanced data
                for row, mapping in enumerate(mappings):
                    # Add null checks for mapping data
                    if not isinstance(mapping, dict):
                        continue
                        
                    self.translation_table.insertRow(row)
                    
                    # Compressed token (with null check)
                    compressed_token = mapping.get("compressed_token", "")
                    self.translation_table.setItem(row, 0, QTableWidgetItem(str(compressed_token)))
                    
                    # Human translation (with null check)
                    human_candidate = mapping.get("human_candidate", "")
                    self.translation_table.setItem(row, 1, QTableWidgetItem(str(human_candidate)))
                    
                    # Confidence (with color coding and category)
                    confidence = mapping.get("confidence", 0.0)
                    if confidence is None:
                        confidence = 0.0
                    confidence_category = mapping.get("confidence_category", "Unknown")
                    if confidence_category is None:
                        confidence_category = "Unknown"
                    confidence_item = QTableWidgetItem(f"{confidence:.3f} ({confidence_category})")
                    
                    if confidence >= 0.8:
                        confidence_item.setBackground(QColor(144, 238, 144))  # Light green
                    elif confidence >= 0.6:
                        confidence_item.setBackground(QColor(255, 255, 144))  # Light yellow
                    elif confidence >= 0.4:
                        confidence_item.setBackground(QColor(255, 210, 128))  # Light orange
                    else:
                        confidence_item.setBackground(QColor(255, 182, 193))  # Light red
                    self.translation_table.setItem(row, 2, confidence_item)
                    
                    # Frequency data (with null checks)
                    freq_data = mapping.get("frequency_data", {})
                    if not isinstance(freq_data, dict):
                        freq_data = {}
                    observation_count = freq_data.get("observation_count", 0) or 0
                    recent_activity = freq_data.get("recent_activity", 0) or 0
                    frequency_text = f"{observation_count} obs ({recent_activity} recent)"
                    self.translation_table.setItem(row, 3, QTableWidgetItem(frequency_text))
                    
                # Add token to filter if not already there
                if compressed_token and self.token_filter.findText(compressed_token) == -1:
                    self.token_filter.addItem(compressed_token)
            
                # Sort by confidence (highest first) using Qt sort order constants
                from PyQt6.QtCore import Qt
                self.translation_table.sortItems(2, Qt.SortOrder.DescendingOrder)  # Sort by confidence column, descending
                
                self.translation_table.resizeColumnsToContents()
                
            elif isinstance(mappings, dict) and mappings.get("status") == "translation_mappings_not_available":
                # Handle case where translation system isn't available yet
                self.translation_table.setRowCount(1)
                self.translation_table.setItem(0, 0, QTableWidgetItem("System initializing..."))
                self.translation_table.setItem(0, 1, QTableWidgetItem("Please wait"))
                self.translation_table.setItem(0, 2, QTableWidgetItem("N/A"))
                self.translation_table.setItem(0, 3, QTableWidgetItem("N/A"))
            else:
                # No mappings available yet or mappings is None
                self.translation_table.setRowCount(1)
                if mappings is None:
                    self.translation_table.setItem(0, 0, QTableWidgetItem("Background processor not initialized"))
                    self.translation_table.setItem(0, 1, QTableWidgetItem("Check system logs"))
                else:
                    self.translation_table.setItem(0, 0, QTableWidgetItem("No translation mappings yet"))
                    self.translation_table.setItem(0, 1, QTableWidgetItem("Send agent messages to generate"))
                self.translation_table.setItem(0, 2, QTableWidgetItem("N/A"))
                self.translation_table.setItem(0, 3, QTableWidgetItem("N/A"))
            
        except Exception as e:
            # Show error in details pane and table
            self.token_details.setText(f"Error refreshing translation data: {str(e)}")
            self.translation_table.setRowCount(1)
            self.translation_table.setItem(0, 0, QTableWidgetItem(f"Error: {str(e)}"))
            self.translation_table.setItem(0, 1, QTableWidgetItem("Check logs"))
            self.translation_table.setItem(0, 2, QTableWidgetItem("N/A"))
            self.translation_table.setItem(0, 3, QTableWidgetItem("N/A"))
    
    def filter_translations(self, filter_text):
        """Filter translations by token"""
        if filter_text == "All Tokens":
            # Show all rows
            for row in range(self.translation_table.rowCount()):
                self.translation_table.setRowHidden(row, False)
        else:
            # Hide rows that don't match the filter
            for row in range(self.translation_table.rowCount()):
                token_item = self.translation_table.item(row, 0)
                if token_item:
                    self.translation_table.setRowHidden(row, token_item.text() != filter_text)
    
    def show_token_details(self):
        """Show details for selected token"""
        selected_items = self.translation_table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            token = self.translation_table.item(row, 0).text()
            translation = self.translation_table.item(row, 1).text()
            confidence = self.translation_table.item(row, 2).text()
            
            details = f"Token: {token}\n"
            details += f"Translation: {translation}\n"
            details += f"Confidence: {confidence}\n"
            details += f"Analysis: This mapping was generated through semantic gravity analysis "
            details += f"and context correlation patterns."
            
            self.token_details.setText(details)
    
    def clear_mappings(self):
        """Clear all translation mappings (future functionality)"""
        self.token_details.setText("Clear mappings functionality would be implemented here.")


class CategorizationAnalysisTab(QWidget):
    """Analysis of the agent categorization system"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize categorization analysis UI"""
        layout = QVBoxLayout(self)
        
        # Control panel
        controls_layout = QHBoxLayout()
        
        # Category filter
        controls_layout.addWidget(QLabel("Category Type:"))
        self.category_filter = QComboBox()
        self.category_filter.addItem("All Categories")
        self.category_filter.addItem("Agent Generated")
        self.category_filter.addItem("User Defined")
        self.category_filter.addItem("Background Generated")
        self.category_filter.currentTextChanged.connect(self.filter_categories)
        controls_layout.addWidget(self.category_filter)
        
        controls_layout.addStretch()
        
        # Refresh button
        refresh_button = QPushButton("Refresh Categories")
        refresh_button.clicked.connect(self.manual_refresh)
        controls_layout.addWidget(refresh_button)
        
        layout.addLayout(controls_layout)
        
        # Create splitter for two-panel layout
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Category list and stats
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Category overview stats
        stats_group = QGroupBox("Category Statistics")
        stats_layout = QGridLayout(stats_group)
        
        self.total_categories_label = QLabel("Total Categories: 0")
        self.agent_categories_label = QLabel("Agent Generated: 0")
        self.user_categories_label = QLabel("User Defined: 0")
        self.active_categories_label = QLabel("Active Categories: 0")
        self.avg_usage_label = QLabel("Average Usage: 0")
        
        stats_layout.addWidget(self.total_categories_label, 0, 0)
        stats_layout.addWidget(self.agent_categories_label, 0, 1)
        stats_layout.addWidget(self.user_categories_label, 1, 0)
        stats_layout.addWidget(self.active_categories_label, 1, 1)
        stats_layout.addWidget(self.avg_usage_label, 2, 0, 1, 2)
        
        left_layout.addWidget(stats_group)
        
        # Category table
        self.category_table = QTableWidget()
        self.category_table.setColumnCount(5)
        self.category_table.setHorizontalHeaderLabels([
            "Category", "Type", "Usage Count", "Last Used", "Confidence"
        ])
        self.category_table.setSortingEnabled(True)
        self.category_table.horizontalHeader().setStretchLastSection(True)
        self.category_table.itemSelectionChanged.connect(self.show_category_details)
        
        left_layout.addWidget(self.category_table)
        
        splitter.addWidget(left_widget)
        
        # Right panel: Category details and evolution
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Category details
        details_group = QGroupBox("Category Details")
        details_layout = QVBoxLayout(details_group)
        
        self.category_name_label = QLabel("Category: None selected")
        self.category_name_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        details_layout.addWidget(self.category_name_label)
        
        self.category_info = QTextEdit()
        self.category_info.setMaximumHeight(120)
        self.category_info.setReadOnly(True)
        details_layout.addWidget(self.category_info)
        
        right_layout.addWidget(details_group)
        
        # Category evolution timeline
        evolution_group = QGroupBox("Category Evolution Timeline")
        evolution_layout = QVBoxLayout(evolution_group)
        
        self.evolution_list = QTableWidget()
        self.evolution_list.setColumnCount(3)
        self.evolution_list.setHorizontalHeaderLabels([
            "Timestamp", "Event Type", "Description"
        ])
        self.evolution_list.horizontalHeader().setStretchLastSection(True)
        
        evolution_layout.addWidget(self.evolution_list)
        right_layout.addWidget(evolution_group)
        
        # Category relationships
        relationships_group = QGroupBox("Related Categories")
        relationships_layout = QVBoxLayout(relationships_group)
        
        self.relationships_list = QTableWidget()
        self.relationships_list.setColumnCount(3)
        self.relationships_list.setHorizontalHeaderLabels([
            "Related Category", "Relationship Type", "Strength"
        ])
        self.relationships_list.horizontalHeader().setStretchLastSection(True)
        
        relationships_layout.addWidget(self.relationships_list)
        right_layout.addWidget(relationships_group)
        
        splitter.addWidget(right_widget)
        
        # Set splitter proportions
        splitter.setSizes([400, 600])
        layout.addWidget(splitter)
        
    def manual_refresh(self):
        """Manually refresh categorization data"""
        # This will be called by the parent refresh system
        pass
        
    def refresh_data(self, agent):
        """Refresh categorization data"""
        try:
            # Get categorization statistics
            cat_stats = agent.get_categorization_stats()
            
            if cat_stats.get("status") == "active":
                # Update statistics
                self.total_categories_label.setText(f"Total Categories: {cat_stats.get('agent_categories', 0)}")
                self.agent_categories_label.setText(f"Agent Generated: {cat_stats.get('agent_generated', 0)}")
                self.user_categories_label.setText(f"User Defined: {cat_stats.get('user_defined', 0)}")
                self.active_categories_label.setText(f"Active Categories: {cat_stats.get('active_categories', 0)}")
                self.avg_usage_label.setText(f"Average Usage: {cat_stats.get('avg_usage', 0):.1f}")
                
                # Get detailed category data
                category_data = self._get_detailed_category_data(agent)
                self._populate_category_table(category_data)
                
            else:
                # Show system not available
                self.category_table.setRowCount(1)
                self.category_table.setItem(0, 0, QTableWidgetItem("Categorization system not active"))
                self.category_table.setItem(0, 1, QTableWidgetItem("Initializing..."))
                self.category_table.setItem(0, 2, QTableWidgetItem("N/A"))
                self.category_table.setItem(0, 3, QTableWidgetItem("N/A"))
                self.category_table.setItem(0, 4, QTableWidgetItem("N/A"))
                
        except Exception as e:
            self.category_info.setText(f"Error refreshing categorization data: {str(e)}")
    
    def _get_detailed_category_data(self, agent):
        """Get detailed category data from the agent"""
        try:
            # Try to get detailed categorization data
            if (hasattr(agent.memory_bank, 'agent_categorizer') and 
                agent.memory_bank.agent_categorizer):
                
                categorizer = agent.memory_bank.agent_categorizer
                
                # Get all categories with usage statistics
                categories = []
                
                # Get categories from categorizer
                if hasattr(categorizer, 'categories'):
                    for category_name, category_data in categorizer.categories.items():
                        category_info = {
                            "name": category_name,
                            "type": self._determine_category_type(category_data),
                            "usage_count": category_data.get("usage_count", 0),
                            "last_used": category_data.get("last_used", "Never"),
                            "confidence": category_data.get("confidence", 0.0),
                            "created": category_data.get("created", "Unknown"),
                            "description": category_data.get("description", ""),
                            "examples": category_data.get("examples", []),
                            "evolution": category_data.get("evolution_history", [])
                        }
                        categories.append(category_info)
                
                # Also get background-generated categories
                if hasattr(categorizer, 'background_processor') and categorizer.background_processor:
                    bg_processor = categorizer.background_processor
                    if hasattr(bg_processor, 'translation_candidates'):
                        for token, candidates in bg_processor.translation_candidates.items():
                            for candidate in candidates:
                                confidence = bg_processor.confidence_scores.get((token, candidate), 0.0)
                                category_info = {
                                    "name": f"{token}→{candidate}",
                                    "type": "Background Translation",
                                    "usage_count": len(bg_processor.compressed_token_cache.get(token, [])),
                                    "last_used": "Recently active",
                                    "confidence": confidence,
                                    "created": "Background generated",
                                    "description": f"Translation mapping from compressed token '{token}' to human concept '{candidate}'",
                                    "examples": [token],
                                    "evolution": []
                                }
                                categories.append(category_info)
                
                return categories
                
        except Exception as e:
            return [{"name": f"Error: {str(e)}", "type": "Error", "usage_count": 0, "last_used": "N/A", "confidence": 0.0}]
            
        return []
    
    def _determine_category_type(self, category_data):
        """Determine the type of category based on its data"""
        source = category_data.get("source", "unknown")
        creation_method = category_data.get("creation_method", "")
        
        if "user" in source.lower() or "manual" in creation_method.lower():
            return "User Defined"
        elif "agent" in source.lower() or "automatic" in creation_method.lower():
            return "Agent Generated"
        elif "background" in source.lower() or "gravity" in creation_method.lower():
            return "Background Generated"
        else:
            return "Unknown"
    
    def _populate_category_table(self, categories):
        """Populate the category table with data"""
        self.category_table.setRowCount(len(categories))
        
        for row, category in enumerate(categories):
            # Category name
            name_item = QTableWidgetItem(category["name"])
            self.category_table.setItem(row, 0, name_item)
            
            # Type with color coding
            type_item = QTableWidgetItem(category["type"])
            if category["type"] == "Agent Generated":
                type_item.setBackground(QColor(144, 238, 144))  # Light green
            elif category["type"] == "User Defined":
                type_item.setBackground(QColor(173, 216, 230))  # Light blue
            elif category["type"] == "Background Generated":
                type_item.setBackground(QColor(255, 255, 144))  # Light yellow
            elif category["type"] == "Background Translation":
                type_item.setBackground(QColor(255, 210, 128))  # Light orange
            self.category_table.setItem(row, 1, type_item)
            
            # Usage count
            usage_item = QTableWidgetItem(str(category["usage_count"]))
            self.category_table.setItem(row, 2, usage_item)
            
            # Last used
            last_used_item = QTableWidgetItem(str(category["last_used"]))
            self.category_table.setItem(row, 3, last_used_item)
            
            # Confidence with color coding
            confidence = category["confidence"]
            confidence_item = QTableWidgetItem(f"{confidence:.3f}")
            if confidence >= 0.8:
                confidence_item.setBackground(QColor(144, 238, 144))  # Light green
            elif confidence >= 0.6:
                confidence_item.setBackground(QColor(255, 255, 144))  # Light yellow
            elif confidence >= 0.4:
                confidence_item.setBackground(QColor(255, 210, 128))  # Light orange
            else:
                confidence_item.setBackground(QColor(255, 182, 193))  # Light red
            self.category_table.setItem(row, 4, confidence_item)
        
        # Sort by usage count (descending)
        self.category_table.sortItems(2, Qt.SortOrder.DescendingOrder)
        self.category_table.resizeColumnsToContents()
    
    def filter_categories(self, filter_text):
        """Filter categories by type"""
        if filter_text == "All Categories":
            # Show all rows
            for row in range(self.category_table.rowCount()):
                self.category_table.setRowHidden(row, False)
        else:
            # Hide rows that don't match the filter
            for row in range(self.category_table.rowCount()):
                type_item = self.category_table.item(row, 1)
                if type_item:
                    should_hide = True
                    if filter_text == "Agent Generated" and type_item.text() == "Agent Generated":
                        should_hide = False
                    elif filter_text == "User Defined" and type_item.text() == "User Defined":
                        should_hide = False
                    elif filter_text == "Background Generated" and type_item.text() in ["Background Generated", "Background Translation"]:
                        should_hide = False
                    
                    self.category_table.setRowHidden(row, should_hide)
    
    def show_category_details(self):
        """Show details for selected category"""
        selected_items = self.category_table.selectedItems()
        if selected_items:
            row = selected_items[0].row()
            
            # Get table items with null checks
            name_item = self.category_table.item(row, 0)
            type_item = self.category_table.item(row, 1)
            usage_item = self.category_table.item(row, 2)
            last_used_item = self.category_table.item(row, 3)
            confidence_item = self.category_table.item(row, 4)
            
            # Extract text safely
            category_name = name_item.text() if name_item else "Unknown"
            category_type = type_item.text() if type_item else "Unknown"
            usage_count = usage_item.text() if usage_item else "0"
            last_used = last_used_item.text() if last_used_item else "Never"
            confidence = confidence_item.text() if confidence_item else "0.000"
            
            # Update category details
            self.category_name_label.setText(f"Category: {category_name}")
            
            details = f"Type: {category_type}\n"
            details += f"Usage Count: {usage_count}\n"
            details += f"Last Used: {last_used}\n"
            details += f"Confidence: {confidence}\n\n"
            
            if "→" in category_name:  # Translation mapping
                token, candidate = category_name.split("→", 1)
                details += f"This is a translation mapping from compressed token '{token}' "
                details += f"to human concept '{candidate}'. Generated through semantic gravity analysis "
                details += f"of contextual patterns and gravitational clustering."
            else:
                details += f"Category '{category_name}' represents a conceptual grouping "
                details += f"identified by the agent's categorization system."
            
            self.category_info.setText(details)
            
            # Clear evolution and relationships (could be enhanced later)
            self.evolution_list.setRowCount(1)
            self.evolution_list.setItem(0, 0, QTableWidgetItem(datetime.now().strftime('%H:%M:%S')))
            self.evolution_list.setItem(0, 1, QTableWidgetItem("Selection"))
            self.evolution_list.setItem(0, 2, QTableWidgetItem(f"Category '{category_name}' selected for analysis"))
            
            self.relationships_list.setRowCount(0)  # Clear for now


class SemanticGravityTab(QWidget):
    """Enhanced semantic gravity analysis with particle detail viewing"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize enhanced semantic gravity UI"""
        layout = QVBoxLayout(self)
        
        # Import the enhanced analyzer
        try:
            from .utils.enhanced_semantic_gravity import EnhancedSemanticGravityAnalyzer
            self.enhanced_analyzer = EnhancedSemanticGravityAnalyzer(self)
            layout.addWidget(self.enhanced_analyzer)
        except ImportError as e:
            # Fallback to basic UI if enhanced analyzer not available
            layout.addWidget(QLabel(f"Enhanced analyzer not available: {e}"))
            self._create_fallback_ui(layout)
            
    def _create_fallback_ui(self, layout):
        """Create fallback UI if enhanced analyzer fails to load"""
        
        # Control panel
        controls_layout = QHBoxLayout()
        
        # Analysis type selector
        controls_layout.addWidget(QLabel("Analysis Type:"))
        self.analysis_type = QComboBox()
        self.analysis_type.addItem("Gravitational Clustering")
        self.analysis_type.addItem("Token Frequency Analysis")
        self.analysis_type.addItem("Spatial Distribution")
        self.analysis_type.addItem("Confidence Evolution")
        self.analysis_type.currentTextChanged.connect(self.change_analysis_view)
        controls_layout.addWidget(self.analysis_type)
        
        controls_layout.addStretch()
        
        # Time window selector
        controls_layout.addWidget(QLabel("Time Window:"))
        self.time_window = QComboBox()
        self.time_window.addItem("Last 5 minutes")
        self.time_window.addItem("Last 15 minutes")
        self.time_window.addItem("Last hour")
        self.time_window.addItem("All time")
        controls_layout.addWidget(self.time_window)
        
        layout.addLayout(controls_layout)
        
        # Create main content area with tabs
        self.content_tabs = QTabWidget()
        
        # Tab 1: Gravitational Clusters
        self.clusters_tab = self._create_clusters_tab()
        self.content_tabs.addTab(self.clusters_tab, "Gravitational Clusters")
        
        # Tab 2: Token Dynamics
        self.dynamics_tab = self._create_dynamics_tab()
        self.content_tabs.addTab(self.dynamics_tab, "Token Dynamics")
        
        # Tab 3: Spatial Analysis
        self.spatial_tab = self._create_spatial_tab()
        self.content_tabs.addTab(self.spatial_tab, "Spatial Analysis")
        
        layout.addWidget(self.content_tabs)
        
        # Status and summary
        self.status_label = QLabel("Semantic gravity analysis ready (fallback mode)")
        self.status_label.setStyleSheet("color: orange; font-style: italic;")
        layout.addWidget(self.status_label)
        
    def _create_clusters_tab(self):
        """Create the gravitational clusters analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Cluster overview
        overview_group = QGroupBox("Cluster Overview")
        overview_layout = QGridLayout(overview_group)
        
        self.cluster_count_label = QLabel("Active Clusters: 0")
        self.avg_cluster_size_label = QLabel("Average Size: 0")
        self.largest_cluster_label = QLabel("Largest Cluster: 0 tokens")
        self.cluster_density_label = QLabel("Density: 0.0")
        
        overview_layout.addWidget(self.cluster_count_label, 0, 0)
        overview_layout.addWidget(self.avg_cluster_size_label, 0, 1)
        overview_layout.addWidget(self.largest_cluster_label, 1, 0)
        overview_layout.addWidget(self.cluster_density_label, 1, 1)
        
        layout.addWidget(overview_group)
        
        # Cluster details table
        self.clusters_table = QTableWidget()
        self.clusters_table.setColumnCount(5)
        self.clusters_table.setHorizontalHeaderLabels([
            "Cluster ID", "Token Count", "Center Position", "Density", "Representative Tokens"
        ])
        self.clusters_table.setSortingEnabled(True)
        self.clusters_table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(self.clusters_table)
        
        return widget
        
    def _create_dynamics_tab(self):
        """Create the token dynamics analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Frequency analysis
        freq_group = QGroupBox("Token Frequency Analysis")
        freq_layout = QVBoxLayout(freq_group)
        
        # Statistics
        freq_stats_layout = QGridLayout()
        self.total_observations_label = QLabel("Total Observations: 0")
        self.unique_tokens_label = QLabel("Unique Tokens: 0")
        self.avg_frequency_label = QLabel("Average Frequency: 0.0")
        self.most_active_token_label = QLabel("Most Active: None")
        
        freq_stats_layout.addWidget(self.total_observations_label, 0, 0)
        freq_stats_layout.addWidget(self.unique_tokens_label, 0, 1)
        freq_stats_layout.addWidget(self.avg_frequency_label, 1, 0)
        freq_stats_layout.addWidget(self.most_active_token_label, 1, 1)
        
        freq_layout.addLayout(freq_stats_layout)
        
        # Token frequency table
        self.frequency_table = QTableWidget()
        self.frequency_table.setColumnCount(6)
        self.frequency_table.setHorizontalHeaderLabels([
            "Token", "Total Count", "Recent Activity", "Growth Rate", "Last Seen", "Burst Pattern"
        ])
        self.frequency_table.setSortingEnabled(True)
        self.frequency_table.horizontalHeader().setStretchLastSection(True)
        
        freq_layout.addWidget(self.frequency_table)
        layout.addWidget(freq_group)
        
        return widget
        
    def _create_spatial_tab(self):
        """Create the spatial analysis tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Spatial statistics
        spatial_group = QGroupBox("Spatial Distribution Analysis")
        spatial_layout = QVBoxLayout(spatial_group)
        
        spatial_stats_layout = QGridLayout()
        self.positioned_tokens_label = QLabel("Positioned Tokens: 0")
        self.spatial_spread_label = QLabel("Spatial Spread: 0.0")
        self.avg_distance_label = QLabel("Avg Distance: 0.0")
        self.hotspots_label = QLabel("Dense Regions: 0")
        
        spatial_stats_layout.addWidget(self.positioned_tokens_label, 0, 0)
        spatial_stats_layout.addWidget(self.spatial_spread_label, 0, 1)
        spatial_stats_layout.addWidget(self.avg_distance_label, 1, 0)
        spatial_stats_layout.addWidget(self.hotspots_label, 1, 1)
        
        spatial_layout.addLayout(spatial_stats_layout)
        
        # Spatial positions table
        self.spatial_table = QTableWidget()
        self.spatial_table.setColumnCount(6)
        self.spatial_table.setHorizontalHeaderLabels([
            "Token", "X Position", "Y Position", "Z Position", "Nearest Neighbors", "Cluster Assignment"
        ])
        self.spatial_table.setSortingEnabled(True)
        self.spatial_table.horizontalHeader().setStretchLastSection(True)
        
        spatial_layout.addWidget(self.spatial_table)
        layout.addWidget(spatial_group)
        
        return widget
        
    def change_analysis_view(self, analysis_type):
        """Change the active analysis view"""
        if analysis_type == "Gravitational Clustering":
            self.content_tabs.setCurrentIndex(0)
        elif analysis_type == "Token Frequency Analysis":
            self.content_tabs.setCurrentIndex(1)
        elif analysis_type == "Spatial Distribution":
            self.content_tabs.setCurrentIndex(2)
        
    def refresh_data(self, agent):
        """Refresh semantic gravity data"""
        try:
            # If we have the enhanced analyzer, delegate to it
            if hasattr(self, 'enhanced_analyzer'):
                # The enhanced analyzer handles its own refresh
                pass
            else:
                # Fallback refresh for basic UI
                self._fallback_refresh_data(agent)
                
        except Exception as e:
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"Error refreshing semantic gravity data: {str(e)}")
                self.status_label.setStyleSheet("color: red;")
    
    def _fallback_refresh_data(self, agent):
        """Fallback refresh method for basic UI"""
        try:
            # Get background processor stats
            bg_stats = agent.get_background_processor_stats()
            
            if bg_stats.get("background_running"):
                # Get detailed semantic gravity data
                gravity_data = self._get_semantic_gravity_data(agent)
                
                # Update clusters analysis
                self._update_clusters_analysis(gravity_data)
                
                # Update token dynamics
                self._update_dynamics_analysis(gravity_data)
                
                # Update spatial analysis
                self._update_spatial_analysis(gravity_data)
                
                self.status_label.setText(f"Last updated: {datetime.now().strftime('%H:%M:%S')} - {gravity_data.get('status', 'Active')}")
                self.status_label.setStyleSheet("color: green;")
                
            else:
                self.status_label.setText("Semantic gravity background processor not running")
                self.status_label.setStyleSheet("color: orange;")
                
        except Exception as e:
            self.status_label.setText(f"Error refreshing semantic gravity data: {str(e)}")
            self.status_label.setStyleSheet("color: red;")
    
    def _get_semantic_gravity_data(self, agent):
        """Get semantic gravity data from the agent"""
        try:
            gravity_data = {
                "status": "unknown",
                "clusters": {},
                "token_frequencies": {},
                "spatial_positions": {},
                "temporal_patterns": {}
            }
            
            # Try to access the background processor directly
            if (hasattr(agent.memory_bank, 'agent_categorizer') and 
                agent.memory_bank.agent_categorizer and
                hasattr(agent.memory_bank.agent_categorizer, 'background_processor')):
                
                bg_processor = agent.memory_bank.agent_categorizer.background_processor
                
                if bg_processor:
                    gravity_data["status"] = "active"
                    
                    # Get token frequency data
                    gravity_data["token_frequencies"] = dict(bg_processor.compressed_token_cache)
                    
                    # Get translation candidates (these represent gravitational attractions)
                    gravity_data["translation_mappings"] = dict(bg_processor.translation_candidates)
                    gravity_data["confidence_scores"] = dict(bg_processor.confidence_scores)
                    
                    # Get context correlations
                    gravity_data["context_correlations"] = dict(bg_processor.context_correlations)
                    
                    # Generate cluster analysis from translation mappings
                    gravity_data["clusters"] = self._analyze_gravitational_clusters(bg_processor)
                    
                    # Get spatial data if available
                    gravity_data["spatial_positions"] = self._get_spatial_positions(agent)
                    
            return gravity_data
            
        except Exception as e:
            return {"status": f"error: {str(e)}", "clusters": {}, "token_frequencies": {}, "spatial_positions": {}}
    
    def _analyze_gravitational_clusters(self, bg_processor):
        """Analyze gravitational clusters from background processor data"""
        clusters = {}
        
        try:
            # Group tokens by their human translations (these form gravitational clusters)
            translation_groups = {}
            
            for token, candidates in bg_processor.translation_candidates.items():
                for candidate in candidates:
                    if candidate not in translation_groups:
                        translation_groups[candidate] = []
                    translation_groups[candidate].append(token)
            
            # Convert to cluster format
            cluster_id = 0
            for human_concept, tokens in translation_groups.items():
                if len(tokens) > 1:  # Only clusters with multiple tokens
                    clusters[f"cluster_{cluster_id}"] = {
                        "center_concept": human_concept,
                        "tokens": tokens,
                        "size": len(tokens),
                        "density": self._calculate_cluster_density(tokens, bg_processor),
                        "representative_tokens": tokens[:3]  # Show first 3 as representatives
                    }
                    cluster_id += 1
            
        except Exception as e:
            clusters["error"] = {"tokens": [f"Error: {str(e)}"], "size": 0, "density": 0.0}
            
        return clusters
    
    def _calculate_cluster_density(self, tokens, bg_processor):
        """Calculate density of a gravitational cluster"""
        try:
            # Calculate density based on confidence scores and frequency
            total_confidence = 0.0
            total_frequency = 0
            count = 0
            
            for token in tokens:
                # Get frequency data
                freq_data = bg_processor.compressed_token_cache.get(token, [])
                total_frequency += len(freq_data)
                
                # Get average confidence for this token
                token_confidences = [
                    conf for (t, candidate), conf in bg_processor.confidence_scores.items() 
                    if t == token
                ]
                if token_confidences:
                    total_confidence += sum(token_confidences) / len(token_confidences)
                    count += 1
            
            if count > 0:
                avg_confidence = total_confidence / count
                avg_frequency = total_frequency / len(tokens)
                # Combine confidence and frequency for density metric
                return (avg_confidence + min(avg_frequency / 10.0, 1.0)) / 2.0
            
        except Exception:
            pass
            
        return 0.0
    
    def _get_spatial_positions(self, agent):
        """Get spatial position data for tokens using field's spatial indexing"""
        spatial_data = {}
        
        try:
            # First try to get data from field's spatial grid for maximum accuracy
            if (hasattr(agent, 'particle_field') and agent.particle_field and 
                hasattr(agent.particle_field, 'spatial_grid')):
                
                field = agent.particle_field
                spatial_grid = field.spatial_grid
                
                # Get particles from spatial grid
                processed_particles = 0
                max_particles = 200  # Limit for performance
                
                for grid_key, particle_ids in spatial_grid.items():
                    if processed_particles >= max_particles:
                        break
                        
                    for particle_id in list(particle_ids)[:10]:  # Limit per grid cell
                        particle = field.get_particle_by_id(particle_id)
                        if particle and hasattr(particle, 'metadata'):
                            # Extract token from particle metadata
                            token = particle.metadata.get('token') or particle.metadata.get('content')
                            if token and hasattr(particle, 'position') and particle.position is not None:
                                try:
                                    # Extract x, y, z from particle position (12D array)
                                    pos = particle.position
                                    spatial_data[str(token)[:20]] = {  # Limit token length for display
                                        "x": float(pos[0]) if len(pos) > 0 else 0.0,
                                        "y": float(pos[1]) if len(pos) > 1 else 0.0,
                                        "z": float(pos[2]) if len(pos) > 2 else 0.0,
                                        "source": "field_spatial_grid",
                                        "grid_key": str(grid_key),
                                        "particle_type": particle.type if hasattr(particle, 'type') else "unknown"
                                    }
                                    processed_particles += 1
                                except (IndexError, TypeError, ValueError) as e:
                                    # Skip particles with invalid positions
                                    continue
                
                self.log(f"Retrieved {len(spatial_data)} spatial positions from field grid", "DEBUG", "_get_spatial_positions")
            
            # Fallback to lexicon store method if field data insufficient
            if len(spatial_data) < 10:
                if (hasattr(agent.memory_bank, 'agent_categorizer') and 
                    agent.memory_bank.agent_categorizer and
                    hasattr(agent.memory_bank.agent_categorizer.memory, 'lexicon_store')):
                    
                    lexicon_store = agent.memory_bank.agent_categorizer.memory.lexicon_store
                    
                    if hasattr(lexicon_store, 'get_tokens_with_positions'):
                        positioned_tokens = lexicon_store.get_tokens_with_positions(limit=100)
                        
                        for token_data in positioned_tokens:
                            token = token_data.get("token")
                            position = token_data.get("position", {})
                            
                            if token and position:
                                spatial_data[token] = {
                                    "x": position.get("x", 0.0),
                                    "y": position.get("y", 0.0),
                                    "z": position.get("z", 0.0),
                                    "source": "lexicon_store",
                                    "grid_key": "legacy",
                                    "particle_type": "lexicon"
                                }
            
        except Exception as e:
            spatial_data["error"] = {
                "x": 0, "y": 0, "z": 0, 
                "source": f"Error: {str(e)}", 
                "grid_key": "error", 
                "particle_type": "error"
            }
            
        return spatial_data
    
    def _update_clusters_analysis(self, gravity_data):
        """Update the clusters analysis tab"""
        clusters = gravity_data.get("clusters", {})
        
        # Update overview statistics
        cluster_count = len([c for c in clusters.values() if isinstance(c, dict) and c.get("size", 0) > 1])
        self.cluster_count_label.setText(f"Active Clusters: {cluster_count}")
        
        if clusters:
            sizes = [c.get("size", 0) for c in clusters.values() if isinstance(c, dict)]
            avg_size = sum(sizes) / len(sizes) if sizes else 0
            max_size = max(sizes) if sizes else 0
            avg_density = sum(c.get("density", 0) for c in clusters.values() if isinstance(c, dict)) / len(clusters)
            
            self.avg_cluster_size_label.setText(f"Average Size: {avg_size:.1f}")
            self.largest_cluster_label.setText(f"Largest Cluster: {max_size} tokens")
            self.cluster_density_label.setText(f"Density: {avg_density:.3f}")
        
        # Update clusters table
        self.clusters_table.setRowCount(len(clusters))
        
        for row, (cluster_id, cluster_data) in enumerate(clusters.items()):
            if isinstance(cluster_data, dict):
                self.clusters_table.setItem(row, 0, QTableWidgetItem(cluster_id))
                self.clusters_table.setItem(row, 1, QTableWidgetItem(str(cluster_data.get("size", 0))))
                self.clusters_table.setItem(row, 2, QTableWidgetItem(cluster_data.get("center_concept", "Unknown")))
                self.clusters_table.setItem(row, 3, QTableWidgetItem(f"{cluster_data.get('density', 0):.3f}"))
                self.clusters_table.setItem(row, 4, QTableWidgetItem(", ".join(cluster_data.get("representative_tokens", []))))
        
        self.clusters_table.resizeColumnsToContents()
    
    def _update_dynamics_analysis(self, gravity_data):
        """Update the token dynamics analysis tab"""
        token_frequencies = gravity_data.get("token_frequencies", {})
        
        # Update overview statistics
        total_observations = sum(len(timestamps) for timestamps in token_frequencies.values())
        unique_tokens = len(token_frequencies)
        avg_frequency = total_observations / unique_tokens if unique_tokens > 0 else 0
        
        self.total_observations_label.setText(f"Total Observations: {total_observations}")
        self.unique_tokens_label.setText(f"Unique Tokens: {unique_tokens}")
        self.avg_frequency_label.setText(f"Average Frequency: {avg_frequency:.1f}")
        
        # Find most active token
        if token_frequencies:
            most_active = max(token_frequencies.items(), key=lambda x: len(x[1]))
            self.most_active_token_label.setText(f"Most Active: {most_active[0]} ({len(most_active[1])} obs)")
        
        # Update frequency table
        self.frequency_table.setRowCount(len(token_frequencies))
        
        for row, (token, timestamps) in enumerate(token_frequencies.items()):
            # Calculate metrics
            total_count = len(timestamps)
            recent_activity = len([t for t in timestamps if t > (datetime.now().timestamp() - 300)])  # Last 5 minutes
            
            # Calculate growth rate
            if len(timestamps) > 1:
                recent_timestamps = [t for t in timestamps if t > (datetime.now().timestamp() - 900)]  # Last 15 minutes
                old_timestamps = [t for t in timestamps if t <= (datetime.now().timestamp() - 900)]
                recent_rate = len(recent_timestamps) / 15.0 if recent_timestamps else 0  # per minute
                old_rate = len(old_timestamps) / max((timestamps[-1] - timestamps[0]) / 60, 1) if old_timestamps else 0
                growth_rate = recent_rate - old_rate
            else:
                growth_rate = 0.0
            
            # Last seen
            last_seen = datetime.fromtimestamp(timestamps[-1]).strftime('%H:%M:%S') if timestamps else "Never"
            
            # Burst pattern detection
            if len(timestamps) > 2:
                intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
                avg_interval = sum(intervals) / len(intervals)
                short_intervals = [i for i in intervals if i < avg_interval * 0.3]
                burst_pattern = "Yes" if len(short_intervals) > len(intervals) * 0.3 else "No"
            else:
                burst_pattern = "Insufficient data"
            
            # Populate table
            self.frequency_table.setItem(row, 0, QTableWidgetItem(token))
            self.frequency_table.setItem(row, 1, QTableWidgetItem(str(total_count)))
            self.frequency_table.setItem(row, 2, QTableWidgetItem(str(recent_activity)))
            
            growth_item = QTableWidgetItem(f"{growth_rate:+.3f}")
            if growth_rate > 0:
                growth_item.setBackground(QColor(144, 238, 144))  # Light green
            elif growth_rate < 0:
                growth_item.setBackground(QColor(255, 182, 193))  # Light red
            self.frequency_table.setItem(row, 3, growth_item)
            
            self.frequency_table.setItem(row, 4, QTableWidgetItem(last_seen))
            self.frequency_table.setItem(row, 5, QTableWidgetItem(burst_pattern))
        
        # Sort by total count (descending)
        self.frequency_table.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.frequency_table.resizeColumnsToContents()
    
    def _update_spatial_analysis(self, gravity_data):
        """Update the spatial analysis tab"""
        spatial_positions = gravity_data.get("spatial_positions", {})
        
        # Update spatial statistics
        positioned_count = len([pos for pos in spatial_positions.values() if isinstance(pos, dict)])
        self.positioned_tokens_label.setText(f"Positioned Tokens: {positioned_count}")
        
        if positioned_count > 1:
            # Calculate spatial metrics
            positions = [pos for pos in spatial_positions.values() if isinstance(pos, dict)]
            
            # Calculate spatial spread (max distance between any two points)
            max_distance = 0.0
            total_distance = 0.0
            distance_count = 0
            
            for i, pos1 in enumerate(positions):
                for pos2 in positions[i+1:]:
                    distance = ((pos1["x"] - pos2["x"])**2 + 
                              (pos1["y"] - pos2["y"])**2 + 
                              (pos1["z"] - pos2["z"])**2)**0.5
                    max_distance = max(max_distance, distance)
                    total_distance += distance
                    distance_count += 1
            
            avg_distance = total_distance / distance_count if distance_count > 0 else 0.0
            
            self.spatial_spread_label.setText(f"Spatial Spread: {max_distance:.2f}")
            self.avg_distance_label.setText(f"Avg Distance: {avg_distance:.2f}")
            
            # Detect dense regions (simplified)
            hotspots = len([pos for pos in positions if pos.get("x", 0)**2 + pos.get("y", 0)**2 < avg_distance**2])
            self.hotspots_label.setText(f"Dense Regions: {hotspots}")
        
        # Update spatial table with null checks and enhanced spatial indexing data
        if not spatial_positions:
            # Show message when no spatial data available
            self.spatial_table.setRowCount(1)
            self.spatial_table.setItem(0, 0, QTableWidgetItem("No spatial data available"))
            self.spatial_table.setItem(0, 1, QTableWidgetItem("N/A"))
            self.spatial_table.setItem(0, 2, QTableWidgetItem("N/A"))
            self.spatial_table.setItem(0, 3, QTableWidgetItem("N/A"))
            self.spatial_table.setItem(0, 4, QTableWidgetItem("Field spatial indexing not initialized"))
            self.spatial_table.setItem(0, 5, QTableWidgetItem("Check particle field status"))
        else:
            self.spatial_table.setRowCount(len(spatial_positions))
            
            for row, (token, position) in enumerate(spatial_positions.items()):
                if isinstance(position, dict):
                    # Add null checks for all position data
                    token_str = str(token) if token is not None else "Unknown"
                    x_pos = position.get('x', 0.0) if position.get('x') is not None else 0.0
                    y_pos = position.get('y', 0.0) if position.get('y') is not None else 0.0
                    z_pos = position.get('z', 0.0) if position.get('z') is not None else 0.0
                    
                    # Enhanced spatial indexing information
                    source = position.get('source', 'unknown')
                    grid_key = position.get('grid_key', 'N/A')
                    particle_type = position.get('particle_type', 'unknown')
                    
                    self.spatial_table.setItem(row, 0, QTableWidgetItem(token_str))
                    self.spatial_table.setItem(row, 1, QTableWidgetItem(f"{x_pos:.3f}"))
                    self.spatial_table.setItem(row, 2, QTableWidgetItem(f"{y_pos:.3f}"))
                    self.spatial_table.setItem(row, 3, QTableWidgetItem(f"{z_pos:.3f}"))
                    
                    # Show grid key as neighbor info (spatial indexing information)
                    neighbor_info = f"Grid: {grid_key}" if grid_key != 'N/A' else "No grid data"
                    self.spatial_table.setItem(row, 4, QTableWidgetItem(neighbor_info))
                    
                    # Show source and particle type as cluster info
                    cluster_info = f"{source} ({particle_type})"
                    self.spatial_table.setItem(row, 5, QTableWidgetItem(cluster_info))
                else:
                    # Handle non-dict position data
                    self.spatial_table.setItem(row, 0, QTableWidgetItem(str(token)))
                    self.spatial_table.setItem(row, 1, QTableWidgetItem("Invalid"))
                    self.spatial_table.setItem(row, 2, QTableWidgetItem("Invalid"))
                    self.spatial_table.setItem(row, 3, QTableWidgetItem("Invalid"))
                    self.spatial_table.setItem(row, 4, QTableWidgetItem("Error"))
                    self.spatial_table.setItem(row, 5, QTableWidgetItem("Data corrupt"))
        
        self.spatial_table.resizeColumnsToContents()


class MemoryThermalTab(QWidget):
    """Memory thermal state analysis"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize memory thermal UI"""
        layout = QVBoxLayout(self)
        
        # Placeholder for memory thermal analysis
        layout.addWidget(QLabel("Memory Thermal States"))
        layout.addWidget(QLabel("This will show hot/warm/cool memory distributions and thermal evolution."))
        
    def refresh_data(self, agent):
        """Refresh memory thermal data"""
        pass  # Placeholder