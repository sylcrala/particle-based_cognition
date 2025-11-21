"""
Particle-based Cognition Engine - Memories GUI tab page
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

from PyQt6.QtWidgets import QWidget, QTabWidget, QTableWidget, QTableWidgetItem, QStackedLayout, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog, QLineEdit, QMessageBox
import os
from apis.api_registry import api
from qdrant_client import QdrantClient

class MemoriesTab(QWidget):
    def __init__(self):
        super().__init__()
        # memory setup
        self.config = api.get_api("config")
        self.agent_config = self.config.AGENT_CONFIG
        self.mem_dir = self.agent_config["memory_dir"]
        self.memory_api = None
        self.qdrant_client = None

        # UI setup
        self.foundation_layout = QVBoxLayout()
        self.setLayout(self.foundation_layout)

        self.header_layout = QVBoxLayout()
        self.foundation_layout.addLayout(self.header_layout, stretch = 1)

        self.content_layout = QVBoxLayout()
        self.foundation_layout.addLayout(self.content_layout, stretch = 10)
        self.content_widget = QTabWidget()
        self.content_layout.addWidget(self.content_widget)

        # Header
        header_label = QLabel("Memories")
        header_font = header_label.font()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header_label.setFont(header_font)
        self.header_layout.addWidget(header_label)

        # Control buttons layout
        controls_layout = QHBoxLayout()
        self.header_layout.addLayout(controls_layout)
        
        self.refresh_btn = QPushButton("Refresh Collections")   
        self.refresh_btn.clicked.connect(self.refresh_collections)
        controls_layout.addWidget(self.refresh_btn)
        
        # Search functionality
        search_label = QLabel("Search:")
        controls_layout.addWidget(search_label)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search memory content...")
        controls_layout.addWidget(self.search_input)
        
        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.search_memories)
        controls_layout.addWidget(self.search_btn)


    def refresh_collections(self):
        try:
            # Get the existing memory API and its client
            self.memory_api = api.get_api("_agent_memory")
            if self.memory_api is None:
                QMessageBox.critical(self, "Error", "Memory API not available. Please ensure the agent is running.")
                return
                
            # Access the existing Qdrant client from the memory bank
            self.qdrant_client = self.memory_api.client
            if self.qdrant_client is None:
                QMessageBox.critical(self, "Error", "Unable to access memory client. Please check your configuration.")
                return
                
            # Clear existing tabs
            self.content_widget.clear()

            # Get collections from the client
            collections_response = self.qdrant_client.get_collections()
            self.collections = collections_response.collections  # Extract the actual collections list
            
            if not self.collections:
                # No collections found, create a simple message
                no_collections_widget = QWidget()
                no_collections_layout = QVBoxLayout()
                no_collections_label = QLabel("No memory collections found.")
                no_collections_layout.addWidget(no_collections_label)
                no_collections_widget.setLayout(no_collections_layout)
                self.content_widget.addTab(no_collections_widget, "No Collections")
                return
                
            # Create widgets for each collection
            for collection in self.collections:
                collection_widget = self.create_collection_widget(collection)
                self.content_widget.addTab(collection_widget, collection.name)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to refresh collections: {str(e)}")
            print(f"Debug - Error details: {e}")  # For debugging

    def create_collection_widget(self, collection):
        widget = QWidget()
        layout = QVBoxLayout()
        header = QHBoxLayout()
        body = QVBoxLayout()
        widget.setLayout(layout)
        layout.addLayout(header)
        layout.addLayout(body)

        # Collection title
        title_label = QLabel(f"Collection: {collection.name}")
        title_font = title_label.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header.addWidget(title_label)

        # Memory count - get detailed collection info to access points_count
        try:
            # Get detailed collection info to access points_count
            collection_info = self.qdrant_client.get_collection(collection.name)
            points_count = collection_info.points_count
        except Exception as e:
            print(f"Error getting collection info for {collection.name}: {e}")
            points_count = "Unknown"
            
        count_label = QLabel(f"Points Count: {points_count}")
        header.addWidget(count_label)

        # Status
        status_label = QLabel(f"Status: {collection_info.status}")
        header.addWidget(status_label)

        try:
            # Get memories from this collection
            # Use scroll to get all points from the collection
            scroll_result = self.qdrant_client.scroll(
                collection_name=collection.name,
                limit=100,  # Get up to 100 memories at once
                with_payload=True,
                with_vectors=False  # Don't need the vectors for display
            )
            
            memories = scroll_result[0]  # First element is the list of points
            
            # Create memory viewer table
            memory_viewer = QTableWidget()
            memory_viewer.setColumnCount(4)
            memory_viewer.setHorizontalHeaderLabels(["Point ID", "Key", "Content", "Timestamp", "Metadata"])
            memory_viewer.setRowCount(len(memories))
            
            for i, memory in enumerate(memories):
                payload = memory.payload or {}

                # Point ID
                memory_viewer.setItem(i, 0, QTableWidgetItem(str(memory.id)))

                # Key
                key = payload.get("key", "No key")
                memory_viewer.setItem(i, 1, QTableWidgetItem(str(key))) 

                # Content (value)
                content = payload.get("value", "No content")
                if isinstance(content, dict):
                    content = str(content)
                memory_viewer.setItem(i, 1, QTableWidgetItem(str(content)[:100] + "..." if len(str(content)) > 100 else str(content)))
                
                # Timestamp
                timestamp = payload.get("timestamp", "Unknown")
                memory_viewer.setItem(i, 2, QTableWidgetItem(str(timestamp)))
                
                # Metadata (simplified)
                metadata_str = str(payload)[:50] + "..." if len(str(payload)) > 50 else str(payload)
                memory_viewer.setItem(i, 3, QTableWidgetItem(metadata_str))
            
            # Adjust column widths
            memory_viewer.resizeColumnsToContents()
            body.addWidget(memory_viewer)
            
        except Exception as e:
            # If there's an error getting memories, show an error message
            error_label = QLabel(f"Error loading memories: {str(e)}")
            body.addWidget(error_label)
            print(f"Debug - Error loading memories for {collection.name}: {e}")
        
        return widget

    def search_memories(self):
        """Search across all collections for matching content"""
        search_term = self.search_input.text().strip()
        if not search_term:
            QMessageBox.information(self, "Search", "Please enter a search term.")
            return
            
        if not hasattr(self, 'qdrant_client') or self.qdrant_client is None:
            QMessageBox.warning(self, "Search", "Please refresh collections first.")
            return
            
        try:
            # Create search results widget
            search_results_widget = QWidget()
            search_layout = QVBoxLayout()
            search_results_widget.setLayout(search_layout)
            
            # Search header
            search_header = QLabel(f"Search Results for: '{search_term}'")
            search_header_font = search_header.font()
            search_header_font.setPointSize(12)
            search_header_font.setBold(True)
            search_header.setFont(search_header_font)
            search_layout.addWidget(search_header)
            
            # Results table
            results_table = QTableWidget()
            results_table.setColumnCount(5)
            results_table.setHorizontalHeaderLabels(["Collection", "Point ID", "Content", "Timestamp", "Score"])
            
            all_results = []
            
            # Search each collection
            for collection in getattr(self, 'collections', []):
                try:
                    # Use scroll to search through the collection
                    scroll_result = self.qdrant_client.scroll(
                        collection_name=collection.name,
                        limit=100,
                        with_payload=True,
                        with_vectors=False,
                        scroll_filter=None  # We'll filter manually for text search
                    )
                    
                    memories = scroll_result[0]
                    
                    # Manual text search through payloads
                    for memory in memories:
                        payload = memory.payload or {}
                        content_str = str(payload.get("content", "")).lower()
                        
                        if search_term.lower() in content_str:
                            all_results.append({
                                "collection": collection.name,
                                "id": memory.id,
                                "key": payload.get("key", "No key"),
                                "content": payload.get("value", "No content"),
                                "timestamp": payload.get("timestamp", "Unknown"),
                                "score": content_str.count(search_term.lower())  # Simple relevance score
                            })
                            
                except Exception as e:
                    print(f"Error searching collection {collection.name}: {e}")
                    continue
            
            # Sort results by score (descending)
            all_results.sort(key=lambda x: x["score"], reverse=True)
            
            # Populate results table
            results_table.setRowCount(len(all_results))
            for i, result in enumerate(all_results):
                results_table.setItem(i, 0, QTableWidgetItem(result["collection"]))
                results_table.setItem(i, 1, QTableWidgetItem(str(result["id"])))
                results_table.setItem(i, 2, QTableWidgetItem(str(result["key"])))
                content = str(result["value"])
                results_table.setItem(i, 2, QTableWidgetItem(content))
                results_table.setItem(i, 3, QTableWidgetItem(str(result["timestamp"])))
                results_table.setItem(i, 4, QTableWidgetItem(str(result["score"])))
            
            results_table.resizeColumnsToContents()
            search_layout.addWidget(results_table)
            
            # Add search results as a new tab
            self.content_widget.addTab(search_results_widget, f"Search: {search_term}")
            self.content_widget.setCurrentWidget(search_results_widget)
            
            if not all_results:
                no_results = QLabel(f"No results found for '{search_term}'")
                search_layout.addWidget(no_results)
                
        except Exception as e:
            QMessageBox.critical(self, "Search Error", f"Search failed: {str(e)}")
            print(f"Search error details: {e}")


