import asyncio
import os
import signal
import sys
import time
import math
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from vispy import app
app.use_app("pyqt6")
from vispy import scene
from vispy.scene.visuals import Text, XYZAxis, Line, Markers
from pathlib import Path
import threading
import colorsys
from math import sin

# import API registry and initialize APIs
from apis.api_registry import api
from shared_services import config
from shared_services import logging
from shared_services import system_metrics
from apis.personal_tools.todo_list import todolist_api
from apis.research.external_resources import ExternalResources

# TODO: Implement PyQt6 GUI components here
# TODO: Use vispy for 3D field visualizer
# TODO: Refactor LLM usage as external component rather than core dependency (via external resources, similar to spaCy)
# TODO: integrate more particle types
# TODO: expand lexicon parsing

# Global logger
logger = api.get_api("logger")

def log_to_console(message, level="INFO", context = None):
    """Send log messages to both system logger and GUI"""

    context = context or "no_context"

    if logger:
        logger.log(message, level, context=context, source="MainApplication_GUI")

class FieldVisualizer(scene.SceneCanvas):
    def __init__(self, particle_provider):
        scene.SceneCanvas.__init__(self, keys="interactive")
        self.unfreeze()
        self.particle_provider = particle_provider
        self.agent_ready = False

        # 3d view
        self.view = self.central_widget.add_view()
        self.view.camera = "turntable"
        self.view.camera.fov = 45
        self.view.camera.distance = 5

        self.axis = XYZAxis(parent = self.view.scene)

        # position history for trails
        self.position_history = {}
        self.trail_length = 10

        # particle element dicts
        self.particle_visuals = {}
        self.shimmer_visuals = {}
        self.ghost_trail_visuals = {}

        try:
            self.status_text = Text("Waiting for agent initialization...", 
                                parent=self.view.scene,
                                color='white',
                                pos=(0, 0, 0),
                                font_size=20)
        except Exception as e:
            log_to_console(f"Error initializing status text: {e}", level="ERROR", context="FieldVisualizer")

        particle_types = ["memory", "lingual", "sensory", "core"] # add others as needed for new particle types
        try:
            for p_type in particle_types:
                self.particle_visuals[p_type] = Markers(parent=self.view.scene)
                self.shimmer_visuals[p_type] = Markers(parent=self.view.scene)
                self.ghost_trail_visuals[p_type] = Line(parent=self.view.scene)

                self.particle_visuals[p_type].set_data(
                    pos=np.array([[0, 0, 0]], dtype=np.float32),
                    size=np.array([0.001], dtype=np.float32),
                    face_color=np.array([[0, 0, 0, 0]], dtype=np.float32),
                )
                self.shimmer_visuals[p_type].set_data(
                    pos=np.array([[0, 0, 0]], dtype=np.float32),
                    size=np.array([0.001], dtype=np.float32),
                    face_color=np.array([[0, 0, 0, 0]], dtype=np.float32),
                )
                self.ghost_trail_visuals[p_type].set_data(
                    pos=np.array([[0, 0, 0]], dtype=np.float32),
                    color=np.array([[0, 0, 0, 0]], dtype=np.float32),
                )
        except Exception as e:
            log_to_console(f"Error initializing particle visuals: {e}", level="ERROR", context="FieldVisualizer")

        try:
            self.connection_lines = Line(parent=self.view.scene, connect = "segments", color = np.array([[0.5, 0.5, 1.0, 0.3]]), width = 1, antialias=True)
            self.connection_lines.set_data(
                pos=np.array([[0, 0, 0]], dtype=np.float32),
                color=np.array([[0, 0, 0, 0]], dtype=np.float32),
            )
        except Exception as e:
            log_to_console(f"Error initializing connection lines: {e}", level="ERROR", context="FieldVisualizer")

        self.x_dim = 0
        self.y_dim = 1
        self.z_dim = 2

        self.show_entanglements = True
        self.show_trails = True
        self.show_shimmer = True
        self.show_spatial_grid = True


        try:
            # Add spatial grid visualization
            self.grid_lines = Line(parent=self.view.scene, 
                                connect='segments',
                                color=np.array([[0.2, 0.2, 0.2, 0.3]]),
                                width=0.5,
                                antialias=True)
            self.grid_lines.set_data(
                pos=np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32),
                color=np.array([[0.2, 0.2, 0.2, 0.3], [0.2, 0.2, 0.2, 0.3]], dtype=np.float32),
            )
        except Exception as e:
            log_to_console(f"Error initializing spatial grid: {e}", level="ERROR", context="FieldVisualizer")
            self.show_spatial_grid = False


        QtCore.QTimer.singleShot(100, self.start_update_timer)

    def start_update_timer(self):
        try:
            self.update_timer = QtCore.QTimer()
            self.update_timer.timeout.connect(self.update_visualization)
            self.update_timer.start(30) # Update at ~30 FPS 
        except Exception as e:
            log_to_console(f"Error starting update timer: {e}", level="ERROR", context="FieldVisualizer")

    def set_dimension_mapping(self, x_dim, y_dim, z_dim):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

    def toggle_entanglements(self, visible):
        self.show_entanglements = visible
        self.connection_lines.visible = visible

    def toggle_trails(self, visible):
        self.show_trails = visible
        for trail in self.ghost_trail_visuals.values():
            trail.visible = visible

    def toggle_shimmer(self, visible):
        self.show_shimmer = visible
        for shimmer in self.shimmer_visuals.values():
            shimmer.visible = visible

    def safe_min_max(self, array, default_min=-1.0, default_max=1.0):
        """Safely compute min/max on potentially empty arrays"""
        if array is None or len(array) == 0:
            return default_min, default_max
        return np.min(array), np.max(array)
    
    def safe_update_visual(self, visual, visual_type, data=None):
        """Safely update visual without causing empty array errors"""
        try:
            if data is None or len(data) == 0:
                # Use invisible placeholder instead of empty arrays
                if visual_type == "markers":
                    visual.set_data(
                        pos=np.array([[0, 0, 0]], dtype=np.float32),
                        size=np.array([0.001], dtype=np.float32),
                        face_color=np.array([[0, 0, 0, 0]], dtype=np.float32)
                    )
                elif visual_type == "lines":
                    visual.set_data(
                        pos=np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32),
                        color=np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32)
                    )
            else:
                # Normal data update
                if visual_type == "markers":
                    visual.set_data(**data)
                elif visual_type == "lines":
                    visual.set_data(**data)
                    
        except Exception as e:
            print(f"Visual update error: {e}")
        
    def safe_color_bounds(self, color_values):
        """Ensure color values are within [0,1] range, logs if out of bounds"""
        original = color_values
        try:
            bounded = np.clip(color_values, 0.0, 1.0)

            if not np.array_equal(original, bounded):
                intensity = np.sum(np.maximum(original - 1.0, 0.0))
                log_to_console(f"Color values out of bounds, adjusted | Original colors: {original}, Bounded colors: {bounded}, Intensity: {intensity}", level="WARNING", context="FieldVisualizer")
        except Exception as e:
            log_to_console(f"Error bounding color values: {e}", level="ERROR", context="FieldVisualizer")
            import traceback
            log_to_console(f"Full traceback:\n{traceback.format_exc()}", level="ERROR", context="FieldVisualizer")

        return bounded

    def toggle_spatial_grid(self, visible):
        """Toggle visibility of the spatial indexing grid"""
        self.show_spatial_grid = visible
        self.grid_lines.visible = visible
    
    
    def update_visualization(self):
        """Update the 3D visualization with the latest particle data"""

        try:
            # QUICK SAFETY CHECK
            if not hasattr(self, 'particle_visuals') or not self.particle_visuals:
                return

            if self.particle_provider is None:
                field_api = api.get_api("_agent_field")
                if field_api and hasattr(field_api, "get_all_particles"):
                    self.particle_provider = field_api
                    log_to_console("Visualizer connected to particle field", context="update_visualization")

            if not self.particle_provider:
                if self.agent_ready:
                    self.agent_ready = False
                    self.status_text.visible = True
                    self.grid_lines.visible = self.show_spatial_grid
                return

            try:
                particles = self.particle_provider.get_all_particles()
                if not particles:
                    return

                # disabling status text upon agent ready and particle detection
                if particles and not self.agent_ready:
                    log_to_console(f"Visualizer found {len(particles)} particles in field", context="update_visualization")
                    self.agent_ready = True
                    self.status_text.visible = False
                    self.grid_lines.visible = self.show_spatial_grid

                # Update spatial grid visualization if enabled
                if self.show_spatial_grid and particles:
                    try:
                        positions = [p.render_particle()["position"] for p in particles]
                        pos_array = np.array(positions)
                        min_bounds = (np.min(pos_array, axis=0))
                        max_bounds = (np.max(pos_array, axis=0))
                        
                        # Create grid lines
                        grid_points = []
                        grid_extent = max(np.max(np.abs([min_bounds, max_bounds])) + 1.0, 2.0) # extent based on particle spread
                        grid_size = 0.5


                        # Create horizontal grid lines (in XZ plane)
                        for i in np.arange(-grid_extent, grid_extent + grid_size, grid_size):
                            grid_points.extend([[-grid_extent, i, -grid_extent], [grid_extent, i, -grid_extent]])
                            grid_points.extend([[-grid_extent, i, grid_extent], [grid_extent, i, grid_extent]])
                        
                        # Create depth grid lines (in XY plane)
                        for i in np.arange(-grid_extent, grid_extent + grid_size, grid_size):
                            grid_points.extend([[-grid_extent, -grid_extent, i], [grid_extent, -grid_extent, i]])
                            grid_points.extend([[-grid_extent, grid_extent, i], [grid_extent, grid_extent, i]])
                        
                        # Create width grid lines (in YZ plane)
                        for i in np.arange(-grid_extent, grid_extent + grid_size, grid_size):
                            grid_points.extend([[i, -grid_extent, -grid_extent], [i, grid_extent, -grid_extent]])
                            grid_points.extend([[i, -grid_extent, grid_extent], [i, grid_extent, grid_extent]])
                        
                        # Update grid lines
                        if grid_points:
                            grid_colors = np.tile([0.2, 0.2, 0.2, 0.15], (len(grid_points), 1))
                            self.grid_lines.set_data(
                                pos=np.array(grid_points),
                                color=grid_colors
                            )
                        else:
                            self.safe_update_visual(self.grid_lines, "lines", None)
                    except Exception as e:
                        log_to_console(f"Error updating spatial grid: {e}", level="ERROR", context="update_visualization")
                        self.safe_update_visual(self.grid_lines, "lines", None)

                # Group particles by type
                grouped_particles = {}
                for p_type in self.particle_visuals:
                    grouped_particles[p_type] = []
                
                # Get all particle render info
                particle_render_data = {}
                for particle in particles:
                    # Use particle's built-in rendering method with dimension mapping
                    render_info = particle.render_particle(dim_mapping={
                        "x": self.x_dim, 
                        "y": self.y_dim, 
                        "z": self.z_dim
                    })
                    
                    # Store by ID for entanglement resolution
                    particle_render_data[str(particle.id)] = render_info
                    
                    # Group by type
                    p_type = render_info["type"]
                    if p_type not in grouped_particles:
                        p_type = "default"
                        
                    grouped_particles[p_type].append(render_info)
                    
                    # Update position history for trails
                    if self.show_trails:
                        particle_id = str(particle.id)
                        if particle_id not in self.position_history:
                            self.position_history[particle_id] = [render_info["position"]] * self.trail_length
                        else:
                            self.position_history[particle_id].append(render_info["position"])
                            self.position_history[particle_id] = self.position_history[particle_id][-self.trail_length:]
                
                # Process entanglements if enabled
                connection_positions = []
                if self.show_entanglements:
                    for particle_id, render_info in particle_render_data.items():
                        for entanglement in render_info.get("entanglements", []):
                            target_id = entanglement["target_id"]
                            if str(target_id) in particle_render_data:
                                # Get positions of both particles
                                pos1 = render_info["position"]
                                pos2 = particle_render_data[str(target_id)]["position"]
                                
                                # Add line segment
                                connection_positions.extend([pos1, pos2])
                
                # Update connection lines 
                if connection_positions:
                    connection_colors = np.tile([0.5, 0.5, 1.0, 0.3], (len(connection_positions), 1))
                    self.connection_lines.set_data(
                        pos=np.array(connection_positions),
                        color=connection_colors
                    )
                else:
                    # Use single transparent point instead of empty array
                    self.connection_lines.set_data(
                        pos=np.array([[0, 0, 0]]), 
                        color=np.array([[0, 0, 0, 0]])  # Fully transparent
                    )
        

                # Update visuals for each particle type
                for p_type, particles_of_type in grouped_particles.items():
                    if not particles_of_type:
                        # Clear visuals if no particles of this type
                        self.safe_update_visual(self.particle_visuals[p_type], "markers", None)
                        self.safe_update_visual(self.shimmer_visuals[p_type], "markers", None)
                        self.safe_update_visual(self.ghost_trail_visuals[p_type], "lines", None)
                        continue
                        
                    # Prepare arrays for visualization
                    positions = []
                    sizes = []
                    colors = []
                    shimmer_positions = []
                    shimmer_sizes = []
                    shimmer_colors = []
                    trail_data = []
                    
                    for render_info in particles_of_type:
                        # Get position
                        pos = render_info["position"]
                        positions.append(pos)
                        
                        # Get size
                        size = render_info["size"] * 10  # Scale for visibility
                        sizes.append(size)
                        
                        # Get color from hue and saturation
                        hue = render_info["color_hue"] / 360.0  # Normalize to [0,1]
                        raw_saturation = render_info["color_saturation"]
                        
                        if raw_saturation > 1.0:
                            overflow_intensity = raw_saturation - 1.0
                            #log_to_console(f"Saturation out of bounds: {raw_saturation} (overflow: {overflow_intensity})", level="WARNING", context="update_visualization")

                            saturation = 0.85 + (0.15 * min(overflow_intensity, 1.0))
                        else:
                            saturation = raw_saturation

                        r, g, b = colorsys.hsv_to_rgb(hue, saturation, 1.0)
                        opacity = render_info["quantum_state"]["opacity"]
                        colors.append([r, g, b, opacity])

                        glow_value = render_info.get("glow", 0.0)
                        glow_intensity = render_info.get("glow_intensity", 0.0)
                        glow_polarity = render_info.get("glow_polarity", 1)
                        particle_frequency = render_info.get("pulse_rate", 1.0)

                        shimmer_r, shimmer_g, shimmer_b = r, g, b
                        shimmer_opacity = opacity * 0.7  # base shimmer opacity

                        # Check for shimmer effect
                        if self.show_shimmer and render_info["quantum_state"]["animation"] == "shimmer" and glow_intensity > 0.1:
                            if glow_polarity > 0:
                                shimmer_positions.append(pos)
                                shimmer_sizes.append(size * (1.2 + glow_intensity * 0.3))  # grows with intensity
                                glow_brightness = min(1.0 + glow_intensity * 0.4, 1.0)
                                shimmer_r = min(r * glow_brightness, 1.0)
                                shimmer_g = min(g * glow_brightness, 1.0)
                                shimmer_b = min(b * glow_brightness, 1.0)
                                pulse_frequency = particle_frequency + (glow_intensity * 0.5)
                                shimmer_opacity = opacity * (0.7 + 0.3 * math.sin(time.time() * pulse_frequency))

                                shimmer_colors.append([shimmer_r, shimmer_g, shimmer_b, shimmer_opacity])

                            elif glow_polarity < 0:
                                shimmer_positions.append(pos)
                                shimmer_sizes.append(size * (1.4 + glow_intensity * 0.5))  # Larger shadow area
                                shadow_strength = min(glow_intensity * 0.6, 0.8)
                                shimmer_r = max(r * (1.0 - shadow_strength), 0.0)
                                shimmer_g = max(g * (1.0 - shadow_strength), 0.0)
                                shimmer_b = max(b * (1.0 - shadow_strength), 0.0)
                                shimmer_b = min(shimmer_b + (shadow_strength * 0.2), 1.0)
                                pulse_frequency = particle_frequency + (glow_intensity * 0.3)
                                shimmer_opacity = opacity * (0.3 + 0.4 * math.sin(time.time() * pulse_frequency))

                                shimmer_colors.append([shimmer_r, shimmer_g, shimmer_b, shimmer_opacity])



                        # Check for ghost trails
                        if self.show_trails and render_info["quantum_state"]["ghost_trails"]:
                            # Get trail positions from history
                            particle_id = render_info["id"]
                            if str(particle_id) in self.position_history:
                                trail_positions = self.position_history[str(particle_id)]
                                trail_opacities = np.linspace(0.1, opacity, len(trail_positions))
                                
                                for i, trail_pos in enumerate(trail_positions):
                                    if i < len(trail_positions) - 1:  # Skip the last position (current position)
                                        trail_opacity = trail_opacities[i]
                                        trail_size = size * 0.8 * (i / len(trail_positions))
                                        trail_data.append((
                                            trail_pos,
                                            trail_size,
                                            [r, g, b, trail_opacity]
                                        ))
                    
                    # Convert to numpy arrays and update main particles
                    if positions:
                        self.particle_visuals[p_type].set_data(
                            pos=np.array(positions),
                            size=np.array(sizes),
                            face_color=np.array(colors),
                            edge_width=0
                        )
                    else:
                        self.particle_visuals[p_type].set_data([], size=[], face_color=np.empty((0, 4)))
                    
                    # Update shimmer effects
                    if shimmer_positions and self.show_shimmer:
                        shimmer_colors = np.clip(np.array(shimmer_colors), 0.0, 1.0).tolist() if shimmer_colors else [] #temporary clipping for test
                        assert len(shimmer_positions) == len(shimmer_sizes) == len(shimmer_colors), "Shimmer arrays length mismatch"
                        self.shimmer_visuals[p_type].set_data(
                            pos=np.array(shimmer_positions),
                            size=np.array(shimmer_sizes),
                            face_color=np.array(shimmer_colors),
                            edge_width=0
                        )
                    else:
                        # Use a single transparent point instead of empty arrays
                        # This avoids the "zero-size array" error
                        self.shimmer_visuals[p_type].set_data(
                            pos=np.array([[0, 0, 0]]),  # Single point at origin
                            size=np.array([0.0001]),    # Nearly invisible size
                            face_color=np.array([[0, 0, 0, 0]]),  # Fully transparent
                            edge_width=0
                        )

                # Update ghost trails - SAFE EMPTY HANDLING
                if not trail_data or len(trail_data) == 0 or not self.show_trails:
                    # Use single transparent point instead of empty array
                    self.safe_update_visual(self.ghost_trail_visuals[p_type], "lines", None)
                    
                else:
                    try:
                        trail_pos = np.array([t[0] for t in trail_data])
                        trail_colors = np.array([t[2] for t in trail_data])
                        trail_colors = np.clip(trail_colors, 0, 1)
                        
                        self.ghost_trail_visuals[p_type].set_data(
                            pos=trail_pos,
                            connect="strip",
                            color=trail_colors,
                        )
                    except Exception as e:
                        log_to_console(f"Error updating ghost trails: {e}", level="ERROR", context="update_visualization")
                        # Use single transparent point on error
                        self.safe_update_visual(self.ghost_trail_visuals[p_type], "lines", None)

                # Trigger canvas update
                self.update()

            except Exception as e:
                log_to_console(f"Error updating visualization: {e}", level="ERROR", context="update_visualization")
                import traceback
                log_to_console(f"Full traceback: {traceback.format_exc()}", level="ERROR", context="update_visualization")

                # Reset all visuals to safe empty state
                try:
                    for p_type in self.particle_visuals:
                        self.safe_update_visual(self.particle_visuals[p_type], "markers", None)
                        self.safe_update_visual(self.shimmer_visuals[p_type], "markers", None)
                        self.safe_update_visual(self.ghost_trail_visuals[p_type], "lines", None)
                    self.safe_update_visual(self.connection_lines, "lines", None)
                    self.safe_update_visual(self.grid_lines, "lines", None)
                except:
                    pass  # Last resort

                self.agent_ready = False
                self.status_text.visible = True

        except Exception as e:
            log_to_console(f"Unexpected error in visualization update: {e}", level="ERROR", context="update_visualization")
            import traceback
            log_to_console(f"Full traceback: {traceback.format_exc()}", level="ERROR", context="update_visualization")

            # Reset all visuals to safe empty state
            try:
                for p_type in self.particle_visuals:
                    self.safe_update_visual(self.particle_visuals[p_type], "markers", None)
                    self.safe_update_visual(self.shimmer_visuals[p_type], "markers", None)
                    self.safe_update_visual(self.ghost_trail_visuals[p_type], "lines", None)
                self.safe_update_visual(self.connection_lines, "lines", None)
                self.safe_update_visual(self.grid_lines, "lines", None)
            except:
                pass  # Last resort

            self.agent_ready = False
            self.status_text.visible = True

class MainWindow(QtWidgets.QMainWindow): 
    def __init__(self):
        super().__init__()
        
        self.field = self.get_field_api()
        
        # Set up the UI
        self.setWindowTitle("Local Kit - Quantum Particle System")
        self.resize(1280, 800)
        
        # Create main widget and layout
        main_widget = QtWidgets.QWidget()
        main_layout = QtWidgets.QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)
        
        # Create visualization panel (left side)
        visualization_panel = QtWidgets.QVBoxLayout()
        
        # Add VisPy canvas for particle visualization
        self.visualizer = FieldVisualizer(self.field) if self.field is not None else FieldVisualizer(None) 
        visualization_panel.addWidget(self.visualizer.native, stretch=4)
        
        # Add dimension controls
        dimension_group = QtWidgets.QGroupBox("Dimension Mapping")
        dimension_layout = QtWidgets.QGridLayout(dimension_group)
        
        # Labels for dimensions
        dimension_layout.addWidget(QtWidgets.QLabel("X Axis:"), 0, 0)
        dimension_layout.addWidget(QtWidgets.QLabel("Y Axis:"), 1, 0)
        dimension_layout.addWidget(QtWidgets.QLabel("Z Axis:"), 2, 0)
        
        # Create dimension selector combos
        dimension_names = [
            "Length (x)", "Width (y)", "Height (z)", 
            "Creation Time (w)", "Current Time (t)", "Age (a)",
            "Frequency (f)", "Memory Phase (m)", "Valence (v)",
            "Identity (i)", "Intent (n)", "Circadian Phase (p)"
        ]
        
        self.x_dim_selector = QtWidgets.QComboBox()
        self.y_dim_selector = QtWidgets.QComboBox()
        self.z_dim_selector = QtWidgets.QComboBox()
        
        for i, name in enumerate(dimension_names):
            self.x_dim_selector.addItem(name, i)
            self.y_dim_selector.addItem(name, i)
            self.z_dim_selector.addItem(name, i)
        
        # Set default selections
        self.x_dim_selector.setCurrentIndex(0)  # Length (x)
        self.y_dim_selector.setCurrentIndex(1)  # Width (y)
        self.z_dim_selector.setCurrentIndex(2)  # Height (z)
        
        # Connect change events
        self.x_dim_selector.currentIndexChanged.connect(self.update_dimension_mapping)
        self.y_dim_selector.currentIndexChanged.connect(self.update_dimension_mapping)
        self.z_dim_selector.currentIndexChanged.connect(self.update_dimension_mapping)
        
        # Add selectors to layout
        dimension_layout.addWidget(self.x_dim_selector, 0, 1)
        dimension_layout.addWidget(self.y_dim_selector, 1, 1)
        dimension_layout.addWidget(self.z_dim_selector, 2, 1)
        
        visualization_panel.addWidget(dimension_group)
        
        # Add visualization options group
        viz_options = QtWidgets.QGroupBox("Visualization Options")
        viz_options_layout = QtWidgets.QVBoxLayout(viz_options)
        
        # Create checkboxes for options
        self.show_entanglements = QtWidgets.QCheckBox("Show Entanglements")
        self.show_trails = QtWidgets.QCheckBox("Show Ghost Trails")
        self.show_shimmer = QtWidgets.QCheckBox("Show Shimmer Effects")
        self.show_grid = QtWidgets.QCheckBox("Show Spatial Grid")
        
        # Set defaults
        self.show_entanglements.setChecked(True)
        self.show_trails.setChecked(True)
        self.show_shimmer.setChecked(True)
        self.show_grid.setChecked(True)
        
        # Connect events
        self.show_entanglements.toggled.connect(self.visualizer.toggle_entanglements)
        self.show_trails.toggled.connect(self.visualizer.toggle_trails)
        self.show_shimmer.toggled.connect(self.visualizer.toggle_shimmer)
        self.show_grid.toggled.connect(self.visualizer.toggle_spatial_grid)
        
        # Add to layout
        viz_options_layout.addWidget(self.show_entanglements)
        viz_options_layout.addWidget(self.show_trails)
        viz_options_layout.addWidget(self.show_shimmer)
        viz_options_layout.addWidget(self.show_grid)

        visualization_panel.addWidget(viz_options)
        
        # Add legend for particle types
        self.add_particle_legend(visualization_panel)
        
        # Add visualization panel to main layout
        main_layout.addLayout(visualization_panel, 3)  # 3:1 ratio
        
        # Create control panel (right side)
        control_panel = QtWidgets.QVBoxLayout()
        
        # System controls
        system_group = QtWidgets.QGroupBox("System Controls")
        system_layout = QtWidgets.QVBoxLayout(system_group)
        
        # Add buttons for system control
        reflect_btn = QtWidgets.QPushButton("Trigger Reflection")
        reflect_btn.clicked.connect(self.trigger_reflection)
        system_layout.addWidget(reflect_btn)
        
        save_btn = QtWidgets.QPushButton("Save System State")
        save_btn.clicked.connect(self.save_system_state)
        system_layout.addWidget(save_btn)
        
        # TODO: Add system-wide age slider for time exploration
        # This would be implemented later after memory system fixes
        
        control_panel.addWidget(system_group)
        
        # System metrics
        metrics_group = QtWidgets.QGroupBox("System Metrics")
        metrics_layout = QtWidgets.QVBoxLayout(metrics_group)
        
        # Add labels for metrics
        self.particle_count = QtWidgets.QLabel("Total Particles: 0")
        self.system_energy = QtWidgets.QLabel("System Energy: 0.00")
        self.average_certainty = QtWidgets.QLabel("Average Certainty: 0.00")
        self.average_energy = QtWidgets.QLabel("Average Particle Energy: 0.00")
        self.average_activation = QtWidgets.QLabel("Average Activation Level: 0.00")

        metrics_layout.addWidget(self.particle_count)
        metrics_layout.addWidget(self.system_energy)
        metrics_layout.addWidget(self.average_certainty)
        metrics_layout.addWidget(self.average_energy)
        metrics_layout.addWidget(self.average_activation)

        # Update metrics timer
        self.metrics_timer = QtCore.QTimer()
        self.metrics_timer.timeout.connect(self.update_metrics)
        self.metrics_timer.start(1000)  # Update every second
        
        control_panel.addWidget(metrics_group)

        # Chat / Interaction panel
        chat_group = QtWidgets.QGroupBox("Communications")
        chat_layout = QtWidgets.QVBoxLayout(chat_group)

        self.chat_display = QtWidgets.QTextEdit()
        self.chat_display.setReadOnly(True)
        chat_layout.addWidget(self.chat_display)
        self.chat_input = QtWidgets.QLineEdit()
        self.chat_input.returnPressed.connect(lambda: self.send_chat_message())
        chat_layout.addWidget(self.chat_input)

        control_panel.addWidget(chat_group, stretch=2)
        
        # Log display
        log_group = QtWidgets.QGroupBox("System Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        
        self.log_display = QtWidgets.QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.document().setMaximumBlockCount(100)  # Limit for performance
        log_layout.addWidget(self.log_display)
        
        control_panel.addWidget(log_group, stretch=3)
        
        # Add control panel to main layout
        main_layout.addLayout(control_panel, 1)  # 3:1 ratio

        # Add agent status indicator
        self.agent_status = QtWidgets.QLabel("Agent Status: Initializing...")
        self.agent_status.setStyleSheet("color: orange; font-weight: bold;")
        control_panel.addWidget(self.agent_status)
        
        # Start a timer to check for agent availability
        self.agent_check_timer = QtCore.QTimer()
        self.agent_check_timer.timeout.connect(self.check_agent_availability)
        self.agent_check_timer.start(1000)  # Check every second

    def get_field_api(self):
        """Safely get the field API, returning None if not available"""
        try:
            field = api.get_api("_agent_field")
            if field and hasattr(field, "get_all_particles"):
                return field
        except Exception:
            return None
        return None
    
    def check_agent_availability(self):
        """Check if the agent and its components are available"""
        agent_available = api.get_api("agent") is not None
        field_available = api.get_api("_agent_field") is not None
        memory_available = api.get_api("_agent_memory") is not None
        
        if agent_available and field_available and memory_available:
            self.agent_status.setText("Agent Status: Ready")
            self.agent_status.setStyleSheet("color: green; font-weight: bold;")
            # Enable buttons that require the agent
            for child in self.findChildren(QtWidgets.QPushButton):
                if child.text() in ["Trigger Reflection", "Save System State"]:
                    child.setEnabled(True)
        else:
            components = []
            if not agent_available:
                components.append("Agent")
            if not field_available:
                components.append("Field")
            if not memory_available:
                components.append("Memory")
            
            missing = ", ".join(components)
            self.agent_status.setText(f"Agent Status: Initializing... (Waiting for: {missing})")
            self.agent_status.setStyleSheet("color: orange; font-weight: bold;")
            # Disable buttons that require the agent
            for child in self.findChildren(QtWidgets.QPushButton):
                if child.text() in ["Trigger Reflection", "Save System State"]:
                    child.setEnabled(False)

    def send_chat_message(self):
        """Handle chat input from UI"""
        message = self.chat_input.text().strip()
        if not message:
            return
        
        agent = api.get_api("_agent_anchor")

        self.chat_input.clear()
        self.update_chat_display(f"<b>You:</b> {message}")
        
        # Send event to agent event handler
        try:
            response = agent.send_message(
                message=message,
                source="gui_user"
            )
            self.update_chat_display(f"<b>Misty:</b> {response}")
        except Exception as e:
            logger.log(f"Error injecting chat message: {e}", level="ERROR", context="MainWindow")
            self.update_chat_display("<i>Error sending message to agent.</i>")
               
        
    @QtCore.pyqtSlot(str)
    def update_chat_display(self, message):
        """Thread-safe method to update chat display"""
        self.chat_display.append(message)

    def add_particle_legend(self, layout):
        """Add a legend showing particle types and colors"""
        legend_group = QtWidgets.QGroupBox("Particle Types")
        legend_layout = QtWidgets.QGridLayout(legend_group)
        
        # Define colors for each particle type
        types = {
            "Memory": 180,  # Cyan
            "Lingual": 285,  # Magenta
            "Sensory": 30,  # Orange
            "Core": 120,  # Green
        }
        
        # Add color indicators and labels
        row = 0
        for type_name, hue in types.items():
            # Convert hue to RGB
            r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 1.0, 1.0)
            rgb_color = (int(r * 255), int(g * 255), int(b * 255))
            
            # Create color indicator
            color_indicator = QtWidgets.QFrame()
            color_indicator.setFixedSize(16, 16)
            color_indicator.setStyleSheet(f"background-color: rgb({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]}); border: 1px solid black;")
            
            # Add to grid
            legend_layout.addWidget(color_indicator, row, 0)
            legend_layout.addWidget(QtWidgets.QLabel(type_name), row, 1)
            row += 1
        
        layout.addWidget(legend_group)
    
    def update_dimension_mapping(self):
        """Update which dimensions are mapped to which axis"""
        x_dim = self.x_dim_selector.currentData()
        y_dim = self.y_dim_selector.currentData()
        z_dim = self.z_dim_selector.currentData()
        
        self.visualizer.set_dimension_mapping(x_dim, y_dim, z_dim)
        log_to_console(f"Updated dimension mapping: X={x_dim}, Y={y_dim}, Z={z_dim}")
    
    def update_metrics(self):
        """Update system metrics display"""
        try:
            field = api.get_api("_agent_field")
            if not field:
                self.particle_count.setText("Total Particles: N/A")
                self.system_energy.setText("System Energy: N/A")
                self.average_certainty.setText("Average Certainty: N/A")
                self.average_energy.setText("Average Particle Energy: N/A")
                self.average_activation.setText("Average Activation Level: N/A")
                return

            particles = field.get_all_particles()
            if not particles:
                self.particle_count.setText("Total Particles: 0")
                self.system_energy.setText("System Energy: 0.00")
                self.average_certainty.setText("Average Certainty: 0.00")
                self.average_energy.setText("Average Particle Energy: 0.00")
                self.average_activation.setText("Average Activation Level: 0.00")
                return
            
            # Update metrics safely
            self.particle_count.setText(f"Total Particles: {len(particles)}")
            
            # Safely access energy attribute
            total_energy = 0
            for p in particles:
                if hasattr(p, 'energy'):
                    total_energy += p.energy
            self.system_energy.setText(f"System Energy: {total_energy:.2f}")
            
            # Safely access superposition attribute
            valid_particles = 0
            certainty_sum = 0
            for p in particles:
                if hasattr(p, 'superposition') and isinstance(p.superposition, dict):
                    # Handle both string and integer indices
                    if isinstance(p.superposition, dict) and 'certain' in p.superposition:
                        certainty_sum += p.superposition['certain']
                        valid_particles += 1
            
            avg_certainty = certainty_sum / valid_particles if valid_particles > 0 else 0
            self.average_certainty.setText(f"Average Certainty: {avg_certainty:.2f}")
            
            # Average particle energy
            avg_energy = total_energy / len(particles) if len(particles) > 0 else 0
            self.average_energy.setText(f"Average Particle Energy: {avg_energy:.2f}")

            # Average activation level
            activation_sum = 0
            for p in particles:
                if hasattr(p, 'activation'):
                    activation_sum += p.activation
            avg_activation = activation_sum / len(particles) if len(particles) > 0 else 0
            self.average_activation.setText(f"Average Activation Level: {avg_activation:.2f}")
            

        except Exception as e:
            self.particle_count.setText("Total Particles: Error")
            self.system_energy.setText(f"System Energy: Error ({str(e)[:20]}...)")
            self.average_certainty.setText("Average Certainty: Error")
            self.average_energy.setText("Average Energy: Error")
            self.average_activation.setText("Average Activation: Error")
            log_to_console(f"Error updating metrics: {str(e)}", "ERROR")
    
    def trigger_reflection(self):
        """Trigger a reflection cycle in the agent"""
        try:
            agent = api.get_api("_agent_anchor")
            if not agent:
                self.add_to_log("Agent not initialized yet", "WARNING")
                return
                
            try:
                agent.emit_event("reflection_triggered")

            except Exception as e:
                self.add_to_log(f"Error triggering reflection: {str(e)}", "ERROR")
                return
        except Exception as e:
            self.add_to_log(f"Error triggering reflection: {str(e)}", "ERROR")

    def save_system_state(self):
        """Save the current system state"""
        try:
            memory = api.get_api("_agent_memory")
            if not memory:
                self.add_to_log("Memory system not initialized yet", "WARNING")
                return

            if hasattr(memory, 'emergency_save'):
                memory.emergency_save()
                self.add_to_log("System state saved", "INFO")
            else:
                self.add_to_log("Memory system doesn't support manual saving", "WARNING")
        except Exception as e:
            self.add_to_log(f"Error saving system state: {str(e)}", "ERROR")
    
    def add_to_log(self, message, level="INFO"):
        """Add a message to the log display"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level}] {message}"
        
        # Apply styling based on log level
        color = {
            "INFO": "black",
            "WARNING": "orange",
            "ERROR": "red",
            "SUCCESS": "green",
            "DEBUG": "blue"
        }.get(level, "black")
        
        self.log_display.append(f"<font color='{color}'>{formatted_msg}</font>")
        
        # Auto-scroll to bottom
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())


def initialize_agent(main_window=None):
    """Initialize the agent system"""
    log_to_console("Initializing agent systems...", "INFO", "initialization")
    
    if main_window:
        main_window.add_to_log("Loading model and initializing agent...", "INFO")

    try:
        # Import and initialize agent
        log_to_console("Importing Agent Core...", "INFO", "initialization")
        from apis.agent.core import AgentCore
        log_to_console("Agent Core imported successfully", "SUCCESS", "initialization")

        log_to_console("Creating AgentCore instance and registering API...", "INFO", "initialization")
        agent_core = AgentCore()
        api.register_api("agent", agent_core)
        log_to_console("AgentCore instance created and API registered", "SUCCESS", "initialization")

        log_to_console("Starting agent loop in a separate daemon...", "INFO", "initialization")
        # Start agent loop in a separate thread
        agent_run_thread = threading.Thread(
            target=lambda: asyncio.run(agent_core.run()),
            daemon=True
        )
        agent_run_thread.start()
        log_to_console("Agent thread started successfully", "SUCCESS", "initialization")

        log_to_console("Startup successful", "SUCCESS", "initialization")
        if main_window:
            main_window.add_to_log("Agent initialized successfully!", "SUCCESS")
            
        return agent_run_thread
    
    except ImportError as ie:
        error_msg = f"ImportError initializing agent: {str(ie)}"
        log_to_console(error_msg, "ERROR", "initialization")
        if main_window:
            main_window.add_to_log(error_msg, "ERROR")
        import traceback
        log_to_console(f"Full traceback:\n {traceback.format_exc()}", level="ERROR", context="initialization")
        return None
            
    except Exception as e:
        error_msg = f"Error initializing agent: {str(e)}"
        log_to_console(error_msg, "ERROR", "initialization")
        if main_window:
            main_window.add_to_log(error_msg, "ERROR")
        import traceback
        log_to_console(f"Full traceback:\n {traceback.format_exc()}", level="ERROR", context="initialization")
        return None

def main():
    """Main entry point"""
    try:
        print(" **** STARTUP DIAGNOSTICS **** ")
        print("Python version:", sys.version)
        print("Current working directory:", os.getcwd())
        #print("Loaded modules:", list(sys.modules.keys()))
        print(" **** END DIAGNOSTICS **** ")

        # Set up PyQt application
        app = QtWidgets.QApplication(sys.argv)
        
        # Set style
        app.setStyle("Fusion")
        
        # Create and show main window with initial "waiting" state
        main_window = MainWindow()
        main_window.show()

        # Process app events to ensure window is shown before potentially lengthy initialization
        app.processEvents()
        
        # Log initial message
        main_window.add_to_log("GUI started successfully", "SUCCESS")
        main_window.add_to_log("Waiting for agent initialization...", "INFO")

        # Initialize agent after GUI launches (in a separate thread)
        agent_thread = threading.Thread(
            target=lambda: initialize_agent(main_window),
            daemon=True
        )
        agent_thread.start()
        
        # Run the application
        exit_code = app.exec()
        
        # Perform cleanup
        log_to_console("Shutting down...", "INFO")
        agent = api.get_api("agent")
        if agent:
            agent.shutdown()
        
        # Wait for agent thread to terminate
        if agent_thread.is_alive():
            main_window.add_to_log("Waiting for agent thread to terminate...", level="INFO", context="main")
            agent_thread.join(timeout=5)
        
        return exit_code
    
    except KeyboardInterrupt as app_quit:
        log_to_console(f"System shutdown trigger detected, shutdown process initializing...")
        asyncio.run(api.handle_shutdown())
        return 1

    except Exception as main_e:
        log_to_console(f"Fatal error in main: {main_e}", "ERROR", "main")
        import traceback
        log_to_console(f"Full traceback:\n {traceback.format_exc()}", level="ERROR", context="main")
        return 1

def signal_handler(sig, frame):
    print(f"[CRASH] Received signal {sig}")
    import traceback
    traceback.print_stack(frame)

    asyncio.run(api.handle_shutdown())

    sys.exit(1)

#signal.signal(signal.SIGSEGV, signal_handler)  # Segmentation fault
#signal.signal(signal.SIGABRT, signal_handler)  # Abort
#signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C

if __name__ == "__main__":
    sys.exit(main())
