"""
Particle-based Cognition Engine - GUI visualizer tab utilities
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

from apis.api_registry import api
from gui.utils.particle_popup import show_particle_details
import colorsys

# Set up OpenGL environment for Fedora 42 compatibility
def setup_opengl_env():
    """Configure OpenGL for XWayland + NVIDIA"""
    import os
    
    # Use system OpenGL (should now work through XWayland)
    os.environ["VISPY_GL_LIB"] = "libGL.so.1"
    
    # Let VisPy know we're in X11 mode
    os.environ["VISPY_BACKEND"] = "pyqt6"
    
    # NVIDIA XWayland optimizations
    os.environ["__GL_VENDOR_LIBRARY_NAME"] = "nvidia"
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

config = api.get_api("config")

if config.wayland_active:
    print("Wayland active - applying OpenGL settings for XWayland")
    # Apply OpenGL settings before importing VisPy
    setup_opengl_env()

from PyQt6.QtWidgets import (
    QWidget, 
    QStackedLayout,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel
)
from PyQt6.QtGui import QPalette
from PyQt6.QtCore import QTimer
from vispy import app
app.use_app('pyqt6')
from vispy import scene
from vispy.scene.visuals import Text, XYZAxis, Line, Markers, GridLines
import numpy as np
import datetime as dt
import threading


class VisualizerTab(QWidget):
    """The dedicated visualizer tab class - holds the vispy canvas (which handles 3D rendering and controls + legend/information)"""
    def __init__(self):
        super().__init__()
        self.logger = api.get_api("logger")
        
        # set layout
        self.base_layout = QHBoxLayout()
        self.setLayout(self.base_layout)
        #self.bar_layout = QHBoxLayout()
        #self.base_layout.addLayout(self.bar_layout, stretch=1)
        self.content_layout = QStackedLayout()
        self.base_layout.addLayout(self.content_layout, stretch=10)
        self.utility_layout = QVBoxLayout()
        self.base_layout.addLayout(self.utility_layout, stretch=2)
        
        """
        # set palette
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.cyan)
        self.setPalette(palette)
        self.setAutoFillBackground(True) 
        """
        # set up content area
        self.visualizer_canvas = VisualizerCanvas()

        # set up bar area
        #self.visualizer_pause_btn = QPushButton("Pause/unpause visualizer")
        #self.visualizer_pause_btn.clicked.connect(self.toggle_visualizer_pause)
        #self.bar_layout.addWidget(self.visualizer_pause_btn)

        #self.visualizer_pwr_btn = QPushButton("Enable/disable visualizer")
        #self.visualizer_pwr_btn.clicked.connect(self.toggle_visualizer_enabled)
        #self.bar_layout.addWidget(self.visualizer_pwr_btn)

        # set up utility area
        self.utility_content_1 = QHBoxLayout()
        self.utility_content_2 = QHBoxLayout()
        from gui.tabs.visualizer.utils.field_stats import FieldStats
        self.utility_layout.addWidget(FieldStats(), stretch=3)
        self.utility_layout.addWidget(QLabel("add 'current activity' display here"), stretch=3)
        self.utility_layout.addLayout(self.utility_content_1, stretch=2)
        self.utility_layout.addLayout(self.utility_content_2, stretch=2)
        from gui.tabs.logging.utils.logstream import LogStream
        self.utility_layout.addWidget(LogStream(view_limit=100), stretch=3)

        # utility functions - content set 1
        self.save_state_btn = QPushButton("Save agent state")
        self.save_state_btn.clicked.connect(self.save_agent_state)
        self.utility_content_1.addWidget(self.save_state_btn)


        # utility functions - content set 2
        self.toggle_pause_btn = QPushButton("Pause/unpause")
        self.toggle_pause_btn.clicked.connect(self.toggle_visualizer_pause)
        self.utility_content_2.addWidget(self.toggle_pause_btn)
        self.toggle_enable_btn = QPushButton("Enable/disable")
        self.toggle_enable_btn.clicked.connect(self.toggle_visualizer_enabled)
        self.utility_content_2.addWidget(self.toggle_enable_btn)

    
    def save_agent_state(self):
        agent = api.get_api("agent")
        agent.save()
        self.log("Manual agent state save triggered via visualizer", "INFO", "save_agent_state")

    def toggle_visualizer_enabled(self):
        self.visualizer_canvas.visualizer_enabled = not self.visualizer_canvas.visualizer_enabled
        if self.visualizer_canvas.visualizer_enabled == False:
            self.content_layout.removeWidget(self.visualizer_canvas.native)
        if self.visualizer_canvas.visualizer_enabled == True:
            self.content_layout.addWidget(self.visualizer_canvas.native)

    def toggle_visualizer_pause(self):
        self.visualizer_canvas.visualizer_enabled = not self.visualizer_canvas.visualizer_enabled
        


    def log(self, message, level="INFO", context = None):
        """Send log messages to system logger"""
        
        if context != None:
            context = context
        else:
            context = "no_context"

        if self.logger:
            self.logger.log(message, level, context=context, source="VisualizerTab")


        
class VisualizerCanvas(scene.SceneCanvas):
    """The vispy canvas holding 3D field visualization, controls, and information overlays"""
    def __init__(self):
        self.logger = api.get_api("logger")
        self.visualizer_enabled = False # flag to enable the visualizer - default is False to prevent ghost API calls, must be enabled in app via a button or toggle
        self.agent_ready = False

        # TODO add dedicated thread for visualizer canvas to prevent GUI hanging when under heavy canvas load

        scene.SceneCanvas.__init__(self, keys="interactive", bgcolor="white") #bgcolor=(0.01, 0.01, 0.03, 1.0)
        self.unfreeze()

        self.particle_source = None

        # camera setup
        self.view = self.central_widget.add_view()
        self.view.camera = "fly"
        self.view.camera.fov = 45
        self.view.camera.distance = 20
        self.view.camera.center = (0, 0, 0)

        self.view.camera.pan_sensitivity = 0.5
        self.view.camera.zoom_sensitivity = 0.8

        self.camera = self.view.camera

        self.setup_camera_controls()

        # particle setup - dependent on agent being ready
        particle_types = ["memory", "lingual", "sensory", "core"]

        self.position_history = {}
        self.trail_length = 10

        self.interactive_particles = {}
        self.shimmer_visuals = {}
        self.trail_visuals = {}
        
        self.show_entanglements = True
        self.show_trails = False
        self.show_shimmer = True
        self.show_spatial_grid = True

        # create 3D rendering axis
        self.axis = XYZAxis(parent = self.view.scene) # 3D XYZ axis
        if self.show_spatial_grid:
            self.setup_gridlines()
        else:
            self.grid_lines = None

        # help text box
        self.help_overlay = Text(
            """ 
Controls:                                   |   "R" to reset camera
WASD for movement                           |   "F" to focus on particles center
Mouse to pan camera                         |   "H" to toggle help overlay
Left click particles for details (disabled) |   "G" to toggle spatial grid
Escape to close particle details (disabled) |   "P" to save agent state
""",
            pos=(10, 30),
            font_size=10,
            color=(0.7, 0.7, 0.7, 0.8),
            parent=self.view,
            anchor_x='left',
            anchor_y='bottom'
        )

        self.detail_overlay = None

        self.events.mouse_press.connect(self.on_canvas_click)
        self.particle_metadata = {"memory": [], "lingual": [], "sensory": [], "core": [], "unknown": []}

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_particles)
        self.update_timer.start(100)  # 40 FPS


    def log(self, message, level="INFO", context = None):
        """Send log messages to system logger"""
        
        if context != None:
            context = context
        else:
            context = "no_context"

        if self.logger:
            self.logger.log(message, level, context=context, source="VisualizerCanvas")

    #*# -- GRIDLINES -- #*#
    def setup_gridlines(self):
        """Create cube boundary gridlines for spatial reference"""
        try:
            bounds = {
                'x': (-5.0, 5.0),
                'y': (-5.0, 5.0), 
                'z': (-5.0, 5.0)
            }
            
            x_min, x_max = bounds['x']
            y_min, y_max = bounds['y'] 
            z_min, z_max = bounds['z']

            lines = []
            colors = []

            # Main cube edges (bright)
            cube_edges = [
                # Bottom face
                [x_min, y_min, z_min], [x_max, y_min, z_min],
                [x_max, y_min, z_min], [x_max, y_max, z_min], 
                [x_max, y_max, z_min], [x_min, y_max, z_min],
                [x_min, y_max, z_min], [x_min, y_min, z_min],
                # Top face
                [x_min, y_min, z_max], [x_max, y_min, z_max],
                [x_max, y_min, z_max], [x_max, y_max, z_max],
                [x_max, y_max, z_max], [x_min, y_max, z_max], 
                [x_min, y_max, z_max], [x_min, y_min, z_max],
                # Vertical edges
                [x_min, y_min, z_min], [x_min, y_min, z_max],
                [x_max, y_min, z_min], [x_max, y_min, z_max],
                [x_max, y_max, z_min], [x_max, y_max, z_max],
                [x_min, y_max, z_min], [x_min, y_max, z_max],
            ]
            
            cube_colors = [[0.9, 0.9, 0.9, 0.8]] * len(cube_edges)
            
            # Add a few reference grid lines (just center lines on each face)
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2  
            center_z = (z_min + z_max) / 2
            
            reference_lines = [
                # Center horizontal line on front/back faces
                [x_min, center_y, z_min], [x_max, center_y, z_min],
                [x_min, center_y, z_max], [x_max, center_y, z_max],
                # Center vertical line on front/back faces  
                [center_x, y_min, z_min], [center_x, y_max, z_min],
                [center_x, y_min, z_max], [center_x, y_max, z_max],
            ]
            
            reference_colors = [[0.5, 0.5, 0.5, 0.4]] * len(reference_lines)

            lines.extend(cube_edges)
            lines.extend(reference_lines)
            colors.extend(cube_colors)
            colors.extend(reference_colors)

            # Create visual
            lines = np.array(lines, dtype=np.float32)
            colors = np.array(colors, dtype=np.float32)

            self.grid_lines = Line(
                parent=self.view.scene,
                antialias=True
            )
            
            self.grid_lines.set_data(
                pos=lines,
                color=colors,
                connect='segments'
            )
            
            self.log("Simple boundary cube created", "DEBUG", "_create_simple_boundary_cube")
            
        except Exception as e:
            self.log(f"Simple cube creation error: {e}", "ERROR", "_create_simple_boundary_cube")
            self.grid_lines = None

    def toggle_spatial_grid(self):
        """Toggle the spatial grid on/off"""
        try:
            if self.grid_lines:
                self.grid_lines.visible = not self.grid_lines.visible
                self.log(f"Grid visibility: {self.grid_lines.visible}", "DEBUG", "toggle_spatial_grid")
            else:
                # Create grid if it doesn't exist
                if self.show_spatial_grid:
                    self.grid_lines = GridLines(parent=self.view.scene)
                    self.log("Grid created and enabled", "DEBUG", "toggle_spatial_grid")
        except Exception as e:
            self.log(f"Grid toggle error: {e}", "ERROR", "toggle_spatial_grid")


    #*# -- KEY PRESS HANDLERS -- #*#

    def on_key_press(self, event):
        """Handles key press events"""
        if event.key == "r":    # reset camera
            self.view.camera.center = (0, 0, 0)
            self.view.camera.distance = 15
        elif event.key == 'f':  # Focus on particles
            if self.particle_source:
                particles = self.particle_source.get_alive_particles()
                if particles:
                    # calculate center of all particles
                    positions = [p.render()['position'][:3] for p in particles]
                    center = np.mean(positions, axis=0)
                    self.view.camera.center = center
                    self.view.camera.distance = 30
        elif event.key == 'h':  # toggle help overlay
            self.help_overlay.visible = not self.help_overlay.visible
        elif event.key == 'g':  # toggle spatial grid
            self.toggle_spatial_grid()
        elif event.key == "p":
            agent = api.get_api("agent")
            agent.save()
            self.log("Manual agent state save triggered via visualizer", "INFO", "on_key_press")
        elif event.key == 'Escape':  # close detail overlay
            if self.detail_overlay:
                self.detail_overlay = None
            else:
                pass  # No overlay to close

    def on_canvas_click(self, event):
        """Handles mouse click events"""
        if event.button == 1:  # Left click
            if event.pos is None:
                return
            # Check for interactive particles at click position
            for id, interactive_particle in self.interactive_particles.items():
                if interactive_particle.contains(event.pos):
                    interactive_particle.on_click(event)
                    break


    #*# -- CAMERA -- #*#

    def setup_camera_controls(self):
        """Set up camera controls and bindings"""
        self.events.key_press.connect(self.on_key_press)
        pass


    #*# -- VISUALIZATION / RENDERING -- #*#

    def update_particles(self):
        """Updates the particle visuals on the canvas based on the current field state"""
        # debug statement
        #self.log("update_particles() called", "DEBUG", "update_particles")  # Add this line

        # update particle source
        if self.particle_source is None:
            self.particle_source = api.get_api("_agent_field")
            if self.particle_source is None:
                self.log("Particle source not available - skipping update particles rotation", "WARNING", "update_particles")
                return 
        
        if not self.visualizer_enabled:
            return
        
        # debug statement
        #self.log("Updating particles...", "DEBUG", "update_particles")
        
        try: # update loop
            alive_particles = self.particle_source.get_alive_particles()
            current_particle_ids = set(str(p.id) for p in alive_particles)

            self._update_interactive_particles(alive_particles, current_particle_ids)

            all_entanglements = []

            for particle in alive_particles:
                particle_id = str(particle.id)
                render_data = particle.render()
                current_pos = render_data["position"]

                # update position history
                if particle_id not in self.position_history:
                    self.position_history[particle_id] = []
                
                self.position_history[particle_id].append(current_pos)
                if len(self.position_history[particle_id]) > self.trail_length:
                    self.position_history[particle_id].pop(0)

                # collect entanglements
                if self.show_entanglements and "entanglements" in render_data:
                    for entanglement in render_data["entanglements"]:
                        all_entanglements.append({
                            "source_pos": current_pos,
                            "target_id": entanglement["target_id"],
                            "strength": entanglement["strength"],
                            "type": entanglement["type"]
                        })
            
            # update trails if enabled
            if self.show_trails:
                self._update_particle_trails()
            
            # update entanglements if enabled
            if self.show_entanglements:
                self._update_entanglement_connections(all_entanglements, alive_particles)
            
            # Update shimmer effects if enabled
            if self.show_shimmer:
                self._update_shimmer_effects(alive_particles)
            
            # Update metadata storage for interaction
            self._update_particle_metadata(alive_particles)

            # Clean up old data
            self._cleanup_old_visuals(current_particle_ids)
            
            # Update the canvas
            self.update()
            
        except Exception as e:
            self.log(f"Particle update error: {str(e)}", "ERROR", "update_particles")

    def _update_interactive_particles(self, alive_particles, current_particle_ids):
        """Update the interactive particle objects"""
        try:
            # Remove dead particles
            dead_particle_ids = set(self.interactive_particles.keys()) - current_particle_ids
            for dead_id in dead_particle_ids:
                if self.interactive_particles[dead_id]:
                    self.interactive_particles[dead_id].parent = None
                del self.interactive_particles[dead_id]
            
            # Update existing and create new particles
            for particle in alive_particles:
                particle_id = str(particle.id)
                
                if particle_id in self.interactive_particles:
                    # Update existing interactive particle
                    self.interactive_particles[particle_id].update_particle_data(particle)
                else:
                    # Create new interactive particle
                    interactive_particle = InteractableParticle(
                        particle_data=particle,
                        canvas_ref=self,
                        parent=self.view.scene,
                        antialias=True
                    )
                    self.interactive_particles[particle_id] = interactive_particle
                    
        except Exception as e:
            self.log(f"Interactive particle update error: {e}", "ERROR", "_update_interactive_particles")

    def _cleanup_old_visuals(self, current_particle_ids):
        """Clean up visual effects for particles that no longer exist"""
        try:
            # Clean up trails
            old_trail_ids = set(self.trail_visuals.keys()) - current_particle_ids
            for old_id in old_trail_ids:
                if old_id in self.trail_visuals and self.trail_visuals[old_id]:
                    self.trail_visuals[old_id].parent = None
                self.trail_visuals.pop(old_id, None)

            old_shimmer_ids = set(self.shimmer_visuals.keys()) - current_particle_ids
            for old_id in old_shimmer_ids:
                if old_id in self.shimmer_visuals and self.shimmer_visuals[old_id]:
                    self.shimmer_visuals[old_id].parent = None
                self.shimmer_visuals.pop(old_id, None)
            
            # Clean up position history
            old_history_ids = set(self.position_history.keys()) - current_particle_ids
            for old_id in old_history_ids:
                self.position_history.pop(old_id, None)
                
        except Exception as e:
            self.log(f"Cleanup error: {e}", "ERROR", "_cleanup_old_visuals")

    def _update_particle_trails(self):
        """Update trail visuals for particles"""
        try:
            for particle_id, positions in self.position_history.items():
                if len(positions) > 1:
                    if particle_id not in self.trail_visuals:
                        self.trail_visuals[particle_id] = Line(
                            parent=self.view.scene, 
                            antialias=True
                        )
                    
                    # Create trail with fading opacity
                    trail_positions = np.array(positions, dtype=np.float32)
                    
                    # Create color array with fading alpha
                    trail_colors = []
                    for i, pos in enumerate(positions):
                        alpha = (i + 1) / len(positions) * 0.5  # Fade from 0 to 0.5
                        trail_colors.append([0.5, 0.5, 0.5, alpha])
                    
                    trail_colors = np.array(trail_colors, dtype=np.float32)
                    
                    # Update trail visual
                    if self.trail_visuals[particle_id]:
                        self.trail_visuals[particle_id].set_data(
                            pos=trail_positions, 
                            color=trail_colors,
                            connect='strip'  
                        )
                            
        except Exception as e:
            self.log(f"Trail update error: {str(e)}", "ERROR", "_update_particle_trails")

    def _update_entanglement_connections(self, entanglements, all_particles):
        """Update connection lines between entangled particles"""
        try:
            if not entanglements:
                return
                
            # Create particle ID to position mapping
            particle_positions = {}
            for particle in all_particles:
                render_data = particle.render()
                particle_positions[str(particle.id)] = render_data['position']
            
            # Build connection lines
            connection_lines = []
            connection_colors = []
            
            for entanglement in entanglements:
                target_id = entanglement['target_id']
                if target_id in particle_positions:
                    source_pos = entanglement['source_pos']
                    target_pos = particle_positions[target_id]
                    strength = entanglement['strength']
                    
                    # Add line segment
                    connection_lines.extend([source_pos, target_pos])
                    
                    # Color based on connection strength
                    alpha = min(strength, 1.0)
                    connection_colors.extend([
                        [1.0, 1.0, 0.0, alpha],  # Yellow connections
                        [1.0, 1.0, 0.0, alpha]
                    ])
            
            if connection_lines:
                connection_lines = np.array(connection_lines, dtype=np.float32)
                connection_colors = np.array(connection_colors, dtype=np.float32)
                
                # Update or create connection visual
                # TODO
                
        except Exception as e:
            self.log(f"Entanglement update error: {str(e)}", "ERROR", "_update_entanglement_connections")

    def _update_shimmer_effects(self, particles):
        """Update shimmer effects for uncertain particles"""
        # TODO
        try:
            current_time = dt.datetime.now().timestamp()
            
            for particle in particles:
                render_data = particle.render()
                particle_id = str(particle.id)
                
                if render_data['quantum_state']['animation'] == 'shimmer':
                    if particle_id not in self.shimmer_visuals:
                        # Create shimmer visual for this particle
                        # (Implementation depends on your shimmer visual setup)
                        pass
                    
                    # Update shimmer effect
                    # (Animate shimmer based on current_time and uncertainty)
                    
        except Exception as e:
            self.log(f"Shimmer update error: {str(e)}", "ERROR", "_update_shimmer_effects")

    def _update_particle_metadata(self, particles):
        """Update particle metadata storage for interactions"""
        try:
            # Get current particle IDs
            current_particle_ids = set(str(p.id) for p in particles)
            
            # Clean up old trail visuals for particles that no longer exist
            old_trail_ids = set(self.trail_visuals.keys()) - current_particle_ids
            for old_id in old_trail_ids:
                if self.trail_visuals[old_id]:
                    self.trail_visuals[old_id].parent = None  # Remove from scene
                del self.trail_visuals[old_id]
                
            # Clean up old position history
            old_history_ids = set(self.position_history.keys()) - current_particle_ids
            for old_id in old_history_ids:
                del self.position_history[old_id]
            
            # Clear previous metadata
            for p_type in self.particle_metadata:
                self.particle_metadata[p_type].clear()
            
            # Rebuild metadata storage
            for particle in particles:
                p_type = getattr(particle, 'type', 'unknown')
                if p_type not in self.particle_metadata:
                    self.particle_metadata[p_type] = []
                
                render_data = particle.render()
                metadata_entry = {
                    'id': render_data['id'],
                    'position': render_data['position'],
                    'particle_ref': particle,  # Keep reference for interaction
                    'render_data': render_data
                }
                self.particle_metadata[p_type].append(metadata_entry)
                
        except Exception as e:
            self.log(f"Metadata update error: {str(e)}", "ERROR", "_update_particle_metadata")


class InteractableParticle(Markers):
    """Custom vispy markers class for interactable particles"""
    def __init__(self, particle_data, canvas_ref, **kwargs):
        super().__init__(**kwargs)
        self.unfreeze()
        self.particle_data = particle_data
        self.particle_id = str(particle_data.id)
        self.canvas_ref = canvas_ref

        self.render_data = particle_data.render()
        render_data = self.render_data
        self.display_size = render_data['size']

        # set up marker data
        self.set_data(
            pos = np.array([render_data['position'][:3]], dtype=np.float32),
            size = np.array([self.display_size], dtype=np.float32),
            face_color = np.array([self._get_particle_color(render_data)], dtype=np.float32),
        )
        self.freeze()

        # interaction
        self.interactive = True
        self.events.mouse_press.connect(self.on_click)

    def contains(self, event_pos):
        """Check if the click is within the particle's area"""
        if event_pos is None:
            return False
        
        # Get the particle's current position and size
        pos = self.render_data['position']
        size = self.display_size

        # Calculate distance from click to particle center
        dx = event_pos[0] - pos[0]
        dy = event_pos[1] - pos[1]
        distance = np.sqrt(dx**2 + dy**2)

        # Check if within particle radius
        return distance <= size / 2
    


    def _get_particle_color(self, render_data):
        """Determine color based on render data"""
        h = (float(render_data['color_hue']) % 360) / 360.0
        s = min(max(abs(float(render_data['color_saturation'])), 0.3), 1.0)
        v = min(max(abs(render_data["glow_intensity"]), 0.3), 1.0)
        raw_opacity = float(render_data["quantum_state"]["opacity"])

        opacity = max(raw_opacity, 0.7)

        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return [r, g, b, opacity]
    
    def on_click(self, event):
        """Handle click events on the particle"""
        if self.canvas_ref:
            self.canvas_ref.log(f"Particle {self.particle_id} clicked", "INFO", "InteractableParticle")
            show_particle_details(self.particle_id, parent=self.canvas_ref)
            event.handled = True


    def update_particle_data(self, new_particle_data):
        """Update this particle's visual data"""
        self.unfreeze()
        self.particle_data = new_particle_data
        render_data = new_particle_data.render()
        self.render_data = render_data
        
        self.set_data(
            pos=np.array([render_data['position'][:3]], dtype=np.float32),
            size=np.array([render_data['size']], dtype=np.float32),
            face_color=np.array([self._get_particle_color(render_data)], dtype=np.float32)
        )
        self.freeze()


# registering visualizer to APIregistry for global access
api.register_api("visualizer", VisualizerCanvas)