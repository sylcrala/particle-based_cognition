import os
import sys
from apis.api_registry import api

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
    QVBoxLayout,
)
from PyQt6.QtGui import QPalette
from PyQt6.QtCore import Qt
from vispy import app
app.use_app('pyqt6')
from vispy import scene
from vispy.scene.visuals import Text, XYZAxis, Line, Markers, GridLines
from pathlib import Path
import asyncio
import numpy as np
import math
import threading




class VisualizerTab(QWidget):
    """The dedicated visualizer tab class - holds the vispy canvas (which handles 3D rendering and controls + legend/information)"""
    def __init__(self):
        super().__init__()
        self.logger = api.get_api("logger")

        # set layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # set palette
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.cyan)
        self.setPalette(palette)
        self.setAutoFillBackground(True) 

        # set up vispy canvas
        self.visualizer_canvas = VisualizerCanvas()
        
        # Add canvas to layout
        self.layout.addWidget(self.visualizer_canvas.native)


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

        scene.SceneCanvas.__init__(self, keys="interactive")
        self.unfreeze()
        self.agent_ready = False

        try:
            self.particle_source = api.get_api("_agent_field")
            if self.particle_source:
                self.log("Particle field API connected successfully", "INFO", "__init__()")
            else:
                self.log("Particle field API not yet available - will retry during updates", "WARNING", "__init__()")
        except Exception as field_not_found:
            self.log(f"Particle source API '_agent_field' not found - cannot populate visualizer canvas | Error: {field_not_found}", "ERROR", "__init__()")
            self.particle_source = None

        # camera setup
        self.view = self.central_widget.add_view()
        self.view.camera = "turntable" # or "fly"
        self.view.camera.fov = 45
        self.view.camera.distance = 5

        # create 3D grid
        self.axis = XYZAxis(parent = self.view.scene) # 3D XYZ axis
        self.grid_lines = GridLines(
            parent=self.view.scene,
            scale=(1, 1, 1),
            color=(0.2, 0.2, 0.2, 0.3)
        )

        # particle setup - dependent on agent being ready
        particle_types = ["memory", "lingual", "sensory", "core"]

        self.particle_visuals = {}
        self.shimmer_visuals = {}
        self.trail_visuals = {}
        
        self.show_entanglements = True
        self.show_trails = True
        self.show_shimmer = True
        self.show_spatial_grid = True


        # 3d overlay controls
        # TODO

        self.events.mouse_press.connect(self.on_mouse_press)
        self.particle_metadata = {"memory": [], "lingual": [], "sensory": [], "core": [], "unknown": []}
        
        #if self.particle_source != None:
        try:
            # particle visual foundations with compatibility settings
            for p_type in particle_types:
                try:
                    # Initialize Markers for particles
                    self.particle_visuals[p_type] = Markers(
                        parent=self.view.scene,
                        antialias=False  # Disable antialiasing for compatibility
                    )
                    
                    # Disable Line visuals for now due to shader compatibility issues
                    self.shimmer_visuals[p_type] = None
                    self.trail_visuals[p_type] = None

                    # Set minimal initial data
                    self.particle_visuals[p_type].set_data(
                        pos=np.array([[0, 0, 0]], dtype=np.float32),
                        size=np.array([0.001], dtype=np.float32),
                        face_color=np.array([[0, 0, 0, 0]], dtype=np.float32)
                    )

                    # Don't set data for Line visuals initially to avoid shader issues
                    # They will be populated when particles are actually rendered
                    
                    self.log(f"Initialized visuals for particle type: {p_type}", "DEBUG", "__init__")
                    
                except Exception as visual_error:
                    self.log(f"Error initializing visuals for {p_type}: {visual_error}", "WARNING", "__init__")
                    # Create placeholder None values for failed visuals
                    self.particle_visuals[p_type] = None
                    self.shimmer_visuals[p_type] = None
                    self.trail_visuals[p_type] = None
        except Exception as e:
            self.log(f"Error setting up particle visuals: {e}", "ERROR", "__init__()")

            # Disable connection lines for now due to shader compatibility issues
            self.connection_lines = None

        # Start update timer in separate thread for async operations
        self.update_thread = threading.Thread(
            target=self.run_update_loop,
            daemon=True
        )
        self.update_thread.start()


    def create_manual_grid(self):
        """Create a manual grid as fallback for GridLines compatibility issues"""
        try:
            # Create grid points manually
            grid_range = 2.0
            grid_step = 0.2
            grid_points = []
            
            # Create grid lines in X direction
            for y in np.arange(-grid_range, grid_range + grid_step, grid_step):
                for z in np.arange(-grid_range, grid_range + grid_step, grid_step):
                    grid_points.extend([
                        [-grid_range, y, z],
                        [grid_range, y, z]
                    ])
            
            # Create grid lines in Y direction  
            for x in np.arange(-grid_range, grid_range + grid_step, grid_step):
                for z in np.arange(-grid_range, grid_range + grid_step, grid_step):
                    grid_points.extend([
                        [x, -grid_range, z],
                        [x, grid_range, z]
                    ])
                    
            # Create grid lines in Z direction
            for x in np.arange(-grid_range, grid_range + grid_step, grid_step):
                for y in np.arange(-grid_range, grid_range + grid_step, grid_step):
                    grid_points.extend([
                        [x, y, -grid_range],
                        [x, y, grid_range]
                    ])
            
            if grid_points:
                manual_grid = Line(
                    pos=np.array(grid_points, dtype=np.float32),
                    color=(0.2, 0.2, 0.2, 0.3),
                    connect='segments',
                    parent=self.view.scene,
                    antialias=False  # Disable antialiasing to reduce shader complexity
                )
                return manual_grid
                
        except Exception as manual_error:
            self.log(f"Manual grid creation also failed: {manual_error}", "ERROR", "__init__")
            return None

    def log(self, message, level="INFO", context = None):
        """Send log messages to system logger"""
        
        if context != None:
            context = context
        else:
            context = "no_context"

        if self.logger:
            self.logger.log(message, level, context=context, source="VisualizerCanvas")


    def run_update_loop(self):
        """Run the async update loop in a separate thread"""
        try:
            self.log("Starting visualizer update loop...", "INFO", "run_update_loop")
            asyncio.run(self.start_update_timer())
        except Exception as e:
            self.log(f"Update loop error: {e}", "ERROR", "run_update_loop")
            import traceback
            traceback.print_exc()

    async def start_update_timer(self):
        """Starts the visualizer update timer"""
        # Wait a bit for agent to initialize
        self.log("Waiting for agent initialization...", "INFO", "start_update_timer")
        await asyncio.sleep(3.0)
        
        # Try to reconnect to particle source if not available
        if not self.particle_source:
            try:
                self.particle_source = api.get_api("_agent_field")
                if self.particle_source:
                    self.log("Particle field API connected after retry", "INFO", "start_update_timer")
            except Exception as e:
                self.log(f"Failed to connect to particle field: {e}", "WARNING", "start_update_timer")
        
        update_count = 0
        while True:  # Keep running even if particle source not available yet
            try:
                if self.particle_source:
                    await self.update_particles()
                    if update_count % 100 == 0:  # Log every 5 seconds (100 * 0.05s)
                        self.log(f"Visualizer update #{update_count}", "DEBUG", "start_update_timer")
                else:
                    # Try to reconnect periodically
                    if update_count % 20 == 0:  # Every second
                        try:
                            self.particle_source = api.get_api("_agent_field")
                            if self.particle_source:
                                self.log("Particle field API connected", "INFO", "start_update_timer")
                        except:
                            pass
                            
                update_count += 1
                await asyncio.sleep(0.05)
                
            except Exception as e:
                self.log(f"Visualization update error: {e}", "ERROR", "start_update_timer()")
                await asyncio.sleep(1.0)


    def on_mouse_press(self, event):
        """Handles mouse clicks - mainly used for particle selected at the moment"""
        if event.button == 1: # left click
            # get 3D coords from 2D click 
            clicked_particle = self.find_closest_particle(event.pos)

            if clicked_particle:
                self.show_particle_details(clicked_particle)

    def find_closest_particle(self, screen_pos):
        """Find particle using direct field references"""
        tr = self.scene.node_transform(self.view.scene)
        min_dist = float("inf")
        closest = None
        
        for p_type in self.particle_visuals:
            if len(self.particle_metadata[p_type]) > 0:
                positions = self.particle_visuals[p_type]._data['pos']
                
                for i, field_particle in enumerate(self.particle_metadata[p_type]):
                    if i < len(positions):
                        pos_3d = positions[i]
                        screen_3d = tr.map(pos_3d)
                        distance = np.linalg.norm(screen_3d[:2] - screen_pos)
                        
                        if distance < min_dist and distance < 50:  # 50px tolerance
                            min_dist = distance
                            closest = {
                                'field_particle': field_particle,  # Direct field reference!
                                'position': pos_3d,
                                'type': p_type
                            }
        return closest

    def show_particle_details(self, particle_info):
        """Show details using live field data"""
        if hasattr(self, 'detail_overlay'):
            self.detail_overlay.parent = None
        
        field_particle = particle_info['field_particle']
        
        # Get LIVE data from field particle
        detail_text = f"""
LIVE FIELD PARTICLE
==================
ID: {str(field_particle.id)[:8]}...
Type: {field_particle.type.upper()}
Position: {field_particle.position[:3]}
Energy: {field_particle.energy:.3f}
Activation: {field_particle.activation:.3f}
Policy: {getattr(field_particle, 'policy', 'unknown')}
Alive: {field_particle.alive}
Linked Particles: {len(field_particle.linked_particles.get('children', []))}

QUANTUM STATE:
Certainty: {field_particle.superposition['certain']:.3f}
Collapsed: {hasattr(field_particle, 'collapsed_state')}

SPATIAL INFO:
Grid Key: {self.particle_source._get_grid_key(field_particle.position)}
Neighbors: {len(self.particle_source.get_spatial_neighbors(field_particle, radius=0.6))}

Click elsewhere to close
"""
        
        self.detail_overlay = Text(
            detail_text,
            pos=(50, 50),
            font_size=10,
            color='white',
            parent=self.view
        )


    #*# -- VISUALIZATION / RENDERING -- #*#

    def convert_hues_to_rgba(self, hues, saturations, brightness, opacities):
        """Converts arrays of hues (particle type), saturations (frequency), brightness (valence), and opacities (certainty) to RGBA color values"""
        import colorsys

        colors = []
        for hue, sat, bright, opacity in zip(hues, saturations, brightness, opacities):
            # converting from 0-360 to 0-1
            h = (hue % 360) / 360.0
            s = min(max(sat, 0.0), 1.0)
            v = min(max(bright, 0.0), 1.0)

            # convert hsv to rgb
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            colors.append([r, g, b, opacity])

        return np.array(colors, dtype=np.float32)

    def update_field_attention_indicator(self):
        """Show conscious attention region from field"""
        try:
            attention_region = self.particle_source.get_conscious_attention_region()
            if attention_region:
                # Highlight the conscious attention grid sector
                grid_key = attention_region
                center_pos = np.array([
                    grid_key[0] * self.particle_source.grid_size,
                    grid_key[1] * self.particle_source.grid_size, 
                    grid_key[2] * self.particle_source.grid_size
                ])
                
                # Create attention region indicator
                if hasattr(self, 'attention_marker'):
                    self.attention_marker.parent = None
                
                self.attention_marker = Markers(parent=self.view.scene)
                self.attention_marker.set_data(
                    pos=np.array([center_pos]),
                    size=np.array([20]),  # Larger marker
                    face_color=np.array([[1.0, 1.0, 0.0, 0.3]]),  # Yellow glow
                    edge_color=np.array([[1.0, 1.0, 0.0, 0.8]]),
                    edge_width=2
                )
                
        except Exception as e:
            self.log(f"Error updating attention indicator: {e}", "ERROR", "update_field_attention_indicator")

    def update_connections_from_field(self):
        """Draw connections using actual field linkages"""
        if not self.show_entanglements:
            return
        
        connection_points = []
        connection_colors = []
        
        alive_particles = self.particle_source.get_alive_particles()
        
        for particle in alive_particles:
            # Use actual field linkage data
            children = particle.linked_particles.get('children', [])
            
            for child_id in children:
                child_particle = self.particle_source.get_particle_by_id(child_id)
                if child_particle and child_particle.alive:
                    # Draw line between actual field positions
                    connection_points.extend([
                        particle.position[:3],  # Parent position
                        child_particle.position[:3]  # Child position
                    ])
                    
                    # Color based on actual connection strength
                    distance = np.linalg.norm(particle.position[:3] - child_particle.position[:3])
                    strength = max(0.2, 1.0 - (distance * 0.5))  # Closer = stronger
                    
                    connection_colors.extend([
                        [0.5, 0.5, 1.0, strength],
                        [0.5, 0.5, 1.0, strength]
                    ])
        
        if connection_points and self.connection_lines is not None:
            try:
                self.connection_lines.set_data(
                    pos=np.array(connection_points, dtype=np.float32),
                    color=np.array(connection_colors, dtype=np.float32)
                )
            except Exception as e:
                self.log(f"Error updating connection lines: {e}", "WARNING", "update_connections_from_field")

    async def update_particles(self):
        """Updates the particle visuals on the canvas based on the current field state"""
        if not self.particle_source:
            return
        
        alive_particles = self.particle_source.get_alive_particles()

        by_type = {
            "memory": [],
            "lingual": [],
            "sensory": [],
            "core": [],
            "unknown": []
        }

        for particle in alive_particles:
            render_data = await particle.render()
            p_type = render_data.get("type", "unknown") # if invalid or new particle type is detected, add to "unknown" category
            by_type[p_type].append({
                "render_data": render_data,
                "particle_ref": particle        # reference to actual particle in field
            })

        for p_type, particle_list in by_type.items():
            if particle_list:
                # stacking position arrays
                positions = np.stack([p["render_data"]["position"] for p in particle_list])
                sizes = np.array([p["render_data"]["size"] for p in particle_list])

                # pull color values
                colors = self.convert_hues_to_rgba(
                    hues = np.array([p["render_data"]["color_hue"] for p in particle_list]),
                    saturations = np.array([p["render_data"]["color_saturation"] for p in particle_list]),
                    brightness = np.array([p["render_data"]["glow"] for p in particle_list]),
                    opacities = np.array([p["render_data"]["quantum_state"]["opacity"] for p in particle_list])
                )

                # update vispy visuals with error handling
                if p_type in self.particle_visuals and self.particle_visuals[p_type] is not None:
                    try:
                        self.particle_visuals[p_type].set_data(
                            pos=positions,
                            size=sizes,
                            face_color=colors
                        )
                    except Exception as e:
                        self.log(f"Error updating {p_type} particle visuals: {e}", "WARNING", "update_particles")


                self.particle_metadata[p_type] = [p["particle_ref"] for p in particle_list]