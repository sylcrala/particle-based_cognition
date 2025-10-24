import os
import sys
from PyQt6.QtWidgets import (
    QWidget, 
    QVBoxLayout,
)
from PyQt6.QtGui import QPalette
from PyQt6.QtCore import Qt, QTimer
from vispy import app
app.use_app('pyqt6')
from vispy import scene
from vispy.scene.visuals import Text, XYZAxis, Line, Markers
from pathlib import Path
import asyncio
import numpy as np
import math
from traceback import format_exc

from apis.api_registry import api

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
        self.visualizer_enabled = False # flag to enblae the visualizer - default is False to prevent ghost API calls, must be enabled in app via a button or toggle

        scene.SceneCanvas.__init__(self, keys="interactive")
        self.unfreeze()
        self.agent_ready = False

        try:
            self.particle_source = api.get_api("_agent_field")
        except Exception as field_not_found:
            self.log(f"Particle source API '_agent_field' not found - cannot populate visualizer canvas | Error: {field_not_found}", "ERROR", "__init__()")
            self.particle_source = None

        # camera setup
        self.view = self.central_widget.add_view()
        self.view.camera = "turntable" # or "fly"
        self.view.camera.fov = 45
        self.view.camera.distance = 5

        # create 3D rendering axis
        self.axis = XYZAxis(parent = self.view.scene) # 3D XYZ axis

        # particle setup - dependent on agent being ready
        particle_types = ["memory", "lingual", "sensory", "core"]

        self.position_history = {}
        self.trail_length = 10

        self.particle_visuals = {}
        self.shimmer_visuals = {}
        self.trail_visuals = {}
        
        self.show_entanglements = True
        self.show_trails = True
        self.show_shimmer = True
        self.show_spatial_grid = True


        try:
            # particle visual foundations - filled with data from indidivual particles
            for p_type in particle_types:
                self.particle_visuals[p_type] = Markers(parent = self.view.scene)
                self.shimmer_visuals[p_type] = Line(parent = self.view.scene)
                self.trail_visuals[p_type] = Line(parent = self.view.scene)

                self.particle_visuals[p_type].set_data(
                    pos = np.array([[0, 0, 0]], dtype=np.float32),
                    size = np.array([0.001], dtype=np.float32),
                    face_color = np.array([[0, 0, 0, 0]], dtype=np.float32)
                )

                self.shimmer_visuals[p_type].set_data(
                    pos = np.array([[0, 0, 0]], dtype=np.float32),
                    size = np.array([0.001], dtype=np.float32),
                    face_color = np.array([[0, 0, 0, 0]], dtype=np.float32)
                )

                self.trail_visuals[p_type].set_data(
                    pos = np.array([[0, 0, 0]], dtype=np.float32),
                    color = np.array([[0, 0, 0, 0]], dtype=np.float32)
                )
        except Exception as e:
            self.log(f"Error setting up particle visuals: {e}", "ERROR", "__init__()")
            self.log(f"Traceback: {format_exc()}", "ERROR", "__init__()")

        try:
            # particle connection lines / linkage
            self.connection_lines = Line(parent = self.view.scene, connect = "segments", color = np.array([[0.5, 0.5, 1.0, 0.3]]), width = 1, antialias=True)
            self.connection_lines.set_data(
            pos=np.array([[0, 0, 0]], dtype=np.float32),
            color=np.array([[0, 0, 0, 0]], dtype=np.float32),
        )
        except Exception as e:
            self.log(f"Error initializing connection lines: {e}", "ERROR", "__init__()")

        try:
            self.grid_lines = Line(parent = self.view.scene, connect = "segments", color = np.array([[0.2, 0.2, 0.2, 0.3]]), width = 0.5, antialias=True)
            self.grid_lines.set_data(
                pos=np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32),
                color=np.array([[0.2, 0.2, 0.2, 0.3], [0.2, 0.2, 0.2, 0.3]], dtype=np.float32),
            )
        except Exception as e:
            self.log(f"Error initializing grid lines: {e}", "ERROR", "__init__()")
            self.log(f"Traceback: {format_exc()}", "ERROR", "__init__()")

        QTimer.singleShot(100, self.start_update_timer)


    def log(self, message, level="INFO", context = None):
        """Send log messages to system logger"""
        
        if context != None:
            context = context
        else:
            context = "no_context"

        if self.logger:
            self.logger.log(message, level, context=context, source="VisualizerCanvas")

    def start_update_timer(self):
        """Starts the visualizer update timer"""
        try:
            self.update_timer = QTimer()
            self.update_timer.timeout.connect(self.update_visualization)
            self.update_timer.start(20)  # Update every 20 ms
        except Exception as e:
            self.log(f"Error starting update timer: {e}", "ERROR", "start_update_timer()")

    def update_visualization(self):
        """Update the visualization canvas with current particle data"""
        try:
            if not self.particle_source:
                self.particle_source = api.get_api("_agent_field")
                if not self.particle_source:
                    self.log("Particle field not detected, waiting 20 seconds for system startup...", "WARNING", "update_visualization()")
                    asyncio.sleep(20)
                else:
                    self.log("Particle field detected, populating visualizer...", "INFO", "update_visualization()") 
            
            try:
                # build spatial grid lines, generalized bounds of -50 to 50 for now
                if self.show_spatial_grid:
                    try:
                        #positions = [p.render()["position"] for p in particles]
                        #pos_array = np.array(positions)
                        #min_bounds = (np.min(pos_array, axis=0))
                        #max_bounds = (np.max(pos_array, axis=0))
                        min_bounds = -50.0
                        max_bounds = 50.0

                        # create grid lines
                        grid_points = []
                        grid_extent = max(np.max(np.abs([min_bounds, max_bounds])) + 1.0, 2.0)
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
                        
                        grid_colors = np.tile([0.2, 0.2, 0.2, 0.15], (len(grid_points), 1))
                        self.grid_lines.set_data(
                            pos=np.array(grid_points),
                            color=grid_colors
                        )
                    
                    except Exception as e:
                        self.log(f"Error updating spatial grid lines: {e}", "ERROR", "update_visualization()")

                # gather particle information from field
                particles = self.particle_source.get_all_particles()
                if not particles:
                    self.log("No particles found in particle source", "WARNING", "update_visualization()")
                    return

            except Exception as e:
                self.log(f"Error during individual particle rendering process: {e}")

        except Exception as e:
            self.log(f"Error updating visualization: {e}", "ERROR", "update_visualization()")
            return