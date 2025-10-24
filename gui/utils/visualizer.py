import os
import sys
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

        # create 3D grid
        self.axis = XYZAxis(parent = self.view.scene) # 3D XYZ axis
        self.grid = GridLines(parent = self.view.scene) # 3D grid lines

        # particle setup - dependent on agent being ready
        particle_types = ["memory", "lingual", "sensory", "core"]

        self.particle_visuals = {}
        self.shimmer_visuals = {}
        self.trail_visuals = {}
        
        self.show_entanglements = True
        self.show_trails = True
        self.show_shimmer = True
        self.show_spatial_grid = True

        if self.particle_source != None:
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
                self.grid_lines = GridLines(parent = self.view.scene, color = np.array([[0.2, 0.2, 0.2, 0.3]]), width = 0.5)
                self.grid_lines.set_data(
                    pos=np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32),
                    color=np.array([[0.2, 0.2, 0.2, 0.3], [0.2, 0.2, 0.2, 0.3]], dtype=np.float32),
                )
            except Exception as e:
                self.log(f"Error initializing grid lines: {e}", "ERROR", "__init__()")


    def start_update_timer(self):
        """Starts the visualizer update timer"""
        # TODO finish this, link it to the framework update loops






    def log(self, message, level="INFO", context = None):
        """Send log messages to system logger"""
        
        if context != None:
            context = context
        else:
            context = "no_context"

        if self.logger:
            self.logger.log(message, level, context=context, source="VisualizerCanvas")